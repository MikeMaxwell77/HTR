"""Train a Keras CNN + two BiLSTMs word recognizer with CTC loss."""
import argparse
import csv
import json
import os
import hashlib
import zipfile
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np

if __package__:
    from .word_preprocessing import preprocess_image, read_grayscale, standardize_image
else:
    from word_preprocessing import preprocess_image, read_grayscale, standardize_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_state(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("assets/htr_state.json"))


def epoch_checkpoints(output_dir):
    return sorted((output_dir / "checkpoints").glob("epoch-[0-9]*.keras"))


def restore_checkpoint(model, output_dir, signature):
    import keras
    paths = epoch_checkpoints(output_dir)
    if not paths:
        raise ValueError(f"No epoch checkpoints found in {output_dir / 'checkpoints'}")
    path = paths[-1]
    state = checkpoint_state(path)
    if state["signature"] != signature:
        raise ValueError("Checkpoint dataset, split, vocabulary or model settings differ. Use the original settings or a new output directory.")
    # Recreate this run's width-dependent loss while restoring Adam's saved state.
    model = keras.models.load_model(path, custom_objects={"ctc_loss": model.loss})
    print(f"Resumed {path}: {state['completed_epoch']} completed epochs; "
          f"optimizer step {int(model.optimizer.iterations.numpy())}", flush=True)
    return model, state


def make_epoch_saver(output_dir, signature, state=None, keep=5):
    import keras

    class EpochSaver(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.state = state or {"completed_epoch": 0, "history": {}, "best_val_loss": None}

        def on_epoch_end(self, epoch, logs=None):
            logs = {key: float(value) for key, value in (logs or {}).items()}
            if not all(np.isfinite(value) for value in logs.values()):
                raise RuntimeError("Non-finite training metrics; keeping the previous valid checkpoint")
            for key, value in logs.items():
                self.state["history"].setdefault(key, []).append(value)
            previous = self.state["best_val_loss"]
            current = logs.get("val_loss", float("inf"))
            improved = np.isfinite(current) and (previous is None or current < previous)
            if improved:
                temporary_best = output_dir / ".best.pending.keras"
                keras.Model(self.model.input, self.model.output).save(temporary_best)
                temporary_best.replace(output_dir / "best.keras")
                self.state["best_val_loss"] = current
            self.state.update(completed_epoch=epoch + 1, signature=signature)
            directory = output_dir / "checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            pending = directory / ".epoch.pending.keras"
            # Full compiled model includes weights and Adam slots/iteration count.
            self.model.save(pending)
            with zipfile.ZipFile(pending, "a") as archive:
                archive.writestr("assets/htr_state.json", json.dumps(self.state))
            destination = directory / f"epoch-{epoch + 1:06d}.keras"
            pending.replace(destination)
            (output_dir / "history.json").write_text(json.dumps(self.state["history"], indent=2), encoding="utf-8")
            if keep:
                for obsolete in epoch_checkpoints(output_dir)[:-keep]:
                    obsolete.unlink()
            print(f"\nSaved epoch {epoch + 1}: {destination}", flush=True)

    return EpochSaver()


def read_records(csv_path, words_dir, limit=None):
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "target"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires image_path and target columns")
        records = [(str(words_dir / row["image_path"]), row["target"])
                   for row in reader if row["image_path"] and row["target"]]
    records = records[:limit] if limit else records
    if len(records) < 2:
        raise ValueError("At least two labeled images are required")
    if len({path for path, _ in records}) != len(records):
        raise ValueError("Duplicate image paths would leak across train/validation splits")
    return records


def canvas_size(records, width=None, height=None):
    # CTC needs an extra timestep between consecutive identical characters.
    min_steps = max(len(word) + sum(a == b for a, b in zip(word, word[1:]))
                    for _, word in records)
    required_width = min_steps * 4
    width = width if width is not None else max(128, required_width)
    height = height if height is not None else 32
    if width < required_width or height < 8 or width % 4 or height % 8:
        raise ValueError(f"CTC requires width >= {required_width} (multiple of 4) and "
                         "height >= 8 (multiple of 8); increase --width for long labels")
    return width, height


def build_ctc_model(width, height, num_characters, lstm_units=128):
    import keras
    from keras import layers, ops

    if lstm_units < 2 or lstm_units % 2:
        raise ValueError("lstm_units must be a positive even integer >= 2")
    image = layers.Input((width, height, 1), name="image")
    x = image
    for filters, pool in [(32, (2, 2)), (64, (2, 2)), (128, (1, 2))]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool)(x)
    x = layers.Reshape((width // 4, height // 8 * 128))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(0.15, name="lstm_1_dropout")(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=True))(x)
    x = layers.Dropout(0.15, name="lstm_2_dropout")(x)
    logits = layers.Dense(num_characters + 1, name="character_logits")(x)
    model = keras.Model(image, logits, name="word_recognizer")

    def ctc_loss(labels, predictions):
        labels = ops.cast(labels, "int32")
        lengths = ops.sum(ops.cast(labels != 0, "int32"), axis=1)
        output_lengths = ops.full((ops.shape(predictions)[0],), width // 4, dtype="int32")
        return ops.ctc_loss(labels, predictions, lengths, output_lengths, mask_index=0)

    model.compile(optimizer=keras.optimizers.Adam(clipnorm=5.0), loss=ctc_loss)
    return model


def make_dataset(records, vocabulary, width, height, batch_size, shuffle=False, seed=42):
    import tensorflow as tf
    char_to_id = {char: i + 1 for i, char in enumerate(vocabulary)}
    max_label = max(len(word) for _, word in records)

    def generate():
        for path, word in records:
            labels = np.zeros(max_label, dtype=np.int32)
            labels[:len(word)] = [char_to_id[c] for c in word]
            yield preprocess_image(path, width, height), labels

    data = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((width, height, 1), tf.float32),
        tf.TensorSpec((max_label,), tf.int32)))
    # Tell Keras the finite dataset size so it can restart each epoch cleanly.
    data = data.apply(tf.data.experimental.assert_cardinality(len(records)))
    if shuffle:
        data = data.shuffle(min(len(records), 256), seed=seed)
    return data.batch(batch_size).prefetch(1)


def decode_batch_predictions(logits, vocabulary):
    import keras
    lengths = np.full(len(logits), logits.shape[1], dtype=np.int32)
    decoded, _ = keras.ops.ctc_decode(logits, lengths, mask_index=0)
    return ["".join(vocabulary[i - 1] for i in row if i > 0)
            for row in keras.ops.convert_to_numpy(decoded[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data/word_dataset_info.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/ctc_word")
    parser.add_argument("--png-dir", type=Path, help="Optional standardized, losslessly compressed PNG export")
    parser.add_argument("--width", type=int, help="Resize canvas width before rotation (default: 128, increased for long labels)")
    parser.add_argument("--height", type=int, default=32, help="Resize canvas height before rotation (default: 32)")
    parser.add_argument("--lstm-units", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Optional small-run image limit")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume the latest completed epoch in output-dir")
    parser.add_argument("--resume-if-available", action="store_true", help="Resume if checkpoints exist; otherwise start a new run")
    parser.add_argument("--keep-checkpoints", type=int, default=5, help="Keep this many latest epoch checkpoints; 0 keeps all")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Stop after this many epochs without improvement; 0 disables")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or min(args.batch_size, args.epochs) < 1:
        parser.error("Use positive batch size/epochs and a validation fraction between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be >= 2")
    if min(args.keep_checkpoints, args.early_stopping_patience) < 0:
        parser.error("Checkpoint retention and early stopping patience must be >= 0")
    if epoch_checkpoints(args.output_dir) and not (args.resume or args.resume_if_available):
        parser.error("Output directory already has checkpoints. Use --resume or a new --output-dir")
    records = read_records(args.csv, args.words_dir, args.limit)
    width, height = canvas_size(records, args.width, args.height)
    vocabulary = sorted(set("".join(word for _, word in records)))
    print(f"Canvas: {width} x {height}; rotated input: ({width}, {height}, 1)")
    if args.png_dir:
        for index, (path, _) in enumerate(records):
            standardize_image(path, width, height, args.png_dir / f"{index:08d}.png")
    import keras
    keras.utils.set_random_seed(args.seed)
    indices = np.random.default_rng(args.seed).permutation(len(records))
    n_val = min(len(records) - 1, max(1, round(len(records) * args.validation_fraction)))
    validation = [records[i] for i in indices[:n_val]]
    training = [records[i] for i in indices[n_val:]]
    train_data = make_dataset(training, vocabulary, width, height, args.batch_size, True, args.seed)
    val_data = make_dataset(validation, vocabulary, width, height, args.batch_size)
    model = build_ctc_model(width, height, len(vocabulary), args.lstm_units)
    metadata = {"characters": vocabulary, "blank_index": 0, "width": width, "height": height,
                "rotation": "90 degrees clockwise", "normalization": "divide by 255",
                "padding": "white; left-aligned, vertically centered before rotation",
                "resize": "fit canvas preserving aspect ratio; Pillow LANCZOS",
                "png_compression": "lossless PNG, compression level 9 when exported",
                "lstm_output_dropout": 0.15,
                "lstm_units": [args.lstm_units, args.lstm_units // 2],
                "output": "logits", "seed": args.seed,
                "training": training, "validation": validation}
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    state = None
    if args.resume or (args.resume_if_available and epoch_checkpoints(args.output_dir)):
        model, state = restore_checkpoint(model, args.output_dir, signature)
    initial_epoch = state["completed_epoch"] if state else 0
    if initial_epoch >= args.epochs:
        print(f"Already completed {initial_epoch} epochs (target: {args.epochs}).")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    saver = make_epoch_saver(args.output_dir, signature, state, args.keep_checkpoints)
    callbacks = [keras.callbacks.TerminateOnNaN(), saver,
                 keras.callbacks.CSVLogger(str(args.output_dir / "training.csv"), append=bool(state))]
    if args.early_stopping_patience:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True))
    model.summary()
    model.fit(train_data, validation_data=val_data, epochs=args.epochs,
              initial_epoch=initial_epoch, callbacks=callbacks)
    # Portable inference export; epoch checkpoints retain optimizer state separately.
    keras.Model(model.input, model.output).save(args.output_dir / "word_recognition.keras")
    predictions = []
    for images, _ in val_data:
        predictions.extend(decode_batch_predictions(model.predict_on_batch(images), vocabulary))
    with (args.output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "target", "prediction"])
        writer.writerows((path, word, prediction) for (path, word), prediction in zip(validation, predictions))
    accuracy = sum(word == pred for (_, word), pred in zip(validation, predictions)) / len(validation)
    print(f"Validation exact word accuracy: {accuracy:.4f}")
    print(f"Model and preprocessing metadata saved to {args.output_dir}")


if __name__ == "__main__":
    main()
