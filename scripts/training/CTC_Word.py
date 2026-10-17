"""Train a Keras CNN + two BiLSTMs word recognizer with CTC loss."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np

if __package__:
    from .word_preprocessing import preprocess_image, read_grayscale, standardize_image
else:
    from word_preprocessing import preprocess_image, read_grayscale, standardize_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or min(args.batch_size, args.epochs) < 1:
        parser.error("Use positive batch size/epochs and a validation fraction between 0 and 1")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be >= 2")
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
    model.summary()
    history = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=10, restore_best_weights=True)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save an uncompiled inference model: no custom loss needed when loading.
    keras.Model(model.input, model.output).save(args.output_dir / "word_recognition.keras")
    metadata = {"characters": vocabulary, "blank_index": 0, "width": width, "height": height,
                "rotation": "90 degrees clockwise", "normalization": "divide by 255",
                "padding": "white; left-aligned, vertically centered before rotation",
                "resize": "fit canvas preserving aspect ratio; Pillow LANCZOS",
                "png_compression": "lossless PNG, compression level 9 when exported",
                "lstm_output_dropout": 0.15,
                "lstm_units": [args.lstm_units, args.lstm_units // 2],
                "output": "logits", "seed": args.seed,
                "training": training, "validation": validation}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
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
