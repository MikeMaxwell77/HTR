import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scripts.training.word_preprocessing import standardize_image, preprocess_image
from scripts.training.CTC_Word import canvas_size, build_ctc_model, decode_batch_predictions
from scripts.training.CTC_Word import make_epoch_saver, restore_checkpoint, epoch_checkpoints


class WordPipelineTests(unittest.TestCase):
    def test_rotation_padding_and_png_are_pixel_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            exported = Path(directory) / "rotated.png"
            pixels = np.arange(32, dtype=np.uint8).reshape(4, 8)
            Image.fromarray(pixels).save(source)
            rotated = standardize_image(source, 8, 8, exported)
            restored = np.asarray(rotated.transpose(Image.Transpose.ROTATE_90))
            np.testing.assert_array_equal(restored[2:6], pixels)
            self.assertTrue(np.all(restored[:2] == 255))
            with Image.open(exported) as image:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(rotated))
            self.assertEqual(preprocess_image(source, 8, 8).shape, (8, 8, 1))
            with self.assertRaises(ValueError):
                standardize_image(source, 0, 8)

    def test_resize_preserves_aspect_ratio_and_png_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.png"
            exported = Path(directory) / "resized.png"
            Image.new("L", (200, 100), 0).save(source)
            rotated = standardize_image(source, 128, 32, exported)
            restored = np.asarray(rotated.transpose(Image.Transpose.ROTATE_90))
            self.assertEqual(restored.shape, (32, 128))
            self.assertTrue(np.all(restored[:, :64] == 0))
            self.assertTrue(np.all(restored[:, 64:] == 255))
            with Image.open(exported) as image:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(rotated))

    def test_ctc_repeated_characters_require_extra_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("L", (2, 2), 255).save(source)
            self.assertEqual(canvas_size([(source, "book")]), (128, 32))
            self.assertEqual(canvas_size([(source, "book")], 20, 8), (20, 8))
            self.assertEqual(canvas_size([(source, "a" * 53)]), (420, 32))
            with self.assertRaises(ValueError):
                canvas_size([(source, "book")], 16, 8)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") and importlib.util.find_spec("keras"),
                         "TensorFlow/Keras are not installed")
    def test_training_decode_and_model_round_trip(self):
        import keras
        model = build_ctc_model(20, 8, 2, lstm_units=8)
        recurrent = [layer for layer in model.layers if isinstance(layer, keras.layers.Bidirectional)]
        self.assertEqual([layer.forward_layer.units for layer in recurrent], [8, 4])
        self.assertEqual(model.get_layer("lstm_1_dropout").rate, 0.15)
        self.assertEqual(model.get_layer("lstm_2_dropout").rate, 0.15)
        images = np.ones((2, 20, 8, 1), dtype=np.float32)
        labels = np.array([[1, 1], [2, 0]], dtype=np.int32)
        self.assertTrue(np.isfinite(model.train_on_batch(images, labels)))
        logits = np.full((1, 5, 3), -20.0, dtype=np.float32)
        logits[0, np.arange(5), [1, 1, 0, 1, 2]] = 20
        self.assertEqual(decode_batch_predictions(logits, ["a", "b"]), ["aab"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            keras.Model(model.input, model.output).save(path)
            loaded = keras.models.load_model(path)
            np.testing.assert_allclose(model(images, training=False).numpy(),
                                       loaded(images, training=False).numpy(), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") and importlib.util.find_spec("keras"),
                         "TensorFlow/Keras are not installed")
    def test_epoch_checkpoint_restores_optimizer_and_retains_latest(self):
        model = build_ctc_model(20, 8, 2, lstm_units=8)
        images = np.ones((2, 20, 8, 1), dtype=np.float32)
        labels = np.array([[1, 1], [2, 0]], dtype=np.int32)
        model.train_on_batch(images, labels)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            saver = make_epoch_saver(output, "same-data", keep=1)
            saver.set_model(model)
            saver.on_epoch_end(0, {"loss": 2.0, "val_loss": 3.0})
            restored, state = restore_checkpoint(
                build_ctc_model(20, 8, 2, lstm_units=8), output, "same-data")
            self.assertEqual(state["completed_epoch"], 1)
            self.assertEqual(int(restored.optimizer.iterations.numpy()), 1)
            for original, loaded in zip(model.optimizer.variables, restored.optimizer.variables):
                np.testing.assert_array_equal(original.numpy(), loaded.numpy())
            with self.assertRaises(ValueError):
                restore_checkpoint(model, output, "different-data")
            restored.train_on_batch(images, labels)
            saver = make_epoch_saver(output, "same-data", state, keep=1)
            saver.set_model(restored)
            saver.on_epoch_end(1, {"loss": 1.5, "val_loss": 2.0})
            self.assertEqual(int(restored.optimizer.iterations.numpy()), 2)
            self.assertEqual([p.name for p in epoch_checkpoints(output)], ["epoch-000002.keras"])
            self.assertEqual(saver.state["history"]["loss"], [2.0, 1.5])
            self.assertTrue((output / "best.keras").is_file())


if __name__ == "__main__":
    unittest.main()
