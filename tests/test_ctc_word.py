import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from scripts.training.word_preprocessing import standardize_image, preprocess_image
from scripts.training.CTC_Word import canvas_size, build_ctc_model, decode_batch_predictions


class WordPipelineTests(unittest.TestCase):
    def test_rotation_padding_and_png_are_pixel_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            exported = Path(directory) / "rotated.png"
            pixels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
            Image.fromarray(pixels).save(source)
            rotated = standardize_image(source, 8, 8, exported)
            restored = np.asarray(rotated.transpose(Image.Transpose.ROTATE_90))
            np.testing.assert_array_equal(restored[3:5, :3], pixels)
            self.assertTrue(np.all(restored[:, 3:] == 255))
            with Image.open(exported) as image:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(rotated))
            self.assertEqual(preprocess_image(source, 8, 8).shape, (8, 8, 1))
            with self.assertRaises(ValueError):
                standardize_image(source, 2, 8)

    def test_ctc_repeated_characters_require_extra_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("L", (2, 2), 255).save(source)
            self.assertEqual(canvas_size([(source, "book")]), (20, 8))
            with self.assertRaises(ValueError):
                canvas_size([(source, "book")], 16, 8)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") and importlib.util.find_spec("keras"),
                         "TensorFlow/Keras are not installed")
    def test_training_decode_and_model_round_trip(self):
        import keras
        model = build_ctc_model(20, 8, 2, lstm_units=8)
        recurrent = [layer for layer in model.layers if isinstance(layer, keras.layers.Bidirectional)]
        self.assertEqual([layer.forward_layer.units for layer in recurrent], [8, 4])
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
            np.testing.assert_allclose(model(images).numpy(), loaded(images).numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
