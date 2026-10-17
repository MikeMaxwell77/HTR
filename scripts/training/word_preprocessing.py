"""Pixel-preserving preprocessing shared by training and inference."""
from pathlib import Path

import numpy as np
from PIL import Image


def read_grayscale(path):
    with Image.open(path) as image:
        if image.mode not in ("1", "L"):
            raise ValueError(f"{path}: expected grayscale (1/L), got {image.mode}; conversion would lose information")
        return image.convert("L")


def standardize_image(path, width, height, png_path=None):
    """Pad original to (width, height), then rotate clockwise; never resample.

    Returned PIL size is (height, width), so array axis 0 follows reading order.
    width/height describe the original, horizontal canvas.
    """
    image = read_grayscale(path)
    if image.width > width or image.height > height:
        raise ValueError(f"{path}: image {image.size} exceeds canvas {(width, height)}")
    canvas = Image.new("L", (width, height), 255)
    canvas.paste(image, (0, (height - image.height) // 2))
    rotated = canvas.transpose(Image.Transpose.ROTATE_270)
    if png_path is not None:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(png_path, format="PNG", compress_level=9)
    return rotated


def preprocess_image(path, width, height):
    return np.asarray(standardize_image(path, width, height), dtype=np.float32)[..., None] / 255.0
