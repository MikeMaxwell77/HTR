"""Aspect-ratio-preserving resize and lossless PNG export for word images."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def read_grayscale(path):
    with Image.open(path) as image:
        if image.mode not in ("1", "L"):
            raise ValueError(f"{path}: expected grayscale (1/L), got {image.mode}; conversion would lose information")
        return image.convert("L")


def standardize_image(path, width, height, png_path=None):
    """Resize to fit (width, height), pad white, then rotate clockwise.

    Returned PIL size is (height, width), so array axis 0 follows reading order.
    width/height describe the original, horizontal canvas.
    """
    if width < 1 or height < 1:
        raise ValueError("Canvas dimensions must be positive")
    image = ImageOps.contain(read_grayscale(path), (width, height), Image.Resampling.LANCZOS)
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
