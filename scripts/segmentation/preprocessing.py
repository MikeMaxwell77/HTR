"""Resize and normalize word crops to match model training."""
import tensorflow as tf
from keras import ops

image_width = 128
image_height = 32

def distortion_free_resize(image, img_size=(image_width, image_height)):
    w, h = img_size
    #https://www.tensorflow.org/api_docs/python/tf/image/resize
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    #check tha amount of padding needed to be done.
    pad_height = h - ops.shape(image)[0]
    pad_width = w - ops.shape(image)[1]

    #add padding to both sides
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    #we transpose the image because handwriting is more of an up down
    image = ops.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image, img_size=(image_width, image_height)):
    #convert to tensor for CNN and other tensorflwo functions
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = distortion_free_resize(image, img_size)
    #scale
    image = ops.cast(image, tf.float32) / 255.0
    return image
