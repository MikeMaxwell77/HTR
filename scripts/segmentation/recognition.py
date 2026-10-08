"""Saved-model loading and CTC decoding with the original vocabulary."""
import numpy as np
import tensorflow as tf
from keras import ops
from keras.layers import StringLookup
from keras.models import load_model
from .config import MODEL_PATH

characters = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ops.nn.ctc_decode(pred, sequence_lengths=input_len)[0][0][
        :, :21
    ]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = (
            tf.strings.reduce_join(num_to_char(res))
            .numpy()
            .decode("utf-8")
            .replace("[UNK]", "")
        )
        output_text.append(res)
    return output_text


def load_recognition_model(model_path=MODEL_PATH):
    return load_model(str(model_path))
