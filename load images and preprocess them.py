# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 19:40:33 2025

@author: mikey
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:27:15 2025

@author: mikey
"""
"""
/////////////DOCUMENTATION/////////////////
PILLOW: https://pillow.readthedocs.io/en/stable/reference/Image.html


"""
"""
# Save the image as a PNG file
img.save("output.png")

# Save the image with an explicit format
img.save("output.jpeg", format="JPEG")
"""
import numpy as np
import pandas as pd
#image processing
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image


def load_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image {image_path}")
        return None
    return image

df = pd.read_csv('word_dataset_info.csv')

image_path = df.iloc[1,0]
# Load an image and resize it to the target size
img = image.load_img("words/"+image_path, target_size=(150, 1500))
img.show()

