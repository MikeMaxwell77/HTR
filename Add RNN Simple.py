# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:19:54 2025
@author: mikey
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, SimpleRNN
from keras.layers import Reshape, TimeDistributed
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image

# Load dataset
df = pd.read_csv('word_dataset_info.csv')

# Filter for the 4 target words
values_to_keep = ['the', 'of', 'to', 'and']
filtered_df = df[df['target'].isin(values_to_keep)]
print("step 1")

# Function to load and preprocess images
X_size = 600
Y_size = 150
def load_and_preprocess_image(image_path, target_size=(X_size, Y_size)):
    try:
        img = image.load_img("words/" + image_path, color_mode='grayscale', target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Load and preprocess all images
X = []
valid_paths = []

for image_path in filtered_df["image_path"]:
    img_array = load_and_preprocess_image(image_path)
    if img_array is not None:
        X.append(img_array)
        valid_paths.append(image_path)

X = np.array(X)
print("step 2")

# Get labels for only the successfully loaded images
y = filtered_df.loc[filtered_df["image_path"].isin(valid_paths)]["target"].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)
print("step 3")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
print("step 4")

# Build CNN-RNN hybrid model
model = Sequential()

# CNN layers for feature extraction
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(Y_size, X_size, 1)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# calculate the dimensions after CNN processing
# after 3 max pooling layers with pool_size (2,2), dimensions are reduced by factor of 8

#https://keras.io/api/layers/recurrent_layers/rnn/
# Reshape the CNN output for RNN input
# We treat each row as a time step, and the features are the columns
model.add(Reshape((18, -1)))
#https://keras.io/api/layers/recurrent_layers/rnn/
# RNN layers
model.add(SimpleRNN(64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(16, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

# Output layer for 4 classes
model.add(Dense(4, activation='softmax'))

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

print("\n//////////////////////fit//////////////////////")
# Train the model
history = model.fit(
    X_train, 
    y_train, 
    epochs=5, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Visualization of training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("CNN_RNN_Basic_combo.keras")

