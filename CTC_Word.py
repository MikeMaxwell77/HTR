# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:19:06 2025

@author: mikey
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:19:54 2025

@author: mikey
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import keras
from keras import ops
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from keras.layers import Reshape, LSTM, Bidirectional
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
#set working directory
os.chdir(r"C:\Users\mikey\OneDrive\Documents\USCB\Data Mining\Project")

X_size=600
Y_size=100
#define the image preprocessing function
def load_and_preprocess_image(image_path, target_size=(X_size, Y_size)):
    try:
        img = image.load_img("words/" + image_path, color_mode='grayscale', target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# CTC loss layer implementation
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, inputs):
        y_true, y_pred, input_length, label_length = inputs
        batch_loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(batch_loss)
        return y_pred

#main execution
print("Loading and preparing data...")

#load dataset
df = pd.read_csv('word_dataset_info.csv')

#drop empty rows
df = df.dropna(subset=['target'])
#make df target all lowercase
df['target'] = df['target'].str.lower()

df = df[df['target'].str.match(r'^[a-z]+$')]
#make remove all the special character words
# Filter for the 4 target words
#values_to_keep = ['the', 'of', 'to', 'and']
filtered_df = df#['target'].isin(values_to_keep)
"""
we need to set the ctc loss function max and from here we have it be 17
////////////////////////////
max_length = 0
for i in range(len(filtered_df)):
    if(max_length<len(filtered_df['target'].iloc[i])):
        max_length = len(filtered_df['target'].iloc[i])
////////////////////////
max length is 17
        """
    

# Load and preprocess all images
X = []
valid_indices = []
#add image path to valid_indices
for i, image_path in enumerate(filtered_df["image_path"]):
    #test and limit this on 
    if i >= 100:
       break
    print(i)
    img_array = load_and_preprocess_image(image_path)
    if img_array is not None:
        X.append(img_array)
        valid_indices.append(i)

X = np.array(X)
print(f"Loaded {len(X)} valid images")

#create character map for CTC (including blank for CTC)
characters = ['-'] + list('abcdefghijklmnopqrstuvwxyz')  # '-' is the blank character
char_to_num = {char: i for i, char in enumerate(characters)}
num_to_char = {i: char for i, char in enumerate(characters)}
num_chars = len(characters)

print(f"Character set size: {num_chars}")

#encode targets as character sequences
def encode_target(word):
    return [char_to_num[c] for c in word]

# get word labels
words = filtered_df.iloc[valid_indices]["target"].values
y = [encode_target(word) for word in words]

# get the maximum length of any word (17)
max_length = max(len(word) for word in y)
print(f"Maximum word length: {max_length}")

# pad sequences to same length for batch processing
y_padded = tf.keras.preprocessing.sequence.pad_sequences(
    y, maxlen=max_length, padding='post', value=char_to_num['-']
)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_padded, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# build CTC model
def build_ctc_model(input_shape=(X_size, Y_size, 1), num_chars=len(characters)):
    # Input
    input_img = Input(shape=input_shape, name="image")
    
    # CNN feature extraction
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    # Calculate new dimensions after pooling
    new_height = input_shape[0] // 8
    new_width = input_shape[1] // 8
    
    # Reshape to sequences - treat each row as a sequence step
    x = Reshape((new_height, new_width * 128))(x)
    
    # RNN layers for sequence processing
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    
    # Output layer with softmax over characters at each time step
    outputs = Dense(num_chars, activation='softmax', name="char_output")(x)
    
    # Define the prediction model
    prediction_model = Model(inputs=input_img, outputs=outputs)
    
    # Define inputs for CTC loss
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    
    # Define CTC loss
    ctc_output = CTCLayer(name='ctc_loss')([labels, outputs, input_length, label_length])
    
    # Training model with CTC loss
    training_model = Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=ctc_output
    )
    
    # Compile model with dummy loss since the real loss is added in the CTC layer
    training_model.compile(optimizer='adam')
    
    return training_model, prediction_model

#create the model
print("Building and compiling model...")
training_model, prediction_model = build_ctc_model(input_shape=(X_size, Y_size, 1))

#print model summary
training_model.summary()

#batch size and epochs
batch_size = 10
epochs = 50

#prepare data for training
def train_generator():
    while True:
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train[batch_indices].reshape(-1, X_size, Y_size, 1)
            batch_y = y_train[batch_indices]
            batch_y = batch_y.astype(np.int32)
            
            # Calculate input lengths
            # This is the number of time steps in the output sequence from the CNN+RNN
            input_length = np.ones((len(batch_indices), 1), dtype=np.int32) * (X_size // 8)
            
            # Calculate label lengths
            label_length = np.array([[np.sum(l != char_to_num['-'])] for l in batch_y], dtype=np.int32)
            
            # Yield the batch data
            inputs = {
                'image': batch_x,
                'labels': batch_y,
                'input_length': input_length,
                'label_length': label_length
            }
            # Dummy output for the CTC loss function
            outputs = {'ctc_loss': np.zeros((len(batch_indices),))}
            
            yield (inputs, outputs)

#prepare data for validation
def val_generator():
    while True:
        indices = np.arange(len(X_test))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train[batch_indices].reshape(-1, X_size, Y_size, 1)
            batch_y = y_test[batch_indices]
            batch_y = batch_y.astype(np.int32)
            
            # Calculate input lengths
            input_length = np.ones((len(batch_indices), 1), dtype=np.int32) * (X_size // 8)
            
            # Calculate label lengths
            label_length = np.array([[np.sum(l != char_to_num['-'])] for l in batch_y], dtype=np.int32)
            
            # Yield the batch data
            inputs = {
                'image': batch_x,
                'labels': batch_y,
                'input_length': input_length,
                'label_length': label_length
            }
            # Dummy output for the CTC loss function
            outputs = {'ctc_loss': np.zeros((len(batch_indices),))}
            
            yield (inputs, outputs)

#train the model
print("\n//////////////////////fit//////////////////////")
history = training_model.fit(
    train_generator(),
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_generator(),
    validation_steps=len(X_test) // batch_size,
    epochs=epochs
)

#implement CTC decoder for prediction
def decode_batch_predictions(pred):
    """
    Decode the output of the CTC model to get word predictions
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Convert indices to characters
    output_text = []
    for res in results:
        decoded = []
        for c in res:
            if c < 0:
                # Skip invalid values
                continue
            # Convert index to character
            c = int(c)
            if c > 0:  # Skip the blank character
                decoded.append(num_to_char[c])
        output_text.append(''.join(decoded))
    
    return output_text

# Evaluate the model
def evaluate_model():
    print("Evaluating model...")
    # Get predictions
    predictions = []
    ground_truth = []
    
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        
        # Get predictions
        pred = prediction_model.predict(batch_x)
        decoded_preds = decode_batch_predictions(pred)
        
        # Get ground truth
        for j in range(len(batch_y)):
            # Convert label to text
            label = ''.join([num_to_char[c] for c in batch_y[j] if c > 0])
            ground_truth.append(label)
        
        # Add predictions
        predictions.extend(decoded_preds)
    
    # Calculate accuracy
    accurate = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predictions[i]:
            accurate += 1
    
    accuracy = accurate / len(ground_truth)
    print(f"Word Recognition Accuracy: {accuracy:.4f}")
    
    # Print some examples
    print("\nSample Predictions:")
    for i in range(min(10, len(predictions))):
        print(f"Predicted: '{predictions[i]}', True: '{ground_truth[i]}'")

# Evaluate the model
evaluate_model()

# Plot training history
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CTC Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the prediction model
prediction_model.save("word_recognition_model.keras")
print("Model saved as 'word_recognition_model.keras'")