import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle

# Load the dataset
data = pd.read_csv('isl_gesture_dataset.csv')

# Split the dataset into features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last one (landmarks)
y = data.iloc[:, -1].values   # The last column (gesture labels)

# Encode the labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(126,)),  # 21 landmarks * 3 coordinates * 2 hands
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Number of unique gestures
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model and label encoder for later use
model.save('isl_gesture_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)



#Hello my friend how r u

#  I am not sure