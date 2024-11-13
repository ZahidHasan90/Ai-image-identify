import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input
from keras.utils import to_categorical
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Image size for resizing
IMG_SIZE = 64

# Function to load images and labels from specified folders
def load_data():
    data = []
    labels = []
    # Specify the paths for 'crow', 'hen', and 'stork' folders
    categories = {
        'crow': 'C:/project/cat/crow',
        'hen': 'C:/project/cat/hen',
        'stork': 'C:/project/cat/stork'
    }

    for category, path in categories.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist. Please create it and add images.")

        # Label encoding: 'crow' is 0, 'hen' is 1, 'stork' is 2
        label = 0 if category == 'crow' else 1 if category == 'hen' else 2
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img)
                    labels.append(label)
    
    data = np.array(data, dtype=np.float32) / 255.0  # Normalize and convert to float32
    labels = to_categorical(labels, num_classes=3)  # One-hot encode labels
    return data, labels

# Load the data and labels
data, labels = load_data()

# Shuffle data
idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data[idx]
labels = labels[idx]

# Split data into training and validation sets
num_samples = len(data)
num_train = int(num_samples * 0.8)
x_train = data[:num_train]
y_train = labels[:num_train]
x_val = data[num_train:]
y_val = labels[num_train:]

# Build the model with an Input layer
model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(3))  # Change output layer to 3 neurons
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Create the GUI using tkinter
root = tk.Tk()
root.title("Crow, Hen, or Stork Classifier")
root.geometry("500x500")

# Function to handle the image prediction
def predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Display selected image
    image_tk = Image.open(file_path).resize((IMG_SIZE, IMG_SIZE))
    image_tk = ImageTk.PhotoImage(image_tk)
    img_label.configure(image=image_tk)
    img_label.image = image_tk

    # Prepare the image for prediction
    image_selected = cv2.imread(file_path)
    if image_selected is not None:
        image_selected = cv2.cvtColor(image_selected, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image_selected = cv2.resize(image_selected, (IMG_SIZE, IMG_SIZE))
        image_selected = np.expand_dims(image_selected, axis=0) / 255.0  # Normalize

        # Predict and display the result
        prediction = model.predict(image_selected)
        class_idx = np.argmax(prediction)
        if class_idx == 0:
            result_label.configure(text="This is a crow!")
        elif class_idx == 1:
            result_label.configure(text="This is a hen!")
        else:
            result_label.configure(text="This is a stork!")

# Create GUI widgets
img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, font=("Helvetica", 18))
result_label.pack()

button = tk.Button(root, text="Choose Image", command=predict)
button.pack()

root.mainloop()
