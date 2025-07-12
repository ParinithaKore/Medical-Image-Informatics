import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



# Set the paths to the training and testing directories

train_dir = r"C:\Users\Deepak\Downloads\Lung Disease Dataset\train"

test_dir = r"C:\Users\Deepak\Downloads\Lung Disease Dataset\test"



# Data augmentation and loading images from directory

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



# Load training data

train_generator = datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),  # Resize images to this size

    batch_size=32,

    class_mode='categorical',  # Use 'categorical' for multi-class classification

    shuffle=True

)



# Load validation data (you can use the same generator with a split)

val_generator = datagen.flow_from_directory(

    train_dir,

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    shuffle=True

)



# Build the CNN model

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(5, activation='softmax'))  # Adjust the number of output classes



# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Train the model

history = model.fit(train_generator, 

                    validation_data=val_generator, 

                    epochs=10, 

                    verbose=1)



# Save the model

model.save('lungs_disease_model.h5')
