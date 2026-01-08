import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# ---------------- Dataset Paths ---------------- #

# Directory containing real and fake face images
REAL_DIR = "real_and_fake_face_detection/real_and_fake_face/training_real/"
FAKE_DIR = "real_and_fake_face_detection/real_and_fake_face/training_fake/"


# ---------------- Dataset Inspection ---------------- #

# List image files
real_images = os.listdir(REAL_DIR)
fake_images = os.listdir(FAKE_DIR)


# Utility function to load and resize images for visualization
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img[..., ::-1]   # Convert BGR to RGB


# Display sample real images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_image(REAL_DIR + real_images[i]))
    plt.suptitle("Sample Real Images", fontsize=20)
    plt.axis("off")
plt.show()

# Display sample fake images
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_image(FAKE_DIR + fake_images[i]))
    plt.suptitle("Sample Fake Images", fontsize=20)
    plt.axis("off")
plt.show()


# ---------------- Data Augmentation ---------------- #

# Path containing both real and fake folders
DATASET_PATH = "real_and_fake_face"

# Image augmentation and normalization
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)

# Training dataset
train_data = data_generator.flow_from_directory(
    DATASET_PATH,
    target_size=(96, 96),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

# Validation dataset
validation_data = data_generator.flow_from_directory(
    DATASET_PATH,
    target_size=(96, 96),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)


# ---------------- Model Architecture ---------------- #

# Load MobileNetV2 as base feature extractor
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(96, 96, 3)
)

tf.keras.backend.clear_session()

# Build the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(2, activation="softmax")
])

# Freeze base model layers
model.layers[0].trainable = False


# ---------------- Model Compilation ---------------- #

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ---------------- Learning Rate Scheduler ---------------- #

def learning_rate_schedule(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch <= 15:
        return 0.0001
    else:
        return 0.00001


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)


# ---------------- Model Training ---------------- #

history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data,
    callbacks=[lr_scheduler]
)


# ---------------- Save Trained Model ---------------- #

model.save("deepfake_detection_model.h5")


# ---------------- Training Visualization ---------------- #

epochs_range = range(20)

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

# Loss graph
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, train_loss)
plt.plot(epochs_range, val_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Training", "Validation"])
plt.grid(True)
plt.style.use("classic")

# Accuracy graph
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, train_acc)
plt.plot(epochs_range, val_acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(["Training", "Validation"])
plt.grid(True)
plt.style.use("classic")
plt.show()
