import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Load the trained deepfake detection model
model = load_model('deepfake_detection_model.h5')


# Function to prepare an image before prediction
def preprocess_image(image_path):
    # Read the image from the given file path
    image = cv2.imread(image_path)

    # Resize image to match model input dimensions
    image = cv2.resize(image, (96, 96))

    # Convert image into array format
    image = img_to_array(image)

    # Add batch dimension for model compatibility
    image = np.expand_dims(image, axis=0)

    # Normalize pixel values to range [0, 1]
    image = image / 255.0

    return image


# Function to classify the image as Real or Fake
def predict_image(image_path):
    processed_image = preprocess_image(image_path)

    # Get prediction from the model
    prediction = model.predict(processed_image)

    # Determine the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]

    return "Fake" if predicted_class == 0 else "Real"


# ---------------- Example Execution ---------------- #

# Sample image path for testing the model
image_path = "real_and_fake_face_detection/real_and_fake_face/training_real/real_00001.jpg"

# Run prediction on the image
result = predict_image(image_path)

# Display the prediction result
print(f"Prediction Result: The image is classified as {result}")
