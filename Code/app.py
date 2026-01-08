import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Load the pre-trained deepfake detection model
model = load_model('deepfake_detection_model.h5')


# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (96, 96))       # Resize to model input size
    image = img_to_array(image)               # Convert image to array
    image = np.expand_dims(image, axis=0)     # Add batch dimension
    image = image / 255.0                     # Normalize pixel values
    return image


# Function to predict whether the image is real or fake
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "Fake" if predicted_class == 0 else "Real"


# ---------------- Streamlit UI ---------------- #

st.markdown(
    "<h1 style='text-align: center; color: #555;'>AI-Based Deepfake Image Detection</h1>",
    unsafe_allow_html=True
)

st.image("coverpage.png")


# Section explaining deepfakes
st.header("What Are Deepfakes?")
st.write("""
Deepfakes refer to digitally manipulated images or videos where a person’s appearance is altered using artificial intelligence.
These manipulations are typically created using deep learning techniques such as neural networks, making them extremely
realistic and difficult to identify with the naked eye.

While deepfake technology has positive applications in fields like film and education, it also introduces serious concerns.
It can be misused to spread false information, impersonate individuals, or damage reputations. As a result, detecting
deepfakes has become increasingly important.

This application uses a trained AI model to analyze uploaded images and identify subtle inconsistencies—such as abnormal
textures, lighting issues, or facial irregularities—that help determine whether an image is real or artificially generated.
""")


# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image for verification",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert uploaded file into OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Perform prediction
    result = predict_image(image)

    # Define output message and color based on prediction
    if result == "Fake":
        color = "red"
        description = (
            "The system has identified this image as a deepfake. "
            "Fake images often contain hidden artifacts such as unnatural textures, "
            "inconsistent facial alignment, or irregular lighting patterns. "
            "The model detects these subtle cues to differentiate manipulated images from real ones."
        )

    else:
        color = "green"
        description = (
            "The system has classified this image as authentic. "
            "Real images typically do not exhibit the visual inconsistencies found in deepfakes. "
            "Based on the learned patterns from the training dataset, this image aligns with real media characteristics."
        )

    # Display result
    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>Prediction: {result}</h2>",
        unsafe_allow_html=True
    )

    st.write(description)


# Model performance section
st.title("Model Performance Overview")

st.markdown("### Training Accuracy")
st.markdown("**Achieved Accuracy:** 95%")
st.image("Figure_2.png")

st.markdown("### Training Loss")
st.image("Figure_1.png")


# Footer
st.markdown("""
---
### 📩 Contact Information
For questions or collaboration opportunities, reach out at  
📧 **contact@example.com**

### 🌐 Stay Connected
[Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com) | [Facebook](https://facebook.com)
""")
