import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import cv2
import os

# --------------------------------------------
# ✅ CONFIGURATION
# --------------------------------------------
logo_path = "C:\\Users\\kul_a\\Usecases\\MosChip-DigitaSky_Ultimate-Logo.jpg"

logo = Image.open(logo_path)
st.sidebar.image(logo,width =200)
# Path to your model file (.h5)
model_path = r"C:\Users\kul_a\Usecases\Emotion_little_vgg.h5"

# Labels used by your model
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise','fear','disgust']

# --------------------------------------------
# ✅ MODEL LOADING (cached)
# --------------------------------------------

@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_emotion_model()

# --------------------------------------------
# ✅ Streamlit UI
# --------------------------------------------

st.title("Emotion Detector")
st.write("Upload an image and I will try to detect the emotion!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --------------------------------------------
    # ✅ Preprocess the image
    # --------------------------------------------
    image = image.resize((48, 48))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # --------------------------------------------
    # ✅ Make prediction
    # --------------------------------------------
    if model:
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]

        st.success(f"Predicted Emotion: **{predicted_label}**")
    else:
        st.warning("Model not loaded. Cannot predict.")
