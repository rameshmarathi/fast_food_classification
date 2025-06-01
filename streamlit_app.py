# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Fast Food Classifier", layout="centered")

# --- Other Configuration ---
# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

MODEL_PATH = 'best_resnet50v2.h5'
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASS_LABELS = ['burger', 'cake', 'chicken', 'donuts', 'fries', 'hot_dog', 'ice_cream', 'pizza', 'sandwich', 'waffles']  # Update to match your dataset

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Ensure '{MODEL_PATH}' is in the correct location and is a valid Keras model file.")
        return None

model = load_my_model()

# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """Preprocesses the uploaded image to fit model input requirements."""
    image_resized = image_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    if image_resized.mode != 'RGB':
        image_resized = image_resized.convert('RGB')
    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch

# --- Streamlit App UI ---
st.title("🍔 Fast Food Image Classifier")
st.markdown("""
Upload an image of a fast food item. This app uses a ResNet50V2 model to predict the type of food.
""")

uploaded_file = st.file_uploader("Choose a fast food image...", type=["jpg", "jpeg", "png"])

if model is None:
    st.warning("Model could not be loaded. Please check the console or logs.")
else:
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            with st.spinner('Analyzing the image...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class_index = np.argmax(prediction[0])

                if predicted_class_index < len(CLASS_LABELS):
                    predicted_class_label = CLASS_LABELS[predicted_class_index]
                    confidence = np.max(prediction[0]) * 100

                    st.subheader("🔍 Prediction:")
                    st.success(f"**{predicted_class_label.replace('_', ' ').title()}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")

                    st.write("---")
                    st.write("Confidence Scores for all classes:")
                    for i, label in enumerate(CLASS_LABELS):
                        st.write(f"- {label.replace('_', ' ').title()}: {prediction[0][i]*100:.2f}%")
                else:
                    st.error(f"Prediction index {predicted_class_index} is out of bounds for CLASS_LABELS.")
                    st.error("Please ensure CLASS_LABELS in app.py matches the training classes and their order.")
        except Exception as e:
            st.error(f"An error occurred processing the image: {e}")
            st.error("Please ensure you uploaded a valid image file.")
    else:
        st.info("Awaiting an image upload to start analysis.")

st.markdown("---")
st.markdown("Developed for fast food classification using ResNet50V2 and TensorFlow/Keras.")
