import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ------------------------------
# Load model
# ------------------------------
MODEL_PATH = "agrimind_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.error(f"❌ {MODEL_PATH} not found. Make sure it is in the same folder as app.py")
    st.stop()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AgriMind 🌱", layout="centered")
st.title("🌱 AgriMind — Crop Health Detector")
st.write("Upload a leaf image, and AgriMind will tell you if it is Healthy or Diseased.")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess image
    img = img.resize((128,128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
# Prediction
prediction = model.predict(img_array)[0][0]

if prediction < 0.5:
    st.error(f"⚠️ Diseased Leaf (Confidence: {(1-prediction)*100:.2f}%)")
    st.info("Suggested Action: Remove affected leaves, use fungicide if needed.")
else:
    st.success(f"🌿 Healthy Leaf (Confidence: {prediction*100:.2f}%)")


# Footer
st.markdown("---")
st.write("💡 Built by Vaishnavi S | AI-powered crop health detection")

