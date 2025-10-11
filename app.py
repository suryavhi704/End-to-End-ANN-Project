# app.py
"""
Streamlit app to load a saved Keras ANN (model5) and predict MNIST digits from a local image path
or uploaded image. The app auto-detects the model input shape and preprocesses the image to match.
"""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile

st.set_page_config(page_title="MNIST ANN Predictor", layout="centered")

st.title("MNIST ANN Model — Streamlit Deployment")
st.write("Load your saved Keras ANN and predict digits from a local image or upload.")

# -------- Sidebar: model selection --------
st.sidebar.header("Model settings")

# Option: upload model file or use existing path
uploaded_model_file = st.sidebar.file_uploader("model5.h5", type=["h5", "keras", "hdf5"])
model_path_input = st.sidebar.text_input(r"C:\Users\DELL\Desktop\Resume end to end data science project\ANN\model5.h5", value="model5.h5")

@st.cache_resource(show_spinner=False)
def load_ann(model_path: str):
    """Load Keras model (cached)."""
    # load_model can accept directory or .h5 file.
    return load_model(model_path)

def get_model_to_use():
    if uploaded_model_file is not None:
        # save uploaded model to a temp file and load it
        tf_tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tf_tmp.write(uploaded_model_file.getbuffer())
        tf_tmp.flush()
        model = load_ann(tf_tmp.name)
        return model, tf_tmp.name
    else:
        if not os.path.exists(model_path_input):
            st.sidebar.error(f"Model file not found at `{model_path_input}`. Please upload or provide a valid path.")
            return None, None
        model = load_ann(model_path_input)
        return model, model_path_input

model, _ = get_model_to_use()
if model is None:
    st.warning("Please upload a `.h5` model or provide a valid model path in the sidebar.")
    st.stop()

st.sidebar.success("Model loaded ✅")
st.sidebar.write(f"Model input shape: `{model.input_shape}`")

# -------- Image input area --------
st.header("Image input")

use_path = st.radio("How will you provide the image?", ("Upload image", "Provide local image path"), index=0)

image = None
image_path_used = None

if use_path == "Upload image":
    uploaded_file = st.file_uploader("Upload an image of a digit (png/jpg). Example: a handwritten digit.", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path_used = getattr(uploaded_file, "name", "uploaded_image")
else:
    image_path = st.text_input("Enter local image path (absolute or relative)", value="")
    if image_path:
        if not os.path.exists(image_path):
            st.error("Image path not found. Please check the path.")
        else:
            image = Image.open(image_path)
            image_path_used = image_path

# optional: show raw image
if image is not None:
    st.subheader("Original image")
    st.image(image, use_column_width=False)
else:
    st.info("Provide an image to enable prediction.")

# -------- Preprocessing & Prediction --------
def prepare_input_for_model(pil_img: Image.Image, model, target_size=(28, 28)):
    """
    Convert PIL image to model input:
      - grayscale
      - resize to target_size
      - auto-invert if background is light
      - normalize to [0,1]
      - reshape according to model.input_shape (handles flattened (784) and (28,28,1), etc.)
    """
    # convert & resize
    img = pil_img.convert("L").resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32")

    # detect background color and invert if needed:
    # If mean intensity is high (close to white background), invert so digits become white on dark background like MNIST
    if np.mean(arr) > 127:
        arr = 255.0 - arr

    arr /= 255.0  # normalize

    # Inspect model input shape to reshape accordingly
    in_shape = model.input_shape  # e.g. (None, 784) or (None, 28, 28, 1)
    # Convert any None to length check
    # Many Keras models have input_shape like (None, 784) or (None, 28, 28, 1)
    try:
        dims = len(in_shape)
    except Exception:
        dims = 2

    # Flatten-case (Dense network trained on flattened 28x28)
    if dims == 2 and in_shape[1] in (784, 28*28):
        x = arr.flatten().reshape(1, -1)
        return x
    # Channels-last 4D input (batch, height, width, channels)
    if dims == 4:
        # expected channels dimension is in_shape[3]
        channels = in_shape[3] if in_shape[3] is not None else 1
        if channels == 1:
            x = arr.reshape(1, target_size[0], target_size[1], 1)
            return x
        elif channels == 3:
            # replicate grayscale to 3 channels
            x = np.stack([arr, arr, arr], axis=-1)
            x = x.reshape(1, target_size[0], target_size[1], 3)
            return x
    # 3D input (batch, height, width)
    if dims == 3:
        x = arr.reshape(1, target_size[0], target_size[1])
        return x

    # fallback: flatten
    x = arr.flatten().reshape(1, -1)
    return x

def predict_image(pil_img: Image.Image, model):
    x = prepare_input_for_model(pil_img, model)
    preds = model.predict(x)
    # preds shape likely (1,10)
    probs = preds[0]
    # If outputs not normalized, apply softmax
    s = np.sum(probs)
    if not (0.99 <= s <= 1.01):
        probs = tf.nn.softmax(probs).numpy()
    predicted = int(np.argmax(probs))
    confidence = float(np.max(probs))
    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(int(i), float(probs[i])) for i in top3_idx]
    return predicted, confidence, top3, probs

# Button: Predict
if image is not None:
    if st.button("Predict"):
        with st.spinner("Preprocessing image and predicting..."):
            try:
                pred, conf, top3, probs = predict_image(image, model)
            except Exception as e:
                st.error(f"Error while predicting: {e}")
            else:
                st.success(f"Predicted digit: **{pred}** (confidence: {conf:.2%})")
                st.write("Top 3 predictions (digit : probability):")
                for dig, p in top3:
                    st.write(f"- **{dig}** : {p:.2%}")
                # Show probability bar
                st.subheader("All class probabilities")
                prob_labels = [f"{i}" for i in range(len(probs))]
                st.bar_chart(data=np.array(probs).reshape(1, -1), height=200, use_container_width=True)
                st.balloons()
