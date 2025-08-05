import streamlit as st
from PIL import Image
import numpy as np
import cv2 # Import OpenCV for image processing
import joblib # Import joblib to load the model
import os # For path operations
import pandas as pd # Import pandas for DataFrame creation

st.set_page_config(page_title="Real Time Detection", page_icon="🔬")
st.title("🔬 Real-Time Malaria Image Detection")

st.write("Upload a cell image for classification (Parasitized or Uninfected).")

# Define paths for all three model files
MODEL_PATHS = {
    "Random Forest using GridSearchCV": 'random_forest_malaria_model.joblib',
    "Random Forest using Bayesian Optimization": 'random_forest_bayesian.joblib',
    "Random Forest using RandomizedSearchCV": 'random_forest_malaria_detector.joblib'
}

# --- Model Loading ---
# Ensure all model files exist and load them using st.cache_resource
loaded_models = {}
for model_name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        st.error(f"Error: Model file for '{model_name}' not found at '{path}'. "
                 "Please ensure all model .joblib files are in the same directory as this script, "
                 "or provide the correct path.")
        st.stop() # Stop execution if any model is not found

    @st.cache_resource(hash_funcs={joblib.load: lambda _: None}) # Add hash_funcs for joblib.load
    def load_rf_model_cached(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Failed to load the model from {model_path}. Error: {e}")
            st.stop()

    loaded_models[model_name] = load_rf_model_cached(path)
    st.sidebar.success(f"Model '{model_name}' loaded successfully!")


# Re-implement the extract_areas function from random_forest.py
# This function must be exactly the same as used during training for all models
def extract_areas(
    image_array, # Pass numpy array instead of file path
    n              = 5,
    min_area       = 0,
    use_adaptive   = True,
):
    # Convert PIL Image to OpenCV format (numpy array) if not already
    img  = image_array

    # Ensure image is 3-channel if it's grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # ❶ Threshold
    if use_adaptive:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            11, 2
        )
    else:
        # For a 0-255 range threshold, use OTSU for automatic thresholding
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) >= min_area]
    areas = sorted(areas, reverse=True)[:n]
    areas += [0]*(n - len(areas)) # Pad with zeros if less than n areas
    return areas

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Uploaded Image", use_container_width=True)

    st.write("🧪 Predicting with all models...")

    # Convert PIL Image to OpenCV format (NumPy array)
    img_cv = np.array(img_pil.convert('RGB')) # Convert to RGB first to ensure 3 channels
    img_cv = img_cv[:, :, ::-1].copy() # Convert RGB to BGR as OpenCV uses BGR by default

    # Extract features using the same function as training
    features = extract_areas(img_cv)

    # Convert features to a Pandas DataFrame row, matching the training feature names
    features_df = pd.DataFrame([features], columns=[f"area{i+1}" for i in range(len(features))])

    # --- Display Predictions from Each Model ---
    for model_name, model_obj in loaded_models.items():
        st.markdown(f"---") # Separator for each model's results
        st.subheader(f"Results from {model_name}")

        try:
            prediction = model_obj.predict(features_df)[0]
            prediction_proba = model_obj.predict_proba(features_df)[0] # Array of probabilities

            class_labels = model_obj.classes_ # Get the class labels from this specific model

            # Find the index of "Parasitized" and "Uninfected" in the class_labels array
            parasitized_idx = -1
            uninfected_idx = -1
            try:
                parasitized_idx = list(class_labels).index("Parasitized")
            except ValueError:
                pass # 'Parasitized' might not be in class_labels if model was trained on different labels

            try:
                uninfected_idx = list(class_labels).index("Uninfected")
            except ValueError:
                pass # 'Uninfected' might not be in class_labels

            st.success(f"Prediction: **{prediction}**")

            # Display probabilities correctly based on their class index
            if parasitized_idx != -1:
                st.info(f"Probability of Parasitized: **{prediction_proba[parasitized_idx]:.2f}**")
            if uninfected_idx != -1:
                st.info(f"Probability of Uninfected: **{prediction_proba[uninfected_idx]:.2f}**")

            # Optional: Highlight the highest probability more clearly
            if prediction == "Parasitized":
                if parasitized_idx != -1:
                    st.warning(f"Highest Probability: **{prediction_proba[parasitized_idx]:.2f}** for Parasitized")
            else: # Prediction is Uninfected
                if uninfected_idx != -1:
                    st.success(f"Highest Probability: **{prediction_proba[uninfected_idx]:.2f}** for Uninfected")

        except Exception as e:
            st.error(f"An error occurred during prediction for {model_name}: {e}")
            st.write("Please ensure the uploaded image is valid and the model is correctly loaded.")