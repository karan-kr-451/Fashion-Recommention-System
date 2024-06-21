import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import faiss
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
from src.pipeline.feature_extraction import FeatureExtractor

st.set_page_config(page_title="Fashion Recommender System", layout="wide")

# Load the feature list and filenames
feature_list = np.array(pickle.load(open('features.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize Faiss index
dimension = feature_list.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(feature_list)

# Initialize the model
@st.cache_resource
def load_model():
    extractor = FeatureExtractor()
    return extractor

extractor = load_model()

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join('uploads', uploaded_file.name)
    except Exception as e:
        st.error(f"Error occurred in file upload: {e}")
        return None

# Feature extraction function
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation function
def recommend(features, index, k=5):
    features = np.expand_dims(features, axis=0)
    distances, indices = index.search(features, k)
    return indices[0]

# Streamlit UI
# st.set_page_config(page_title="Fashion Recommender System", layout="wide")

# Header and sidebar
st.title('Fashion Recommender System')
st.sidebar.title('Upload Image')

uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded file
        st.sidebar.image(Image.open(file_path), caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        try:
            features = feature_extraction(file_path, extractor.model)
        except Exception as e:
            st.error(f"Error occurred during feature extraction: {e}")
            st.stop()

        # Recommendation
        indices = recommend(features, index)

        # Show recommended images
        st.header('Recommended Images')
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[i]], use_column_width=True)

        # Remove uploaded file after processing
        os.remove(file_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Karan Kumar")
