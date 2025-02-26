import certifi
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union
from io import BytesIO
import json

# Constants
TRAINED_DB_PATH = "db"

# SSL Certificate setup
os.environ['SSL_CERT_FILE'] = certifi.where()

#load models
@st.cache_resource
def load_model() -> tf.keras.Model:
    """
    Load the pre-trained ResNet50 model.

    Returns:
        tf.keras.Model: Pre-trained ResNet50 model.
    """
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_features(image_path: Union[str, BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    """
    Extract features from an image using the given model.

    Args:
        image_path (Union[str, BytesIO]): Path to the image file or file-like object.
        model (tf.keras.Model): Pre-trained model for feature extraction.

    Returns:
        Union[np.ndarray, None]: Extracted features or None if extraction fails.
    """
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img).flatten()
        tf.keras.backend.clear_session()
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        tf.keras.backend.clear_session()
        return None

#feature in database
@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(db_path: str, model: tf.keras.Model) -> tuple[np.ndarray, list[str]]:
    """
    Extract features from all images in the database.

    Args:
        db_path (str): Path to the image database directory.
        model (tf.keras.Model): Pre-trained model for feature extraction.

    Returns:
        tuple[np.ndarray, list[str]]: A tuple containing the feature vectors and corresponding image paths.
    """
    feature_list = []
    image_paths = []
    try:
        for img_path in os.listdir(db_path):
            if img_path.endswith(".jpg"):
                path = os.path.join(db_path, img_path)
                features = extract_features(path, model)
                if features is not None:
                    feature_list.append(features)
                    image_paths.append(path)
        feature_vectors = np.vstack(feature_list)
        return feature_vectors, image_paths
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")
        return np.array([]), []


def find_similar_images(image_path: Union[str, BytesIO], feature_vectors: np.ndarray, image_paths: list[str],
                        model: tf.keras.Model, threshold: float = 0.5, top_n: int = 5) -> list[str]:

    """
    Find similar images based on the given image and feature vectors.

    Args:
        image_path (Union[str, BytesIO]): Path to the query image file or file-like object.
        feature_vectors (np.ndarray): Feature vectors of the images in the database.
        image_paths (list[str]): List of image paths corresponding to the feature vectors.
        model (tf.keras.Model): Pre-trained model for feature extraction.
        threshold (float, optional): Similarity threshold. Defaults to 0.5.
        top_n (int, optional): Number of top similar images to return. Defaults to 5.

    Returns:
        list[str]: List of paths to the similar images.
    """
    query_features = extract_features(image_path, model)
    if query_features is None:
        return []

    similarities = cosine_similarity([query_features], feature_vectors)
    similarities_indices = [i for i in range(len(similarities[0])) if similarities[0][i] > threshold]
    similarities_indices = sorted(similarities_indices, key=lambda i: similarities[0][i], reverse=True)
    similar_images = [image_paths[i] for i in similarities_indices[:top_n]]
    tf.keras.backend.clear_session()
    return similar_images


def init_session_state():
    """
    Initialize session state variables.
    """
    if "feature_vectors" not in st.session_state:
        st.session_state.feature_vectors = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = None

def load_json_data(img_path, json_path="/Users/luongdinhdung/Downloads/vs_search/ecommerce-visual-search/data.json"):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf_8") as json_file:
            json_data = json.load(json_file)

            for item in json_data:
                string1 = item.get("path")
                string2 = img_path
                if (os.path.splitext(string1)[0] == os.path.splitext(string2)[0]):
                    return item
    return None

# item = load_json_data("db/jeans7.jpg")
# print(item)

def print_info(img_json) -> None:
    if (img_json is not None):
        str_name = f"""Name: {img_json.get("name")}"""
        str_price = f"""Price: {img_json.get("price")}"""
        str_type = f"""Type: {img_json.get("type")}"""
        str_size = f"""Size: {img_json.get("size")}"""
        st.write(str_name)
        st.write(str_type)
        st.write(str_size)
        st.write(str_price)
    # st.write(f"{str_name}\n{str_type}\n{str_size}\n{str_price}")

def main():
    """
    Main function to run the Streamlit app for visual image search.
    """
    
    st.title("Visual Image Search Engine")
    st.write(
        "Upload an image and find similar images from the database. It uses ResNet50 model for feature extraction and cosine similarity for finding similar images. Currently it supports only .jpg images.")
    
    init_session_state()
    model = load_model()

    if st.session_state.feature_vectors is None:
        with st.spinner("Loading database..."):
            st.session_state.feature_vectors, st.session_state.image_paths = get_feature_vectors_from_db(
                TRAINED_DB_PATH, model)
            st.success("Database loaded successfully!")
    
    

# with col1:
    uploaded_img_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_img_file is not None:
        uploaded_img = Image.open(uploaded_img_file)
        # uploaded_img.thumbnail(150,150)
        # st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
        st.image(uploaded_img, caption="Uploaded Image", width=200)
        st.write("")

# with col2:

    with st.spinner("Extracting features..."):
        if (uploaded_img_file is None):
            return
        query_features = extract_features(uploaded_img_file, model)
        if query_features is not None:
            st.success("Features extracted successfully!")
        else:
            st.error("Failed to extract features from the uploaded image.")
            return

    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.01)
    top_n = st.slider("Number of Similar Images", 1, 10, 5)

# with col3:
    if st.button("Find Similar Images"):
        with st.spinner("Searching for similar images..."):
            similar_images = find_similar_images(
                uploaded_img_file,
                st.session_state.feature_vectors,
                st.session_state.image_paths,
                model,
                threshold,
                top_n
            )
            if similar_images:
                st.success("Similar images found!")
                #2 columns here
                
                for i, similar_image in enumerate(similar_images):
                    col1, col2 = st.columns(2)
                    image = Image.open(similar_image)
                    with col1:
                        st.image(image, caption=f"Similar Image {i + 1}", width=200)
                        st.write(f"{similar_image}")
                    img_json = load_json_data(img_path=similar_image)
                    print(img_json)
                    # print(similar_image)
                    with col2:
                        print_info(img_json)
                    # st.write(f"Name: {img}")
                    # st.write(f"Price: {img_json.get("price")}")
                    # st.json(img_json, expanded=True)
            else:
                st.write("No similar images found!")

            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()