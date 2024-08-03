import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load your custom model
model = load_model("Alzheimer's VGG16 model.keras")

# Function to preprocess the image
def preprocess_image(img, target_size=(224, 224)):
    # Resize the image
    img = img.resize(target_size)
    
    # Convert image to array and rescale
    img_array = np.array(img) / 255.0
    
    # Expand dimensions to create a batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict the image
def predict_image(img):
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    return predictions

# Streamlit app
st.title("Custom Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    with st.spinner('Classifying...'):
        predictions = predict_image(img)
    
    # Display results
    st.subheader("Predictions:")
    

    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented','VeryMildDemented']  
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")
