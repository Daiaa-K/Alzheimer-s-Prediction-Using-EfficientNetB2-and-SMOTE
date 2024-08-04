import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load your custom model
model = load_model("alzheimer's_model.keras")

labels = ['MildDemented', 'ModerateDemented', 'NonDemented','VeryMildDemented'] 

# Streamlit app title

st.title('Image Classification with Keras')
st.write('Upload an image to classify it using a pre-trained model.')

# Sidebar for file upload
st.sidebar.header('Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image has 3 channels (RGB)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a spinner while processing
    with st.spinner('Processing...'):
        # Preprocess the image
        image = image.resize((224, 224))  # Resize the image to the size your model expects
        image_array = np.array(image) 
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        predicted_label = labels[np.argmax(predictions)]

    # Display the prediction
    st.success(f'Prediction: {predicted_label}')
else:
    st.info('Please upload an image to classify.')

# Footer
st.sidebar.markdown("""
---
Created by [Diaa_K](https://github.com/your-github-profile)
""")
