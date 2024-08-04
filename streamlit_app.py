import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load your custom model
model = load_model("Alzheimer's VGG16 model.keras")

labels = ['MildDemented', 'ModerateDemented', 'NonDemented','VeryMildDemented'] 

# Streamlit app title

st.title('Image Classification with Keras')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image has 3 channels (RGB)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image to the size your model expects
    image_array = np.array(image) # Rescale the image
    image_array = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image_array)
    predicted_label = labels[np.argmax(predictions,axis=1)]

    # Display the prediction
    st.write(f'Prediction: {predicted_label}')
