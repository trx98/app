import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

# Load the saved model
model = tf.keras.models.load_model('best_model.h5')

# Define class labels (replace these with your actual class labels)
class_labels = ['biological', 'cardboard', 'paper', 'plastic', 'e-waste']

# Function to preprocess image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Adjust target_size based on your model
    img = img_to_array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict image category
def predict_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]

# Streamlit app
st.title("Garbage Classification App")
st.write("Upload an image to classify")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Predict the image class
    st.write("Classifying...")
    result = predict_image(uploaded_file)
    
    # Show prediction result
    st.write(f"Predicted label: {result}")
