import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Creating title for Streamlit app
st.title("AI-Powered Fabric Defect Detection")

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)

# Function to generate Grad-CAM heatmap
def generate_gradcam(model, image, layer_name):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Uploading file for processing
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Preprocess the image for the model
    img = cv2.resize(image, (224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam(model, img, 'Conv_1')
    
    # Superimpose heatmap on original image
    heatmap = cv2.resize(heatmap[0], (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Display results
    st.image(image, caption="Original Image", channels="BGR")
    st.image(superimposed_img, caption="Defect Heatmap", channels="BGR")
    
    if prediction[0][0] > 0.5:
        st.write("Defect Detected")
        st.write(f"Confidence: {prediction[0][0]:.2f}")
    else:
        st.write("No Defect Detected")
        st.write(f"Confidence: {1 - prediction[0][0]:.2f}")
    
    # Generate and display explanation
    st.subheader("Explanation")
    st.write("The heatmap above shows the areas of the fabric that the AI model focused on when making its decision. Red areas indicate regions that strongly influenced the model's prediction of a defect, while blue areas had less influence.")
    
    # Contextual analysis (placeholder)
    st.subheader("Contextual Analysis")
    st.write("Based on the detected pattern, this defect might be caused by:")
    st.write("1. Irregular tension in the weaving process")
    st.write("2. Contamination in the raw materials")
    st.write("3. Malfunction in the spinning equipment")
    
    # Recommendations (placeholder)
    st.subheader("Recommendations")
    st.write("1. Check and adjust the tension settings on the weaving machines")
    st.write("2. Inspect the quality of incoming raw materials")
    st.write("3. Perform maintenance on the spinning equipment")

# Remove OpenCV window handling as it's not needed in Streamlit
# cv2.waitKey(0)
# cv2.destroyAllWindows()