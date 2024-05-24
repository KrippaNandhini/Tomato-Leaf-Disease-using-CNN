
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('my_cnn_model.h5')
# Define the same architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224,224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the weights
model.load_weights('my_cnn_weights.weights.h5')

# Evaluate the model with loaded weights (example)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Model accuracy with loaded weights: {accuracy}')

#model.load_weights('my_cnn_weights.weights.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224)) 
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to perform classification
def classify_image(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Streamlit app
st.title('Tomato Leaf Disease Classifier')
st.write('This app classifies tomato leaf diseases. The expected outcome is one of the following:')
st.markdown("- Tomato Early Blight")
st.markdown("- Tomato Late Blight")
st.markdown("- Tomato Leaf Mold")
st.markdown("- Tomato Septoria Leaf spot")
st.markdown("- Tomato Spider Mites")
st.markdown("- Tomato Target Spot")
st.markdown("- Tomato Yellow Leaf Curl Virus")
st.markdown("- Tomato Mosaic Virus")
st.markdown("- Tomato Bacterial Spot")
st.markdown("- Tomato Healthy")
st.markdown('''<style>[data-testid="stMarkdownContainer"] ul{list-style-position: inside;}</style>''', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image with single leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if st.button('Classify'):
        with st.spinner('Classifying...'):
            prediction = classify_image(image)
            st.success('Classification done!')

        # Displaying the predicted class or classes (depending on your output)
        st.subheader('Prediction:')
        # Define your class dictionary
        class_dict = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Late_blight': 2, 'Tomato_Leaf_Mold': 3, 'Tomato_Septoria_leaf_spot': 4, 'Tomato_Spider_mites': 5, 'Tomato_Target_Spot': 6, 'Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato_mosaic_virus': 8, 'Tomato_healthy': 9}

# Get the class name from the dictionary
        # Find the index of the maximum value in the prediction array
        prediction_index = np.argmax(prediction)
        class_name = list(class_dict.keys())[list(class_dict.values()).index(prediction_index)]
        st.write(f"Predicted Class: {class_name}")
