import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('cnn_model.keras')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

def main():
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
    uploaded_file = st.file_uploader("Choose an image with a single leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = classify_image(image)
                st.success('Classification done!')

            # Displaying the predicted class or classes (depending on your output)
            st.subheader('Prediction:')
            class_dict = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Late_blight': 2, 'Tomato_Leaf_Mold': 3, 'Tomato_Septoria_leaf_spot': 4, 'Tomato_Spider_mites': 5, 'Tomato_Target_Spot': 6, 'Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato_mosaic_virus': 8, 'Tomato_healthy': 9}
            prediction_index = np.argmax(prediction)
            class_name = list(class_dict.keys())[list(class_dict.values()).index(prediction_index)]
            st.write(f"Predicted Class: {class_name}")

if __name__ == "__main__":
    main()
