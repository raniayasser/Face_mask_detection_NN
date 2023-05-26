import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("C:/Users/lap shop/Desktop/nn/face mask model.h5")

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image /= 255.0
    return image

def main():
    st.title("New Fase mask detection")
    st.write("Upload an image of a crop leaf to predict the disease")
    file = st.file_uploader("Choose an image", type=["jpg", "JPG"])
    if file is not None:
        image = np.array(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        classes = ['With_mask',
                   'Without_mask' ]
        class_name = classes[np.argmax(prediction)]
        st.write("Prediction: ", class_name)
if __name__ == "__main__":
    main()