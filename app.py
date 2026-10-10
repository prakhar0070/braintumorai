import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = 'models/mobilenetv2_final.h5'
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']   # same order as class_names in training

st.set_page_config(page_title='Brain Tumor MRI Classifier', page_icon='🧠', layout='centered')


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


model = load_model()

st.title('🧠 Brain Tumor MRI Classification')
st.write('Upload a brain MRI image and the model will predict the tumor type.')
st.caption('Educational project only. This is not a medical diagnosis.')

file = st.file_uploader('Upload an MRI image', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)

    # raw 0-255 pixels: scaling is already inside the model
    arr = np.array(img.resize((224, 224)), dtype='float32')
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx].replace('_', ' ').title()
    st.subheader(f'Prediction: {label}')
    st.metric('Confidence', f'{probs[idx]:.1%}')

    if probs[idx] < 0.6:
        st.warning('Low confidence. The image may be unclear or unlike the training data.')

    st.write('Probability for each class')
    st.bar_chart({n.replace('_', ' ').title(): float(p) for n, p in zip(CLASS_NAMES, probs)})
