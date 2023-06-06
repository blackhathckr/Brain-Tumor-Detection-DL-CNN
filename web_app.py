import streamlit as st
import joblib as jb 
import numpy as np
import tensorflow as tf
import keras

def detection(img):
    model=jb.load('brain_tumor_detection_model.h5')
    test_image = tf.keras.utils.load_img(img,target_size=(224,224))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    print(result)

    if result[0][0]==1:
        st.title("Brain Tumor Detected")
    else:
        st.title("No Brain Tumor Detected")

def brain_tumor_detection_webapp():
    st.title("Brain Tumor Detection ")
    img=st.file_uploader(label="Upload Image",accept_multiple_files=False)
    if st.button("Detect"):
        detection(img)

brain_tumor_detection_webapp()
    