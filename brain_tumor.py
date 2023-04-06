# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:35:27 2023

@author: Aondona Moses Iorumbur
"""


import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import tensorflow


st.title("Brain Tumor Detection APP")

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
# Load saved model 
def loadModel():
    model = keras.models.load_model("brain-tumor-mri-classificationModel.hdf5")
    return model


# Formate uploaded image for prediction
def predictionModel(img):
    model = loadModel()
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = img.reshape(1,150,150,3)
    prediction = model.predict(img)
    prediction = np.argmax(prediction,axis=1)[0]

    return prediction


def main():
    st.markdown("<h6 style='text-align: center'>Please upload only Brain Tumor images</h6>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center'>Otherwise it's gives wrong output/prediction</h6>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a Brain MRI File", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        new_image = image.resize((200, 200))
        st.image(new_image, use_column_width=False)
        
        if st.button("RUN TEST"):
            st.write("Please wait......")
            label = predictionModel(image)
            if label == 0:
                st.write(" **TEST RESULT: Glioma Tumor Detected**")
            elif label == 1:
                st.write("**TEST RESULT: No Tumor Detected**")
            elif label == 2:
                st.write("**TEST RESULT: Meningioma Tumor Detected**")
            else:
                st.write("**TEST RESULT: Pituitary Tumor Detected**")
    else:
        st.write("")
        
        
tab_title = ["**Welcome Page**",
             "**Prediction Page**",
             "**Info Page**"]

tabs = st.tabs(tab_title)

with tabs[0]:
    image = Image.open("brainImage.jpg")
    new_image = image.resize((400, 400))
    st.image(new_image)
    st.markdown("<h5>Student: Aondona Moses Iorumbur</h5>", unsafe_allow_html=True)
    st.markdown("<h5>Matric No.: 2016/1/59336PP</h5>", unsafe_allow_html=True)
    st.markdown("<h5>Supervisor/Mentor: Dr. M.O. Dada</h5>", unsafe_allow_html=True)

with tabs[1]:
    if __name__ == "__main__":
        main()
        
with tabs[2]:
    glioma_tumor = Image.open("Glioma Tumor.jpg")
    no_tumor = Image.open("No Tumor.jpg")
    mening_tumor = Image.open("Meningioma Tumor.jpg")
    pituit_tumor = Image.open("Pituitary Tumor.jpg")
    
    glioma_tumor = glioma_tumor.resize((150, 150))
    no_tumor = no_tumor.resize((150, 150))
    mening_tumor = mening_tumor.resize((150, 150))
    pituit_tumor = pituit_tumor.resize((150, 150))
    
    
    st.markdown("<h4 style='text-align: center'>BRAIN TUMOR</h6>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Brain tumor diagnosis is often performed by examining an MRI due to its high accuracy, specificity, and sensitivity. This imaging modality uses non-ionizing radiation to image soft tissues. Specifically, it detects the concentration or density of protons. The whiter areas in an MRI represent regions of high proton density whereas the darker areas represent regions of low proton density (i.e. air, water). Since tumors have high proton density, they show up as a lighter colour.</p>", unsafe_allow_html=True)
    st.image([glioma_tumor, no_tumor, mening_tumor, pituit_tumor], caption=["Glioma Tumor","No Tumor", "Meningioma Tumor", "Pituitary Tumor"])
    st.markdown("<h4 style='text-align: center'>BACKGROUND</h6>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Users can upload an image file of a Brain MRI. Once the image is uploaded, users can submit the image where the app will detect one of the case No Brain Tumor, Glioma Tumor,  Meningioma Tumor or Pituitary Tumor. The dataset was taken from: Kaggle. This Deep Learning model uses Transfer Learning Technique with an accuracy of 99.89%.</p>", unsafe_allow_html=True)