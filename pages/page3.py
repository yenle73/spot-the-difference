import os
import cv2 as cv
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from  PIL import Image, ImageEnhance

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# SSIM value
def calculate_ssim(image1, image2):
  image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
  image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

  return ssim(image1_gray, image2_gray, full=True)[0]

# Threshold
def threshold(diff, value = 0, choice = ''):
    if choice == 'otsu':
        thresh = cv.threshold(diff, int(value), 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    else:
        thresh = cv.threshold(diff, int(value), 255, cv.THRESH_BINARY_INV)[1]
    return thresh

# Draw diff img
def draw_diff(img1, img2):
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    score, diff = ssim(img1_gray, img2_gray, full=True)

    return (diff * 255).astype("uint8")


st.markdown("<h2 style='text-align: center; color: #10316B;'>SSIM Comparison</h2>", unsafe_allow_html=True)

#Add file uploader to allow users to upload photos
uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'], key='file1')

uploaded_file_2 = st.file_uploader("", type=['jpg','png','jpeg'], key='file2')

#Add 'before' and 'after' columns
if uploaded_file_1 is not None and uploaded_file_2 is not None:
    image1 = Image.open(uploaded_file_1)
    image2 = Image.open(uploaded_file_2)
    
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Originals</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Image 1</p>',unsafe_allow_html=True)
        st.image(image1)

    with col2:
        st.markdown('<p style="text-align: center;">Image 2</p>',unsafe_allow_html=True)
        st.image(image2)

img_array1 = np.array(image1)
img_array2 = np.array(image2)

st.markdown("<h2 style='text-align: center; color: #10316B;'>Final results</h2>", unsafe_allow_html=True)
diff = draw_diff(img_array1, img_array1)
st.image(diff)