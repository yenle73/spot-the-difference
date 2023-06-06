import cv2 as cv
import numpy as np
import os
from  PIL import Image, ImageEnhance
import streamlit as st
import imutils

st.markdown("<h2 style='text-align: center; color: #10316B;'>Pixel-wise Comparison</h2>", unsafe_allow_html=True)

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
        st.image(image1)

    with col2:
        st.image(image2)

    img_array1 = np.array(image1)
    img_array2 = np.array(image2)

    # Grayscale
    gray1 = cv.cvtColor(img_array1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img_array2, cv.COLOR_BGR2GRAY)
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Gray Scale</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Gray 1</p>',unsafe_allow_html=True)
        st.image(gray1)
    with col2:
        st.markdown('<p style="text-align: center;">Gray 2</p>',unsafe_allow_html=True)
        st.image(gray2)

    # Find the difference between the two images using absdiff
    diff = cv.absdiff(gray1, gray2)
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Absolute differences</h3>", unsafe_allow_html=True)
    st.image(diff)

    # Apply threshold
    thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Apply threshold</h3>", unsafe_allow_html=True)
    st.image(thresh)

    # Dilation
    kernel = np.ones((5,5), np.uint8)
    dilate = cv.dilate(thresh, kernel, iterations=2)
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Dilation</h3>", unsafe_allow_html=True)
    st.image(dilate)

    # Find contours
    contours = cv.findContours(dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Loop over each contour
    for contour in contours:
        if cv.contourArea(contour) > 100:
            # Calculate bounding box
            x, y, w, h = cv.boundingRect(contour)
            # Draw rectangle - bounding box
            cv.rectangle(img_array1, (x,y), (x+w, y+h), (0,0,255), 3)
            cv.rectangle(img_array2, (x,y), (x+w, y+h), (0,0,255), 3)

    # Display the images with bounding boxes
    st.markdown("<h2 style='text-align: center; color: #10316B;'>Final results</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.image(img_array1)
    with col2:
        st.image(img_array2)









