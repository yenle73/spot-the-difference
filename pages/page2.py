import cv2
import numpy as np
import os
from  PIL import Image, ImageEnhance
import streamlit as st


def Sobel_compare(image1, image2):

  gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  st.markdown("<h3 style='text-align: center; color: #10316B;'>Gray Scale</h3>", unsafe_allow_html=True)
  col1, col2 = st.columns( [0.5, 0.5])
  with col1:
    st.markdown('<p style="text-align: center;">Gray 1</p>',unsafe_allow_html=True)
    st.image(gray1)
  with col2:
    st.markdown('<p style="text-align: center;">Gray 2</p>',unsafe_allow_html=True)
    st.image(gray2)

  # Apply the Sobel operator to both images
  sobel_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
  sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
  sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)

  # Compute the gradient magnitudes
  gradient_magnitude1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
  gradient_magnitude2 = np.sqrt(sobel_x2**2 + sobel_y2**2)

  # Normalize the gradient magnitudes to the range [0, 255]
  gradient_magnitude1 = cv2.normalize(gradient_magnitude1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  gradient_magnitude2 = cv2.normalize(gradient_magnitude2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

  # Apply a threshold to obtain binary edges
  threshold = 50
  edges1 = cv2.threshold(gradient_magnitude1, threshold, 255, cv2.THRESH_BINARY)[1]
  edges2 = cv2.threshold(gradient_magnitude2, threshold, 255, cv2.THRESH_BINARY)[1]

  st.markdown("<h3 style='text-align: center; color: #10316B;'>Detecting edges</h3>", unsafe_allow_html=True)
  col1, col2 = st.columns( [0.5, 0.5])
  with col1:
    st.markdown('<p style="text-align: center;">Edges 1</p>',unsafe_allow_html=True)
    st.image(edges1)
  with col2:
    st.markdown('<p style="text-align: center;">Edges 2</p>',unsafe_allow_html=True)
    st.image(edges2)

  # Compute the absolute difference between the binary edges
  diff = cv2.absdiff(edges1, edges2)
  st.markdown("<h3 style='text-align: center; color: #10316B;'>Absolute differences between the edges</h3>", unsafe_allow_html=True)
  st.image(diff)

  threshold = 50
  diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

  # Find contours of the differences
  contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter out smaller bounding boxes
  min_area = 100  # Minimum bounding box area
  filtered_contours = []
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h >= min_area:
      filtered_contours.append(contour)

  # Draw bounding boxes around the contours on both images
  for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 3)

  # Display the images with bounding boxes
  st.markdown("<h2 style='text-align: center; color: #10316B;'>Final results</h2>", unsafe_allow_html=True)
  col1, col2 = st.columns( [0.5, 0.5])
  with col1:
    st.markdown('<p style="text-align: center;">Image 1</p>',unsafe_allow_html=True)
    st.image(image1)
  with col2:
    st.markdown('<p style="text-align: center;">Image 2</p>',unsafe_allow_html=True)
    st.image(image2)

st.markdown("<h2 style='text-align: center; color: #10316B;'>Edge Detection Comparison</h2>", unsafe_allow_html=True)

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

    Sobel_compare(img_array1, img_array2)

