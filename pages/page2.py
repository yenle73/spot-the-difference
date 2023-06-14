import cv2
import numpy as np
import os
from  PIL import Image, ImageEnhance
import streamlit as st

def feature_based_matching(img1, img2):
  # Convert to grayscale
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  hh, ww = gray1.shape

  st.markdown("<h3 style='text-align: center; color: #10316B;'>Gray Scale</h3>", unsafe_allow_html=True)
  col1, col2 = st.columns( [0.5, 0.5])
  with col1:
    st.image(gray1)
  with col2:
    st.image(gray2)

  # Tạo đối tượng SIFT và tính toán keypoint và descriptor cho 2 ảnh
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(gray1, None)
  kp2, des2 = sift.detectAndCompute(gray2, None)

  # Tìm các match giữa 2 ảnh bằng RANSAC
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)

  good_matches = []
  for m, n in matches:
    if m.distance < 0.7 * n.distance: # Chọn các match tốt
      good_matches.append(m)

    # Draw matches
  img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  st.markdown("<h3 style='text-align: center; color: #10316B;'>Matches using Flann Based Matcher</h3>", unsafe_allow_html=True)
  st.image(img_matches)

  MIN_MATCH_COUNT = 10
  if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # Tìm ma trận homography bằng RANSAC
    matchesMask = mask.ravel().tolist()
    
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Homography matrix between 2 images</h3>", unsafe_allow_html=True)
    st.table(M)

    # Tính toán tọa độ của các điểm trên ảnh thứ nhất bằng cách sử dụng ma trận homography
    h, w, d = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # Tính toán difference giữa ảnh thứ nhất và ảnh thứ hai đã được chuyển đổi bằng ma trận homography
    diff = cv2.absdiff(cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])), img2)
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Differences between images</h3>", unsafe_allow_html=True)
    st.image(diff)

    # Áp dụng threshold và tìm contour để tách các vùng khác biệt
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tính toán tọa độ của bounding box cho từng contour và vẽ chúng lên ảnh thứ hai
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h > hh * ww * 0.0001:
           cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
           cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Hiển thị kết quả
    st.markdown("<h3 style='text-align: center; color: #10316B;'>Final results</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.image(img1)

    with col2:
        st.image(img2)
  else:
    st.warning("Không đủ điểm match để tính toán ma trận homography!")

st.markdown("<h2 style='text-align: center; color: #10316B;'>Feature Based Comparison</h2>", unsafe_allow_html=True)

#Add file uploader to allow users to upload photos
uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'], key='file1')

uploaded_file_2 = st.file_uploader("", type=['jpg','png','jpeg'], key='file2')

#Add originals
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

    feature_based_matching(img_array1, img_array2)

