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

# Get rectangles from diff
def get_rect(diff, thr = 0):
    diff = threshold(diff, int(thr))

    contours = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    rect = []

    for c in contours:
        area = cv.contourArea(c)
        if area > 40:
            x,y,w,h = cv.boundingRect(c)
            rect.append([int(x), int(y), int(x + w), int(y+ h)])
    rect = remove_overlap_rectangles(rect)
    return rect

def remove_overlap_rectangles(rectangles):
    rectangles.sort(key=lambda x: x[0])
    result = []
    for i in range(len(rectangles)):
        if i == 0:
            result.append(rectangles[i])
        else:
            if check_overlap_rectangle(result[-1], rectangles[i]):
                if rectangles[i][2] > result[-1][2]:
                    result[-1][2] = rectangles[i][2]
                if rectangles[i][3] > result[-1][3]:
                    result[-1][3] = rectangles[i][3]
            else:
                result.append(rectangles[i])
    return result

# Draw rectangles on mask
def draw_rect(diff, color = 'r', thr = 0):
    rect = get_rect(diff, thr)
    # Highlight differences
    mask = np.ones(diff.shape, dtype='uint8') * 255
    mask = cv.merge((mask, mask, mask))

    for r in rect:
            if color == 'b':
                cv.rectangle(mask, (r[0], r[1]),(r[2], r[3]), (255, 0, 0), 1)
            elif color == 'g':
                cv.rectangle(mask, (r[0], r[1]),(r[2], r[3]), (0, 255, 0), 1)
            else:
                cv.rectangle(mask, (r[0], r[1]),(r[2], r[3]), (0, 0, 255), 1)

            # cv.drawContours(mask, [c], 0, (0,255,0), -1)
    return mask

# Draw rectangles on img
def draw_img_rect(img, diff, color = 'r', thr = 0):
    line = int(img.shape[0]/200)
    rect = get_rect(diff, thr)
    for r in rect:
        if color == 'b':
            cv.rectangle(img, (r[0], r[1]),(r[2], r[3]), (255, 0, 0), line)
        elif color == 'g':
            cv.rectangle(img, (r[0], r[1]),(r[2], r[3]), (0, 255, 0), line)
        else:
            cv.rectangle(img, (r[0], r[1]),(r[2], r[3]), (0, 0, 255), line)
    return img

# Overlap 2 diffs on mask
def overlap(diff1, diff2, thr = 0):
    rect1 = get_rect(diff1, thr)
    rect2 = get_rect(diff2, thr)

    # Highlight differences
    mask = np.ones(diff1.shape, dtype='uint8') * 255
    mask = cv.merge((mask, mask, mask))

    for r in rect1:
        cv.rectangle(mask, (r[0], r[1]),(r[2], r[3]), (0, 0, 255), 1)
    for r in rect2:
        cv.rectangle(mask, (r[0], r[1]),(r[2], r[3]), (0, 255, 0), 1)
    return mask

# Fill rectangles on mask then calculate similar
def ssim_rect(diff1, diff2, thr = 0):
    rect1 = get_rect(diff1, thr)
    rect2 = get_rect(diff2, thr)

    # Highlight differences
    mask1 = np.zeros(diff1.shape, dtype='uint8')
    mask2 = np.zeros(diff1.shape, dtype='uint8')

    for r in rect1:
        cv.rectangle(mask1, (r[0], r[1]),(r[2], r[3]), (255, 255, 255), -1)

    for r in rect2:
        cv.rectangle(mask2, (r[0], r[1]),(r[2], r[3]), (0, 255, 0), 1)
    # cv2_imshow(mask1)
    # cv2_imshow(mask2)
    ssim_score = ssim(mask1, mask2, full=True)[0]
    return ssim_score

def check_overlap_rectangle(rect1, rect2):
    if rect1[0] >= rect2[2] or rect1[2] <= rect2[0] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3]:
        return False
    return True

def calculate_intersect_area(rect1, rect2):
    # Check if the rectangles intersect.
    if not check_overlap_rectangle(rect1, rect2):
        return 0

    # Calculate the intersect area.
    intersect_x_min = max(rect1[0], rect2[0])
    intersect_x_max = min(rect1[2], rect2[2])
    intersect_y_min = max(rect1[1], rect2[1])
    intersect_y_max = min(rect1[3], rect2[3])

    intersect_width = intersect_x_max - intersect_x_min
    intersect_height = intersect_y_max - intersect_y_min

    return intersect_width * intersect_height


st.markdown("<h2 style='text-align: center; color: #10316B;'>Structural Similarity Index Comparison</h2>", unsafe_allow_html=True)

#Add file uploader to allow users to upload photos
uploaded_file_1 = st.file_uploader("", type=['jpg','png','jpeg'], key='file1')

uploaded_file_2 = st.file_uploader("", type=['jpg','png','jpeg'], key='file2')

#Add 'before' and 'after' columns
if uploaded_file_1 is not None and uploaded_file_2 is not None:
    image1 = Image.open(uploaded_file_1)
    image2 = Image.open(uploaded_file_2)

    img1 = image1.convert('RGB')
    img2 = image2.convert('RGB')
    images1 = np.array(img1)
    images2 = np.array(img2)



    diff = draw_diff(images1, images2)


    calculate_ssim(images1, images2)

    final1 = draw_img_rect(images1, diff, 'g', 200)
    final2 = draw_img_rect(images2, diff, 'g', 200)

    st.markdown("<h3 style='text-align: center; color: #10316B;'>Originals</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.image(image1)

    with col2:
        st.image(image2)

    img_array1 = np.array(image1)
    img_array2 = np.array(image2)

    st.markdown("<h2 style='text-align: center; color: #10316B;'>Final results</h2>", unsafe_allow_html=True)
    with col1:
        st.images(img_array1)

    with col2:
        st.images(img_array2)

    img_array1 = np.array(final1)
    img_array2 = np.array(final2)
