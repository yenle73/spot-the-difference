from st_pages import Page, Section, add_page_title, show_pages
import streamlit as st

st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h1>", unsafe_allow_html=True)

show_pages([
    Page("pages/page1.py", "Pixel-wise Comparison", ":notebook:"),
    Page("pages/page2.py", "Edge Detection Comparison", ":blue_book:"),
    Page("pages/page3.py", "Structural Similarity Index Comparison", ":bar_chart:"),
])