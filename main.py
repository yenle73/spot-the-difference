from st_pages import Page, Section, add_page_title, show_pages
import streamlit as st

st.markdown("<h1 style='text-align: center; color: #10316B;'>Movie Recommendation System</h1>", unsafe_allow_html=True)

show_pages([
    Page("main.py", "Home Page", ":house:"),
    Page("pages/page1.py", "Content Based Recommender", ":notebook:"),
    Page("pages/page2.py", "Mood and Genre Based Recommender", ":blue_book:"),
    Page("pages/page3.py", "Collaborative Recommender", ":bar_chart:"),
])