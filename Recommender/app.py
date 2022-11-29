import streamlit as st 

from predict import homepage
from explore import explore

page = st.sidebar.selectbox("Recommend or Explore", ("Recommend", "Explore"))

if page == "Recommend" :
    homepage()
else :
    explore()