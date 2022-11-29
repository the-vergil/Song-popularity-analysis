from pickle import EMPTY_LIST
import numpy as np
import pandas as pd
import streamlit as st
from recommend import recommend_songs


@st.cache
def get_data():
    data = pd.read_csv("dataset\data.csv")
    return data


def homepage():
    st.title("Song Recommendation Engine")

    st.write("##### Write a song name to get the recommendation and choose the number of recommendations")

    data = get_data()

    song = st.text_input("Song Name")
    song = song.lower()
    st.write("Song name : ", song)

    n_rec = st.slider("Number of Recommendations", 1, 10, 1)

    rec = st.button("Recommend")

    if rec:
        if recommend_songs(song, data, n_rec) == "Song does not found in Spotify or in database":
            st.write(f'''"{song}" does not found in Spotify''')
        else:
            st.write(pd.DataFrame(recommend_songs(song, data, n_rec)))
