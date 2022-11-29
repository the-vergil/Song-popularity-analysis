import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.manifold import TSNE


def explore():
    st.title("Exploring dataset")
    st.write("Developer : Chet")
