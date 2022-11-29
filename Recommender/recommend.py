#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


# # Read Data

# In[2]:


data = pd.read_csv("E:\Song Recommender\dataset\data.csv")
genre_data = pd.read_csv("E:\Song Recommender\dataset\data_by_genres.csv")
year_data = pd.read_csv("E:\Song Recommender\dataset\data_by_year.csv")


# In[3]:


# print(f"data : {data.shape}")
# print(f"genre_data : {genre_data.shape}")
# print(f"year_data : {year_data.shape}")


# In[4]:


# data.head(3)


# In[5]:


# data.info()


# In[6]:


# genre_data.head()


# In[7]:


# genre_data.info()


# In[8]:


# year_data.head()


# In[9]:


# year_data.info()


# # EDA

# ## Song dataset

# ### Finding the pearson correlation coefficient between features and popularity of song

# In[10]:


## data
data_pc = data.drop(["artists", "id", "name", "release_date"], axis=1)
corr_features_popularity = {}
for col in data_pc.columns :
    corr_features_popularity[col] = data_pc[col].corr(data_pc["popularity"])


# In[11]:


corr_features_popularity_Series = pd.Series(corr_features_popularity)


# In[12]:


corr_features_popularity_df = pd.DataFrame(corr_features_popularity_Series, columns=["Pearson Coefficient"])


# In[13]:


# px.bar(corr_features_popularity_df,x="Pearson Coefficient", orientation="h")


# - The year feature is highly correlated with the popularity of songs
# - The acousticness feature is highly negatively correlated with the popularity of songs

# ### Heatmap to find the correlation between features

# In[14]:


# data_pc.corr()
plt.figure(figsize=(18,12))
sns.heatmap(data_pc.corr(), annot=True)
# plt.show()


# - From the heatmap it is clear that there a high multicolinearity between the features of songs

# ## Genre dataset

# In[15]:


data_genre = genre_data.copy()


# In[16]:


# data_genre.head(2)


# In[17]:


top_10_genres = data_genre.nlargest(5, "popularity")
fig = px.bar(top_10_genres, x="genres", y=["acousticness", "danceability", "energy", "liveness", "speechiness", "valence"], barmode="group")
# fig.show()


# - The above graph will help us to analyze songs features with respect to genres

# ### Year Dataset

# In[18]:


data_year = year_data.copy()


# In[19]:


# data_year.head(2)


# In[20]:


sound_features = ["acousticness", "danceability", "energy", "liveness", "speechiness", "valence"]
# px.line(data_year, x="year", y=sound_features)


# In[21]:


# px.line(data_year, x="year", y="popularity")


# # Cluster Genres with K-means

# In[22]:


cluster_pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=10))])
X = data_genre.select_dtypes(np.number)
cluster_pipeline.fit(X)
data_genre["cluster"] = cluster_pipeline.predict(X)


# In[23]:


tsne_pipeline = Pipeline([("scaler", StandardScaler()), ("tsne", TSNE())])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(genre_embedding, columns=["x", "y"])
projection["genres"] = data_genre["genres"]
projection["cluster"] = data_genre["cluster"]


# In[24]:


fig = px.scatter(projection, x="x", y="y", color="cluster")
# fig.show()


# # Cluster songs with K-means

# In[25]:


data1 = data.copy()


# In[26]:


song_cluster_pipeline = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=10))])
X = data1.select_dtypes(np.number)
song_cluster_pipeline.fit(X)
data1["cluster"] = song_cluster_pipeline.predict(X)


# In[27]:


song_cluster_pca_pipeline = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(n_components=2))])
song_embedding = song_cluster_pca_pipeline.fit_transform(X)
projection_song = pd.DataFrame(song_embedding, columns=["x","y"])
projection_song["song"] = data1["name"]
projection_song["cluster"] = data1["cluster"]


# In[28]:


fig_2 = px.scatter(projection_song, x="x", y="y", color="cluster")
# fig_2.show()


# - Based on the analysis and visualizations, we can say that the songs with similar feature values tend to be closer to each other also similar genres are clustered together

# # Spotify API

# In[42]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from collections import defaultdict


# In[233]:


import yaml
with open("configuration.yaml", "r") as credentials :
    credential = yaml.safe_load(credentials)


# In[234]:


credential


# In[237]:


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=credential["spotify"]["client_id"], 
                                                           client_secret=credential["spotify"]["client_secret"]))


# In[238]:


def find_song(song_name) : 
    results = sp.search(q="track: {}".format(song_name), limit=1)
    results2 = results["tracks"]["items"][0]
    track_id = results2["id"]
    audio_features = sp.audio_features(track_id)[0]
    audio_features
    song_data = defaultdict()
    year = int(results["tracks"]["items"][0]["album"]["release_date"].split("-")[0])
    song_data["name"] = [song_name]
    song_data["year"] = [year]
    song_data["explicit"] = [int(results2["explicit"])]
    song_data["duration_ms"] = [results2["duration_ms"]]
    song_data["popularity"] = [results2["popularity"]]

    for key, value in audio_features.items() :
        song_data[key] = value

    return pd.DataFrame(song_data)


# In[239]:


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

features_cols = ["mode", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", 
                 "loudness", "speechiness", "tempo", "valence", "popularity", "key", "year", "explicit"]


# In[240]:


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[spotify_data.name == song].iloc[0]
        return song_data
    
    except IndexError:
        song_data = find_song(song)
        return song_data


# In[241]:


def get_mean_vector(song, spotify_data) :
    song_vector = []
    
    song_Data = get_song_data(song, spotify_data)
        
    song_vector.append(song_Data[features_cols].values)

    return np.mean(song_vector, axis=0)


# In[242]:


def recommend_songs(song, spotify_data, n_songs):
    
    try :
        song_center = get_mean_vector(song, spotify_data)
    except IndexError :
        print(f"Song does not found in Spotify or in database")
        return
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[features_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1,-1))
    distances = cdist(scaled_song_center, scaled_data, "cosine")
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    cols = ["name", "release_date", "artists", "popularity"]
    result = rec_songs[cols].to_dict(orient = "records")
    
    song_names = []
    for i in range(n_songs) :
        song_names.append(result[i]["name"])
    song_links = []
    for song in song_names :
        song_links.append(list(find_song(song)["uri"])[0])
    for i in range(n_songs) :
        result[i]["link"] = song_links[i]
    return result


# In[243]:


# recommend_songs("arcade", data, 5)


# In[244]:


# recommend_songs("more than you know", data, 5)

