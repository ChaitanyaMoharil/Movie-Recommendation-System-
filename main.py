import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Corrected background color styling
st.markdown("""
    <style>
    body {
        background-color: teal;
    }
    .stApp {
        background-color: teal;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app with HTML styling
st.markdown('<h1 style="color: #FFD700;">RATEU: MOVIE MANAGEMENT SYSTEM</h1>', unsafe_allow_html=True)

# Load movie data
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # Helper functions to process the data
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_dataframe = movies[['movie_id', 'title', 'tags', 'genres']]
    new_dataframe['tags'] = new_dataframe['tags'].apply(lambda x: " ".join(x).lower())

    return new_dataframe

# Vectorize data and calculate similarity
@st.cache_data
def vectorize_data(new_dataframe):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_dataframe['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

# Recommendation function
def recommend(movie, new_dataframe, similarity):
    if movie not in new_dataframe['title'].values:
        return None
    movie_index = new_dataframe[new_dataframe['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_dataframe.iloc[i[0]].title)

    return recommended_movies

# Random movie suggestion if the movie is not found
def suggest_random_movies(new_dataframe):
    return random.sample(list(new_dataframe['title']), 5)

# Watchlist storage
watchlist = {}

# Watchlist management functions
def add_to_watchlist(user_id, movie_id):
    if user_id not in watchlist:
        watchlist[user_id] = []
    if movie_id not in watchlist[user_id]:
        watchlist[user_id].append(movie_id)
        return True  # Successfully added
    return False  # Already in watchlist

def remove_from_watchlist(user_id, movie_id):
    if user_id in watchlist and movie_id in watchlist[user_id]:
        watchlist[user_id].remove(movie_id)
        return True  # Successfully removed
    return False  # Not in watchlist

def get_watchlist(user_id):
    return watchlist.get(user_id, [])

# Load data
new_dataframe = load_data()
similarity = vectorize_data(new_dataframe)

# Input for user ID
user_id = st.text_input("Enter your User ID:", "")

if user_id:
    # Filter by genre
    all_genres = set([genre for sublist in new_dataframe['genres'] for genre in sublist])
    selected_genre = st.selectbox('Select a genre to filter movies:', sorted(all_genres))

    # Filter movies by selected genre
    filtered_movies = new_dataframe[new_dataframe['genres'].apply(lambda x: selected_genre in x)]['title'].values

    # Select a movie for recommendation
    selected_movie = st.selectbox('Select a movie you like:', filtered_movies)

    # Recommend movies on button click
    if st.button('Recommend'):
        recommendations = recommend(selected_movie, new_dataframe, similarity)
        if recommendations:
            st.write('Here are the recommended movies:')
            for movie in recommendations:
                st.write(movie)
        else:
            st.write("Sorry....Here are 5 movies we think you will love:")
            random_movies = suggest_random_movies(new_dataframe)
            for movie in random_movies:
                st.write(movie)

    # Watchlist functionality
    st.header("Manage Your Watchlist")
    movie_for_watchlist = st.selectbox("Select a movie to add/remove from your watchlist:",
                                       new_dataframe['title'].values)

    # Get the movie_id of the selected movie
    movie_id = new_dataframe[new_dataframe['title'] == movie_for_watchlist]['movie_id'].values[0]

    if st.button("Add to Watchlist"):
        if add_to_watchlist(user_id, movie_id):
            st.success(f"'{movie_for_watchlist}' added to your watchlist.")
        else:
            st.warning(f"'{movie_for_watchlist}' is already in your watchlist.")

    if st.button("Remove from Watchlist"):
        if remove_from_watchlist(user_id, movie_id):
            st.success(f"'{movie_for_watchlist}' removed from your watchlist.")
        else:
            st.warning(f"'{movie_for_watchlist}' is not in your watchlist.")

    # Display watchlist
    if st.button("Show Watchlist"):
        user_watchlist = get_watchlist(user_id)
        if user_watchlist:
            st.write("Your Watchlist:")
            watchlist_movies = new_dataframe[new_dataframe['movie_id'].isin(user_watchlist)]['title'].values
            for movie in watchlist_movies:
                st.write(movie)
        else:
            st.write("Your watchlist is empty.")
