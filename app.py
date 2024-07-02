import streamlit as st
import requests
import pickle
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from dotenv import load_dotenv
import joblib
from prediction_model.config.config import COUNT_VECTORIZER_PATH, SIMILARITY_MATRIX_PATH, MOVIE_LIST_PATH

# Load variables from .env file
tmdb_api_key = "f772f057340a7021d5fc62995e6a3f97"

# Load data from the pickle file
# file_path = 'src\prediction_model\trained_models\movie_list.pkl'
# with open(file_path, 'rb') as file:
#     data = joblib.load(file)

data=joblib.load(r'src\prediction_model\trained_models\list.pkl')


# Load the sentiment analysis model
with open(r'src\prediction_model\trained_models\sentiment_model.pkl', 'rb') as model_file:
    tfidf_vectorizer, naive_bayes = joblib.load(model_file)

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb_api_key}&language=en-US"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_movie_names = []
    recommended_movie_posters = []
    recommended_movie_ids = []

    for i in sim_scores[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        poster_url = fetch_poster(movie_id)
        if poster_url:
            recommended_movie_posters.append(poster_url)
            recommended_movie_names.append(movies.iloc[i[0]].title)
            recommended_movie_ids.append(movie_id)

    return recommended_movie_names, recommended_movie_posters, recommended_movie_ids

# Function to get movie reviews from TMDb API
def get_movie_reviews(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews'
    params = {'api_key': 'f772f057340a7021d5fc62995e6a3f97'}

    response = requests.get(url, params=params)
    data = response.json()

    return data.get('results', [])

# Function to predict sentiment using the loaded model
def predict_sentiment(review):
    transformed_review = tfidf_vectorizer.transform([review])
    prediction = naive_bayes.predict(transformed_review)
    return prediction[0]

# Assuming the data is a list of dictionaries
df = pd.DataFrame(data)

# Create TF-IDF vectorizer and calculate similarity matrix
# tfidf_vectorizer_movies = TfidfVectorizer(stop_words='english')
# tfidf_matrix_movies = tfidf_vectorizer_movies.fit_transform(df['overview'].astype(str))
# similarity_movies = linear_kernel(tfidf_matrix_movies, tfidf_matrix_movies)
# similarity_movies=pickle.load('model\similarity.pkl')
# Load data from the pickle file
file_path = r'src\prediction_model\trained_models\similarity_matrix1.pkl'
with open(file_path, 'rb') as file:
    similarity_movies = joblib.load(file)

# Title dropdown instead of sidebar slider
selected_movie_title = st.selectbox("Select Movie Title", ['None'] + df['title'].tolist())

# Fetch data from TMDb API
if selected_movie_title != 'None':
    selected_movie_id = df[df['title'] == selected_movie_title]['movie_id'].values[0]
    poster_path = fetch_poster(selected_movie_id)

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Display TMDb poster image
    with col1:
        if poster_path:
            st.subheader("   ")
            img = Image.open(requests.get(poster_path, stream=True).raw)
            st.image(img, width=300)
        else:
            st.warning("Poster not available for this movie.")

    # Display information for the selected movie
    with col2:
        st.title(f" {selected_movie_title}")
        st.subheader("Overview")
        st.write(df[df['movie_id'] == selected_movie_id]['overview'].values[0])


        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Genre")
            genres_list = df[df['movie_id'] == selected_movie_id]['genres'].values[0]
            st.write(', '.join(genres_list))

        with col6:
            st.subheader("Keywords")
            keywords_list = df[df['movie_id'] == selected_movie_id]['keywords'].values[0]
            keywords_list=keywords_list[:3]
            st.write(', '.join(keywords_list))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Cast")
            cast_list = df[df['movie_id'] == selected_movie_id]['cast'].values[0]
            st.write(' '.join(cast_list))
        with col4:
            st.subheader("Director")
            director_list = df[df['movie_id'] == selected_movie_id]['crew'].values[0]  # Assuming the director is in the 'crew' field
            st.write(' '.join(director_list))
    # Sentiment Analysis Section
    

    # Fetch reviews for the given movie ID
    reviews = get_movie_reviews(selected_movie_id)
    reviews = reviews[:2]
    if reviews:
        st.subheader("Reviews and Sentiment Analysis:")
        for i, review in enumerate(reviews):
            # Display review content
            full_content = review['content']
            author = review['author']
            sentiment_prediction = predict_sentiment(review['content'])
            result_text = "Positive" if sentiment_prediction == 1 else "Negative"
            color = "#B3FFAE" if sentiment_prediction == 1 else "#FF6464"
            text_color = "black" if sentiment_prediction == 1 else "white"

            # Display review content with "Read More" expander
            with st.expander(f"Review  by {author}"):
                

                st.markdown(
                    f"<div style='background-color:{color}; padding: 10px; border-radius: 5px; color:{text_color}'>"
                    f"<p style='font-size: 18px;'>Full Review: {full_content}</p>"
                    f"<p style='font-size: 18px;'>Sentiment: {result_text}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
    else:
        st.warning("No reviews found for the selected movie.")


    # Movie Recommender System
    st.header('Recommendations')

    # if st.button('Show Recommendations'):
    recommended_movie_names, recommended_movie_posters, recommended_movie_ids = recommend(selected_movie_title, df, similarity_movies)
    col5, col6, col7, col8, col9 = st.columns(5)  # Use st.columns for stable layout

    for i, (name, poster, movie_id) in enumerate(zip(recommended_movie_names, recommended_movie_posters, recommended_movie_ids)):
        with locals()[f"col{i + 5}"]:
            st.text(name)
            # Create a clickable link to the TMDb page for the movie using HTML
            st.image(poster)
            # st.markdown(f"[See](https://www.themoviedb.org/movie/{movie_id})")

    
else:
    st.warning("Please select a movie to show recommendations and sentiment analysis.")