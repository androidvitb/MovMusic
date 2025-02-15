import streamlit as st
import numpy as np
import pandas as pd
import json
from difflib import get_close_matches
from typing import List, Dict, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import joblib
from fuzzywuzzy import process
import gensim
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from typing import Union, List, Dict
from difflib import get_close_matches
import spacy

# # Set up credentials
SPOTIFY_CLIENT_ID = "acc33b2088c041628c9f94386ca2a6ed"  # Such as  "ee3e7f4789a56c4e40a2a3fc8bc99d5e"
SPOTIFY_CLIENT_SECRET =   "03ab90d72355453098a5db05bf4cff68" # Such as"ee3e7f4789a56c4e40a2a3fc8bc99d5e"
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



st.set_page_config(
    page_title="Movie-to-Music Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Original CSS remains the same */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp { background: #f0f2f6; }
        
        .main-header {
            font-family: 'Inter', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
        }
        
        .section-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .movie-card {
            background: linear-gradient(to right, #EFF6FF, #FFFFFF);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #3B82F6;
        }
        
        .track-card {
            background: white;
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border: 1px solid #E5E7EB;
            transition: all 0.2s ease;
        }
        
        .track-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        @media (prefers-color-scheme: dark) {
            .stApp { background: #1a1a1a; }
            .section-container { background: #2d2d2d; }
            .movie-card { background: linear-gradient(to right, #2d2d2d, #363636); }
            .track-card { 
                background: #2d2d2d;
                border-color: #404040;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .animate-fade-in { animation: fadeIn 0.5s ease-in; }
        
        .recommendation-score {
            font-size: 1.2rem;
            font-weight: bold;
            color: #3B82F6;
            text-align: center;
            padding: 0.5rem;
            border-radius: 8px;
            background: rgba(59, 130, 246, 0.1);
        }
        
        .statistics-card {
            background: #F8FAFC;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #E5E7EB;
            text-align: center;
 .insights-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #E5E7EB;
        }
        
        .mood-tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            margin: 0.25rem;
            background: #EFF6FF;
            color: #2563EB;
            border: 1px solid #BFDBFE;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-card {
            background: #F8FAFC;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1E3A8A;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: #6B7280;
        }
    </style>
""", unsafe_allow_html=True)



nlp = spacy.load("en_core_web_sm")
def word_tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

@st.cache_data
def load_movies(path: str):
    return pd.read_csv(path)

@st.cache_data
def load_music(path: str):
    return pd.read_csv(path)


def render_movie_recommendation(rec: Dict, index: int, track_name: str, saved_movies: List):
    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
    
    st.markdown(f'### {rec["title"]}')
    st.write(f'⭐ Rating: {rec["vote_average"]:.1f}')
    st.write(f'📈 Popularity: {int(rec["popularity"])}')
    st.write(rec["overview"])

    col1, col2 = st.columns([3, 1])
    with col1:
        rating = st.slider(
            "Rate this movie",
            1, 5, 3,
            key=f"movie_rating_{index}_{track_name}",
            help="1 = Poor, 5 = Excellent"
        )
    
    with col2:
        if st.button("💾 Save", key=f"save_movie_{index}_{track_name}"):
            movie_data = {
                'track': track_name,
                'movie': rec['title'],
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            }
            saved_movies.append(movie_data)
            save_tracks_to_file("saved_movies.txt", saved_movies)
            st.success("✅ Movie saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)

class GenreRecommendationSystem:
    def __init__(self, movie_data_path="tmdb_5000_movies.csv", music_data_path="extended_data_by_genres.csv", model_path="word2vec.model"):
        self.movie_data_path = movie_data_path
        self.music_data_path = music_data_path
        self.model_path = model_path
        self.nlp = spacy.load("en_core_web_sm") 

        # Load datasets
        self.movies_df = pd.read_csv(self.movie_data_path)
        self.music_df = pd.read_csv(self.music_data_path)

        # Preprocess genres
        self.movies_df["processed_genres"] = self.movies_df["genres"].apply(self.preprocess_genres)
        self.music_df["processed_genres"] = self.music_df["genres"].apply(lambda x: x.lower())

        # Load or train Word2Vec model
        self.word2vec_model = self.load_or_train_model()

        # Generate genre embeddings
        self.movies_df["genre_embeddings"] = self.movies_df["processed_genres"].apply(self.genre_vector)
        self.music_df["genre_embeddings"] = self.music_df["processed_genres"].apply(self.genre_vector)

        # Compute TF-IDF similarity matrix
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["processed_genres"])
        self.genre_sim_matrix = cosine_similarity(self.tfidf_matrix)


    def get_movie_mood(self, movie_genres: List[str]) -> List[str]:
        with open("genre_to_mood.json", "r") as f:
            mood_mapping = json.load(f)
        moods = set()
        for genre in movie_genres:
            if genre in mood_mapping:
                moods.update(mood_mapping[genre])
        return list(moods)
    def find_movie(self, movie_title: str) -> Union[None, pd.Series]:
        best_match = process.extractOne(movie_title, self.movies_df["title"].tolist(), score_cutoff=80)
        if best_match:
            return self.movies_df[self.movies_df["title"] == best_match[0]].iloc[0]
        return None
        
    def preprocess_genres(self, genres_str):
        """
        Extracts genre names from a JSON string and returns a space-separated string of genres.
        """
        try:
            genres_list = [genre["name"].lower() for genre in json.loads(genres_str)]
            return " ".join(genres_list)
        except:
            return ""

    def word_tokenize(self, text):
        """
        Tokenizes input text using SpaCy.
        Removes stop words and punctuation and returns lemmatized tokens.
        """
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def load_or_train_model(self):
        """
        Loads an existing Word2Vec model if available, otherwise trains a new one and saves it.
        """
        if os.path.exists(self.model_path):
            print("Loaded existing Word2Vec model.")
            return Word2Vec.load(self.model_path)
        
        print("Training new Word2Vec model...")
        all_genres = self.movies_df["processed_genres"].tolist() + self.music_df["processed_genres"].tolist()
        tokenized_genres = [self.word_tokenize(genres) for genres in all_genres]

        model = Word2Vec(sentences=tokenized_genres, vector_size=100, window=5, min_count=1, workers=4)
        model.save(self.model_path)
        print("New Word2Vec model trained and saved.")
        return model

    def genre_vector(self, genre_text):
        """
        Converts a genre text into a numerical vector using the trained Word2Vec model.
        """
        words = self.word_tokenize(genre_text)
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    def recommend_music(self, movie_title, num_recommendations=5):
        """
        Given a movie title, recommends the top N music tracks based on genre similarity.
        """
        idx = self.movies_df[self.movies_df["title"].str.lower() == movie_title.lower()].index
        if len(idx) == 0:
            return "Movie not found!"
        
        idx = idx[0]
        movie_vector = self.movies_df.iloc[idx]["genre_embeddings"]

        # Compute similarity with music tracks
        similarities = cosine_similarity([movie_vector], np.stack(self.music_df["genre_embeddings"].values))[0]
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        return self.music_df.iloc[top_indices]["track_names"].tolist()


class BidirectionalRecommendationSystem(GenreRecommendationSystem):
    def __init__(self, movie_data_path="tmdb_5000_movies.csv", music_data_path="extended_data_by_genres.csv", model_path="word2vec.model"):
        super().__init__(movie_data_path, music_data_path, model_path)
        
    def find_track(self, track_name: str) -> Union[None, pd.Series]:
        best_match = process.extractOne(track_name, self.music_df["track_names"].tolist(), score_cutoff=80)
        if best_match:
            return self.music_df[self.music_df["track_names"] == best_match[0]].iloc[0]
        return None

    def recommend_movies(self, track_name: str, num_recommendations=5):
        """
        Given a music track name, recommends the top N movies based on genre similarity.
        """
        idx = self.music_df[self.music_df["track_names"].str.lower() == track_name.lower()].index
        if len(idx) == 0:
            return "Track not found!"
        
        idx = idx[0]
        track_vector = self.music_df.iloc[idx]["genre_embeddings"]

        # Compute similarity with movies
        similarities = cosine_similarity([track_vector], np.stack(self.movies_df["genre_embeddings"].values))[0]
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        return self.movies_df.iloc[top_indices][["title", "overview", "vote_average", "popularity"]].to_dict('records')


def get_spotify_preview(track_name, limit=1):
    results = sp.search(q=track_name, type='track', limit=limit)
    
    songs = []
    for track in results['tracks']['items']:
        song_data = {
            "track_name": track["name"],
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track["preview_url"],
            "album_name": track["album"]["name"],
            "album_cover": track["album"]["images"][0]["url"] if track["album"]["images"] else None
        }
        songs.append(song_data)
    
    return pd.DataFrame(songs)

def save_tracks_to_file(file_path: str, tracks: List[Dict[str, Union[str, int]]]):
    with open(file_path, 'w') as file:
        for track in tracks:
            file.write(json.dumps(track) + '\n')

def load_tracks_from_file(file_path: str) -> List[Dict[str, Union[str, int]]]:
    try:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        return []

def render_movie_insights(movie: pd.Series):
    st.markdown('<div class="insights-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rating", f"⭐ {movie['vote_average']:.1f}")
    with col2:
        st.metric("Popularity", f"📈 {int(movie['popularity'])}")
    with col3:
        if 'budget' in movie:
            budget_m = movie['budget'] / 1_000_000
            st.metric("Budget", f"💰 ${budget_m:.1f}M")
    
    moods = recommender.get_movie_mood(movie['genres'])
    st.markdown("### Movie Mood")
    for mood in moods:
        st.markdown(f'<span class="mood-tag">{mood}</span>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_recommendation_stats(saved_tracks: List[Dict]):
    if saved_tracks:
        genres = [track['genre'] for track in saved_tracks]
        genre_counts = pd.Series(genres).value_counts()
        
        fig = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="Your Music Genre Distribution"
        )
        st.plotly_chart(fig)



def render_track_recommendation(rec: Dict, index: int, movie_title: str, saved_tracks: List):

    st.markdown(f'<div class="track-title">🎵 {rec}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="track-genre">Genre: {rec}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    spotify_data = get_spotify_preview(rec)
    if not spotify_data.empty:
        track_info = spotify_data.iloc[0]
        st.image(track_info["album_cover"], caption=track_info["album_name"], width=200)
        st.write(f"🎵 {track_info['track_name']} - {track_info['artist']}")
        st.markdown(f"[Listen on Spotify]({track_info['spotify_url']})")

    col1, col2 = st.columns([3, 1])
    with col1:
        rating = st.slider(
            "Rate this track",
            1, 5, 3,
            key=f"rating_{index}_{movie_title}",
            help="1 = Poor, 5 = Excellent"
        )
    
    with col2:
        if st.button("💾 Save", key=f"save_{index}_{movie_title}"):
            track_data = {
                'movie': movie_title,
                'track': rec['track_names'],
                'genre': rec['genres'],
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            }
            saved_tracks.append(track_data)
            save_tracks_to_file("saved_tracks.txt", saved_tracks)
            st.success("✅ Track saved!")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">Movie-and-Music Recommender</h1>', unsafe_allow_html=True)
    
    mode = st.radio("Select Mode", ["Movie → Music", "Music → Movies"], horizontal=True)

    with st.sidebar:
        st.markdown("### 🎯 Preferences")
        num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
        
        if mode == "Movie → Music":
            min_rating = st.slider("Minimum movie rating", 0.0, 10.0, 7.0)
            st.markdown("### 🎵 Genre Filters")
            genre_filter = st.multiselect(
                "Filter music genres",
                options=[
                    'electronic', 'rock', 'epic', 'bass music', 'drum and bass',
                    'world', 'cinematic', 'orchestral', 'folk', 'classical',
                    'ambient', 'abstract', 'pop', 'vocal', 'acoustic'
                ]
            )
        else:
            min_rating = st.slider("Minimum track popularity", 0.0, 100.0, 70.0)
            st.markdown("### 🎬 Movie Filters")
            movie_genre_filter = st.multiselect(
                "Filter movie genres",
                options=[
                    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                    'Thriller', 'War', 'Western'
                ]
            )

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if mode == "Movie → Music":
            movie_title = st.text_input(
                "",
                placeholder="Enter a movie title (e.g., The Dark Knight, Avatar, Inception...)",
                help="Type the name of a movie to get music recommendations"
            )
            
            if movie_title:
                movie = recommender.find_movie(movie_title)
                if movie is not None:
                    st.markdown("### 🎬 Movie Details")
                    render_movie_insights(movie)
                    
                    recommendations = recommender.recommend_music(
                        movie_title,
                        num_recommendations=num_recommendations
                    )
                    
                    if isinstance(recommendations, str):
                        st.warning(recommendations)
                    else:
                        st.markdown("### 🎵 Recommended Tracks")
                        for i, rec in enumerate(recommendations, 1):
                            render_track_recommendation(rec, i, movie_title, saved_tracks)
                else:
                    st.error("Movie not found. Please try another title.")
        
        else:  # Music → Movies mode
            track_name = st.text_input(
                "",
                placeholder="Enter a music track name",
                help="Type the name of a song to get movie recommendations"
            )
            
            if track_name:
                track = recommender.find_track(track_name)
                if track is not None:
                    st.markdown("### 🎵 Track Details")
                    spotify_data = get_spotify_preview(track["track_names"])
                    if not spotify_data.empty:
                        track_info = spotify_data.iloc[0]
                        st.image(track_info["album_cover"], caption=track_info["album_name"], width=200)
                        st.write(f"🎵 {track_info['track_name']} - {track_info['artist']}")
                        st.markdown(f"[Listen on Spotify]({track_info['spotify_url']})")
                    
                    recommendations = recommender.recommend_movies(
                        track_name,
                        num_recommendations=num_recommendations
                    )
                    
                    if isinstance(recommendations, str):
                        st.warning(recommendations)
                    else:
                        st.markdown("### 🎬 Recommended Movies")
                        for i, rec in enumerate(recommendations, 1):
                            render_movie_recommendation(rec, i, track_name, saved_movies)
                else:
                    st.error("Track not found. Please try another title.")
    
    with col2:
        st.markdown("### 📊 Your Stats")
        if mode == "Movie → Music":
            render_recommendation_stats(saved_tracks)
            st.markdown("### 📝 Recently Saved Tracks")
            for track in saved_tracks[-5:]:
                st.markdown(f"""
                    <div class="saved-track-card">
                        <strong>🎬 {track['movie']}</strong><br>
                        🎵 {track['track']}<br>
                        🎯 {track['genre']}<br>
                        ⭐ {"★" * track['rating']}{"☆" * (5-track['rating'])}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### 📝 Recently Saved Movies")
            for movie in saved_movies[-5:]:
                st.markdown(f"""
                    <div class="saved-movie-card">
                        <strong>🎵 {movie['track']}</strong><br>
                        🎬 {movie['movie']}<br>
                        ⭐ {"★" * movie['rating']}{"☆" * (5-track['rating'])}
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    recommender = BidirectionalRecommendationSystem("tmdb_5000_movies.csv", "extended_data_by_genres.csv")
    saved_tracks = load_tracks_from_file("saved_tracks.txt")
    saved_movies = load_tracks_from_file("saved_movies.txt")
    main()