import streamlit as st
import pandas as pd
import json
from difflib import get_close_matches
from typing import List, Dict, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


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

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List, Dict
from difflib import get_close_matches

import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List, Dict
from difflib import get_close_matches

class GenreRecommendationSystem:
    def __init__(self, movies_path: str, music_path: str):
        self.movies_df = self._load_movie_data(movies_path)
        self.music_genres_df = self._load_music_data(music_path)
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
        self.movie_genre_matrix = self._create_movie_genre_matrix()
        self.music_genre_matrix = self._create_music_genre_matrix()

    def _load_movie_data(self, path: str) -> pd.DataFrame:
        try:
            movies_df = pd.read_csv(path)
            movies_df['genres'] = movies_df['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)])
            return movies_df
        except Exception as e:
            raise ValueError(f"Error loading movie data: {e}")

    def _load_music_data(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error loading music data: {e}")

    def _create_movie_genre_matrix(self) -> pd.DataFrame:
        # Vectorize movie genres using CountVectorizer
        movie_genres_text = self.movies_df['genres'].apply(lambda x: ','.join(x))
        return self.vectorizer.fit_transform(movie_genres_text)

    def _create_music_genre_matrix(self) -> pd.DataFrame:
        # Vectorize music genres using CountVectorizer
        music_genres_text = self.music_genres_df['genres'].apply(lambda x: ','.join(x))
        return self.vectorizer.transform(music_genres_text)

    def find_movie(self, movie_title: str) -> Union[None, pd.Series]:
        matches = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False, na=False)]
        return matches.iloc[0] if not matches.empty else None

    def get_movie_mood(self, movie_genres: List[str]) -> List[str]:
        # Define mood mapping for various genres
        mood_mapping = {
            'Action': ['energetic', 'intense'],
            'Drama': ['emotional', 'reflective'],
            'Comedy': ['upbeat', 'light'],
            'Horror': ['dark', 'tense'],
            'Romance': ['gentle', 'warm'],
            'Adventure': ['exciting', 'dynamic'],
            'Fantasy': ['mystical', 'otherworldly'],
            'Science Fiction': ['futuristic', 'abstract'],
            'Crime': ['suspenseful', 'gritty'],
            'Thriller': ['suspenseful', 'high tension'],
            'Animation': ['whimsical', 'fun'],
            'Family': ['warm', 'heartfelt']
        }
        moods = set()
        for genre in movie_genres:
            if genre in mood_mapping:
                moods.update(mood_mapping[genre])
        return list(moods)

    def get_related_music_genres(self, movie_genres: List[str]) -> List[str]:
            # Create the vector for the movie genres
            movie_genre_text = ','.join(movie_genres)
            movie_vector = self.vectorizer.transform([movie_genre_text])
            
            # Compute cosine similarities between the movie and all music genres
            similarities = cosine_similarity(movie_vector, self.music_genre_matrix).flatten()
            
            # Get the indices of the top 5 most similar music genres
            related_indices = similarities.argsort()[-5:][::-1]
            
            # Log the related indices and the top genres
            print("Related indices:", related_indices)
            print("Top related genres:", self.music_genres_df.iloc[related_indices]['genres'].tolist())
            
            # Return the top related genres
            return self.music_genres_df.iloc[related_indices]['genres'].tolist()

    def recommend_music_based_on_movie(self, movie_title: str, num_recommendations: int = 5, 
                                     genre_filter: List[str] = None) -> Union[List[Dict[str, str]], str]:
        movie = self.find_movie(movie_title)
        if movie is None:
            return f"Movie not found. Did you mean one of these: {', '.join(get_close_matches(movie_title, self.movies_df['title'].tolist(), n=3))}?"

        movie_genres = movie['genres']
        related_music_genres = self.get_related_music_genres(movie_genres)
        
        if not related_music_genres:
            return "No matching music genres found for this movie's genres."
            
        if genre_filter:
            related_music_genres = [genre for genre in related_music_genres if genre in genre_filter]
            
        matching_music = self.music_genres_df[
            self.music_genres_df['genres'].str.lower().isin([g.lower() for g in related_music_genres])
        ]
        
        if matching_music.empty:
            return "No music recommendations found for the related genres."
            
        recommendations = matching_music.sample(n=min(num_recommendations, len(matching_music)))
        return recommendations[['genres', 'track_names']].to_dict('records')




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
    st.markdown('<div class="track-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="track-title">🎵 {rec["track_names"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="track-genre">Genre: {rec["genres"]}</div>', unsafe_allow_html=True)
    
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
    st.markdown('<h1 class="main-header">Movie-to-Music Recommender</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Preferences")
        num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
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
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
                
                recommendations = recommender.recommend_music_based_on_movie(
                    movie_title,
                    num_recommendations=num_recommendations,
                    genre_filter=genre_filter
                )
                
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.markdown("### 🎵 Recommended Tracks")
                    for i, rec in enumerate(recommendations, 1):
                        render_track_recommendation(rec, i, movie_title, saved_tracks)
            else:
                st.error("Movie not found. Please try another title.")
    
    with col2:
        st.markdown("### 📊 Your Music Stats")
        render_recommendation_stats(saved_tracks)
        
        st.markdown("### 📝 Recently Saved")
        recent_tracks = saved_tracks[-5:]
        for track in recent_tracks:
            st.markdown(f"""
                <div class="saved-track-card">
                    <strong>🎬 {track['movie']}</strong><br>
                    🎵 {track['track']}<br>
                    🎯 {track['genre']}<br>
                    ⭐ {"★" * track['rating']}{"☆" * (5-track['rating'])}
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    recommender = GenreRecommendationSystem("tmdb_5000_movies.csv", "extended_data_by_genres.csv")
    saved_tracks = load_tracks_from_file("saved_tracks.txt")
    main()