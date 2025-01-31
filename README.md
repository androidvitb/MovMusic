# 🎮 Movie-to-Music Recommendation System 🎵

A Python-based web application that recommends music tracks based on the genres of movies entered by the user. This project uses **Streamlit** for the frontend, **Pyngrok** for exposing the app to the internet, and leverages movie and music datasets for genre-based recommendations.

### Repository Under: AcWoC'25
### Club: Android Club, VIT Bhopal University
---

## 📚 Features

- **Movie Genre Recognition**: Detects genres of a given movie.
- **Music Genre Mapping**: Matches movie genres to related music genres.
- **Music Recommendations**: Recommends tracks based on movie genres.
- **User-Friendly Interface**: Interactive web app for easy usage.
- **Link With Spotify**: For Song link is given by Spotify
- **Info. of Song**: Model also give album,image,Link.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Required Libraries:
  - `streamlit`
  - `pyngrok`

### Datasets

1. **Movies Dataset**: `tmdb_5000_movies.csv`  
   Contains movie information, including genres and titles.

2. **Music Dataset**: `extended_data_by_genres.csv`  
   Contains music genres and associated track names.

---

## 🔧 How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/YourUsername/MovieMusicRecommender.git
   cd MovieMusicRecommender
   ```
2. **Run the Script**:
    - Run all the Code Cells given in `MovMusic.ipynb` one by one.
   - Replace `YOUR_AUTH_TOKEN` in the last cell with your [Ngrok Auth Token](https://dashboard.ngrok.com/).
   - Run the last cell

3. **Access the Application**:
   - The app will be hosted locally, but the **Pyngrok** integration will provide a public URL.
   - Open the provided URL to access the app in your browser.

---

## 🖥️ Application Workflow

1. **Input**:
   - Enter the movie dataset path, music dataset path, and a movie title in the Streamlit web app.

2. **Processing**:
   - The app identifies the movie's genres.
   - Maps the genres to related music genres using a predefined mapping.
   - Searches the music dataset for matching tracks.

3. **Output**:
   - Displays a list of recommended music tracks for the movie's genres.

---

## 🧬 Key Components

### `GenreRecommendationSystem`
A Python class that handles:
- Loading and preprocessing movie and music datasets.
- Mapping movie genres to related music genres.
- Generating music recommendations.
- Spotify link to listen the song 

### Streamlit Interface
- Provides a user-friendly UI for entering inputs and displaying results.

### Pyngrok Integration
- Exposes the local Streamlit server to the internet, making it accessible from anywhere.

---

## 🗃️ Example Usage

- **Input**:  
  Movie Title: *Inception*

- **Output**:  
  Recommended Tracks:
  1. Genre: Electronic - Track: *Dreamscape Beats*
  2. Genre: Ambient - Track: *Space Journey*
  3. Genre: Cinematic - Track: *Epic Movie Score*

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

---

## 🙏 Acknowledgments

- **Datasets**:
  - [TMDB Movies Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata)
  - Custom Music Dataset
- **Libraries**:
  - [Streamlit](https://streamlit.io/)
  - [Pyngrok](https://pyngrok.readthedocs.io/)
  - [Pandas](https://pandas.pydata.org/)
