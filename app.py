
import os
import io
import zipfile
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
LOCAL_MOVIES = "movies.csv"

@st.cache_data(show_spinner=True)
def load_movies_df():
    """
    Returns a pandas DataFrame of movies with columns: movieId, title, genres.
    If movies.csv isn't present, downloads MovieLens small and extracts it.
    """
    if os.path.exists(LOCAL_MOVIES):
        return pd.read_csv(LOCAL_MOVIES)

    # Download the MovieLens small zip
    resp = requests.get(MOVIELENS_ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("ml-latest-small/movies.csv") as f:
            df = pd.read_csv(f)
    # Save for subsequent runs (and faster caching)
    df.to_csv(LOCAL_MOVIES, index=False)
    return df

@st.cache_data(show_spinner=False)
def build_similarity(df: pd.DataFrame):
    """
    Build the content-based similarity matrix from movie genres.
    Returns (vectorizer, cosine_sim matrix).
    """
    df = df.copy()
    df["genres"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
    cv = CountVectorizer(max_features=5000, stop_words="english")
    matrix = cv.fit_transform(df["genres"])
    sim = cosine_similarity(matrix)
    return cv, sim

def recommend(title: str, df: pd.DataFrame, sim_matrix, top_n: int = 5):
    if title not in df["title"].values:
        return []
    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    picks = [df.iloc[i[0]].title for i in scores[1 : top_n + 1]]
    return picks

# --------------- UI ---------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering using MovieLens (genres with CountVectorizer + cosine similarity).")

with st.spinner("Loading dataset..."):
    movies_df = load_movies_df()

_, sim_matrix = build_similarity(movies_df)

# Search + select UI
query = st.text_input("Search a movie title (optional):")
titles = movies_df["title"].tolist()
if query:
    filtered = [t for t in titles if query.lower() in t.lower()]
    if not filtered:
        st.info("No titles matched your search. Showing full list.")
        filtered = titles
else:
    filtered = titles

selected = st.selectbox("Choose a movie you like:", filtered, index=0 if filtered else None)
top_n = st.slider("How many recommendations?", 3, 15, 5)

if st.button("Recommend"):
    if not selected:
        st.warning("Please select a movie title.")
    else:
        recs = recommend(selected, movies_df, sim_matrix, top_n=top_n)
        if not recs:
            st.error("Selected movie not found in database.")
        else:
            st.subheader("Top Recommendations")
            for i, m in enumerate(recs, 1):
                st.write(f"{i}. {m}")

with st.expander("What is this doing?"):
    st.markdown(
        """
        **Method:** Content-based filtering using genres.
        1. Load MovieLens *small* dataset (automatically downloaded on first run).
        2. Convert genre strings to a bag‑of‑words representation using `CountVectorizer`.
        3. Compute cosine similarity between all movies.
        4. For the selected movie, return the top‑N most similar titles.
        """
    )
