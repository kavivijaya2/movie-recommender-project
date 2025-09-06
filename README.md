# 🎬 Movie Recommendation System (Streamlit)

Content-based movie recommender using MovieLens genres (CountVectorizer + cosine similarity).

## Files
- `app.py` — Streamlit application (auto-downloads MovieLens small on first run)
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

On first run, the app downloads the MovieLens **small** dataset and caches `movies.csv` locally.
