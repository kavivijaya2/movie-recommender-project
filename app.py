
import streamlit as st
import joblib
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

def basic_clean(text):
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def preprocess_for_vectorizer(text):
    t = basic_clean(text)
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
    # tokens = [ps.stem(w) for w in tokens]  # optional
    return " ".join(tokens)

@st.cache_resource
def load_pipeline():
    return joblib.load("model_logreg.pkl")

st.set_page_config(page_title="Fake News Classifier", page_icon="📰")
st.title("📰 Fake News Classifier")
st.write("Paste a news **headline or paragraph** and get a prediction: **FAKE** or **REAL**.")

user_input = st.text_area("Enter news text:", height=200, placeholder="Type or paste news content here...")
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        pipe = load_pipeline()
        cleaned = preprocess_for_vectorizer(user_input)
        pred = pipe.predict([cleaned])[0]
        label = "REAL ✅" if pred==1 else "FAKE ❌"
        st.subheader(f"Prediction: {label}")
        st.caption("Model: TF‑IDF + Logistic Regression")
