import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load your trained model and TF-IDF vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Offline prediction function
def predict_news(news_text):
    cleaned = clean_text(news_text)
    if cleaned.strip() == "":
        return "⚠️ Please enter meaningful news text."
    try:
        tfidf = vectorizer.transform([cleaned])
        proba = model.predict_proba(tfidf)[0]
        st.sidebar.write(f"🔍 Debug: Real news = {proba[1]:.4f}, Fake news = {proba[0]:.4f}")
        threshold = 0.3
        return "🟢 Real News" if proba[1] > threshold else "🔴 Fake News"
    except Exception as e:
        return f"❌ Error during prediction: {e}"

# Streamlit UI config
st.set_page_config(page_title="📰 Fake News Detector", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        color: #0a9396;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0;
    }
    .subheader {
        color: #005f73;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #555;
        margin-top: 50px;
    }
    .stButton>button {
        background-color: #0a9396;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
    }
    .stButton>button:hover {
        background-color: #005f73;
        color: #caf0f8;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Check if a news article is Real or Fake using Offline AI Model</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    news_input = st.text_area("Enter a news headline or article", height=200)

with col2:
    st.markdown("### Instructions")
    st.markdown("""
    - Paste the news headline or paragraph.
    - Click **Check News** to get the prediction.
    - This version uses an offline AI model trained on a Kaggle dataset.
    """)

# Button to predict
if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        result = predict_news(news_input)

        if "Real News" in result:
            st.success(f"Prediction: {result}")
        elif "Fake News" in result:
            st.error(f"Prediction: {result}")
        else:
            st.warning(result)

# Footer
st.markdown('<div class="footer">👨‍💻 Created by <b>Kaneeshkar</b></div>', unsafe_allow_html=True)
