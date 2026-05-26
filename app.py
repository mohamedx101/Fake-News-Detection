import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection System")
st.markdown("### Detect whether a news article is Real or Fake")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

news_text = st.text_area("Enter the news article text:", height=300)

if st.button("🔍 Check News"):
    if news_text.strip():
        text_vec = vectorizer.transform([news_text])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        
        if prediction == 1:
            st.success("✅ This news appears to be **REAL**")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")
        else:
            st.error("🚨 This news appears to be **FAKE**")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
    else:
        st.warning("Please enter some text")