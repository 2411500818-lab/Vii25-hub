# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob

nltk.download("stopwords")
nltk.download("punkt")

st.set_page_config(page_title="Analisis Twitter", layout="wide")
st.title("📊 Analisis Twitter – TF-IDF, WordCloud & Sentiment")

uploaded_file = st.file_uploader("Upload file CSV Twitter", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    stop_words = set(stopwords.words("indonesian"))
    stemmer = StemmerFactory().create_stemmer()

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^A-Za-z\s]", " ", text)
        text = text.lower()
        words = word_tokenize(text)
        words = [w for w in words if w not in stop_words]
        words = [stemmer.stem(w) for w in words]
        return " ".join(words)

    df["cleaned_text"] = df["full_text"].apply(clean_text)

    st.subheader("Hasil Preprocessing")
    st.dataframe(df[["full_text", "cleaned_text"]].head())

    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(df["cleaned_text"])

    tfidf_scores = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    word_scores = pd.DataFrame({
        "word": words,
        "score": tfidf_scores
    }).sort_values(by="score", ascending=False)

    st.subheader("Top 15 Kata TF-IDF")
    st.dataframe(word_scores.head(15))

    st.subheader("Bar Chart TF-IDF")
    fig, ax = plt.subplots()
    top10 = word_scores.head(10)
    ax.barh(top10["word"], top10["score"])
    ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("WordCloud")
    word_freq = dict(zip(word_scores["word"], word_scores["score"]))
    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate_from_frequencies(word_freq)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    def analyze_sentiment(text):
        if pd.isna(text) or text == "":
            return "Neutral"
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["full_text"].apply(analyze_sentiment)

    st.subheader("Distribusi Sentiment")
    sentiment_counts = df["sentiment"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(sentiment_counts.index, sentiment_counts.values)
    st.pyplot(fig2)

    st.subheader("Naive Bayes (Dummy Label dari Sentiment)")
    label_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
    y = df["sentiment"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.info("Silakan upload file CSV Twitter terlebih dahulu")
