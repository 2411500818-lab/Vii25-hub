import streamlit as st
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob

st.set_page_config(page_title="Analisis Twitter NLP", layout="wide")
st.title("🧠 Analisis NLP Twitter (Tanpa Sklearn)")

uploaded_file = st.file_uploader("Upload CSV Twitter", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Asli")
    st.write(f"Total data: {len(df)}")
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    # ================= PREPROCESSING =================
    st.sidebar.title("⚙️ Preprocessing")
    lower = st.sidebar.checkbox("Case Folding", True)
    remove_url = st.sidebar.checkbox("Remove URL", True)
    remove_symbol = st.sidebar.checkbox("Remove Symbol", True)

    def clean_text(text):
        text = str(text)
        if remove_url:
            text = re.sub(r"http\S+|www\S+", "", text)
        if remove_symbol:
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
        if lower:
            text = text.lower()
        return text.strip()

    if st.button("🚀 Jalankan Analisis"):
        df["clean_text"] = df[text_col].apply(clean_text)

        # ================= STATISTIK =================
        st.subheader("📊 Statistik")
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Data", len(df))
        col2.metric("Total Kata", df["clean_text"].str.split().str.len().sum())
        col3.metric("Rata-rata Panjang Teks", round(df["clean_text"].str.len().mean(), 2))

        # ================= TF-IDF MANUAL =================
        st.subheader("📐 TF-IDF (Manual)")

        docs = df["clean_text"].tolist()
        N = len(docs)

        tf = []
        df_count = Counter()

        for doc in docs:
            words = doc.split()
            counts = Counter(words)
            tf.append(counts)
            for w in counts:
                df_count[w] += 1

        tfidf_scores = {}
        for i, doc_tf in enumerate(tf):
            for word, freq in doc_tf.items():
                idf = math.log(N / (df_count[word]))
                tfidf_scores[word] = tfidf_scores.get(word, 0) + freq * idf

        tfidf_df = pd.DataFrame(
            tfidf_scores.items(),
            columns=["Kata", "TF-IDF"]
        ).sort_values(by="TF-IDF", ascending=False)

        st.dataframe(tfidf_df.head(20))

        # ================= WORDCLOUD =================
        st.subheader("☁️ WordCloud")
        if not tfidf_df.empty:
            wc = WordCloud(width=800, height=400, background_color="white")
            wc.generate_from_frequencies(dict(tfidf_df.head(100).values))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

        # ================= SENTIMENT =================
        st.subheader("😊 Analisis Sentimen")

        def sentiment(text):
            p = TextBlob(text).sentiment.polarity
            if p > 0.1:
                return "Positive"
            elif p < -0.1:
                return "Negative"
            return "Neutral"

        df["Sentiment"] = df["clean_text"].apply(sentiment)

        sent = df["Sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sent, labels=sent.index, autopct="%1.1f%%")
        st.pyplot(fig)

        st.download_button(
            "⬇️ Download Hasil",
            df.to_csv(index=False),
            "hasil_nlp.csv",
            "text/csv"
        )
