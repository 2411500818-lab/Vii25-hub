import streamlit as st
import pandas as pd
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

st.set_page_config(page_title="NLP Twitter Analysis", layout="wide")
st.title("🧠 Analisis NLP Twitter (Tanpa sklearn)")

uploaded_file = st.file_uploader("Upload file CSV Twitter", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Asli")
    st.write(f"Total data: {len(df)}")
    st.dataframe(df)

    text_col = st.selectbox("Pilih kolom teks", df.columns)

    st.sidebar.title("⚙️ Preprocessing")
    lowercase = st.sidebar.checkbox("Case Folding", True)
    remove_url = st.sidebar.checkbox("Remove URL", True)
    remove_symbol = st.sidebar.checkbox("Remove Symbol", True)

    def clean_text(text):
        text = str(text)
        if remove_url:
            text = re.sub(r"http\S+|www\S+", "", text)
        if remove_symbol:
            text = re.sub(r"[^a-zA-Z\s]", " ", text)
        if lowercase:
            text = text.lower()
        return text.strip()

    if st.button("🚀 Jalankan Analisis"):
        df["clean_text"] = df[text_col].apply(clean_text)

        st.subheader("📊 Statistik Data")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Data", len(df))
        c2.metric("Total Kata", df["clean_text"].str.split().str.len().sum())
        c3.metric("Rata-rata Panjang Teks", round(df["clean_text"].str.len().mean(), 2))

        # ================= TF-IDF MANUAL =================
        st.subheader("📐 TF-IDF Manual")

        docs = df["clean_text"].tolist()
        N = len(docs)

        term_freq = []
        doc_freq = Counter()

        for doc in docs:
            tf = Counter(doc.split())
            term_freq.append(tf)
            for word in tf:
                doc_freq[word] += 1

        tfidf_scores = {}
        for tf in term_freq:
            for word, count in tf.items():
                idf = math.log((N + 1) / (doc_freq[word] + 1)) + 1
                tfidf_scores[word] = tfidf_scores.get(word, 0) + count * idf

        tfidf_df = pd.DataFrame(tfidf_scores.items(), columns=["Kata", "TF-IDF"])
        tfidf_df = tfidf_df.sort_values(by="TF-IDF", ascending=False)

        st.dataframe(tfidf_df.head(20))

        fig, ax = plt.subplots()
        top10 = tfidf_df.head(10)
        ax.barh(top10["Kata"], top10["TF-IDF"])
        ax.invert_yaxis()
        st.pyplot(fig)

        # ================= WORDCLOUD =================
        st.subheader("☁️ WordCloud")
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(dict(tfidf_df.head(100).values))
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # ================= SENTIMENT =================
        st.subheader("😊 Analisis Sentimen")

        def sentiment_label(text):
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            return "Neutral"

        df["sentiment"] = df["clean_text"].apply(sentiment_label)

        sent_count = df["sentiment"].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(sent_count, labels=sent_count.index, autopct="%1.1f%%")
        st.pyplot(fig2)

        # ================= NAIVE BAYES MANUAL =================
        st.subheader("🤖 Naive Bayes (Manual)")

        labels = df["sentiment"].unique()
        label_counts = df["sentiment"].value_counts().to_dict()
        total_docs = len(df)

        word_counts = {label: Counter() for label in labels}
        total_words = {label: 0 for label in labels}

        for _, row in df.iterrows():
            label = row["sentiment"]
            words = row["clean_text"].split()
            for w in words:
                word_counts[label][w] += 1
                total_words[label] += 1

        vocab = set(tfidf_df["Kata"])
        vocab_size = len(vocab)

        def predict_nb(text):
            words = text.split()
            scores = {}
            for label in labels:
                log_prob = math.log(label_counts[label] / total_docs)
                for w in words:
                    word_freq = word_counts[label].get(w, 0) + 1
                    prob = word_freq / (total_words[label] + vocab_size)
                    log_prob += math.log(prob)
                scores[label] = log_prob
            return max(scores, key=scores.get)

        df["nb_prediction"] = df["clean_text"].apply(predict_nb)

        st.subheader("📋 Hasil Prediksi Naive Bayes")
        st.dataframe(df[[text_col, "sentiment", "nb_prediction"]].head(10))

        accuracy = (df["sentiment"] == df["nb_prediction"]).mean() * 100
        st.metric("Akurasi Naive Bayes", f"{accuracy:.2f}%")

        st.download_button(
            "⬇️ Download Hasil Analisis",
            df.to_csv(index=False),
            "hasil_nlp_twitter.csv",
            "text/csv"
        )
