#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import nltk

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Sentiment analysis function
def analyze_sentiments(texts):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(text) for text in texts]
    sentiment_scores = [s['compound'] for s in sentiments]
    sentiment_labels = ['Positive' if score >= 0 else 'Negative' for score in sentiment_scores]
    return sentiment_scores, sentiment_labels

# Word cloud generation
def generate_wordcloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Theme identification using LDA
def identify_themes(texts, num_topics=5, num_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_matrix)
    terms = vectorizer.get_feature_names_out()
    themes = {}
    for index, topic in enumerate(lda.components_):
        themes[f"Theme {index + 1}"] = [terms[i] for i in topic.argsort()[-num_words:]]
    return themes

# Most common words
def most_common_words(texts, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)
    sum_words = text_matrix.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:num_words]
    return words_freq

# Bigrams extraction
def extract_bigrams(texts, num_bigrams=10):
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    text_matrix = vectorizer.fit_transform(texts)
    sum_bigrams = text_matrix.sum(axis=0)
    bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)[:num_bigrams]
    return bigrams_freq

# Descriptive statistics for numeric columns
def descriptive_statistics(df):
    return df.describe().transpose()

# Histogram plotting for numeric columns
def plot_histograms(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df[column].dropna(), bins=20, alpha=0.7, edgecolor='black')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot(plt)

# Streamlit app
def main():
    st.title("Sentiment Analysis and Numeric Data Dashboard")

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])
    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file, delimiter="\n", header=None)
            data.columns = ["text"]

        # Identify text and numeric columns
        text_columns = data.select_dtypes(include=['object']).columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Sentiment Analysis on Text Columns
        if not text_columns.empty:
            st.write("### Sentiment Analysis on Text Columns")
            for col in text_columns:
                st.write(f"**Column:** {col}")
                texts = data[col].astype(str).dropna().tolist()
                sentiment_scores, sentiment_labels = analyze_sentiments(texts)
                sentiment_df = pd.DataFrame({
                    'Text': texts,
                    'Sentiment Score': sentiment_scores,
                    'Sentiment': sentiment_labels
                })
                st.dataframe(sentiment_df)

                st.write("#### Word Cloud")
                wordcloud = generate_wordcloud(texts)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

                st.write("#### Theme Identification")
                themes = identify_themes(texts)
                for theme, words in themes.items():
                    st.write(f"**{theme}:**", ', '.join(words))

                st.write("#### Most Common Words")
                common_words = most_common_words(texts)
                for word, freq in common_words:
                    st.write(f"{word}: {freq}")

                st.write("#### Bigrams")
                bigrams = extract_bigrams(texts)
                for bigram, freq in bigrams:
                    st.write(f"{bigram}: {freq}")

        # Descriptive Statistics for Numeric Columns
        if not numeric_columns.empty:
            st.write("### Descriptive Statistics for Numeric Columns")
            st.dataframe(descriptive_statistics(data[numeric_columns]))

            st.write("### Histograms for Numeric Columns")
            plot_histograms(data[numeric_columns])

if __name__ == "__main__":
    main()

