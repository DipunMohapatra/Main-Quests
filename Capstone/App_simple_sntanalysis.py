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
import plotly.express as px
import plotly.graph_objects as go
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
        fig = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
        st.plotly_chart(fig)

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

        # Allow user to select a column for analysis
        column_to_analyze = st.selectbox("Select a column for analysis", data.columns)

        # Identify text and numeric columns
        text_columns = data.select_dtypes(include=['object']).columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        # Sentiment Analysis on Selected Text Column
        if column_to_analyze in text_columns:
            st.write("### Sentiment Analysis on Selected Text Column")
            
            texts = data[column_to_analyze].astype(str).dropna().tolist()
            sentiment_scores, sentiment_labels = analyze_sentiments(texts)
            sentiment_df = pd.DataFrame({
                'Text': texts,
                'Sentiment Score': sentiment_scores,
                'Sentiment': sentiment_labels
            })
            
            # Filtering options
            sentiment_filter = st.radio("Filter comments by sentiment", ("All", "Positive", "Negative"))
            if sentiment_filter != "All":
                sentiment_df = sentiment_df[sentiment_df['Sentiment'] == sentiment_filter]

            st.dataframe(sentiment_df)

            # Plot pie chart for sentiment distribution
            st.write("#### Sentiment Distribution")
            sentiment_counts = sentiment_df['Sentiment'].value_counts()
            fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Sentiment Distribution", hole=0.3)
            st.plotly_chart(fig)

            # Plot word cloud
            st.write("#### Word Cloud")
            wordcloud = generate_wordcloud(sentiment_df['Text'].tolist())
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # Plot theme identification
            st.write("#### Theme Identification")
            themes = identify_themes(sentiment_df['Text'].tolist())
            for theme, words in themes.items():
                st.write(f"**{theme}:**", ', '.join(words))

            # Plot most common words
            st.write("#### Most Common Words")
            common_words = most_common_words(sentiment_df['Text'].tolist())
            common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
            fig = px.bar(common_words_df, x='Frequency', y='Word', orientation='h', title='Most Common Words')
            st.plotly_chart(fig)

            # Plot bigrams
            st.write("#### Bigrams")
            bigrams = extract_bigrams(sentiment_df['Text'].tolist())
            bigrams_df = pd.DataFrame(bigrams, columns=['Bigram', 'Frequency'])
            fig = px.bar(bigrams_df, x='Frequency', y='Bigram', orientation='h', title='Bigrams')
            st.plotly_chart(fig)

        # Descriptive Statistics for Numeric Columns
        if column_to_analyze in numeric_columns:
            st.write("### Descriptive Statistics for Numeric Columns")
            st.dataframe(descriptive_statistics(data[[column_to_analyze]]))

            st.write("### Histograms for Numeric Columns")
            plot_histograms(data[[column_to_analyze]])

if __name__ == "__main__":
    main()

