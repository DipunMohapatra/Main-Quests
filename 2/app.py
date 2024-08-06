#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import concurrent.futures  # For parallel processing
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the sentiment analysis model once at the start
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load sentiment data
@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    data = pd.read_excel(uploaded_file)
    return data

# Improved Sentiment analysis and visualization
def analyze_sentiments(df, text_column):
    # Convert the specified column to string type
    df[text_column] = df[text_column].astype(str)

    # Initialize NLTK sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment scores using NLTK VADER
    df['nltk_sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['nltk_sentiment_category'] = df['nltk_sentiment_scores'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )

    # Compute sentiment scores using a transformer-based model (batch processing)
    texts = df[text_column].tolist()

    # Use batch processing and parallelization
    transformer_results = []
    batch_size = 16  # You can adjust this based on your hardware capabilities

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(sentiment_pipeline, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        for future in concurrent.futures.as_completed(futures):
            transformer_results.extend(future.result())

    # Extract sentiment label and score
    df['transformer_sentiment_label'] = [result['label'] for result in transformer_results]
    df['transformer_sentiment_score'] = [result['score'] for result in transformer_results]

    # Map transformer sentiment labels to categories
    df['transformer_sentiment_category'] = df['transformer_sentiment_label'].apply(
        lambda x: 'Positive' if x == 'POSITIVE' else ('Negative' if x == 'NEGATIVE' else 'Neutral')
    )

    # Compare and decide final sentiment category based on the most confident prediction
    df['final_sentiment_category'] = np.where(
        df['transformer_sentiment_score'] > 0.6,  # Threshold for confident prediction
        df['transformer_sentiment_category'],
        df['nltk_sentiment_category']
    )

    return df

def plot_sentiment_pie_chart(df):
    # Count the number of occurrences of each sentiment category
    sentiment_counts = df['final_sentiment_category'].value_counts()
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, 
                                 values=sentiment_counts.values,
                                 hoverinfo='label+percent',
                                 textinfo='value+percent')])
    fig.update_traces(marker=dict(colors=['#00CC96', '#EF553B', '#636EFA']))
    fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
    
    st.plotly_chart(fig)

def display_comments_by_sentiment(df, sentiment_type, text_column):
    filtered_df = df[df['final_sentiment_category'] == sentiment_type]
    st.subheader(f"{sentiment_type} Comments")
    if not filtered_df.empty:
        st.write(filtered_df[[text_column, 'nltk_sentiment_scores', 'transformer_sentiment_score']].reset_index(drop=True))
    else:
        st.write("No comments found.")

def plot_top_words(df, text_column, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')  # Use 'english' for built-in stop words
    word_counts = vectorizer.fit_transform(df[text_column].dropna())
    
    sum_words = word_counts.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    top_words_df = pd.DataFrame(words_freq[:num_words], columns=['Word', 'Frequency'])

    # Plot bar chart using Plotly Express
    fig = px.bar(top_words_df, x='Frequency', y='Word', orientation='h', 
                 title=f'Top {num_words} Occurring Words',
                 template="plotly_dark")
    st.plotly_chart(fig)

def plot_top_bigrams(df, text_column, num_bigrams=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2))  # Use 'english' for built-in stop words
    bigram_counts = vectorizer.fit_transform(df[text_column].dropna())
    
    sum_bigrams = bigram_counts.sum(axis=0)
    bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    
    top_bigrams_df = pd.DataFrame(bigrams_freq[:num_bigrams], columns=['Bigram', 'Frequency'])

    # Plot bar chart using Plotly Express
    fig = px.bar(top_bigrams_df, x='Frequency', y='Bigram', orientation='h', 
                 title=f'Top {num_bigrams} Occurring Bigrams',
                 template="plotly_dark")
    st.plotly_chart(fig)

def generate_wordcloud(text, title):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='black', colormap='plasma').generate(' '.join(text))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, color='white')
    st.pyplot(plt)

def identify_themes(df, text_column, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')  # Use 'english' for built-in stop words
    text_data = vectorizer.fit_transform(df[text_column])
    
    # Use LDA to identify themes
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)
    
    # Display the themes
    words = vectorizer.get_feature_names_out()
    themes = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-6:-1]
        top_features = [words[i] for i in top_features_ind]
        themes.append(" ".join(top_features))
    
    return themes

def plot_numeric_analysis(df, numeric_column):
    # Create a histogram using Plotly
    fig = px.histogram(df, x=numeric_column, nbins=20, title='Distribution of Values',
                       labels={numeric_column: 'Value', 'count': 'Frequency'}, template="plotly_dark")
    fig.update_xaxes(tickmode='linear', tick0=0)
    fig.update_yaxes(tickmode='auto')
    st.plotly_chart(fig)
    
    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df[numeric_column].describe())

# Main app
def main():
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Data Analysis Dashboard")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
            font-family: 'Open Sans', sans-serif;
        }
        .css-1lcbmhc, .css-1y4p8pa {
            background-color: #1e1e1e;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        h1, h2, h3, h4, h5, h6, p, label, .stTextInput label {
            color: white;
        }
        .stTabs [role="tablist"] .tab-label {
            color: white !important;
        }
        .stTabs [role="tablist"] .st-tab:hover {
            background-color: #333333;
        }
        .stTabs [role="tablist"] .st-tab[data-selected="true"] {
            border-bottom: 2px solid #2196f3 !important;
            color: #2196f3 !important;
        }
        .st-infobox {
            background-color: #333333;
            border: 1px solid #555555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Upload your dataset and select a column for analysis.")

    # File uploader for data analysis
    uploaded_file = st.file_uploader("Upload your Excel dataset for Analysis:", type=["xlsx"])
    
    if uploaded_file:
        df = load_sentiment_data(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Select a column for analysis
        column_options = df.columns.tolist()
        selected_column = st.selectbox("Select a column for analysis:", column_options)

        if selected_column:
            if np.issubdtype(df[selected_column].dtype, np.number):
                # Perform numeric analysis if column is of numeric type
                st.subheader("Numeric Analysis")
                plot_numeric_analysis(df, selected_column)
            else:
                # Perform sentiment analysis if column is of string type
                st.subheader("Sentiment Analysis")
                df = analyze_sentiments(df, selected_column)

                # Visualize sentiment distribution and comments
                plot_sentiment_pie_chart(df)

                # Display comments by sentiment
                sentiment_choice = st.radio("Select sentiment to view comments:", ['Positive', 'Negative', 'Neutral'])
                display_comments_by_sentiment(df, sentiment_choice, selected_column)

                # Word Cloud
                st.subheader("Word Cloud")
                generate_wordcloud(df[selected_column].dropna(), "Word Cloud of Comments")

                # Top Words
                st.subheader("Top Words")
                plot_top_words(df, selected_column)

                # Top Bigrams
                st.subheader("Top Bigrams")
                plot_top_bigrams(df, selected_column)

                # Thematic identification
                st.subheader("Thematic Identification")
                themes = identify_themes(df, selected_column)
                for i, theme in enumerate(themes):
                    st.write(f"**Theme {i + 1}:** {theme}")

if __name__ == "__main__":
    main()

