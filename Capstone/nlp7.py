import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load sentiment data with caching
@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the sentiment analysis pipeline with GPU support
@st.cache_resource
def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

# Sentiment analysis with batch processing and threading
def analyze_sentiments(df, text_column):
    sentiment_pipeline = load_sentiment_pipeline()
    df[text_column] = df[text_column].astype(str)
    
    # Perform sentiment analysis in parallel using threading
    comments = df[text_column].tolist()
    sentiments = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(sentiment_pipeline, comments[i:i+10]) for i in range(0, len(comments), 10)]
        for future in as_completed(futures):
            sentiments.extend(future.result())

    # Extract sentiment ratings and add them to the DataFrame
    df['Sentiment Rating'] = [result['label'] for result in sentiments]
    df['Sentiment Category'] = df['Sentiment Rating'].apply(lambda x: 'Positive' if x in ['4 stars', '5 stars'] else 'Negative')
    
    return df

# Visualize sentiment distribution as a pie chart
def plot_sentiment_pie_chart(df):
    sentiment_counts = df['Sentiment Category'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.4)])
    fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
    st.plotly_chart(fig)

# Display comments by sentiment
def display_comments_by_sentiment(df, sentiment_type, text_column):
    filtered_df = df[df['Sentiment Category'] == sentiment_type]
    st.subheader(f"{sentiment_type} Comments")
    if not filtered_df.empty:
        st.dataframe(filtered_df[[text_column, 'Sentiment Rating']].reset_index(drop=True))
    else:
        st.write("No comments found.")

# Caching expensive operations
@st.cache_data
def generate_wordcloud(df, text_column, sentiment_type=None):
    if sentiment_type:
        text = df[df['Sentiment Category'] == sentiment_type][text_column].dropna().tolist()
    else:
        text = df[text_column].dropna().tolist()
    
    wordcloud = WordCloud(stopwords=set(), background_color='black', colormap='plasma').generate(' '.join(text))
    return wordcloud

# Plot top words
def plot_top_words(df, text_column, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(df[text_column].dropna())
    sum_words = word_counts.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_words_df = pd.DataFrame(words_freq[:num_words], columns=['Word', 'Frequency'])
    fig = px.bar(top_words_df, x='Frequency', y='Word', orientation='h', title=f'Top {num_words} Occurring Words', template="plotly_dark")
    st.plotly_chart(fig)

# Plot top bigrams
def plot_top_bigrams(df, text_column, num_bigrams=10):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2))
    bigram_counts = vectorizer.fit_transform(df[text_column].dropna())
    sum_bigrams = bigram_counts.sum(axis=0)
    bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    top_bigrams_df = pd.DataFrame(bigrams_freq[:num_bigrams], columns=['Bigram', 'Frequency'])
    fig = px.bar(top_bigrams_df, x='Frequency', y='Bigram', orientation='h', title=f'Top {num_bigrams} Occurring Bigrams', template="plotly_dark")
    st.plotly_chart(fig)

# Generate and plot themes using LDA
def plot_themes(df, text_column, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_data = vectorizer.fit_transform(df[text_column].dropna())
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)
    words = vectorizer.get_feature_names_out()
    themes = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-6:-1]
        top_features = [words[i] for i in top_features_ind]
        themes.append(" ".join(top_features))
    
    for i, theme in enumerate(themes):
        st.write(f"**Theme {i + 1}:** {theme}")

# Generate descriptive statistics and histogram for numeric columns
def plot_numeric_analysis(df, numeric_column):
    fig = px.histogram(df, x=numeric_column, nbins=20, title='Distribution of Ratings', labels={numeric_column: 'Rating', 'count': 'Frequency'}, template="plotly_dark")
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
    fig.update_yaxes(tickmode='auto')
    st.plotly_chart(fig)
    st.subheader("Descriptive Statistics")
    st.write(df[numeric_column].describe())

# Main app function
def main():
    st.set_page_config(page_title="RPC Feedback Analysis Dashboard", page_icon="📊", layout="wide")
    st.title("📊 RPC Data Analysis Dashboard")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select an analysis type from the tabs below:")

    sentiment_tab = st.tabs(["😊 Sentiment Analysis"])[0]
    
    with sentiment_tab:
        st.header("Sentiment Analysis")
        st.info("Upload your dataset and select the column to analyze.", icon="ℹ️")

        uploaded_sentiment_file = st.file_uploader("Upload your Excel dataset:", type=["xlsx"])
        
        if uploaded_sentiment_file:
            df = load_sentiment_data(uploaded_sentiment_file)
            if df.empty:
                st.warning("No data loaded. Please check your file.")
                return
            
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Allow the user to select one column for analysis
            selected_column = st.selectbox("Select a column for analysis:", df.columns)

            if selected_column:
                column_data = df[selected_column]
                if np.issubdtype(column_data.dtype, np.number):
                    # Handle numeric data
                    plot_numeric_analysis(df, selected_column)
                else:
                    # Handle text data for NLP tasks
                    st.subheader(f"Text Analysis for Column: {selected_column}")
                    if st.button(f"Analyze Sentiments in {selected_column}"):
                        with st.spinner('Analyzing sentiments...'):
                            df = analyze_sentiments(df, selected_column)
                            st.session_state.analysis_results = df
                            
                        st.header("Results")
                        plot_sentiment_pie_chart(df)  # Sentiment distribution chart

                        st.subheader("Word Cloud")
                        sentiment_filter = st.radio("Filter by Sentiment", ['All', 'Positive', 'Negative'], key=f"wordcloud_{selected_column}")
                        wordcloud = generate_wordcloud(df, selected_column, sentiment_filter if sentiment_filter != 'All' else None)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f"Word Cloud ({sentiment_filter if sentiment_filter else 'All'})", color='white')
                        st.pyplot(plt)

                        st.subheader("Top Words")
                        plot_top_words(df, selected_column)
                        
                        st.subheader("Top Bigrams")
                        plot_top_bigrams(df, selected_column)

                        st.subheader("Thematic Identification")
                        plot_themes(df, selected_column)

                    if "analysis_results" in st.session_state:
                        df = st.session_state.analysis_results
                        sentiment_choice = st.radio(
                            "Select sentiment to view comments:",
                            ['Positive', 'Negative'],
                            index=0,
                            key=f'sentiment_choice_{selected_column}'
                        )
                        display_comments_by_sentiment(df, sentiment_choice, selected_column)

if __name__ == "__main__":
    main()
