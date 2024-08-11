import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK stopwords
nltk.download('stopwords')

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
        futures = [executor.submit(sentiment_pipeline, comments[i:i+50]) for i in range(0, len(comments), 50)]
        for future in as_completed(futures):
            try:
                sentiments.extend(future.result())
            except Exception as e:
                st.error(f"Error during sentiment analysis: {e}")
                return df

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

# Generate wordcloud
def generate_wordcloud(text, title):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='black', colormap='plasma').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, color='white')
    st.pyplot(plt)

# Function to plot the most common words
def plot_top_words(df, text_column, num_words=10):
    if df[text_column].dropna().empty:
        st.warning("No text data available for common words analysis.")
        return
    
    vectorizer = CountVectorizer(stop_words='english')
    try:
        word_counts = vectorizer.fit_transform(df[text_column].dropna())
        sum_words = word_counts.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        if not words_freq:
            st.warning("No common words found. The text data may be too sparse or preprocessed incorrectly.")
            return
        
        top_words_df = pd.DataFrame(words_freq[:num_words], columns=['Word', 'Frequency'])
        fig = px.bar(top_words_df, x='Frequency', y='Word', orientation='h', title=f'Top {num_words} Most Common Words', template="plotly_dark")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating common words: {e}")

# Function to plot top bigrams
def plot_top_bigrams(df, text_column, num_bigrams=10):
    if df[text_column].dropna().empty:
        st.warning("No text data available for bigram analysis.")
        return
    
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2))
    try:
        bigram_counts = vectorizer.fit_transform(df[text_column].dropna())
        sum_bigrams = bigram_counts.sum(axis=0)
        bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
        bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
        
        if not bigrams_freq:
            st.warning("No bigrams found. The text data may be too sparse or preprocessed incorrectly.")
            return
        
        top_bigrams_df = pd.DataFrame(bigrams_freq[:num_bigrams], columns=['Bigram', 'Frequency'])
        fig = px.bar(top_bigrams_df, x='Frequency', y='Bigram', orientation='h', title=f'Top {num_bigrams} Occurring Bigrams', template="plotly_dark")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating bigrams: {e}")

# Function to plot top trigrams
def plot_top_trigrams(df, text_column, num_trigrams=10):
    if df[text_column].dropna().empty:
        st.warning("No text data available for trigram analysis.")
        return
    
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(3, 3))
    try:
        trigram_counts = vectorizer.fit_transform(df[text_column].dropna())
        sum_trigrams = trigram_counts.sum(axis=0)
        trigrams_freq = [(trigram, sum_trigrams[0, idx]) for trigram, idx in vectorizer.vocabulary_.items()]
        trigrams_freq = sorted(trigrams_freq, key=lambda x: x[1], reverse=True)
        
        if not trigrams_freq:
            st.warning("No trigrams found. The text data may be too sparse or preprocessed incorrectly.")
            return
        
        top_trigrams_df = pd.DataFrame(trigrams_freq[:num_trigrams], columns=['Trigram', 'Frequency'])
        fig = px.bar(top_trigrams_df, x='Frequency', y='Trigram', orientation='h', title=f'Top {num_trigrams} Occurring Trigrams', template="plotly_dark")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error generating trigrams: {e}")

# Generate and plot themes using LDA
def plot_themes(df, text_column, num_topics=3):
    if df[text_column].dropna().empty:
        st.warning("No text data available for thematic analysis.")
        return
    
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    try:
        text_data = vectorizer.fit_transform(df[text_column].dropna())
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(text_data)
        words = vectorizer.get_feature_names_out()
        themes = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-6:-1]
            top_features = [words[i] for i in top_features_ind]
            themes.append(" ".join(top_features))
        
        st.subheader("Identified Themes")
        for i, theme in enumerate(themes):
            st.write(f"**Theme {i + 1}:** {theme}")
    except Exception as e:
        st.error(f"Error generating thematic analysis: {e}")

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
    st.set_page_config(page_title="Data Analysis Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Data Analysis Dashboard")

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
                        filtered_text = df[selected_column].tolist() if sentiment_filter == 'All' else df[df['Sentiment Category'] == sentiment_filter][selected_column].tolist()
                        generate_wordcloud(filtered_text, f"Word Cloud ({sentiment_filter})")

                        st.subheader("Top Words")
                        plot_top_words(df, selected_column)

                        st.subheader("Top Bigrams")
                        plot_top_bigrams(df, selected_column)

                        st.subheader("Top Trigrams")
                        plot_top_trigrams(df, selected_column)

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
