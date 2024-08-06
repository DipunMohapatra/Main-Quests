#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import matplotlib.pyplot as plt
from transformers import pipeline
import concurrent.futures
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Set up the page configuration as the first Streamlit command
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # Added download for POS tagger

# Initialize the sentiment analysis model using DistilBERT
@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading the sentiment model: {e}")
        return None

sentiment_pipeline = load_sentiment_pipeline()

def get_wordnet_pos(word):
    """
    Map POS tag to the first character lemmatize() accepts.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    """
    Enhanced text preprocessing that includes:
    - Lowercasing
    - Removing URLs and mentions
    - Removing punctuation and numbers
    - Correcting common misspellings
    - Removing stopwords
    - Lemmatization

    Args:
        text (str): The text to preprocess.
        
    Returns:
        str: The preprocessed text.
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Correct misspellings
    text = str(TextBlob(text).correct())

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords and lemmatize
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens if word not in stop_words])
    
    return text

@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    """Load sentiment data from an uploaded Excel file."""
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Data successfully loaded.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

def analyze_sentiments(df, text_column, positive_threshold=0.6, negative_threshold=0.4):
    """
    Analyze sentiments in a DataFrame column using a transformer-based model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text data.
        positive_threshold (float): Threshold above which sentiment is considered positive.
        negative_threshold (float): Threshold below which sentiment is considered negative.
        
    Returns:
        pd.DataFrame: DataFrame with sentiment scores and categories.
    """
    if sentiment_pipeline is None:
        st.error("Sentiment model is not loaded.")
        return df

    if text_column not in df:
        st.error(f"Column '{text_column}' does not exist in the data.")
        return df

    df[text_column] = df[text_column].astype(str)

    # Preprocess text data
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)

    # Filter out empty strings
    df = df[df['preprocessed_text'].str.strip() != ""]

    if df.empty:
        st.error("No valid text data available for sentiment analysis.")
        return df

    # Compute sentiment scores using a transformer-based model
    texts = df['preprocessed_text'].tolist()
    transformer_results = []

    # Batch processing for efficiency
    batch_size = 16
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(sentiment_pipeline, texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        for future in concurrent.futures.as_completed(futures):
            try:
                transformer_results.extend(future.result())
            except Exception as e:
                st.error(f"Error in sentiment analysis: {e}")

    # Extract sentiment label and score with thresholding
    sentiment_labels = []
    sentiment_scores = []

    for result, original_text in zip(transformer_results, texts):
        label = result['label']
        score = result['score']
        
        # Adjust label based on threshold
        if score >= positive_threshold and label == 'POSITIVE':
            label = 'POSITIVE'
        elif score <= negative_threshold and label == 'NEGATIVE':
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'

        sentiment_labels.append(label)
        sentiment_scores.append(score)

        # Debugging output to understand misclassification
        st.write(f"Original Text: {original_text}")
        st.write(f"Preprocessed Text: {df.loc[df['preprocessed_text'] == original_text, text_column].values[0]}")
        st.write(f"Predicted Sentiment: {label} | Score: {score}")

    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores

    st.write("Sentiment analysis completed.")
    return df

def plot_sentiment_pie_chart(df):
    """Plot a pie chart of sentiment distribution with percentages."""
    if 'sentiment_label' not in df:
        st.error("Sentiment analysis has not been performed.")
        return

    sentiment_counts = df['sentiment_label'].value_counts(normalize=True).reset_index()
    sentiment_counts.columns = ['Sentiment', 'Percentage']
    sentiment_counts['Percentage'] *= 100  # Convert to percentages

    if sentiment_counts.empty:
        st.error("No sentiment data available to plot.")
        return

    chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Percentage", type="quantitative"),
        color=alt.Color(field="Sentiment", type="nominal"),
        tooltip=[alt.Tooltip('Sentiment', type='nominal'),
                 alt.Tooltip('Percentage', format='.2f', title='Percentage (%)')]
    ).properties(
        title="Sentiment Distribution"
    )

    st.altair_chart(chart, use_container_width=True)

def display_comments_by_sentiment(df, sentiment_type, text_column):
    """Display comments filtered by a specific sentiment type."""
    if 'sentiment_label' not in df:
        st.error("Sentiment analysis has not been performed.")
        return

    # Convert sentiment_type to uppercase to match transformer results
    sentiment_type = sentiment_type.upper()

    filtered_df = df[df['sentiment_label'] == sentiment_type]
    st.subheader(f"{sentiment_type.capitalize()} Comments")
    if not filtered_df.empty:
        st.dataframe(filtered_df[[text_column, 'sentiment_score']].reset_index(drop=True))
    else:
        st.write(f"No comments found for {sentiment_type.lower()} sentiment.")

def plot_top_words(df, text_column, num_words=10):
    """Plot the top occurring words in the text data."""
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(df[text_column].dropna())
    
    sum_words = word_counts.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    top_words_df = pd.DataFrame(words_freq[:num_words], columns=['Word', 'Frequency'])

    chart = alt.Chart(top_words_df).mark_bar().encode(
        x=alt.X('Frequency', sort='-y'),
        y=alt.Y('Word', sort='-x'),
        tooltip=['Word', 'Frequency']
    ).properties(
        title=f'Top {num_words} Occurring Words'
    )

    st.altair_chart(chart, use_container_width=True)

def plot_top_bigrams(df, text_column, num_bigrams=10):
    """Plot the top occurring bigrams in the text data."""
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(2, 2))
    bigram_counts = vectorizer.fit_transform(df[text_column].dropna())
    
    sum_bigrams = bigram_counts.sum(axis=0)
    bigrams_freq = [(bigram, sum_bigrams[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]
    bigrams_freq = sorted(bigrams_freq, key=lambda x: x[1], reverse=True)
    
    top_bigrams_df = pd.DataFrame(bigrams_freq[:num_bigrams], columns=['Bigram', 'Frequency'])

    chart = alt.Chart(top_bigrams_df).mark_bar().encode(
        x=alt.X('Frequency', sort='-y'),
        y=alt.Y('Bigram', sort='-x'),
        tooltip=['Bigram', 'Frequency']
    ).properties(
        title=f'Top {num_bigrams} Occurring Bigrams'
    )

    st.altair_chart(chart, use_container_width=True)

def generate_wordcloud_based_on_sentiment(df, sentiment_type):
    """Generate a word cloud based on the sentiment."""
    # Convert sentiment to uppercase to match transformer results
    sentiment_type = sentiment_type.upper()

    # Filter the dataframe by sentiment
    sentiment_texts = df[df['sentiment_label'] == sentiment_type]['preprocessed_text'].dropna().values

    if len(sentiment_texts) == 0:
        st.write(f"No {sentiment_type.lower()} sentiments found.")
        return

    # Join all the text for this sentiment
    text = " ".join(sentiment_texts)

    # Generate word cloud
    stop_words = set(nltk.corpus.stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='black', colormap='plasma').generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment_type} Sentiment", color='white')
    st.pyplot(plt)

def identify_themes(df, text_column, num_topics=3):
    """Identify themes in the text data using Latent Dirichlet Allocation (LDA)."""
    vectorizer = CountVectorizer(stop_words='english')
    text_data = vectorizer.fit_transform(df[text_column])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_data)
    
    words = vectorizer.get_feature_names_out()
    themes = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[:-6:-1]
        top_features = [words[i] for i in top_features_ind]
        themes.append(" ".join(top_features))
    
    return themes

def plot_numeric_analysis(df, numeric_column):
    """Plot a histogram and descriptive statistics for a numeric column."""
    hist = alt.Chart(df).mark_bar().encode(
        alt.X(numeric_column, bin=alt.Bin(maxbins=30), title='Rating'),
        y='count()',
        tooltip=[numeric_column, 'count()']
    ).properties(
        title='Distribution of Ratings'
    )

    st.altair_chart(hist, use_container_width=True)
    
    st.subheader("Descriptive Statistics")
    st.write(df[numeric_column].describe())

def main():
    """Main function to run the Streamlit app."""
    st.title("📊 Data Analysis Dashboard")
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select an analysis type from the tabs below:")

    # Create a tab for sentiment analysis
    tabs = st.tabs(["😊 Sentiment Analysis"])
    
    # Access the sentiment tab by indexing the list of tabs
    with tabs[0]:
        st.header("Sentiment Analysis")
        st.info("Upload your dataset and select the column for sentiment analysis.", icon="ℹ️")

        uploaded_sentiment_file = st.file_uploader("Upload your Excel dataset for Sentiment Analysis:", type=["xlsx"])
        
        if uploaded_sentiment_file:
            df = load_sentiment_data(uploaded_sentiment_file)
            if not df.empty:
                st.write("Data Preview:")
                st.dataframe(df.head())

                column_options = df.columns.tolist()
                selected_column = st.selectbox("Select a column for analysis:", column_options)

                if selected_column:
                    if np.issubdtype(df[selected_column].dtype, np.number):
                        plot_numeric_analysis(df, selected_column)
                    else:
                        df = analyze_sentiments(df, selected_column)

                        plot_sentiment_pie_chart(df)

                        sentiment_choice = st.radio("Select sentiment to view comments and word cloud:", ['Positive', 'Negative', 'Neutral'])
                        display_comments_by_sentiment(df, sentiment_choice, selected_column)

                        st.subheader("Word Cloud")
                        generate_wordcloud_based_on_sentiment(df, sentiment_choice)

                        st.subheader("Top Words")
                        plot_top_words(df, 'preprocessed_text')

                        st.subheader("Top Bigrams")
                        plot_top_bigrams(df, 'preprocessed_text')

                        st.subheader("Thematic Identification")
                        themes = identify_themes(df, 'preprocessed_text')
                        for i, theme in enumerate(themes):
                            st.write(f"**Theme {i + 1}:** {theme}")

if __name__ == "__main__":
    main()

