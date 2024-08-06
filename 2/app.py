#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from transformers import pipeline
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the sentiment analysis pipeline with a pre-trained transformer model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load sentiment data
@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    data = pd.read_excel(uploaded_file)  # Read Excel file
    return data

# Sentiment analysis using the pre-trained transformer model
def analyze_sentiments(df, text_column):
    # Convert the specified column to string type
    df[text_column] = df[text_column].astype(str)

    # Compute sentiment scores using the transformer model
    texts = df[text_column].tolist()
    
    # Analyze sentiment using the pre-trained model
    results = sentiment_pipeline(texts)

    # Extract sentiment label and score
    df['sentiment_label'] = [result['label'] for result in results]
    df['sentiment_score'] = [result['score'] for result in results]

    return df

def plot_sentiment_pie_chart(df):
    # Count the number of occurrences of each sentiment category
    sentiment_counts = df['sentiment_label'].value_counts()
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, 
                                 values=sentiment_counts.values,
                                 hoverinfo='label+percent',
                                 textinfo='value+percent')])
    fig.update_traces(marker=dict(colors=['#00CC96', '#EF553B', '#636EFA']))
    fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
    
    st.plotly_chart(fig)

def display_comments_by_sentiment(df, sentiment_type, text_column):
    filtered_df = df[df['sentiment_label'] == sentiment_type]
    st.subheader(f"{sentiment_type} Comments")
    if not filtered_df.empty:
        st.write(filtered_df[[text_column, 'sentiment_score']].reset_index(drop=True))
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
        page_title="Sentiment Analysis Dashboard",
        page_icon="😊",
        layout="wide",
    )

    st.title("😊 Sentiment Analysis Dashboard")
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
                sentiment_choice = st.radio("Select sentiment to view comments:", ['POSITIVE', 'NEGATIVE'])
                display_comments_by_sentiment(df, sentiment_choice, selected_column)

                # Top Words
                st.subheader("Top Words")
                plot_top_words(df, selected_column)

if __name__ == "__main__":
    main()

