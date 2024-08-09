import streamlit as st
import pandas as pd
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Load sentiment data with caching
@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load the sentiment analysis pipeline, leveraging GPU if available
@st.cache_resource
def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# Sentiment analysis with multi-threading and GPU support
def analyze_sentiments(df, text_column):
    sentiment_pipeline = load_sentiment_pipeline()
    df[text_column] = df[text_column].astype(str)
    
    batch_size = 32
    sentiments = []
    
    def process_batch(batch):
        return sentiment_pipeline(batch)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_batch, df[text_column].iloc[i:i+batch_size].tolist()) for i in range(0, len(df), batch_size)]
        for future in as_completed(futures):
            sentiments.extend(future.result())
    
    # Extract sentiments and ensure only Positive/Negative labels
    df['sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_scores'] = [s['score'] for s in sentiments]
    df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x == 'POSITIVE' else 'Negative')
    
    return df

# Visualize sentiment distribution as a pie chart
def plot_sentiment_pie_chart(df):
    sentiment_counts = df['sentiment_category'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hoverinfo='label+percent', textinfo='value+percent')])
    fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
    st.plotly_chart(fig)

# Display comments by sentiment
def display_comments_by_sentiment(df, sentiment_type, text_column):
    filtered_df = df[df['sentiment_category'] == sentiment_type]
    st.subheader(f"{sentiment_type} Comments")
    if not filtered_df.empty:
        st.write(filtered_df[[text_column, 'sentiment_scores']].reset_index(drop=True))
    else:
        st.write("No comments found.")

# Plot top occurring words
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

# Generate word cloud
def generate_wordcloud(text, title):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='black', colormap='plasma').generate(' '.join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, color='white')
    st.pyplot(plt)

# Identify themes using LDA
def identify_themes(df, text_column, num_topics=3):
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

# Plot histogram for numeric data
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
        st.info("Upload your dataset and select the column for sentiment analysis.", icon="ℹ️")

        uploaded_sentiment_file = st.file_uploader("Upload your Excel dataset for Sentiment Analysis:", type=["xlsx"])
        
        if uploaded_sentiment_file:
            df = load_sentiment_data(uploaded_sentiment_file)
            if df.empty:
                st.warning("No data loaded. Please check your file.")
                return
            
            st.write("Data Preview:")
            st.dataframe(df.head())

            column_options = df.columns.tolist()
            selected_column = st.selectbox("Select a column for analysis:", column_options)

            if "sentiment_choice" not in st.session_state:
                st.session_state.sentiment_choice = 'Positive'
            
            if st.button("Analyze"):
                if selected_column:
                    if pd.api.types.is_numeric_dtype(df[selected_column]):
                        plot_numeric_analysis(df, selected_column)
                    else:
                        with st.spinner('Analyzing sentiments...'):
                            df = analyze_sentiments(df, selected_column)
                            st.session_state.analysis_results = df
                        plot_sentiment_pie_chart(df)
                
            if "analysis_results" in st.session_state:
                df = st.session_state.analysis_results
                sentiment_choice = st.radio(
                    "Select sentiment to view comments:",
                    ['Positive', 'Negative'],  # Only Positive and Negative options
                    index=['Positive', 'Negative'].index(st.session_state.sentiment_choice),
                    key='sentiment_choice'
                )
                display_comments_by_sentiment(df, sentiment_choice, selected_column)
                st.subheader("Word Cloud")
                generate_wordcloud(df[selected_column].dropna(), "Word Cloud of Comments")
                st.subheader("Top Words")
                plot_top_words(df, selected_column)
                st.subheader("Top Bigrams")
                plot_top_bigrams(df, selected_column)
                st.subheader("Thematic Identification")
                themes = identify_themes(df, selected_column)
                for i, theme in enumerate(themes):
                    st.write(f"**Theme {i + 1}:** {theme}")

if __name__ == "__main__":
    main()
