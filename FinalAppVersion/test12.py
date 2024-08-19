import streamlit as st
import pandas as pd
import torch
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize session state variables
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'sentiment_filter' not in st.session_state:
    st.session_state.sentiment_filter = None
if 'wordcloud_sentiment' not in st.session_state:
    st.session_state.wordcloud_sentiment = None

def main():
    # Load sentiment data with enhanced error handling
    @st.cache_data
    def load_sentiment_data(uploaded_file) -> pd.DataFrame:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                if df.empty:
                    st.error("The uploaded file is empty. Please check the file content.")
                    return pd.DataFrame()
                return df
            else:
                st.error("Please upload a valid Excel file (.xlsx).")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

    # Load the sentiment analysis pipeline with GPU support and caching efficiency
    @st.cache_resource(show_spinner=False)
    def load_sentiment_pipeline():
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

    # Perform sentiment analysis in batches to improve performance with a progress bar
    def analyze_sentiments(comments, batch_size=25):
        sentiment_pipeline = load_sentiment_pipeline()
        
        valid_comments = [comment for comment in comments if isinstance(comment, str) and comment.strip()]
        
        if not valid_comments:
            return [], []
        
        sentiment_labels = []
        progress_bar = st.progress(0)
        total_batches = len(valid_comments) // batch_size + (1 if len(valid_comments) % batch_size > 0 else 0)
        
        for i in range(0, len(valid_comments), batch_size):
            batch = valid_comments[i:i + batch_size]
            results = sentiment_pipeline(batch)
            sentiment_labels.extend([result['label'] for result in results])
            progress_bar.progress((i + batch_size) / len(valid_comments))
        
        progress_bar.empty()
        return sentiment_labels, valid_comments

    # Visualize sentiment distribution as a pie chart
    def plot_sentiment_pie_chart(df):
        sentiment_counts = df['Sentiment'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.4)])
        fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
        st.plotly_chart(fig)

    # Function to generate word frequencies and create an interactive bar chart
    def plot_interactive_wordcloud(df, text_column, sentiment_filter):
        filtered_df = df[df['Sentiment'] == sentiment_filter]
        if filtered_df[text_column].dropna().empty:
            st.warning(f"No text data available for analysis under {sentiment_filter} sentiment.")
            return
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        try:
            tfidf_matrix = vectorizer.fit_transform(filtered_df[text_column].dropna())
            sum_words = tfidf_matrix.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            
            if not words_freq:
                st.warning(f"No words found under {sentiment_filter} sentiment.")
                return
            
            words_df = pd.DataFrame(words_freq, columns=['Word', 'TF-IDF Score'])
            
            fig = px.bar(words_df, x='TF-IDF Score', y='Word', orientation='h', title=f'Most Common Words for {sentiment_filter}', template="plotly_dark")
            fig.update_traces(marker_color='rgba(50, 171, 96, 0.6)', marker_line_color='rgba(50, 171, 96, 1.0)', marker_line_width=1.5)
            st.plotly_chart(fig)

            # Interactivity: select a word and filter comments
            selected_word = st.selectbox("Select a word to filter comments:", words_df['Word'])
            filtered_comments = filtered_df[filtered_df[text_column].str.contains(selected_word, case=False, na=False)]
            st.write(f"Comments containing the word '{selected_word}':")
            st.write(filtered_comments[[text_column, 'Sentiment']])
            
        except Exception as e:
            st.error(f"Error generating interactive wordcloud: {e}")

    # Generate wordcloud based on the selected sentiment with a progress bar
    def generate_wordcloud(text, title, max_words=100, max_font_size=80, collocations=False):
        if not text:
            st.warning("No text available to generate word cloud.")
            return
        
        stop_words = set(nltk.corpus.stopwords.words('english'))
        wordcloud = WordCloud(
            stopwords=stop_words,
            background_color='black',
            color_func=lambda *args, **kwargs: "white",
            max_words=max_words,
            max_font_size=max_font_size,
            collocations=collocations,
            width=800,
            height=400,
            prefer_horizontal=0.9
        ).generate(' '.join(text))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # Function to plot histogram and descriptive statistics for numeric data with a progress bar
    def analyze_numeric_column(df, column):
        st.subheader(f"Histogram and Descriptive Statistics for {column}")
        
        # Initialize progress bar
        progress_bar = st.progress(0)
        
        # Calculate descriptive statistics
        stats = df[column].describe()
        
        # Update progress bar after statistics calculation
        progress_bar.progress(33)
        
        # Plot histogram
        fig = px.histogram(df, x=column, nbins=20, title=f'Histogram of {column}', template="plotly_dark")
        st.plotly_chart(fig)
        
        # Update progress bar after plotting histogram
        progress_bar.progress(66)
        
        # Display descriptive statistics
        st.write(stats)
        
        # Final update to progress bar
        progress_bar.progress(100)
        progress_bar.empty()

    # Main app layout and interaction
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

                if pd.api.types.is_numeric_dtype(column_data):
                    # Handle numeric data
                    if st.button(f"Analyze {selected_column}"):
                        analyze_numeric_column(df, selected_column)
                elif column_data.dtype == object:  # Ensure the column contains text data
                    comments = column_data.dropna().tolist()  # Remove NaN and convert to list

                    if st.button(f"Analyze Sentiments in {selected_column}"):
                        with st.spinner('Analyzing sentiments...'):
                            sentiment_labels, valid_comments = analyze_sentiments(comments)
                            
                            if sentiment_labels and valid_comments:
                                # Store the results in session state
                                st.session_state.sentiment_df = pd.DataFrame({
                                    selected_column: valid_comments,
                                    'Sentiment': sentiment_labels
                                })
                                st.session_state.analysis_done = True
                                st.success("Sentiment analysis completed.")

            # Check if analysis is done and session state is available
            if st.session_state.analysis_done:
                sentiment_df = st.session_state.sentiment_df

                st.write("Sentiment Analysis Results:")
                st.dataframe(sentiment_df.head())

                # Visualize the sentiment distribution
                plot_sentiment_pie_chart(sentiment_df)

                # Display an interactive word cloud (bar chart) with filtering capability
                st.subheader("Interactive Word Cloud")
                st.session_state.sentiment_filter = st.selectbox("Filter by Sentiment", sentiment_df['Sentiment'].unique(), key="filter")
                
                if st.session_state.sentiment_filter:
                    plot_interactive_wordcloud(sentiment_df, selected_column, st.session_state.sentiment_filter)
                    st.subheader(f"Word Cloud for {st.session_state.sentiment_filter} Sentiment")
                    generate_wordcloud(sentiment_df[sentiment_df['Sentiment'] == st.session_state.sentiment_filter][selected_column].dropna().tolist(), f"Word Cloud for {st.session_state.sentiment_filter}")

if __name__ == "__main__":
    main()
