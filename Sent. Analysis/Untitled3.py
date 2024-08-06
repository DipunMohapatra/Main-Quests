#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import pandas as pd
from time_series_analysis import TimeSeriesAnalysis
from sentiment_analysis import SentimentAnalysis
from visualization import Visualizer

def main():
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Data Analysis Dashboard")

    # Custom CSS for UI enhancements
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
    st.sidebar.markdown("Select an analysis type from the tabs below:")

    # Create tabs for different analysis types
    analysis_tab, sentiment_tab = st.tabs(["📈 Time Series Analysis", "😊 Sentiment Analysis"])

    # Time Series Analysis
    with analysis_tab:
        st.header("Time Series Analysis")
        st.info("Upload your dataset and select the sheet for analysis.", icon="ℹ️")

        # File uploader for time series analysis
        uploaded_file = st.file_uploader("Upload your Excel dataset for Time Series:", type=["xlsx"])
        
        if uploaded_file:
            try:
                sheet_name = st.text_input("Enter the sheet name for Time Series:", "")
                if sheet_name:
                    data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    ts_analysis = TimeSeriesAnalysis(data)

                    # Dynamic column selection
                    time_column = st.selectbox("Select the Time column:", data.columns)
                    value_column = st.selectbox("Select the Value column:", data.columns)

                    # Perform time series analysis
                    ts_analysis.perform_analysis(time_column, value_column)

            except Exception as e:
                st.error(f"Error loading data: {e}")

    # Sentiment Analysis
    with sentiment_tab:
        st.header("Sentiment Analysis")
        st.info("Upload your dataset and select the column for sentiment analysis.", icon="ℹ️")

        # File uploader for sentiment analysis
        uploaded_sentiment_file = st.file_uploader("Upload your Excel dataset for Sentiment Analysis:", type=["xlsx"])
        
        if uploaded_sentiment_file:
            try:
                df = pd.read_excel(uploaded_sentiment_file)
                st.write("Data Preview:")
                st.dataframe(df.head())

                # Dynamic column selection
                column_options = df.columns.tolist()
                selected_column = st.selectbox("Select a column for analysis:", column_options)

                if selected_column:
                    sentiment_analysis = SentimentAnalysis(df, selected_column)
                    sentiment_analysis.perform_analysis()

            except Exception as e:
                st.error(f"Error loading data: {e}")

if __name__ == "__main__":
    main()


# In[ ]:


# time_series_analysis.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
from visualization import Visualizer

class TimeSeriesAnalysis:
    def __init__(self, data):
        self.data = data
        self.visualizer = Visualizer()

    def perform_analysis(self, time_column, value_column):
        try:
            self.data['Time'] = pd.to_datetime(self.data[time_column])
            self.data['Value'] = self.data[value_column]

            st.subheader("Original Time Series Data")
            st.write(self.data.head())

            # Decompose the time series using STL
            stl = STL(self.data['Value'], seasonal=13, period=12)
            result = stl.fit()
            self.data['Trend'] = result.trend
            self.data['Seasonal'] = result.seasonal
            self.data['Residual'] = result.resid

            # Split train and test for the residual component
            train_residual = self.data.iloc[:-int(len(self.data) * 0.2)]
            test_residual = self.data.iloc[-int(len(self.data) * 0.2):]

            # Build ARIMA model on the residual component
            model = ARIMA(train_residual['Residual'], order=(10, 1, 10), seasonal_order=(1, 1, 1, 12)).fit()
            forecasts = model.forecast(len(test_residual))

            # Combine forecasts with trend and seasonal components
            trend_forecast = self.data['Trend'].iloc[-len(test_residual):].values
            seasonal_forecast = self.data['Seasonal'].iloc[-len(test_residual):].values
            final_forecasts = trend_forecast + seasonal_forecast + forecasts

            # Plot the final forecasts
            st.subheader("Hybrid STL + SARIMA Forecast")
            self.visualizer.plot_forecasts(train_residual, test_residual, final_forecasts, pd.DataFrame(), 'Hybrid STL + SARIMA Forecast')

            # Evaluate model fit
            mae = mean_absolute_error(test_residual['Value'], final_forecasts)
            mse = mean_squared_error(test_residual['Value'], final_forecasts)
            rmse = np.sqrt(mse)

            st.write(f'Mean Absolute Error (MAE): {mae}')
            st.write(f'Mean Squared Error (MSE): {mse}')
            st.write(f'Root Mean Squared Error (RMSE): {rmse}')

            # Get user input for forecast horizon
            months_to_forecast = st.slider("Select the number of months to forecast:", min_value=1, max_value=36, value=12, step=1)

            # Generate future forecast
            last_date = self.data['Time'].max()
            future_dates = pd.date_range(last_date, periods=months_to_forecast + 1, freq='M')[1:]

            future_trend = self.data['Trend'].iloc[-12:].values.reshape(-1, 1).repeat(months_to_forecast, axis=1).flatten()[:months_to_forecast]
            future_seasonal = self.data['Seasonal'].iloc[-12:].values[:months_to_forecast]

            future_residual = model.forecast(months_to_forecast)

            future_forecast = future_trend + future_seasonal + future_residual

            future_df = pd.DataFrame({'Time': future_dates, 'Forecast': future_forecast})

            # Plot future forecasts
            st.subheader("Future Forecast")
            self.visualizer.plot_forecasts(train_residual, test_residual, final_forecasts, future_df, 'Future Forecast')

        except Exception as e:
            st.error(f"Error performing time series analysis: {e}")


# In[ ]:


# sentiment_analysis.py
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import streamlit as st
from visualization import Visualizer

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalysis:
    def __init__(self, data, text_column):
        self.data = data
        self.text_column = text_column
        self.visualizer = Visualizer()

    def analyze_sentiments(self):
        # Convert the specified column to string type
        self.data[self.text_column] = self.data[self.text_column].astype(str)
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Compute sentiment scores
        self.data['sentiment_scores'] = self.data[self.text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
        self.data['sentiment_category'] = self.data['sentiment_scores'].apply(
            lambda x: 'Positive' if x >= 0 else 'Negative' 
        )

    def identify_themes(self, num_topics=3):
        vectorizer = CountVectorizer(stop_words='english')  # Use 'english' for built-in stop words
        text_data = vectorizer.fit_transform(self.data[self.text_column])
        
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

    def perform_analysis(self):
        try:
            if np.issubdtype(self.data[self.text_column].dtype, np.number):
                # Perform numeric analysis if column is of numeric type
                self.visualizer.plot_numeric_analysis(self.data, self.text_column)
            else:
                # Perform sentiment analysis if column is of string type
                self.analyze_sentiments()

                # Visualize sentiment distribution and comments
                self.visualizer.plot_sentiment_pie_chart(self.data)

                # Display comments by sentiment
                sentiment_choice = st.radio("Select sentiment to view comments:", ['Positive', 'Negative', 'Neutral'])
                self.visualizer.display_comments_by_sentiment(self.data, sentiment_choice, self.text_column)

                # Word Cloud
                st.subheader("Word Cloud")
                self.visualizer.generate_wordcloud(self.data[self.text_column].dropna(), "Word Cloud of Comments")

                # Top Words
                st.subheader("Top Words")
                self.visualizer.plot_top_words(self.data, self.text_column)

                # Top Bigrams
                st.subheader("Top Bigrams")
                self.visualizer.plot_top_bigrams(self.data, self.text_column)

                # Thematic identification
                st.subheader("Thematic Identification")
                themes = self.identify_themes()
                for i, theme in enumerate(themes):
                    st.write(f"**Theme {i + 1}:** {theme}")

        except Exception as e:
            st.error(f"Error performing sentiment analysis: {e}")


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

