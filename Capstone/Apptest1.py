#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Load time series data
@st.cache_data
def load_time_series_data(uploaded_file, sheet_name: str) -> pd.DataFrame:
    data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    return data

# Load sentiment data
@st.cache_data
def load_sentiment_data(uploaded_file) -> pd.DataFrame:
    data = pd.read_excel(uploaded_file)
    return data

# Plot forecasts using Plotly
def plot_forecasts(train, test, forecasts, future_df, title: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Time'], y=train['Value'], name='Train', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=test['Time'], y=test['Value'], name='Test', line=dict(color='magenta')))
    fig.add_trace(go.Scatter(x=test['Time'], y=forecasts, name='Forecast', line=dict(color='lime')))
    if not future_df.empty:
        fig.add_trace(go.Scatter(x=future_df['Time'], y=future_df['Forecast'], name='Future Forecast', line=dict(color='yellow')))
    fig.update_layout(template="plotly_dark", font=dict(size=16, color='white'), title_text=title,
                      width=900, title_x=0.5, height=500, xaxis_title='Date',
                      yaxis_title='Value', legend_title_text='Legend')
    st.plotly_chart(fig)

# Sentiment analysis and visualization
def analyze_sentiments(df, text_column):
    # Convert the specified column to string type
    df[text_column] = df[text_column].astype(str)
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment scores
    df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_category'] = df['sentiment_scores'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    return df

def plot_sentiment_pie_chart(df):
    # Count the number of occurrences of each sentiment category
    sentiment_counts = df['sentiment_category'].value_counts()
    
    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, 
                                 values=sentiment_counts.values,
                                 hoverinfo='label+percent',
                                 textinfo='value+percent')])
    fig.update_traces(marker=dict(colors=['#00CC96', '#EF553B', '#636EFA']))
    fig.update_layout(title="Sentiment Distribution", template="plotly_dark")
    
    st.plotly_chart(fig)

def display_comments_by_sentiment(df, sentiment_type, text_column):
    filtered_df = df[df['sentiment_category'] == sentiment_type]
    st.subheader(f"{sentiment_type} Comments")
    if not filtered_df.empty:
        st.write(filtered_df[[text_column, 'sentiment_scores']].reset_index(drop=True))
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
    fig = px.histogram(df, x=numeric_column, nbins=20, title='Distribution of Ratings',
                       labels={numeric_column: 'Rating', 'count': 'Frequency'}, template="plotly_dark")
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
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
    st.sidebar.markdown("Select an analysis type from the tabs below:")

    # Create tabs for different analysis types
    analysis_tab, sentiment_tab = st.tabs(["📈 Time Series Analysis", "😊 Sentiment Analysis"])
    
    with analysis_tab:
        st.header("Time Series Analysis")
        st.info("Upload your dataset and select the sheet for analysis.", icon="ℹ️")

        # File uploader for time series analysis
        uploaded_file = st.file_uploader("Upload your Excel dataset for Time Series:", type=["xlsx"])
        
        if uploaded_file:
            # Input sheet name from the user
            sheet_name = st.text_input("Enter the sheet name for Time Series:", "")
            
            if sheet_name:
                data = load_time_series_data(uploaded_file, sheet_name)
                
                # Select columns for analysis
                time_column = st.selectbox("Select the Time column:", data.columns)
                value_column = st.selectbox("Select the Value column:", data.columns)

                # Convert time column to datetime
                data['Time'] = pd.to_datetime(data[time_column])
                data['Value'] = data[value_column]
                
                st.subheader("Original Time Series Data")
                st.write(data.head())
                
                # Decompose the time series using STL
                stl = STL(data['Value'], seasonal=13, period=12)
                result = stl.fit()
                data['Trend'] = result.trend
                data['Seasonal'] = result.seasonal
                data['Residual'] = result.resid
                
                # Split train and test for the residual component
                train_residual = data.iloc[:-int(len(data) * 0.2)]
                test_residual = data.iloc[-int(len(data) * 0.2):]
                
                # Build ARIMA model on the residual component
                model = ARIMA(train_residual['Residual'], order=(10, 1, 10), seasonal_order=(1, 1, 1, 12)).fit()
                yeojohnson_forecasts = model.forecast(len(test_residual))
                
                # Combine forecasts with trend and seasonal components
                trend_forecast = data['Trend'].iloc[-len(test_residual):].values
                seasonal_forecast = data['Seasonal'].iloc[-len(test_residual):].values
                final_forecasts = trend_forecast + seasonal_forecast + yeojohnson_forecasts
                
                # Plot the final forecasts
                st.subheader("Hybrid STL + SARIMA Forecast")
                plot_forecasts(train_residual, test_residual, final_forecasts, pd.DataFrame(), 'Hybrid STL + SARIMA Forecast')
                
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
                last_date = data['Time'].max()
                future_dates = pd.date_range(last_date, periods=months_to_forecast + 1, freq='M')[1:]
                
                future_trend = data['Trend'].iloc[-12:].values.reshape(-1, 1).repeat(months_to_forecast, axis=1).flatten()[:months_to_forecast]
                future_seasonal = data['Seasonal'].iloc[-12:].values[:months_to_forecast]
                
                future_residual = model.forecast(months_to_forecast)
                
                future_forecast = future_trend + future_seasonal + future_residual
                
                future_df = pd.DataFrame({'Time': future_dates, 'Forecast': future_forecast})
                
                # Plot future forecasts
                st.subheader("Future Forecast")
                plot_forecasts(train_residual, test_residual, final_forecasts, future_df, 'Future Forecast')

    with sentiment_tab:
        st.header("Sentiment Analysis")
        st.info("Upload your dataset and select the column for sentiment analysis.", icon="ℹ️")

        # File uploader for sentiment analysis
        uploaded_sentiment_file = st.file_uploader("Upload your Excel dataset for Sentiment Analysis:", type=["xlsx"])
        
        if uploaded_sentiment_file:
            df = load_sentiment_data(uploaded_sentiment_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Select a column for analysis
            column_options = df.columns.tolist()
            selected_column = st.selectbox("Select a column for analysis:", column_options)

            if selected_column:
                if np.issubdtype(df[selected_column].dtype, np.number):
                    # Perform numeric analysis if column is of numeric type
                    plot_numeric_analysis(df, selected_column)
                else:
                    # Perform sentiment analysis if column is of string type
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
