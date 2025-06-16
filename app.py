# app.py

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Streamlit Setup ---
st.set_page_config(page_title="AI Market Radar", layout="centered")
st.title("📈 AI Market Radar")
st.caption("Get live price, volume, and AI-powered sentiment analysis.")

# --- FinBERT Model Loading ---
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def get_finbert_sentiment(headlines):
    tokenizer, model = load_finbert()
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_scores = scores.mean(dim=0).tolist()
    labels = ["negative", "neutral", "positive"]
    return dict(zip(labels, sentiment_scores))

# --- Ticker Input ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA):", value="AAPL").upper()

if ticker:
    try:
        # --- Price Data ---
        today = datetime.now()
        past = today - timedelta(days=30)
        data = yf.download(ticker, start=past, end=today, auto_adjust=False)

        # 🔧 Flatten MultiIndex if present (fix for OHLC display)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # 🧪 Debug: Show raw data preview
        st.subheader("🧪 Raw Data Debug")
        st.write(f"Rows downloaded: {len(data)}")
        st.dataframe(data.tail())

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError("No price data available for today.")

        current_price = hist["Close"].iloc[-1]
        volume_today = hist["Volume"].iloc[-1]
        avg_volume = stock.info.get("averageVolume", 1)

        st.subheader(f"{ticker} – Market Snapshot")
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Volume Today", f"{volume_today:,}")
        st.metric("20D Avg Volume", f"{avg_volume:,}")

        # --- Interactive Candlestick Chart with Plotly ---
        st.subheader("Price Chart (Last 30 Days)")

        if not data.empty and all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color='green',
                decreasing_line_color='red',
                name="Candlestick"
            )])

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No OHLC price data available for charting.")

        # --- FinBERT Sentiment Analysis ---
        st.subheader("🧠 Sentiment Analysis")
        NEWS_API_KEY = "YOUR_NEWSAPI_KEY"  # ← Replace with your real key

        news_api_url = (
            f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=5&apiKey=11c0eca5f0284ac79d05f6a14749dc65"
        )

        try:
            news_data = requests.get(news_api_url).json()
            articles = news_data.get("articles", [])
            headlines = [a["title"] for a in articles if "title" in a]
        except Exception:
            headlines = []

        if not headlines:
            st.warning("No recent headlines found for sentiment analysis.")
            sentiment = "Neutral 🤔"
            polarity = 0.0
        else:
            sentiment_result = get_finbert_sentiment(headlines)
            polarity = sentiment_result["positive"] - sentiment_result["negative"]

            if polarity > 0.2:
                sentiment = "Bullish 📈"
            elif polarity < -0.2:
                sentiment = "Bearish 📉"
            else:
                sentiment = "Neutral 🤔"

        st.metric("Sentiment", sentiment, f"{polarity:.2f}")

        # --- AI Insight ---
        st.subheader("🔍 AI Insight")
        if len(data["Close"]) >= 2:
            price_change = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100
            price_change = round(float(price_change), 2)
        else:
            raise ValueError("Not enough historical price data for insight.")

        vol_spike = volume_today / avg_volume

        st.markdown(f"""
        - **Ticker**: `{ticker}`
        - **Sentiment**: `{sentiment}`
        - **Price Change (30d)**: `{price_change:.2f}%`
        - **Volume Spike**: `{vol_spike:.2f}x`

        **Insight:** {ticker} is showing a **{sentiment.lower()}** trend.
        {"Volume spike indicates momentum." if vol_spike > 1.5 else ""}
        """)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
