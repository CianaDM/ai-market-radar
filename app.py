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
        today = datetime.date.today()
        past = today - datetime.timedelta(days=30)
        tsx_ticker = f"TSX:{ticker}" if ":" not in ticker else ticker

        data = yf.download(ticker, start=past, end=today)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.columns = [col.capitalize() for col in data.columns]

        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(data.columns)):
            st.error(f"Missing required price columns: {required_cols - set(data.columns)}")
            st.stop()

        data = data.dropna(subset=list(required_cols))

        st.subheader("Candlestick Chart (30 Days)")

        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )])

        fig.update_layout(
            height=700,
            template="plotly_white",
            showlegend=False,
            margin=dict(t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # === Canadian News via Finnhub ===
        # === Finnhub News Sentiment ===
        st.subheader("🧠 Sentiment Summary")

        sentiment_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={tsx_ticker}&token={d18t4qpr01qkcat4n8e0d18t4qpr01qkcat4n8eg}"
        sentiment_res = requests.get(sentiment_url)
        if sentiment_res.status_code == 200:
            sentiment_data = sentiment_res.json()
            bullish = sentiment_data.get("sentiment", {}).get("bullishPercent", 0)
            bearish = sentiment_data.get("sentiment", {}).get("bearishPercent", 0)
            neutral = 100 - bullish - bearish
            st.markdown(f"""
            - 🟢 **Bullish**: {bullish:.0f}%
            - 🔴 **Bearish**: {bearish:.0f}%
            - ⚪ **Neutral**: {neutral:.0f}%
            """)
        else:
            st.caption("Sentiment data unavailable.")


        st.subheader("📢 Latest Canadian Headlines")
        try:
            finnhub_key = st.secrets["FINNHUB_API_KEY"] if "FINNHUB_API_KEY" in st.secrets else "Yd18t4qpr01qkcat4n8e0d18t4qpr01qkcat4n8eg"
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={tsx_ticker}&from={past}&to={today}&token={finnhub_key}"
            res = requests.get(news_url)
            if res.status_code == 200:
                news_items = res.json()[:5]
                if news_items:
                    for article in news_items:
                        dt = datetime.datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
                        st.markdown(f"**{article['headline']}**  ")
                        st.caption(f"{dt} — [Source]({article['url']})")
                        st.markdown("---")
                else:
                    st.warning("No recent Canadian headlines found.")
            else:
                st.warning(f"Finnhub API error: {res.status_code}")
        except Exception as e:
            st.warning(f"Error fetching news: {e}")

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

