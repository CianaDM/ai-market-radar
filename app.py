import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")

st.set_page_config(page_title="AI Market Radar", layout="centered")

st.title("📈 AI Market Radar")
st.caption("Get live price, volume, and AI-powered sentiment analysis.")

# --- Input ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA):", value="AAPL").upper()

# --- Main Block ---
if ticker:
    try:
        today = datetime.today().date()
        past = today - timedelta(days=30)
        tsx_ticker = ticker
        symbol_for_api = f"TSX:{ticker}" if ticker.endswith(".TO") else ticker
        finnhub_key = st.secrets["FINNHUB_API_KEY"] if "FINNHUB_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"

        # --- Price Data ---
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

        # === Sentiment Summary using VADER ===
st.subheader("🧠 Sentiment Summary (via VADER)")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    if news_items:
        sentiments = []
        for article in news_items:
            score = analyzer.polarity_scores(article['headline'])["compound"]
            sentiments.append((score, article["headline"]))

        avg_score = sum([s[0] for s in sentiments]) / len(sentiments)

        if avg_score >= 0.25:
            sentiment_label = "🟢 Bullish"
        elif avg_score <= -0.25:
            sentiment_label = "🔴 Bearish"
        else:
            sentiment_label = "⚪ Neutral"

        st.markdown(f"**Sentiment:** {sentiment_label}  \n"
                    f"**Average Score:** {avg_score:+.2f}  \n"
                    f"**Analyzed Headlines:** {len(sentiments)}")

        # show top positive and negative headlines
        top_pos = max(sentiments, key=lambda x: x[0])
        top_neg = min(sentiments, key=lambda x: x[0])

        with st.expander("💬 Strongest Headlines"):
            st.markdown(f"**Most Positive:** _{top_pos[1]}_  (`{top_pos[0]:+.2f}`)")
            st.markdown(f"**Most Negative:** _{top_neg[1]}_  (`{top_neg[0]:+.2f}`)")

    else:
        st.caption("Sentiment data unavailable.")

except Exception as e:
    st.error(f"VADER Sentiment error: {e}")
