import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Only download once
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="AI Market Radar", layout="wide")

# --- Input ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", value="AAPL")

if ticker:
    try:
        # --- Controls ---
        range_days = st.slider("Select date range (days):", min_value=5, max_value=90, value=30, step=5)
        today = datetime.date.today()
        past = today - datetime.timedelta(days=range_days)

        # --- Price Data ---
        data = yf.download(ticker, start=past, end=today)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.columns = [col.capitalize() for col in data.columns]
        data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        # --- Technicals ---
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        data["MA20"] = data["Close"].rolling(window=20).mean()

        # --- Current Stats ---
        st.subheader(f"{ticker} – Market Snapshot")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
        st.metric("Volume Today", f"{hist['Volume'].iloc[-1]:,}")
        st.metric("20D Avg Volume", f"{stock.info.get('averageVolume', 1):,}")

        # --- Chart ---
        st.subheader(f"Candlestick Chart with RSI & Volume ({range_days} Days)")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.6, 0.25, 0.15],
                            subplot_titles=("Price with 20D MA", "Volume", "RSI (14)"))

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="Price",
            increasing_line_color='green', decreasing_line_color='red'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="20D MA",
                                 line=dict(color='blue', width=1)), row=1, col=1)

        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume",
                             marker_color="lightgrey", opacity=0.5), row=2, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                                 line=dict(color="orange", width=1)), row=3, col=1)

        fig.update_layout(height=800, template="plotly_white", showlegend=True, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # --- News + Sentiment ---
        st.subheader("🧠 Sentiment Summary (via VADER)")
        newsapi_key = st.secrets["NEWS_API_KEY "] if "NEWS_API_KEY " in st.secrets else "your_api_key_here"
        news_url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&pageSize=10&sortBy=publishedAt&apiKey={newsapi_key}"

        response = requests.get(news_url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if articles:
                analyzer = SentimentIntensityAnalyzer()
                scores = []
                for article in articles:
                    headline = article.get("title", "")
                    score = analyzer.polarity_scores(headline)["compound"]
                    scores.append((score, headline))

                avg_score = sum(s[0] for s in scores) / len(scores)
                label = "🟢 Bullish" if avg_score >= 0.25 else "🔴 Bearish" if avg_score <= -0.25 else "⚪ Neutral"
                st.markdown(f"**Sentiment**: {label} (avg score: {avg_score:.2f})")

                with st.expander("📰 Latest Headlines"):
                    for _, headline in scores:
                        st.markdown(f"- {headline}")
            else:
                st.info("No recent news available.")
        else:
            st.error(f"NewsAPI Error: {response.status_code}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
