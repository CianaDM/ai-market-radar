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
st.cache_data.clear()  # Clear Streamlit's cached charting data


st.title("📈 AI Market Radar")
st.caption("Get live price, volume, and AI-powered sentiment analysis.")

# --- Input ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA):", value="AAPL").upper()

# --- Main Block ---
if ticker:
    try:
        # --- Dates and Ticker Prep ---
        today = datetime.today().date()
        past = today - timedelta(days=30)
        tsx_ticker = ticker
        symbol_for_api = f"TSX:{ticker}" if ticker.endswith(".TO") else ticker
        finnhub_key = st.secrets["FINNHUB_API_KEY"] if "FINNHUB_API_KEY" in st.secrets else "your_fallback_key"
        newsapi_key = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else "your_newsapi_key"

        # === Price Data Helper ===
        @st.cache_data
        def get_price_data(ticker, start, end):
            data = yf.download(ticker, start=start, end=end)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            data.columns = [col.capitalize() for col in data.columns]
            return data

        # === Price Data Section ===
        data = get_price_data(ticker, past, today)

        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(set(data.columns)):
            st.error(f"Missing required price columns: {required_cols - set(data.columns)}")
            st.stop()

        data = data.dropna(subset=list(required_cols))

        st.subheader("📈 Candlestick Chart (30 Days)")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="green",
            decreasing_line_color="red",
            name="Price"
        ))

        fig.update_layout(
            height=600,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=False,
            margin=dict(t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)


        # --- Sentiment Analysis (VADER over NewsAPI headlines) ---
        st.subheader("🧠 Sentiment Summary (via VADER)")
        news_url = f"https://newsapi.org/v2/everything?q={ticker}&from={past}&sortBy=publishedAt&apiKey={newsapi_key}"
        res = requests.get(news_url)
        news_items = []

        if res.status_code == 200:
            news_items = res.json().get("articles", [])[:10]
        else:
            st.caption(f"NewsAPI Error: {res.status_code}")

        if news_items:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()

            sentiments = []
            for article in news_items:
                score = analyzer.polarity_scores(article['title'])["compound"]
                sentiments.append((score, article["title"]))

            avg_score = sum([s[0] for s in sentiments]) / len(sentiments)
            sentiment_label = "🟢 Bullish" if avg_score >= 0.25 else "🔴 Bearish" if avg_score <= -0.25 else "⚪ Neutral"
            st.markdown(f"**Average Sentiment:** `{avg_score:.2f}` → {sentiment_label}")
        else:
            st.caption("Sentiment data unavailable.")

        # --- Headlines in Expander ---
        with st.expander("📢 Latest Headlines"):
            if news_items:
                for article in news_items:
                    date = article["publishedAt"].split("T")[0]
                    st.markdown(f"**{article['title']}**")
                    st.caption(f"{date} — [Source]({article['url']})")
                    st.markdown("---")
            else:
                st.caption("No recent news available.")

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

