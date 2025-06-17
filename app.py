import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

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

        # --- Sentiment ---
        st.subheader("🧠 Sentiment Summary")
        sentiment_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={symbol_for_api}&token={finnhub_key}"
        response = requests.get(sentiment_url)

        if response.status_code == 200:
            sentiment_data = response.json()
            sentiment_section = sentiment_data.get("sentiment", {})
            bullish = sentiment_section.get("bullishPercent")
            bearish = sentiment_section.get("bearishPercent")

            if bullish is not None and bearish is not None:
                neutral = 100 - bullish - bearish
                st.markdown(f"""
                - 🟢 **Bullish**: {bullish:.0f}%
                - 🔴 **Bearish**: {bearish:.0f}%
                - ⚪ **Neutral**: {neutral:.0f}%
                """)
            else:
                st.caption("Sentiment data unavailable.")
        else:
            st.caption("Sentiment data unavailable.")

        # --- Expandable Headlines ---
        with st.expander("📢 Latest Headlines"):
            news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol_for_api}&from={past}&to={today}&token={finnhub_key}"
            res = requests.get(news_url)
            if res.status_code == 200:
                news_items = res.json()[:5]
                if news_items:
                    for article in news_items:
                        dt = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
                        st.markdown(f"**{article['headline']}**")
                        st.caption(f"{dt} — [Source]({article['url']})")
                        st.markdown("---")
                else:
                    st.info("No recent headlines found.")
            else:
                st.warning(f"Finnhub API error: {res.status_code}")

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
