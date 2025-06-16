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
    scores = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()
    labels = ["negative", "neutral", "positive"]

    results = []
    for i, headline in enumerate(headlines):
        label_scores = dict(zip(labels, scores[i]))
        results.append({
            "headline": headline,
            "negative": round(label_scores["negative"], 2),
            "neutral": round(label_scores["neutral"], 2),
            "positive": round(label_scores["positive"], 2),
        })

    # Aggregate sentiment score (average positive - negative)
    avg_pos = sum(r["positive"] for r in results) / len(results)
    avg_neg = sum(r["negative"] for r in results) / len(results)
    polarity = avg_pos - avg_neg

    return results, polarity


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
        NEWS_API_KEY = "11c0eca5f0284ac79d05f6a14749dc65"  # ← Replace with your real key

        news_api_url = (
            f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
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
            finbert_results, polarity = get_finbert_sentiment(headlines)
            finbert_results = sorted(finbert_results, key=lambda x: x["positive"], reverse=True)

            if polarity > 0.2:
                sentiment = "Bullish 📈"
            elif polarity < -0.2:
                sentiment = "Bearish 📉"
            else:
                sentiment = "Neutral 🤔"

        st.metric("Sentiment", sentiment, f"{polarity:.2f}")

        with st.expander("ℹ️ How this sentiment score is calculated"):
            st.markdown(f"""
            We analyze the most recent headlines for **{ticker}** using [**FinBERT**](https://huggingface.co/ProsusAI/finbert), a language model trained on financial documents.

            **Process:**
            1. Fetch the last 5 headlines mentioning `{ticker}`
            2. Each headline is scored as:
                - 🟢 Positive
                - ⚪ Neutral
                - 🔴 Negative
            3. We compute:
            ```python
            Sentiment Score = Average(Positive) - Average(Negative)
            ```

            - **Bullish** if score > +0.2
            - **Bearish** if score < –0.2
            - **Neutral** otherwise
            """)

            st.markdown("**🔍 Raw FinBERT Scores by Headline:**")
            for row in finbert_results:
                st.markdown(f"""
                > *{row['headline']}*  
                🟢 Positive: `{row['positive']}`  
                ⚪ Neutral: `{row['neutral']}`  
                🔴 Negative: `{row['negative']}`  
                """)


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



# 📊 Multi-Ticker Screener Add-On for Mining Stocks (Senior & Junior)
# Includes debug and fallback handling for junior miners + improved NewsAPI queries

st.header("⛏️ Mining Sector Screener")

# Senior and Junior Miner Ticker Sets
senior_miners = {
    "AEM": "Agnico Eagle Mines",
    "NEM": "Newmont Corporation",
    "GOLD": "Barrick Gold",
    "FNV": "Franco-Nevada",
    "WPM": "Wheaton Precious Metals"
}

junior_miners = {
    "ROS.V": "Roscan Gold",
    "MOZ.TO": "Marathon Gold",
    "OSK.TO": "Osisko Mining",
    "AR.TO": "Argonaut Gold",
    "NXS.V": "Nexus Gold"
}

def get_metrics_for_ticker(ticker, name=None):
    try:
        st.write(f"🔎 Processing: {ticker} ({name})")
        data = yf.download(ticker, period="1mo", interval="1d", auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        if len(data) < 2:
            st.warning(f"⚠️ Not enough data for {ticker}")
            return None

        price_change = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
        avg_vol = data["Volume"].rolling(window=20).mean().iloc[-1]
        today_vol = data["Volume"].iloc[-1]
        vol_spike = today_vol / avg_vol if avg_vol and avg_vol > 0 else 1

        # 🆕 Use company name for better news search
        query = name or ticker
        query = f'"{query}"'  # add quotes to enforce phrase matching
        news_api_url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey=11c0eca5f0284ac79d05f6a14749dc65"
        news_data = requests.get(news_api_url).json()
        articles = news_data.get("articles", [])
        headlines = [a["title"] for a in articles if "title" in a]
        st.write("📰 Headlines:", headlines)

        if not headlines:
            st.warning(f"⚠️ No headlines found for {ticker}")
            results = []
            polarity = 0.0
        else:
            results, polarity = get_finbert_sentiment(headlines)

        signal = polarity > 0.3 and vol_spike > 1.5

        return {
            "Ticker": ticker,
            "Company": name or ticker,
            "Sentiment": round(polarity, 2),
            "Price Change (%)": round(price_change, 2),
            "Volume Spike (x)": round(vol_spike, 2),
            "Signal": "✅" if signal else "",
            "Headlines": results
        }
    except Exception as e:
        st.error(f"❌ Error with {ticker}: {e}")
        return None
    
    # Display Screener Tabs
tab1, tab2 = st.tabs(["Senior Miners", "Junior Miners"])

def display_screener(data):
    if data:
        df = pd.DataFrame([{k: v for k, v in d.items() if k != "Headlines"} for d in data])
        df = df.sort_values(by="Sentiment", ascending=False)
        st.dataframe(df, use_container_width=True)

        for d in data:
            with st.expander(f"🗞️ Headlines for {d['Company']} ({d['Ticker']})"):
                if not d["Headlines"]:
                    st.write("No headlines available.")
                for row in d["Headlines"]:
                    st.markdown(f"""
                    > *{row['headline']}*  
                    🟢 Positive: `{row['positive']}`  
                    ⚪ Neutral: `{row['neutral']}`  
                    🔴 Negative: `{row['negative']}`
                    """)
    else:
        st.info("No data available.")

with tab1:
    st.subheader("🟡 Senior Mining Companies")
    senior_data = []
    for ticker, name in senior_miners.items():
        row = get_metrics_for_ticker(ticker, name)
        if row:
            senior_data.append(row)
    display_screener(senior_data)

with tab2:
    st.subheader("⚒️ Junior Mining Companies")
    junior_data = []
    for ticker, name in junior_miners.items():
        row = get_metrics_for_ticker(ticker, name)
        if row:
            junior_data.append(row)
    display_screener(junior_data)

