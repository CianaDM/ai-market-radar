# app.py

import streamlit as st
import yfinance as yf
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import requests
import plotly.graph_objects as go
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="AI Market Radar", layout="wide")

# ===============================
# Load FinBERT Once
# ===============================
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

# ===============================
# FinBERT Sentiment Function
# ===============================
def get_finbert_sentiment(headlines):
    tokenizer, model = load_finbert()
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = F.softmax(outputs.logits, dim=1).tolist()
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

    avg_pos = sum(r["positive"] for r in results) / len(results)
    avg_neg = sum(r["negative"] for r in results) / len(results)
    polarity = avg_pos - avg_neg

    return results, polarity

# ===============================
# UI – Single Stock Section
# ===============================
st.title("📈 AI Market Radar")
st.caption("Get live price, volume, and AI-powered sentiment analysis.")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA):", value="AAPL").upper()

if ticker:
    # Slider placeholder for bottom positioning
    slider_slot = st.empty()
    range_days = 30

    try:
        # Use slider inside try block to reactively update chart
        :", min_value=5, max_value=180, value=30, step=5)

        today = datetime.date.today()
        past = today - datetime.timedelta(days=range_days)
        data = yf.download(ticker, start=past, end=today)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.columns = [col.capitalize() for col in data.columns]

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(data.columns)):
            st.error(f"Missing required price columns: {required_cols - set(data.columns)}")
            st.stop()

        data = data.dropna(subset=list(required_cols))

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        current_price = hist["Close"].iloc[-1]
        volume_today = hist["Volume"].iloc[-1]
        avg_volume = stock.info.get("averageVolume", 1)

        st.subheader(f"{ticker} – Market Snapshot")
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Volume Today", f"{volume_today:,}")
        st.metric("20D Avg Volume", f"{avg_volume:,}")

        # === Indicators ===
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        data["MA20"] = data["Close"].rolling(window=20).mean()

        st.subheader(f"Candlestick Chart with RSI & Volume ({range_days} Days)")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.65, 0.25, 0.1],
            subplot_titles=("Price with 20D MA", "Volume", "RSI (14)")
        )

        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["MA20"],
            name="20D MA",
            mode="lines",
            line=dict(color='blue', width=1)
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color="lightgrey",
            opacity=0.5
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            mode="lines",
            line=dict(color="orange", width=1)
        ), row=3, col=1)

        fig.update_layout(
            height=950,
            template="plotly_white",
            showlegend=True,
            margin=dict(t=40, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Move slider to render AFTER chart
        range_days = slider_slot.slider("Select date range (days):", min_value=5, max_value=180, value=range_days, step=5):", min_value=5, max_value=180, value=range_days, step=5)

    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")



st.subheader("🧠 Sentiment Analysis")

# === Fetch Headlines ===
news_api_url = f"https://newsapi.org/v2/everything?q=\"{ticker}\"&language=en&sortBy=publishedAt&pageSize=5&apiKey=11c0eca5f0284ac79d05f6a14749dc65"
news_data = requests.get(news_api_url).json()
articles = news_data.get("articles", [])
headlines = [a["title"] for a in articles if "title" in a]

if not headlines:
    st.warning("No recent headlines found for sentiment analysis.")
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

    st.subheader("🔍 AI Insight")
    st.markdown(f"""
    - **Ticker**: `{ticker}`
    - **Sentiment**: `{sentiment}`
    - **Price Change (30d)**: `{(data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100:.2f}%`
    - **Volume Spike**: `{volume_today / avg_volume:.2f}x`

    **Insight:** {ticker} is showing a **{sentiment.lower()}** trend.{" Volume spike indicates momentum." if volume_today > 1.5 * avg_volume else ""}
    """)


# ===============================
# Sector Screener – Mining Stocks
# ===============================
st.header("⛏️ Mining Sector Screener")

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
            return None

        price_change = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
        avg_vol = data["Volume"].rolling(window=20).mean().iloc[-1]
        today_vol = data["Volume"].iloc[-1]
        vol_spike = today_vol / avg_vol if avg_vol and avg_vol > 0 else 1

        query = f'"{name or ticker}"'
        news_api_url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey=11c0eca5f0284ac79d05f6a14749dc65"
        time.sleep(1.1)
        news_data = requests.get(news_api_url).json()
        articles = news_data.get("articles", [])
        headlines = [a["title"] for a in articles if "title" in a]

        if not headlines:
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

tab1, tab2 = st.tabs(["Senior Miners", "Junior Miners"])

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
