import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Candlestick Test", layout="wide")

# --- Input ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA):", value="AAPL").upper()

if ticker:
    try:
        # --- Date Range ---
        today = datetime.date.today()
        past = today - datetime.timedelta(days=30)

        # --- Download data ---
        data = yf.download(ticker, start=past, end=today)
        data = data.dropna(subset=["Open", "High", "Low", "Close"])

        st.subheader("📈 Candlestick Chart (30 Days)")

        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="blue",
            decreasing_line_color="orange"
        )])

        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(t=40, b=40),
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading chart: {e}")
