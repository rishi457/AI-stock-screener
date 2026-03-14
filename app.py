import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import feedparser
from urllib.parse import quote_plus
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Stock Intelligence Platform", layout="wide")

st.title("AI Stock Intelligence Platform")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Stock Analysis",
        "AI Direction Prediction",
        "AI Market Scanner",
        "Daily AI Radar",
        "Portfolio Analyzer",
        "News Intelligence",
        "AI Chatbot"
    ]
)

# -----------------------------
# Clean Close Series
# -----------------------------

def clean_close(data):

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    return pd.Series(close.values.flatten(), index=data.index)

# -----------------------------
# Indicators
# -----------------------------

def add_indicators(data):

    close = clean_close(data)

    data["RSI"] = ta.momentum.RSIIndicator(close).rsi()

    data["MA50"] = close.rolling(50).mean()

    data["MA200"] = close.rolling(200).mean()

    data["Return"] = close.pct_change()

    data["Volatility"] = data["Return"].rolling(20).std()

    return data

# -----------------------------
# Technical Score
# -----------------------------

def technical_score(data):

    close = clean_close(data)

    price = float(close.iloc[-1])

    ma50 = float(data["MA50"].iloc[-1])

    ma200 = float(data["MA200"].iloc[-1])

    rsi = float(data["RSI"].iloc[-1])

    volatility = float(data["Volatility"].iloc[-1])

    growth = (price - close.iloc[-120]) / close.iloc[-120]

    score = 0

    if price > ma50:
        score += 2

    if ma50 > ma200:
        score += 2

    if rsi < 40:
        score += 1

    if volatility < 0.03:
        score += 1

    if growth > 0.20:
        score += 3

    return score

# -----------------------------
# Fundamental Score
# -----------------------------

def fundamental_score(symbol):

    try:

        ticker = yf.Ticker(symbol + ".NS")

        info = ticker.info

        pe = info.get("trailingPE",0)

        roe = info.get("returnOnEquity",0)

        debt = info.get("debtToEquity",0)

        score = 0

        if pe and pe < 25:
            score += 1

        if roe and roe > 0.15:
            score += 1

        if debt and debt < 100:
            score += 1

        return score

    except:

        return 0

# -----------------------------
# Google News Fetch
# -----------------------------

def get_news(symbol):

    try:

        query = quote_plus(symbol + " stock India")

        url = f"https://news.google.com/rss/search?q={query}"

        feed = feedparser.parse(url)

        headlines = []

        for entry in feed.entries[:10]:

            headlines.append(entry.title)

        return headlines

    except:

        return []

# -----------------------------
# News Sentiment
# -----------------------------

def news_score(symbol):

    headlines = get_news(symbol)

    score = 0

    for h in headlines:

        polarity = TextBlob(h).sentiment.polarity

        if polarity > 0.2:
            score += 1

        elif polarity < -0.2:
            score -= 1

    return score

# -----------------------------
# News Summary
# -----------------------------

def summarize_news(symbol):

    headlines = get_news(symbol)

    positive = 0
    negative = 0

    for h in headlines:

        polarity = TextBlob(h).sentiment.polarity

        if polarity > 0:
            positive += 1
        elif polarity < 0:
            negative += 1

    if positive > negative:

        return "Overall news sentiment is positive."

    elif negative > positive:

        return "Recent news sentiment is negative."

    else:

        return "News sentiment is neutral."

# -----------------------------
# AI Direction Prediction
# -----------------------------

def predict_direction(symbol):

    try:

        data = yf.download(symbol + ".NS", period="2y")

        if data.empty or len(data) < 250:

            return "Not enough data"

        data = add_indicators(data)

        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        data = data.dropna()

        if len(data) < 50:

            return "Not enough data"

        X = data[["RSI","MA50","MA200","Volatility"]]

        y = data["Target"]

        model = RandomForestClassifier(n_estimators=200)

        model.fit(X,y)

        latest = X.iloc[-1:]

        prediction = model.predict(latest)[0]

        if prediction == 1:
            return "UP"
        else:
            return "DOWN"

    except:

        return "Prediction failed"

# -----------------------------
# Analyze Stock
# -----------------------------

def analyze_stock(symbol):

    try:

        data = yf.download(symbol + ".NS", period="1y")

        if len(data) < 200:

            return None

        data = add_indicators(data)

        close = clean_close(data)

        price = float(close.iloc[-1])

        tech = technical_score(data)

        fund = fundamental_score(symbol)

        news = news_score(symbol)

        final = tech + fund + news

        return {
            "Stock":symbol,
            "Price":round(price,2),
            "Tech Score":tech,
            "Fund Score":fund,
            "News Score":news,
            "Final Score":final
        }

    except:

        return None

# -----------------------------
# Market Scanner
# -----------------------------

def scan_market():

    stocks = pd.read_csv("nse_stocks.csv")

    symbols = stocks["SYMBOL"].tolist()

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        data = list(executor.map(analyze_stock,symbols))

    for r in data:

        if r:
            results.append(r)

    df = pd.DataFrame(results)

    df = df.sort_values("Final Score",ascending=False)

    return df

# -----------------------------
# Pages
# -----------------------------

if menu == "Stock Analysis":

    ticker = st.text_input("Enter ticker (Example RELIANCE)")

    if ticker:

        data = yf.download(ticker + ".NS", start="2020-01-01")

        if data.empty:

            st.error("Stock not found")

        else:

            close = clean_close(data)

            st.line_chart(close)

            data = add_indicators(data)

            score = technical_score(data)

            st.write("Technical Score:", score)

elif menu == "AI Direction Prediction":

    ticker = st.text_input("Enter ticker")

    if ticker:

        direction = predict_direction(ticker)

        if direction == "UP":

            st.success("AI Prediction: Stock may go UP tomorrow")

        elif direction == "DOWN":

            st.error("AI Prediction: Stock may go DOWN tomorrow")

        else:

            st.warning(direction)

elif menu == "AI Market Scanner":

    if st.button("Scan NSE Market"):

        df = scan_market()

        st.dataframe(df.head(100))

elif menu == "Daily AI Radar":

    if st.button("Generate Radar"):

        df = scan_market()

        st.subheader("Top 5 Stocks Today")

        st.dataframe(df.head(5))

elif menu == "Portfolio Analyzer":

    tickers = st.text_input("Enter tickers separated by comma")

    if tickers:

        tickers = tickers.split(",")

        prices = yf.download(tickers,start="2022-01-01")["Close"]

        returns = prices.pct_change().dropna()

        annual_return = returns.mean().mean()*252

        risk = returns.std().mean()*np.sqrt(252)

        st.write("Expected Return:",round(annual_return*100,2),"%")

        st.write("Risk:",round(risk*100,2),"%")

elif menu == "News Intelligence":

    company = st.text_input("Enter stock")

    if company:

        headlines = get_news(company)

        for h in headlines:

            sentiment = TextBlob(h).sentiment.polarity

            if sentiment > 0:
                st.success(h)

            elif sentiment < 0:
                st.error(h)

            else:
                st.write(h)

        st.subheader("AI News Summary")

        st.info(summarize_news(company))

elif menu == "AI Chatbot":

    q = st.text_input("Ask investing question")

    if q:

        q = q.lower()

        if "buy" in q:

            st.write("Look for strong earnings growth and low debt.")

        elif "risk" in q:

            st.write("Diversify across sectors.")

        else:

            st.write("Analyze fundamentals and technical trends.")