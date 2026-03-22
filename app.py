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
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Stock Intelligence Platform", layout="wide")
st.title("🚀 AI Stock Intelligence Platform")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Stock Analysis",
        "AI Direction Prediction",
        "LSTM Prediction",
        "AI Market Scanner",
        "Daily AI Radar",
        "Portfolio Analyzer",
        "News Intelligence"
    ]
)

# ---------------- CLEAN ----------------
def clean_close(data):
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    return pd.Series(close.values.flatten(), index=data.index)

# ---------------- INDICATORS ----------------
def add_indicators(data):
    close = clean_close(data)
    data["RSI"] = ta.momentum.RSIIndicator(close).rsi()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()
    data["Return"] = close.pct_change()
    data["Volatility"] = data["Return"].rolling(20).std()
    return data

# ---------------- TECH SCORE ----------------
def technical_score(data):
    close = clean_close(data)
    price = float(close.iloc[-1])
    ma50 = float(data["MA50"].iloc[-1])
    ma200 = float(data["MA200"].iloc[-1])
    rsi = float(data["RSI"].iloc[-1])
    volatility = float(data["Volatility"].iloc[-1])

    lookback = min(120, len(close)-1)
    growth = (price - close.iloc[-lookback]) / close.iloc[-lookback]

    score = 0
    if price > ma50: score += 2
    if ma50 > ma200: score += 2
    if rsi < 40: score += 1
    if volatility < 0.03: score += 1
    if growth > 0.20: score += 3

    return score

# ---------------- FUNDAMENTAL ----------------
def fundamental_score(symbol):
    try:
        info = yf.Ticker(symbol + ".NS").info
        pe = info.get("trailingPE",0)
        roe = info.get("returnOnEquity",0)
        debt = info.get("debtToEquity",0)

        score = 0
        if pe and pe < 25: score += 1
        if roe and roe > 0.15: score += 1
        if debt and debt < 100: score += 1

        return score
    except:
        return 0

# ---------------- NEWS ----------------
def get_news(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(symbol)}"
        feed = feedparser.parse(url)
        return [e.title for e in feed.entries[:10]]
    except:
        return []

def news_score(symbol):
    headlines = get_news(symbol)
    score = 0
    for h in headlines:
        p = TextBlob(h).sentiment.polarity
        if p > 0.2: score += 1
        elif p < -0.2: score -= 1
    return score

# ---------------- RANDOM FOREST ----------------
def predict_direction(symbol):
    data = yf.download(symbol + ".NS", period="2y")
    if data.empty or len(data) < 250:
        return "Not enough data"

    data = add_indicators(data)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    X = data[["RSI","MA50","MA200","Volatility"]]
    y = data["Target"]

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X,y)

    pred = model.predict(X.iloc[-1:])[0]
    return "UP" if pred==1 else "DOWN"

# ---------------- LSTM ----------------
def lstm_predict(symbol):
    data = yf.download(symbol + ".NS", period="5y")
    if data.empty or len(data) < 100:
        return None, None, None

    close = data["Close"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test)

    # Next day prediction
    last_seq = scaled[-60:]
    last_seq = np.reshape(last_seq,(1,60,1))
    next_pred = scaler.inverse_transform(model.predict(last_seq))

    return preds, y_test, next_pred

# ---------------- MARKET SCAN ----------------
def analyze_stock(symbol):
    try:
        data = yf.download(symbol + ".NS", period="1y")
        if len(data)<200: return None

        data = add_indicators(data)
        return {
            "Stock":symbol,
            "Price":round(clean_close(data).iloc[-1],2),
            "Score":technical_score(data)+fundamental_score(symbol)+news_score(symbol)
        }
    except:
        return None

def scan_market():
    stocks = pd.read_csv("nse_stocks.csv")["SYMBOL"].tolist()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(analyze_stock,stocks))
    return pd.DataFrame([r for r in results if r]).sort_values("Score",ascending=False)

# ---------------- PORTFOLIO ----------------
def portfolio_forecast(annual_return, investment, years=5):
    return investment * (1 + annual_return) ** years

# ---------------- PAGES ----------------
if menu == "Stock Analysis":
    ticker = st.text_input("Enter ticker")
    if ticker:
        data = yf.download(ticker + ".NS")
        st.line_chart(data["Close"])

elif menu == "AI Direction Prediction":
    ticker = st.text_input("Enter ticker")
    if ticker:
        st.write(predict_direction(ticker))

elif menu == "LSTM Prediction":
    ticker = st.text_input("Enter ticker")
    if ticker:
        preds, actual, next_pred = lstm_predict(ticker)
        if preds is not None:
            df = pd.DataFrame({
                "Actual": actual.flatten(),
                "Predicted": preds.flatten()
            })
            st.line_chart(df)
            st.success(f"Next Day Prediction: ₹{round(next_pred[0][0],2)}")

elif menu == "AI Market Scanner":
    if st.button("Scan"):
        st.dataframe(scan_market().head(50))

elif menu == "Daily AI Radar":
    if st.button("Generate"):
        st.dataframe(scan_market().head(5))

elif menu == "Portfolio Analyzer":

    tickers = st.text_input("Enter tickers separated by comma")

    if tickers:
        tickers = [t.strip() for t in tickers.split(",")]

        prices = yf.download(tickers, period="5y")["Close"]
        prices = prices.dropna(axis=1, how="all").dropna()

        returns = prices.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        weights = np.ones(len(returns.columns)) / len(returns.columns)
        portfolio_returns = returns.dot(weights)

        annual_return = portfolio_returns.mean() * 252
        risk = portfolio_returns.std() * np.sqrt(252)

        st.success(f"Expected Return: {round(annual_return*100,2)}%")
        st.warning(f"Risk: {round(risk*100,2)}%")

        # ---------- 2030 FORECAST ----------
        investment = st.number_input("Enter current investment (₹)", value=10000)
        future = portfolio_forecast(annual_return, investment, years=5)

        st.subheader("📈 Portfolio Value in 2030")
        st.success(f"Projected Value: ₹{round(future,2)}")

        # ---------- NIFTY ----------
        nifty = yf.download("^NSEI", period="5y")["Close"]
        nifty_returns = nifty.pct_change().dropna()

        df = pd.DataFrame({
            "Portfolio": (1+portfolio_returns).cumprod(),
            "NIFTY": (1+nifty_returns).cumprod()
        })

        st.line_chart(df)

        # ---------- OPTIMIZATION ----------
        def neg_sharpe(w):
            r = np.sum(returns.mean()*w)*252
            std = np.sqrt(np.dot(w.T,np.dot(returns.cov()*252,w)))
            return -(r/std)

        n = len(returns.columns)
        opt = minimize(neg_sharpe, np.ones(n)/n,
                       bounds=[(0,1)]*n,
                       constraints={'type':'eq','fun':lambda w: np.sum(w)-1})

        st.subheader("Optimal Weights")
        st.write(pd.Series(opt.x, index=returns.columns))

        # ---------- HEATMAP ----------
        fig, ax = plt.subplots()
        sns.heatmap(returns.corr(), ax=ax)
        st.pyplot(fig)

elif menu == "News Intelligence":
    company = st.text_input("Enter stock")
    if company:
        for h in get_news(company):
            st.write(h)
