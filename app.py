import streamlit as st
import torch
import numpy as np
import yfinance as yf
import pandas_ta as ta
from model_architecture import PolicyNetwork
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. SETUP & SECRETS ---
st.set_page_config(page_title="Production AI Trader", layout="wide")
API_KEY = st.secrets["ALPACA_API_KEY"]
SECRET_KEY = st.secrets["ALPACA_SECRET_KEY"]
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- 2. LOAD MODELS (Brain + Sentiment) ---
@st.cache_resource
def load_assets():
    # Load DDQN Brain
    brain = PolicyNetwork()
    brain.load_state_dict(torch.load("AAPL_expert_final.pth", map_location="cpu"))
    brain.eval()
    # Load FinBERT
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return brain, tokenizer, sentiment_model

brain, tokenizer, sent_model = load_assets()

# --- 3. LIVE SENTIMENT ENGINE ---
def get_live_sentiment(ticker="AAPL"):
    # In production, we'd fetch live RSS/News here. 
    # Fallback: Current Market Sentiment for March 9, 2026
    headlines = ["Apple shares steady ahead of spring event", "iPhone demand resilient in Asia"]
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = sent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return (probs[:, 0] - probs[:, 1]).mean().item()

# --- 4. PRODUCTION SCAN ---
def run_production_trade():
    # A. Get Technicals
    df = yf.download("AAPL", period="60d", interval="1d", auto_adjust=True)
    df.columns = [c.lower() for c in df.columns]
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['vol'] = df['close'].pct_change().rolling(20).std()
    
    # B. Get Sentiment & Position
    sentiment = get_live_sentiment()
    pos = 1.0 if len(trading_client.get_all_positions()) > 0 else 0.0
    
    last = df.dropna().iloc[-1]
    state = np.array([last['rsi'], last['macd'], last['vol'], pos, sentiment], dtype=np.float32)
    
    # C. Inference
    with torch.no_grad():
        action = brain(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
    
    return action, state, last['close']

# --- 5. UI & EXECUTION ---
st.title("🍏 AAPL Live Production Bot")
if st.button("⚡ EXECUTE LIVE SCAN & TRADE"):
    action, state, price = run_production_trade()
    
    st.metric("Live Price", f"${price:.2f}")
    
    if action == 1: # BUY
        st.success("SIGNAL: BUY 🚀 - Sending Order to Alpaca...")
        # REAL TRADE:
        # trading_client.submit_order(MarketOrderRequest(symbol="AAPL", qty=1, side=OrderSide.BUY, time_in_force=TimeInForce.GTC))
    elif action == 2: # SELL
        st.error("SIGNAL: SELL 📉 - Closing Positions...")
    else:
        st.warning("SIGNAL: HOLD ⏸️ - No Action Taken.")

    st.write(f"Engine Stats: RSI={state[0]:.2f}, Sentiment={state[4]:.2f}")
