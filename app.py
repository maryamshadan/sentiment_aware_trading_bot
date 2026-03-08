import streamlit as st
import torch
import yfinance as yf
import pandas_ta as ta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from alpaca.trading.client import TradingClient
from model_architecture import PolicyNetwork # Your DDQN structure

# --- 1. MEMORY-EFFICIENT ASSET LOADING ---
@st.cache_resource
def load_all_models():
    # Load FinBERT (Sentiment)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model_sent = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Load DDQN (The Brain)
    brain = PolicyNetwork()
    brain.load_state_dict(torch.load("AAPL_expert_final.pth", map_location="cpu"))
    brain.eval()
    
    return tokenizer, model_sent, brain

tokenizer, model_sent, brain = load_all_models()

# --- 2. THE REAL-TIME SENTIMENT ENGINE ---
def get_sentiment(ticker):
    # In a full app, use NewsAPI or Alpaca News here
    # For now, we simulate the live feed fetch
    headlines = [f"{ticker} shows strong quarterly growth", f"Analysts upgrade {ticker} to Buy"]
    
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sent(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Score = Positive - Negative (Result between -1 and 1)
    sentiment_score = (probs[:, 0] - probs[:, 1]).mean().item()
    return sentiment_score

# --- 3. THE END-TO-END EXECUTION ---
st.title("🍏 AAPL End-to-End Trading Bot")

if st.button("RUN LIVE STRATEGY"):
    # A. Fetch Technicals
    data = yf.download("AAPL", period="60d", interval="1d")
    data.columns = [c.lower() for c in data.columns]
    data['rsi'] = ta.rsi(data['close'], length=14)
    # ... Add MACD and Volatility logic here ...
    
    # B. Fetch Live Sentiment
    score = get_sentiment("AAPL")
    
    # C. Prepare 5D State [RSI, MACD, Vol, Pos, Sentiment]
    last_row = data.dropna().iloc[-1]
    state = torch.FloatTensor([last_row['rsi'], 0.0, 0.01, 0.0, score]).unsqueeze(0)
    
    # D. Brain Decision
    with torch.no_grad():
        action = brain(state).argmax().item()
    
    # E. Display & Trade
    st.write(f"Current RSI: {last_row['rsi']:.2f} | FinBERT Score: {score:.2f}")
    if action == 1: st.success("ACTION: BUY 🚀")
    elif action == 2: st.error("ACTION: SELL 📉")
    else: st.info("ACTION: HOLD ⏸️")
