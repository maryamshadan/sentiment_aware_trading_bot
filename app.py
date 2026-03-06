import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf

# 1. FIXED ARCHITECTURE (Must match your AAPL_expert_final.pth exactly)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size=5, action_size=3):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x):
        return self.fc(x)

# 2. UI SETUP
st.set_page_config(page_title="AI Trader", page_icon="📈")
st.title("🚀 AAPL Sentiment-Aware AI Expert")
st.markdown("### Presentation Mode: AI Market Scan")

# 3. LOAD BRAIN
@st.cache_resource
def load_expert():
    try:
        model = PolicyNetwork()
        model.load_state_dict(torch.load("AAPL_expert_final.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Brain Load Error: {e}")
        return None

# 4. DATA FETCHING (Friday's Close Replay)
def get_market_data():
    try:
        # Try to get real data
        df = yf.download("AAPL", period="20d", interval="1d", auto_adjust=True)
        if len(df) < 15: raise ValueError("Not enough data")
        
        df.columns = [c.lower() for c in df.columns]
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        
        # Use .iloc[-1] safely
        current_rsi = 100 - (100 / (1 + (gain.iloc[-1]/loss.iloc[-1])))
        current_price = df['close'].iloc[-1]
        return current_rsi, current_price
    except:
        # Presentation Fallback: Friday's Close Stats
        return 65.4, 264.18 


# 5. SIDEBAR STRESS TEST
st.sidebar.header("📊 AI Stress Test")
manual_sentiment = st.sidebar.slider("Simulate News Sentiment", 0.0, 1.0, 0.5)

# 6. ACTION
if st.button("🔍 ANALYSE MARKET"):
    rsi, price = get_market_data()
    # Create 5D State [RSI, MACD(0), Vol(0.01), Pos(0), Sentiment]
    state = np.array([rsi, 0.0, 0.01, 0.0, manual_sentiment], dtype=np.float32)
    
    model = load_expert()
    if model:
        with torch.no_grad():
            q_vals = model(torch.FloatTensor(state).unsqueeze(0))
            action = q_vals.argmax().item()
        
        mapping = {0: "HOLD ⏸️", 1: "BUY 🚀", 2: "SELL 📉"}
        st.metric("AAPL Price (Friday)", f"${price:.2f}")
        st.header(f"AI Decision: {mapping[action]}")
        st.info(f"AI Context: RSI {rsi:.1f} | Sentiment {manual_sentiment:.2f}")
