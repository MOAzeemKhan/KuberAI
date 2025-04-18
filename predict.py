import pandas as pd
import numpy as np
import random
from stable_baselines3 import SAC
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import itertools
import os

# --- Hardcoded Parameters (to be later linked to frontend) ---
INVESTMENT_AMOUNT = 50000  # ₹ 50,000
RISK_LEVEL = 'High'      # 'Low', 'Medium', 'High'
HORIZON = 'Short Term'      # 'Short Term', 'Long Term'

TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2024-04-01'

# --- Load Trained Tickers ---
with open('model/trained_tickers.txt', 'r') as f:
    SYMBOLS = [line.strip() for line in f.readlines()]

def get_trade_env():
    df_raw = YahooDownloader(start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=SYMBOLS).fetch_data()
    df_raw = df_raw.drop_duplicates(subset=["date", "tic"], keep="first")

    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=True, use_turbulence=True)
    df_processed = fe.preprocess_data(df_raw)

    list_ticker = df_processed["tic"].unique().tolist()
    list_date = list(pd.date_range(df_processed['date'].min(), df_processed['date'].max()).astype(str))
    combo = list(itertools.product(list_date, list_ticker))

    df_full = pd.DataFrame(combo, columns=["date", "tic"]).merge(df_processed, on=["date", "tic"], how="left")
    df_full = df_full[df_full['date'].isin(df_processed['date'])]
    df_full = df_full.sort_values(["date", "tic"]).fillna(0)

    trade = data_split(df_full, TRADE_START_DATE, TRADE_END_DATE)

    stock_dim = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dim + len(INDICATORS) * stock_dim

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4
    }

    env_trade = StockTradingEnv(df=trade, **env_kwargs)
    return env_trade, trade

# ================= EXPLAINABILITY ===================

def generate_explanation(top5_df):
    """Generate per-stock and portfolio-level explanation based on selections."""
    
    # === Step 1: Mapping sectors manually ===
    sector_mapping = {
        'aapl': 'Technology', 'msft': 'Technology', 'meta': 'Technology', 'ibm': 'Technology',
        'hd': 'Consumer Discretionary', 'cat': 'Industrials', 'amzn': 'Consumer Discretionary',
        'intc': 'Technology', 't': 'Telecommunications', 'v': 'Financial Services',
        'gs': 'Financial Services', 'tsla': 'Automotive', 'nvda': 'Technology',
        'goog': 'Technology', 'googl': 'Technology', 'brk.b': 'Financial Services',
        'jpm': 'Financial Services', 'unh': 'Healthcare', 'vzb': 'Telecommunications',
        'xom': 'Energy', 'wmt': 'Consumer Staples', 'cvx': 'Energy', 'pg': 'Consumer Staples',
        'ma': 'Financial Services', 'dis': 'Communication Services', 'ko': 'Consumer Staples',
        'pepsico': 'Consumer Staples', 'nke': 'Consumer Discretionary', 'mrk': 'Healthcare',
        'pfe': 'Healthcare', 'abbv': 'Healthcare', 'cost': 'Consumer Staples', 'csco': 'Technology',
        'adbe': 'Technology', 'crm': 'Technology', 'abnb': 'Consumer Discretionary', 'baba': 'Consumer Discretionary',
        'qcom': 'Technology', 'mcd': 'Consumer Discretionary', 'orcl': 'Technology',
        'spy': 'Index ETF', 'qqq': 'Index ETF', 'dia': 'Index ETF', 'voo': 'Index ETF',
        'vti': 'Index ETF', 'iwm': 'Index ETF', 'arkk': 'Innovation ETF', 'xlk': 'Sector ETF (Tech)',
        'xlv': 'Sector ETF (Healthcare)', 'xlf': 'Sector ETF (Financials)', 'tsm': 'Technology',
        'bmy': 'Healthcare', 'intuit': 'Technology', 'low': 'Consumer Discretionary', 'ge': 'Industrials',
        'tgt': 'Consumer Staples', 'sofi': 'Financial Technology', 'snow': 'Technology', 'pltr': 'Technology',
        'shop': 'Technology', 'docu': 'Technology', 'square': 'Financial Technology', 'coin': 'Financial Technology',
        'snap': 'Communication Services', 'pypl': 'Financial Technology', 'etsy': 'Consumer Discretionary',
        'roku': 'Communication Services', 'zillow': 'Real Estate', 'amd': 'Technology',
        'asml': 'Technology', 'regn': 'Healthcare', 'isrg': 'Healthcare', 'jd': 'Consumer Discretionary',
        'pdd': 'Consumer Discretionary', 'ba': 'Industrials', 'ups': 'Industrials', 'fdx': 'Industrials',
        'gm': 'Automotive'
    }

    explanation_lines = []
    sectors = set()

    for idx, row in top5_df.iterrows():
        ticker = row['ticker'].lower()
        sector = sector_mapping.get(ticker, "General Sector")
        sectors.add(sector)

        # Volatility classification
        volatility_label = "stable returns" if row['volatility'] < 0.03 else "higher volatility"

        # Momentum classification
        if pd.isna(row['momentum']):
            momentum_label = "consistent performance"
        elif row['momentum'] > 0:
            momentum_label = "strong upward momentum"
        else:
            momentum_label = "cautious performance recently"

        stock_line = f"- 📈 {row['ticker'].upper()}: {sector} sector, {volatility_label}, and {momentum_label}."
        explanation_lines.append(stock_line)

    # === Step 2: Portfolio Summary based on sectors and risk ===
    risk_comment = {
        "Low": "focuses heavily on stability and defensive sectors.",
        "Medium": "balances stability with moderate growth exposure.",
        "High": "leans towards higher growth but riskier opportunities."
    }

    horizon_comment = {
        "Short Term": "targeting quick momentum plays within 6 months.",
        "Long Term": "aiming for compounding steady growth over multiple years."
    }

    portfolio_summary = f"\nOverall, your portfolio {risk_comment.get(RISK_LEVEL, '')} It is {horizon_comment.get(HORIZON, '')} It is diversified across {len(sectors)} sectors."

    # === Step 3: Assemble ===
    explanation_text = "\n".join(explanation_lines) + "\n" + portfolio_summary
    return explanation_text

# ---- Predict and Recommend ----
env_trade, trade = get_trade_env()
model = SAC.load("model/sac_model_v2.zip")

state = env_trade.reset()
if isinstance(state, tuple):
    state = state[0]

action, _ = model.predict(state, deterministic=True)

actions_df = pd.DataFrame({
    "ticker": trade["tic"].unique(),
    "allocation_score": action
})

# Latest prices for calculating momentum & volatility
latest_prices = trade.sort_values(by=["date"]).groupby("tic").tail(120)

# Volatility
volatility_df = latest_prices.groupby("tic")["close"].std().reset_index()
volatility_df.rename(columns={"close": "volatility"}, inplace=True)

# Momentum
horizon_days = 30 if HORIZON == "Short Term" else 120
momentum_df = latest_prices.groupby("tic").apply(
    lambda x: (x.iloc[-1]["close"] - x.iloc[-horizon_days]["close"]) / x.iloc[-horizon_days]["close"]
).reset_index(name="momentum")

# Merge all
# Merge all
volatility_df.rename(columns={"tic": "ticker"}, inplace=True)
momentum_df.rename(columns={"tic": "ticker"}, inplace=True)

actions_df = actions_df.merge(volatility_df, on="ticker", how="left")
actions_df = actions_df.merge(momentum_df, on="ticker", how="left")


# Alpha (risk level) random assignment
if RISK_LEVEL == "Low":
    alpha = random.uniform(0.8, 1.0)
elif RISK_LEVEL == "Medium":
    alpha = random.uniform(0.6, 0.8)
else:
    alpha = random.uniform(0.4, 0.6)

beta = 0.5  # Fixed momentum boost

# Final score calculation
actions_df["final_score"] = actions_df["allocation_score"] - (alpha * actions_df["volatility"]) + (beta * actions_df["momentum"])

# Sort & pick Top 5
top5 = actions_df.sort_values(by="final_score", ascending=False).head(5)

# Normalize allocations
total_score = top5["final_score"].sum()
top5["weight"] = top5["final_score"] / total_score
top5["amount_to_invest"] = (top5["weight"] * INVESTMENT_AMOUNT).round(2)
top5["recommendation"] = "Buy"

# --- Print Recommendations ---
print("\n🧠 Top 5 Investment Recommendations:\n")
for idx, row in top5.iterrows():
    print(f"📈 {row['ticker'].upper()} ({row['recommendation']}) --> ₹{row['amount_to_invest']:.2f}")

# After printing stock recommendations
explanation = generate_explanation(top5)
print("\n🧠 Why this portfolio was suggested:\n")
print(explanation)

print("\nInvestment Amount:", INVESTMENT_AMOUNT)
print("Risk Level:", RISK_LEVEL)
print("Horizon:", HORIZON)
