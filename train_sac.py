import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import SAC
from finrl.config import INDICATORS
from stable_baselines3.common.logger import configure

import os
import itertools
import datetime
import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.main import check_and_make_directories
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR

check_and_make_directories([TRAINED_MODEL_DIR])

# Dates
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2020-07-01'


# Full Stock List
symbols = [
    'aapl', 'msft', 'meta', 'ibm', 'hd', 'cat', 'amzn', 'intc', 't', 'v', 'gs',
    'tsla', 'nvda', 'goog', 'googl', 'brk.b', 'jpm', 'unh', 'vzb', 'xom', 'wmt',
    'cvx', 'pg', 'ma', 'dis', 'ko', 'pepsico', 'nke', 'mrk', 'pfe', 'abbv',
    'cost', 'csco', 'adbe', 'crm', 'abnb', 'baba', 'qcom', 'mcd', 'orcl',
    'spy', 'qqq', 'dia', 'voo', 'vti', 'iwm', 'arkk', 'xlk', 'xlv', 'xlf',
    'tsm', 'bmy', 'intuit', 'low', 'ge', 'tgt', 'meta', 'sofi', 'snow', 'pltr',
    'shop', 'docu', 'square', 'coin', 'snap', 'pypl', 'etsy', 'roku', 'zillow',
    'amd', 'asml', 'regn', 'isrg', 'jd', 'pdd', 'ba', 'ups', 'fdx', 'gm'
]

# Step 1: Download data
print("🔽 Downloading data...")
df_raw = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, ticker_list=symbols).fetch_data()
df_raw = df_raw.drop_duplicates(subset=["date", "tic"], keep="first")

# Step 2: Feature Engineering
print("⚙️ Feature engineering...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False
)
processed = fe.preprocess_data(df_raw)

# Step 3: Fill missing combinations
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combo = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combo, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic']).fillna(0)

# Save FINAL list of clean tickers for prediction phase
clean_tickers = processed_full["tic"].unique().tolist()
with open('model/trained_tickers.txt', 'w') as f:
    for ticker in clean_tickers:
        f.write(ticker + '\n')
print(f"✅ Saved {len(clean_tickers)} clean tickers to model/trained_tickers.txt")

# Step 4: Train-test split
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(f"📈 Train samples: {len(train)}, Trade samples: {len(trade)}")

# Step 5: Env setup
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"🧠 Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# Step 6: Train the SAC model
print("🏋️ Training SAC...")
e_train = StockTradingEnv(df=train, **env_kwargs)
agent = DRLAgent(env=e_train)
model_sac = agent.get_model("sac")

trained_sac = agent.train_model(
    model=model_sac,
    tb_log_name="sac_v2",
    total_timesteps=50000
)

# Step 7: Save Model
trained_sac.save("model/sac_model_v2.zip")
print("✅ SAC V2 model trained and saved successfully!")