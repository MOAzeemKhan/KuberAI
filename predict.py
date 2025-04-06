from flask import Flask, jsonify
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import itertools

app = Flask(__name__)

SYMBOLS = ['aapl', 'msft', 'meta', 'ibm', 'hd', 'cat', 'amzn', 'intc', 't', 'v', 'gs']

TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2023-05-01'

def get_trade_env():
    df_raw = YahooDownloader(start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=SYMBOLS).fetch_data()

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

@app.route('/predict', methods=['GET'])
def predict():
    env_trade, trade = get_trade_env()
    model = SAC.load("model/sac_model.zip")

    state = env_trade.reset()
    if isinstance(state, tuple):
        state = state[0]

    action, _ = model.predict(state, deterministic=True)

    actions_df = pd.DataFrame({
        "ticker": trade["tic"].unique(),
        "allocation": action
    })

    top5 = actions_df.sort_values(by="allocation", ascending=False).head(5)
    top5["allocation"] = (top5["allocation"] * 100).round(2)
    top5["recommendation"] = "Buy"

    return jsonify(top5.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
