# Reinforcement Learning for Intraday Futures Strategies

## 📌 Project Overview
This repository demonstrates the design and implementation of reinforcement learning (RL) agents for intraday futures trading.  
The goal is to **generate alphas, optimize execution, and manage inventory risk** in high‑frequency environments using realistic market simulations.

Key highlights:
- **RL specialization**: Deep Q‑Learning, PPO, and SAC agents applied to intraday equity/futures proxies.
- **Market microstructure simulation**: Tick‑level and 1‑minute bar data from `yfinance` ETFs (SPY, QQQ, DIA, IWM, GLD, SLV, USO, IEF, TLT).
- **Reproducible pipelines**: Modular ingestion, feature engineering, training, and backtesting.


---

## ⚙️ Data Sources
- **Provider**: Yahoo Finance (`yfinance`)
- **Tickers**:
  - SPY (S&P 500 ETF proxy for ES futures)
  - QQQ (Nasdaq‑100 ETF proxy for NQ futures)
  - DIA (Dow Jones ETF proxy for YM futures)
  - IWM (Russell 2000 ETF proxy for RTY futures)
  - GLD (Gold ETF proxy for GC futures)
  - SLV (Silver ETF proxy for SI futures)
  - USO (Crude Oil ETF proxy for CL futures)
  - IEF (10‑Year Treasury ETF proxy for ZN futures)
  - TLT (20‑Year Treasury ETF proxy for ZB futures)

---

## 🧠 Methodology
- **Environment**: Custom Gym‑style intraday futures environment with transaction costs, slippage, and latency.
- **Agents**:
  - DQN (Deep Q‑Learning)
  - PPO (Proximal Policy Optimization)
  - SAC (Soft Actor‑Critic)
- **Features**:
  - OHLCV bars
  - Order book imbalance proxies
  - Realized volatility
  - Microprice indicators
- **Evaluation**:
  - Walk‑forward validation
  - Risk‑adjusted performance metrics
  - Robustness under simulated liquidity regimes

---

## 📊 Current Results (placeholders)
| Metric                  | DQN   | PPO   | SAC   |
|--------------------------|-------|-------|-------|
| Annualized Return (%)    | NaN   | NaN   | NaN   |
| Sharpe Ratio             | NaN   | NaN   | NaN   |
| Max Drawdown (%)         | NaN   | NaN   | NaN   |
| Win Rate (%)             | NaN   | NaN   | NaN   |
| Avg Trade Duration (min) | NaN   | NaN   | NaN   |
| Turnover (%)             | NaN   | NaN   | NaN   |

> ⚠️ Metrics are currently placeholders (`NaN`) pending full backtests.  

---

## 🚀 How to Run
1. **Install dependencies**
   ```bash
   make setup

2.  **Ingest Data**
    python scripts/fetch_data.py --config configs/data/ingestion.yaml

3. **Train Agent**
    python src/training/train.py --config configs/experiment/dqn.yaml

4. **Evaluate**
    python src/evaluation/evaluate.py --config configs/experiment/dqn.yaml

