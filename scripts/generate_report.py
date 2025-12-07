#!/usr/bin/env python3
"""Generate report artifacts (PNGs + interactive HTML) outside the notebook.

Run from repository root:
    python scripts/generate_report.py

This script is intentionally lightweight: it re-uses the project's ingestion
and feature-engineering modules to fetch data, compute features, and
write a few representative visualizations to `reports/visualizations` and
interactive HTML to `reports/interactive`.
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_report")


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(p: Path):
    with open(p, "r") as fh:
        return yaml.safe_load(fh)


def main():
    logger.info("Starting report generation")
    cfg_ing = load_yaml(ROOT / "config" / "data" / "ingestion.yaml")
    cfg_feat = load_yaml(ROOT / "config" / "data" / "features.yaml")

    tickers = cfg_ing.get("tickers", ["SPY", "QQQ", "IWM"])
    period = cfg_ing.get("period", "12y")
    interval = cfg_ing.get("interval", "1d")

    # Import project modules (must be importable from repo root)
    try:
        from src.data.ingestion import DataIngester
        from src.data.feature_pipeline import FeatureEngineer
    except Exception as e:
        logger.error("Failed to import project modules: %s", e)
        raise

    out_vis = ROOT / "reports" / "visualizations"
    out_int = ROOT / "reports" / "interactive"
    safe_mkdir(out_vis)
    safe_mkdir(out_int)

    logger.info(
        "Fetching data for %s (period=%s, interval=%s)", tickers, period, interval
    )
    ing = DataIngester()
    df = ing.fetch(tickers=tickers, period=period, interval=interval)

    # engineer features
    fe = FeatureEngineer(cfg_feat)
    df_features = fe.engineer_features(df)

    # attempt to get 'close' and 'volume' into DataFrames with tickers as columns
    if isinstance(df.index, pd.MultiIndex):
        close = df["close"].unstack(level=-1)
        volume = df["volume"].unstack(level=-1)
    else:
        close = df[["close"]].copy()
        volume = df[["volume"]].copy()

    returns = np.log(close / close.shift(1))

    # 1) Returns distribution (combined)
    plt.figure(figsize=(8, 4))
    returns.dropna(how="all").stack().hist(bins=120)
    plt.title("Returns Distribution (all tickers)")
    p1 = out_vis / "01_returns_distribution.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", p1)

    # 2) Volatility over time (30-day rolling std of log returns)
    vol30 = returns.rolling(30).std()
    plt.figure(figsize=(10, 4))
    for col in vol30.columns[:4]:
        plt.plot(vol30.index, vol30[col], label=str(col))
    plt.legend()
    plt.title("30-day Rolling Volatility")
    p2 = out_vis / "02_volatility_over_time.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", p2)

    # 3) Technical indicators (take first ticker)
    first = returns.columns[0]
    ti = (
        df_features.xs(first, level=-1, drop_level=False)
        if isinstance(df_features.index, pd.MultiIndex)
        else df_features
    )
    # pick some columns if present
    cols = [c for c in ti.columns if "rsi" in c or "macd" in c or "returns" in c]
    plt.figure(figsize=(10, 5))
    if len(cols) >= 2:
        plt.plot(ti.index, ti[cols[0]], label=cols[0])
        plt.plot(ti.index, ti[cols[1]], label=cols[1])
    else:
        plt.plot(close.index, close[first], label=str(first))
    plt.legend()
    plt.title(f"Technical Indicators / Price - {first}")
    p3 = out_vis / "03_technical_indicators.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", p3)

    # 4) Price vs Volume (first ticker)
    plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax1.plot(close.index, close[first], color="tab:blue")
    ax1.set_ylabel("Price", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.fill_between(
        volume.index, 0, volume[first].fillna(0), color="tab:gray", alpha=0.3
    )
    ax2.set_ylabel("Volume", color="tab:gray")
    plt.title(f"Price vs Volume - {first}")
    p4 = out_vis / "04_price_vs_volume.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", p4)

    # 5) Feature correlation heatmap (across features for the first ticker if multiindex)
    if isinstance(df_features.index, pd.MultiIndex):
        try:
            df_corr = df_features.xs(first, level=-1).dropna()
        except Exception:
            df_corr = df_features.dropna()
    else:
        df_corr = df_features.dropna()

    corr = df_corr.corr()
    plt.figure(figsize=(8, 6))
    import seaborn as sns

    sns.heatmap(corr, cmap="RdBu_r", center=0)
    plt.title("Feature Correlations")
    p5 = out_vis / "05_feature_correlations.png"
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", p5)

    # Save a quick CSV snapshot of features
    csv_out = ROOT / "reports" / "data_snapshot.csv"
    df_features.head(1000).to_csv(csv_out)
    logger.info("Saved feature snapshot %s", csv_out)

    # Make a tiny interactive HTML for price + returns of the first 2 tickers (if available)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        for col in close.columns[:2]:
            fig.add_trace(
                go.Scatter(x=close.index, y=close[col], name=f"price-{col}"),
                row=1,
                col=1,
            )
        for col in returns.columns[:2]:
            fig.add_trace(
                go.Scatter(x=returns.index, y=returns[col], name=f"returns-{col}"),
                row=2,
                col=1,
            )
        fig.update_layout(height=700, title_text="Price and Returns (interactive)")
        p_int = out_int / "price_returns.html"
        fig.write_html(str(p_int), include_plotlyjs="cdn")
        logger.info("Saved interactive %s", p_int)
    except Exception as e:
        logger.warning("Failed to create interactive chart: %s", e)

    logger.info(
        "Report generation completed. Visualizations in %s and %s", out_vis, out_int
    )


if __name__ == "__main__":
    main()
