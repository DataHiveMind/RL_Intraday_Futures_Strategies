"""Feature engineering pipeline for market data."""

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute and manage derived features from OHLCV data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer with optional config."""
        self.config = config or {}
        self.features_computed = {}

    def compute_returns(
        self, data: pd.DataFrame, window: int = 1, method: str = "log"
    ) -> pd.Series:
        """Compute returns.

        Args:
            data: DataFrame with 'close' column.
            window: Return horizon (e.g., 1 for 1-bar returns).
            method: 'log' for log returns, 'simple' for simple returns.

        Returns:
            Series of returns.
        """
        if method == "log":
            returns = (
                data["close"]
                .pct_change(window)
                .apply(lambda x: x.log() if x > 0 else 0)
            )
        else:  # simple
            returns = data["close"].pct_change(window)
        return returns

    def compute_volatility(
        self, data: pd.DataFrame, window: int = 5, method: str = "realized"
    ) -> pd.Series:
        """Compute realized volatility over a rolling window.

        Args:
            data: DataFrame with 'close' column.
            window: Rolling window size in bars.
            method: 'realized' for rolling std of returns.

        Returns:
            Series of volatility.
        """
        if method == "realized":
            returns = self.compute_returns(data, window=1, method="log")
            volatility = returns.rolling(window).std()
        else:
            volatility = data["close"].rolling(window).std()
        return volatility

    def compute_moving_average(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """Compute simple moving average."""
        return data["close"].rolling(window).mean()

    def compute_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = data["close"].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(
        self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Compute MACD and signal line."""
        ema_fast = data["close"].ewm(span=fast).mean()
        ema_slow = data["close"].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def compute_volume_imbalance(
        self, data: pd.DataFrame, window: int = 5
    ) -> pd.Series:
        """Compute volume imbalance proxy (buy vs. sell pressure)."""
        volume = data["volume"]
        close_change = data["close"].diff()
        buy_volume = volume.where(close_change > 0, 0)
        sell_volume = volume.where(close_change < 0, 0)
        buy_pressure = buy_volume.rolling(window).sum()
        sell_pressure = sell_volume.rolling(window).sum()
        total = buy_pressure + sell_pressure
        imbalance = (buy_pressure - sell_pressure) / total.replace(0, 1e-8)
        return imbalance

    def compute_spread_proxy(self, data: pd.DataFrame) -> pd.Series:
        """Compute bid-ask spread proxy as (high - low) / close."""
        spread = (data["high"] - data["low"]) / data["close"]
        return spread

    def engineer_features(
        self, data: pd.DataFrame, feature_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compute all configured features and return augmented DataFrame.

        Args:
            data: Raw OHLCV DataFrame.
            feature_config: Feature computation config (optional).

        Returns:
            DataFrame with original + computed features.
        """
        feature_config = feature_config if feature_config is not None else self.config.get("features", {})
        if feature_config is None:
            feature_config = {}
        result = data.copy()

        if feature_config.get("returns", {}).get("include"):
            windows = feature_config.get("returns", {}).get("window", [1])
            for w in windows:
                result[f"returns_{w}"] = self.compute_returns(data, window=w)

        if feature_config.get("volatility", {}).get("include"):
            windows = feature_config.get("volatility", {}).get("window", [5])
            for w in windows:
                result[f"volatility_{w}"] = self.compute_volatility(data, window=w)

        if feature_config.get("moving_average", {}).get("include"):
            windows = feature_config.get("moving_average", {}).get("windows", [5])
            for w in windows:
                result[f"ma_{w}"] = self.compute_moving_average(data, window=w)

        if feature_config.get("rsi", {}).get("include"):
            window = feature_config.get("rsi", {}).get("window", 14)
            result["rsi"] = self.compute_rsi(data, window=window)

        if feature_config.get("macd", {}).get("include"):
            macd_config = feature_config.get("macd", {})
            macd_features = self.compute_macd(
                data,
                fast=macd_config.get("fast", 12),
                slow=macd_config.get("slow", 26),
                signal=macd_config.get("signal", 9),
            )
            for key, val in macd_features.items():
                result[f"macd_{key}"] = val

        if feature_config.get("imbalance", {}).get("include"):
            windows = feature_config.get("imbalance", {}).get("window", [5])
            for w in windows:
                result[f"imbalance_{w}"] = self.compute_volume_imbalance(data, window=w)

        if feature_config.get("spread_proxy", {}).get("include"):
            result["spread_proxy"] = self.compute_spread_proxy(data)

        logger.info(f"Engineered {len(result.columns) - len(data.columns)} features")
        return result.dropna()


__all__ = ["FeatureEngineer"]
