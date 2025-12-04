"""Data ingestion module for fetching market data from external sources."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngester:
    """Fetch and ingest market data from external providers (e.g., yfinance).

    Supports downloading OHLCV data with configurable parameters:
    - tickers: list of symbols to download
    - period: rolling window (e.g., "5d")
    - interval: bar frequency (e.g., "1m" for 1-minute)
    - auto_adjust: adjust for splits/dividends
    """

    def __init__(
        self,
        provider: str = "yfinance",
        tickers: Optional[List[str]] = None,
        period: str = "5d",
        interval: str = "1m",
        auto_adjust: bool = True,
        prepost: bool = False,
        threads: int = 4,
    ):
        """Initialize the data ingester.

        Args:
            provider: Data provider name (currently supports 'yfinance').
            tickers: List of ticker symbols to ingest.
            period: Historical period (e.g., "5d", "1mo", "1y").
            interval: Bar interval (e.g., "1m", "5m", "1h", "1d").
            auto_adjust: Automatically adjust for splits/dividends.
            prepost: Include pre/post-market data.
            threads: Number of parallel download threads.
        """
        self.provider = provider
        self.tickers = tickers or []
        self.period = period
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.prepost = prepost
        self.threads = threads

        if provider == "yfinance":
            try:
                import importlib.util

                spec = importlib.util.find_spec("yfinance")
                if spec is None:
                    raise ImportError("yfinance not found")
            except ImportError as err:
                raise ImportError(
                    "yfinance required for DataIngester; install: pip install yfinance"
                ) from err

    def fetch(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch OHLCV data for given tickers.

        Args:
            tickers: List of ticker symbols. If None, uses self.tickers.

        Returns:
            DataFrame with MultiIndex (date, ticker) and columns [Open, High, Low, Close, Volume].
        """
        tickers = tickers or self.tickers
        if not tickers:
            raise ValueError("No tickers provided")

        if self.provider == "yfinance":
            return self._fetch_yfinance(tickers)

        raise ValueError(f"Provider '{self.provider}' not supported")

    def _fetch_yfinance(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch data from yfinance.

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume] indexed by (Datetime, Ticker).
        """
        import yfinance as yf

        logger.info(
            f"Fetching {len(tickers)} tickers from yfinance: period={self.period}, interval={self.interval}"
        )
        data = yf.download(
            tickers,
            period=self.period,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            prepost=self.prepost,
            threads=self.threads,
            progress=False,
        )

        # yfinance returns columns like Open, High, Low, Close, Volume (uppercase)
        # Convert to MultiIndex (Datetime, Ticker) if multiple tickers, else add ticker column
        if len(tickers) == 1:
            data["ticker"] = tickers[0]
            data = data.reset_index().set_index(["Datetime", "ticker"])
        else:
            data = data.stack()
            data.index.names = ["Datetime", "ticker"]

        data.columns = data.columns.str.lower()  # Normalize to lowercase
        logger.info(f"Fetched data shape: {data.shape}")
        return data

    def save_raw(self, data: pd.DataFrame, path: str) -> None:
        """Save raw ingested data to a CSV file.

        Args:
            data: DataFrame to save.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path)
        logger.info(f"Saved raw data to {path}")


def ingest_from_config(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data ingestion config and fetch data.

    Args:
        config: Dictionary with keys like 'provider', 'tickers', 'period', 'interval'.

    Returns:
        Ingested DataFrame.
    """
    ingester = DataIngester(**config)
    return ingester.fetch()


__all__ = ["DataIngester", "ingest_from_config"]
