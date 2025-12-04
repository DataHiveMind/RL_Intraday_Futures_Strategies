"""Data loaders for reading market data from storage."""

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load market data from filesystem or database."""

    def __init__(self, storage_path: str = "data/store"):
        """Initialize loader pointing to storage directory."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file from storage."""
        filepath = self.storage_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath, index_col=[0, 1], parse_dates=[0])

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load parquet file from storage."""
        filepath = self.storage_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        logger.info(f"Loading data from {filepath}")
        return pd.read_parquet(filepath)

    def save_csv(self, data: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV in storage."""
        filepath = self.storage_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")

    def save_parquet(self, data: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to parquet in storage."""
        filepath = self.storage_path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(filepath)
        logger.info(f"Saved data to {filepath}")

    def list_files(self, suffix: str = ".csv") -> List[str]:
        """List all files with given suffix in storage."""
        return [f.name for f in self.storage_path.rglob(f"*{suffix}")]


__all__ = ["DataLoader"]
