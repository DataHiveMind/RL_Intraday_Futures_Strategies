"""Arctic DB storage backend for efficient time-series data management.

Uses modern arcticdb API: Arctic(db_url) -> get_library() -> write/read/update/delete.
"""

import importlib.util
import logging
from typing import Dict, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


class ArcticDBStore:
    """Wrapper for modern ArcticDB time-series storage.

    Arctic DB is designed for high-performance storage of pandas DataFrames
    with native support for time-series data, versioning, and efficient storage.

    Uses the modern arcticdb API: Arctic(db_url) -> get_library() -> write/read/update/delete.
    """

    def __init__(
        self,
        db_url: str = "lmdb://financial_data",
        library_name: str = "futures_intraday",
    ):
        """Initialize Arctic DB store.

        Args:
            db_url: Arctic DB connection URI (e.g., "lmdb://path", "mem://").
                   Defaults to LMDB backend for persistence.
            library_name: Name of the Arctic DB library/namespace.
        """
        self.db_url = db_url
        self.library_name = library_name
        self._lib = None
        self._initialized = False
        self._mock_storage = None

    def _flatten_multiindex_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to single level (ArcticDB limitation).

        Args:
            df: DataFrame with potentially MultiIndex columns.

        Returns:
            DataFrame with flattened column names.
        """
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            logger.debug(f"Flattened MultiIndex columns to {len(df.columns)} columns")
        return df

    def _ensure_initialized(self):
        """Lazy initialization of Arctic connection."""
        if self._initialized:
            return

        # Try modern arcticdb API
        try:
            spec = importlib.util.find_spec("arcticdb")
            if spec is None:
                raise ImportError("arcticdb not found")
        except Exception:
            spec = None

        if spec is not None:
            try:
                import arcticdb as adb

                ac = adb.Arctic(self.db_url)
                self._lib = ac.get_library(self.library_name, create_if_missing=True)
                logger.info(
                    f"Connected to Arctic DB at {self.db_url}, "
                    f"library: {self.library_name}"
                )
                self._initialized = True
                return
            except Exception as e:
                logger.warning(
                    f"ArcticDB initialization failed ({e}); "
                    f"falling back to mock storage"
                )

        # Fall back to in-memory mock store if arcticdb not available
        logger.warning(
            "Arctic DB not available; using mock storage. Install: pip install arcticdb"
        )
        self._mock_storage = {}
        self._initialized = True

    def write(
        self, symbol: str, data: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> None:
        """Write DataFrame to Arctic DB.

        Args:
            symbol: Unique symbol/identifier for the data.
            data: DataFrame with time-series data (may have MultiIndex columns).
            metadata: Optional metadata dict to attach.
        """
        self._ensure_initialized()

        # Flatten MultiIndex columns if present
        data = self._flatten_multiindex_columns(data)

        if self._mock_storage is not None:
            self._mock_storage[symbol] = {"data": data, "metadata": metadata}
            logger.debug(f"Mock store: wrote {symbol} ({len(data)} rows)")
        else:
            if self._lib is None:
                raise RuntimeError("Arctic DB library is not initialized.")
            try:
                self._lib.write(symbol, data)
                logger.info(f"Wrote {symbol} to Arctic DB ({len(data)} rows)")
            except Exception as e:
                logger.error(f"Failed to write {symbol}: {e}")
                raise

    def read(self, symbol: str) -> pd.DataFrame:
        """Read DataFrame from Arctic DB.

        Args:
            symbol: Unique symbol/identifier.

        Returns:
            Retrieved DataFrame.
        """
        self._ensure_initialized()

        if self._mock_storage is not None:
            if symbol not in self._mock_storage:
                raise KeyError(f"Symbol not found in mock store: {symbol}")
            logger.debug(f"Mock store: read {symbol}")
            return self._mock_storage[symbol]["data"].copy()
        else:
            if self._lib is None:
                raise RuntimeError("Arctic DB library is not initialized.")
            try:
                data = self._lib.read(symbol).data
                logger.info(f"Read {symbol} from Arctic DB ({len(data)} rows)")
                return data
            except Exception as e:
                logger.error(f"Failed to read {symbol}: {e}")
                raise

    def update(
        self, symbol: str, data: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> None:
        """Update (upsert) data in Arctic DB.

        For mock storage: concatenates with existing data if present.
        For real storage: uses Arctic's update to append/upsert.

        Args:
            symbol: Unique symbol/identifier.
            data: DataFrame rows to append/upsert.
            metadata: Optional metadata dict.
        """
        self._ensure_initialized()

        # Flatten MultiIndex columns if present
        data = self._flatten_multiindex_columns(data)

        if self._mock_storage is not None:
            if symbol in self._mock_storage:
                existing = self._mock_storage[symbol]["data"]
                self._mock_storage[symbol]["data"] = pd.concat(
                    [existing, data], ignore_index=False
                )
                logger.debug(
                    f"Mock store: updated {symbol} "
                    f"({len(data)} new rows, "
                    f"{len(self._mock_storage[symbol]['data'])} total)"
                )
            else:
                self._mock_storage[symbol] = {"data": data, "metadata": metadata}
                logger.debug(f"Mock store: created {symbol} ({len(data)} rows)")
        else:
            if self._lib is None:
                raise RuntimeError("Arctic DB library is not initialized.")
            try:
                self._lib.update(symbol, data)
                logger.info(f"Updated {symbol} in Arctic DB ({len(data)} rows)")
            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                raise

    def list_symbols(self) -> List[str]:
        """List all symbols in the library.

        Returns:
            List of symbol names.
        """
        self._ensure_initialized()

        if self._mock_storage is not None:
            return list(self._mock_storage.keys())
        else:
            if self._lib is None:
                logger.error("Arctic DB library is not initialized.")
                return []
            try:
                symbols = self._lib.list_symbols()
                logger.debug(f"Listed {len(symbols)} symbols from Arctic DB")
                return symbols
            except Exception as e:
                logger.error(f"Failed to list symbols: {e}")
                return []

    def delete(self, symbol: str) -> None:
        """Delete a symbol from Arctic DB.

        Args:
            symbol: Symbol identifier to delete.
        """
        self._ensure_initialized()

        if self._mock_storage is not None:
            if symbol in self._mock_storage:
                del self._mock_storage[symbol]
                logger.debug(f"Mock store: deleted {symbol}")
            else:
                raise KeyError(f"Symbol not found in mock store: {symbol}")
        else:
            if self._lib is None:
                raise RuntimeError("Arctic DB library is not initialized.")
            try:
                self._lib.delete(symbol)
                logger.info(f"Deleted {symbol} from Arctic DB")
            except Exception as e:
                logger.error(f"Failed to delete {symbol}: {e}")
                raise

    def display_data(self, symbol: str, head: int = 10) -> None:
        """Display stored data for inspection (CLI debugging).

        Args:
            symbol: Symbol to display.
            head: Number of rows to display.
        """
        try:
            df = self.read(symbol)
            print(f"\n=== {symbol} ===")
            print(f"Shape: {df.shape}")
            print(f"Index: {df.index.name} ({df.index.dtype})")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst {head} rows:")
            print(df.head(head))
        except Exception as e:
            logger.error(f"Failed to display {symbol}: {e}")


__all__ = ["ArcticDBStore"]
