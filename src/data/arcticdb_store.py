"""Arctic DB storage backend for efficient time-series data management."""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ArcticDBStore:
    """Lightweight wrapper for Arctic DB / time-series storage.

    Arctic DB is designed for high-performance storage of pandas DataFrames
    with native support for time-series data, versioning, and efficient storage.
    """

    def __init__(
        self, library_name: str = "futures_intraday", uri: Optional[str] = None
    ):
        """Initialize Arctic DB store.

        Args:
            library_name: Name of the Arctic DB library/namespace.
            uri: Arctic DB connection URI (e.g., "mem://", "lmdb:///path").
                 If None, defaults to in-memory store for testing.
        """
        self.library_name = library_name
        self.uri = uri or "mem://"
        self._lib = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of Arctic connection."""
        if self._initialized:
            return
        try:
            from arctic import Arctic
        except ImportError:
            logger.warning(
                "Arctic DB not available; using mock storage. Install: pip install arctic"
            )
            self._mock_storage = {}
            self._initialized = True
            return

        try:
            self._ac = Arctic(self.uri)
            if not self._ac.library_exists(self.library_name):
                self._ac.initialize_library(self.library_name)
            self._lib = self._ac[self.library_name]
            logger.info(f"Connected to Arctic DB library: {self.library_name}")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Arctic DB initialization failed: {e}; using mock storage")
            self._mock_storage = {}
            self._initialized = True

    def write(
        self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> None:
        """Write DataFrame to Arctic DB.

        Args:
            key: Unique symbol/identifier for the data.
            data: DataFrame with time-series data.
            metadata: Optional metadata dict to attach.
        """
        self._ensure_initialized()
        if hasattr(self, "_mock_storage"):
            self._mock_storage[key] = {"data": data, "metadata": metadata}
            logger.debug(f"Mock store: wrote {key}")
        else:
            try:
                self._lib.write(key, data, metadata=metadata or {})
                logger.info(f"Wrote {key} to Arctic DB")
            except Exception as e:
                logger.error(f"Failed to write {key}: {e}")
                raise

    def read(self, key: str) -> pd.DataFrame:
        """Read DataFrame from Arctic DB.

        Args:
            key: Unique symbol/identifier.

        Returns:
            Retrieved DataFrame.
        """
        self._ensure_initialized()
        if hasattr(self, "_mock_storage"):
            if key not in self._mock_storage:
                raise KeyError(f"Key not found in mock store: {key}")
            logger.debug(f"Mock store: read {key}")
            return self._mock_storage[key]["data"]
        else:
            try:
                data = self._lib.read(key)
                logger.info(f"Read {key} from Arctic DB")
                return data.data
            except Exception as e:
                logger.error(f"Failed to read {key}: {e}")
                raise

    def update(
        self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None
    ) -> None:
        """Update (append) data in Arctic DB.

        Args:
            key: Unique symbol/identifier.
            data: DataFrame rows to append.
            metadata: Optional metadata dict.
        """
        self._ensure_initialized()
        if hasattr(self, "_mock_storage"):
            if key in self._mock_storage:
                existing = self._mock_storage[key]["data"]
                self._mock_storage[key]["data"] = pd.concat([existing, data])
            else:
                self._mock_storage[key] = {"data": data, "metadata": metadata}
            logger.debug(f"Mock store: updated {key}")
        else:
            try:
                self._lib.update(key, data, metadata=metadata or {})
                logger.info(f"Updated {key} in Arctic DB")
            except Exception as e:
                logger.error(f"Failed to update {key}: {e}")
                raise

    def list_symbols(self):
        """List all symbols in the library."""
        self._ensure_initialized()
        if hasattr(self, "_mock_storage"):
            return list(self._mock_storage.keys())
        else:
            try:
                return self._lib.list_symbols()
            except Exception as e:
                logger.error(f"Failed to list symbols: {e}")
                return []

    def delete(self, key: str) -> None:
        """Delete a symbol from Arctic DB."""
        self._ensure_initialized()
        if hasattr(self, "_mock_storage"):
            if key in self._mock_storage:
                del self._mock_storage[key]
                logger.debug(f"Mock store: deleted {key}")
        else:
            try:
                self._lib.delete(key)
                logger.info(f"Deleted {key} from Arctic DB")
            except Exception as e:
                logger.error(f"Failed to delete {key}: {e}")
                raise


__all__ = ["ArcticDBStore"]
