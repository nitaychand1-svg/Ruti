"""[FIX-13] Production caching layer."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional


class PredictionCache:
    """Thread-safe prediction cache with TTL."""

    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value asynchronously."""
        async with self._lock:
            return self._get_internal(key)

    async def set(self, key: str, value: Any) -> None:
        """Set cached value asynchronously."""
        async with self._lock:
            self._set_internal(key, value)

    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get for non-async contexts."""
        return self._get_internal(key)

    def set_sync(self, key: str, value: Any) -> None:
        """Synchronous set."""
        self._set_internal(key, value)

    # Internal helpers -------------------------------------------------
    def _get_internal(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None

        value, expiry = entry
        if datetime.now() > expiry:
            self._cache.pop(key, None)
            return None
        return value

    def _set_internal(self, key: str, value: Any) -> None:
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            self._cache.pop(oldest_key, None)

        expiry = datetime.now() + timedelta(seconds=self.ttl)
        self._cache[key] = (value, expiry)
