"""[FIX-13] Production caching layer."""
import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta

class PredictionCache:
    """Thread-safe prediction cache with TTL."""
    
    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            if datetime.now() > expiry:
                del self._cache[key]
                return None
            
            return value
    
    async def set(self, key: str, value: Any):
        """Set cached value."""
        async with self._lock:
            if len(self._cache) >= self.maxsize:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            expiry = datetime.now() + timedelta(seconds=self.ttl)
            self._cache[key] = (value, expiry)
    
    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get for non-async contexts."""
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        if datetime.now() > expiry:
            del self._cache[key]
            return None
        
        return value
    
    def set_sync(self, key: str, value: Any):
        """Synchronous set."""
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        expiry = datetime.now() + timedelta(seconds=self.ttl)
        self._cache[key] = (value, expiry)
