"""
Thread-safe in-memory cache with TTL support using asyncio.Lock.
"""
import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds


class CacheService:
    """
    Simple in-memory cache with per-key TTL.

    Keys used by this service:
      - "products_all"  : list[ProductBase] fetched from MongoDB
      - "tfidf_matrix"  : (vectorizer, matrix) tuple built by RecommenderService
    """

    def __init__(self, default_ttl_minutes: float = 30.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.default_ttl_seconds = default_ttl_minutes * 60
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.debug("Cache MISS: %s", key)
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                logger.debug("Cache EXPIRED: %s", key)
                return None
            logger.debug("Cache HIT: %s", key)
            return entry.value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[float] = None
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        async with self._lock:
            self._store[key] = CacheEntry(value, ttl)
            self._created_at = time.monotonic()
            logger.debug("Cache SET: %s (TTL=%.0fs)", key, ttl)

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
            logger.debug("Cache INVALIDATED: %s", key)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._created_at = time.monotonic()
            logger.info("Cache CLEARED")

    def age_minutes(self) -> float:
        """Minutes since the last cache write (approximate)."""
        return (time.monotonic() - self._created_at) / 60.0

    async def is_alive(self, key: str) -> bool:
        """Return True if key exists and has not expired."""
        return await self.get(key) is not None


# Singleton instance – imported by other modules
cache_service = CacheService()
