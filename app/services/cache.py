"""
Bộ nhớ đệm (Cache) trong RAM, thread-safe, hỗ trợ TTL (thời gian sống) sử dụng asyncio.Lock.
"""
import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    """Đại diện cho một mục trong cache, bao gồm giá trị và thời điểm hết hạn."""
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl_seconds: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl_seconds


class CacheService:
    """
    Dịch vụ quản lý cache trong bộ nhớ với TTL theo từng khóa.

    Các khóa (keys) thường dùng:
      - "products_all" : Danh sách [ProductBase] lấy từ MongoDB.
      - "tfidf_matrix" : Bộ (vectorizer, matrix) do RecommenderService tạo ra.
    """

    def __init__(self, default_ttl_minutes: float = 30.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()  # Đảm bảo an toàn khi truy cập đồng thời
        self.default_ttl_seconds = default_ttl_minutes * 60
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """Lấy giá trị từ cache theo key. Trả về None nếu không thấy hoặc đã hết hạn."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.debug("Cache MISS: %s", key)
                return None
            
            # Kiểm tra xem dữ liệu đã quá hạn chưa
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                logger.debug("Cache EXPIRED: %s", key)
                return None
            
            logger.debug("Cache HIT: %s", key)
            return entry.value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[float] = None
    ) -> None:
        """Lưu giá trị vào cache với một key và thời gian sống (TTL) tùy chọn."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        async with self._lock:
            self._store[key] = CacheEntry(value, ttl)
            self._created_at = time.monotonic()
            logger.debug("Cache SET: %s (TTL=%.0fs)", key, ttl)

    async def invalidate(self, key: str) -> None:
        """Xóa một key cụ thể khỏi cache."""
        async with self._lock:
            self._store.pop(key, None)
            logger.debug("Cache INVALIDATED: %s", key)

    async def clear(self) -> None:
        """Xóa toàn bộ cache."""
        async with self._lock:
            self._store.clear()
            self._created_at = time.monotonic()
            logger.info("Cache CLEARED")

    def age_minutes(self) -> float:
        """Số phút kể từ lần cập nhật cache gần nhất."""
        return (time.monotonic() - self._created_at) / 60.0

    async def is_alive(self, key: str) -> bool:
        """Kiểm tra xem một key có tồn tại và còn hạn hay không."""
        return await self.get(key) is not None


# Tạo instance singleton để dùng chung trong toàn bộ ứng dụng
cache_service = CacheService()
