"""
Bộ nhớ đệm (Cache) trong RAM, thread-safe, hỗ trợ TTL (thời gian sống) sử dụng asyncio.Lock.
"""
import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)
#  dùng một cấu trúc dữ liệu Dictionary (dict) cơ bản của Python tên là self._store để lưu trữ dữ liệu dưới dạng Key-Value.

class CacheEntry:
    """Đại diện cho một mục trong cache, bao gồm giá trị và thời điểm hết hạn."""
    # Tối ưu RAM, tránh cấp phát bộ nhớ động (__dict__) dư thừa cho object khi có hàng ngàn sản phẩm
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl_seconds: float) -> None:
        self.value = value
        # Dùng monotonic() (thời gian đơn điệu) chống lỗi đồng bộ giờ hệ điều hành, đảm bảo đếm giờ chính xác
        self.expires_at = time.monotonic() + ttl_seconds


class CacheService:
    """
    Dịch vụ quản lý cache trong bộ nhớ với TTL theo từng khóa.

    Các khóa (keys) thường dùng:
      - "products_all" : Danh sách [ProductBase] lấy từ MongoDB.
      - "tfidf_matrix" : Bộ (vectorizer, matrix) do RecommenderService tạo ra.
    """

    def __init__(self, default_ttl_minutes: float = 30.0) -> None:
        # Biến _store hoạt động như một thanh RAM (Lưu trữ Key-Value)
        self._store: dict[str, CacheEntry] = {}
        # Khóa Mutex (asyncio.Lock): Chống lỗi Race Condition (Đụng độ dữ liệu) khi nhiều user truy cập/ghi đè Cache cùng lúc
        self._lock = asyncio.Lock()  
        self.default_ttl_seconds = default_ttl_minutes * 60
        self._created_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """Lấy giá trị từ cache theo key. Trả về None nếu không thấy hoặc đã hết hạn."""
        async with self._lock: # Yêu cầu quyền truy cập an toàn (Đợi luồng khác nhả khóa)
            entry = self._store.get(key)
            if entry is None:
                logger.debug("Cache MISS: %s", key) # Không có dữ liệu, sẽ phải chạy AI tính toán lại
                return None
            
            # Cơ chế tự hủy dữ liệu quá hạn (Tránh Memory Leak - Rò rỉ RAM)
            if time.monotonic() > entry.expires_at:
                del self._store[key] # Dọn rác
                logger.debug("Cache EXPIRED: %s", key)
                return None
            
            logger.debug("Cache HIT: %s", key) # Có sẵn dữ liệu, trả về ngay lập tức
            return entry.value

    async def set(
        self, key: str, value: Any, ttl_seconds: Optional[float] = None
    ) -> None:
        """Lưu giá trị vào cache với một key và thời gian sống (TTL) tùy chọn."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        async with self._lock: # Yêu cầu quyền truy cập an toàn khi ghi dữ liệu
            self._store[key] = CacheEntry(value, ttl)
            self._created_at = time.monotonic()
            logger.debug("Cache SET: %s (TTL=%.0fs)", key, ttl)

    async def invalidate(self, key: str) -> None:
        """Xóa một key cụ thể khỏi cache (Thường dùng khi CSDL có thay đổi)."""
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


# Design Pattern Singleton: Tạo 1 instance duy nhất để dùng chung toàn app, trỏ về cùng 1 vùng RAM
cache_service = CacheService()
