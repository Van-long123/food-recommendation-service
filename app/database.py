"""
Kết nối MongoDB bất đồng bộ bằng driver motor.
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger(__name__)

# Biến toàn cục để lưu trữ client và database instance
_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    """Thiết lập kết nối tới MongoDB."""
    global _client, _db
    try:
        # Khởi tạo client với URI từ cấu hình, timeout 10 giây
        _client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=10000)
        
        # Kiểm tra kết nối bằng lệnh ping
        await _client.admin.command("ping")
        
        # Chọn database
        _db = _client[settings.MONGODB_DB_NAME]
        
        logger.info(
            "Connected to MongoDB | DB: %s | Collection: %s",
            settings.MONGODB_DB_NAME,
            settings.MONGODB_COLLECTION,
        )
    except Exception as exc:
        logger.error("Failed to connect to MongoDB: %s", exc)
        raise


async def close_mongo_connection() -> None:
    """Đóng kết nối MongoDB khi ứng dụng dừng."""
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Trả về instance database đang hoạt động."""
    if _db is None:
        raise RuntimeError("Database chưa được khởi tạo. Hãy gọi connect_to_mongo() trước.")
    return _db


def get_collection():
    """Trả về collection 'products' để thao tác dữ liệu sản phẩm."""
    db = get_database()
    return db[settings.MONGODB_COLLECTION]
