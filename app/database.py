"""
Async MongoDB connection using motor driver.
"""
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_to_mongo() -> None:
    """Establish MongoDB connection."""
    global _client, _db
    try:
        _client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=10000)
        # Ping to verify connection
        await _client.admin.command("ping")
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
    """Close MongoDB connection."""
    global _client
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _db is None:
        raise RuntimeError("Database not initialised. Call connect_to_mongo() first.")
    return _db


def get_collection():
    """Return the products collection."""
    db = get_database()
    return db[settings.MONGODB_COLLECTION]
