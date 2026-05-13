"""
Điểm bắt đầu của ứng dụng FastAPI.

Vòng đời ứng dụng (Lifespan):
  - Startup: Kết nối MongoDB, nạp dữ liệu vào cache (warm up) bằng cách tính toán ma trận TF-IDF.
  - Background task: Tự động cập nhật lại cache định kỳ theo CACHE_TTL_MINUTES.
  - Shutdown: Đóng kết nối MongoDB an toàn.
"""
import asyncio
import logging
import logging.config
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.models.product import HealthResponse
from app.services.cache import cache_service
from app.services.recommender import CACHE_KEY_PRODUCTS, CACHE_KEY_TFIDF
from app.services.recommender import recommender_service
from app.routers.recommendation import router as recommendation_router

# Logging - Cấu hình ghi log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Background refresh task - Tác vụ chạy ngầm để cập nhật AI matrix

_refresh_task: asyncio.Task | None = None


async def _background_refresh() -> None:
    """Chạy lặp vô hạn để làm mới ma trận TF-IDF định kỳ."""
    interval = settings.CACHE_TTL_MINUTES * 60
    while True:
        await asyncio.sleep(interval)
        try:
            await recommender_service.refresh_cache()
        except Exception as exc:
            logger.error("Cập nhật cache ngầm thất bại: %s", exc)


# Lifespan - Quản lý khởi động và tắt ứng dụng

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task

    # ---- Giai đoạn Khởi động (Startup) ----
    logger.info("Đang khởi động Recommendation Service...")
    await connect_to_mongo()

    logger.info("Đang nạp cache: lấy sản phẩm và xây dựng ma trận TF-IDF...")
    try:
        await recommender_service.refresh_cache()
    except Exception as exc:
        logger.error("Nạp cache ban đầu thất bại (dịch vụ vẫn sẽ khởi chạy): %s", exc)

    # Khởi chạy tác vụ cập nhật định kỳ
    _refresh_task = asyncio.create_task(_background_refresh())
    logger.info("Đã bắt đầu tác vụ cập nhật ngầm (chu kỳ=%dm)", settings.CACHE_TTL_MINUTES)

    yield  # Ứng dụng hoạt động tại đây

    # ---- Giai đoạn Tắt máy (Shutdown) ----
    if _refresh_task and not _refresh_task.done():
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass

    await close_mongo_connection()
    logger.info("Recommendation Service đã dừng an toàn.")


# App - Khởi tạo ứng dụng FastAPI

app = FastAPI(
    title="Food Recommendation Service",
    description="Microservice gợi ý sản phẩm dựa trên nội dung (Content-Based Filtering)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- CORS - Cho phép các domain khác truy cập API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers - Tích hợp các cụm API ----
logger.info("Đang tích hợp recommendation router...")
app.include_router(recommendation_router)


# Health check - Kiểm tra sức khỏe hệ thống

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Kiểm tra trạng thái dịch vụ",
)
async def health_check() -> HealthResponse:
    """Trả về thông tin về số lượng sản phẩm trong cache và trạng thái ma trận AI."""
    products = await cache_service.get(CACHE_KEY_PRODUCTS)
    matrix_data = await cache_service.get(CACHE_KEY_TFIDF)

    return HealthResponse(
        status="ok",
        products_cached=len(products) if products else 0,
        matrix_built=matrix_data is not None,
        cache_age_minutes=round(cache_service.age_minutes(), 2),
    )


@app.post("/cache-refresh", tags=["admin"])
async def manual_cache_refresh():
    """Endpoint thủ công để yêu cầu tính toán lại ma trận AI ngay lập tức."""
    try:
        await recommender_service.refresh_cache()
        return {"success": True, "message": "Đã tính toán lại ma trận AI thành công!"}
    except Exception as exc:
        logger.exception("Thủ công làm mới cache thất bại: %s", exc)
        return JSONResponse(status_code=500, content={"success": False, "detail": str(exc)})


# Xử lý lỗi toàn cục (Ẩn stack trace khi gặp lỗi không mong muốn)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Lỗi chưa được xử lý tại %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau.",
        },
    )


# Chạy trực tiếp bằng python app/main.py

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
