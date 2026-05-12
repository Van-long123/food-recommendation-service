"""
FastAPI application entry point.

Lifecycle:
  - startup: connect to MongoDB, warm up cache (load products + build TF-IDF matrix)
  - background task: refresh cache every CACHE_TTL_MINUTES
  - shutdown: close MongoDB connection
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background refresh task
# ---------------------------------------------------------------------------

_refresh_task: asyncio.Task | None = None


async def _background_refresh() -> None:
    """Periodically rebuild the TF-IDF matrix."""
    interval = settings.CACHE_TTL_MINUTES * 60
    while True:
        await asyncio.sleep(interval)
        try:
            await recommender_service.refresh_cache()
        except Exception as exc:
            logger.error("Background refresh failed: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task

    # ---- Startup ----
    logger.info("Starting up Recommendation Service...")
    await connect_to_mongo()

    logger.info("Warming up cache: loading products and building TF-IDF matrix...")
    try:
        await recommender_service.refresh_cache()
    except Exception as exc:
        logger.error("Cache warm-up failed (service will still start): %s", exc)

    _refresh_task = asyncio.create_task(_background_refresh())
    logger.info("Background refresh task started (interval=%dm)", settings.CACHE_TTL_MINUTES)

    yield  # Application runs here

    # ---- Shutdown ----
    if _refresh_task and not _refresh_task.done():
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass

    await close_mongo_connection()
    logger.info("Recommendation Service shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Food Recommendation Service",
    description="Content-Based Filtering microservice (TF-IDF + Cosine Similarity)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers ----
logger.info("Including recommendation router...")
app.include_router(recommendation_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Service health check",
)
async def health_check() -> HealthResponse:
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
    """Manual trigger to rebuild TF-IDF matrix."""
    try:
        await recommender_service.refresh_cache()
        return {"success": True, "message": "AI matrix rebuilt successfully!"}
    except Exception as exc:
        logger.exception("Manual refresh failed: %s", exc)
        return JSONResponse(status_code=500, content={"success": False, "detail": str(exc)})


# ---------------------------------------------------------------------------
# Global exception handler (hide stack traces in production)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
