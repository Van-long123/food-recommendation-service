"""
Các Pydantic models(Định nghĩa cấu trúc dữ liệu và validate dữ liệu) dành cho tài liệu Sản phẩm từ MongoDB.
Được thiết kế tương thích với schema productModel của Node.js.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class Ratings(BaseModel):
    """Cấu trúc dữ liệu đánh giá sản phẩm."""
    totalRating: float = 0.0
    numberOfRatings: int = 0


class ProductBase(BaseModel):
    """
    Dữ liệu sản phẩm tối thiểu lấy từ MongoDB để phục vụ tính toán gợi ý.
    """

    id: str = Field(alias="_id")  # Map trường _id của MongoDB vào id
    title: str
    slug: Optional[str] = None
    description: Optional[str] = ""
    unit: Optional[str] = None
    thumbnail: Optional[str] = None
    images: Optional[List[str]] = []
    stock: int = 0
    price: float = 0.0
    discountPercentage: float = 0.0
    originalPrice: float = 0.0
    # status: "active" | "inactive"
    status: Optional[str] = None
    featured: bool = False
    isBestPrice: bool = False
    isOnlineExclusive: bool = False
    tags: Optional[List[str]] = []
    ratings: Ratings = Field(default_factory=Ratings)
    primary_category_id: Optional[str] = None
    deleted: bool = False

    @field_validator("id", mode="before")
    @classmethod
    def coerce_object_id(cls, v: Any) -> str:
        """Chuyển đổi ObjectId (từ MongoDB) thành chuỗi string."""
        return str(v)

    @field_validator("primary_category_id", mode="before")
    @classmethod
    def coerce_category_id(cls, v: Any) -> Optional[str]:
        """Chuyển đổi ID danh mục thành chuỗi string, chấp nhận None."""
        if v is None:
            return None
        return str(v)

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v: Any) -> List[str]:
        """Đảm bảo trường tags luôn là một danh sách chuỗi."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(t) for t in v]
        return [str(v)]

    @field_validator("images", mode="before")
    @classmethod
    def ensure_images_list(cls, v: Any) -> List[str]:
        """Đảm bảo trường images luôn là một danh sách chuỗi."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(i) for i in v]
        return []

    model_config = {"populate_by_name": True}


class RecommendationItem(BaseModel):
    """
    Cấu trúc của một sản phẩm gợi ý trả về cho Client.
    Bao gồm thêm điểm số tương đồng (similarity_score).
    """

    id: str = Field(alias="_id")
    title: str
    slug: Optional[str] = None
    price: float
    unit: Optional[str] = None
    images: List[str] = []
    ratings: Ratings
    primary_category_id: Optional[str] = None
    featured: bool = False
    isBestPrice: bool = False
    isOnlineExclusive: bool = False
    similarity_score: float

    model_config = {"populate_by_name": True}


class RecommendationResponse(BaseModel):
    """Cấu trúc phản hồi API cho endpoint gợi ý sản phẩm."""

    success: bool = True
    product_id: str
    total: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    """Cấu trúc phản hồi kiểm tra trạng thái dịch vụ (Health Check)."""
    status: str
    products_cached: int
    matrix_built: bool
    cache_age_minutes: float
