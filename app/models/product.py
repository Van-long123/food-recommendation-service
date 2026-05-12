"""
Pydantic models for Product documents from MongoDB.
Aligned with the Node.js productModel schema.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class Ratings(BaseModel):
    totalRating: float = 0.0
    numberOfRatings: int = 0


class ProductBase(BaseModel):
    """Minimal projection returned from MongoDB for recommendations."""

    id: str = Field(alias="_id")
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
    # status field: "active" | "inactive"
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
        """Convert ObjectId or any type to string."""
        return str(v)

    @field_validator("primary_category_id", mode="before")
    @classmethod
    def coerce_category_id(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v)

    @field_validator("tags", mode="before")
    @classmethod
    def ensure_tags_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(t) for t in v]
        return [str(v)]

    @field_validator("images", mode="before")
    @classmethod
    def ensure_images_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(i) for i in v]
        return []

    model_config = {"populate_by_name": True}


class RecommendationItem(BaseModel):
    """A single recommendation result returned to the client."""

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
    """API response envelope for recommendation endpoint."""

    success: bool = True
    product_id: str
    total: int
    recommendations: List[RecommendationItem]


class HealthResponse(BaseModel):
    status: str
    products_cached: int
    matrix_built: bool
    cache_age_minutes: float
