"""
Router: GET /api/product-recommendation
"""
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.product import RecommendationResponse, RecommendationItem, Ratings
from app.services.recommender import recommender_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recommendations"])


@router.get(
    "/product-recommendation",
    response_model=RecommendationResponse,
    summary="Get content-based product recommendations",
    responses={
        200: {"description": "List of recommended products"},
        404: {"description": "Product not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def get_recommendations(
    product_id: Annotated[
        str,
        Query(
            description="ID of the product currently being viewed",
            min_length=1,
        ),
    ],
    limit: Annotated[
        int,
        Query(description="Number of recommendations to return", ge=1, le=20),
    ] = 8,
    category_boost: Annotated[
        bool,
        Query(description="Boost products in the same category"),
    ] = True,
) -> RecommendationResponse:
    """
    Returns *limit* products most similar to *product_id* using
    TF-IDF + Cosine Similarity with weighted scoring.
    """
    try:
        results, found = await recommender_service.get_recommendations(
            product_id=product_id,
            top_n=limit,
            category_boost=category_boost,
        )
    except Exception as exc:
        logger.exception("Unexpected error in get_recommendations: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while computing recommendations.",
        ) from exc

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Product with id '{product_id}' not found in the active product catalogue.",
        )

    recommendation_items = []
    for item in results:
        recommendation_items.append(
            RecommendationItem(
                _id=item["_id"],
                title=item["title"],
                slug=item.get("slug"),
                price=item["price"],
                images=item.get("images", []),
                ratings=Ratings(**item["ratings"]),
                primary_category_id=item.get("primary_category_id"),
                featured=item.get("featured", False),
                isBestPrice=item.get("isBestPrice", False),
                isOnlineExclusive=item.get("isOnlineExclusive", False),
                similarity_score=item["similarity_score"],
            )
        )

    return RecommendationResponse(
        success=True,
        product_id=product_id,
        total=len(recommendation_items),
        recommendations=recommendation_items,
    )
