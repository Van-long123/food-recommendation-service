"""
Router định nghĩa các API endpoint cho chức năng gợi ý sản phẩm.
"""
import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.product import RecommendationResponse, RecommendationItem, Ratings
from app.services.recommender import recommender_service

logger = logging.getLogger(__name__)

# Khởi tạo router với tiền tố /api và gắn tag để phân loại trong tài liệu Swagger
router = APIRouter(prefix="/api", tags=["recommendations"])


@router.get(
    "/product-recommendation",
    response_model=RecommendationResponse,
    summary="Lấy danh sách sản phẩm gợi ý dựa trên nội dung",
    responses={
        200: {"description": "Danh sách các sản phẩm gợi ý tương đồng"},
        404: {"description": "Không tìm thấy sản phẩm mục tiêu"},
        422: {"description": "Lỗi validate dữ liệu đầu vào"},
        500: {"description": "Lỗi server nội bộ"},
    },
)
async def get_recommendations(
    product_id: Annotated[
        str,
        Query(
            description="ID của sản phẩm đang được xem để tìm sản phẩm tương tự",
            min_length=1,
        ),
    ],
    limit: Annotated[
        int,
        Query(description="Số lượng sản phẩm gợi ý muốn lấy", ge=1, le=20),
    ] = 8,
    category_boost: Annotated[
        bool,
        Query(description="Có ưu tiên các sản phẩm cùng danh mục hay không"),
    ] = True,
) -> RecommendationResponse:
    """
    Trả về danh sách *limit* sản phẩm có nội dung tương đồng nhất với *product_id*.
    Sử dụng thuật toán TF-IDF + Cosine Similarity kết hợp với chấm điểm trọng số.
    """
    try:
        # Gọi service để thực hiện tính toán gợi ý
        results, found = await recommender_service.get_recommendations(
            product_id=product_id,
            top_n=limit,
            category_boost=category_boost,
        )
    except Exception as exc:
        logger.exception("Lỗi không xác định trong get_recommendations: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Đã xảy ra lỗi hệ thống khi tính toán gợi ý sản phẩm.",
        ) from exc

    # Nếu không tìm thấy sản phẩm mục tiêu trong database
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Sản phẩm với id '{product_id}' không tồn tại trong danh mục hoạt động.",
        )

    # Chuyển đổi dữ liệu thô từ service sang cấu hình Pydantic model
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

    # Trả về kết quả cuối cùng theo đúng cấu trúc response mong muốn
    return RecommendationResponse(
        success=True,
        product_id=product_id,
        total=len(recommendation_items),
        recommendations=recommendation_items,
    )
