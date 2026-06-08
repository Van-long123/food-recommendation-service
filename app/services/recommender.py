"""
Công cụ gợi ý sản phẩm dựa trên nội dung (Content-Based Filtering).
Sử dụng phương pháp TF-IDF kết hợp Độ tương đồng Cosine (Cosine Similarity) và chấm điểm theo trọng số.
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

from app.config import settings
from app.database import get_collection
from app.models.product import ProductBase, Ratings
from app.services.cache import cache_service

logger = logging.getLogger(__name__)

# Các khóa dùng cho Cache
CACHE_KEY_PRODUCTS = "products_all"
CACHE_KEY_TFIDF = "tfidf_matrix"

# Các từ khóa phổ biến trong thực phẩm cần bỏ qua khi so sánh tiêu đề để tránh khớp sai.
FOOD_STOP_WORDS = {
    "tươi", "ngon", "sạch", "đặc", "biệt", "cao", "cấp",
    "kg", "gram", "g", "hộp", "chai", "lon",
    "fresh", "premium", "organic",
}


class RecommenderService:
    """
    Dịch vụ gợi ý dựa trên nội dung sản phẩm.

    Công thức tính điểm (tổng trọng số):
            final_score = (tier_boost)
                                    + 1.00 * title_score        (Độ tương đồng tiêu đề)
                                    + 0.70 * content_score      (Độ tương đồng nội dung TF-IDF)
                                    + 0.12 * price_similarity   (Sự tương đồng về giá)
                                    + 0.06 * rating_score       (Điểm đánh giá)
                                    + 0.04 * popularity_score   (Độ phổ biến)
                                    + 0.03 * bonus_score        (Điểm thưởng đặc biệt)

    Logic phân tầng (Tier):
        TIER 1 (+10.0): Độ tương đồng nội dung >= SIMILARITY_THRESHOLD
                        HOẶC tiêu đề khớp Jaccard >= TITLE_SIMILARITY_THRESHOLD.
                        → Ưu tiên cao nhất, không phân biệt danh mục.
        TIER 2 (+0.0):  Dành cho sản phẩm cùng danh mục khi không đủ sản phẩm Tier 1.
        LOẠI  (-1.0):   Không đủ ngưỡng tương đồng và khác danh mục.
    """

    # Cấu hình trọng số (Dễ dàng điều chỉnh)
    W_TITLE: float = 1.0
    W_CONTENT: float = 0.70
    W_PRICE: float = 0.12
    W_RATING: float = 0.06
    W_POPULARITY: float = 0.04
    W_BONUS: float = 0.03

    # Điểm thưởng cho các thuộc tính đặc biệt
    BONUS_FEATURED: float = 0.05
    BONUS_BEST_PRICE: float = 0.03
    BONUS_ONLINE_EXCL: float = 0.02

    # Điểm cộng thêm để đảm bảo thứ tự giữa các tầng
    TIER_1_BOOST: float = 10.0
    TIER_2_BOOST: float = 0.0

    # Ngưỡng tối thiểu để được coi là tương đồng nội dung
    SIMILARITY_THRESHOLD: float = 0.25

    # Ngưỡng tối thiểu để được coi là tương đồng tiêu đề (Jaccard)
    TITLE_SIMILARITY_THRESHOLD: float = 0.15

    # Xây dựng Corpus (Tập văn bản)
    # Giai đoạn "Học"
    def build_corpus(self, products: List[ProductBase]) -> List[str]:
        """
        Tạo tập văn bản từ `title`, `description`, `unit` và token giá.
        
        Trọng số trong văn bản:
            tiêu đề x4, mô tả x2, đơn vị x1, giá x1
        """
        def _price_token(price_val: float) -> str:
            """Chuyển đổi giá tiền thành các phân khúc văn bản để TF-IDF xử lý."""
            try:
                p = float(price_val)
            except Exception:
                return "price_unknown"
            if p <= 0:
                return "price_0"
            if p < 50000:
                return "price_lt_50k"
            if p < 100000:
                return "price_50k_100k"
            if p < 200000:
                return "price_100k_200k"
            if p < 500000:
                return "price_200k_500k"
            return "price_ge_500k"

        corpus: List[str] = []
        for p in products:
            title = (p.title or "").lower().strip()
            description = (p.description or "").lower().strip()
            unit = (getattr(p, "unit", "") or "").lower().strip()
            price_token = _price_token(getattr(p, "price", 0))

            unit_token = f"unit_{unit.replace(' ', '_')}" if unit else "unit_unknown"

            # Ghép các phần lại với trọng số khác nhau bằng cách lặp lại chuỗi
            parts = (
                (title + " ") * 4
                + (description + " ") * 2
                + (unit_token + " ")
                + (price_token)
            )
            corpus.append(parts.strip())
        return corpus

    # Ma trận TF-IDF
    # Giai đoạn "Tính toán"
    def build_tfidf_matrix(
        self, corpus: List[str]
    ) -> Tuple[TfidfVectorizer, spmatrix]:
        """
        Huấn luyện bộ Vectorizer trên *corpus* và trả về (vectorizer, matrix).
        Sử dụng ngram_range=(1, 2) để bắt được cả các cụm 2 từ (ví dụ: "thịt heo").
        """
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None, # Không dùng stop_words mặc định vì là tiếng Việt
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix

    # Phân loại loại thực phẩm (Mở rộng trong tương lai)

    @staticmethod
    def _product_text(p: ProductBase) -> str:
        """Tổng hợp toàn bộ thông tin văn bản của sản phẩm."""
        title = (p.title or "").lower().strip()
        description = (p.description or "").lower().strip()
        unit = (getattr(p, "unit", "") or "").lower().strip()
        price = getattr(p, "price", 0)

        unit_token = f"unit_{unit.replace(' ', '_')}" if unit else "unit_unknown"

        def _price_token_for_text(price_val: float) -> str:
            try:
                p = float(price_val)
            except Exception:
                return "price_unknown"
            if p <= 0:
                return "price_0"
            if p < 50000:
                return "price_lt_50k"
            if p < 100000:
                return "price_50k_100k"
            if p < 200000:
                return "price_100k_200k"
            if p < 500000:
                return "price_200k_500k"
            return "price_ge_500k"

        price_token = _price_token_for_text(price)
        return f"{title} {description} {unit_token} {price_token}".strip()

    @staticmethod
    def _detect_food_type(text: str) -> Optional[str]:
        """Phát hiện loại thực phẩm (Thịt, Cá, Rau...) - Hiện tại chưa dùng."""
        return None

    @classmethod
    def compute_food_type_score(
        cls, products: List[ProductBase], target_product: ProductBase
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Tính điểm dựa trên loại thực phẩm giống nhau."""
        target_text = cls._product_text(target_product)
        target_type = cls._detect_food_type(target_text)
        if not target_type:
            return np.zeros(len(products), dtype=float), None

        scores = np.zeros(len(products), dtype=float)
        for i, p in enumerate(products):
            if cls._detect_food_type(cls._product_text(p)) == target_type:
                scores[i] = 1.0
        return scores, target_type

    # Các hàm tính điểm phụ trợ (Scoring)

    @staticmethod
    def compute_price_similarity(
        products: List[ProductBase], target_price: float
    ) -> np.ndarray:
        """
        Tính độ tương đồng về giá bằng hàm Gaussian.
        Sản phẩm có giá càng gần với sản phẩm mục tiêu thì điểm càng cao.
        """
        if target_price <= 0:
            return np.ones(len(products))

        prices = np.array([p.price for p in products], dtype=float)
        # Công thức: exp(-|price_i - target_price| / target_price)
        scores = np.exp(-np.abs(prices - target_price) / target_price)
        
        # Chuẩn hóa về khoảng [0, 1]
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return np.ones(len(products))
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_rating_score(products: List[ProductBase]) -> np.ndarray:
        """
        Tính điểm đánh giá theo phong cách Bayesian.
        Kết hợp cả số điểm trung bình và số lượng đánh giá.
        """
        total_ratings = np.array(
            [p.ratings.totalRating for p in products], dtype=float
        )
        num_ratings = np.array(
            [p.ratings.numberOfRatings for p in products], dtype=float
        )

        max_num = num_ratings.max()
        log_denom = math.log(max_num + 1) if max_num > 0 else 1.0

        # Điểm = (Sao / 5) * log(Số lượt + 1)
        scores = (total_ratings / 5.0) * (np.log(num_ratings + 1) / log_denom)
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return np.ones(len(products))
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_popularity_score(products: List[ProductBase]) -> np.ndarray:
        """
        Tính điểm phổ biến dựa trên số lượng đánh giá.
        Sản phẩm được mua/đánh giá nhiều hơn sẽ có điểm cao hơn.
        """
        num_ratings = np.array(
            [p.ratings.numberOfRatings for p in products], dtype=float
        )
        max_num = num_ratings.max()
        log_denom = math.log(max_num + 1) if max_num > 0 else 1.0
        scores = np.log(num_ratings + 1) / log_denom
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return np.ones(len(products))
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_bonus_score(products: List[ProductBase]) -> np.ndarray:
        """Cộng điểm thưởng cho các cờ: featured (nổi bật), isBestPrice, isOnlineExclusive."""
        scores = np.zeros(len(products), dtype=float)
        for i, p in enumerate(products):
            if p.featured:
                scores[i] += RecommenderService.BONUS_FEATURED
            if p.isBestPrice:
                scores[i] += RecommenderService.BONUS_BEST_PRICE
            if p.isOnlineExclusive:
                scores[i] += RecommenderService.BONUS_ONLINE_EXCL
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return scores
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_title_similarity(target_title: str, products: List[ProductBase]) -> np.ndarray:
        """
        Tính độ tương đồng tiêu đề bằng chỉ số Jaccard (tỷ lệ từ chung / tổng số từ).
        Giúp bắt các sản phẩm có tên gọi giống nhau dù khác danh mục.
        """
        target_words = set((target_title or "").lower().split())
        # Loại bỏ các từ dừng thực phẩm (tươi, ngon, kg...)
        target_words = {w for w in target_words if w.strip()} - FOOD_STOP_WORDS
        if not target_words:
            return np.zeros(len(products), dtype=float)

        scores = np.zeros(len(products), dtype=float)
        for i, p in enumerate(products):
            p_words = set((p.title or "").lower().split())
            p_words = {w for w in p_words if w.strip()} - FOOD_STOP_WORDS
            if not p_words:
                continue
            intersection = len(target_words & p_words)
            union = len(target_words | p_words)
            scores[i] = intersection / union if union > 0 else 0.0

        return scores

    # Tải dữ liệu (Data loading)

    async def _load_products(self) -> List[ProductBase]:
        """Lấy danh sách sản phẩm còn hoạt động và chưa bị xóa từ MongoDB."""
        collection = get_collection()

        query_filter = {
            "deleted": {"$ne": True},
            "status": "active",
        }

        projection = {
            "_id": 1, "title": 1, "slug": 1, "description": 1, "unit": 1,
            "images": 1, "thumbnail": 1, "stock": 1, "price": 1,
            "discountPercentage": 1, "originalPrice": 1, "status": 1,
            "deleted": 1, "ratings": 1, "primary_category_id": 1,
            "featured": 1, "isBestPrice": 1, "isOnlineExclusive": 1,
        }

        cursor = collection.find(query_filter, projection).limit(
            settings.MAX_PRODUCTS_FETCH
        )

        products: List[ProductBase] = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            try:
                products.append(ProductBase(**doc))
            except Exception as exc:
                logger.warning(
                    "Bỏ qua sản phẩm lỗi %s: %s", doc.get("_id"), exc
                )

        logger.info("Đã tải %d sản phẩm từ MongoDB", len(products))
        return products

    # Xây dựng ma trận có sử dụng Cache

    async def _get_products_and_matrix(
        self,
    ) -> Tuple[List[ProductBase], spmatrix]:
        """Trả về (products, tfidf_matrix), ưu tiên lấy từ cache nếu có."""
        products: Optional[List[ProductBase]] = await cache_service.get(
            CACHE_KEY_PRODUCTS
        )
        matrix_data: Optional[Tuple[TfidfVectorizer, spmatrix]] = (
            await cache_service.get(CACHE_KEY_TFIDF)
        )

        if products is not None and matrix_data is not None:
            _, matrix = matrix_data
            return products, matrix

        # Nếu cache trống - Tiến hành tính toán lại từ đầu
        products = await self._load_products()
        if not products:
            return [], None  # type: ignore[return-value]

        t0 = time.perf_counter()
        corpus = self.build_corpus(products)
        vectorizer, matrix = self.build_tfidf_matrix(corpus)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Đã xây dựng ma trận TF-IDF trong %.3fs | shape=%s", elapsed, matrix.shape
        )

        # Lưu vào cache với thời gian TTL từ config
        ttl = settings.CACHE_TTL_MINUTES * 60
        await cache_service.set(CACHE_KEY_PRODUCTS, products, ttl_seconds=ttl)
        await cache_service.set(
            CACHE_KEY_TFIDF, (vectorizer, matrix), ttl_seconds=ttl
        )

        return products, matrix

    # Phương thức Gợi ý chính (Main recommendation)

    async def get_recommendations(
        self,
        product_id: str,
        top_n: int = 8,
        category_boost: bool = True,
    ) -> Tuple[List[dict], bool]:
        """
        Hàm chính để lấy danh sách gợi ý cho một sản phẩm.

        Quy trình:
        1. Lấy dữ liệu sản phẩm và ma trận TF-IDF (từ cache).
        2. Tính toán độ tương đồng nội dung (Cosine) và tiêu đề (Jaccard).
        3. Tính toán các điểm phụ trợ (Giá, Đánh giá, Phổ biến).
        4. Tổng hợp điểm cuối cùng theo trọng số.
        5. Phân tầng sản phẩm (Tier 1: Rất giống, Tier 2: Cùng danh mục).
        6. Trả về top N sản phẩm có điểm cao nhất.
        """
        t_start = time.perf_counter()

        products, matrix = await self._get_products_and_matrix()

        if not products or matrix is None:
            logger.warning("Không có sản phẩm nào để tính toán gợi ý")
            return [], False

        # Tìm chỉ mục (index) của sản phẩm mục tiêu
        id_to_idx = {p.id: i for i, p in enumerate(products)}
        target_idx = id_to_idx.get(product_id)

        if target_idx is None:
            logger.warning(
                "product_id=%s không tìm thấy trong danh sách đã nạp", product_id
            )
            return [], False

        target_product = products[target_idx]

        # ---- 1. Cosine similarity (Tương đồng nội dung - Tín hiệu chính) ----
        # Lấy vector của target product
        target_vec = matrix[target_idx]
        # Tính Cosine Similarity, So sánh sản phẩm hiện tại với toàn bộ sản phẩm khác
        cosine_scores = cosine_similarity(target_vec, matrix).flatten()

        # ---- 2. Tính các điểm số phụ trợ ----
        title_scores = self.compute_title_similarity(target_product.title, products)
        price_scores = self.compute_price_similarity(products, target_product.price)
        rating_scores = self.compute_rating_score(products)
        popularity_scores = self.compute_popularity_score(products)
        bonus_scores = self.compute_bonus_score(products)

        # ---- 3. Tính điểm nền (Base score) bằng tổng trọng số ----
        base_scores = (
            self.W_TITLE * title_scores
            + self.W_CONTENT * cosine_scores
            + self.W_PRICE * price_scores
            + self.W_RATING * rating_scores
            + self.W_POPULARITY * popularity_scores
            + self.W_BONUS * bonus_scores
        )

        # Khởi tạo mảng điểm cuối cùng với giá trị -1 (mặc định bị loại)
        final_scores = np.full(len(products), -1.0, dtype=float)

        target_category = getattr(target_product, "primary_category_id", None)

        # ---- 4. Phân tầng và Lọc sản phẩm (Tier assignment) ----
        for i, p in enumerate(products):
            if i == target_idx:
                continue

            # Chỉ gợi ý sản phẩm còn hàng
            if getattr(p, "stock", 0) <= 0:
                continue

            is_content_similar = cosine_scores[i] >= self.SIMILARITY_THRESHOLD
            is_title_similar = title_scores[i] >= self.TITLE_SIMILARITY_THRESHOLD

            if is_content_similar or is_title_similar:
                # Tier 1: Đủ tương đồng về nội dung HOẶC tiêu đề
                # → Được ưu tiên hàng đầu, bất kể danh mục nào
                final_scores[i] = base_scores[i] + self.TIER_1_BOOST
            elif (
                category_boost
                and target_category
                and getattr(p, "primary_category_id", None) == target_category
            ):
                # Tier 2: Fallback cho các sản phẩm cùng danh mục khi Tier 1 không đủ
                final_scores[i] = base_scores[i] + self.TIER_2_BOOST
            # else: bị loại (giữ nguyên điểm -1)

        # ---- 5. Lấy Top-N kết quả ----
        # Sắp xếp giảm dần và lấy N chỉ mục đầu tiên
        top_indices = np.argsort(final_scores)[::-1][:top_n]

        results: List[dict] = []
        for idx in top_indices:
            if final_scores[idx] < 0:
                continue
            p = products[idx]
            results.append(
                {
                    "_id": p.id,
                    "title": p.title,
                    "slug": p.slug,
                    "price": p.price,
                    "unit": getattr(p, "unit", None),
                    "images": p.images or [],
                    "thumbnail": getattr(p, "thumbnail", None),
                    "ratings": p.ratings.model_dump(),
                    "primary_category_id": getattr(p, "primary_category_id", None),
                    "featured": getattr(p, "featured", False),
                    "isBestPrice": getattr(p, "isBestPrice", False),
                    "isOnlineExclusive": getattr(p, "isOnlineExclusive", False),
                    "similarity_score": round(float(final_scores[idx]), 4),
                }
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Recommendation | product_id=%s | results=%d | time=%.1fms",
            product_id,
            len(results),
            elapsed_ms,
        )

        return results, True

    # Làm mới Cache (Dùng cho tác vụ chạy ngầm)

    async def refresh_cache(self) -> None:
        """Xóa cache cũ và xây dựng lại toàn bộ ma trận AI từ MongoDB."""
        logger.info(
            "Background refresh: Đang xóa cache và xây dựng lại ma trận TF-IDF..."
        )
        await cache_service.invalidate(CACHE_KEY_PRODUCTS)
        await cache_service.invalidate(CACHE_KEY_TFIDF)
        await self._get_products_and_matrix()
        logger.info("Background refresh hoàn tất.")


# Khởi tạo instance singleton để sử dụng toàn ứng dụng
recommender_service = RecommenderService()