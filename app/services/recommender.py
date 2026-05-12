"""
Content-Based Filtering recommendation engine.
Uses TF-IDF + Cosine Similarity with weighted scoring.
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

# Cache keys
CACHE_KEY_PRODUCTS = "products_all"
CACHE_KEY_TFIDF = "tfidf_matrix"

FOOD_STOP_WORDS = {
    "tươi",
    "ngon",
    "sạch",
    "đặc",
    "biệt",
    "cao",
    "cấp",
    "kg",
    "gram",
    "g",
    "hộp",
    "chai",
    "lon",
    "fresh",
    "premium",
    "organic",
}


class RecommenderService:
    """
    Content-Based Filtering recommender.

    Scoring formula (weighted sum):
            final_score = (tier_boost)
                                    + 1.00 * title_score
                                    + 0.70 * content_score
                                    + 0.12 * price_similarity
                                    + 0.06 * rating_score
                                    + 0.04 * popularity_score
                                    + 0.03 * bonus_score

    Tier logic:
        TIER 1 (+10.0): cosine_similarity >= SIMILARITY_THRESHOLD
                        HOẶC title Jaccard similarity >= TITLE_SIMILARITY_THRESHOLD.
                        → Ưu tiên cao nhất, không phân biệt category.
        TIER 2 (+0.0):  Fallback cùng category để bù đủ số lượng khi thiếu TIER 1.
        LOẠI  (-1.0):   Không đủ ngưỡng similarity và khác category.
    """

    # Weight configuration (easy to tune)
    W_TITLE: float = 1.0
    W_CONTENT: float = 0.70
    W_PRICE: float = 0.12
    W_RATING: float = 0.06
    W_POPULARITY: float = 0.04
    W_BONUS: float = 0.03

    BONUS_FEATURED: float = 0.05
    BONUS_BEST_PRICE: float = 0.03
    BONUS_ONLINE_EXCL: float = 0.02

    # Tier boost
    TIER_1_BOOST: float = 10.0
    TIER_2_BOOST: float = 0.0

    # Minimum cosine similarity to be considered "content similar".
    SIMILARITY_THRESHOLD: float = 0.25

    # Minimum Jaccard title similarity to be considered "title similar".
    # Cho phép bắt các sản phẩm cùng loại nhưng khác category
    # (vd: "Thịt Ba Chỉ Heo" ↔ "Sườn Cốt Lết Heo" đều có "heo").
    TITLE_SIMILARITY_THRESHOLD: float = 0.15

    # ------------------------------------------------------------------
    # Corpus building
    # ------------------------------------------------------------------

    def build_corpus(self, products: List[ProductBase]) -> List[str]:
        print(">>> DANG TINH TOAN AI <<<")
        """
        Build corpus using only `title`, `description`, `unit` and a price token.

        Weights:
            title x4, description x2, unit x1, price token x1

        Price is converted to a coarse token so numeric differences are
        represented as categorical text for TF-IDF.
        """
        def _price_token(price_val: float) -> str:
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

            parts = (
                (title + " ") * 4
                + (description + " ") * 2
                + (unit_token + " ")
                + (price_token)
            )
            corpus.append(parts.strip())
        return corpus

    # ------------------------------------------------------------------
    # TF-IDF matrix
    # ------------------------------------------------------------------

    def build_tfidf_matrix(
        self, corpus: List[str]
    ) -> Tuple[TfidfVectorizer, spmatrix]:
        """
        Fit a TF-IDF vectorizer on *corpus* and return (vectorizer, matrix).

        stop_words=None because the corpus may contain Vietnamese text.
        """
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix

    # ------------------------------------------------------------------
    # Food type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _product_text(p: ProductBase) -> str:
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
        return None

    @classmethod
    def compute_food_type_score(
        cls, products: List[ProductBase], target_product: ProductBase
    ) -> Tuple[np.ndarray, Optional[str]]:
        target_text = cls._product_text(target_product)
        target_type = cls._detect_food_type(target_text)
        if not target_type:
            return np.zeros(len(products), dtype=float), None

        scores = np.zeros(len(products), dtype=float)
        for i, p in enumerate(products):
            if cls._detect_food_type(cls._product_text(p)) == target_type:
                scores[i] = 1.0
        return scores, target_type

    # ------------------------------------------------------------------
    # Auxiliary scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_price_similarity(
        products: List[ProductBase], target_price: float
    ) -> np.ndarray:
        """
        Gaussian-like similarity based on price proximity.
        score_i = exp(-|price_i - target_price| / max(target_price, 1))
        Normalised to [0, 1].
        """
        if target_price <= 0:
            return np.ones(len(products))

        prices = np.array([p.price for p in products], dtype=float)
        scores = np.exp(-np.abs(prices - target_price) / target_price)
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return np.ones(len(products))
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_rating_score(products: List[ProductBase]) -> np.ndarray:
        """
        Bayesian-style rating score.
        score_i = (totalRating_i / 5) * log(numRatings_i + 1) / log(max_numRatings + 1)
        Normalised to [0, 1].
        """
        total_ratings = np.array(
            [p.ratings.totalRating for p in products], dtype=float
        )
        num_ratings = np.array(
            [p.ratings.numberOfRatings for p in products], dtype=float
        )

        max_num = num_ratings.max()
        log_denom = math.log(max_num + 1) if max_num > 0 else 1.0

        scores = (total_ratings / 5.0) * (np.log(num_ratings + 1) / log_denom)
        rng = scores.max() - scores.min()
        if rng < 1e-9:
            return np.ones(len(products))
        return (scores - scores.min()) / rng

    @staticmethod
    def compute_popularity_score(products: List[ProductBase]) -> np.ndarray:
        """
        Popularity score based on number of ratings.
        score_i = log(numRatings_i + 1) / log(max_numRatings + 1)
        Normalised to [0, 1].
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
        """Additive bonus for featured / isBestPrice / isOnlineExclusive flags."""
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
        Compute Jaccard similarity between the target title and candidate titles
        to explicitly boost products with similar names.
        """
        target_words = set((target_title or "").lower().split())
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

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_products(self) -> List[ProductBase]:
        """Fetch active, non-deleted products from MongoDB."""
        collection = get_collection()

        query_filter = {
            "deleted": {"$ne": True},
            "status": "active",
        }

        projection = {
            "_id": 1,
            "title": 1,
            "slug": 1,
            "description": 1,
            "unit": 1,
            "images": 1,
            "thumbnail": 1,
            "stock": 1,
            "price": 1,
            "discountPercentage": 1,
            "originalPrice": 1,
            "status": 1,
            "deleted": 1,
            "ratings": 1,
            "primary_category_id": 1,
            "featured": 1,
            "isBestPrice": 1,
            "isOnlineExclusive": 1,
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
                    "Skipping malformed product %s: %s", doc.get("_id"), exc
                )

        logger.info("Loaded %d products from MongoDB", len(products))
        return products

    # ------------------------------------------------------------------
    # Cache-aware matrix build
    # ------------------------------------------------------------------

    async def _get_products_and_matrix(
        self,
    ) -> Tuple[List[ProductBase], spmatrix]:
        """Return (products, tfidf_matrix), using cache when available."""
        products: Optional[List[ProductBase]] = await cache_service.get(
            CACHE_KEY_PRODUCTS
        )
        matrix_data: Optional[Tuple[TfidfVectorizer, spmatrix]] = (
            await cache_service.get(CACHE_KEY_TFIDF)
        )

        if products is not None and matrix_data is not None:
            _, matrix = matrix_data
            return products, matrix

        # Cache miss — rebuild everything
        products = await self._load_products()
        if not products:
            return [], None  # type: ignore[return-value]

        t0 = time.perf_counter()
        corpus = self.build_corpus(products)
        vectorizer, matrix = self.build_tfidf_matrix(corpus)
        elapsed = time.perf_counter() - t0
        logger.info(
            "TF-IDF matrix built in %.3fs | shape=%s", elapsed, matrix.shape
        )

        ttl = settings.CACHE_TTL_MINUTES * 60
        await cache_service.set(CACHE_KEY_PRODUCTS, products, ttl_seconds=ttl)
        await cache_service.set(
            CACHE_KEY_TFIDF, (vectorizer, matrix), ttl_seconds=ttl
        )

        return products, matrix

    # ------------------------------------------------------------------
    # Main recommendation method
    # ------------------------------------------------------------------

    async def get_recommendations(
        self,
        product_id: str,
        top_n: int = 8,
        category_boost: bool = True,
    ) -> Tuple[List[dict], bool]:
        """
        Return (recommendations, product_found).

        Tier assignment:

        TIER 1: cosine_similarity >= SIMILARITY_THRESHOLD
            HOẶC title Jaccard similarity >= TITLE_SIMILARITY_THRESHOLD.
            → Ưu tiên cao nhất, không phân biệt category.
        TIER 2: fallback cùng category khi thiếu TIER 1.
        LOẠI  : Không đủ cả hai ngưỡng và khác category.
        """
        t_start = time.perf_counter()

        products, matrix = await self._get_products_and_matrix()

        if not products or matrix is None:
            logger.warning("No products available for recommendations")
            return [], False

        # Find target product index
        id_to_idx = {p.id: i for i, p in enumerate(products)}
        target_idx = id_to_idx.get(product_id)

        if target_idx is None:
            logger.warning(
                "product_id=%s not found in cached products", product_id
            )
            return [], False

        target_product = products[target_idx]

        # ---- Cosine similarity (content — primary signal) ----
        target_vec = matrix[target_idx]
        cosine_scores = cosine_similarity(target_vec, matrix).flatten()

        # ---- Auxiliary scores ----
        title_scores = self.compute_title_similarity(target_product.title, products)
        price_scores = self.compute_price_similarity(products, target_product.price)
        rating_scores = self.compute_rating_score(products)
        popularity_scores = self.compute_popularity_score(products)
        bonus_scores = self.compute_bonus_score(products)

        # ---- Base score ----
        base_scores = (
            self.W_TITLE * title_scores
            + self.W_CONTENT * cosine_scores
            + self.W_PRICE * price_scores
            + self.W_RATING * rating_scores
            + self.W_POPULARITY * popularity_scores
            + self.W_BONUS * bonus_scores
        )

        # Initialise all scores to -1 (excluded)
        final_scores = np.full(len(products), -1.0, dtype=float)

        target_category = getattr(target_product, "primary_category_id", None)

        # ---- Tier assignment ----
        for i, p in enumerate(products):
            if i == target_idx:
                continue

            # Chỉ recommend sản phẩm còn hàng
            if getattr(p, "stock", 0) <= 0:
                continue

            is_content_similar = cosine_scores[i] >= self.SIMILARITY_THRESHOLD
            is_title_similar = title_scores[i] >= self.TITLE_SIMILARITY_THRESHOLD

            if is_content_similar or is_title_similar:
                # Tier 1: đủ similar về nội dung HOẶC tên sản phẩm
                # → không phân biệt category
                final_scores[i] = base_scores[i] + self.TIER_1_BOOST
            elif (
                category_boost
                and target_category
                and getattr(p, "primary_category_id", None) == target_category
            ):
                # Tier 2: fallback cùng category nếu chưa đủ số lượng
                final_scores[i] = base_scores[i] + self.TIER_2_BOOST
            # else: loại (final_scores[i] giữ nguyên -1)

        # ---- Top-N ----
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

    # ------------------------------------------------------------------
    # Forced refresh (used by background task)
    # ------------------------------------------------------------------

    async def refresh_cache(self) -> None:
        """Invalidate cache and rebuild products + TF-IDF matrix."""
        logger.info(
            "Background refresh: clearing cache and rebuilding TF-IDF matrix..."
        )
        await cache_service.invalidate(CACHE_KEY_PRODUCTS)
        await cache_service.invalidate(CACHE_KEY_TFIDF)
        await self._get_products_and_matrix()
        logger.info("Background refresh complete.")


# Singleton instance
recommender_service = RecommenderService()