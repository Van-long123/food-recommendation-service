# 🥗 Food Recommendation Service

A Python microservice that delivers **Content-Based Filtering** product recommendations using **TF-IDF + Cosine Similarity**. It is designed to slot into the SmartFood e-commerce architecture:

```
Nuxt 3 Frontend → Node.js API → Python Recommendation Service → MongoDB Atlas
```

---

## 📋 Table of Contents

1. [Quick Start (local)](#1-quick-start-local)
2. [Run with Docker](#2-run-with-docker)
3. [Node.js Integration](#3-nodejs-integration)
4. [How the Algorithm Works](#4-how-the-algorithm-works)
5. [Tuning the Weights](#5-tuning-the-weights)
6. [API Reference](#6-api-reference)
7. [Project Structure](#7-project-structure)

---

## 1. Quick Start (local)

### Prerequisites
- Python 3.11+
- MongoDB Atlas (or local MongoDB)

### Steps

```bash
# 1. Enter the service directory
cd recommendation-service

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
# Then edit .env and set MONGODB_URI, MONGODB_DB_NAME, etc.

# 5. Start the service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service is now available at **http://localhost:8000**

- Swagger UI: http://localhost:8000/docs  
- ReDoc:      http://localhost:8000/redoc  
- Health:     http://localhost:8000/health  

---

## 2. Run with Docker

```bash
# Build the image
docker build -t food-recommendation-service .

# Run the container
docker run -d \
  --name rec-service \
  -p 8000:8000 \
  --env-file .env \
  food-recommendation-service
```

### docker-compose (recommended for integration)

```yaml
version: "3.9"

services:
  recommendation-service:
    build: ./recommendation-service
    ports:
      - "8000:8000"
    env_file:
      - ./recommendation-service/.env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 3. Node.js Integration

### Full Flow

```
GET /api/products/:id/recommendations (Node.js)
  └─► GET http://localhost:8000/api/product-recommendation?product_id=:id&limit=8 (Python)
        └─► MongoDB Atlas (products collection)
```

### Install axios (if not already present)

```bash
npm install axios
```

### Node.js controller example

```js
// src/controllers/productController.js
const axios = require('axios')

const RECOMMENDATION_SERVICE_URL =
  process.env.RECOMMENDATION_SERVICE_URL || 'http://localhost:8000'

exports.getProductRecommendations = async (req, res) => {
  const { id } = req.params
  const limit = parseInt(req.query.limit) || 8

  try {
    const { data } = await axios.get(
      `${RECOMMENDATION_SERVICE_URL}/api/product-recommendation`,
      {
        params: { product_id: id, limit, category_boost: true },
        timeout: 5000,
      }
    )
    return res.json(data)
  } catch (error) {
    if (error.response?.status === 404) {
      return res.status(404).json({ message: 'Product not found' })
    }
    console.error('Recommendation service error:', error.message)
    return res.status(500).json({ message: 'Could not fetch recommendations' })
  }
}
```

### Node.js route

```js
// src/routes/productRoutes.js
const express = require('express')
const router = express.Router()
const { getProductRecommendations } = require('../controllers/productController')

router.get('/:id/recommendations', getProductRecommendations)

module.exports = router
```

### Nuxt 3 composable

```ts
// composables/useRecommendations.ts
export function useRecommendations(productId: string) {
  const { data, pending, error } = useFetch(
    `/api/products/${productId}/recommendations`,
    { query: { limit: 8 } }
  )
  return { recommendations: data, loading: pending, error }
}
```

### Nuxt 3 component

```vue
<!-- components/ProductRecommendations.vue -->
<script setup lang="ts">
const props = defineProps<{ productId: string }>()
const { recommendations, loading } = useRecommendations(props.productId)
</script>

<template>
  <section v-if="!loading && recommendations?.recommendations?.length">
    <h2>Sản phẩm tương tự</h2>
    <div class="product-grid">
      <ProductCard
        v-for="item in recommendations.recommendations"
        :key="item._id"
        :product="item"
      />
    </div>
  </section>
</template>
```

---

## 4. How the Algorithm Works

### Step 1 — Build Text Corpus

For each product a single string is assembled with **field weighting**:

| Field | Repetitions | Reason |
|---|---|---|
| `title` | × 3 | Most discriminative signal |
| `tags` | × 2 | Structured semantic keywords |
| `primary_category_id` | × 2 | Category membership |
| `description` | × 1 | Supporting context |

### Step 2 — TF-IDF Vectorisation

Each corpus string is converted to a sparse vector using `sklearn.TfidfVectorizer`:

- **TF-IDF** = Term Frequency × Inverse Document Frequency  
  → common words across all products score low, rare words score high.
- **`sublinear_tf=True`** applies `log(1 + tf)` to dampen very frequent terms.
- **`ngram_range=(1, 2)`** captures both single words and two-word phrases.
- `stop_words=None` because the corpus contains Vietnamese text.

### Step 3 — Cosine Similarity

```
similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
```

Two TF-IDF vectors with the same term distributions → similarity close to 1.  
Orthogonal vectors (no shared terms) → similarity = 0.

### Step 4 — Weighted Final Score

```
final_score = 0.60 × content_score        (cosine similarity)
            + 0.20 × price_similarity      (Gaussian proximity)
            + 0.10 × rating_score          (Bayesian quality score)
            + 0.10 × bonus_score           (featured / bestPrice / exclusive flags)
```

**Price similarity** penalises products far from the target price:
```
price_sim_i = exp(−|price_i − target_price| / target_price)
```

**Rating score** rewards well-rated products with many reviews:
```
rating_score_i = (totalRating_i / 5) × log(numRatings_i + 1) / log(maxNumRatings + 1)
```

### Step 5 — Optional Category Boost

When `category_boost=true` and the target product has a `primary_category_id`, products in the same category receive a ×1.05 multiplier on their final score (capped at 1.0).

---

## 5. Tuning the Weights

All weights are class-level constants in `app/services/recommender.py`:

```python
class RecommenderService:
    # Main scoring weights (must sum to 1.0)
    W_CONTENT: float = 0.6   # TF-IDF cosine similarity
    W_PRICE:   float = 0.2   # Price proximity
    W_RATING:  float = 0.1   # Rating quality
    W_BONUS:   float = 0.1   # Flag bonuses

    # Flag bonuses (within the W_BONUS budget)
    BONUS_FEATURED:      float = 0.05
    BONUS_BEST_PRICE:    float = 0.03
    BONUS_ONLINE_EXCL:   float = 0.02
```

> **Tip:** If you want to de-emphasise price and rely more on content, try  
> `W_CONTENT=0.7, W_PRICE=0.1, W_RATING=0.1, W_BONUS=0.1`.

**TF-IDF parameters** are in `build_tfidf_matrix()`:

```python
TfidfVectorizer(
    max_features=5000,   # increase for larger/richer catalogues
    ngram_range=(1, 2),  # (1,3) to capture longer phrases
    min_df=1,            # ignore terms in fewer than min_df docs
    sublinear_tf=True,
)
```

---

## 6. API Reference

### `GET /api/product-recommendation`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `product_id` | string | ✅ | — | MongoDB `_id` of the reference product |
| `limit` | int | ❌ | 8 | Max results (1–20) |
| `category_boost` | bool | ❌ | true | Boost same-category products |

**200 OK**
```json
{
  "success": true,
  "product_id": "6638abc123def456",
  "total": 8,
  "recommendations": [
    {
      "_id": "6638xyz789",
      "title": "Thịt bò Úc nhập khẩu 500g",
      "slug": "thit-bo-uc-nhap-khau-500g",
      "price": 185000,
      "images": ["https://..."],
      "ratings": { "totalRating": 4.7, "numberOfRatings": 213 },
      "primary_category_id": "64abc...",
      "featured": true,
      "isBestPrice": false,
      "isOnlineExclusive": false,
      "similarity_score": 0.8734
    }
  ]
}
```

**404 Not Found** — `product_id` not in the active catalogue.  
**422 Unprocessable Entity** — invalid query parameters.  
**500 Internal Server Error** — generic error (no stack trace exposed).

---

### `GET /health`

```json
{
  "status": "ok",
  "products_cached": 1243,
  "matrix_built": true,
  "cache_age_minutes": 4.5
}
```

---

## 7. Project Structure

```
recommendation-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, CORS, health check
│   ├── config.py            # pydantic-settings configuration
│   ├── database.py          # motor async MongoDB connection
│   ├── models/
│   │   ├── __init__.py
│   │   └── product.py       # Pydantic schemas (ProductBase, RecommendationResponse)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── recommender.py   # TF-IDF + Cosine Similarity engine
│   │   └── cache.py         # Async in-memory cache with TTL
│   └── routers/
│       ├── __init__.py
│       └── recommendation.py  # GET /api/product-recommendation
├── requirements.txt
├── .env                     # Your local secrets (not committed)
├── .env.example             # Template for .env
├── Dockerfile
└── README.md
```
