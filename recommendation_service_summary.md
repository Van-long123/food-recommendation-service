# 🥗 Recommendation Service — Implementation Summary

## Project Structure

```
recommendation-service/
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI entry, lifespan, CORS, health
│   ├── config.py            ← pydantic-settings env config
│   ├── database.py          ← motor async MongoDB driver
│   ├── models/
│   │   └── product.py       ← ProductBase, RecommendationResponse (Pydantic)
│   ├── services/
│   │   ├── recommender.py   ← TF-IDF + Cosine Similarity engine
│   │   └── cache.py         ← Async in-memory TTL cache
│   └── routers/
│       └── recommendation.py ← GET /api/product-recommendation
├── requirements.txt
├── .env                     ← pre-filled with Atlas URI from Node.js project
├── .env.example
├── Dockerfile
└── README.md
```

## Key Design Decisions

### Schema Alignment with productModel.js
- Uses `deleted` (not `isDeleted`) and `status: "active"` (not `isActive: true`) — matching actual Node.js Joi schema
- MongoDB filter: `{ stock: {$gt: 0}, deleted: {$ne: true}, status: "active" }`

### Scoring Formula
| Component | Weight | Method |
|---|---|---|
| TF-IDF Cosine Similarity | 0.60 | sklearn, ngram (1,2), sublinear_tf |
| Price proximity | 0.20 | Gaussian: `exp(-|Δprice| / target)` |
| Rating quality | 0.10 | Bayesian: `(rating/5) × log(n+1)/log(max+1)` |
| Flag bonuses | 0.10 | featured +0.05, bestPrice +0.03, exclusive +0.02 |

### Cache Strategy
- Singleton `CacheService` with `asyncio.Lock` (thread-safe)
- Keys: `products_all` + `tfidf_matrix`
- TTL: 30 min (configurable via `CACHE_TTL_MINUTES`)
- On startup: warm-up loads products + builds matrix
- Background asyncio task refreshes every TTL interval

## How to Run

```bash
cd d:\Ky8\DATN\recommendation-service
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Integration with Node.js API

```js
// In Node.js product controller
const { data } = await axios.get('http://localhost:8000/api/product-recommendation', {
  params: { product_id: id, limit: 8, category_boost: true }
})
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/product-recommendation` | Get recommendations by `product_id` |
| GET | `/health` | Cache status + products count |
| GET | `/docs` | Swagger UI |
