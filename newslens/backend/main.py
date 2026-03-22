"""
main.py — NewsLens FastAPI Backend
------------------------------------
Exposes a /search endpoint that:
  1. Embeds the user's natural-language query
  2. Queries Endee for nearest-neighbor articles
  3. Returns ranked results with title, snippet, category, score

Endpoints:
  GET /           → health check
  GET /search     → semantic search  (?q=your+query&top_k=5)
  GET /stats      → index statistics

Run locally (outside Docker):
  uvicorn main:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from endee import Endee

from embedder import embed_text, EMBEDDING_DIM

load_dotenv()

# ─────────────────────────────────────────────
# Config (from environment / .env)
# ─────────────────────────────────────────────
ENDEE_HOST       = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME       = os.getenv("ENDEE_INDEX_NAME", "news_articles")

# ─────────────────────────────────────────────
# Globals (set during startup)
# ─────────────────────────────────────────────
endee_client = None
news_index   = None


# ─────────────────────────────────────────────
# Lifespan: connect to Endee on startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global endee_client, news_index

    print(f"[startup] Connecting to Endee at {ENDEE_HOST}")
    endee_client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    endee_client.set_base_url(f"{ENDEE_HOST}/api/v1")

    # Verify the index exists (must run indexer first)
    try:
        news_index = endee_client.get_index(INDEX_NAME)
        print(f"[startup] Connected to index '{INDEX_NAME}'")
    except Exception as e:
        print(f"[startup] WARNING: Index '{INDEX_NAME}' not found. Run the indexer first.")
        print(f"[startup] Error: {e}")

    yield  # app runs here

    print("[shutdown] Closing Endee connection")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="NewsLens Semantic Search API",
    description="Search 30+ news articles by meaning, not just keywords. Powered by Endee vector database.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the local frontend HTML file to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────
class ArticleResult(BaseModel):
    id: str
    title: str
    snippet: str
    category: str
    source: str
    url: str
    similarity_score: float  # 0.0 – 1.0 cosine similarity


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[ArticleResult]


class StatsResponse(BaseModel):
    index_name: str
    total_vectors: int
    dimension: int
    space_type: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "service": "NewsLens Semantic Search",
        "index": INDEX_NAME,
        "endee_host": ENDEE_HOST,
    }


@app.get("/search", response_model=SearchResponse, summary="Semantic search over news articles")
def search(
    q: str = Query(..., description="Natural language search query", min_length=2),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
):
    """
    Perform semantic search over indexed news articles.

    The query is embedded using the same sentence-transformer model
    used during indexing, then Endee finds the nearest-neighbor vectors
    using cosine similarity.

    Example queries:
    - "climate change and sea level"
    - "breakthroughs in cancer treatment"
    - "stock market and inflation"
    - "electric vehicles and battery technology"
    """
    if news_index is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not available. Please run the indexer script first.",
        )

    # 1. Embed the query
    try:
        query_vector = embed_text(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 2. Query Endee for nearest neighbors
    try:
        raw_results = news_index.query(vector=query_vector, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Endee query failed: {e}")

    # 3. Format results
    articles = []
    for r in raw_results:
        meta = r.meta or {}
        body = meta.get("body", "")
        snippet = body[:200].rstrip() + "…" if len(body) > 200 else body

        articles.append(
            ArticleResult(
                id=r.id,
                title=meta.get("title", "Untitled"),
                snippet=snippet,
                category=meta.get("category", "General"),
                source=meta.get("source", "Unknown"),
                url=meta.get("url", "#"),
                similarity_score=round(float(r.similarity), 4),
            )
        )

    return SearchResponse(query=q, total_results=len(articles), results=articles)


@app.get("/stats", response_model=StatsResponse, summary="Index statistics")
def stats():
    """Return basic statistics about the Endee index."""
    if news_index is None:
        raise HTTPException(status_code=503, detail="Index not available.")

    try:
        info = news_index.info()
        return StatsResponse(
            index_name=INDEX_NAME,
            total_vectors=info.get("count", 0),
            dimension=EMBEDDING_DIM,
            space_type="cosine",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch stats: {e}")
