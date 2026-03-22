"""
main.py — NewsLens FastAPI Backend
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from endee import Endee
from embedder import embed_text, EMBEDDING_DIM

load_dotenv()

ENDEE_HOST       = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME       = os.getenv("ENDEE_INDEX_NAME", "news_articles")

endee_client = None
news_index   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global endee_client, news_index

    print(f"[startup] Connecting to Endee at {ENDEE_HOST}")
    endee_client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    endee_client.set_base_url(f"{ENDEE_HOST}/api/v1")

    try:
        news_index = endee_client.get_index(INDEX_NAME)
        print(f"[startup] Connected to index '{INDEX_NAME}'")
    except Exception as e:
        print(f"[startup] WARNING: Index '{INDEX_NAME}' not found. Run indexer first. ({e})")

    yield
    print("[shutdown] Done.")


app = FastAPI(
    title="NewsLens Semantic Search API",
    description="Semantic news search powered by Endee vector database.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class ArticleResult(BaseModel):
    id: str
    title: str
    snippet: str
    category: str
    source: str
    url: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[ArticleResult]


class StatsResponse(BaseModel):
    index_name: str
    dimension: int
    space_type: str


@app.get("/")
def root():
    return {"status": "ok", "service": "NewsLens", "index": INDEX_NAME}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2),
    top_k: int = Query(default=5, ge=1, le=20),
):
    if news_index is None:
        raise HTTPException(status_code=503, detail="Index not ready. Run the indexer first.")

    try:
        query_vector = embed_text(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        # query() returns a list of dicts: {"id": ..., "similarity": ..., "meta": {...}}
        raw_results = news_index.query(vector=query_vector, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Endee query failed: {e}")

    articles = []
    for r in raw_results:
        # Results are dicts — access with r["key"] not r.key
        meta    = r.get("meta") or {}
        body    = meta.get("body", "")
        snippet = body[:200].rstrip() + "…" if len(body) > 200 else body

        articles.append(ArticleResult(
            id               = str(r.get("id", "")),
            title            = meta.get("title", "Untitled"),
            snippet          = snippet,
            category         = meta.get("category", "General"),
            source           = meta.get("source", "Unknown"),
            url              = meta.get("url", "#"),
            similarity_score = round(float(r.get("similarity", 0)), 4),
        ))

    return SearchResponse(query=q, total_results=len(articles), results=articles)


@app.get("/stats", response_model=StatsResponse)
def stats():
    if news_index is None:
        raise HTTPException(status_code=503, detail="Index not available.")
    return StatsResponse(
        index_name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
    )