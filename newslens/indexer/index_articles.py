"""
index_articles.py — NewsLens Indexer
--------------------------------------
This script reads a CSV of news articles, generates vector embeddings
for each article, and upserts them into an Endee index.

Run this ONCE before starting the backend. Re-running is safe —
Endee upsert will overwrite existing vectors with matching IDs.

Usage:
  python index_articles.py
  python index_articles.py --csv ./data/my_articles.csv
  python index_articles.py --reset   # drops and recreates the index first

Environment variables (or .env file):
  ENDEE_HOST         default: http://localhost:8080
  ENDEE_AUTH_TOKEN   default: (empty = no auth)
  ENDEE_INDEX_NAME   default: news_articles
  DATA_CSV_PATH      default: ../data/sample_articles.csv
"""

import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from endee import Endee, Precision

# Add backend dir to path so we can import the shared embedder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from embedder import embed_batch, build_article_text, EMBEDDING_DIM

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ENDEE_HOST       = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME       = os.getenv("ENDEE_INDEX_NAME", "news_articles")
DEFAULT_CSV_PATH = os.getenv("DATA_CSV_PATH", "../data/sample_articles.csv")
BATCH_SIZE       = 64   # how many articles to embed + upsert at once


def connect_to_endee() -> Endee:
    """Create and configure an Endee client."""
    print(f"[indexer] Connecting to Endee at {ENDEE_HOST}")
    client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")
    return client


def create_or_get_index(client: Endee, reset: bool = False):
    """
    Create the Endee index if it doesn't exist.
    If reset=True, drop it first and start fresh.

    Index parameters:
    - dimension=384      → matches all-MiniLM-L6-v2 output size
    - space_type=cosine  → cosine similarity for semantic matching
    - precision=INT8     → quantized storage (8x smaller, ~1% accuracy loss)
    """
    # List existing indexes
    existing = [idx.name for idx in client.list_indexes()]

    if INDEX_NAME in existing:
        if reset:
            print(f"[indexer] --reset flag set. Deleting existing index '{INDEX_NAME}'")
            client.delete_index(INDEX_NAME)
        else:
            print(f"[indexer] Index '{INDEX_NAME}' already exists. Upserting into it.")
            return client.get_index(INDEX_NAME)

    print(f"[indexer] Creating new index '{INDEX_NAME}' (dim={EMBEDDING_DIM}, cosine, INT8)")
    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    return client.get_index(INDEX_NAME)


def load_articles(csv_path: str) -> pd.DataFrame:
    """Load and validate the articles CSV."""
    print(f"[indexer] Loading articles from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[indexer] ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"id", "title", "body"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[indexer] ERROR: CSV missing columns: {missing}")
        print(f"[indexer] Found columns: {list(df.columns)}")
        sys.exit(1)

    # Fill optional columns
    for col, default in [("category", "General"), ("source", "Unknown"), ("url", "#")]:
        if col not in df.columns:
            df[col] = default

    # Drop rows with missing title or body
    before = len(df)
    df = df.dropna(subset=["title", "body"])
    after = len(df)
    if before != after:
        print(f"[indexer] Dropped {before - after} rows with missing title/body")

    print(f"[indexer] Loaded {len(df)} articles")
    return df


def index_articles(csv_path: str, reset: bool = False):
    """Main indexing pipeline."""
    start_time = time.time()

    # 1. Connect
    client = connect_to_endee()
    index = create_or_get_index(client, reset=reset)

    # 2. Load data
    df = load_articles(csv_path)

    # 3. Prepare texts for embedding
    #    We combine title + body using the shared helper from embedder.py
    print("[indexer] Preparing article texts for embedding...")
    texts = [
        build_article_text(row["title"], row["body"])
        for _, row in df.iterrows()
    ]

    # 4. Generate embeddings in batches
    print(f"[indexer] Generating embeddings for {len(texts)} articles (batch_size={BATCH_SIZE})...")
    vectors = embed_batch(texts, batch_size=BATCH_SIZE)

    # 5. Build Endee vector items
    #    meta stores everything we want to return in search results
    items = []
    for i, (vector, row) in enumerate(zip(vectors, df.itertuples(index=False))):
        items.append({
            "id": str(row.id),
            "vector": vector,
            "meta": {
                "title":    row.title,
                "body":     row.body,
                "category": row.category,
                "source":   row.source,
                "url":      row.url,
            },
        })

    # 6. Upsert to Endee in batches
    print(f"[indexer] Upserting {len(items)} vectors into Endee...")
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        index.upsert(batch)
        print(f"[indexer]   Upserted {min(i + BATCH_SIZE, len(items))} / {len(items)}")

    elapsed = time.time() - start_time
    print(f"\n[indexer] ✅ Done! Indexed {len(items)} articles in {elapsed:.1f}s")
    print(f"[indexer] Index '{INDEX_NAME}' is ready. Start the backend and search!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NewsLens Article Indexer")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Path to articles CSV file (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Endee index before indexing",
    )
    args = parser.parse_args()

    index_articles(csv_path=args.csv, reset=args.reset)
