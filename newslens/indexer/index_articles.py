"""
index_articles.py — NewsLens Indexer
--------------------------------------
Reads a CSV of news articles, generates embeddings, upserts into Endee.

Usage:
  python index_articles.py
  python index_articles.py --csv ./data/my_articles.csv
  python index_articles.py --reset
"""

import argparse
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from endee import Endee, Precision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from embedder import embed_batch, build_article_text, EMBEDDING_DIM

load_dotenv()

ENDEE_HOST       = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
INDEX_NAME       = os.getenv("ENDEE_INDEX_NAME", "news_articles")
DEFAULT_CSV_PATH = os.getenv("DATA_CSV_PATH", "../data/sample_articles.csv")
BATCH_SIZE       = 64


def connect_to_endee() -> Endee:
    print(f"[indexer] Connecting to Endee at {ENDEE_HOST}")
    client = Endee(ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(f"{ENDEE_HOST}/api/v1")
    return client


def create_or_get_index(client: Endee, reset: bool = False):
    """
    list_indexes() returns plain strings (index names), not objects.
    So we compare directly: if INDEX_NAME in existing_names.
    """
    # list_indexes() returns a list of strings
    existing = client.list_indexes()

    if INDEX_NAME in existing:
        if reset:
            print(f"[indexer] --reset flag: deleting index '{INDEX_NAME}'")
            client.delete_index(INDEX_NAME)
        else:
            print(f"[indexer] Index '{INDEX_NAME}' already exists. Upserting into it.")
            return client.get_index(INDEX_NAME)

    print(f"[indexer] Creating index '{INDEX_NAME}' (dim={EMBEDDING_DIM}, cosine, INT8)")
    client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        space_type="cosine",
        precision=Precision.INT8,
    )
    return client.get_index(INDEX_NAME)


def load_articles(csv_path: str) -> pd.DataFrame:
    print(f"[indexer] Loading articles from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[indexer] ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"id", "title", "body"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[indexer] ERROR: CSV missing columns: {missing}")
        sys.exit(1)

    for col, default in [("category", "General"), ("source", "Unknown"), ("url", "#")]:
        if col not in df.columns:
            df[col] = default

    before = len(df)
    df = df.dropna(subset=["title", "body"])
    if before != len(df):
        print(f"[indexer] Dropped {before - len(df)} rows with missing data")

    print(f"[indexer] Loaded {len(df)} articles")
    return df


def index_articles(csv_path: str, reset: bool = False):
    start_time = time.time()

    client = connect_to_endee()
    index  = create_or_get_index(client, reset=reset)
    df     = load_articles(csv_path)

    print("[indexer] Preparing texts for embedding...")
    texts = [
        build_article_text(row["title"], row["body"])
        for _, row in df.iterrows()
    ]

    print(f"[indexer] Generating embeddings for {len(texts)} articles...")
    vectors = embed_batch(texts, batch_size=BATCH_SIZE)

    items = []
    for vector, row in zip(vectors, df.itertuples(index=False)):
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

    print(f"[indexer] Upserting {len(items)} vectors into Endee...")
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        index.upsert(batch)
        print(f"[indexer]   {min(i + BATCH_SIZE, len(items))} / {len(items)} done")

    elapsed = time.time() - start_time
    print(f"\n[indexer] Done! Indexed {len(items)} articles in {elapsed:.1f}s")
    print(f"[indexer] Start the backend and open the frontend to search!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NewsLens Indexer")
    parser.add_argument("--csv",   default=DEFAULT_CSV_PATH)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    index_articles(csv_path=args.csv, reset=args.reset)