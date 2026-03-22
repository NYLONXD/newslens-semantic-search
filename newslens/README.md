# NewsLens — Semantic News Search Engine

> Search news articles by **meaning**, not keywords. Built with Endee vector database, sentence-transformers, and FastAPI.

---

## Project Overview & Problem Statement

Traditional keyword search fails when the user's words don't exactly match the document's words. A search for *"heart attack prevention"* misses an article titled *"Reducing Cardiovascular Risk with Mediterranean Diet"* — even though it's exactly what the user wanted.

**NewsLens** solves this using **semantic search**: every article is converted into a numerical vector that captures its *meaning*, and queries are matched by geometric proximity in that vector space rather than by word overlap. This means:

- Synonyms work automatically ("car" matches "automobile")
- Conceptual queries work ("space exploration" matches articles about NASA, SpaceX, lunar landings)
- Paraphrases work ("heart disease prevention" matches "cardiovascular risk reduction")

---

## System Design & Technical Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                      INDEXING PIPELINE (run once)               │
│                                                                 │
│  articles.csv  →  Sentence Transformer  →  Endee Vector DB      │
│  (title, body)    (all-MiniLM-L6-v2)       (HNSW, cosine)      │
│                   384-dim vectors           INT8 precision       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE (real-time)                 │
│                                                                 │
│  User query  →  Sentence Transformer  →  Endee top-k search     │
│  (text)         (same model)             (cosine similarity)    │
│                 384-dim vector           returns ranked results  │
│                                                ↓                │
│                              FastAPI  →  Frontend UI            │
│                              (REST API)   (HTML + CSS + JS)     │
└─────────────────────────────────────────────────────────────────┘
```

### Embedding Model

`all-MiniLM-L6-v2` from the [sentence-transformers](https://www.sbert.net/) library:
- 384-dimensional output vectors
- Optimized for semantic similarity tasks
- Fast enough to run on CPU
- Free and open source (Apache 2.0)

Each article is embedded as: `"{title}. {title}. {first 512 chars of body}"` — the title is repeated to give it slightly more weight without manual weighting logic.

### Why Cosine Similarity?

Cosine similarity measures the *angle* between two vectors, making it invariant to vector magnitude. When vectors are normalized (unit length), cosine similarity equals the dot product — which Endee's HNSW index computes extremely efficiently.

### Endee Index Configuration

| Parameter | Value | Reason |
|---|---|---|
| `dimension` | 384 | Matches all-MiniLM-L6-v2 output |
| `space_type` | cosine | Semantic similarity |
| `precision` | INT8 | 8× smaller storage, ~1% accuracy loss |

---

## How Endee Is Used

Endee is the **core persistence and retrieval engine** of NewsLens. It is used in two places:

### 1. Indexing (`indexer/index_articles.py`)

```python
from endee import Endee, Precision

# Connect to the Endee server
client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create an index
client.create_index(
    name="news_articles",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,
)

# Store article vectors with metadata
index = client.get_index("news_articles")
index.upsert([
    {
        "id": "1",
        "vector": [0.023, -0.114, ...],   # 384-dim embedding
        "meta": {
            "title": "Scientists Discover...",
            "body":  "Marine biologists...",
            "category": "Science",
            "source": "National Geographic",
            "url": "https://example.com/...",
        }
    },
    # ... more articles
])
```

### 2. Querying (`backend/main.py`)

```python
# Embed the user's search query
query_vector = embed_text("ocean exploration deep sea")  # → 384-dim list

# Find the nearest neighbors in Endee
results = index.query(vector=query_vector, top_k=5)

# Each result has: result.id, result.similarity, result.meta
for r in results:
    print(r.meta["title"], r.similarity)
```

Endee handles all the heavy lifting: storing vectors on disk, building the HNSW graph index for fast approximate nearest-neighbor search, and returning results in milliseconds even as the dataset scales.

---

## Project Structure

```
newslens/
│
├── data/
│   └── sample_articles.csv      # 30 news articles across 6 categories
│
├── indexer/
│   └── index_articles.py        # Embeds articles and loads them into Endee
│
├── backend/
│   ├── main.py                  # FastAPI app — /search, /stats endpoints
│   ├── embedder.py              # Shared embedding utility (used by indexer + API)
│   ├── Dockerfile               # Containerizes the FastAPI backend
│   └── requirements.txt         # (copy of root requirements.txt)
│
├── frontend/
│   └── index.html               # Search UI (plain HTML, no build step)
│
├── docker-compose.yml           # Spins up Endee + FastAPI together
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
└── README.md
```

---

## Setup & Execution Instructions

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) v2
- Python 3.10+ (for running the indexer locally)
- ~2 GB RAM, ~1 GB disk space

### Option A — Quick Start (Docker Compose)

This is the recommended path. Everything runs in containers.

**Step 1: Fork and clone the Endee repository**

```bash
# Star + fork https://github.com/endee-io/endee first, then:
git clone https://github.com/YOUR_USERNAME/endee
cd endee
```

**Step 2: Clone this project into the Endee fork**

```bash
git clone https://github.com/YOUR_USERNAME/newslens
cd newslens
```

**Step 3: Copy the environment config**

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

**Step 4: Start Endee and the backend**

```bash
docker compose up -d
```

Wait ~30 seconds for Endee to fully start. Check it's running:

```bash
curl http://localhost:8080
# Should return: {"status":"ok"} or similar
```

**Step 5: Install Python dependencies and run the indexer**

```bash
pip install -r requirements.txt
cd indexer
python index_articles.py
```

You'll see output like:
```
[indexer] Connecting to Endee at http://localhost:8080
[indexer] Creating new index 'news_articles' (dim=384, cosine, INT8)
[indexer] Loading model: all-MiniLM-L6-v2
[indexer] Generating embeddings for 30 articles...
[indexer] Upserting 30 vectors into Endee...
[indexer] ✅ Done! Indexed 30 articles in 8.3s
```

**Step 6: Open the frontend**

Open `frontend/index.html` in your browser (just double-click it). Try a search!

---

### Option B — Local Development (no Docker)

Use this if you want to edit code with hot-reload.

**Step 1: Start Endee only**

```bash
docker compose up endee -d
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Run the indexer**

```bash
cd indexer
python index_articles.py
```

**Step 4: Start the backend with hot-reload**

```bash
cd backend
ENDEE_HOST=http://localhost:8080 uvicorn main:app --reload
```

**Step 5: Open the frontend**

Open `frontend/index.html` in your browser.

---

### Using Your Own Dataset

The indexer accepts any CSV with these columns:

| Column | Required | Description |
|---|---|---|
| `id` | ✅ | Unique identifier (string or number) |
| `title` | ✅ | Article headline |
| `body` | ✅ | Article text |
| `category` | optional | Topic category |
| `source` | optional | Publication name |
| `url` | optional | Link to original article |

```bash
python indexer/index_articles.py --csv ./data/my_articles.csv
```

To start fresh with a new dataset:

```bash
python indexer/index_articles.py --csv ./data/my_articles.csv --reset
```

---

## API Reference

The FastAPI backend exposes three endpoints:

### `GET /search`

```
GET http://localhost:8000/search?q=climate+change&top_k=5
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Natural language query |
| `top_k` | int | 5 | Number of results (max 20) |

**Response:**
```json
{
  "query": "climate change",
  "total_results": 5,
  "results": [
    {
      "id": "19",
      "title": "Wildfires Burn Record Acreage Across Canada",
      "snippet": "Canada recorded its worst wildfire season...",
      "category": "Environment",
      "source": "CBC",
      "url": "https://example.com/canada-wildfires",
      "similarity_score": 0.8712
    }
  ]
}
```

### `GET /stats`

Returns index statistics (total vectors, dimension, space type).

### `GET /`

Health check endpoint.

Interactive API docs available at: `http://localhost:8000/docs`

---

## Useful Commands

```bash
# View Endee logs
docker logs -f endee-server

# View backend logs
docker logs -f newslens-backend

# Stop everything
docker compose down

# Stop and wipe Endee data (forces re-indexing next time)
docker compose down -v

# Re-index with a fresh index
python indexer/index_articles.py --reset
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Vector database | [Endee](https://github.com/endee-io/endee) |
| Embedding model | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via sentence-transformers |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Containerization | Docker + Docker Compose |
| Data | Custom 30-article news CSV (replace with any dataset) |

---

## Extending the Project

**Scale to a real dataset:** Replace `sample_articles.csv` with a dataset from [Kaggle's All the News](https://www.kaggle.com/datasets/snapcrack/all-the-news) (200k+ articles) or the [CC-News dataset](https://huggingface.co/datasets/cc_news) on HuggingFace. The indexer handles it automatically.

**Better embeddings:** Swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (768-dim, higher quality) by changing `EMBEDDING_MODEL` in `.env` and updating `EMBEDDING_DIM` in `embedder.py`. Remember to update the Endee index dimension too.

**Hybrid search:** Combine Endee's vector search with Endee's built-in BM25 keyword scoring for even better results on rare terms.

**Add filters:** Endee supports metadata filtering. Add `?category=Science` filtering by passing filter parameters to `index.query()`.

---

## License

MIT
