# NewsLens — Semantic News Search Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![Endee](https://img.shields.io/badge/Vector_DB-Endee-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Search news articles by **meaning**, not keywords — powered by [Endee](https://github.com/endee-io/endee) vector database.

---

## What Is This?

Traditional keyword search fails when your words don't match the document's words. Searching *"heart attack prevention"* misses an article titled *"Reducing Cardiovascular Risk"* — even though it's exactly what you want.

**NewsLens** solves this with **semantic search**:
- Every article is converted into a 384-dimensional vector that captures its *meaning*
- Queries are matched by proximity in that vector space — not by word overlap
- Synonyms, paraphrases, and related concepts all work automatically

### Example
| Query | What it finds |
|---|---|
| `"climate change"` | Wildfire, coral reef, Arctic ice articles |
| `"AI in medicine"` | CRISPR, Alzheimer's drug, AI diagnosis articles |
| `"space missions"` | Starlink, Moon landing, fusion energy articles |

---

## System Architecture

```
┌──────────────────────────── INDEXING (run once) ───────────────────────────┐
│                                                                             │
│   articles.csv  ──►  Sentence Transformer  ──►  Endee Vector DB            │
│   (title, body)       all-MiniLM-L6-v2           HNSW index                │
│                        384-dim vectors             cosine similarity        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── QUERY (real-time) ──────────────────────────────┐
│                                                                             │
│   User query  ──►  Sentence Transformer  ──►  Endee top-k search           │
│                     (same model)               returns ranked results       │
│                                                       │                     │
│                              FastAPI ◄────────────────┘                    │
│                                 │                                           │
│                              Frontend (HTML)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How Endee Is Used

Endee is the **core of this project** — it stores all article vectors and performs the nearest-neighbor search at query time.

### 1. Creating the index

```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="news_articles",
    dimension=384,        # matches all-MiniLM-L6-v2 output
    space_type="cosine",  # cosine similarity for semantic matching
    precision=Precision.INT8,  # 8x smaller storage, ~1% accuracy loss
)
```

### 2. Storing article vectors

```python
index = client.get_index("news_articles")

index.upsert([
    {
        "id": "1",
        "vector": [0.023, -0.114, ...],  # 384-dimensional embedding
        "meta": {
            "title": "Scientists Discover New Deep-Sea Fish",
            "body": "Marine biologists from...",
            "category": "Science",
            "source": "National Geographic",
            "url": "https://example.com/article"
        }
    }
])
```

### 3. Searching at query time

```python
# Embed the search query using the same model
query_vector = embed_text("ocean exploration deep sea")

# Endee finds the most similar article vectors
results = index.query(vector=query_vector, top_k=5)

for r in results:
    print(r["meta"]["title"], r["similarity"])
```

---

## Project Structure

```
newslens/
│
├── data/
│   └── sample_articles.csv      # 30 news articles across 6 categories
│
├── indexer/
│   └── index_articles.py        # Embeds articles → stores in Endee
│
├── backend/
│   ├── main.py                  # FastAPI: /search, /stats, / endpoints
│   ├── embedder.py              # Shared embedding utility
│   ├── Dockerfile               # Containerizes the API
│   └── requirements.txt
│
├── frontend/
│   └── index.html               # Search UI (no build step needed)
│
├── docker-compose.yml           # Runs Endee + backend together
├── requirements.txt             # Python dependencies
├── setup.sh                     # One-command venv setup
├── .env.example                 # Config template
└── README.md
```

---

## Setup & Running

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Endee)
- Python 3.10+
- ~3 GB disk space (for PyTorch + model download)

---

### Step 1 — Fork & Star Endee

> This is **mandatory** for the submission.

1. Go to [github.com/endee-io/endee](https://github.com/endee-io/endee)
2. Click **Star** ⭐
3. Click **Fork** and fork it to your account

---

### Step 2 — Clone This Project

```bash
git clone https://github.com/NYLONXD/newslens-semantic-search
cd newslens-semantic-search
```

---

### Step 3 — Create Virtual Environment

```bash
# Mac / Linux
bash setup.sh

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Every time you open a new terminal, activate the venv first:**
```bash
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### Step 4 — Start Endee

```bash
docker compose up endee -d
```

Verify it's running:
```bash
curl http://localhost:8080
# or open http://localhost:8080 in your browser
```

---

### Step 5 — Index the Articles

```bash
cd indexer
python index_articles.py
```

Expected output:
```
[indexer] Connecting to Endee at http://localhost:8080
[indexer] Creating index 'news_articles' (dim=384, cosine, INT8)
[indexer] Loading model: all-MiniLM-L6-v2
[indexer] Generating embeddings for 30 articles...
[indexer] Upserting 30 vectors into Endee...
[indexer] Done! Indexed 30 articles in 12.4s
```

---

### Step 6 — Start the Backend

```bash
cd ../backend
ENDEE_HOST=http://localhost:8080 uvicorn main:app --reload
```

API will be live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

### Step 7 — Open the Frontend

Just open `frontend/index.html` in your browser (double-click it).

---

## API Reference

### `GET /search`

```
GET http://localhost:8000/search?q=climate+change&top_k=5
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Your search query |
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
      "url": "https://example.com/...",
      "similarity_score": 0.8712
    }
  ]
}
```

### `GET /stats`
Returns index name, vector dimension, and similarity metric.

### `GET /`
Health check. Returns `{"status": "ok"}`.

---

## Using a Larger Dataset

The included dataset has 30 sample articles. To use a real dataset:

1. Download [All the News](https://www.kaggle.com/datasets/snapcrack/all-the-news) from Kaggle (200k+ articles)
2. Make sure it has `id`, `title`, `body` columns
3. Run:
```bash
python indexer/index_articles.py --csv ./data/your_dataset.csv --reset
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `venv\Scripts\activate` first |
| `Cannot connect to Endee` | Run `docker compose up endee -d` and wait 15s |
| `Index not found` | Run the indexer script first |
| Port 8080 in use | Change `"8080:8080"` to `"8081:8080"` in docker-compose.yml |
| Slow install | PyTorch is 2GB — just wait, it's normal |

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector database | [Endee](https://github.com/endee-io/endee) |
| Embedding model | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Backend | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Infrastructure | Docker + Docker Compose |

---

## License

MIT
