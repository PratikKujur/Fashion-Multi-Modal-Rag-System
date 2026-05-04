---
title: Fashion Multi-Modal RAG System
description: A production-ready Retrieval-Augmented Generation system for fashion with text + image search, LLM-powered recommendations, and vector-based semantic retrieval
sdk: streamlit
app_file: streamlit_app.py
license: mit
---

# Fashion Multi-Modal RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system that combines **text and image understanding** to deliver intelligent fashion search, visual similarity matching, and conversational styling advice. Built with **FastAPI**, **Streamlit**, **Qdrant**, and **Groq**.

---

## Key Features

- **Multi-Modal Search** - Search fashion items using text queries ("red summer dress") or upload images for visual similarity matching
- **Semantic Vector Retrieval** - CLIP-powered image embeddings + multilingual text embeddings stored in Qdrant for fast, context-aware search
- **LLM-Powered Chat Assistant** - Conversational fashion assistant backed by Groq (Llama 3.3 70B) with context-aware retrieval
- **Personalized Recommendations** - Item-to-item similarity recommendations powered by vector search
- **Cross-Encoder Reranking** - Results refined with cross-encoder reranker for higher precision
- **Multi-Source Data Ingestion** - Supports HuggingFace Datasets, Parquet/Arrow files, and PDF fashion guides
- **Production-Ready Architecture** - FastAPI backend with automatic startup from Streamlit frontend, health checks, and connection recovery

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Vector DB** | Qdrant |
| **Text Embeddings** | `sentence-transformers/distiluse-base-multilingual-cased` |
| **Image Embeddings** | `openai/clip-vit-base-patch32` |
| **LLM** | Groq (`llama-3.3-70b-versatile`) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Data Processing** | PyArrow, Pandas, PIL, PyPDF2 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                       │
│  (Text Search | Image Search | Recommendations | Chat)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP requests
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────┐│
│  │ Text Encoder│  │Image Encoder │  │  Reranker  │  │  LLM   ││
│  │(Sentence-   │  │   (CLIP)     │  │(Cross-     │  │(Groq/  ││
│  │Transformers)│  │              │  │ Encoder)   │  │ Llama) ││
│  └──────┬──────┘  └──────┬───────┘  └──────┬─────┘  └───┬────┘│
│         │                │                 │             │      │
│         └────────────────┴─────────────────┴─────────────┘      │
│                              │                                  │
│                     FashionService                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Database                       │
│                                                                 │
│   fashion-text (512d)          fashion-image (512d)            │
│   └─ text embeddings           └─ CLIP image embeddings        │
│   └─ payloads: text, meta      └─ payloads: base64, category   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python >= 3.11
- CUDA-compatible GPU (optional, recommended for faster embeddings)
- Qdrant instance (local or cloud)

### Quick Start

1. **Clone and setup:**
```bash
git clone <repository-url>
cd fashion-Rag-v2
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
```

2. **Configure environment:**

Create a `.env` file in the project root:

```env
# Qdrant Vector Database
QDRANT_URL=http://localhost:6333        # or your Qdrant Cloud URL
QDRANT_API_KEY=your_qdrant_api_key

# HuggingFace (for embedding models)
HF_API_TOKEN=your_huggingface_api_token

# Groq API (for LLM inference)
GROQ_API_KEY=your_groq_api_key
```

3. **Start the application:**
```bash
streamlit run streamlit_app.py
```

The app auto-starts the FastAPI backend internally. Access it at `http://localhost:8501`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/search` | Text-based fashion search with optional category filter |
| `POST` | `/api/search/image` | Visual similarity search by uploaded image |
| `GET` | `/api/recommend` | Get item-to-item recommendations |
| `GET` | `/api/chat` | Conversational fashion assistant |
| `GET` | `/health` | Health check |

### Examples

**Text search:**
```bash
curl "http://localhost:8000/api/search?query=red+summer+dress&category=dress&limit=5"
```

**Image search:**
```bash
curl -X POST -F "image=@dress.jpg" "http://localhost:8000/api/search/image?limit=5"
```

**Recommendations:**
```bash
curl "http://localhost:8000/api/recommend?item_id=131596499_6&limit=5"
```

**Chat:**
```bash
curl "http://localhost:8000/api/chat?message=What+should+I+wear+for+a+beach+wedding"
```

---

## Data Ingestion

The system supports three data sources, processed via the ingestion pipeline:

### 1. HuggingFace Datasets (Primary)

```bash
python offline_embd_ingestion/ingestion_pipeline.py
```

The pipeline loads datasets (e.g., `Marqo/polyvore`), generates CLIP embeddings for images and sentence-transformer embeddings for text, then uploads to Qdrant with base64-encoded image payloads.

### 2. Parquet/Arrow Files

Place `.arrow` files in a directory and run:

```python
process_and_upload(parquet_dir="path/to/parquet/files")
```

Images are stored as raw bytes in payloads for local deployments.

### 3. PDF Fashion Guides

```python
process_and_upload(pdf_dir="path/to/fashion/pdfs")
```

PDFs are parsed (using `unstructured` or PyPDF2 fallback), chunked with LangChain's `RecursiveCharacterTextSplitter`, and indexed in the text collection.

### Ingestion Features

- **Checkpoint-based** resume support (skips already-processed files/items)
- **Batch uploads** to Qdrant for efficient indexing
- **Deduplication** via hash-based point IDs
- **Force re-ingest** mode with `force_reingest=True`

---

## Configuration

All settings are managed via `app/core/config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `QDRANT_URL` | - | Qdrant instance URL (required) |
| `QDRANT_API_KEY` | - | Qdrant API key |
| `HF_API_TOKEN` | - | HuggingFace API token |
| `GROQ_API_KEY` | - | Groq API key for LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/distiluse-base-multilingual-cased` | Text embedding model |
| `HF_IMAGE_MODEL` | `openai/clip-vit-base-patch32` | Image embedding model |
| `TEXT_COLLECTION_NAME` | `fashion-text` | Qdrant text collection |
| `IMAGE_COLLECTION_NAME` | `fashion-image` | Qdrant image collection |
| `EMBEDDING_DIM` | `512` | Embedding vector dimension |
| `TOP_K` | `20` | Initial retrieval count |
| `RERANK_TOP_K` | `5` | Results after reranking |

---

## Project Structure

```
fashion-Rag-v2/
├── app/                          # Backend application
│   ├── api/                      # FastAPI routes and app setup
│   │   ├── main.py               # FastAPI app with lifespan management
│   │   └── routes.py             # API endpoints (search, recommend, chat)
│   ├── core/                     # Core utilities
│   │   ├── config.py             # Pydantic settings management
│   │   └── logging.py            # Application logging
│   ├── embeddings/               # Embedding models
│   │   ├── text_encoder.py       # Sentence transformer text encoder
│   │   └── image_encoder.py      # CLIP image encoder
│   ├── rag/                      # LLM generation
│   │   └── generator.py          # Groq/HuggingFace LLM wrapper
│   ├── reranker/                 # Result reranking
│   │   └── cross_encoder.py      # Cross-encoder reranker
│   ├── retrieval/                # Vector database layer
│   │   └── qdrant_client.py      # Qdrant manager with reconnect logic
│   ├── services/                 # Business logic
│   │   └── fashion_service.py    # Orchestrates search, recommend, chat
│   └── utils/                    # Helper utilities
│       └── image_utils.py        # Image loading/resizing
├── offline_embd_ingestion/       # Data ingestion pipeline
│   ├── ingestion_pipeline.py     # Multi-source data processor
│   └── del_embd.py               # Embedding deletion utility
├── api/                          # API package marker
│   └── main.py                   # Entrypoint
├── streamlit_app.py              # Streamlit frontend (auto-starts backend)
├── app.py                        # Gradio frontend (legacy)
├── tests/                        # Test suite
├── experiments/                  # Experiment notebooks/scripts
├── pyproject.toml                # Project dependencies
├── requirements.txt              # Pip requirements
├── Dockerfile                    # Container configuration
└── .env                          # Environment variables (gitignored)
```

---

## Docker

Build and run with Docker:

```bash
docker build -t fashion-rag .
docker run -p 8501:8501 -p 8000:8000 --env-file .env fashion-rag
```

---

## License

MIT
