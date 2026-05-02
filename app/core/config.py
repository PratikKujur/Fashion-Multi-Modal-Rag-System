import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # Qdrant Settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    TEXT_COLLECTION_NAME: str = "fashion-text"
    IMAGE_COLLECTION_NAME: str = "fashion-image"

    # HuggingFace Settings
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_EMBEDDING_MODEL: str = "sentence-transformers/distiluse-base-multilingual-cased"
    HF_IMAGE_MODEL: str = "openai/clip-vit-base-patch32"
    HF_LLM_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"

    # Groq Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Embedding Settings
    EMBEDDING_DIM: int = 512
    USE_LOCAL_EMBEDDINGS: bool = False

    # Reranker Settings
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval Settings
    TOP_K: int = 20
    RERANK_TOP_K: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings():
    return Settings()
