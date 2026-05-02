from app.core.config import get_settings

settings = get_settings()

def get_qdrant_client():
    from app.retrieval.qdrant_client import QdrantManager
    return QdrantManager()

def get_embedding_service():
    from app.embeddings.text_encoder import TextEncoder
    from app.embeddings.image_encoder import ImageEncoder
    return TextEncoder(), ImageEncoder()

def get_llm_service():
    from app.rag.generator import LLMGenerator
    return LLMGenerator()

def get_fashion_service():
    from app.services.fashion_service import FashionService
    return FashionService()
