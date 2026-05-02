from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging import logger
from app.core.config import get_settings
from app.api.routes import router
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic for the FastAPI application."""
    settings = get_settings()
    logger.info("Starting Fashion Multi-Modal RAG System...")

    # Load Qdrant client
    logger.info("Connecting to Qdrant...")
    from app.retrieval.qdrant_client import QdrantManager
    app.state.qdrant = QdrantManager()
    logger.info("Qdrant connected successfully")

    # Create payload indexes for filtering
    logger.info("Creating payload indexes...")
    app.state.qdrant.create_payload_index("category")
    app.state.qdrant.create_payload_index("item_id")
    logger.info("Payload indexes created")

    # Load embedding models
    logger.info("Loading embedding models...")
    from app.embeddings.text_encoder import TextEncoder
    from app.embeddings.image_encoder import ImageEncoder
    app.state.text_encoder = TextEncoder()
    app.state.image_encoder = ImageEncoder()
    logger.info("Embedding models loaded")

    # Load reranker
    logger.info("Loading reranker...")
    from app.reranker.cross_encoder import CrossEncoderReranker
    app.state.reranker = CrossEncoderReranker()
    logger.info("Reranker loaded")

    # Load LLM
    logger.info("Initializing LLM...")
    from app.rag.generator import LLMGenerator
    app.state.llm = LLMGenerator()
    logger.info("LLM initialized")

    # Load fashion service
    logger.info("Initializing fashion service...")
    from app.services.fashion_service import FashionService
    app.state.fashion_service = FashionService(
        qdrant=app.state.qdrant,
        text_encoder=app.state.text_encoder,
        image_encoder=app.state.image_encoder,
        reranker=app.state.reranker,
        llm=app.state.llm
    )
    logger.info("Fashion service ready")

    logger.info("All components loaded successfully!")
    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, "qdrant"):
        app.state.qdrant.close()
    logger.info("Shutdown complete")

app = FastAPI(
    title="Fashion Multi-Modal RAG API",
    description="API for fashion item retrieval and recommendation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fashion-rag"}

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
