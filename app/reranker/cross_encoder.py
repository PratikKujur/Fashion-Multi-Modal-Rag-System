from app.core.config import get_settings
from app.core.logging import logger

class CrossEncoderReranker:
    def __init__(self):
        settings = get_settings()
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")
            logger.info("Loaded FlashRank reranker")
        except Exception as e:
            logger.error(f"Error loading FlashRank: {e}")
            self.ranker = None
            logger.warning("Reranker not available")

    def rerank(self, query, documents, top_k=5):
        if not self.ranker or not documents:
            return documents[:top_k] if documents else []
        try:
            if isinstance(documents[0], str):
                docs = [{"text": doc} for doc in documents]
            else:
                docs = documents
            results = self.ranker.rerank(query, docs)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents[:top_k]
