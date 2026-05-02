from app.core.config import get_settings
from app.core.logging import logger
import torch

class TextEncoder:
    def __init__(self):
        settings = get_settings()
        self.model_name = settings.HF_EMBEDDING_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded text model: {self.model_name} on {self.device}")

    def encode(self, text):
        try:
            embedding = self.model.encode(text)
            logger.info(f"Encoded text: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Text encoding error: {e}")
            return None
