from app.core.config import get_settings
from app.core.logging import logger
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import io

class ImageEncoder:
    def __init__(self):
        settings = get_settings()
        self.model_name = settings.HF_IMAGE_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        logger.info(f"Loaded image model: {self.model_name} on {self.device}")

    def encode(self, image):
        try:
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            embedding = outputs.pooler_output.cpu().numpy()[0]
            logger.info(f"Encoded image: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return None
