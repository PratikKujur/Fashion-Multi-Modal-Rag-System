import os
from dotenv import load_dotenv
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np
from PIL import Image

load_dotenv()

QDRANT_API_KEY = os.getenv("vdb_api")
QDRANT_ENDPOINT = os.getenv("cluster_endpoint")

TEXT_COLLECTION_NAME = "fashion-text"
IMAGE_COLLECTION_NAME = "fashion-image"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")

qdrant_client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY,
)


def query_text(text: str, limit: int = 5):
    embedding = text_model.encode(text)
    results = qdrant_client.query_points(
        collection_name=TEXT_COLLECTION_NAME,
        query=embedding.tolist(),
        limit=limit,
    )
    return results.points


def query_image(image_path: str, limit: int = 5):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    embedding = outputs.pooler_output.cpu().numpy()[0]
    results = qdrant_client.query_points(
        collection_name=IMAGE_COLLECTION_NAME,
        query=embedding.tolist(),
        limit=limit,
    )
    return results.points


def get_collection_info():
    text_count = qdrant_client.count(collection_name=TEXT_COLLECTION_NAME).count
    image_count = qdrant_client.count(collection_name=IMAGE_COLLECTION_NAME).count
    return {"text_points": text_count, "image_points": image_count}


if __name__ == "__main__":
    info = get_collection_info()
    print(f"Database loaded. Text points: {info['text_points']}, Image points: {info['image_points']}")

    if info['text_points'] > 0:
        print("\nSample text query:")
        results = query_text("A skirt with flat strappy sandals")
        for r in results[:3]:
            print(f"  - {r.payload.get('text', '')[:100]}...")
