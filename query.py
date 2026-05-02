import os
from dotenv import load_dotenv
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np
from PIL import Image

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_URL")

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


def query_image(image_path: str, category: str = None, limit: int = 5):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    embedding = outputs.pooler_output.cpu().numpy()[0]

    filters = None
    if category:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        filters = Filter(must=[FieldCondition(key="category", match=MatchValue(value=category))])

    text_results = qdrant_client.query_points(
        collection_name=TEXT_COLLECTION_NAME,
        query=embedding.tolist(),
        limit=limit,
        query_filter=filters,
    ).points

    image_results = qdrant_client.query_points(
        collection_name=IMAGE_COLLECTION_NAME,
        query=embedding.tolist(),
        limit=limit,
        query_filter=filters,
    ).points

    combined = list(text_results) + list(image_results)
    seen_ids = set()
    unique = []
    for r in combined:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            unique.append(r)
    unique.sort(key=lambda r: r.score, reverse=True)
    return unique[:limit]


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

    if info['image_points'] > 0:
        print("\nSample image query:")
        
        image_path = 'D:\Projects\Fashion-Multi-Modal-Rag-System\dataset\jeans.jpg'
        results = query_image(image_path, limit=5)
        for r in results[:5]:
            print(f"  - {r.payload.get('name', 'Item')} ({r.payload.get('category', '')}) - Score: {r.score:.4f}")
        
