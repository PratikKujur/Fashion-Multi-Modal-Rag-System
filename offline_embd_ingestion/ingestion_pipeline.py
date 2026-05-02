import os
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dotenv import load_dotenv
import pandas as pd
import pyarrow as pa
from PIL import Image
from PyPDF2 import PdfReader
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import numpy as np
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_URL")

TEXT_VECTOR_SIZE = 512
IMAGE_VECTOR_SIZE = 512
TEXT_COLLECTION_NAME = "fashion-text"
IMAGE_COLLECTION_NAME = "fashion-image"
BATCH_SIZE = 128
VERBOSE = True
CHECKPOINT_FILE = "ingestion_checkpoint.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")

qdrant_client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY,
)


def load_checkpoint() -> Dict[str, List[str]]:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": [], "processed_items": []}


def save_checkpoint(checkpoint: Dict[str, List[str]]):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def ensure_collections_exist():
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if TEXT_COLLECTION_NAME not in collection_names:
        if VERBOSE:
            print(f"Creating collection: {TEXT_COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=TEXT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=TEXT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    if IMAGE_COLLECTION_NAME not in collection_names:
        if VERBOSE:
            print(f"Creating collection: {IMAGE_COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=IMAGE_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=IMAGE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def get_existing_ids(collection_name: str) -> Set[int]:
    existing_ids = set()
    try:
        offset = None
        while True:
            response = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, offset = response
            for point in points:
                existing_ids.add(point.id)
            if offset is None:
                break
        if VERBOSE:
            print(f"Found {len(existing_ids)} existing IDs in {collection_name}")
    except Exception as e:
        if VERBOSE:
            print(f"Error getting existing IDs: {e}")
    return existing_ids


def get_text_embedding(text: str) -> np.ndarray:
    return text_model.encode(text)


def get_image_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.pooler_output.cpu().numpy()[0]


def process_pdf_files(pdf_dir: str, existing_text_ids: Set[int], checkpoint: Dict) -> Tuple[List[Dict], List[Dict]]:
    pdf_path = Path(pdf_dir)
    text_points = []
    image_points = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    for pdf_file in pdf_path.glob("*.pdf"):
        if pdf_file.name in checkpoint["processed_files"]:
            if VERBOSE:
                print(f"Skipping already processed PDF: {pdf_file.name}")
            continue

        if VERBOSE:
            print(f"Processing PDF: {pdf_file.name}")

        full_text = ""
        try:
            elements = partition_pdf(
                filename=str(pdf_file),
                extract_images_in_pdf=False,
                infer_table_structure=True,
                strategy="fast",
            )
            texts = [str(el) for el in elements if hasattr(el, 'text') and el.text]
            full_text = " ".join(texts)
        except Exception as e:
            if VERBOSE:
                print(f"  unstructured failed, falling back to PyPDF2: {e}")
            reader = PdfReader(str(pdf_file))
            texts = [page.extract_text() or "" for page in reader.pages]
            full_text = " ".join(texts)

        if full_text.strip():
            chunks = text_splitter.split_text(full_text)
            if VERBOSE:
                print(f"  Split into {len(chunks)} text chunks")
            for i, chunk in enumerate(chunks):
                point_id = abs(hash(str(pdf_file.name) + str(i) + chunk[:50]))
                if point_id in existing_text_ids:
                    continue
                embedding = get_text_embedding(chunk)
                text_points.append({
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "source": "pdf",
                        "filename": pdf_file.name,
                        "text": chunk,
                        "chunk_index": i,
                        "category": "fashion_guide",
                    },
                })

        checkpoint["processed_files"].append(pdf_file.name)
        save_checkpoint(checkpoint)

    return text_points, image_points


def process_parquet_files(parquet_dir: str, existing_text_ids: Set[int], existing_image_ids: Set[int], checkpoint: Dict) -> Tuple[List[Dict], List[Dict]]:
    data_dir = Path(parquet_dir)
    text_points = []
    image_points = []

    arrow_files = list(data_dir.glob("*.arrow"))
    for arrow_file in arrow_files:
        if arrow_file.name in checkpoint["processed_files"]:
            if VERBOSE:
                print(f"Skipping already processed Arrow file: {arrow_file.name}")
            continue

        if VERBOSE:
            print(f"Processing Arrow file: {arrow_file.name}")

        with open(arrow_file, 'rb') as f:
            table = pa.ipc.open_stream(f).read_all()
            df = table.to_pandas()

        for idx, item in tqdm(df.iterrows(), total=len(df), desc=f"Processing {arrow_file.name}"):
            if pd.notna(item.get("text")):
                point_id = abs(hash(str(item.get("item_ID", idx)) + "_text"))
                if point_id not in existing_text_ids:
                    text = item["text"]
                    embedding = get_text_embedding(text)
                    text_points.append({
                        "id": point_id,
                        "vector": embedding.tolist(),
                        "payload": {
                            "source": "parquet",
                            "item_id": str(item.get("item_ID", "")),
                            "text": text,
                            "category": item.get("category", "unknown"),
                            "type": "text",
                        },
                    })

            if item.get("image") and item["image"].get("bytes"):
                try:
                    point_id = abs(hash(str(item.get("item_ID", idx)) + "_image"))
                    if point_id not in existing_image_ids:
                        image = Image.open(io.BytesIO(item["image"]["bytes"]))
                        # Save image to disk with absolute path
                        image_dir = Path.cwd() / "dataset" / "images"
                        image_dir.mkdir(parents=True, exist_ok=True)
                        image_filename = f"{item.get('item_ID', idx)}_{idx}.jpg"
                        image_path = image_dir / image_filename
                        image.save(image_path)
                        embedding = get_image_embedding(image)
                        image_points.append({
                            "id": point_id,
                            "vector": embedding.tolist(),
                            "payload": {
                                "source": "parquet",
                                "item_id": str(item.get("item_ID", "")),
                                "category": item.get("category", "unknown"),
                                "type": "image",
                                "image_path": str(image_path.absolute()),
                            },
                        })
                except Exception as e:
                    print(f"Error processing image at idx {idx}: {e}")

        checkpoint["processed_files"].append(arrow_file.name)
        save_checkpoint(checkpoint)

    return text_points, image_points


def batch_upload_text(points: List[Dict], batch_size: int = BATCH_SIZE):
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading text to Qdrant"):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=TEXT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in batch
            ],
        )


def batch_upload_image(points: List[Dict], batch_size: int = BATCH_SIZE):
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading images to Qdrant"):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=IMAGE_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in batch
            ],
        )


def process_and_upload(pdf_dir: str = None, parquet_dir: str = None, force_reingest: bool = False):
    ensure_collections_exist()

    checkpoint = load_checkpoint()

    if force_reingest:
        checkpoint = {"processed_files": [], "processed_items": []}
        save_checkpoint(checkpoint)
        existing_text_ids = set()
        existing_image_ids = set()
    else:
        existing_text_ids = get_existing_ids(TEXT_COLLECTION_NAME)
        existing_image_ids = get_existing_ids(IMAGE_COLLECTION_NAME)

    all_text_points = []
    all_image_points = []

    if pdf_dir and Path(pdf_dir).exists():
        print("Processing PDF files...")
        pdf_text, pdf_images = process_pdf_files(pdf_dir, existing_text_ids, checkpoint)
        all_text_points.extend(pdf_text)
        all_image_points.extend(pdf_images)
        print(f"Processed {len(pdf_text)} new PDF text chunks, {len(pdf_images)} new PDF images")

    if parquet_dir and Path(parquet_dir).exists():
        print("Processing Parquet/Arrow files...")
        parquet_text, parquet_images = process_parquet_files(parquet_dir, existing_text_ids, existing_image_ids, checkpoint)
        all_text_points.extend(parquet_text)
        all_image_points.extend(parquet_images)
        print(f"Processed {len(parquet_text)} new parquet text items, {len(parquet_images)} new parquet images")

    if all_text_points:
        print(f"Total new text points to upload: {len(all_text_points)}")
        batch_upload_text(all_text_points)
        print("Text upload complete!")

    if all_image_points:
        print(f"Total new image points to upload: {len(all_image_points)}")
        batch_upload_image(all_image_points)
        print("Image upload complete!")

    if not all_text_points and not all_image_points:
        print("No new data to upload.")

    if all_text_points or all_image_points:
        print("Ingestion complete. Clearing checkpoint.")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    process_and_upload(
        pdf_dir=r"D:\Projects\Fashion-Multi-Modal-Rag-System\dataset\dress_pdfs",
        parquet_dir=r"D:\Projects\Fashion-Multi-Modal-Rag-System\dataset",
    )
