from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("vdb_api")
QDRANT_ENDPOINT = os.getenv("cluster_endpoint")
TEXT_COLLECTION_NAME = "fashion-text"
IMAGE_COLLECTION_NAME = "fashion-image"

client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

# Delete all embeddings by removing entire collections
client.delete_collection(collection_name=TEXT_COLLECTION_NAME)
print(f"Deleted collection: {TEXT_COLLECTION_NAME}")

client.delete_collection(collection_name=IMAGE_COLLECTION_NAME)
print(f"Deleted collection: {IMAGE_COLLECTION_NAME}")

print("All embeddings removed.")
