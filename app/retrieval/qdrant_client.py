from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from app.core.config import get_settings
from app.core.logging import logger

class QdrantManager:
    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
        )
        self.text_collection = settings.TEXT_COLLECTION_NAME
        self.image_collection = settings.IMAGE_COLLECTION_NAME

    def create_payload_index(self, field_name):
        """Create payload index for filtering on specified field."""
        for collection in [self.text_collection, self.image_collection]:
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created payload index for '{field_name}' in {collection}")
            except Exception as e:
                logger.warning(f"Index for '{field_name}' may already exist in {collection}: {e}")

    def search(self, vector, collection_type="text", limit=5, filters=None):
        collection = self.text_collection if collection_type == "text" else self.image_collection
        logger.info(f"Searching {collection}, vector is None: {vector is None}")
        
        if vector is None:
            logger.error("Cannot search with None vector!")
            return []
            
        query_filter = None
        if filters:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            query_filter = Filter(must=conditions)
        try:
            results = self.client.query_points(
                collection_name=collection,
                query=vector.tolist() if hasattr(vector, 'tolist') else vector,
                limit=limit,
                query_filter=query_filter
            ).points
            logger.info(f"Found {len(results)} results in {collection}")
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def close(self):
        self.client.close()
