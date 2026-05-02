from app.core.logging import logger

class Pipeline:
    def __init__(self, qdrant, text_encoder, image_encoder, reranker, llm):
        self.qdrant = qdrant
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.reranker = reranker
        self.llm = llm
        logger.info("Pipeline initialized")

    def process_text_query(self, query, category=None, limit=5):
        query_vector = self.text_encoder.encode(query)
        filters = {"category": category} if category else None
        results = self.qdrant.search(query_vector, limit=limit, filters=filters)
        return results

    def process_image_query(self, image, category=None, limit=5):
        image_vector = self.image_encoder.encode(image)
        filters = {"category": category} if category else None
        results = self.qdrant.search(image_vector, limit=limit, filters=filters)
        return results
