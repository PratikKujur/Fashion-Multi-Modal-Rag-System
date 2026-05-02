from app.core.logging import logger

class FashionService:
    def __init__(self):
        from main import app
        self.qdrant = app.state.qdrant
        self.text_encoder = app.state.text_encoder
        self.image_encoder = app.state.image_encoder
        self.reranker = app.state.reranker
        self.llm = app.state.llm
        logger.info("FashionService initialized with all components")

    def _combine_results(self, text_results, image_results, limit):
        combined = list(text_results) + list(image_results)
        seen_ids = set()
        unique = []
        for r in combined:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique.append(r)
        unique.sort(key=lambda r: r.score, reverse=True)
        return unique[:limit]

    async def search(self, query, category=None, limit=5):
        query_vector = self.text_encoder.encode(query)
        if query_vector is None:
            logger.error("Text encoding returned None")
            return {"query": query, "results": [], "count": 0}
        filters = {"category": category} if category else None
        text_results = self.qdrant.search(query_vector, collection_type="text", limit=limit, filters=filters)
        image_results = self.qdrant.search(query_vector, collection_type="image", limit=limit, filters=filters)
        final_results = self._combine_results(text_results, image_results, limit)
        formatted = []
        for r in final_results:
            payload = r.payload
            formatted.append({
                "item_id": payload.get("item_id", ""),
                "name": payload.get("name", payload.get("text", "Item")[:50]),
                "category": payload.get("category", "unknown"),
                "description": payload.get("text", payload.get("description", ""))[:100],
                "score": round(r.score, 4),
                "source": payload.get("source", "unknown"),
                "type": payload.get("type", "unknown"),
                "image_path": payload.get("image_path", "")
            })
        return {"query": query, "results": formatted, "count": len(formatted)}

    async def search_by_image(self, image, limit=5):
        image_bytes = await image.read()
        image_vector = self.image_encoder.encode(image_bytes)
        if image_vector is None:
            logger.error("Image encoding returned None")
            return {"results": [], "count": 0}
        image_results = self.qdrant.search(image_vector, collection_type="image", limit=limit)
        formatted = []
        for r in image_results:
            payload = r.payload
            formatted.append({
                "item_id": payload.get("item_id", "unknown"),
                "category": payload.get("category", "unknown"),
                "score": round(r.score, 4),
                "source": payload.get("source", "unknown"),
                "type": payload.get("type", "unknown"),
                "image_path": payload.get("image_path", "")
            })
        return {"results": formatted, "count": len(formatted)}

    async def recommend(self, item_id, limit=5):
        filters = {"item_id": item_id}
        results = self.qdrant.search([0.0]*512, collection_type="image", limit=1, filters=filters)
        if not results:
            return {"item_id": item_id, "recommendations": []}
        similar = self.qdrant.search(results[0].vector, collection_type="image", limit=limit+1)
        recommendations = []
        for r in similar:
            if r.payload.get("item_id") != item_id:
                recommendations.append({
                    "item_id": r.payload.get("item_id", ""),
                    "category": r.payload.get("category", "unknown"),
                    "score": round(r.score, 4),
                    "image_path": r.payload.get("image_path", "")
                })
        return {"item_id": item_id, "recommendations": recommendations[:limit]}

    async def chat(self, message, session_id=None):
        query_vector = self.text_encoder.encode(message)
        context = "You are a fashion assistant."
        retrieved_items = []
        if query_vector is not None:
            text_results = self.qdrant.search(query_vector, collection_type="text", limit=3)
            image_results = self.qdrant.search(query_vector, collection_type="image", limit=3)
            combined = self._combine_results(text_results, image_results, 5)
            context_items = []
            for r in combined:
                name = r.payload.get("name", r.payload.get("item_id", ""))
                category = r.payload.get("category", "")
                desc = r.payload.get("description", r.payload.get("text", ""))
                image_path = r.payload.get("image_path", "")
                context_items.append(f"{name} ({category}): {desc}")
                retrieved_items.append({
                    "item_id": r.payload.get("item_id", ""),
                    "name": name,
                    "category": category,
                    "score": round(r.score, 4),
                    "image_path": image_path
                })
            context = "Relevant fashion items:\n" + "\n".join(context_items)
        response = self.llm.generate(message, context=context)
        return {"response": response, "retrieved_items": retrieved_items}
