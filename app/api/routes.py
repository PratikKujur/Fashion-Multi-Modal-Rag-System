from fastapi import APIRouter, Depends, UploadFile, File
from typing import Optional

router = APIRouter()

def get_fashion_service():
    from main import app
    return app.state.fashion_service

@router.get("/search")
async def search_fashion(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
    service=Depends(get_fashion_service)
):
    """Search fashion items using text query."""
    return await service.search(query=query, category=category, limit=limit)

@router.post("/search/image")
async def search_by_image(
    image: UploadFile = File(...),
    limit: int = 5,
    service=Depends(get_fashion_service)
):
    """Search fashion items using image query."""
    return await service.search_by_image(image=image, limit=limit)

@router.get("/recommend")
async def recommend(
    item_id: str,
    limit: int = 5,
    service=Depends(get_fashion_service)
):
    """Get recommendations based on an item."""
    return await service.recommend(item_id=item_id, limit=limit)

@router.get("/chat")
async def chat(
    message: str,
    session_id: Optional[str] = None,
    service=Depends(get_fashion_service)
):
    """Chat with fashion assistant."""
    return await service.chat(message=message, session_id=session_id)
