import uvicorn

if __name__ == "__main__":
    from app.core.config import get_settings
    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
