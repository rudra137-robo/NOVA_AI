from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "service": "SASTRA AI",
        "version": "0.1.0"
    }