"""
Product search routes.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import cv2
import numpy as np

from modules.visual_search import search_similar_products
from models.schemas import ProductSearchResponse, ProductMatch
from config import settings

router = APIRouter(prefix="/api/v1/product", tags=["Product Search"])


async def upload_to_cv2(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to cv2 image."""
    contents = await upload_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


@router.post("/search", response_model=ProductSearchResponse)
async def search_product(
    query_image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return")
):
    """
    Upload a product image to find similar products in inventory.
    Returns: Top-K similar products with similarity scores
    """
    try:
        # Convert upload to cv2 image
        img = await upload_to_cv2(query_image)
        
        # Get inventory directory from settings
        inventory_dir = settings.INVENTORY_DIR
        
        # Run visual search
        matches = search_similar_products(img, inventory_dir, top_k=top_k)
        
        # Convert to response model
        match_models = [ProductMatch(**match) for match in matches]
        
        return ProductSearchResponse(
            matches=match_models,
            total_matches=len(match_models)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

