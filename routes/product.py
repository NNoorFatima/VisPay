"""
Product search routes.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import cv2
import numpy as np
from pathlib import Path
import uuid

from modules.visual_search import search_similar_products
from modules.preprocessing import preprocess_image_for_visual_search
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
        original_filename = query_image.filename or "query.jpg"
        query_id = uuid.uuid4().hex[:8]
        
        # Persist original upload
        uploads_dir = Path(settings.UPLOAD_DIR) / "visual_search"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        original_ext = Path(original_filename).suffix or ".jpg"
        saved_query_name = f"{query_id}{original_ext}"
        saved_query_path = uploads_dir / saved_query_name
        cv2.imwrite(str(saved_query_path), img)
        query_image_url = f"/uploads/visual_search/{saved_query_name}"
        
        # Persist preprocessed preview (largest configured scale)
        preview = preprocess_image_for_visual_search(img.copy())
        processed_dir = Path(settings.PROCESSED_DIR) / "visual_search"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_name = f"{query_id}_processed.jpg"
        processed_path = processed_dir / processed_name
        if preview is None:
            preview = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(processed_path), preview)
        query_processed_image_url = f"/processed/visual_search/{processed_name}"
        
        # Get inventory directory from settings
        inventory_dir = settings.INVENTORY_DIR
        
        # Run visual search
        matches = search_similar_products(img, inventory_dir, top_k=top_k)
        
        inventory_base_url = "/inventory"
        match_models: List[ProductMatch] = []
        for match in matches:
            match["product_image_url"] = f"{inventory_base_url}/{match['product_image']}"
            match["color_similarity"] = round(match.get("color_similarity", 0.0) * 100, 2)
            match_models.append(ProductMatch(**match))
        
        return ProductSearchResponse(
            matches=match_models,
            total_matches=len(match_models),
            query_image_path=str(saved_query_path),
            query_image_url=query_image_url,
            query_processed_image_path=str(processed_path),
            query_processed_image_url=query_processed_image_url
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

