"""
Product search routes.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List
import cv2
import numpy as np
from pathlib import Path
import uuid
import logging

from modules.visual_search import search_similar_products
from models.schemas import ProductSearchResponse, ProductMatch
from config import settings

logger = logging.getLogger(__name__)

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
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return"),
    color_weight: float = Query(default=0.3, ge=0.0, le=1.0, description="Weight for color similarity (0-1)")
):
    """
    Upload a product image to find similar products in inventory using ResNet50 deep features.
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
        
        # Save a preview (resized for display)
        preview = cv2.resize(img, (400, 400))
        processed_dir = Path(settings.PROCESSED_DIR) / "visual_search"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_name = f"{query_id}_processed.jpg"
        processed_path = processed_dir / processed_name
        cv2.imwrite(str(processed_path), preview)
        query_processed_image_url = f"/processed/visual_search/{processed_name}"
        
        # Get inventory directory and index path from settings
        inventory_dir = settings.INVENTORY_DIR
        index_path = getattr(settings, 'VISUAL_SEARCH_INDEX_PATH', None)
        
        # Run visual search (ResNet50 + color histograms)
        logger.info(f"Starting visual search with top_k={top_k}, color_weight={color_weight}")
        try:
            matches = search_similar_products(
                img, 
                inventory_dir, 
                top_k=top_k,
                color_weight=color_weight,
                index_path=index_path
            )
            logger.info(f"Search returned {len(matches)} matches")
        except Exception as search_error:
            logger.error(f"Error during visual search: {search_error}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Visual search failed: {str(search_error)}"
            )
        
        inventory_base_url = "/inventory"
        match_models: List[ProductMatch] = []
        try:
            for match in matches:
                match["product_image_url"] = f"{inventory_base_url}/{match['product_image']}"
                # Convert color_similarity from 0-1 to 0-100 percentage
                match["color_similarity"] = round(match.get("color_similarity", 0.0) * 100, 2)
                # Ensure semantic_similarity is included if present
                if "semantic_similarity" not in match:
                    match["semantic_similarity"] = None
                match_models.append(ProductMatch(**match))
        except Exception as model_error:
            logger.error(f"Error creating match models: {model_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing search results: {str(model_error)}"
            )
        
        return ProductSearchResponse(
            matches=match_models,
            total_matches=len(match_models),
            query_image_path=str(saved_query_path),
            query_image_url=query_image_url,
            query_processed_image_path=str(processed_path),
            query_processed_image_url=query_processed_image_url
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in product search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

