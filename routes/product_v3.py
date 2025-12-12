"""
Product search routes (V3-only).

This version of product.py is compatible only with the v3 hybrid visual search
implementation (modules.visual_search_v3). It calls the v3 API and exposes
clip_weight and visual_weight query parameters (instead of color_weight).
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List
import cv2
import numpy as np
from pathlib import Path
import uuid
import logging

from modules.visual_searchv3 import search_similar_products
from models.schemas import ProductSearchResponse, ProductMatch
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/product", tags=["Product Search"])


async def upload_to_cv2(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to cv2 image (BGR numpy array)."""
    contents = await upload_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


@router.post("/search", response_model=ProductSearchResponse)
async def search_product_v3(
    query_image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return"),
    clip_weight: float = Query(default=0.4, ge=0.0, le=1.0, description="Weight for CLIP semantic similarity (0-1)"),
    visual_weight: float = Query(default=0.6, ge=0.0, le=1.0, description="Weight for visual similarity (0-1)"),
    category_filter: bool = Query(default=True, description="Filter results to same predicted category")
):
    """
    Upload a product image to find similar products in inventory using the V3 hybrid search
    (CLIP + ResNet). Returns: Top-K similar products with similarity scores.
    """
    try:
        # Convert upload to cv2 image (BGR numpy array)
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
        try:
            preview = cv2.resize(img, (400, 400))
        except Exception:
            preview = img
        processed_dir = Path(settings.PROCESSED_DIR) / "visual_search"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_name = f"{query_id}_processed.jpg"
        processed_path = processed_dir / processed_name
        cv2.imwrite(str(processed_path), preview)
        query_processed_image_url = f"/processed/visual_search/{processed_name}"

        # Inventory and index path from settings
        inventory_dir = settings.INVENTORY_DIR
        index_path = getattr(settings, "VISUAL_SEARCH_INDEX_PATH", None)

        # Run visual search (v3 hybrid)
        logger.info(
            "Starting V3 visual search",
            extra={"top_k": top_k, "clip_weight": clip_weight, "visual_weight": visual_weight}
        )
        try:
            matches = search_similar_products(
                query_img=img,
                inventory_dir=inventory_dir,
                top_k=top_k,
                category_filter=category_filter,
                clip_weight=clip_weight,
                visual_weight=visual_weight,
                index_path=index_path,
            )
            logger.info(f"V3 search returned {len(matches)} matches")
        except Exception as search_error:
            logger.error(f"Error during V3 visual search: {search_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Visual search failed: {str(search_error)}")

        # Build response models
        inventory_base_url = "/inventory"
        match_models: List[ProductMatch] = []
        try:
            for match in matches:
                # v3 returns 'filename' and also compatibility keys; ensure we set product_image
                product_img = match.get("product_image") or match.get("filename")
                if not product_img:
                    # skip entries without filename
                    continue
                match["product_image"] = product_img
                match["product_image_url"] = f"{inventory_base_url}/{product_img}"

                # v3 sets color_similarity to 0.0 by default; convert 0-1 -> 0-100 for the API
                match["color_similarity"] = round(match.get("color_similarity", 0.0) * 100, 2)

                # Ensure semantic_similarity exists (map from clip_sim if necessary)
                if "semantic_similarity" not in match:
                    match["semantic_similarity"] = match.get("clip_sim") or None

                # Ensure any missing keys required by ProductMatch are provided as None/defaults
                # (ProductMatch schema may validate fields; using dict unpacking)
                match_models.append(ProductMatch(**match))
        except Exception as model_error:
            logger.error(f"Error creating match models: {model_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing search results: {str(model_error)}")

        return ProductSearchResponse(
            matches=match_models,
            total_matches=len(match_models),
            query_image_path=str(saved_query_path),
            query_image_url=query_image_url,
            query_processed_image_path=str(processed_path),
            query_processed_image_url=query_processed_image_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in V3 product search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")