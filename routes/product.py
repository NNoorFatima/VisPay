# """
# Product search routes.
# """
# from fastapi import APIRouter, File, UploadFile, HTTPException, Query
# from typing import List
# import cv2
# import numpy as np
# from pathlib import Path
# import uuid
# import logging

# from modules.visual_searchv3 import search_similar_products
# from models.schemas import ProductSearchResponse, ProductMatch
# from config import settings

# logger = logging.getLogger(__name__)

# router = APIRouter(prefix="/api/v1/product", tags=["Product Search"])


# async def upload_to_cv2(upload_file: UploadFile) -> np.ndarray:
#     """Convert uploaded file to cv2 image."""
#     contents = await upload_file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(status_code=400, detail="Invalid image file")
#     return img


# @router.post("/search", response_model=ProductSearchResponse)
# async def search_product(
#     query_image: UploadFile = File(...),
#     top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return"),
#     color_weight: float = Query(default=0.3, ge=0.0, le=1.0, description="Weight for color similarity (0-1)")
# ):
#     """
#     Upload a product image to find similar products in inventory using ResNet50 deep features.
#     Returns: Top-K similar products with similarity scores
#     """
#     try:
#         # Convert upload to cv2 image
#         img = await upload_to_cv2(query_image)
#         original_filename = query_image.filename or "query.jpg"
#         query_id = uuid.uuid4().hex[:8]
        
#         # Persist original upload
#         uploads_dir = Path(settings.UPLOAD_DIR) / "visual_search"
#         uploads_dir.mkdir(parents=True, exist_ok=True)
#         original_ext = Path(original_filename).suffix or ".jpg"
#         saved_query_name = f"{query_id}{original_ext}"
#         saved_query_path = uploads_dir / saved_query_name
#         cv2.imwrite(str(saved_query_path), img)
#         query_image_url = f"/uploads/visual_search/{saved_query_name}"
        
#         # Save a preview (resized for display)
#         preview = cv2.resize(img, (400, 400))
#         processed_dir = Path(settings.PROCESSED_DIR) / "visual_search"
#         processed_dir.mkdir(parents=True, exist_ok=True)
#         processed_name = f"{query_id}_processed.jpg"
#         processed_path = processed_dir / processed_name
#         cv2.imwrite(str(processed_path), preview)
#         query_processed_image_url = f"/processed/visual_search/{processed_name}"
        
#         # Get inventory directory and index path from settings
#         inventory_dir = settings.INVENTORY_DIR
#         index_path = getattr(settings, 'VISUAL_SEARCH_INDEX_PATH', None)
        
#         # Run visual search (ResNet50 + color histograms)
#         logger.info(f"Starting visual search with top_k={top_k}, color_weight={color_weight}")
#         try:
#             matches = search_similar_products(
#                 img, 
#                 inventory_dir, 
#                 top_k=top_k,
#                 color_weight=color_weight,
#                 index_path=index_path
#             )
#             logger.info(f"Search returned {len(matches)} matches")
#         except Exception as search_error:
#             logger.error(f"Error during visual search: {search_error}", exc_info=True)
#             raise HTTPException(
#                 status_code=500, 
#                 detail=f"Visual search failed: {str(search_error)}"
#             )
        
#         inventory_base_url = "/inventory"
#         match_models: List[ProductMatch] = []
#         try:
#             for match in matches:
#                 match["product_image_url"] = f"{inventory_base_url}/{match['product_image']}"
#                 # Convert color_similarity from 0-1 to 0-100 percentage
#                 match["color_similarity"] = round(match.get("color_similarity", 0.0) * 100, 2)
#                 # Ensure semantic_similarity is included if present
#                 if "semantic_similarity" not in match:
#                     match["semantic_similarity"] = None
#                 match_models.append(ProductMatch(**match))
#         except Exception as model_error:
#             logger.error(f"Error creating match models: {model_error}", exc_info=True)
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Error processing search results: {str(model_error)}"
#             )
        
#         return ProductSearchResponse(
#             matches=match_models,
#             total_matches=len(match_models),
#             query_image_path=str(saved_query_path),
#             query_image_url=query_image_url,
#             query_processed_image_path=str(processed_path),
#             query_processed_image_url=query_processed_image_url
#         )
    
#     except HTTPException:
#         # Re-raise HTTP exceptions as-is
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in product search: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

"""
Product search routes (V3-only)a.

This version of product.py is compatible only with the v3 hybrid visual search
implementation (modules.visual_search_v3). It calls the v3 API and exposes
clip_weight and visual_weight query parameters (instead of color_weight).
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List, Optional
import cv2
import numpy as np
from pathlib import Path
import uuid
import logging
from modules.visual_searchv3 import clean_for_json
from modules.visual_searchv3 import search_similar_products
from models.schemas import ProductSearchResponse, ProductMatch
from fastapi.responses import JSONResponse
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


# @router.post("/search", response_model=ProductSearchResponse)
# async def search_product_v3(
#     query_image: UploadFile = File(...),
#     top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return"),
#     clip_weight: float = Query(default=0.4, ge=0.0, le=1.0, description="Weight for CLIP semantic similarity (0-1)"),
#     visual_weight: float = Query(default=0.6, ge=0.0, le=1.0, description="Weight for visual similarity (0-1)"),
#     category_filter: bool = Query(default=True, description="Filter results to same predicted category")
# ):
#     """
#     Upload a product image to find similar products in inventory using the V3 hybrid search
#     (CLIP + ResNet). Returns: Top-K similar products with similarity scores.
#     """
#     try:
#         # Convert upload to cv2 image (BGR numpy array)
#         img = await upload_to_cv2(query_image)
#         original_filename = query_image.filename or "query.jpg"
#         query_id = uuid.uuid4().hex[:8]

#         # Persist original upload
#         uploads_dir = Path(settings.UPLOAD_DIR) / "visual_search"
#         uploads_dir.mkdir(parents=True, exist_ok=True)
#         original_ext = Path(original_filename).suffix or ".jpg"
#         saved_query_name = f"{query_id}{original_ext}"
#         saved_query_path = uploads_dir / saved_query_name
#         cv2.imwrite(str(saved_query_path), img)
#         query_image_url = f"/uploads/visual_search/{saved_query_name}"

#         # Save a preview (resized for display)
#         try:
#             preview = cv2.resize(img, (400, 400))
#         except Exception:
#             preview = img
#         processed_dir = Path(settings.PROCESSED_DIR) / "visual_search"
#         processed_dir.mkdir(parents=True, exist_ok=True)
#         processed_name = f"{query_id}_processed.jpg"
#         processed_path = processed_dir / processed_name
#         cv2.imwrite(str(processed_path), preview)
#         query_processed_image_url = f"/processed/visual_search/{processed_name}"

#         # Inventory and index path from settings
#         inventory_dir = settings.INVENTORY_DIR
#         index_path = getattr(settings, "VISUAL_SEARCH_INDEX_PATH", None)

#         # Run visual search (v3 hybrid)
#         logger.info(
#             "Starting V3 visual search",
#             extra={"top_k": top_k, "clip_weight": clip_weight, "visual_weight": visual_weight}
#         )
#         try:
#             matches = search_similar_products(
#                 query_img=img,
#                 inventory_dir=inventory_dir,
#                 top_k=top_k,
#                 category_filter=category_filter,
#                 clip_weight=clip_weight,
#                 visual_weight=visual_weight,
#                 index_path=index_path,
#             )
#             logger.info(f"V3 search returned {len(matches)} matches")
#         except Exception as search_error:
#             logger.error(f"Error during V3 visual search: {search_error}", exc_info=True)
#             raise HTTPException(status_code=500, detail=f"Visual search failed: {str(search_error)}")

#         # Build response models
#         inventory_base_url = "/inventory"
#         match_models: List[ProductMatch] = []
#         try:
#             for match in matches:
#                 # v3 returns 'filename' and also compatibility keys; ensure we set product_image
#                 product_img = match.get("product_image") or match.get("filename")
#                 if not product_img:
#                     # skip entries without filename
#                     continue
#                 match["product_image"] = product_img
#                 match["product_image_url"] = f"{inventory_base_url}/{product_img}"

#                 # v3 sets color_similarity to 0.0 by default; convert 0-1 -> 0-100 for the API
#                 match["color_similarity"] = round(match.get("color_similarity", 0.0) * 100, 2)

#                 # Ensure semantic_similarity exists (map from clip_sim if necessary)
#                 if "semantic_similarity" not in match:
#                     match["semantic_similarity"] = match.get("clip_sim") or None

#                 # Ensure any missing keys required by ProductMatch are provided as None/defaults
#                 # (ProductMatch schema may validate fields; using dict unpacking)
#                 match_models.append(ProductMatch(**match))
#         except Exception as model_error:
#             logger.error(f"Error creating match models: {model_error}", exc_info=True)
#             raise HTTPException(status_code=500, detail=f"Error processing search results: {str(model_error)}")

#         return ProductSearchResponse(
#             matches=match_models,
#             total_matches=len(match_models),
#             query_image_path=str(saved_query_path),
#             query_image_url=query_image_url,
#             query_processed_image_path=str(processed_path),
#             query_processed_image_url=query_processed_image_url,
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in V3 product search: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/search")
async def search_product_v3(
    query_image: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top matches to return"),
    clip_weight: float = Query(default=0.4, ge=0.0, le=1.0, description="Weight for CLIP semantic similarity"),
    visual_weight: float = Query(default=0.6, ge=0.0, le=1.0, description="Weight for visual similarity"),
    category_filter: bool = Query(default=True, description="Filter results to same predicted category"),
    user_category: Optional[str] = Query(default=None, description="Manual category override from user (e.g. 'watch')")
):
    """
    V4 Smart Hybrid Search API.
    - Step 1: Upload image. If confidence is low, returns "alert_user": True.
    - Step 2: User confirms category. Frontend calls this API again with ?user_category=watch
    """
    try:
        # 1. Image Setup (Same as before)
        img = await upload_to_cv2(query_image)
        original_filename = query_image.filename or "query.jpg"
        query_id = uuid.uuid4().hex[:8]

        # Persist original
        uploads_dir = Path(settings.UPLOAD_DIR) / "visual_search"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        saved_query_name = f"{query_id}{Path(original_filename).suffix}"
        saved_query_path = uploads_dir / saved_query_name
        cv2.imwrite(str(saved_query_path), img)
        query_image_url = f"/uploads/visual_search/{saved_query_name}"

        # Save preview
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

        # 2. Run Smart Visual Search
        inventory_dir = settings.INVENTORY_DIR
        index_path = getattr(settings, "VISUAL_SEARCH_INDEX_PATH", "data/hybrid_v4_index.pkl")

        logger.info(f"Starting V4 search. User Category Override: {user_category}")

        try:
            # === CALLING THE NEW SMART BACKEND ===
            logger.info("Calling smart backend with {category_filter}, user_category={user_category}")
            search_payload = search_similar_products(
                query_img=img,
                inventory_dir=inventory_dir,
                top_k=top_k,
                category_filter=category_filter,
                clip_weight=clip_weight,
                visual_weight=visual_weight,
                user_category=user_category, # Pass the user input
                index_path=index_path,
            )
            
            # Extract list of matches from the payload
            matches = search_payload.get("results", [])
            status_status = search_payload.get("status", "success")
            
            logger.info(f"Search finished. Status: {status_status}, Matches: {len(matches)}")

        except Exception as search_error:
            logger.error(f"Error during V4 visual search: {search_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Visual search failed: {str(search_error)}")

        # 3. Process Matches (Format for Frontend)
        inventory_base_url = "/inventory"
        formatted_matches = []
        
        try:
            for match in matches:
                # Map keys to ensure frontend compatibility
                product_img = match.get("product_image") or match.get("filename")
                if not product_img: continue

                # Normalize scores
                score = match.get("score", 0.0)
                # Ensure 0-100 scale for confidence display
                match_conf = match.get("match_confidence", int(score * 100))

                formatted_match = {
                    "product_image": product_img,
                    "product_image_url": f"{inventory_base_url}/{product_img}",
                    "similarity_score": float(score),
                    "match_confidence": match_conf,
                    "category": match.get("category", "unknown"),
                    "semantic_similarity": match.get("clip_sim", 0.0),
                    "visual_similarity": match.get("visual_sim", 0.0),
                    "color_similarity": round(match.get("color_similarity", 0.0) * 100, 2),
                    # Pass through any metadata
                    "metadata": match.get("metadata", {})
                }
                formatted_matches.append(formatted_match)

        except Exception as model_error:
            logger.error(f"Error formatting matches: {model_error}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error formatting search results")

        # 4. Construct Smart Response
        # We use JSONResponse to handle the "alert_user" flag dynamically
        response_content = {
            "status": "success", 
            "alert_user": (status_status == "needs_confirmation"), # <--- FRONTEND LISTENS FOR THIS
            "alert_message": search_payload.get("message", ""),
            "detected_category": search_payload.get("category_used", "unknown"),
            "confidence": search_payload.get("confidence", 0.0),
            
            # Standard Data
            "matches": formatted_matches, # This list must be cleaned if it contains NumPy types
            "total_matches": len(formatted_matches),
            "query_image_url": query_image_url,
            "query_processed_image_url": query_processed_image_url
        }
        
        # Apply the cleaner to the entire dictionary before passing it to JSONResponse
        cleaned_content = clean_for_json(response_content)
        
        return JSONResponse(content=cleaned_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")