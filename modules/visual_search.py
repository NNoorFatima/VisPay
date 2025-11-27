"""
Visual search module: classic SIFT feature matching pipeline.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

from modules.preprocessing import preprocess_image_for_visual_search

logger = logging.getLogger(__name__)

MIN_MATCHES = 10  # minimal matches to consider a candidate
LOWE_RATIO = 0.7


def extract_features(img: np.ndarray):
    """Extract SIFT descriptors from a preprocessed grayscale image."""
    if img is None:
        return None
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors


def match_features(desc1, desc2):
    """Return good matches using FLANN + Lowe ratio test."""
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=60)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    return good_matches


def find_best_matches(
    query_img: np.ndarray,
    inventory_dir: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Return top matches based solely on SIFT match counts."""
    processed_query = preprocess_image_for_visual_search(query_img)
    query_desc = extract_features(processed_query)
    if query_desc is None or len(query_desc) == 0:
        logger.warning("No SIFT descriptors extracted from query image")
        return []
    
    inventory_path = Path(inventory_dir)
    if not inventory_path.exists():
        logger.error("Inventory directory does not exist: %s", inventory_dir)
        return []
    
    candidates: List[Dict[str, Any]] = []
    
    for filename in os.listdir(inventory_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        inv_path = inventory_path / filename
        inv_img = cv2.imread(str(inv_path))
        processed_inv = preprocess_image_for_visual_search(inv_img)
        inv_desc = extract_features(processed_inv)
        if inv_desc is None or len(inv_desc) == 0:
            continue
        
        good_matches = match_features(query_desc, inv_desc)
        score = len(good_matches)
        if score >= MIN_MATCHES:
            confidence = min(100, score)  # simple mapping
            candidates.append({
                "product_image": filename,
                "similarity_score": score,
                "match_confidence": confidence,
                "feature_method": "SIFT",
                "color_similarity": None
            })
        else:
            logger.debug("Rejected %s (matches=%d < %d)", filename, score, MIN_MATCHES)
    
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return candidates[:top_k]


def search_similar_products(
    query_img: np.ndarray,
    inventory_dir: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    return find_best_matches(query_img, inventory_dir, top_k)

