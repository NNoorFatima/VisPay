"""
Visual search module for product matching using SIFT features.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

from modules.preprocessing import preprocess_image_for_visual_search

logger = logging.getLogger(__name__)


def extract_features(img: np.ndarray) -> Tuple[Optional[list], Optional[np.ndarray]]:
    """
    Extract SIFT features from image.
    
    Args:
        img: Preprocessed grayscale image
    
    Returns:
        Tuple of (keypoints, descriptors) or (None, None) if extraction fails
    """
    if img is None:
        return None, None
    
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return keypoints, descriptors
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None, None


def match_features(desc1: np.ndarray, desc2: np.ndarray) -> List:
    """
    Match features between two descriptors using FLANN matcher.
    
    Args:
        desc1: Descriptors from query image
        desc2: Descriptors from inventory image
    
    Returns:
        List of good matches (after Lowe's ratio test)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test to keep good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    except Exception as e:
        logger.error(f"Error matching features: {e}")
        return []


def find_best_matches(
    query_img: np.ndarray,
    inventory_dir: str,
    top_k: int = 5
) -> List[Tuple[str, int]]:
    """
    Find best matching products for a query image.
    
    Args:
        query_img: Query image as numpy array (BGR format)
        inventory_dir: Directory containing inventory product images
        top_k: Number of top matches to return
    
    Returns:
        List of tuples (filename, match_score) sorted by score descending
    """
    # Preprocess query image
    processed_query = preprocess_image_for_visual_search(query_img)
    if processed_query is None:
        logger.warning("Query image preprocessing failed")
        return []
    
    # Extract features from query
    _, query_desc = extract_features(processed_query)
    if query_desc is None or len(query_desc) == 0:
        logger.warning("No features extracted from query image")
        return []
    
    results = []
    inventory_path = Path(inventory_dir)
    
    if not inventory_path.exists():
        logger.error(f"Inventory directory does not exist: {inventory_dir}")
        return []
    
    # Process each image in inventory
    for filename in os.listdir(inventory_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        inv_path = inventory_path / filename
        inv_img = cv2.imread(str(inv_path))
        
        if inv_img is None:
            continue
        
        # Preprocess inventory image
        processed_inv = preprocess_image_for_visual_search(inv_img)
        if processed_inv is None:
            continue
        
        # Extract features
        _, inv_desc = extract_features(processed_inv)
        if inv_desc is None or len(inv_desc) == 0:
            continue
        
        # Match features
        good_matches = match_features(query_desc, inv_desc)
        score = len(good_matches)  # More matches = more similar
        
        if score > 0:
            results.append((filename, score))
    
    # Sort by descending match score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def search_similar_products(
    query_img: np.ndarray,
    inventory_dir: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar products and return formatted results.
    
    Args:
        query_img: Query image as numpy array (BGR format)
        inventory_dir: Directory containing inventory product images
        top_k: Number of top matches to return
    
    Returns:
        List of dictionaries with product info and similarity scores
    """
    matches = find_best_matches(query_img, inventory_dir, top_k)
    
    results = []
    for filename, score in matches:
        results.append({
            "product_image": filename,
            "similarity_score": score,
            "match_confidence": min(100, int((score / 100) * 100)) if score > 0 else 0
        })
    
    return results

