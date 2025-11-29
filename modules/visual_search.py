"""
Visual search module using ResNet50 deep features + color histograms.
Based on improved_visuasearch-module.ipynb
"""
import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract deep learning features using pretrained ResNet50"""
    
    def __init__(self, model_name='resnet50'):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load pretrained model
            if model_name == 'resnet50':
                logger.info("Loading ResNet50 model...")
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                # Remove final classification layer to get features
                self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("ResNet50 model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing FeatureExtractor: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize feature extractor: {str(e)}")
        
        # Image preprocessing pipeline (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, img_path: str) -> Optional[np.ndarray]:
        """Extract feature vector from image file path."""
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)  # L2 normalization
            
            return features
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None
    
    def extract_features_from_array(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature vector from numpy array (BGR format from cv2)."""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features from array: {e}")
            return None
    
    def extract_color_histogram(self, img_path: str, bins: int = 32) -> Optional[np.ndarray]:
        """Extract color histogram features from image file."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Calculate histogram for each channel
            hist_features = []
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
                hist = hist.flatten()
                hist_features.extend(hist)
            
            hist_features = np.array(hist_features)
            hist_features = hist_features / np.linalg.norm(hist_features)
            
            return hist_features
        except Exception as e:
            logger.error(f"Error extracting color from {img_path}: {e}")
            return None
    
    def extract_color_histogram_from_array(self, img_array: np.ndarray, bins: int = 32) -> Optional[np.ndarray]:
        """Extract color histogram features from numpy array."""
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            hist_features = []
            for i in range(3):
                hist = cv2.calcHist([img_resized], [i], None, [bins], [0, 256])
                hist = hist.flatten()
                hist_features.extend(hist)
            
            hist_features = np.array(hist_features)
            hist_features = hist_features / np.linalg.norm(hist_features)
            
            return hist_features
        except Exception as e:
            logger.error(f"Error extracting color from array: {e}")
            return None


class FashionSearchIndex:
    """Build and search index of fashion products"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.extractor = feature_extractor
        self.features_db: Dict[str, np.ndarray] = {}
        self.color_db: Dict[str, np.ndarray] = {}
        self.image_paths: List[str] = []
    
    def build_index(self, inventory_dir: str, save_path: Optional[str] = None):
        """Build index from inventory images."""
        logger.info("Building search index...")
        
        inventory_path = Path(inventory_dir)
        if not inventory_path.exists():
            logger.error(f"Inventory directory does not exist: {inventory_dir}")
            return
        
        image_files = [f for f in os.listdir(inventory_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        self.features_db = {}
        self.color_db = {}
        self.image_paths = []
        
        for img_file in image_files:
            img_path = str(inventory_path / img_file)
            
            # Extract deep features
            features = self.extractor.extract_features(img_path)
            if features is not None:
                self.features_db[img_file] = features
            
            # Extract color features
            color_features = self.extractor.extract_color_histogram(img_path)
            if color_features is not None:
                self.color_db[img_file] = color_features
            
            self.image_paths.append(img_path)
        
        logger.info(f"Indexed {len(self.features_db)} images")
        
        # Save index if path provided
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, save_path: str):
        """Save index to disk."""
        index_data = {
            'features_db': self.features_db,
            'color_db': self.color_db,
            'image_paths': self.image_paths
        }
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, load_path: str):
        """Load index from disk."""
        with open(load_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.features_db = index_data['features_db']
        self.color_db = index_data.get('color_db', {})
        self.image_paths = index_data.get('image_paths', [])
        logger.info(f"Index loaded from {load_path} ({len(self.features_db)} images)")
    
    def search(
        self, 
        query_img: np.ndarray, 
        inventory_dir: str,
        top_k: int = 5, 
        color_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        """
        if not self.features_db:
            logger.warning("Search index is empty. No products to search against.")
            return []
        
        # Extract query features from array
        try:
            query_features = self.extractor.extract_features_from_array(query_img)
            query_color = self.extractor.extract_color_histogram_from_array(query_img)
        except Exception as e:
            logger.error(f"Error extracting features from query image: {e}", exc_info=True)
            raise ValueError(f"Failed to extract features from query image: {str(e)}")
        
        if query_features is None:
            logger.warning("Could not extract features from query image")
            return []
        
        # Calculate similarities
        results = []
        try:
            for img_file, inv_features in self.features_db.items():
                try:
                    # Semantic similarity (cosine similarity)
                    semantic_sim = np.dot(query_features, inv_features)
                    
                    # Color similarity
                    color_sim = 0.0
                    if query_color is not None and img_file in self.color_db:
                        color_sim = np.dot(query_color, self.color_db[img_file])
                    
                    # Combined score
                    combined_score = (1 - color_weight) * semantic_sim + color_weight * color_sim
                    
                    results.append({
                        "product_image": img_file,
                        "similarity_score": float(combined_score),
                        "semantic_similarity": float(semantic_sim),
                        "color_similarity": float(color_sim),
                        "match_confidence": int(min(100, max(0, combined_score * 100))),
                        "feature_method": "ResNet50"
                    })
                except Exception as e:
                    logger.warning(f"Error processing {img_file} in search: {e}")
                    continue
            
            # Sort by combined score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error during similarity calculation: {e}", exc_info=True)
            raise ValueError(f"Failed to calculate similarities: {str(e)}")


# Global feature extractor instance (lazy loaded)
_extractor: Optional[FeatureExtractor] = None
_index: Optional[FashionSearchIndex] = None


def get_extractor() -> FeatureExtractor:
    """Get or create global feature extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor(model_name='resnet50')
    return _extractor


def get_index(inventory_dir: str, index_path: Optional[str] = None) -> FashionSearchIndex:
    """Get or create global search index."""
    global _index
    if _index is None:
        try:
            extractor = get_extractor()
            _index = FashionSearchIndex(extractor)
            
            # Try to load existing index
            if index_path and os.path.exists(index_path):
                try:
                    logger.info(f"Loading index from {index_path}")
                    _index.load_index(index_path)
                    if not _index.features_db:
                        logger.warning("Loaded index is empty. Rebuilding...")
                        _index.build_index(inventory_dir, save_path=index_path)
                except Exception as e:
                    logger.warning(f"Could not load index from {index_path}: {e}. Building new index...")
                    _index.build_index(inventory_dir, save_path=index_path)
            else:
                # Build new index
                logger.info(f"Building new index from {inventory_dir}")
                _index.build_index(inventory_dir, save_path=index_path)
            
            if not _index.features_db:
                logger.error("Index built but contains no features. Check inventory directory and image files.")
        except Exception as e:
            logger.error(f"Error initializing search index: {e}", exc_info=True)
            raise ValueError(f"Failed to initialize search index: {str(e)}")
    
    return _index


def search_similar_products(
    query_img: np.ndarray,
    inventory_dir: str,
    top_k: int = 5,
    color_weight: float = 0.3,
    index_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar products using ResNet50 + color histograms.
    """
    index = get_index(inventory_dir, index_path)
    return index.search(query_img, inventory_dir, top_k=top_k, color_weight=color_weight)
