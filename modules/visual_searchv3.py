"""
Version 4.1 - Production Ready High-Accuracy Hybrid Fashion Search
Updates:
- Fixed API compatibility with V3 routes (accepts category_filter, weights)
- Maps output keys to V3 schema (product_image, similarity_score) to prevent frontend errors
- Optimized OpenCLIP ViT-H-14 usage
- Robust FAISS Indexing for variable dataset sizes
"""

import os
import pickle
import random
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# OpenCLIP + FAISS
try:
    import open_clip
    import faiss
except ImportError as e:
    raise ImportError("Missing dependencies. Please install: pip install open_clip_torch faiss-gpu") from e

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ==================== Utilities ====================
# --- modules/visual_searchv3.py (Insert near the top utilities) ---

def clean_for_json(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Recursively converts NumPy arrays and types to standard Python types 
    (list/float) to ensure JSON serializability.
    """
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(item) for item in data]
    # Check for numpy array or numpy numeric types (e.g., np.float64, np.int32)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.generic, np.number)):
        return data.item()
    else:
        return data

# --- END of Helper Function ---
def _ensure_pil(img: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
    """Robustly convert any image input to PIL RGB."""
    try:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, str):
            if not os.path.exists(img):
                logger.warning(f"Image path not found: {img}")
                return None
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            # Handle OpenCV BGR to RGB
            if img.ndim == 3 and img.shape[2] == 3:
                import cv2
                # Check if it looks like BGR (simple heuristic or just assume standard cv2 usage)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
            return Image.fromarray(img).convert("RGB")
    except Exception as e:
        logger.warning(f"_ensure_pil error: {e}")
        return None


# ==================== Upgraded CLIP Matcher (ViT-H-14) ====================
class CLIPMatcher:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading OpenCLIP ViT-H-14 (laion2B) on {self.device}...")

        # Load heavy model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.model.eval()

        # Definitions
        # EXPANDED CATEGORIES (Crucial for minimizing "Unknowns")
        self.category_templates = {
            "dress": ["a photo of a dress", "a woman wearing a dress", "evening dress"],
            "shirt": ["a shirt", "button-up shirt", "casual shirt"],
            "tshirt": ["a t-shirt", "graphic tee", "casual tshirt"],
            "pants": ["pants", "trousers", "jeans", "denim pants"],
            "shorts": ["shorts", "denim shorts", "athletic shorts"],
            "skirt": ["a skirt", "mini skirt", "maxi skirt"],
            "jacket": ["a jacket", "denim jacket", "leather jacket", "blazer"],
            "sweater": ["a sweater", "knit sweater", "hoodie"],
            "shoes": ["shoes", "sneakers", "boots", "heels"],
            "bag": ["a handbag", "purse", "backpack", "tote bag"],
            "activewear": ["sports bra", "leggings", "yoga pants", "gym wear"],
            "outerwear": ["coat", "trench coat", "puffer jacket"],
            "swimwear": ["bikini", "swim trunks", "swimsuit"],
            "watch": ["a wrist watch", "luxury watch", "smart watch"], 
            "jewelry": ["necklace", "earrings", "pendant", "jewelry"],
            "accessory": ["sunglasses", "hat", "cap", "belt", "scarf", "wallet"],
            "formal": ["suit", "tuxedo", "formal wear"]
        }
        
        self.color_templates = [
            f"a photo of {c} clothing" for c in [
                "red", "blue", "black", "white", "green", "yellow", "pink", "purple",
                "orange", "gray", "beige", "brown", "navy", "gold", "silver", "teal", "maroon"
            ]
        ]

    def get_image_embedding(self, img_input) -> Optional[np.ndarray]:
        img = _ensure_pil(img_input)
        if img is None:
            return None
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def _ensemble_predict(self, img_input, templates: List[str]) -> Tuple[str, float]:
        img = _ensure_pil(img_input)
        if img is None:
            return "unknown", 0.0
        
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer(templates).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)
            
        top_idx = probs.argmax().item()
        
        # Clean up the prediction string
        raw_pred = templates[top_idx]
        clean_pred = raw_pred.replace("a photo of a ", "").replace("a photo of ", "")
        
        return clean_pred, float(probs[top_idx])

    # def predict_category(self, img_input) -> Tuple[str, float]:
    #     # Flatten templates for prediction
    #     flat_templates = []
    #     labels_map = [] # stores index -> category key
    #     for label, tmpls in self.category_templates.items():
    #         for t in tmpls:
    #             flat_templates.append(t)
    #             labels_map.append(label)
    #     # We predict the specific template
    #     _, _ = self._ensemble_predict(img_input, flat_templates)
    #     # But we need to return the parent category (e.g., "dress" not "evening dress")
    #     # Re-running logic slightly to map back to key
    #     img = _ensure_pil(img_input)
    #     if img is None: return "unknown", 0.0
    #     image_input = self.preprocess(img).unsqueeze(0).to(self.device)
    #     text_inputs = self.tokenizer(flat_templates).to(self.device)
    #     with torch.no_grad():
    #         image_features = self.model.encode_image(image_input)
    #         text_features = self.model.encode_text(text_inputs)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         text_features /= text_features.norm(dim=-1, keepdim=True)
    #         probs = (image_features @ text_features.T).squeeze(0).softmax(dim=-1)
    #     top_idx = probs.argmax().item()
    #     predicted_label = labels_map[top_idx]
    #     print(f"Predicted category: {predicted_label}")
    #     return predicted_label, float(probs[top_idx])
    def predict_category(self, img_input, threshold: float = 0.02) -> Tuple[str, float]:
        """
        Smart prediction with Confidence Thresholding.
        Returns ('unknown', conf) if below threshold.
        """
        # Flatten templates
        flat_templates = []
        labels_map = [] 
        for label, tmpls in self.category_templates.items():
            for t in tmpls:
                flat_templates.append(t)
                labels_map.append(label)
                
        img = _ensure_pil(img_input)
        if img is None: return "unknown", 0.0

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer(flat_templates).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # Calculate probabilities
            probs = (image_features @ text_features.T).squeeze(0).softmax(dim=-1)

        top_idx = probs.argmax().item()
        confidence = float(probs[top_idx])
        predicted_label = labels_map[top_idx]

        # === SMART LOGIC ===
        if confidence < threshold:
            logger.warning(f"Low confidence ({confidence:.2f}) for '{predicted_label}'. Returning 'unknown'.")
            # We still return the 'best guess' in the second slot just in case the UI wants to show "Did you mean...?"
            return "unknown", confidence
            
        return predicted_label, confidence
    def predict_color(self, img_input) -> Tuple[str, float]:
        return self._ensemble_predict(img_input, self.color_templates)


# ==================== Visual Extractor (ResNet50) ====================
class VisualFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load standard ResNet
        weights = 'IMAGENET1K_V2'
        try:
            resnet = torch.hub.load('pytorch/vision:v0.15.2', 'resnet50', weights=weights)
        except:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, img_input) -> Optional[np.ndarray]:
        img = _ensure_pil(img_input)
        if img is None:
            return None
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model(x).flatten().cpu().numpy()
            f = f / np.linalg.norm(f)
        return f


# ==================== MAIN HYBRID SEARCH SYSTEM ====================
class HybridFashionSearch:
    def __init__(self, device=None, lazy=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lazy = lazy

        self.clip: Optional[CLIPMatcher] = None
        self.visual: Optional[VisualFeatureExtractor] = None

        self.inventory_data: Dict[str, Dict] = {}
        self.category_index: Dict[str, List[str]] = {}
        self.faiss_indices: Dict[str, faiss.Index] = {} 

        if not self.lazy:
            self._init_models()

    def _init_models(self):
        """Lazy loader for heavy AI models"""
        if self.clip is None:
            self.clip = CLIPMatcher(device=self.device)
        if self.visual is None:
            self.visual = VisualFeatureExtractor(device=self.device)

    def build_index(self, inventory_dir: str, save_path: str = "data/hybrid_v4_index.pkl"):
        """Scans folder, predicts categories, extracts features, builds FAISS."""
        logger.info("Building V4 hybrid index...")
        self._init_models()
        
        if not os.path.exists(inventory_dir):
            raise FileNotFoundError(inventory_dir)

        files = [f for f in os.listdir(inventory_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            logger.warning("No images found in inventory directory.")
            return

        for fname in tqdm(files, desc="Indexing"):
            path = os.path.join(inventory_dir, fname)
            try:
                # 1. Predict Category (CLIP)
                cat, _ = self.clip.predict_category(path)
                
                # 2. Extract Embeddings
                clip_feat = self.clip.get_image_embedding(path)
                vis_feat = self.visual.extract_features(path)

                if clip_feat is None or vis_feat is None:
                    continue

                self.inventory_data[fname] = {
                    "clip_features": clip_feat,
                    "visual_features": vis_feat,
                    "category": cat,
                }
                self.category_index.setdefault(cat, []).append(fname)
            except Exception as e:
                logger.error(f"Error indexing {fname}: {e}")

        # Build FAISS index per category
        for cat, fnames in self.category_index.items():
            vectors = np.array([self.inventory_data[f]["clip_features"] for f in fnames]).astype('float32')
            
            # Robust FAISS creation based on dataset size
            d = vectors.shape[1]
            num_samples = vectors.shape[0]
            
            if num_samples < 100:
                # Use simple FlatL2 for small categories
                index = faiss.IndexFlatL2(d)
            else:
                # Use IVF for larger categories
                quantizer = faiss.IndexFlatL2(d)
                # Ensure nlist is not larger than num_samples
                nlist = min(100, int(num_samples / 2))
                if nlist < 1: nlist = 1
                index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8) 
                index.train(vectors)
                
            index.add(vectors)
            
            if hasattr(index, 'nprobe'):
                index.nprobe = 10
                
            self.faiss_indices[cat] = index
            self.inventory_data[cat + "_faiss_map"] = fnames

        # Save to disk
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "inventory_data": self.inventory_data,
                "category_index": self.category_index,
            }, f)
        logger.info(f"Index built and saved to {save_path}")

    def load_index(self, load_path: str):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Index file {load_path} not found.")
            
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            
        self.inventory_data = data["inventory_data"]
        self.category_index = data["category_index"]
        
        # Rebuild FAISS indices in memory
        logger.info("Reconstructing FAISS indices...")
        for cat, fnames in self.category_index.items():
            vectors = []
            valid_fnames = []
            for f in fnames:
                if f in self.inventory_data:
                    vectors.append(self.inventory_data[f]["clip_features"])
                    valid_fnames.append(f)
            
            if not vectors: continue
            
            vectors = np.array(vectors).astype('float32')
            d = vectors.shape[1]
            num_samples = vectors.shape[0]
            
            if num_samples < 100:
                index = faiss.IndexFlatL2(d)
            else:
                quantizer = faiss.IndexFlatL2(d)
                nlist = min(100, int(num_samples / 2))
                if nlist < 1: nlist = 1
                index = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8)
                index.train(vectors)
            
            index.add(vectors)
            if hasattr(index, 'nprobe'):
                index.nprobe = 10
            
            self.faiss_indices[cat] = index
            self.inventory_data[cat + "_faiss_map"] = valid_fnames

    def search(
        self,
        query_input,
        top_k: int = 8,
        user_selected_category: Optional[str] = None,
        category_filter: bool = True
    ) -> Tuple[List[Dict], str]:
        
        self._init_models() # Ensure models are loaded

        # === 1. Input Normalization (Fixes "Watch" vs "watch") ===
        if user_selected_category:
            user_selected_category = user_selected_category.lower().strip()
            # Handle simple plurals (e.g., "watches" -> "watch")
            if user_selected_category.endswith('es'): 
                user_selected_category = user_selected_category[:-2]
            elif user_selected_category.endswith('s') and not user_selected_category.endswith('ss'): 
                user_selected_category = user_selected_category[:-1]

        query_pil = _ensure_pil(query_input)
        if query_pil is None:
            return [], "Error: Could not read query image."

        # === 2. Analyze Query ===
        query_clip = self.clip.get_image_embedding(query_pil)
        query_visual = self.visual.extract_features(query_pil)
        pred_category, conf = self.clip.predict_category(query_pil)
        pred_color, _ = self.clip.predict_color(query_pil)

        # Handle failed feature extraction
        if query_clip is None or query_visual is None:
             return [], "Error: Failed to extract features from query image."

        # === 3. Determine Search Space ===
        use_category = user_selected_category or pred_category
        logger.info(f"Search Category: '{use_category}' (User Selected: {user_selected_category is not None})")

        # === 4. Category Availability Check ===
        if use_category not in self.faiss_indices:
            # Scenario A: User manually asked for category not in inventory.
            if user_selected_category:
                logger.warning(f"User requested '{user_selected_category}' but it is not in inventory.")
                return [], f"Sorry, we have no products in the category: '{user_selected_category}'"
            
            # Scenario B: AI guessed "Unknown" or a category you don't stock.
            # ACTION: Fallback to Smart Search over entire inventory (Better UX).
            if category_filter:
                # === GENIUS CODER FIX 1 (Replacement) ===
                fallback_results = self._get_smart_fallback(query_clip, top_k, pred_color)
                
                if fallback_results:
                     return fallback_results, f"Category '{use_category}' not in stock. Showing smart fallback items."
                else:
                     return [], f"Category '{use_category}' not in stock. No items found for fallback."
                # === END GENIUS CODER FIX 1 ===

        # === 5. Perform FAISS Search ===
        if use_category in self.faiss_indices:
            index = self.faiss_indices[use_category]
            map_list = self.inventory_data[use_category + "_faiss_map"]

            # Retrieve candidates
            D, I = index.search(query_clip.reshape(1, -1).astype('float32'), top_k * 4)
            
            candidates = []
            if len(I) > 0:
                for idx_val in I[0]:
                    if 0 <= idx_val < len(map_list):
                        candidates.append(map_list[idx_val])

            results = []
            for fname in candidates:
                item = self.inventory_data[fname]
                
                # Calculate Similarities
                clip_sim = float(np.dot(query_clip, item["clip_features"]))
                vis_sim = float(np.dot(query_visual, item["visual_features"]))
                
                # Reciprocal Rank Fusion
                fused_score = (clip_sim * 0.5) + (vis_sim * 0.5)

                results.append({
                    "filename": fname,
                    "product_image": fname,
                    "score": fused_score,
                    "similarity_score": fused_score,
                    "clip_sim": clip_sim,
                    "semantic_similarity": clip_sim,
                    "visual_sim": vis_sim,
                    "visual_similarity": vis_sim,
                    "category": item["category"],
                    "predicted_color": pred_color,
                    "match_confidence": int(fused_score * 100),
                })

            # Sort by fused score
            results.sort(key=lambda x: x["score"], reverse=True)

            # === GENIUS CODER FIX: Deduplicate Augmented Results ===
            final_matches = []
            seen_source_files = set()
            
            for match in results:
                # 1. Look up the original filename (e.g., '11766.jpg')
                #    We retrieve the source_file from the inventory data using the indexed filename (fname)
                indexed_fname = match['filename']
                source_file = self.inventory_data.get(indexed_fname, {}).get("source_file")
                
                # Fallback check (shouldn't happen if indexing is correct, but good for safety)
                if not source_file:
                    source_file = indexed_fname

                # 2. If this product hasn't been seen, add it
                if source_file not in seen_source_files:
                    seen_source_files.add(source_file)
                    
                    # Ensure the final match structure uses the original source image 
                    # for the product_image key, so the frontend loads the correct, 
                    # non-augmented image URL.
                    match['product_image'] = source_file
                    match['filename'] = source_file # Ensure filename is also the original
                    
                    final_matches.append(match)
                
                # 3. Stop once we have reached the desired top_k
                if len(final_matches) >= top_k:
                    break
            
            status = f"Found in {use_category}"
            return final_matches, status
            # === END GENIUS CODER FIX ===

            # status = f"Found in {use_category}"
            # return results[:top_k], status
            
        else:
            # Catch-all fallback (This line is generally defensive and should be hit less often than 4.B)
            # === GENIUS CODER FIX 2 (Replacement) ===
            fallback_results = self._get_smart_fallback(query_clip, top_k, pred_color)
            return fallback_results, f"Category '{use_category}' unknown, showing smart fallback."
            # === END GENIUS CODER FIX 2 ===
    def _get_smart_fallback(self, query_clip: np.ndarray, top_k: int, pred_color: str) -> List[Dict]:
        """
        Performs a smart fallback search over ALL inventory items using the CLIP vector.
        This provides visually/semantically relevant items even when the category is unknown/not in inventory.
        """
        
        # Check if a combined index for ALL CLIP vectors exists
        if 'all_clip_vectors' not in self.faiss_indices:
            # NOTE: For production, this should be pre-calculated in build_index.
            # This is a one-time build if it doesn't exist.
            logger.info("Building all-inventory CLIP index for fallback...")
            
            all_fnames = [k for k in self.inventory_data.keys() if not k.endswith("_faiss_map")]
            if not all_fnames: return []
            
            # Collect all CLIP vectors
            all_vectors = np.array([
                self.inventory_data[f]["clip_features"] 
                for f in all_fnames 
                if "clip_features" in self.inventory_data[f]
            ]).astype('float32')

            if all_vectors.size == 0: return []
            
            d = all_vectors.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(all_vectors)
            self.faiss_indices['all_clip_vectors'] = index
            self.inventory_data['all_clip_map'] = all_fnames
            
        index = self.faiss_indices['all_clip_vectors']
        map_list = self.inventory_data['all_clip_map']

        # Perform search against the entire index
        D, I = index.search(query_clip.reshape(1, -1).astype('float32'), top_k)
        
        results = []
        if len(I) > 0:
            for idx_val in I[0]:
                if 0 <= idx_val < len(map_list):
                    fname = map_list[idx_val]
                    item = self.inventory_data[fname]
                    
                    # Similarity score is derived from distance D (L2 distance), convert to score (1 - distance_normalized)
                    # For simplicity here, we'll just use the normalized CLIP score later.
                    clip_sim = float(np.dot(query_clip, item["clip_features"]))
                    
                    results.append({
                        "filename": fname,
                        "product_image": fname,
                        "score": clip_sim,
                        "similarity_score": clip_sim,
                        "clip_sim": clip_sim,
                        "semantic_similarity": clip_sim,
                        "visual_sim": item.get("visual_features", 0.0), # No explicit visual search was done
                        "category": item.get("category", "unknown"),
                        "reason": "smart_fallback_clip",
                        "predicted_color": pred_color,
                        "match_confidence": int(clip_sim * 100),
                    })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        # return results[:top_k]
        # === GENIUS CODER FIX: Deduplicate Augmented Results in Fallback ===
        final_matches = []
        seen_source_files = set()
        
        for match in results:
            indexed_fname = match['filename']
            source_file = self.inventory_data.get(indexed_fname, {}).get("source_file")
            
            if not source_file:
                source_file = indexed_fname

            if source_file not in seen_source_files:
                seen_source_files.add(source_file)
                
                # Ensure the product image URL points to the non-augmented source file
                match['product_image'] = source_file
                match['filename'] = source_file 
                
                final_matches.append(match)
            
            # Stop once we have reached the desired top_k
            if len(final_matches) >= top_k:
                break
        
        return final_matches
        # === END GENIUS CODER FIX ===
    # def _get_random_fallback(self, top_k, color):
    #     all_items = [k for k in self.inventory_data.keys() if not k.endswith("_faiss_map")]
    #     if not all_items: return []
    #     random_items = random.sample(all_items, min(top_k, len(all_items)))
    #     results = []
    #     for f in random_items:
    #         item = self.inventory_data[f]
    #         results.append({
    #             "filename": f,
    #             "product_image": f,
    #             "score": 0.0,
    #             "similarity_score": 0.0,
    #             "category": item.get("category", "unknown"),
    #             "reason": "fallback",
    #             "predicted_color": color,
    #         })
    #     return results


# ==================== Public API (V3 Compatible) ====================
_global_index: Optional[HybridFashionSearch] = None

def get_index(inventory_dir: str, index_path: str = "data/hybrid_v4_index.pkl") -> HybridFashionSearch:
    global _global_index
    if _global_index is not None:
        return _global_index

    if os.path.exists(index_path):
        logger.info("Loading existing V4 index...")
        idx = HybridFashionSearch(lazy=True)
        idx.load_index(index_path)
        _global_index = idx
    else:
        logger.info("Building new V4 index...")
        idx = HybridFashionSearch(lazy=False)
        idx.build_index(inventory_dir, index_path)
        _global_index = idx
    return _global_index

# --- modules/visual_searchv3.py (Inside HybridFashionSearch Class) ---


def search_similar_products(
    query_img,
    inventory_dir: str,
    top_k: int = 8,
    category_filter: bool = True, 
    clip_weight: float = 0.5,
    visual_weight: float = 0.5, 
    user_category: Optional[str] = None,
    index_path: str = "data/hybrid_v4_index.pkl",
    meta_df: Optional[Any] = None
) -> Dict[str, Any]:
    
    idx = get_index(inventory_dir, index_path)
    
    # === CRITICAL FIX: Ensure models are loaded before accessing idx.clip ===
    # This wakes up the GPU/RAM models if they were sleeping (Lazy Loading)
    idx._init_models() 
    status_message = ""
    results = []
    # === WORKFLOW LOGIC ===
    # 1B. Perform the search
    try:
        # This is the line that should define the variables
        results, status_message = idx.search(query_img, top_k=top_k, user_selected_category=user_category, category_filter=True)
    except Exception as e:
        # If the search call fails entirely, we ensure status_message is set
        logger.error(f"Error during user-category search: {e}", exc_info=True)
        status_message = f"Search failed for category '{user_category}': {str(e)}"
    # 1. If User provided a category (User corrected the AI), strictly use it.
    # === GENIUS CODER: Robust Logic Check ===
        final_status = "success"
        if not results:
            final_status = "no_matches"
            
            # Overwrite a generic "Found in X" status with the specific 'No products found' message.
            if not status_message or "Found in" in status_message: 
                status_message = f"No products found in the category: '{user_category}'."
        # === END GENIUS CODER: Robust Logic Check ===
        
        return {
            "status": final_status, # Use the determined status
            "category_used": user_category,
            "results": results,
            "message": status_message # Use the determined message
        }

    # 2. If no user category, try Auto-Detect (Smart Mode)
    img_pil = _ensure_pil(query_img)
    
    # Now this will work because idx.clip is no longer None
    predicted_cat, conf = idx.clip.predict_category(img_pil, threshold=0.01)
    
    # 3. CHECK THRESHOLD: If AI is confused ("unknown")
    if predicted_cat == "unknown":
        # We perform a "fallback" search (random or broad) BUT we warn the user
        results, status = idx.search(query_img, top_k=top_k, user_selected_category=None, category_filter=False)
        
        return {
            "status": "needs_confirmation", 
            "category_used": "unknown",
            "confidence": conf,
            "message": "Fetched relevant products, hopefully!",
            "results": results, 
            "prompt_user": True 
        }

    # 4. Standard Success Case (High Confidence)
    results, status = idx.search(query_img, top_k=top_k, user_selected_category=predicted_cat, category_filter=True)
    return {
        "status": "success",
        "category_used": predicted_cat,
        "confidence": conf,
        "results": results,
        "message": status
    }
    
# def search_similar_products(
#     query_img,
#     inventory_dir: str,
#     top_k: int = 8,
#     category_filter: bool = True, # ADDED: V3 compatibility
#     clip_weight: float = 0.5,     # ADDED: V3 compatibility
#     visual_weight: float = 0.5,   # ADDED: V3 compatibility
#     user_category: Optional[str] = None,
#     index_path: str = "data/hybrid_v4_index.pkl",
#     meta_df: Optional[Any] = None # ADDED: V3 compatibility (ignored in V4 but accepted)
# ) -> List[Dict]:
#     """
#     Wrapper function that matches the signature expected by routes/product.py
#     """
#     # 1. Get or Create Index
#     idx = get_index(inventory_dir, index_path)
    
#     # 2. Perform V4 Search
#     # Note: V4 uses Rank Fusion, so explicit weights might be less impactful, 
#     # but we pass category_filter down.
#     results, status_msg = idx.search(
#         query_img, 
#         top_k=top_k, 
#         user_selected_category=user_category,
#         category_filter=category_filter
#     )
    
#     logger.info(f"Search Status: {status_msg}")
    
#     # 3. Return Results (List[Dict])
#     # The dictionary keys inside `results` are already mapped to be V3 compatible 
#     # inside the HybridFashionSearch.search method.
#     return results

# Optional: Helper to get extractor if needed externally
def get_extractor():
    idx = get_index("dummy") # This is a bit hacky, but V4 integrates extractor inside
    if idx.visual is None:
        idx._init_models()
    return idx.visual


# # """
# Version 3 hybrid visual search implementation (ported from notebook).

# Public API (module-level functions):
# - get_extractor() -> returns visual feature extractor instance (VisualFeatureExtractor)
# - get_index(inventory_dir, index_path=None, meta_df=None) -> returns HybridFashionSearch instance
# - search_similar_products(
#         query_img, inventory_dir, top_k=5, category_filter=True,
#         clip_weight=0.4, visual_weight=0.6, index_path=None, meta_df=None
#   ) -> List[Dict[str, Any]]

# Notes:
# - query_img may be:
#     - a filesystem path (str) to an image
#     - a numpy array (BGR as produced by cv2)
#     - a PIL.Image.Image instance
# - Index is saved/loaded as a pickle file. Default save path is "data/hybrid_fashion_index.pkl".
# - Results include compatibility fields with v1 (product_image, similarity_score, semantic_similarity)
#   and v3-specific fields (filename, score, clip_sim, visual_sim, category, metadata).
# """

# import os
# import io
# import pickle
# import logging
# from typing import Any, Dict, List, Optional, Union

# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torchvision import models, transforms

# # Transformers (BLIP + CLIP)
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from transformers import CLIPProcessor, CLIPModel

# logger = logging.getLogger(__name__)


# def _ensure_pil(img: Union[str, np.ndarray, Image.Image]) -> Optional[Image.Image]:
#     """Convert path / BGR numpy array / PIL Image to PIL RGB Image."""
#     try:
#         if isinstance(img, Image.Image):
#             return img.convert("RGB")
#         if isinstance(img, str):
#             return Image.open(img).convert("RGB")
#         if isinstance(img, np.ndarray):
#             # assume BGR (cv2). Convert to RGB
#             if img.ndim == 3 and img.shape[2] == 3:
#                 import cv2
#                 rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 return Image.fromarray(rgb).convert("RGB")
#             else:
#                 # grayscale or unexpected shape
#                 return Image.fromarray(img).convert("RGB")
#     except Exception as e:
#         logger.warning(f"_ensure_pil: could not convert input to PIL Image: {e}")
#         return None

# # ==================== VLM Attribute Extractor (optional) ====================
# class VLMAttributeExtractor:
#     """
#     Optional BLIP-based attribute extractor. Kept minimal and robust.
#     Note: BLIP is relatively large and slow; used only when explicit attribute extraction is required.
#     """

#     def __init__(self, model_type: str = "blip", device: Optional[torch.device] = None):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         logger.info(f"Initializing VLMAttributeExtractor on {self.device}")
#         try:
#             self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#             self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
#             self.model.eval()
#         except Exception as e:
#             logger.warning(f"Could not load BLIP model: {e}")
#             self.processor = None
#             self.model = None

#     def extract_caption(self, img_input: Union[str, np.ndarray, Image.Image]) -> str:
#         if self.model is None or self.processor is None:
#             return ""
#         img = _ensure_pil(img_input)
#         if img is None:
#             return ""
#         try:
#             inputs = self.processor(images=img, return_tensors="pt").to(self.device)
#             with torch.no_grad():
#                 out = self.model.generate(**inputs, max_length=50)
#             caption = self.processor.decode(out[0], skip_special_tokens=True)
#             return caption
#         except Exception as e:
#             logger.warning(f"BLIP caption error: {e}")
#             return ""

#     # Minimal attribute parsing retained from the notebook (category, color, sleeve)
#     def extract_attributes(self, img_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, str]:
#         caption = self.extract_caption(img_input)
#         return {
#             "caption": caption,
#             "category": self._extract_category(caption),
#             "color": self._extract_color(caption),
#             "sleeve_type": self._extract_sleeve_type(caption),
#         }

#     def _extract_category(self, caption: str) -> str:
#         caption_lower = caption.lower()
#         categories = {
#             "dress": ["dress", "gown", "frock"],
#             "shirt": ["shirt", "blouse", "top"],
#             "tshirt": ["t-shirt", "tshirt", "t shirt"],
#             "pants": ["pants", "trousers", "jeans"],
#             "shorts": ["shorts"],
#             "skirt": ["skirt"],
#             "jacket": ["jacket", "blazer", "coat"],
#             "sweater": ["sweater", "pullover", "cardigan"],
#             "shoes": ["shoes", "sneakers", "boots", "sandals"],
#             "bag": ["bag", "purse", "backpack"],
#             "accessory": ["scarf", "hat", "belt", "tie"],
#         }
#         for category, keywords in categories.items():
#             for kw in keywords:
#                 if kw in caption_lower:
#                     return category
#         return "unknown"

#     def _extract_color(self, caption: str) -> str:
#         caption_lower = caption.lower()
#         colors = [
#             "red",
#             "blue",
#             "green",
#             "black",
#             "white",
#             "yellow",
#             "pink",
#             "purple",
#             "orange",
#             "brown",
#             "grey",
#             "gray",
#             "beige",
#             "navy",
#             "maroon",
#         ]
#         for c in colors:
#             if c in caption_lower:
#                 return c
#         return "unknown"

#     def _extract_sleeve_type(self, caption: str) -> str:
#         caption_lower = caption.lower()
#         if "sleeveless" in caption_lower or "no sleeve" in caption_lower:
#             return "sleeveless"
#         if "long sleeve" in caption_lower:
#             return "long_sleeve"
#         if "short sleeve" in caption_lower:
#             return "short_sleeve"
#         return "unknown"


# # ==================== CLIP-BASED SEMANTIC MATCHER ====================
# class CLIPMatcher:
#     """Wraps CLIP model/processor. Supports path / PIL / numpy arrays."""

#     def __init__(self, device: Optional[torch.device] = None):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         logger.info(f"Initializing CLIPMatcher on {self.device}")
#         try:
#             self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#             self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#             self.model.eval()
#         except Exception as e:
#             logger.error(f"Failed to load CLIP model: {e}")
#             raise

#         # Simple category templates; can be extended
#         self.category_templates = {
#             "shirt": "a photo of a shirt",
#             "tshirt": "a photo of a t-shirt",
#             "dress": "a photo of a dress",
#             "pants": "a photo of pants",
#             "shorts": "a photo of shorts",
#             "skirt": "a photo of a skirt",
#             "jacket": "a photo of a jacket",
#             "sweater": "a photo of a sweater",
#             "shoes": "a photo of shoes",
#             "bag": "a photo of a bag",
#         }

#     def predict_category(self, img_input: Union[str, np.ndarray, Image.Image]) -> (str, float):
#         try:
#             img = _ensure_pil(img_input)
#             texts = list(self.category_templates.values())
#             inputs = self.processor(text=texts, images=img, return_tensors="pt", padding=True).to(self.device)
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 logits_per_image = outputs.logits_per_image
#                 probs = logits_per_image.softmax(dim=1)
#             top_idx = int(probs.argmax().item())
#             top_prob = float(probs[0, top_idx].item())
#             categories = list(self.category_templates.keys())
#             return categories[top_idx], top_prob
#         except Exception as e:
#             logger.warning(f"CLIP predict_category error: {e}")
#             return "unknown", 0.0

#     def get_image_embedding(self, img_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
#         try:
#             img = _ensure_pil(img_input)
#             inputs = self.processor(images=img, return_tensors="pt").to(self.device)
#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**inputs)
#                 image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#             return image_features.cpu().numpy().flatten()
#         except Exception as e:
#             logger.warning(f"CLIP get_image_embedding error: {e}")
#             return None


# # ==================== Visual (ResNet) Feature Extractor ====================
# class VisualFeatureExtractor:
#     """ResNet50 based visual feature extractor. Accepts path / numpy array / PIL."""

#     def __init__(self, device: Optional[torch.device] = None):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         logger.info(f"Initializing VisualFeatureExtractor on {self.device}")
#         try:
#             # Use weights API if available
#             self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         except Exception:
#             # fallback older API
#             self.model = models.resnet50(pretrained=True)
#         self.model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
#         self.model.eval()

#         self.transform = transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]
#         )

#     def extract_features(self, img_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
#         img = _ensure_pil(img_input)
#         if img is None:
#             return None
#         try:
#             img_tensor = self.transform(img).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 features = self.model(img_tensor)
#             features = features.squeeze().cpu().numpy()
#             # L2 normalize
#             norm = np.linalg.norm(features)
#             if norm > 0:
#                 features = features / norm
#             return features
#         except Exception as e:
#             logger.warning(f"VisualFeatureExtractor.extract_features error: {e}")
#             return None


# # ==================== HYBRID SEARCH INDEX ====================
# class HybridFashionSearch:
#     """
#     Hybrid search index:
#       - CLIP semantic features + category templates
#       - ResNet visual features
#       - Optional VLM attributes (BLIP)
#     """

#     def __init__(self, device: Optional[torch.device] = None):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         logger.info("Initializing HybridFashionSearch...")
#         # Components
#         try:
#             self.vlm = VLMAttributeExtractor(device=self.device)
#         except Exception:
#             self.vlm = None
#         self.clip = CLIPMatcher(device=self.device)
#         self.visual = VisualFeatureExtractor(device=self.device)

#         # Storage
#         self.inventory_data: Dict[str, Dict[str, Any]] = {}
#         self.category_index: Dict[str, List[str]] = {}

#     def build_index(self, inventory_dir: str, meta_df=None, save_path: str = "data/hybrid_fashion_index.pkl"):
#         logger.info(f"Building hybrid index from {inventory_dir}")
#         if not os.path.exists(inventory_dir):
#             raise FileNotFoundError(f"Inventory directory does not exist: {inventory_dir}")

#         image_files = [f for f in os.listdir(inventory_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#         for img_file in tqdm(image_files, desc="Indexing inventory"):
#             img_path = os.path.join(inventory_dir, img_file)
#             try:
#                 item_data: Dict[str, Any] = {}

#                 # CLIP category + features
#                 category, conf = self.clip.predict_category(img_path)
#                 item_data["clip_category"] = category
#                 item_data["clip_confidence"] = conf
#                 item_data["clip_features"] = self.clip.get_image_embedding(img_path)

#                 # Visual features
#                 item_data["visual_features"] = self.visual.extract_features(img_path)

#                 # Optional VLM attributes (commented in notebook)
#                 # if self.vlm is not None:
#                 #     item_data['vlm_attributes'] = self.vlm.extract_attributes(img_path)

#                 # Metadata from provided DataFrame (if any)
#                 if meta_df is not None:
#                     try:
#                         img_id = int(os.path.splitext(img_file)[0])
#                         if img_id in meta_df["id"].values:
#                             meta_row = meta_df[meta_df["id"] == img_id].iloc[0]
#                             item_data["metadata"] = {
#                                 "category": meta_row.get("masterCategory", "unknown"),
#                                 "subcategory": meta_row.get("subCategory", "unknown"),
#                                 "article": meta_row.get("articleType", "unknown"),
#                                 "color": meta_row.get("baseColour", "unknown"),
#                             }
#                     except Exception:
#                         # ignore metadata mapping issues
#                         pass

#                 self.inventory_data[img_file] = item_data

#                 # Build category index
#                 if item_data["clip_category"] not in self.category_index:
#                     self.category_index[item_data["clip_category"]] = []
#                 self.category_index[item_data["clip_category"]].append(img_file)

#             except Exception as e:
#                 logger.warning(f"Skipping {img_file} due to error: {e}")
#                 continue

#         logger.info(f"Indexed {len(self.inventory_data)} items across {len(self.category_index)} categories")

#         # Ensure dir exists for save_path
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         self.save_index(save_path)

#     def save_index(self, save_path: str = "data/hybrid_fashion_index.pkl"):
#         index_data = {"inventory_data": self.inventory_data, "category_index": self.category_index}
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         with open(save_path, "wb") as f:
#             pickle.dump(index_data, f)
#         logger.info(f"Hybrid index saved to {save_path}")

#     def load_index(self, load_path: str = "data/hybrid_fashion_index.pkl"):
#         if not os.path.exists(load_path):
#             raise FileNotFoundError(f"Index file not found: {load_path}")
#         with open(load_path, "rb") as f:
#             index_data = pickle.load(f)
#         self.inventory_data = index_data.get("inventory_data", {})
#         self.category_index = index_data.get("category_index", {})
#         logger.info(f"Hybrid index loaded from {load_path}. Categories: {list(self.category_index.keys())}")

#     def search(
#         self,
#         query_input: Union[str, np.ndarray, Image.Image],
#         top_k: int = 5,
#         category_filter: bool = True,
#         clip_weight: float = 0.4,
#         visual_weight: float = 0.6,
#     ) -> List[Dict[str, Any]]:
#         """
#         Perform hybrid search.

#         Returns list of results (dicts) with fields:
#         - filename, score, clip_sim, visual_sim, category, metadata
#         - product_image, similarity_score, semantic_similarity, color_similarity, match_confidence, feature_method
#         """
#         query_pil = _ensure_pil(query_input)
#         if query_pil is None:
#             logger.error("Could not read query image")
#             return []

#         # Stage 1: category
#         query_category, query_conf = self.clip.predict_category(query_pil)

#         if category_filter and query_category not in self.category_index:
#             logger.warning(f"Category '{query_category}' not in inventory; returning empty results")
#             return []

#         # Candidates
#         if category_filter:
#             candidates = self.category_index.get(query_category, [])
#         else:
#             candidates = list(self.inventory_data.keys())

#         # Query features
#         query_clip = self.clip.get_image_embedding(query_pil)
#         query_visual = self.visual.extract_features(query_pil)
#         if query_clip is None or query_visual is None:
#             logger.error("Failed to extract query features (clip/visual)")
#             return []

#         results = []
#         for fname in candidates:
#             item = self.inventory_data.get(fname, {})
#             clip_sim = 0.0
#             visual_sim = 0.0
#             if item.get("clip_features") is not None and query_clip is not None:
#                 try:
#                     clip_sim = float(np.dot(query_clip, item["clip_features"]))
#                 except Exception:
#                     clip_sim = 0.0
#             if item.get("visual_features") is not None and query_visual is not None:
#                 try:
#                     visual_sim = float(np.dot(query_visual, item["visual_features"]))
#                 except Exception:
#                     visual_sim = 0.0

#             combined_score = clip_weight * clip_sim + visual_weight * visual_sim

#             # Build result dict with both v3 and v1-friendly keys
#             res = {
#                 "filename": fname,
#                 "score": float(combined_score),
#                 "clip_sim": float(clip_sim),
#                 "visual_sim": float(visual_sim),
#                 "category": item.get("clip_category", "unknown"),
#                 "metadata": item.get("metadata", {}),
#                 # compatibility keys
#                 "product_image": fname,
#                 "similarity_score": float(combined_score),
#                 "semantic_similarity": float(clip_sim),
#                 "visual_similarity": float(visual_sim),
#                 "color_similarity": 0.0,
#                 "match_confidence": int(min(100, max(0, combined_score * 100))),
#                 "feature_method": "HybridV3",
#             }
#             results.append(res)

#         results.sort(key=lambda x: x["score"], reverse=True)
#         return results[:top_k]


# # ==================== Module-level singletons & helpers ====================
# _global_index: Optional[HybridFashionSearch] = None
# _global_visual_extractor: Optional[VisualFeatureExtractor] = None


# def get_extractor() -> VisualFeatureExtractor:
#     global _global_visual_extractor
#     if _global_visual_extractor is None:
#         _global_visual_extractor = VisualFeatureExtractor()
#     return _global_visual_extractor


# def get_index(inventory_dir: str, index_path: Optional[str] = None, meta_df: Optional[Any] = None) -> HybridFashionSearch:
#     """
#     Get or create global hybrid index. If index_path exists it will be loaded; otherwise built.
#     """
#     global _global_index
#     if _global_index is None:
#         idx = HybridFashionSearch()
#         # default index path if not provided
#         index_path = index_path or "data/hybrid_fashion_index.pkl"
#         if os.path.exists(index_path):
#             try:
#                 idx.load_index(index_path)
#             except Exception as e:
#                 logger.warning(f"Failed to load existing index at {index_path}: {e}. Rebuilding...")
#                 idx.build_index(inventory_dir, meta_df=meta_df, save_path=index_path)
#         else:
#             idx.build_index(inventory_dir, meta_df=meta_df, save_path=index_path)
#         _global_index = idx
#     return _global_index


# def search_similar_products(
#     query_img: Union[str, np.ndarray, Image.Image],
#     inventory_dir: str,
#     top_k: int = 5,
#     category_filter: bool = True,
#     clip_weight: float = 0.4,
#     visual_weight: float = 0.6,
#     index_path: Optional[str] = None,
#     meta_df: Optional[Any] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Compatibility wrapper that mimics previous module's function signature.
#     query_img can be path / numpy array / PIL image.
#     """
#     idx = get_index(inventory_dir, index_path=index_path, meta_df=meta_df)
#     return idx.search(
#         query_img,
#         top_k=top_k,
#         category_filter=category_filter,
#         clip_weight=clip_weight,
#         visual_weight=visual_weight,
#     )