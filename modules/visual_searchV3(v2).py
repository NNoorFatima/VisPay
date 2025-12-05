# """
# Version 3(v2) hybrid visual search implementation (ported from notebook).

# Improvements:
# - Lazy model initialization: heavy VLM/CLIP/ResNet models are not loaded
#   during HybridFashionSearch construction when lazy=True. This lets the
#   service quickly start and only load models if/when building the index or
#   serving searches.
# - get_index now checks for an existing index file before initializing heavy
#   models. If the index pickle exists it is loaded immediately (no model loads).
#   If the index does not exist it initializes models and builds/saves the index.

# Public API (module-level functions):
# - get_extractor() -> returns visual feature extractor instance (VisualFeatureExtractor)
# - get_index(inventory_dir, index_path=None, meta_df=None, load_if_exists=True) -> returns HybridFashionSearch instance
# - search_similar_products(...) -> compatibility wrapper
# """
# import os
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
#         # categories = {
#         #     "dress": ["dress", "gown", "frock"],
#         #     "shirt": ["shirt", "blouse", "top"],
#         #     "tshirt": ["t-shirt", "tshirt", "t shirt"],
#         #     "pants": ["pants", "trousers", "jeans"],
#         #     "shorts": ["shorts"],
#         #     "skirt": ["skirt"],
#         #     "jacket": ["jacket", "blazer", "coat"],
#         #     "sweater": ["sweater", "pullover", "cardigan"],
#         #     "shoes": ["shoes", "sneakers", "boots", "sandals"],
#         #     "bag": ["bag", "purse", "backpack"],
#         #     "accessory": ["scarf", "hat", "belt", "tie"],
#         # }
#         categories = {
#             "dress": ["dress", "gown", "frock", "maxi dress", "mini dress", "midi dress"],
#             "shirt": ["shirt", "blouse", "top", "tunic", "camisole", "button-up shirt"],
#             "tshirt": ["t-shirt", "tshirt", "t shirt", "graphic tee", "long sleeve tee", "crop top"],
#             "pants": ["pants", "trousers", "jeans", "chinos", "leggings", "cargo pants", "joggers"],
#             "shorts": ["shorts", "denim shorts", "cargo shorts", "athletic shorts"],
#             "skirt": ["skirt", "mini skirt", "midi skirt", "maxi skirt", "pencil skirt", "pleated skirt"],
#             "jacket": ["jacket", "blazer", "coat", "windbreaker", "denim jacket", "leather jacket", "parka"],
#             "sweater": ["sweater", "pullover", "cardigan", "hoodie", "knitwear", "crewneck", "turtleneck"],
#             "shoes": ["shoes", "sneakers", "boots", "sandals", "heels", "loafers", "flats", "slippers"],
#             "bag": ["bag", "purse", "backpack", "tote", "crossbody", "duffel bag", "clutch"],
#             "accessory": ["scarf", "hat", "belt", "tie", "gloves", "sunglasses", "watch", "jewelry"],
#             "activewear": ["sports bra", "leggings", "track pants", "tank top", "gym shorts", "hoodie"],
#             "outerwear": ["coat", "overcoat", "trench coat", "peacoat", "puffer jacket", "raincoat"],
#             "sleepwear": ["pajamas", "nightgown", "sleep shorts", "robe"],
#             "underwear": ["bra", "panties", "boxers", "briefs", "undershirt", "lingerie"],
#             "swimwear": ["bikini", "one-piece swimsuit", "swim trunks", "cover-up"],
#             "footwear": ["boots", "sneakers", "sandals", "heels", "flats", "loafers", "slippers"],
#             "headwear": ["hat", "cap", "beanie", "headband", "visor"],
#             "formalwear": ["suit", "blazer", "dress shirt", "evening gown", "cocktail dress", "vest"],
#         }

#         for category, keywords in categories.items():
#             for kw in keywords:
#                 if kw in caption_lower:
#                     return kw #changed to sub category
#         return "unknown"

#     def _extract_color(self, caption: str) -> str:
#         caption_lower = caption.lower()
#         # colors = [
#         #     "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
#         #     "orange", "brown", "grey", "gray", "beige", "navy", "maroon",
#         # ]
#         colors = [
#             "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
#             "orange", "brown", "grey", "gray", "beige", "navy", "maroon",
#             "turquoise", "teal", "lavender", "violet", "indigo", "cyan", "magenta",
#             "peach", "cream", "mint", "olive", "lime", "mustard", "coral", "tan",
#             "khaki", "charcoal", "silver", "gold", "rose", "apricot", "plum",
#             "burgundy", "salmon", "sky blue", "forest green", "sea green", "ice blue",
#             "cream white", "off-white", "rust", "chocolate", "wine", "eggplant",
#             "sand", "camel", "emerald", "sapphire", "ruby"
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
#             "accessory": "a photo of an accessory",
#             "activewear": "a photo of activewear",
#             "outerwear": "a photo of outerwear",
#             "sleepwear": "a photo of sleepwear",
#             "underwear": "a photo of underwear",
#             "swimwear": "a photo of swimwear",
#             "footwear": "a photo of footwear",
#             "headwear": "a photo of headwear",
#             "formalwear": "a photo of formalwear",
#         }

#         # self.category_templates = {
#         #     "shirt": "a photo of a shirt",
#         #     "tshirt": "a photo of a t-shirt",
#         #     "dress": "a photo of a dress",
#         #     "pants": "a photo of pants",
#         #     "shorts": "a photo of shorts",
#         #     "skirt": "a photo of a skirt",
#         #     "jacket": "a photo of a jacket",
#         #     "sweater": "a photo of a sweater",
#         #     "shoes": "a photo of shoes",
#         #     "bag": "a photo of a bag",
#         #     "accessory": "a photo of an accessory",
#         # }

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
#             # Print all categories with their probabilities
#             for idx, category in enumerate(categories):
#                 print(f"{category}: {probs[0, idx].item():.4f}")
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


# # ==================== HYBRID SEARCH INDEX (with lazy models) ====================
# class HybridFashionSearch:
#     """
#     Hybrid search index with lazy model loading.

#     If lazy=True, models are not loaded in __init__. Call _init_models()
#     to initialize CLIP/ResNet/BLIP when necessary.
#     """

#     def __init__(self, device: Optional[torch.device] = None, lazy: bool = True):
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         self._lazy = lazy

#         # Components set to None initially to avoid heavy loads
#         self.vlm: Optional[VLMAttributeExtractor] = None
#         self.clip: Optional[CLIPMatcher] = None
#         self.visual: Optional[VisualFeatureExtractor] = None

#         # Storage
#         self.inventory_data: Dict[str, Dict[str, Any]] = {}
#         self.category_index: Dict[str, List[str]] = {}

#         # If not lazy, initialize immediately
#         if not self._lazy:
#             self._init_models()

#     def _init_models(self):
#         """Initialize heavy models if they are not already loaded."""
#         if self.clip is None:
#             try:
#                 self.clip = CLIPMatcher(device=self.device)
#             except Exception as e:
#                 logger.error(f"Failed to initialize CLIP: {e}")
#                 self.clip = None
#         if self.visual is None:
#             try:
#                 self.visual = VisualFeatureExtractor(device=self.device)
#             except Exception as e:
#                 logger.error(f"Failed to initialize VisualFeatureExtractor: {e}")
#                 self.visual = None
#         if self.vlm is None:
#             try:
#                 # BLIP optional: load if available and needed
#                 self.vlm = VLMAttributeExtractor(device=self.device)
#             except Exception:
#                 self.vlm = None

#     def build_index(self, inventory_dir: str, meta_df=None, save_path: str = "data/hybrid_fashion_index.pkl"):
#         """Build comprehensive index with all features (requires models)."""
#         logger.info(f"Building hybrid index from {inventory_dir}")
#         # Ensure models are available
#         self._init_models()
#         if self.clip is None or self.visual is None:
#             raise RuntimeError("Required models (CLIP/ResNet) are not available to build index.")

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

#                 # Metadata (if provided)
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

#         Uses lazy initialization so that if models were not loaded at startup
#         they will be created now.
#         """
#         # Initialize models if required for search
#         self._init_models()
#         if self.clip is None or self.visual is None:
#             logger.error("Search requested but CLIP/Visual models are not available.")
#             return []

#         query_pil = _ensure_pil(query_input)
#         if query_pil is None:
#             logger.error("Could not read query image")
#             return []

#         # Stage 1: category
#         query_category, query_conf = self.clip.predict_category(query_pil)
#         print(f"Predicted category: {query_category} (conf: {query_conf:.3f})")
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


# def get_index(
#     inventory_dir: str,
#     index_path: Optional[str] = None,
#     meta_df: Optional[Any] = None,
#     load_if_exists: bool = True,
# ) -> HybridFashionSearch:
#     """
#     Get or create global hybrid index.

#     Behavior:
#       - If load_if_exists is True and an index pickle exists at index_path (default
#         data/hybrid_fashion_index.pkl), the index will be loaded WITHOUT initializing
#         heavy models.
#       - If no index file exists (or load_if_exists is False), models will be initialized
#         and the index will be built and saved to index_path.
#     """
#     global _global_index
#     index_path = index_path or "data/hybrid_fashion_index.pkl"

#     if _global_index is not None:
#         return _global_index

#     # If an index file exists and caller allows loading, load it without heavy model init
#     if load_if_exists and os.path.exists(index_path):
#         logger.info(f"Found existing index at {index_path}. Loading index without instantiating heavy models.")
#         idx = HybridFashionSearch(lazy=True)
#         try:
#             idx.load_index(index_path)
#             _global_index = idx
#             return _global_index
#         except Exception as e:
#             logger.warning(f"Failed to load existing index {index_path}: {e}. Will rebuild index.")
#             # fall through to rebuild

#     # No index exists or loading failed -> initialize models and build index
#     logger.info("No existing index found (or loading failed). Initializing models and building index...")
#     idx = HybridFashionSearch(lazy=False)
#     idx.build_index(inventory_dir, meta_df=meta_df, save_path=index_path)
#     _global_index = idx
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
#     Compatibility wrapper exposing the v3 search API.
#     Ensures an index exists (loads existing or builds if absent).
#     """
#     idx = get_index(inventory_dir, index_path=index_path, meta_df=meta_df, load_if_exists=True)
#     return idx.search(
#         query_img,
#         top_k=top_k,
#         category_filter=category_filter,
#         clip_weight=clip_weight,
#         visual_weight=visual_weight,
#     )
