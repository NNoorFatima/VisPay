# Hybrid Fashion Search Engine (v4.1) 
### A production-ready, high-accuracy
visual search engine designed specifically for fashion e-commerce. This
module implements a Hybrid Retrieval System that combines semantic
understanding (**OpenCLIP**) with visual feature extraction (**ResNet50**),
powered by FAISS for scalable similarity search.

## Key Features Hybrid RRF Ranking: 
* **Hybrid RRF Ranking**: Uses Reciprocal Rank Fusion to combine Semantic Similarity (text/concept) and Visual Similarity (pattern/shape) with 50/50 weighting.
* **Zero-Shot Category Detection**: Automatically classifies input images into 17+ fashion categories (e.g., "dress", "jacket", "activewear") using ViT-H-14.
* **Smart Fallback System**: If a category is unknown or out of stock, the system automatically performs a semantic fallback search across the entire inventory to find the closest matches.
* **Adaptive FAISS Indexing**: Automatically selects between FlatL2 (exact search) for small datasets and IVFPQ (quantized search) for large datasets.

* V3 API Compatibility: Fully backward compatible with previous API route schemas.

## Model Architecture & Hyperparameters

### Neural Network Architectures

The system relies on two distinct neural network architectures to generate embeddings:

#### 1. Semantic Matcher: OpenCLIP (ViT-H-14)

* Model: ViT-H-14
* Pretrained Weights: laion2b_s32b_b79k

#### Purpose

Handles category classification, color detection, and semantic understanding (e.g., distinguishing a "formal shirt" from a "casual shirt").

#### Confidence Threshold

0.018 (Predictions below this trigger the "Unknown" fallback).

### 2. Visual Extractor: ResNet50

* Model: ResNet50
* Weights: IMAGENET1K_V2

#### Purpose

Extracts low-level visual features such as fabric texture, specific cuts, and shape outlines.

#### Input Size

224x224 pixels.

### 3. Vector Database: FAISS

* Metric: L2 Distance (Euclidean)
* Index Type:
	+ <= 100 items: IndexFlatL2 (Brute force, highest accuracy)
	+ > 100 items: IndexIVFPQ (Inverted File with Product Quantization) for speed
* Hyperparameter: nprobe = 10 (Searches 10 nearest clusters during query)

## Installation & Dependencies

Ensure you have the required PyTorch and FAISS libraries installed.


``` bash
pip install torch torchvision
pip install open_clip_torch
pip install faiss-gpu  # Use faiss-cpu if no GPU is available
pip install pillow numpy tqdm
```
## API Usage

### Primary Entry Point

The primary entry point for the module is the `search_similar_products` function.

### Parameters

Parameter | Type | Default | Description
---------|------|---------|-------------
query_img | str | PIL.Image | Required
inventory_dir | str | Required | Directory containing the inventory images
top_k | int | 8 | Number of results to return
category_filter | bool | True | If True, restricts search to the predicted category
user_category | str | None | Manually force a specific category (overrides AI)
index_path | str | data/... | Path where the FAISS .pkl index is saved/loaded

The function returns a dictionary designed for frontend consumption:

### JSON Response
```bash
{
  "status": "success",
  "category_used": "jacket",
  "confidence": 0.85,
  "message": "Found in jacket",
  "results": [
    {
      "filename": "jacket_01.jpg",
      "product_image": "jacket_01.jpg",
      "score": 0.92,
      "similarity_score": 0.92,
      "predicted_color": "black",
      "category": "jacket"
    },
    ...
  ]
}
```

## Logic Flow & Algorithms
### Indexing (build_index)
#### Steps

* Scans inventory_dir.
* Passes every image through CLIP to predict its category (e.g., "dress").
* Extracts Visual Features (ResNet) and Semantic Features (CLIP).
* Builds a separate FAISS index for each category to ensure fast filtering.
* Saves metadata to a Pickle file.

### Search Pipeline Preprocessing
#### Steps

* Convert input image to RGB.
* Category Prediction:
	+ The AI predicts the category (e.g., "shoes").
	+ Smart Logic: If confidence < 0.018, it flags the image as "Unknown".

* Search Strategy:
    + Scenario A (High Confidence): Search only within the specific FAISS category index.
    + Scenario B (Unknown/Out of Stock): Trigger \_get_smart_fallback. This searches the entire database using only CLIP vectors to find conceptually similar items regardless of category tags.
* Scoring (Reciprocal Rank Fusion):
    ```bash
    Final Score=(0.5×CLIP_sim ​ )+(0.5×ResNet_sim ​ ) 
    ```
* Deduplication: Ensures the same product source file isn't returned twice if multiple augmented versions exist in the index.

## Project Structure Plaintext
```bash
    modules/ 
        ├── visual_searchv3.py # Main logic file 
    data/ 
        └──hybrid_v4_index.pkl # Generated FAISS index (do not edit manually)
    static/ 
        └── products/ # Inventory images 
```
## Common Errors & Troubleshooting**
* **ImportError: Missing dependencies**: Run the pip install commands listed above.
* **Image path not found**: Ensure inventory_dir is absolute or correctly relative to the execution script.
* **Low Confidence Matches**: If the system returns "unknown" often, try lowering the threshold in CLIPMatcher `predict_category` (currently 0.018).


## Workflow Details & Scenario Handling

Here is the breakdown of how the system handles specific scenarios based on the v4.1 code logic.

### 1. Input Validation & Extraction

**Step:** The system attempts to convert the input to a PIL RGB image (`_ensure_pil`).

**Edge Case:** If the file path is broken or the image is corrupt, the system halts immediately and returns a clean error message, preventing server crashes.

**Models:** If the image is valid, parallel extraction occurs:

* **ResNet50:** Captures texture, shape, and patterns.
* **CLIP:** Captures semantic concept and color.

### 2. User Override Mode (Strict Search)

**Scenario:** The user explicitly selects "Watches" from a dropdown, but uploads a picture of a generic circle.

**Logic:** The system respects the user's intent above the AI's prediction.

**Handling:**

* It bypasses the confidence check.
* It checks if "watch" exists in your FAISS index.
* **Edge Case:** If "watch" is not in your inventory, it returns a "No Match" status rather than guessing random products. This prevents user frustration from irrelevant results.

### 3. AI Auto-Detect Mode (Standard Search)

**Scenario:** User uploads an image without selecting a category.

**Logic:** The system predicts the category using CLIPMatcher.

**Success:** If confidence is high (> 1.8%) and the category exists (e.g., "Dress"), it performs a highly optimized search within that specific FAISS cluster.

### 4. The "Unknown" Scenario (low Confidence)

**Scenario:** User uploads a blurry photo or a non-fashion item (e.g., a dog).

**Logic:** The prediction confidence drops below 0.018.

**Action:** The system triggers `category_filter=False`.

**Result:** It searches the entire inventory for visual matches. The response status is set to `needs_confirmation`, flagging the frontend to possibly ask the user: "We aren't sure what this is, but here are some visual matches."

### 5. The "Smart Fallback" (Out of Stock)

**Scenario:** User uploads a valid "Kimono". The AI correctly identifies it as "Kimono", but your inventory has no items labeled "Kimono".

**Logic:** Instead of returning 0 results, the system activates `_get_smart_fallback`.

**Action:** It ignores category constraints and uses the CLIP vector to find the closest semantic matches (e.g., "Robes" or "Long Cardigans") from the global pool.

**Result:** Returns products with a message: "Category 'Kimono' not in stock. Showing smart fallback items."

### 6. Result Processing

**Hybrid Ranking:** Matches are scored using Reciprocal Rank Fusion (`$Score = 0.5 \times Semantic + 0.5 \times Visual$`).

**Deduplication:** The code checks `source_file` to ensure that if you indexed 5 augmented versions of the same shirt, the user only sees one unique product card in the results.

**Formatting:** Data types (NumPy arrays) are sanitized to standard Python types (Floats/Lists) to ensure the JSON response never breaks the frontend parser.
