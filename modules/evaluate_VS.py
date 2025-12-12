
import os
import random
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import classification_report

# ==================== IMPORTS ====================
# We import your system from the main module.
# Ensure 'visual_searchv3.py' is in the same directory.
try:
    from visual_searchv3 import get_index, search_similar_products, _ensure_pil
except ImportError:
    raise ImportError("Could not find 'visual_searchv3.py'. Make sure this script is in the same folder.")

# ==================== CONFIGURATION ====================
# UPDATE THIS PATH to your actual inventory folder
INVENTORY_DIR = "../static/product_images"  # Update this
INDEX_PATH = "../data/hybrid_fashion_index.pkl"

SAMPLE_SIZE = 35   # Keep this small (50-100) for your panel demo/testing
TOP_K = 3         # How many results to check for the correct item

# ==================== SIMULATION LOGIC ====================
def simulate_user_photo(img_path):
    """
    Takes a clean inventory image and applies 'Real World' noise 
    to mimic a user uploading a photo from their phone.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # 1. Random Crop (Simulate zooming or bad framing)
    # We keep 70-90% of the image to ensure the item is still visible
    crop_scale = random.uniform(0.7, 0.9)
    crop_w, crop_h = int(w * crop_scale), int(h * crop_scale)
    
    # Ensure we don't crash on small images
    if w > crop_w and h > crop_h:
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)
        img = img.crop((x, y, x + crop_w, y + crop_h))
    
    # 2. Brightness Jitter (Simulate indoor/outdoor lighting)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # 3. Mild Blur (Simulate shaky hands)
    # We only apply this 50% of the time
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
    return img

# ==================== MAIN EVALUATION LOOP ====================
def evaluate():
    print(f"--- Starting System Evaluation ---")
    print(f"Target Directory: {INVENTORY_DIR}")
    
    # ---------------------------------------------------------
    # [CRITICAL FIX] PRE-LOAD MODELS INTO MEMORY
    # ---------------------------------------------------------
    print("1. Loading Index and initializing models (This happens ONCE)...")
    idx = get_index(INVENTORY_DIR, INDEX_PATH)
    
    # We explicitly call this to load ViT-H-14 (4GB) into GPU/RAM now.
    # If we don't do this, it will reload from disk 50 times (causing 129s latency).
    idx._init_models()
    
    if idx.clip is None:
        raise ValueError("CRITICAL ERROR: Model failed to initialize.")
    
    print("   -> Models loaded successfully. Ready for rapid inference.")
    # ---------------------------------------------------------

    # Get list of files
    if not os.path.exists(INVENTORY_DIR):
        print(f"Error: Directory {INVENTORY_DIR} not found.")
        return

    all_files = [f for f in os.listdir(INVENTORY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not all_files:
        print("Error: No images found in inventory directory.")
        return

    # Select random sample
    test_files = random.sample(all_files, min(len(all_files), SAMPLE_SIZE))
    print(f"2. Testing on {len(test_files)} random images...\n")
    
    # Metrics Storage
    metrics = {
        "top_1_hits": 0,
        "top_5_hits": 0,
        "top_10_hits": 0,
        "category_correct": 0,
        "latencies": []
    }
    
    y_true_cats = []
    y_pred_cats = []

    # Progress Bar Loop
    for fname in tqdm(test_files, desc="Running Queries"):
        original_path = os.path.join(INVENTORY_DIR, fname)
        
        # A. Create the "User Query"
        query_img = simulate_user_photo(original_path)
        
        # B. Measure Latency
        start_time = time.time()
        
        # Run the search!
        # Note: We pass user_category=None to test the AI's auto-detection capability.
        response = search_similar_products(
            query_img, 
            inventory_dir=INVENTORY_DIR,
            top_k=TOP_K,
            index_path=INDEX_PATH,
            user_category=None 
        )
        
        duration_ms = (time.time() - start_time) * 1000
        metrics["latencies"].append(duration_ms)
        
        # C. Analyze Accuracy
        results = response.get("results", [])
        
        # Did we find the original filename?
        found_ranks = [i for i, r in enumerate(results) if r['filename'] == fname]
        
        if found_ranks:
            rank = found_ranks[0] # 0-indexed
            if rank == 0: metrics["top_1_hits"] += 1
            if rank < 5: metrics["top_5_hits"] += 1
            if rank < 10: metrics["top_10_hits"] += 1
            
        # D. Analyze Category Classification
        # We look up what the 'True' category was in our index
        true_data = idx.inventory_data.get(fname)
        if true_data:
            true_cat = true_data.get("category")
            pred_cat = response.get("category_used")
            
            if true_cat and pred_cat:
                y_true_cats.append(true_cat)
                y_pred_cats.append(pred_cat)
                if true_cat == pred_cat:
                    metrics["category_correct"] += 1

    # ==================== FINAL REPORT ====================
    total = len(test_files)
    
    print("\n" + "="*40)
    print("       EVALUATION REPORT       ")
    print("="*40)
    print(f"Top-1 Accuracy (Exact Match): {metrics['top_1_hits']/total:.2%}")
    print(f"Top-5 Recall   (User sees it):{metrics['top_5_hits']/total:.2%}")
    print(f"Top-10 Recall  (Broad Match): {metrics['top_10_hits']/total:.2%}")
    print("-" * 40)
    print(f"Category Router Accuracy:     {metrics['category_correct']/len(y_true_cats):.2%}")
    print(f"Avg Latency per Query:        {np.mean(metrics['latencies']):.2f} ms")
    print("="*40)
    
    if y_true_cats:
        print("\nDetailed Category Classification Report:")
        # We use zero_division=0 to avoid warnings for missing categories in the random sample
        print(classification_report(y_true_cats, y_pred_cats, zero_division=0))

if __name__ == "__main__":
    evaluate()