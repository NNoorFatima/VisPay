# VisPay Vision API & Intelligence System

A comprehensive dual-intelligence system designed for live commerce and fintech platforms. VisPay combines a high-performance **FastAPI** backend for standard OCR and hybrid visual search with a specialized **Flask** microservice for digital cross-verification using chat logs and email forensics.

---

##  Key Features

### 1. Payment Verification Suite
The system offers two distinct modes for verifying payment receipts:

#### **Mode A: Standard OCR (Image Analysis)**
*Powered by FastAPI & Tesseract/EasyOCR*
- **Text Extraction:** Extracts Transaction ID, Amount, and Date from receipt images.
- **Preprocessing:** Auto-selects between minimal, advanced, and scale-aware preprocessing methods based on image quality.
- **Batch Processing:** Supports bulk verification of multiple receipts.

#### **Mode B: Digital Cross-Verification (Forensics)**
*Powered by Flask & Gmail API*
- **Multi-Source Validation:** Cross-references data from **Receipt Image**, **Chat Logs**, and **Email Confirmations**.
- **Fraud Detection:** Uses edge detection (Canny) to flag potentially edited or tampered images.
- **Causality Checks:** Verifies chronological consistency (e.g., payment time vs. chat conversation time).
- **Confidence Scoring:** Generates a 0-100 trust score based on data matching logic (OCR vs. Chat Amount vs. Email metadata).

### 2. Hybrid Fashion Search Engine (v4.1)
*Powered by OpenCLIP, ResNet50 & FAISS*
- **Hybrid RRF Ranking:** Combines Semantic Similarity (text/concept via OpenCLIP) and Visual Similarity (pattern/shape via ResNet50) with 50/50 weighting.
- **Zero-Shot Classification:** Automatically detects 17+ fashion categories (e.g., "dress", "jacket") using ViT-H-14.
- **Smart Fallback:** If a specific category is out of stock, it searches the global inventory for semantic matches.
- **Adaptive Indexing:** Automatically switches between `IndexFlatL2` (exact) and `IndexIVFPQ` (quantized) based on inventory size.

---

##  Project Structure

```text
VisPay/
├── main.py                   # FastAPI Entry Point (Port 8000) - Standard OCR & Visual Search
├── ocr_digital_images.py     # Flask Entry Point (Port 5000) - Digital Verification
├── config.py                 # Configuration Management
├── models/                   # Pydantic Schemas
├── modules/
│   ├── ocr_verification.py   # Standard OCR Logic
│   ├── visual_searchv3.py    # Hybrid Visual Search Logic (v4.1)
│   ├── image_analysis.py     # Image Quality & Preprocessing
│   └── preprocessing.py      # OpenCV Image Filters
├── routes/                   # FastAPI Routes
├── static/
│   └── product_images/       # Inventory for Visual Search
└── cred.json                 # Google API Credentials (Required for Digital Mode)
```

## Prerequisites
1. Python 3.8+

2. Tesseract OCR installed on the system.

    * Windows: `Installer`
    * Linux: `sudo apt-get install tesseract-ocr`
    * Mac: `brew install tesseract`

3. Google Cloud Credentials (For Digital Verification Mode):
    * Enable Gmail API.
    * Download `credentials.json` and rename/save it as `cred.json` in the root directory.

## Installation
1. Clone and Navigate
```bash 
git clone [https://github.com/your-repo/VisPay.git]
cd VisPay 
```

2. Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirementsOCR.txt  #for digital ocr 
pip install -r requirementsVS.txt  #for Visual search 
pip install -r requirementsEval.txt  #for evaluation of visual search
pip install -r requirements.txt #for scanned receipt (mode A) ocr
```

## Configuration `(.env)` Create a `.env` file in the root:
```bash
API_TITLE=VisPay Vision API
API_DESCRIPTION=Dual Image Intelligence System for Payment Verification and Visual Product Search
API_VERSION=1.0.0

HOST=0.0.0.0
PORT=8000
DEBUG=false

TESSERACT_CMD= 
j
USE_EASYOCR=false    

INVENTORY_DIR=static/product_images

UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760

TESSERACT_CMD=path\to\tesseract.exe

USE_EASYOCR=false

INVENTORY_DIR=static/product_images

OCR_AUTO_PREPROCESSING=false
OCR_PREPROCESSING_METHOD=light  

USE_LLM_VALIDATION=false
GEMINI_API_KEY=key 

# LLM_MODEL=gemini-2.5-pro
LLM_MODEL =gemini-2.5-flash-lite
LOG_LEVEL=DEBUG

# vs
INVENTORY_DIR=static/product_images
VISUAL_SEARCH_INDEX_PATH=data/hybrid_fashion_index.pkl
```

## Running the Application
To utilize the full features, you must run both the FastAPI and Flask servers.
1. Start Standard API (FastAPI)
    * Handles Standard OCR and Visual Search.
    ```bash
    python main.py 
    python ocr_digital_images.py
    npm run dev # to run frontend 
    ```
2. Start Digital Verification Service (Flask)
    ```bash 
    python ocr_digital_images.py
    ```
3. Start frontend 
    ```bash
        npm run dev # to run frontend
    ```

## System Workflows

### A. Digital Verification Workflow (ocr_digital_images.py)

This module performs forensic analysis on receipts using a 4-step pipeline:

#### Aggressive Preprocessing
* Detects blur (Laplacian variance)
* Applies histogram equalization and adaptive thresholding
* Upscales image to improve OCR on low-quality screenshots

#### Data Extraction

* OCR: Extracts raw text using Tesseract with strict config.
* Regex Parsing: Identification of Amount, Transaction ID (fuzzy matching to handle "1D" vs "ID"), and Timestamp.
* Chat Parsing: Scans the uploaded chat log `(.txt)` for payment promises (e.g., "sent 5000", "total 5000").

#### Validation

* Gmail Fetch: Uses the extracted Timestamp (+/- 45 mins) and Amount/TxID to search the Merchant's Gmail for an official bank confirmation email.
* Fraud Check: Uses Canny Edge Detection to check if the text looks digitally superimposed (high edge variance).

#### Scoring

Calculates a Confidence Score (0-100).

Penalties applied for:
* Missing email confirmation
* Timestamp mismatch between chat and receipt
* Visual signs of tampering

### B. Hybrid Visual Search Workflow (v4.1)
--------------------------------------------------------

This module allows users to search for products using images.

#### Architecture

* Semantic Matcher (OpenCLIP ViT-H-14): Understands concepts (e.g., "Floral Summer Dress").
* Visual Extractor (ResNet50): Understands patterns, textures, and cuts.
* Vector DB (FAISS): Stores embeddings for fast retrieval.

#### Search Logic

* Category Prediction: AI predicts the category (e.g., "Shoes") with a confidence threshold (0.018).
* Smart Fallback: If the category is unknown or out of stock, the system falls back to a global semantic search to find similar items regardless of category.
* RRF Scoring:
	+ Final Score = `(0.5 * CLIP_Similarity) + (0.5 * Visual_Similarity)`

## API Endpoints Summary
| Method | Endpoint | Description |
| --- | --- | --- |
| POST | /api/v1/payment/verify | Standard Receipt OCR (Image Only) |
| POST | /api/v1/product/search | Visual Product Search |
| GET | /health | API Health Check |

