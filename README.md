# VisPay

A FastAPI-based dual image intelligence system for payment verification and visual product search in live commerce platforms.

## Features

### 1. Payment Receipt Verification (OCR)

- Extract transaction ID, amount, and date from payment receipts
- Support for both pytesseract and EasyOCR
- Batch processing capability
- Multiple preprocessing techniques for better accuracy

### 2. Visual Product Search

- SIFT-based feature extraction for robust image matching
- FLANN-based feature matching with Lowe's ratio test
- Top-K product retrieval with similarity scores
- Scale and rotation invariant matching

## Project Structure

```
VisPay/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── models/
│   └── schemas.py         # Pydantic models for request/response
├── modules/
│   ├── preprocessing.py   # Image preprocessing utilities
│   ├── ocr_verification.py # OCR and receipt parsing logic
│   └── visual_search.py   # Visual search implementation
├── routes/
│   ├── payment.py         # Payment verification endpointsV
│   └── product.py         # Product search endpoints
├── static/
│   └── product_images/    # Inventory product images
└── uploads/               # Temporary upload directory
```

## Prerequisites

See [PREREQUISITES.md](PREREQUISITES.md) for detailed system requirements and setup instructions.

**Quick Checklist:**

- Python 3.8+
- Tesseract OCR installed
- Virtual environment (recommended)

## Installation

1. **Navigate to project directory**

   ```bash
   cd VisPay
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (if using pytesseract)

   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Mac**: `brew install tesseract`

5. **Configure environment variables**

   Create `.env` file in project root:

   ```env
   TESSERACT_CMD=C:\Users\Noor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
   USE_EASYOCR=false
   INVENTORY_DIR=static/product_images
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# OCR Settings
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows example
USE_EASYOCR=false  # Set to true to use EasyOCR

# Visual Search Settings
INVENTORY_DIR=static/product_images

# File Upload Settings
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE=10485760
```

## Running the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Payment Verification

#### POST `/api/v1/payment/verify`

Upload a payment receipt image for OCR verification.

**Request:**

- `receipt_image`: Image file (multipart/form-data)

**Response:**

```json
{
  "transaction_id": "INV-12345",
  "amount": "99.99",
  "date": "2024-01-15",
  "verification_status": "verified",
  "extraction_method": "pytesseract"
}
```

#### POST `/api/v1/payment/verify-batch`

Verify multiple payment receipts at once.

**Request:**

- `receipts`: List of image files (multipart/form-data)

### Product Search

#### POST `/api/v1/product/search`

Search for similar products using visual matching.

**Request:**

- `query_image`: Product image file (multipart/form-data)
- `top_k`: Number of top matches (query parameter, default: 5, max: 20)

**Response:**

```json
{
  "matches": [
    {
      "product_image": "product1.jpg",
      "similarity_score": 230,
      "match_confidence": 100
    }
  ],
  "total_matches": 1
}
```
