# Prerequisites & Setup Guide

### 1. Python Installation

**Check if Python is installed:**

```bash
python --version
# or
python3 --version
```

**If not installed:**

- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt-get install python3 python3-pip python3-venv`
- **macOS**: `brew install python3` or download from python.org

### 2. Tesseract OCR (Required for pytesseract)

**Windows:**

1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or note the installation path

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**

```bash
brew install tesseract
```

**Verify installation:**

```bash
tesseract --version
```

### 3. Visual C++ Redistributable (Windows only, for OpenCV)

If you encounter errors with `opencv-python`, install:

- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

### 4. Git (Optional, for cloning)

```bash
git --version
```

## Installation Steps

### Step 1: Clone/Navigate to Project

```bash
cd C:\Dev\Common\DIP\VisPay
```

### Step 2: Create Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** EasyOCR will download models on first use (~500MB). This happens automatically.

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
# Then edit .env with your text editor
```

**Linux/macOS:**

```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

**Minimum `.env` configuration:**

```env
# For Windows - Update with your Tesseract path
TESSERACT_CMD=C:\Users\Noor\AppData\Local\Programs\Tesseract-OCR\tesseract.exe

# OCR Engine (false = pytesseract, true = EasyOCR)
USE_EASYOCR=false

# Visual Search - Directory with product images
INVENTORY_DIR=static/product_images
```

### Step 5: Prepare Test Data

**For Visual Search:**

- Place some product images in `static/product_images/` directory
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

**For OCR Testing:**

- Prepare receipt images (`.jpg`, `.png`, etc.)

### Step 6: Verify Installation

```bash
python -c "import cv2; import pytesseract; import easyocr; print('All imports successful!')"
```
