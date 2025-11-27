"""
Payment verification routes.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import cv2
import numpy as np
from typing import List, Optional

from modules.ocr_verification import verify_payment_receipt
from models.schemas import (
    ReceiptVerificationResponse,
    BatchVerificationResponse,
    BatchVerificationItem
)
import easyocr
from config import settings

router = APIRouter(prefix="/api/v1/payment", tags=["Payment Verification"])

# Initialize EasyOCR reader if enabled
_easyocr_reader = None


def get_easyocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None and settings.USE_EASYOCR:
        try:
            _easyocr_reader = easyocr.Reader(['en'])
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
    return _easyocr_reader


async def upload_to_cv2(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to cv2 image."""
    contents = await upload_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img


@router.post("/verify", response_model=ReceiptVerificationResponse)
async def verify_payment(
    receipt_image: UploadFile = File(...),
    preprocessing_method: Optional[str] = Query(
        None,
        description="Preprocessing method: minimal, light, medium, advanced, morphology, scale_aware, or 'auto' (default: auto if OCR_AUTO_PREPROCESSING is enabled)"
    )
):
    """
    Upload a payment receipt image for OCR verification.
    Returns: transaction_id, amount, date, and verification status
    
    Preprocessing methods:
    - auto: Automatically select based on image quality (default if enabled)
    - minimal: Just grayscale + sharpening (best for clear images)
    - light: Grayscale + contrast enhancement + light denoising
    - medium: Light + adaptive thresholding
    - advanced: Denoising + OTSU thresholding (for noisy images)
    - morphology: Advanced + morphological operations (for broken text)
    - scale_aware: Upscales image before processing (for small images)
    
    If no method is specified and OCR_AUTO_PREPROCESSING=true, the system will
    automatically analyze the image and select the best preprocessing method.
    """
    try:
        # Convert upload to cv2 image
        img = await upload_to_cv2(receipt_image)
        
        # Get EasyOCR reader if enabled
        reader = get_easyocr_reader()
        use_easyocr = settings.USE_EASYOCR and reader is not None
        
        # Run OCR verification with optional preprocessing method override
        result = verify_payment_receipt(
            img, 
            use_easyocr=use_easyocr, 
            easyocr_reader=reader,
            preprocessing_method=preprocessing_method
        )
        
        return ReceiptVerificationResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-batch", response_model=BatchVerificationResponse)
async def batch_verify(receipts: List[UploadFile] = File(...)):
    """
    Verify multiple payment receipts at once.
    """
    try:
        reader = get_easyocr_reader()
        use_easyocr = settings.USE_EASYOCR and reader is not None
        
        results = []
        for receipt in receipts:
            try:
                img = await upload_to_cv2(receipt)
                result = verify_payment_receipt(img, use_easyocr=use_easyocr, easyocr_reader=reader)
                results.append(BatchVerificationItem(
                    filename=receipt.filename or "unknown",
                    verification=ReceiptVerificationResponse(**result)
                ))
            except Exception as e:
                results.append(BatchVerificationItem(
                    filename=receipt.filename or "unknown",
                    verification=ReceiptVerificationResponse(
                        verification_status="error",
                        error=str(e)
                    )
                ))
        
        return BatchVerificationResponse(
            total_processed=len(results),
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

