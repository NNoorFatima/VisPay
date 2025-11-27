"""
Image preprocessing utilities for OCR and visual search.
"""
import cv2
import numpy as np
from typing import Optional


def preprocess_image_for_ocr_minimal(img: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing: just grayscale conversion and slight sharpening.
    Best for already clear images.
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Optional: slight sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    return sharpened


def preprocess_image_for_ocr_light(img: np.ndarray) -> np.ndarray:
    """
    Light preprocessing: grayscale, contrast enhancement, and light denoising.
    Good for slightly noisy images.
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Light denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised


def preprocess_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for OCR: grayscale, blur, and adaptive threshold.
    
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh


def preprocess_image_for_ocr_medium(img: np.ndarray) -> np.ndarray:
    """
    Medium preprocessing: grayscale, contrast enhancement, and adaptive threshold.
    Good for moderate quality images.
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Light blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Adaptive thresholding with larger block size
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    
    return thresh


def preprocess_image_for_ocr_advanced(img: np.ndarray) -> np.ndarray:
    """
    Advanced preprocessing: denoising + OTSU thresholding.
    Use for noisy or poor quality images.
    
    """
    if img is None:
        raise ValueError("Input image is None")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # OTSU thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def preprocess_image_for_ocr_morphology(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing with morphological operations.
    Good for images with broken text or artifacts
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # OTSU thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def preprocess_image_for_ocr_scale_aware(img: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
    """
    Scale-aware preprocessing: upscales image before processing.
    Good for small or low-resolution images.
    
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale using INTER_CUBIC for better quality
    height, width = gray.shape
    upscaled = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)), 
                          interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(upscaled)
    
    # Light denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised


def preprocess_image_for_visual_search(
    img: np.ndarray,
    target_size: int = 400
) -> Optional[np.ndarray]:
    """
    Minimal preprocessing for visual search:
    1. Convert to grayscale
    2. Resize to a fixed square (default 400x400)
    """
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size))
    return resized

