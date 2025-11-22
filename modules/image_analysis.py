"""
Image quality analysis for automatic preprocessing method selection.
"""
import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_noise_level(gray: np.ndarray) -> float:
    """
    Calculate noise level using difference between original and smoothed image.
    Higher values indicate more noise.
    """
    # Apply bilateral filter to remove noise while preserving edges
    smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Calculate difference - high difference means high noise
    diff = cv2.absdiff(gray, smoothed)
    noise_variance = np.var(diff)
    
    # Normalize to 0-100 scale
    # Typical noise variance for clean images: 0-50
    # For noisy images: 50-200+
    noise_score = min(100, (noise_variance / 2))
    return noise_score


def calculate_contrast(gray: np.ndarray) -> float:
    """
    Calculate image contrast using standard deviation of pixel values.
    """
    mean, stddev = cv2.meanStdDev(gray)
    # Normalize to 0-100 scale
    contrast_score = min(100, (stddev[0][0] / 2.55))
    return contrast_score


def calculate_brightness(gray: np.ndarray) -> float:
    """
    Calculate average brightness of the image.
    """
    mean_brightness = np.mean(gray)
    # Normalize to 0-100 scale
    brightness_score = (mean_brightness / 255) * 100
    return brightness_score


def calculate_blur_level(gray: np.ndarray) -> float:
    """
    Calculate blur level using variance of Laplacian.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize to 0-100 scale
    blur_score = min(100, (laplacian_var / 10))
    return blur_score


def calculate_resolution_score(img: np.ndarray) -> float:
    """
    Calculate resolution score based on image dimensions.
    """
    height, width = img.shape[:2]
    total_pixels = height * width
    
    # Score based on total pixels
    # 1MP = 1000000 pixels = 100 score
    # 0.5MP = 500000 pixels = 50 score
    resolution_score = min(100, (total_pixels / 10000))
    
    # Boost score if image is very large
    if total_pixels > 2000000:  # > 2MP
        resolution_score = 100
    elif total_pixels > 1000000:  # > 1MP
        resolution_score = min(100, resolution_score * 1.2)
    
    return resolution_score


def calculate_text_density(gray: np.ndarray) -> float:
    """
    Estimate text density using edge detection.
    """
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate edge pixel density
    edge_pixels = np.sum(edges > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    density = (edge_pixels / total_pixels) * 100
    
    return min(100, density * 10)  # Scale up


def analyze_image_quality(img: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive image quality analysis.
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate various quality metrics
    noise_level = calculate_noise_level(gray)
    contrast = calculate_contrast(gray)
    brightness = calculate_brightness(gray)
    blur_level = calculate_blur_level(gray)
    resolution_score = calculate_resolution_score(img)
    text_density = calculate_text_density(gray)
    
    # Overall quality score (0-100, higher = better)
    # Weighted combination of metrics
    overall_quality = (
        contrast * 0.25 +
        (100 - noise_level) * 0.20 +
        blur_level * 0.15 +
        resolution_score * 0.20 +
        text_density * 0.10 +
        (100 - abs(brightness - 50)) * 0.10  # Prefer medium brightness
    )
    
    return {
        "noise_level": noise_level,
        "contrast": contrast,
        "brightness": brightness,
        "blur_level": blur_level,
        "resolution_score": resolution_score,
        "text_density": text_density,
        "overall_quality": overall_quality
    }


def select_preprocessing_method(quality_metrics: Dict[str, float]) -> str:
    """
    Automatically select the best preprocessing method based on image quality.
    More conservative approach - defaults to lighter methods.
    """
    noise = quality_metrics["noise_level"]
    contrast = quality_metrics["contrast"]
    brightness = quality_metrics["brightness"]
    blur = quality_metrics["blur_level"]
    resolution = quality_metrics["resolution_score"]
    overall = quality_metrics["overall_quality"]
    
    
    # VERY SPECIFIC: Very low resolution - needs upscaling
    if resolution < 20:
        return "scale_aware"
    
    # VERY SPECIFIC: Very high quality - minimal processing
    if overall > 75 and noise < 15 and contrast > 65 and blur > 40:
        return "minimal"
    
    # VERY SPECIFIC: Extremely noisy (>70) - heavy denoising needed
    if noise > 70:
        return "advanced"
    
    # SPECIFIC: Very blurry (<15) AND noisy - needs morphology cleanup
    if blur < 15 and noise > 40:
        return "morphology"
    
    # SPECIFIC: Very blurry (<15) but not noisy - try advanced sharpening
    if blur < 15 and noise < 20:
        return "advanced"
    
    # MODERATE: Noisy images (40-70) - medium preprocessing
    if noise > 40:
        if blur < 30:  # Blurry too
            return "morphology"
        else:
            return "medium"
    
    # MODERATE: Low contrast (<35) - enhance contrast
    if contrast < 35:
        if noise < 25:
            return "light"
        else:
            return "medium"
    
    # MODERATE: Medium-low resolution - consider upscaling
    if resolution < 35 and overall < 60:
        return "scale_aware"
    
    # GOOD: High quality overall (>70) - light enhancement
    if overall > 70:
        if contrast > 55:
            return "minimal"  # Try minimal first for high quality
        else:
            return "light"
    
    # GOOD: Decent quality (60-70) - light preprocessing
    if overall > 60:
        return "light"
    
    # FAIR: Moderate quality (50-60) - light to medium
    if overall > 50:
        if noise > 25:
            return "medium"
        else:
            return "light"
    
    if noise > 50 or blur < 20:
        return "medium"  
    
    return "minimal"


def auto_select_preprocessing(img: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Analyze image and automatically select best preprocessing method.
    """
    try:
        quality_metrics = analyze_image_quality(img)
        
        method = select_preprocessing_method(quality_metrics)
        
        logger.debug(f"Auto-selected preprocessing: {method} (quality: {quality_metrics['overall_quality']:.1f})")
        
        return method, quality_metrics
    
    except Exception as e:
        logger.warning(f"Error in auto preprocessing selection: {e}, defaulting to 'minimal'")
        return "minimal", {}

