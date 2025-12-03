"""
Pydantic schemas for request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ImageQuality(BaseModel):
    """Image quality metrics."""
    overall_score: Optional[float] = Field(None, description="Overall quality score (0-100)")
    noise_level: Optional[float] = Field(None, description="Noise level (0-100, higher = more noise)")
    contrast: Optional[float] = Field(None, description="Contrast score (0-100)")
    brightness: Optional[float] = Field(None, description="Brightness score (0-100)")
    blur_level: Optional[float] = Field(None, description="Blur level (0-100, lower = more blur)")
    resolution_score: Optional[float] = Field(None, description="Resolution score (0-100)")


class AuthenticityCheck(BaseModel):
    """Model for a single authenticity check."""
    check: str = Field(..., description="Name of the check (e.g., metadata_integrity, copy_move_detection)")
    score: float = Field(..., description="Score for this check (0.0-1.0)")
    weight: float = Field(..., description="Weight of this check in overall score")
    description: str = Field(..., description="Human-readable description of the check")


class AuthenticityResult(BaseModel):
    """Model for image authenticity analysis."""
    authenticity_score: float = Field(..., description="Overall authenticity score (0.0-1.0)")
    recommendation: str = Field(..., description="Recommendation: AUTHENTIC, LIKELY_AUTHENTIC, SUSPICIOUS, or LIKELY_FORGED")
    confidence_level: str = Field(..., description="Confidence level: HIGH, MEDIUM, or LOW")
    is_suspicious: bool = Field(..., description="Whether image is suspicious")
    indicators: Optional[List[AuthenticityCheck]] = Field(None, description="Per-check authenticity indicators")
    details: Optional[Dict[str, int]] = Field(None, description="Check details (total, passed, failed)")


class ReceiptVerificationResponse(BaseModel):
    """Response model for receipt verification."""
    transaction_id: Optional[str] = None
    amount: Optional[str] = None
    date: Optional[str] = None
    verification_status: str = Field(..., description="Status: verified, partial, failed, or error")
    extraction_method: Optional[str] = None
    preprocessing_method: Optional[str] = Field(None, description="Preprocessing method used")
    auto_selected: Optional[bool] = Field(None, description="Whether preprocessing method was auto-selected")
    image_quality: Optional[ImageQuality] = Field(None, description="Image quality metrics (if auto-selected)")
    llm_corrections: Optional[Dict[str, str]] = Field(None, description="LLM correction explanations (if LLM validation used)")
    llm_confidence: Optional[Dict[str, float]] = Field(None, description="LLM confidence scores (0.0-1.0)")
    llm_explanations: Optional[Dict[str, str]] = Field(None, description="LLM explanations of where fields were found or why null")
    authenticity: Optional[AuthenticityResult] = Field(None, description="Image authenticity analysis and forensic checks")
    processing_id: Optional[str] = Field(None, description="Unique ID for this processing session")
    processed_image_path: Optional[str] = Field(None, description="Server path to saved preprocessed image")
    processed_image_url: Optional[str] = Field(None, description="URL to access the preprocessed image")
    error: Optional[str] = None


class ProductMatch(BaseModel):
    """Model for a product match result."""
    product_image: str = Field(..., description="Filename of matched product image")
    product_image_url: Optional[str] = Field(None, description="URL for matched inventory image")
    feature_method: Optional[str] = Field(None, description="Feature extractor used (ResNet50/SIFT/ORB)")
    similarity_score: float = Field(..., description="Combined similarity score (0-1)")
    match_confidence: int = Field(..., description="Confidence percentage (0-100)")
    color_similarity: Optional[float] = Field(None, description="Color similarity percentage (0-100)")
    semantic_similarity: Optional[float] = Field(None, description="Semantic similarity score (0-1)")


class ProductSearchResponse(BaseModel):
    """Response model for product search."""
    matches: List[ProductMatch] = Field(..., description="List of top matching products")
    total_matches: int = Field(..., description="Total number of matches found")
    query_image_path: Optional[str] = Field(None, description="Disk path to saved query image")
    query_image_url: Optional[str] = Field(None, description="URL to access the uploaded query image")
    query_processed_image_path: Optional[str] = Field(None, description="Disk path to preprocessed query image")
    query_processed_image_url: Optional[str] = Field(None, description="URL to access the preprocessed query image")


class BatchVerificationItem(BaseModel):
    """Model for a single item in batch verification."""
    filename: str
    verification: ReceiptVerificationResponse


class BatchVerificationResponse(BaseModel):
    """Response model for batch verification."""
    status: str = "success"
    total_processed: int
    results: List[BatchVerificationItem]

