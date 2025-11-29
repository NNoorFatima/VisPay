"""
Configuration management using environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    API_TITLE: str = "VisPay Vision API"
    API_DESCRIPTION: str = "Dual Image Intelligence System for Payment Verification and Visual Product Search"
    API_VERSION: str = "1.0.0"
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # OCR Settings
    TESSERACT_CMD: Optional[str] = None  
    USE_EASYOCR: bool = False  # Set to True to use EasyOCR instead of pytesseract
    OCR_PREPROCESSING_METHOD: str = "minimal"  # Options: minimal, light, medium, advanced, morphology, scale_aware, auto (used if auto is disabled)
    OCR_AUTO_PREPROCESSING: bool = True  
    
    # LLM Extraction Settings 
    GEMINI_API_KEY: Optional[str] = None  
    LLM_MODEL: str = "gemini-2.5-pro"  
    
    USE_LLM_VALIDATION: bool = True  
    
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_OCR_EXTRACTION: bool = True  
    
    # Visual Search Settings
    INVENTORY_DIR: str = "static/product_images"
    VISUAL_SEARCH_INDEX_PATH: Optional[str] = None  # Path to save/load feature index (e.g., "data/visual_search_index.pkl")
    VISUAL_SEARCH_COLOR_WEIGHT: float = 0.3  # Weight for color similarity (0-1)  
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Processed Images Settings
    PROCESSED_DIR: str = "processed"  
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.INVENTORY_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

if settings.TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

