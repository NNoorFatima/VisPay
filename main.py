"""
VisPay Vision API - Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
from config import settings

from routes import payment, product

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(payment.router)
app.include_router(product.router)

processed_dir = Path(settings.PROCESSED_DIR)
processed_dir.mkdir(parents=True, exist_ok=True)
app.mount("/processed", StaticFiles(directory=str(processed_dir)), name="processed")


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "VisPay Vision API",
        "version": settings.API_VERSION,
        "endpoints": {
            "payment_verification": "/api/v1/payment/verify",
            "batch_verification": "/api/v1/payment/verify-batch",
            "product_search": "/api/v1/product/search",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
