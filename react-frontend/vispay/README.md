# VisPay Vision - Frontend

Modern React frontend for VisPay Vision API - an intelligent payment verification and visual product search system.

## Features

- **Payment Verification**: Upload receipt images for OCR-based verification with multiple preprocessing methods
- **Visual Product Search**: Find similar products in inventory using ResNet50 deep learning features
- **Real-time Processing**: Live updates with processed images and detailed method information
- **Modern UI**: Built with React, Tailwind CSS, and Radix UI components
- **Method Documentation**: Inline information about processing methods and techniques

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on `http://localhost:8000` (or configure via environment variables)

### Installation

```bash
cd react-frontend/vispay
npm install
```

### Configuration

Create a `.env` file in the `react-frontend/vispay` directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the next available port).

### Production Build

```bash
npm run build
npm run preview
```

## API Integration

### Payment Verification

**Endpoint**: `POST /api/v1/payment/verify`

**Features**:

- Multiple preprocessing methods (auto, minimal, light, medium, advanced, morphology, scale_aware)
- OCR text extraction with EasyOCR or Tesseract
- Optional LLM validation for intelligent field correction
- Image quality analysis
- Processed image preview

**Method Details**:

- Uses OCR (Optical Character Recognition) to extract text from receipt images
- Supports multiple preprocessing methods to enhance image quality
- Validates extracted data (transaction ID, amount, date)
- Optional LLM validation for intelligent field correction
- Detects image manipulation and quality issues

### Visual Product Search

**Endpoint**: `POST /api/v1/product/search`

**Features**:

- ResNet50 deep learning feature extraction
- Color histogram analysis
- Adjustable similarity parameters (top-K, color weight)
- Real-time similarity scoring

**Method Details**:

- Uses ResNet50 deep learning model to extract semantic features from images
- Combines color histogram analysis for color-based matching
- Calculates similarity scores using cosine similarity on feature vectors
- Returns top-K most similar products ranked by combined similarity score
- Supports adjustable color weight to balance semantic vs color matching

## Project Structure

```
src/
├── components/
│   ├── PaymentVerfication.jsx  # Payment receipt verification UI
│   ├── ProductSearch.jsx        # Visual product search UI
│   ├── ResultsPanel.jsx         # Right-side results panel with processed images
│   └── Navigation.jsx           # Top navigation bar
├── services/
│   └── api.js                   # API service layer
├── App.jsx                      # Main application component
└── main.jsx                     # Application entry point
```

## Key Components

### PaymentVerification

- Image upload with drag & drop
- Preprocessing method selection
- Real-time verification status
- Detailed extraction results
- LLM correction display

### ProductSearch

- Product image upload
- Adjustable search parameters (top-K, color weight)
- Similarity score visualization
- Match results with product images

### ResultsPanel

- Processed image display
- Method details and metrics
- Confidence scores
- Processing time
- Image quality metrics (for payment verification)
- LLM explanations (when available)

## API Service

The `services/api.js` file provides:

- `verifyPayment(file, preprocessingMethod)` - Verify payment receipt
- `searchProduct(file, topK, colorWeight)` - Search for similar products
- `getImageUrl(imagePath)` - Get full URL for backend images
- `checkHealth()` - Check API health status

## Styling

The application uses:

- **Tailwind CSS** for utility-first styling
- **Radix UI** for accessible component primitives
- **Lucide React** for icons
- Custom color scheme with dark mode support

## Development Notes

- The Vite dev server includes proxy configuration for API routes
- Processed images are displayed in the right-side panel
- All API responses include detailed method information
- Error handling with user-friendly messages
- Loading states for better UX

## Backend Integration

The frontend communicates with the FastAPI backend:

- **Base URL**: Configurable via `VITE_API_BASE_URL` environment variable
- **Proxy**: Vite dev server proxies `/api`, `/processed`, `/uploads`, and `/inventory` routes
- **CORS**: Backend must allow requests from frontend origin

## License

Part of the VisPay Vision project.
