const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const OCR_API_BASE_URL = "http://localhost:5000";

/**
 * Upload a payment receipt for verification
 * @param {File} imageFile - The receipt image
 * @param {File|null} chatFile - The chat log file (required for 'digital' mode)
 * @param {string|null} preprocessingMethod - Preprocessing option
 * @param {string} mode - 'standard' (FastAPI) or 'digital' (Flask)
 */
export async function verifyPayment(imageFile, chatFile = null, preprocessingMethod = null, mode = 'standard') {
  const formData = new FormData();

  if (mode === 'digital') {
    // --- APPROACH A: FLASK (ocr_digital_images.py) ---
    if (!chatFile) {
      throw new Error("Chat file is required for Digital/Chat verification.");
    }
    
    // Flask expects "image" and "chat_file"
    formData.append("image", imageFile); 
    formData.append("chat_file", chatFile);

    const url = new URL(`${OCR_API_BASE_URL}/verify_payment`);
    
    // Note: The Flask app currently doesn't look for preprocessing_method in the query params 
    // but we can send it just in case you update the backend later.
    if (preprocessingMethod) {
      url.searchParams.append("preprocessing_method", preprocessingMethod);
    }
    
    const response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      // Flask returns { "error": "message" }
      throw new Error(error.error || error.detail || `HTTP error! status: ${response.status}`);
    }

    // Tag source for frontend handling so we know how to display the data
    return { ...await response.json(), _source: 'digital' }; 

  } else {
    // --- APPROACH B: FASTAPI (main.py / routes/payment.py) ---
    
    // FastAPI expects "receipt_image"
    formData.append("receipt_image", imageFile);

    const url = new URL(`${API_BASE_URL}/api/v1/payment/verify`);
    if (preprocessingMethod) {
      url.searchParams.append("preprocessing_method", preprocessingMethod);
    }

    const response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return { ...await response.json(), _source: 'standard' };
  }
}

/**
 * Search for similar products using visual search
 */
export async function searchProduct(file, topK = 5, clipWeight = 0.4, visualWeight = 0.6, userCategory = null) {
  const formData = new FormData();
  formData.append("query_image", file);

  const url = new URL(`${API_BASE_URL}/api/v1/product/search`);

  if (userCategory) {
    url.searchParams.append("user_category", userCategory);
  }
  
  url.searchParams.append("top_k", topK.toString());
  url.searchParams.append("clip_weight", clipWeight.toString());
  url.searchParams.append("visual_weight", visualWeight.toString());

  const response = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get the full URL for an image from the backend
 */
export function getImageUrl(imagePath) {
  if (!imagePath) return null;
  if (imagePath.startsWith("http")) return imagePath;
  const baseUrl = API_BASE_URL.replace(/\/+$/, ""); 
  const path = imagePath.replace(/^\/+/, ""); 
  return `${baseUrl}/${path}`;
}

/**
 * Check API health
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("API is not available");
  }
  return await response.json();
}