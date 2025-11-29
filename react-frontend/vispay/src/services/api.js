const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Upload a payment receipt for verification
 * @param {File} file - The receipt image file
 * @param {string} preprocessingMethod - Optional preprocessing method (auto, minimal, light, medium, advanced, morphology, scale_aware)
 * @returns {Promise<Object>} Verification result
 */
export async function verifyPayment(file, preprocessingMethod = null) {
  const formData = new FormData();
  formData.append("receipt_image", file);

  const url = new URL(`${API_BASE_URL}/api/v1/payment/verify`);
  if (preprocessingMethod) {
    url.searchParams.append("preprocessing_method", preprocessingMethod);
  }

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
 * Search for similar products using visual search
 * @param {File} file - The product image file
 * @param {number} topK - Number of top matches to return (1-20)
 * @param {number} colorWeight - Weight for color similarity (0-1)
 * @returns {Promise<Object>} Search results
 */
export async function searchProduct(file, topK = 5, colorWeight = 0.3) {
  const formData = new FormData();
  formData.append("query_image", file);

  const url = new URL(`${API_BASE_URL}/api/v1/product/search`);
  url.searchParams.append("top_k", topK.toString());
  url.searchParams.append("color_weight", colorWeight.toString());

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
 * @param {string} imagePath - Relative image path from backend
 * @returns {string} Full URL
 */
export function getImageUrl(imagePath) {
  if (!imagePath) return null;
  if (imagePath.startsWith("http")) return imagePath;
  return `${API_BASE_URL}${imagePath.startsWith("/") ? "" : "/"}${imagePath}`;
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("API is not available");
  }
  return await response.json();
}
