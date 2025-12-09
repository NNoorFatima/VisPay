const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const OCR_API_BASE_URL = "http://localhost:5000";
/**
 * Upload a payment receipt for verification
 */
// export async function verifyPayment(file, preprocessingMethod = null) {
//   const formData = new FormData();
//   formData.append("receipt_image", file);

//   const url = new URL(`${API_BASE_URL}/api/v1/payment/verify`);
//   if (preprocessingMethod) {
//     url.searchParams.append("preprocessing_method", preprocessingMethod);
//   }

//   const response = await fetch(url.toString(), {
//     method: "POST",
//     body: formData,
//   });

//   if (!response.ok) {
//     const error = await response
//       .json()
//       .catch(() => ({ detail: "Unknown error" }));
//     throw new Error(error.detail || `HTTP error! status: ${response.status}`);
//   }

//   return await response.json();
// }

export async function verifyPayment(imageFile, chatFile, preprocessingMethod = null) {
  const formData = new FormData();
  formData.append("image", imageFile); // File names match Flask
  formData.append("chat_file", chatFile);

  // *** USE OCR_API_BASE_URL HERE ***
  const url = new URL(`${OCR_API_BASE_URL}/verify_payment`); // Endpoint matches Flask
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
 * CORRECTED VERSION
 */
export async function searchProduct(file, topK = 5, clipWeight = 0.4, visualWeight = 0.6, userCategory = null) {
  const formData = new FormData();
  formData.append("query_image", file);

  // 1. Create the URL Object
  const url = new URL(`${API_BASE_URL}/api/v1/product/search`);

  // 2. Use .append() for ALL parameters (Do not use += string concatenation)
  if (userCategory) {
    url.searchParams.append("user_category", userCategory);
  }
  
  url.searchParams.append("top_k", topK.toString());
  //url.searchParams.append("color_weight", colorWeight.toString());
  // These match the backend route definition
  url.searchParams.append("clip_weight", clipWeight.toString());
  url.searchParams.append("visual_weight", visualWeight.toString());
  // Optional: Add visual_weight (calculated as 1 - color) if your backend uses it
  // const visualWeight = (1 - colorWeight).toFixed(2);
  // url.searchParams.append("visual_weight", visualWeight);

  // 3. Perform Fetch
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
  // Handle relative paths robustly
  const baseUrl = API_BASE_URL.replace(/\/+$/, ""); // Remove trailing slash
  const path = imagePath.replace(/^\/+/, ""); // Remove leading slash
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