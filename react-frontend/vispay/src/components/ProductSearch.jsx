"use client";

import { useState, useEffect } from "react";
import {
  Upload,
  Search,
  Loader2,
  Info,
  AlertTriangle,
  Check,
  X,
} from "lucide-react";
import { searchProduct, getImageUrl } from "../services/api";

export default function ProductSearch({ onResultChange, onProcessingChange }) {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [topK, setTopK] = useState(5);
  // const [colorWeight, setColorWeight] = useState(0.3);
  const [showMethodInfo, setShowMethodInfo] = useState(false);
const [clipWeight, setClipWeight] = useState(0.4); // New state for CLIP weight
const [visualWeight, setVisualWeight] = useState(0.6); // New state for Visual weight
  // New State for Backend Messages (e.g., "Found in Watch" or "No Watches found")
  const [searchMessage, setSearchMessage] = useState(""); 

  // Modal State
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [detectedCategory, setDetectedCategory] = useState("unknown");
  const [manualCategoryInput, setManualCategoryInput] = useState("");
  const [confidenceScore, setConfidenceScore] = useState(0);

  useEffect(() => {
    console.log("Current Image File State:", imageFile ? "Loaded" : "Null");
  }, [imageFile]);

  const updateProcessingState = (processing) => {
    setIsProcessing(processing);
    onProcessingChange?.(processing);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setImageFile(file);
        setResults(null);
        setError(null);
        setSearchMessage(""); // Reset message
        setShowConfirmModal(false);
        performSearch(file);
      };
      reader.readAsDataURL(file);
    }
  };

  const performSearch = async (fileToUse, categoryOverride = null) => {
    if (!fileToUse) return;

    updateProcessingState(true);
    setError(null);
    setSearchMessage(""); // Clear previous messages
    onResultChange?.(null);
    
    if (!categoryOverride) setShowConfirmModal(false);

    try {
      const startTime = Date.now();
      // === GENIUS CODER CHANGE: Use explicit clipWeight and visualWeight ===
      // NOTE: We pass both to the API call.
      const apiResult = await searchProduct(fileToUse, topK, clipWeight, visualWeight, categoryOverride); 
      // === END GENIUS CODER CHANGE ===
      //const apiResult = await searchProduct(fileToUse, topK, colorWeight, categoryOverride);
      console.log("API Response:", apiResult);

      const processingTime = Date.now() - startTime;

      // 1. Handle "Needs Confirmation" (Low Confidence)
      if (apiResult.alert_user && !categoryOverride) {
        setDetectedCategory(apiResult.detected_category);
        setConfidenceScore(apiResult.confidence || 0);
        setShowConfirmModal(true);
        updateProcessingState(false);
        return; 
      }

      // 2. Set Results & Message
      setResults(apiResult.matches);
      // Capture the message from backend (e.g. "Sorry, no watches found")
      if (apiResult.alert_message) {
        setSearchMessage(apiResult.alert_message);
      }

      // 3. Transform for Parent Component (Optional)
      const searchResult = {
        type: "product",
        status: apiResult.matches && apiResult.matches.length > 0 ? "success" : "no_results",
        apiResult,
        queryImageUrl: apiResult.query_image_url ? getImageUrl(apiResult.query_image_url) : null,
      };

      onResultChange?.(searchResult);

    } catch (err) {
      console.error("Search Error:", err);
      setError(err.message || "Failed to search products");
      onResultChange?.({ type: "product", status: "error", error: err.message });
    } finally {
      if (!showConfirmModal) {
         updateProcessingState(false);
      }
    }
  };

  const handleCategoryConfirm = (category) => {
    setShowConfirmModal(false);
    if (imageFile) {
        performSearch(imageFile, category);
    } else {
        setError("Session expired. Please re-upload image.");
    }
  };

  return (
    <div className="p-8 relative">
      <div className="max-w-2xl mx-auto">
        {/* Info Panel */}
        <div className="mb-6 p-4 bg-card border border-border rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-foreground">Visual Product Search</h3>
            </div>
          </div>
        </div>

        {/* Modal */}
        {showConfirmModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
                <div className="bg-card border border-border rounded-xl shadow-2xl max-w-md w-full p-6">
                    <div className="flex items-center gap-3 mb-4 text-amber-500">
                        <AlertTriangle className="w-8 h-8" />
                        <h3 className="text-lg font-bold text-foreground">Verify Category</h3>
                    </div>
                    <p className="text-muted-foreground mb-6">
                        The AI isn't confident. 
                        {detectedCategory !== 'unknown' ? (
                            <span> Is this a <span className="font-bold text-primary">{detectedCategory}</span>?</span>
                        ) : (
                            <span> We couldn't identify this item.</span>
                        )}
                    </p>
                    {detectedCategory !== 'unknown' && (
                        <div className="flex gap-3 mb-6">
                            <button type="button" onClick={() => handleCategoryConfirm(detectedCategory)} className="flex-1 flex items-center justify-center gap-2 bg-primary text-primary-foreground py-2 rounded-lg hover:opacity-90">
                                <Check className="w-4 h-4" /> Yes
                            </button>
                            <button type="button" onClick={() => setDetectedCategory("unknown")} className="flex-1 flex items-center justify-center gap-2 bg-secondary text-secondary-foreground py-2 rounded-lg hover:bg-secondary/80">
                                <X className="w-4 h-4" /> No
                            </button>
                        </div>
                    )}
                    {detectedCategory === 'unknown' && (
                        <div className="mb-6">
                            <label className="block text-sm font-medium mb-2">What is this item?</label>
                            <div className="flex gap-2">
                                <input 
                                    type="text" 
                                    placeholder="e.g. Watch"
                                    className="flex-1 bg-background border border-border rounded-lg px-3 py-2"
                                    value={manualCategoryInput}
                                    onChange={(e) => setManualCategoryInput(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleCategoryConfirm(manualCategoryInput)}
                                />
                                <button type="button" onClick={() => handleCategoryConfirm(manualCategoryInput)} disabled={!manualCategoryInput.trim()} className="bg-primary text-primary-foreground px-4 py-2 rounded-lg disabled:opacity-50">
                                    Search
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        )}

        {/* Upload Area */}
        {!image ? (
          <div className="mb-8">
            <label className="block">
              <div className="border-2 border-dashed border-primary/30 hover:border-primary/60 rounded-lg p-12 text-center cursor-pointer bg-primary/5">
                <Upload className="w-12 h-12 text-primary mx-auto mb-4" />
                <p className="text-lg font-semibold text-foreground mb-1">Upload Image</p>
                <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
              </div>
            </label>
          </div>
        ) : (
          <div className="mb-8">
            <img src={image} alt="Preview" className="w-full max-h-96 object-contain rounded-lg border border-border" />
            <div className="mt-4 flex gap-2">
              <button 
                type="button"
                onClick={() => { setImage(null); setImageFile(null); setResults(null); setShowConfirmModal(false); }}
                className="flex-1 px-4 py-2 text-sm text-primary border border-primary/30 rounded-lg"
              >
                Search Different Product
              </button>
            </div>
          </div>
        )}

        {/* Loader */}
        {isProcessing && (
           <div className="flex items-center justify-center py-12">
             <Loader2 className="w-6 h-6 text-primary animate-spin mr-2" />
             <span>Analyzing...</span>
           </div>
        )}

        {/* --- RESULTS SECTION --- */}
        {results && !isProcessing && !showConfirmModal && (
            <div className="space-y-3">
                {/* SUCCESS MESSAGE (e.g. Found in Dress) */}
                {results.length > 0 && searchMessage && (
                    <div className="p-3 bg-green-500/10 border border-green-500/20 text-green-600 rounded-lg mb-4 text-sm font-medium flex items-center gap-2">
                        <Check className="w-4 h-4" />
                        {searchMessage}
                    </div>
                )}

                {results.map((product, idx) => (
                    <div key={idx} className="p-4 border border-border rounded-lg flex items-center gap-4 hover:border-primary/50 transition-colors">
                        {product.product_image_url && (
                             <img src={getImageUrl(product.product_image_url)} alt="product" className="w-16 h-16 object-cover rounded-md bg-secondary" />
                        )}
                        <div>
                            <p className="font-bold">{product.product_image}</p>
                            <p className="text-sm text-muted-foreground">Confidence: {product.match_confidence}%</p>
                        </div>
                    </div>
                ))}
            </div>
        )}

        {/* --- NO MATCHES FOUND SECTION --- */}
        {results && results.length === 0 && !isProcessing && (
          <div className="text-center py-12">
            <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-foreground font-semibold mb-2">No matches found</p>
            
            {/* Display the REASON why (e.g. "We have no watches") */}
            {searchMessage && (
                <p className="text-sm text-red-500/80 max-w-md mx-auto bg-red-500/5 p-2 rounded">
                    {searchMessage}
                </p>
            )}
            
            {!searchMessage && (
                <p className="text-sm text-muted-foreground">
                Try adjusting the search parameters or upload a different image
                </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}