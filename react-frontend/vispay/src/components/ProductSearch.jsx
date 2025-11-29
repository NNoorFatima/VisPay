"use client";

import { useState } from "react";
import {
  Upload,
  Search,
  Loader2,
  ShoppingCart,
  Info,
  Settings,
} from "lucide-react";
import { searchProduct, getImageUrl } from "../services/api";

export default function ProductSearch({ onResultChange, onProcessingChange }) {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [topK, setTopK] = useState(5);
  const [colorWeight, setColorWeight] = useState(0.3);
  const [showMethodInfo, setShowMethodInfo] = useState(false);

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
        performSearch(file);
      };
      reader.readAsDataURL(file);
    }
  };

  const performSearch = async (file) => {
    updateProcessingState(true);
    setError(null);
    onResultChange?.(null);

    try {
      const startTime = Date.now();
      const apiResult = await searchProduct(file, topK, colorWeight);
      const processingTime = Date.now() - startTime;

      setResults(apiResult.matches);

      // Transform API response to frontend format
      const searchResult = {
        type: "product",
        status:
          apiResult.matches && apiResult.matches.length > 0
            ? "success"
            : "no_results",
        confidence: apiResult.matches?.[0]?.match_confidence || 0,
        processingTime,
        data: {
          "Total Matches": `${apiResult.total_matches} products found`,
          "Top Match": apiResult.matches?.[0]?.product_image || "N/A",
          "Match Confidence": apiResult.matches?.[0]?.match_confidence
            ? `${apiResult.matches[0].match_confidence}%`
            : "N/A",
          "Color Similarity": apiResult.matches?.[0]?.color_similarity
            ? `${apiResult.matches[0].color_similarity}%`
            : "N/A",
        },
        apiResult, // Store full API result
        queryImageUrl: apiResult.query_image_url
          ? getImageUrl(apiResult.query_image_url)
          : null,
        processedImageUrl: apiResult.query_processed_image_url
          ? getImageUrl(apiResult.query_processed_image_url)
          : null,
      };

      onResultChange?.(searchResult);
    } catch (err) {
      setError(err.message || "Failed to search products");
      const errorResult = {
        type: "product",
        status: "error",
        error: err.message,
      };
      onResultChange?.(errorResult);
    } finally {
      updateProcessingState(false);
    }
  };

  const handleRetry = () => {
    if (imageFile) {
      performSearch(imageFile);
    }
  };

  return (
    <div className="p-8">
      <div className="max-w-2xl mx-auto">
        {/* Method Information Panel */}
        <div className="mb-6 p-4 bg-card border border-border rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-foreground">
                Visual Product Search Method
              </h3>
            </div>
            <button
              onClick={() => setShowMethodInfo(!showMethodInfo)}
              className="p-1 hover:bg-primary/10 rounded transition-colors"
            >
              <Settings
                className={`w-4 h-4 text-muted-foreground transition-transform ${
                  showMethodInfo ? "rotate-90" : ""
                }`}
              />
            </button>
          </div>
          {showMethodInfo && (
            <div className="mt-3 space-y-2 text-sm text-muted-foreground">
              <p>
                • Uses ResNet50 deep learning model to extract semantic features
                from images
              </p>
              <p>
                • Combines color histogram analysis for color-based matching
              </p>
              <p>
                • Calculates similarity scores using cosine similarity on
                feature vectors
              </p>
              <p>
                • Returns top-K most similar products ranked by combined
                similarity score
              </p>
              <p>
                • Supports adjustable color weight to balance semantic vs color
                matching
              </p>
            </div>
          )}
        </div>

        {/* Search Parameters */}
        {image && (
          <div className="mb-6 p-4 bg-secondary/30 border border-border rounded-lg space-y-4">
            <div>
              <label className="block text-sm font-semibold text-foreground mb-2">
                Number of Results (Top-K): {topK}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => {
                  setTopK(Number(e.target.value));
                  if (imageFile && !isProcessing) {
                    performSearch(imageFile);
                  }
                }}
                disabled={isProcessing}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold text-foreground mb-2">
                Color Weight: {(colorWeight * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={colorWeight}
                onChange={(e) => {
                  setColorWeight(Number(e.target.value));
                  if (imageFile && !isProcessing) {
                    performSearch(imageFile);
                  }
                }}
                disabled={isProcessing}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Lower values prioritize semantic features, higher values
                prioritize color matching
              </p>
            </div>
          </div>
        )}

        {!image ? (
          <div className="mb-8">
            <label className="block">
              <div className="border-2 border-dashed border-primary/30 hover:border-primary/60 rounded-lg p-12 text-center cursor-pointer transition-colors bg-primary/5">
                <Upload className="w-12 h-12 text-primary mx-auto mb-4" />
                <p className="text-lg font-semibold text-foreground mb-1">
                  Upload Product Image
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Customer sends a photo, find matching products instantly
                </p>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </div>
            </label>
          </div>
        ) : (
          <>
            <div className="mb-8">
              <img
                src={image || "/placeholder.svg"}
                alt="Product"
                className="w-full max-h-96 object-contain rounded-lg border border-border"
              />
              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => {
                    setImage(null);
                    setImageFile(null);
                    setResults(null);
                    setError(null);
                    onResultChange?.(null);
                  }}
                  className="flex-1 px-4 py-2 text-sm text-primary hover:bg-primary/10 rounded-lg border border-primary/30 transition-colors"
                >
                  Search Different Product
                </button>
                {results && (
                  <button
                    onClick={handleRetry}
                    disabled={isProcessing}
                    className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
                  >
                    Retry
                  </button>
                )}
              </div>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-primary/10 border-2 border-primary/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <Search className="w-5 h-5 text-primary flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-foreground">
                      Search Failed
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 text-primary animate-spin mr-2" />
                <span className="text-foreground font-medium">
                  Analyzing product features...
                </span>
              </div>
            )}

            {results && results.length > 0 && !isProcessing && (
              <div>
                <div className="mb-6 p-4 bg-secondary/20 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <Search className="w-5 h-5 text-primary" />
                    <p className="font-semibold text-foreground">
                      Found {results.length} matching products
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Ranked by feature similarity and visual match accuracy
                  </p>
                </div>

                <div className="space-y-3">
                  {results.map((product, index) => (
                    <div
                      key={index}
                      className="p-4 border border-border rounded-lg hover:border-primary/50 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-start gap-4 mb-3">
                        {product.product_image_url && (
                          <img
                            src={getImageUrl(product.product_image_url)}
                            alt={product.product_image}
                            className="w-20 h-20 object-cover rounded-lg border border-border"
                            onError={(e) => {
                              e.target.style.display = "none";
                            }}
                          />
                        )}
                        <div className="flex-1">
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <p className="font-semibold text-foreground group-hover:text-primary transition-colors">
                                {product.product_image
                                  .replace(/\.(jpg|jpeg|png|gif)$/i, "")
                                  .replace(/[_-]/g, " ")}
                              </p>
                              <p className="text-xs text-muted-foreground mt-1">
                                {product.feature_method || "ResNet50"}
                              </p>
                            </div>
                            <div className="text-right">
                              <div className="inline-flex items-center gap-1 px-2 py-1 bg-primary/10 rounded">
                                <span className="text-xs font-semibold text-primary">
                                  {product.match_confidence}%
                                </span>
                                <span className="text-xs text-primary">
                                  match
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            {product.color_similarity !== null && (
                              <span>
                                Color:{" "}
                                <span className="text-foreground font-medium">
                                  {product.color_similarity.toFixed(1)}%
                                </span>
                              </span>
                            )}
                            {product.semantic_similarity !== null && (
                              <span>
                                Semantic:{" "}
                                <span className="text-foreground font-medium">
                                  {(product.semantic_similarity * 100).toFixed(
                                    1
                                  )}
                                  %
                                </span>
                              </span>
                            )}
                            <span>
                              Score:{" "}
                              <span className="text-foreground font-medium">
                                {(product.similarity_score * 100).toFixed(1)}%
                              </span>
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results && results.length === 0 && !isProcessing && (
              <div className="text-center py-12">
                <Search className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-foreground font-semibold mb-2">
                  No matches found
                </p>
                <p className="text-sm text-muted-foreground">
                  Try adjusting the search parameters or upload a different
                  image
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
