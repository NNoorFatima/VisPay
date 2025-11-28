"use client";

import { useState } from "react";
import { Upload, Search, Loader2, ShoppingCart } from "lucide-react";

export default function ProductSearch({ onResultChange }) {
  const [image, setImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        simulateSearch();
      };
      reader.readAsDataURL(file);
    }
  };

  const simulateSearch = () => {
    setIsProcessing(true);
    onResultChange?.(null);

    setTimeout(() => {
      const searchResults = [
        {
          id: 1,
          name: "Premium Silk Dress",
          price: "₹1,899",
          match: 98,
          color: "Navy Blue",
          size: "M, L, XL",
          stock: 15,
        },
        {
          id: 2,
          name: "Classic Silk Blouse",
          price: "₹1,299",
          match: 94,
          color: "Navy Blue",
          size: "S, M, L",
          stock: 22,
        },
        {
          id: 3,
          name: "Elegant Evening Wear",
          price: "₹2,499",
          match: 89,
          color: "Dark Blue",
          size: "M, L, XL, XXL",
          stock: 8,
        },
      ];
      setResults(searchResults);
      onResultChange?.({
        status: "success",
        confidence: searchResults[0].match,
        processingTime: 1624,
        data: {
          "Top Match": searchResults[0].name,
          Price: searchResults[0].price,
          Color: searchResults[0].color,
          "Available Sizes": searchResults[0].size,
          Stock: `${searchResults[0].stock} units`,
          "Total Results": `${searchResults.length} products found`,
        },
      });
      setIsProcessing(false);
    }, 2000);
  };

  return (
    <div className="p-8">
      <div className="max-w-2xl mx-auto">
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
              <button
                onClick={() => {
                  setImage(null);
                  setResults(null);
                  onResultChange?.(null);
                }}
                className="mt-4 w-full px-4 py-2 text-sm text-primary hover:bg-primary/10 rounded-lg border border-primary/30 transition-colors"
              >
                Search Different Product
              </button>
            </div>

            {isProcessing && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 text-primary animate-spin mr-2" />
                <span className="text-foreground font-medium">
                  Analyzing product features...
                </span>
              </div>
            )}

            {results && (
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
                  {results.map((product) => (
                    <div
                      key={product.id}
                      className="p-4 border border-border rounded-lg hover:border-primary/50 transition-colors cursor-pointer group"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <p className="font-semibold text-foreground group-hover:text-primary transition-colors">
                            {product.name}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {product.color}
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="font-bold text-primary">
                            {product.price}
                          </p>
                          <div className="mt-1 inline-flex items-center gap-1 px-2 py-1 bg-primary/10 rounded">
                            <span className="text-xs font-semibold text-primary">
                              {product.match}%
                            </span>
                            <span className="text-xs text-primary">match</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <div className="space-y-1">
                          <p className="text-muted-foreground">
                            Available sizes:{" "}
                            <span className="text-foreground font-medium">
                              {product.size}
                            </span>
                          </p>
                          <p className="text-muted-foreground">
                            In stock:{" "}
                            <span className="text-foreground font-medium">
                              {product.stock} units
                            </span>
                          </p>
                        </div>
                        <button className="p-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity">
                          <ShoppingCart className="w-5 h-5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
