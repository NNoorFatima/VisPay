"use client";

import { useState } from "react";
import {
  Upload,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Copy,
  Check,
} from "lucide-react";

export default function PaymentVerification({ onResultChange }) {
  const [image, setImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        simulateProcessing();
      };
      reader.readAsDataURL(file);
    }
  };

  const simulateProcessing = () => {
    setIsProcessing(true);
    onResultChange?.(null);

    setTimeout(() => {
      const verificationResult = {
        status: "success",
        confidence: 98,
        processingTime: 1847,
        data: {
          "Transaction ID": "TXN2024110987654",
          Amount: "₹2,500.00",
          Date: "27 Nov 2024",
          "Payment Method": "Bank Transfer",
          "OCR Status": "HIGH confidence",
          Manipulation: "None detected",
        },
      };
      setResult(verificationResult);
      onResultChange?.(verificationResult);
      setIsProcessing(false);
    }, 2000);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
                  Upload Payment Receipt
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Drag and drop or click to select a receipt image
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
                alt="Receipt"
                className="w-full max-h-96 object-contain rounded-lg border border-border"
              />
              <button
                onClick={() => {
                  setImage(null);
                  setResult(null);
                  onResultChange?.(null);
                }}
                className="mt-4 w-full px-4 py-2 text-sm text-primary hover:bg-primary/10 rounded-lg border border-primary/30 transition-colors"
              >
                Upload Different Receipt
              </button>
            </div>

            {isProcessing && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-primary animate-spin mr-2" />
                <span className="text-foreground font-medium">
                  Verifying receipt...
                </span>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <div
                  className={`flex items-center gap-3 p-4 rounded-lg border-2 ${
                    result.status === "success"
                      ? "bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-800"
                      : "bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800"
                  }`}
                >
                  {result.status === "success" ? (
                    <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400 flex-shrink-0" />
                  ) : (
                    <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0" />
                  )}
                  <div>
                    <p
                      className={`font-semibold ${
                        result.status === "success"
                          ? "text-green-900 dark:text-green-100"
                          : "text-red-900 dark:text-red-100"
                      }`}
                    >
                      Receipt{" "}
                      {result.status === "success" ? "Verified" : "Failed"}
                    </p>
                    <p
                      className={`text-sm ${
                        result.status === "success"
                          ? "text-green-700 dark:text-green-200"
                          : "text-red-700 dark:text-red-200"
                      }`}
                    >
                      Confidence: {result.confidence}%
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: "Transaction ID", value: "TXN2024110987654" },
                    { label: "Amount", value: "₹2,500.00" },
                    { label: "Date", value: "27 Nov 2024" },
                    { label: "Method", value: "Bank Transfer" },
                  ].map((detail, idx) => (
                    <div key={idx} className="p-4 bg-secondary/30 rounded-lg">
                      <p className="text-xs text-muted-foreground font-medium mb-1">
                        {detail.label}
                      </p>
                      <div className="flex items-center justify-between">
                        <p className="font-semibold text-foreground">
                          {detail.value}
                        </p>
                        <button
                          onClick={() => copyToClipboard(detail.value)}
                          className="p-1 hover:bg-primary/20 rounded transition-colors"
                        >
                          {copied ? (
                            <Check className="w-4 h-4 text-primary" />
                          ) : (
                            <Copy className="w-4 h-4 text-muted-foreground" />
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="space-y-2 p-4 bg-secondary/20 rounded-lg">
                  <p className="text-sm font-semibold text-foreground mb-3">
                    Verification Checks
                  </p>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className="text-sm text-foreground">
                      High confidence OCR extraction
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className="text-sm text-foreground">
                      No manipulation detected
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className="text-sm text-foreground">
                      Amount validation passed
                    </span>
                  </div>
                </div>

                <button className="w-full py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:opacity-90 transition-opacity">
                  Confirm & Process Order
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
