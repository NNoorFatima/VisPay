"use client";

import { useState } from "react";
import {
  Upload,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Copy,
  Check,
  Info,
  Settings,
  FileText,
} from "lucide-react";
import { verifyPayment, getImageUrl } from "../services/api";

const PREPROCESSING_METHODS = [
  { value: null, label: "Auto (Recommended)", description: "Automatically selects the best method based on image quality" },
  { value: "minimal", label: "Minimal", description: "Grayscale + sharpening (best for clear images)" },
  { value: "light", label: "Light", description: "Grayscale + contrast enhancement + light denoising" },
  { value: "medium", label: "Medium", description: "Light + adaptive thresholding" },
  { value: "advanced", label: "Advanced", description: "Denoising + OTSU thresholding (for noisy images)" },
  { value: "morphology", label: "Morphology", description: "Advanced + morphological operations (for broken text)" },
  { value: "scale_aware", label: "Scale Aware", description: "Upscales image before processing (for small images)" },
];

export default function PaymentVerification({ onResultChange, onProcessingChange }) {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  
  // New State for Mode and Chat File
  const [verificationMode, setVerificationMode] = useState("standard"); // 'standard' or 'digital'
  const [chatFile, setChatFile] = useState(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [preprocessingMethod, setPreprocessingMethod] = useState(null);
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
        setResult(null);
        setError(null);
        
        // Only auto-process in Standard mode. In Digital mode, we wait for Chat file.
        if (verificationMode === "standard") {
            processReceipt(file, null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleChatUpload = (e) => {
      const file = e.target.files?.[0];
      if (file) {
          setChatFile(file);
          setError(null);
      }
  };

  const handleVerifyClick = () => {
      if (imageFile) {
          processReceipt(imageFile, chatFile);
      }
  };

  const processReceipt = async (file, cFile) => {
    updateProcessingState(true);
    setError(null);
    onResultChange?.(null);

    try {
      const startTime = Date.now();
      
      // Call API with all necessary params including mode
      const apiResult = await verifyPayment(file, cFile, preprocessingMethod, verificationMode);
      const processingTime = Date.now() - startTime;

      let verificationResult;

      // NORMALIZE THE DATA based on the source
      if (apiResult._source === 'digital') {
          // --- FLASK RESPONSE MAPPING ---
          verificationResult = {
            type: "payment",
            status: apiResult.status === "APPROVED" ? "success" : 
                    apiResult.status === "MANUAL REVIEW" ? "partial" : "failed",
            confidence: apiResult.confidence,
            processingTime,
            data: {
              "Status": apiResult.status,
              "Transaction ID": apiResult.extracted_from_receipt?.transaction_id || "Not found",
              "Amount (Receipt)": apiResult.extracted_from_receipt?.amount || "Not found",
              "Amount (Chat)": apiResult.chat_amount || "Not found",
              "Email Verified": apiResult.email_data ? "Yes" : "No",
              "Fraud Detected": apiResult.fraud_detected ? "Yes" : "No",
              "Receiver": apiResult.extracted_from_receipt?.receiver || "Not found",
            },
            apiResult, // Keep full result for advanced debugging if needed
          };
      } else {
          // --- FASTAPI RESPONSE MAPPING ---
          verificationResult = {
            type: "payment",
            status: apiResult.verification_status === "verified" ? "success" : 
                    apiResult.verification_status === "partial" ? "partial" : "failed",
            confidence: apiResult.verification_status === "verified" ? 95 : 
                       apiResult.verification_status === "partial" ? 70 : 30,
            processingTime,
            data: {
              "Transaction ID": apiResult.transaction_id || "Not found",
              "Amount": apiResult.amount || "Not found",
              "Date": apiResult.date || "Not found",
              "Verification Status": apiResult.verification_status || "unknown",
              "Extraction Method": apiResult.extraction_method || "N/A",
              "Preprocessing Method": apiResult.preprocessing_method || "N/A",
              "Auto Selected": apiResult.auto_selected ? "Yes" : "No",
            },
            apiResult, 
            processedImageUrl: apiResult.processed_image_url ? getImageUrl(apiResult.processed_image_url) : null,
          };
      }

      // Add common extra fields if they exist
      if (apiResult.llm_corrections) {
        verificationResult.data["LLM Corrections"] = Object.keys(apiResult.llm_corrections).length + " fields corrected";
      }

      if (apiResult.image_quality) {
        verificationResult.imageQuality = apiResult.image_quality;
      }

      setResult(verificationResult);
      onResultChange?.(verificationResult);
    } catch (err) {
      setError(err.message || "Failed to verify receipt");
      const errorResult = {
        type: "payment",
        status: "error",
        error: err.message,
      };
      onResultChange?.(errorResult);
    } finally {
      updateProcessingState(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRetry = () => {
    if (imageFile) {
      processReceipt(imageFile, chatFile);
    }
  };

  return (
    <div className="p-8">
      <div className="max-w-2xl mx-auto">
        
        {/* --- MODE SELECTION --- */}
        <div className="mb-6 flex gap-4 bg-secondary/30 p-1.5 rounded-lg border border-border">
            <button 
                onClick={() => { setVerificationMode("standard"); setResult(null); setError(null); }}
                disabled={isProcessing}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                    verificationMode === "standard" 
                    ? "bg-primary text-white shadow-sm" 
                    : "text-muted-foreground hover:bg-primary/10 hover:text-foreground"
                }`}
            >
                Standard OCR (Image Only)
            </button>
            <button 
                onClick={() => { setVerificationMode("digital"); setResult(null); setError(null); }}
                disabled={isProcessing}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                    verificationMode === "digital" 
                    ? "bg-primary text-white shadow-sm" 
                    : "text-muted-foreground hover:bg-primary/10 hover:text-foreground"
                }`}
            >
                Digital Verification (Image + Chat)
            </button>
        </div>

        {/* Method Information Panel */}
        <div className="mb-6 p-4 bg-card border border-border rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Info className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-foreground">
                  {verificationMode === "standard" ? "Standard Payment Verification" : "Digital Cross-Verification"}
              </h3>
            </div>
            <button
              onClick={() => setShowMethodInfo(!showMethodInfo)}
              className="p-1 hover:bg-primary/10 rounded transition-colors"
            >
              <Settings className={`w-4 h-4 text-muted-foreground transition-transform ${showMethodInfo ? 'rotate-90' : ''}`} />
            </button>
          </div>
          {showMethodInfo && (
            <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                {verificationMode === "standard" ? (
                    <>
                        <p>• Uses OCR (Optical Character Recognition) to extract text from receipts</p>
                        <p>• Supports multiple preprocessing methods to enhance image quality</p>
                        <p>• Validates transaction ID, amount, and date against regex patterns</p>
                    </>
                ) : (
                    <>
                        <p>• Cross-references Receipt OCR data with Chat Logs</p>
                        <p>• Verifies email confirmation from Gmail (if configured)</p>
                        <p>• Detects fraud/tampering using edge detection</p>
                        <p>• Calculates a confidence score based on matching data points</p>
                    </>
                )}
            </div>
          )}
        </div>

        {/* Preprocessing Method Selection (Standard Mode Only) */}
        {verificationMode === "standard" && image && (
          <div className="mb-6 p-4 bg-secondary/30 border border-border rounded-lg">
            <label className="block text-sm font-semibold text-foreground mb-2">
              Preprocessing Method
            </label>
            <select
              value={preprocessingMethod || ""}
              onChange={(e) => {
                const value = e.target.value || null;
                setPreprocessingMethod(value);
                if (imageFile && !isProcessing) {
                  processReceipt(imageFile, null);
                }
              }}
              disabled={isProcessing}
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {PREPROCESSING_METHODS.map((method) => (
                <option key={method.value || "auto"} value={method.value || ""}>
                  {method.label} - {method.description}
                </option>
              ))}
            </select>
          </div>
        )}

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

              {/* CHAT FILE UPLOAD (Only for Digital Mode) */}
              {verificationMode === "digital" && (
                  <div className="mt-4 p-4 border border-dashed border-primary/40 rounded-lg bg-primary/5">
                      <div className="flex items-center gap-4">
                          <FileText className="w-8 h-8 text-primary flex-shrink-0"/>
                          <div className="flex-1 overflow-hidden">
                              <p className="text-sm font-semibold">Upload Chat Log</p>
                              <p className="text-xs text-muted-foreground truncate">
                                  {chatFile ? chatFile.name : "Required for digital verification (.txt)"}
                              </p>
                          </div>
                          <label className="px-4 py-2 text-sm bg-secondary hover:bg-secondary/80 rounded-lg cursor-pointer whitespace-nowrap">
                              {chatFile ? "Change" : "Browse"}
                              <input type="file" accept=".txt" className="hidden" onChange={handleChatUpload} />
                          </label>
                      </div>
                  </div>
              )}

              <div className="mt-4 flex gap-2">
                <button
                  onClick={() => {
                    setImage(null);
                    setImageFile(null);
                    setChatFile(null);
                    setResult(null);
                    setError(null);
                    onResultChange?.(null);
                  }}
                  disabled={isProcessing}
                  className="flex-1 px-4 py-2 text-sm text-primary hover:bg-primary/10 rounded-lg border border-primary/30 transition-colors"
                >
                  Upload Different Receipt
                </button>
                
                {/* DYNAMIC VERIFY/RETRY BUTTON */}
                {(verificationMode === "digital" || result) && (
                    <button
                        onClick={result ? handleRetry : handleVerifyClick}
                        disabled={isProcessing || (verificationMode === "digital" && !chatFile)}
                        className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 min-w-[100px]"
                    >
                        {isProcessing ? "Verifying..." : result ? "Retry" : "Verify Receipt"}
                    </button>
                )}
              </div>
            </div>

            {error && (
              <div className="mb-6 p-4 bg-red-500/10 border-2 border-red-500/30 rounded-lg">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                  <div>
                    <p className="font-semibold text-foreground">
                      Verification Failed
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-primary animate-spin mr-2" />
                <span className="text-foreground font-medium">
                  Verifying receipt...
                </span>
              </div>
            )}

            {result && !isProcessing && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div
                  className={`flex items-center gap-3 p-4 rounded-lg border-2 ${
                    result.status === "success"
                      ? "bg-green-500/10 border-green-500/50"
                      : result.status === "partial"
                      ? "bg-yellow-500/10 border-yellow-500/50"
                      : "bg-red-500/10 border-red-500/50"
                  }`}
                >
                  {result.status === "success" ? (
                    <CheckCircle2 className="w-6 h-6 text-green-500 flex-shrink-0" />
                  ) : (
                    <AlertCircle className={`w-6 h-6 flex-shrink-0 ${result.status === "partial" ? "text-yellow-500" : "text-red-500"}`} />
                  )}
                  <div>
                    <p className="font-semibold text-foreground">
                      Receipt{" "}
                      {result.status === "success" ? "Verified" : 
                       result.status === "partial" ? "Partially Verified" : "Failed"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Confidence: {Math.round(result.confidence)}%
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(result.data || {}).map(([label, value], idx) => (
                    <div key={idx} className="p-4 bg-secondary/30 rounded-lg">
                      <p className="text-xs text-muted-foreground font-medium mb-1">
                        {label}
                      </p>
                      <div className="flex items-center justify-between">
                        <p className="font-semibold text-foreground text-sm break-words">
                          {String(value)}
                        </p>
                        <button
                          onClick={() => copyToClipboard(String(value))}
                          className="p-1 hover:bg-primary/20 rounded transition-colors flex-shrink-0 ml-2"
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

                {/* Show LLM Corrections if available */}
                {result.apiResult?.llm_corrections && (
                  <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
                    <p className="text-sm font-semibold text-foreground mb-2">
                      LLM Corrections Applied
                    </p>
                    <div className="space-y-1 text-xs text-muted-foreground">
                      {Object.entries(result.apiResult.llm_corrections).map(([field, correction]) => (
                        <div key={field} className="flex justify-between">
                          <span className="font-medium">{field}:</span>
                          <span>{correction}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Footer details */}
                <div className="space-y-2 p-4 bg-secondary/20 rounded-lg">
                  <p className="text-sm font-semibold text-foreground mb-3">
                    Verification Details
                  </p>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0" />
                      <span className="text-foreground">
                        Source: {verificationMode === "digital" ? "Digital (Flask)" : "Standard (FastAPI)"}
                      </span>
                    </div>
                    {result.processingTime && (
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-primary flex-shrink-0" />
                        <span className="text-foreground">
                          Processing Time: {result.processingTime}ms
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}