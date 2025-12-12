"use client";

import { useState } from "react";
import { Upload, Loader2, Zap, Check, Info, AlertTriangle, FileText } from "lucide-react";
import { verifyPayment } from "../services/api"; // Assuming api.js is in ../services/

export default function PaymentVerification({ onResultChange, onProcessingChange }) {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [chatFile, setChatFile] = useState(null);
  const [chatFileName, setChatFileName] = useState(null);
  
  // Preprocessing is set to 'aggressive' by default, as requested
  const preprocessingMethod = 'Digital'; 
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const updateProcessingState = (processing) => {
    setIsProcessing(processing);
    onProcessingChange?.(processing);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    
    // Clear previous error
    setError(null);

    if (file) {
      const fileName = file.name.toLowerCase();
      const allowedExtensions = ['.jpg', '.jpeg', '.png'];
      const isValid = allowedExtensions.some(ext => fileName.endsWith(ext));

      if (!isValid) {
        setImage(null);
        setImageFile(null);
        setError("Invalid image format. Please upload only JPG, JPEG, or PNG.");
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setImageFile(file);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleChatUpload = (e) => {
    const file = e.target.files?.[0];
    setChatFile(null);
    setChatFileName(null);
    setError(null);

    if (file) {
      const fileName = file.name.toLowerCase();
      // --- CHAT FILE VALIDATION (.txt only) ---
      if (!fileName.endsWith('.txt')) {
        setError("Invalid file format. Please upload a .txt file.");
      } else {
        setChatFile(file);
        setChatFileName(file.name);
      }
    }
  };

  const handleVerification = async () => {
    if (!imageFile || !chatFile) {
      setError("Please upload both the receipt image and the chat (.txt) file.");
      return;
    }
    
    updateProcessingState(true);
    setError(null);
    onResultChange?.(null);

    try {
      // Create a new File object explicitly named 'chat1.txt' for the backend
      const chatFileToSend = new File([chatFile], "chat1.txt", { type: chatFile.type });

      const startTime = Date.now();
      
      // Perform the single API call
      const apiResult = await verifyPayment(imageFile, chatFileToSend, preprocessingMethod);
      const processingTime = Date.now() - startTime;
      
      console.log("API Response:", apiResult);

      const status = apiResult.status === "APPROVED" ? "success" : 
                     apiResult.status === "MANUAL REVIEW" ? "partial" : 
                     "review_required";

      const verificationResult = {
        type: "payment",
        status: status,
        confidence: apiResult.confidence,
        data: {
          extracted_amount: apiResult.extracted_from_receipt?.amount ?? null,
          chat_amount: apiResult.chat_amount ?? null,
          transaction_id: apiResult.extracted_from_receipt?.transaction_id ?? null,
          email_tx_id: apiResult.email_data?.transaction_id ?? null,
          timestamp: apiResult.extracted_from_receipt?.timestamp ?? null,
          receiver_account: apiResult["receiver account"] ?? null,
          status: apiResult.status,
        },
        apiResult: {
          ...apiResult, // Pass all API data for Method Details panel
          preprocessing_method: preprocessingMethod, // Send the hardcoded value
        },
        processingTime,
      };

      onResultChange?.(verificationResult);

    } catch (err) {
      console.error("Verification Error:", err);
      setError(err.message || "Failed to verify payment. Check server connection and CORS.");
      onResultChange?.({ type: "payment", status: "error", error: err.message });
    } finally {
      updateProcessingState(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setImageFile(null);
    setChatFile(null);
    setChatFileName(null);
    setError(null);
    onResultChange?.(null);
    updateProcessingState(false);
  }

  const isFormValid = imageFile && chatFile && !isProcessing;

  return (
    <div className="p-8 relative">
      <div className="max-w-2xl mx-auto">
        
        {/* Info Panel */}
        <div className="mb-6 p-4 bg-primary/5 border border-primary/20 rounded-lg">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-primary mt-0.5" />
            <div className="text-sm text-primary-dark">
              {/* <h3 className="font-semibold text-foreground">API Configuration Note</h3>
              <p className="text-muted-foreground">
                Ensure your Flask backend is running on **http://localhost:5000** and that CORS is enabled. The OCR extraction method is set to **Aggressive OCR** by default.
              </p> */}
              <h3 className="font-semibold text-foreground">OCR verification notes</h3>
              <p className="text-muted-foreground">
                Upload the complete payment receipt image. The system will extract payment details from both sources for verification.
              </p>
            </div>
          </div>
        </div>

        {/* Form Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Column 1: Image Upload */}
          <div>
            <h2 className="text-lg font-semibold mb-3">1. Upload Receipt Image</h2>
            {!image ? (
              <label className="block">
                <div className="border-2 border-dashed border-primary/30 hover:border-primary/60 rounded-lg p-12 text-center cursor-pointer bg-primary/5 h-64 flex flex-col items-center justify-center">
                  <Upload className="w-12 h-12 text-primary mx-auto mb-4" />
                  <p className="text-lg font-semibold text-foreground mb-1">Upload Receipt</p>
                  <p className="text-sm text-muted-foreground">JPG, JPEG, or PNG only</p>
                  {/* --- IMAGE INPUT ACCEPT FILTER --- */}
                  <input type="file" accept=".jpg,.jpeg,.png" onChange={handleImageUpload} className="hidden" />
                </div>
              </label>
            ) : (
              <div className="rounded-lg border border-border overflow-hidden relative">
                <img src={image} alt="Receipt Preview" className="w-full h-64 object-contain bg-secondary/20" />
                <button 
                  type="button"
                  onClick={resetForm}
                  className="absolute top-2 right-2 bg-background/80 text-foreground p-2 rounded-full hover:bg-background transition-colors"
                  title="Remove Image"
                >
                  <Zap className="w-4 h-4 text-primary" />
                </button>
              </div>
            )}
          </div>

          {/* Column 2: Chat Upload & Controls */}
          <div className="space-y-6">
            <div className="space-y-3">
              <h2 className="text-lg font-semibold">2. Upload Customer Chat (.txt)</h2>
              <label className="block">
                <div className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer h-[160px] flex flex-col items-center justify-center 
                  ${chatFile ? 'border-primary/60 bg-primary/5' : 'border-border hover:border-primary/30'}`}>
                  <FileText className="w-8 h-8 text-primary mx-auto mb-2" />
                  <p className="text-lg font-semibold text-foreground mb-1">
                    {chatFileName || "Click to select chat file"}
                  </p>
                  <p className="text-sm text-muted-foreground">Accepts only .txt file</p>
                  {/* --- CHAT INPUT ACCEPT FILTER --- */}
                  <input 
                    type="file" 
                    accept=".txt" 
                    onChange={handleChatUpload} 
                    className="hidden" 
                  />
                </div>
              </label>
              {chatFile && (
                <p className="text-sm text-primary/80 mt-2">
                  Ready to use.
                </p>
              )}
            </div>

            {/* Default Extraction Method Info
            <div className="space-y-2 pt-2">
              <h2 className="text-lg font-semibold">3. Extraction Method</h2>
              <p className="text-sm text-muted-foreground">
                Preprocessing is set to **Aggressive OCR** (default).
              </p>
            </div> */}

            {/* Action Button & Loader */}
            <div className="pt-2">
              {error && (
                <div className="flex items-center gap-2 p-3 mb-4 text-sm text-red-600 bg-red-100 border border-red-300 rounded-lg">
                  <AlertTriangle className="w-4 h-4" />
                  {error}
                </div>
              )}

              <button
                onClick={handleVerification}
                disabled={!isFormValid}
                className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-3 rounded-lg font-bold text-lg hover:opacity-90 transition-opacity disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing Receipt...
                  </>
                ) : (
                  <>
                    <Check className="w-6 h-6" />
                    Run Verification
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}