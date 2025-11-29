"use client";

import { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import PaymentVerification from "./components/PaymentVerfication";
import ProductSearch from "./components/ProductSearch";
import Navigation from "./components/Navigation";
import ResultsPanel from "./components/ResultsPanel";

function PaymentPage({ onResultChange, onProcessingChange }) {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-2">
          Payment Verification
        </h1>
        <p className="text-base text-muted-foreground max-w-2xl">
          Verify payment receipts using OCR and detect fraudulent documents
        </p>
      </div>
      <div className="bg-card rounded-xl border border-border shadow-lg overflow-hidden">
        <PaymentVerification
          onResultChange={onResultChange}
          onProcessingChange={onProcessingChange}
        />
      </div>
    </div>
  );
}

function SearchPage({ onResultChange, onProcessingChange }) {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-2">
          Visual Product Search
        </h1>
        <p className="text-base text-muted-foreground max-w-2xl">
          Match customer images to inventory using advanced feature recognition
        </p>
      </div>
      <div className="bg-card rounded-xl border border-border shadow-lg overflow-hidden">
        <ProductSearch
          onResultChange={onResultChange}
          onProcessingChange={onProcessingChange}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [processedResult, setProcessedResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      <Navigation />

      <main className="flex h-screen overflow-hidden">
        {/* Main Content - Left Side (2/3) */}
        <div className="flex-[2] overflow-y-auto border-r border-border">
          <Routes>
            <Route
              path="/payment"
              element={
                <PaymentPage
                  onResultChange={setProcessedResult}
                  onProcessingChange={setIsProcessing}
                />
              }
            />
            <Route
              path="/search"
              element={
                <SearchPage
                  onResultChange={setProcessedResult}
                  onProcessingChange={setIsProcessing}
                />
              }
            />
            <Route path="/" element={<Navigate to="/payment" replace />} />
          </Routes>
        </div>

        {/* Results Panel - Right Side (1/3) */}
        <div className="flex-1 overflow-y-auto bg-card border-l border-border">
          <ResultsPanel result={processedResult} isProcessing={isProcessing} />
        </div>
      </main>
    </div>
  );
}
