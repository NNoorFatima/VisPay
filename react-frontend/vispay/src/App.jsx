"use client";

import { useState } from "react";
import { ImageIcon, Zap, Shield, Search } from "lucide-react";
import PaymentVerification from "./components/PaymentVerfication";
import ProductSearch from "./components/ProductSearch";
import Navigation from "./components/Navigation";
import ResultsPanel from "./components/ResultsPanel";

export default function App() {
  const [activeTab, setActiveTab] = useState("verify");
  const [processedResult, setProcessedResult] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      <Navigation />

      <main className="flex h-screen overflow-hidden">
        {/* Main Content - Left Side (2/3) */}
        <div className="flex-[2] overflow-y-auto border-r border-border">
          <div className="p-8">
            {/* Hero Section */}
            <div className="mb-8">
              <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-2 text-balance">
                VisPay Vision
              </h1>
              <p className="text-base text-muted-foreground max-w-2xl text-balance">
                Intelligent payment verification and visual product search
                powered by advanced image intelligence
              </p>
            </div>

            {/* Feature Pills */}
            <div className="grid grid-cols-2 gap-3 mb-6">
              <div
                className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                  activeTab === "verify"
                    ? "bg-primary/10 border-primary"
                    : "bg-card border-border hover:border-primary/50"
                }`}
                onClick={() => setActiveTab("verify")}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Shield className="w-5 h-5 text-primary" />
                  <h2 className="font-semibold text-foreground text-sm">
                    Payment Verification
                  </h2>
                </div>
                <p className="text-xs text-muted-foreground">
                  Verify receipts using OCR and detect fraudulent documents
                </p>
              </div>

              <div
                className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                  activeTab === "search"
                    ? "bg-primary/10 border-primary"
                    : "bg-card border-border hover:border-primary/50"
                }`}
                onClick={() => setActiveTab("search")}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Search className="w-5 h-5 text-primary" />
                  <h2 className="font-semibold text-foreground text-sm">
                    Visual Product Search
                  </h2>
                </div>
                <p className="text-xs text-muted-foreground">
                  Match images to inventory using feature recognition
                </p>
              </div>
            </div>

            {/* Tab Content */}
            <div className="bg-card rounded-xl border border-border shadow-lg overflow-hidden">
              {activeTab === "verify" && (
                <PaymentVerification onResultChange={setProcessedResult} />
              )}
              {activeTab === "search" && (
                <ProductSearch onResultChange={setProcessedResult} />
              )}
            </div>

            {/* Benefits Section */}
            <div className="grid grid-cols-3 gap-3 mt-6">
              <div className="p-3 rounded-lg bg-card border border-border">
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center mb-2">
                  <Zap className="w-4 h-4 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-1 text-xs">
                  Fast Processing
                </h3>
                <p className="text-xs text-muted-foreground">
                  Real-time analysis within seconds
                </p>
              </div>

              <div className="p-3 rounded-lg bg-card border border-border">
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center mb-2">
                  <Shield className="w-4 h-4 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-1 text-xs">
                  Enhanced Security
                </h3>
                <p className="text-xs text-muted-foreground">
                  Detect manipulated images
                </p>
              </div>

              <div className="p-3 rounded-lg bg-card border border-border">
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center mb-2">
                  <ImageIcon className="w-4 h-4 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground mb-1 text-xs">
                  Visual Intelligence
                </h3>
                <p className="text-xs text-muted-foreground">
                  Advanced feature matching
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Results Panel - Right Side (1/3) */}
        <div className="flex-1 overflow-y-auto bg-card border-l border-border">
          <ResultsPanel result={processedResult} />
        </div>
      </main>
    </div>
  );
}
