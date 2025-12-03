import { CheckCircle, AlertCircle, Loader, Image as ImageIcon, Info } from "lucide-react";

export default function ResultsPanel({ result, isProcessing }) {
  if (!result && !isProcessing) {
    return (
      <div className="p-6 h-full overflow-y-auto">
        <div className="bg-card rounded-xl border border-border p-6 sticky top-4">
          <h3 className="font-semibold text-foreground mb-4">
            Processing Result
          </h3>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <p className="text-sm text-muted-foreground">
              Upload an image to see results here
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (isProcessing && !result) {
    return (
      <div className="p-6 h-full overflow-y-auto">
        <div className="bg-card rounded-xl border border-border p-6 sticky top-4">
          <h3 className="font-semibold text-foreground mb-4">
            Processing Result
          </h3>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-3">
              <Loader className="w-6 h-6 text-primary animate-spin" />
            </div>
            <p className="text-sm text-foreground font-medium">
              Processing...
            </p>
          </div>
        </div>
      </div>
    );
  }

  const isSuccess = result.status === "success" || result.status === "partial";
  const isPayment = result.type === "payment";
  const isProduct = result.type === "product";

  return (
    <div className="p-6 h-full overflow-y-auto">
      <div className="bg-card rounded-xl border border-border p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-2 mb-4">
          {isSuccess ? (
            <CheckCircle className="w-5 h-5 text-primary" />
          ) : (
            <AlertCircle className="w-5 h-5 text-primary" />
          )}
          <h3 className="font-semibold text-foreground">
            {isPayment 
              ? (isSuccess ? "Verification Complete" : "Verification Failed")
              : (isSuccess ? "Search Complete" : "Search Failed")}
          </h3>
        </div>

        {/* Processed Images Section */}
        {(result.processedImageUrl || result.queryImageUrl) && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-primary" />
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Processed Images
              </p>
            </div>
            <div className="space-y-3">
              {result.processedImageUrl && (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">Preprocessed Image</p>
                  <div className="relative rounded-lg border border-border overflow-hidden bg-muted/20">
                    <img
                      src={result.processedImageUrl}
                      alt="Processed"
                      className="w-full h-auto object-contain"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    <div className="hidden items-center justify-center p-8 text-muted-foreground text-sm">
                      Image not available
                    </div>
                  </div>
                </div>
              )}
              {result.queryImageUrl && (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">Query Image</p>
                  <div className="relative rounded-lg border border-border overflow-hidden bg-muted/20">
                    <img
                      src={result.queryImageUrl}
                      alt="Query"
                      className="w-full h-auto object-contain"
                      onError={(e) => {
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    <div className="hidden items-center justify-center p-8 text-muted-foreground text-sm">
                      Image not available
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Status */}
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            Status
          </p>
          <p
            className={`text-sm font-semibold ${
              isSuccess ? "text-primary" : "text-primary"
            }`}
          >
            {result.status === "success" 
              ? (isPayment ? "Verified" : "Found Matches")
              : result.status === "partial"
              ? "Partially Verified"
              : result.status === "no_results"
              ? "No Matches"
              : "Review Required"}
          </p>
        </div>

        {/* Confidence Score */}
        {result.confidence !== undefined && (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Confidence
            </p>
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all bg-primary"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {result.confidence}% {isPayment ? "confidence" : "match"}
            </p>
          </div>
        )}

        {/* Method Details */}
        {result.apiResult && (
          <div className="bg-muted/30 rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 text-primary" />
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Method Details
              </p>
            </div>
            
            {isPayment && result.apiResult.extraction_method && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Extraction Method</p>
                <p className="text-sm font-medium text-foreground">
                  {result.apiResult.extraction_method}
                </p>
              </div>
            )}

            {isPayment && result.apiResult.preprocessing_method && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Preprocessing</p>
                <p className="text-sm font-medium text-foreground">
                  {result.apiResult.preprocessing_method}
                  {result.apiResult.auto_selected && (
                    <span className="ml-2 text-xs text-muted-foreground">(Auto-selected)</span>
                  )}
                </p>
              </div>
            )}

            {isProduct && result.apiResult.matches?.[0]?.feature_method && (
              <div>
                <p className="text-xs text-muted-foreground mb-1">Feature Method</p>
                <p className="text-sm font-medium text-foreground">
                  {result.apiResult.matches[0].feature_method}
                </p>
              </div>
            )}

            {isPayment && result.apiResult.image_quality && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs text-muted-foreground mb-2">Image Quality Metrics</p>
                <div className="space-y-1 text-xs">
                  {result.apiResult.image_quality.overall_score !== null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Overall:</span>
                      <span className="text-foreground font-medium">
                        {result.apiResult.image_quality.overall_score.toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {result.apiResult.image_quality.contrast !== null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Contrast:</span>
                      <span className="text-foreground font-medium">
                        {result.apiResult.image_quality.contrast.toFixed(1)}%
                      </span>
                    </div>
                  )}
                  {result.apiResult.image_quality.blur_level !== null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Blur:</span>
                      <span className="text-foreground font-medium">
                        {result.apiResult.image_quality.blur_level.toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {isPayment && result.apiResult.llm_explanations && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs font-medium text-muted-foreground mb-2">LLM Explanations</p>
                <div className="space-y-2 text-xs">
                  {Object.entries(result.apiResult.llm_explanations).map(([field, explanation]) => (
                    <div key={field}>
                      <p className="font-medium text-foreground">{field}:</p>
                      <p className="text-muted-foreground">{explanation}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {isPayment && result.apiResult.authenticity && (
              <div className="mt-3 pt-3 border-t border-border">
                <p className="text-xs font-medium text-muted-foreground mb-3">Image Authenticity Analysis</p>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-muted-foreground">Authenticity Score</span>
                      <span className={`text-sm font-bold ${
                        result.apiResult.authenticity.authenticity_score > 0.8 ? 'text-green-600' :
                        result.apiResult.authenticity.authenticity_score > 0.65 ? 'text-blue-600' :
                        result.apiResult.authenticity.authenticity_score > 0.45 ? 'text-yellow-600' :
                        'text-red-600'
                      }`}>
                        {(result.apiResult.authenticity.authenticity_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          result.apiResult.authenticity.authenticity_score > 0.8 ? 'bg-green-600' :
                          result.apiResult.authenticity.authenticity_score > 0.65 ? 'bg-blue-600' :
                          result.apiResult.authenticity.authenticity_score > 0.45 ? 'bg-yellow-600' :
                          'bg-red-600'
                        }`}
                        style={{ width: `${result.apiResult.authenticity.authenticity_score * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
                    <p className={`text-sm font-semibold ${
                      result.apiResult.authenticity.recommendation === 'AUTHENTIC' ? 'text-green-600' :
                      result.apiResult.authenticity.recommendation === 'LIKELY_AUTHENTIC' ? 'text-blue-600' :
                      result.apiResult.authenticity.recommendation === 'SUSPICIOUS' ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {result.apiResult.authenticity.recommendation.replace(/_/g, ' ')}
                    </p>
                  </div>

                  {result.apiResult.authenticity.indicators && result.apiResult.authenticity.indicators.length > 0 && (
                    <div>
                      <p className="text-xs text-muted-foreground mb-2">Forensic Checks</p>
                      <div className="space-y-2">
                        {result.apiResult.authenticity.indicators.map((indicator) => (
                          <div key={indicator.check} className="flex items-start gap-2 p-2 bg-muted/20 rounded text-xs">
                            <div className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${
                              indicator.score > 0.7 ? 'bg-green-600' :
                              indicator.score > 0.5 ? 'bg-yellow-600' :
                              'bg-red-600'
                            }`} />
                            <div className="flex-1 min-w-0">
                              <p className="text-foreground font-medium capitalize">
                                {indicator.check.replace(/_/g, ' ')}
                              </p>
                              <p className="text-muted-foreground text-xs">{indicator.description}</p>
                              <p className="text-foreground font-semibold text-xs mt-1">
                                Score: {(indicator.score * 100).toFixed(1)}% (Weight: {(indicator.weight * 100).toFixed(0)}%)
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Data */}
        {result.data && (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Details
            </p>
            <div className="bg-muted/50 rounded-lg p-3 space-y-2">
              {Object.entries(result.data).map(([key, value]) => (
                <div
                  key={key}
                  className="flex justify-between items-start gap-2"
                >
                  <span className="text-xs text-muted-foreground capitalize">
                    {key}:
                  </span>
                  <span className="text-xs font-medium text-foreground text-right break-words max-w-[180px]">
                    {typeof value === "object"
                      ? JSON.stringify(value)
                      : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Time */}
        {result.processingTime && (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
              Processing Time
            </p>
            <p className="text-sm text-foreground font-medium">
              {result.processingTime}ms ({(result.processingTime / 1000).toFixed(2)}s)
            </p>
          </div>
        )}

        {/* Error Message */}
        {result.error && (
          <div className="bg-primary/10 border border-primary/30 rounded-lg p-3">
            <p className="text-xs font-medium text-foreground mb-1">
              Error
            </p>
            <p className="text-xs text-muted-foreground">
              {result.error}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
