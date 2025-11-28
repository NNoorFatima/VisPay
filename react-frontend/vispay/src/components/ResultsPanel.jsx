import { CheckCircle, AlertCircle, Loader } from "lucide-react";

export default function ResultsPanel({ result }) {
  if (!result) {
    return (
      <div className="bg-card rounded-xl border border-border p-6 h-full sticky top-4">
        <h3 className="font-semibold text-foreground mb-4">
          Processing Result
        </h3>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mb-3">
            <Loader className="w-6 h-6 text-muted-foreground animate-spin" />
          </div>
          <p className="text-sm text-muted-foreground">
            Upload an image to see results here
          </p>
        </div>
      </div>
    );
  }

  const isSuccess = result.status === "success";

  return (
    <div className="bg-card rounded-xl border border-border p-6 h-full sticky top-4">
      <div className="flex items-center gap-2 mb-4">
        {isSuccess ? (
          <CheckCircle className="w-5 h-5 text-emerald-500" />
        ) : (
          <AlertCircle className="w-5 h-5 text-amber-500" />
        )}
        <h3 className="font-semibold text-foreground">
          {isSuccess ? "Verification Complete" : "Processing Alert"}
        </h3>
      </div>

      <div className="space-y-4">
        {/* Status */}
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            Status
          </p>
          <p
            className={`text-sm font-semibold ${
              isSuccess ? "text-emerald-600" : "text-amber-600"
            }`}
          >
            {result.status === "success" ? "Verified" : "Review Required"}
          </p>
        </div>

        {/* Confidence Score */}
        {result.confidence && (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Confidence
            </p>
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  result.confidence >= 80
                    ? "bg-emerald-500"
                    : result.confidence >= 60
                    ? "bg-amber-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${result.confidence}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {result.confidence}% match
            </p>
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
                  <span className="text-xs font-medium text-foreground text-right break-words max-w-[120px]">
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
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Processing Time
            </p>
            <p className="text-xs text-muted-foreground">
              {result.processingTime}ms
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
