import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const statusStyles = {
  running: "bg-blue-50 text-blue-700 border-blue-100",
  completed: "bg-emerald-50 text-emerald-700 border-emerald-100",
  failed: "bg-red-50 text-red-700 border-red-100",
  queued: "bg-slate-50 text-slate-700 border-slate-100",
};

const formatTime = (timestamp) => {
  if (!timestamp) return "Unknown";
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
};

export default function RunHistoryPanel({
  runs,
  runOrder,
  selectedRunId,
  onSelect,
  comparedRunIds,
  onToggleCompare,
}) {
  if (runOrder.length === 0) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Run History</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">No runs yet</CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="text-sm">Run History</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {runOrder.map((runId) => {
          const run = runs[runId];
          if (!run) return null;
          const statusClass = statusStyles[run.status] || statusStyles.queued;
          const isSelected = selectedRunId === runId;
          const isCompared = comparedRunIds.includes(runId);
          return (
            <button
              key={runId}
              type="button"
              className={`w-full rounded-md border px-3 py-2 text-left text-xs transition ${
                isSelected ? "border-primary/50 bg-primary/5" : "border-muted/60 bg-white"
              }`}
              onClick={() => onSelect(runId)}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="font-semibold">{runId}</div>
                <span className={`rounded-full border px-2 py-0.5 text-[10px] ${statusClass}`}>
                  {run.status}
                </span>
              </div>
              <div className="mt-1 text-[11px] text-muted-foreground">
                {formatTime(run.updatedAt || run.startedAt)}
              </div>
              <label className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={isCompared}
                  onChange={(e) => {
                    e.stopPropagation();
                    onToggleCompare(runId);
                  }}
                />
                Compare
              </label>
            </button>
          );
        })}
      </CardContent>
    </Card>
  );
}

RunHistoryPanel.propTypes = {
  runs: PropTypes.object.isRequired,
  runOrder: PropTypes.arrayOf(PropTypes.string).isRequired,
  selectedRunId: PropTypes.string,
  onSelect: PropTypes.func.isRequired,
  comparedRunIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  onToggleCompare: PropTypes.func.isRequired,
};
