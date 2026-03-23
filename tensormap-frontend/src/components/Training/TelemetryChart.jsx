import PropTypes from "prop-types";

const buildPoints = (data, width, height, padding) => {
  if (!data || data.length === 0) return "";
  const xs = data.map((d, index) => d.x ?? index + 1);
  const ys = data.map((d) => d.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;

  return data
    .map((point, index) => {
      const rawX = point.x ?? index + 1;
      const rawY = point.y;
      const x = padding + ((rawX - minX) / xSpan) * (width - padding * 2);
      const y = height - padding - ((rawY - minY) / ySpan) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");
};

export default function TelemetryChart({ title, series, height = 180 }) {
  const width = 520;
  const padding = 18;
  const hasData = series.some((s) => s.data && s.data.length > 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold">{title}</h4>
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
          {series.map((s) => (
            <div key={s.label} className="flex items-center gap-1">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-md border bg-white px-3 py-3">
        {hasData ? (
          <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
            <line
              x1={padding}
              y1={height - padding}
              x2={width - padding}
              y2={height - padding}
              stroke="#e5e7eb"
            />
            <line
              x1={padding}
              y1={padding}
              x2={padding}
              y2={height - padding}
              stroke="#e5e7eb"
            />
            {series.map((s) => (
              <polyline
                key={s.label}
                points={buildPoints(s.data, width, height, padding)}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
              />
            ))}
          </svg>
        ) : (
          <div className="flex h-[140px] items-center justify-center text-xs text-muted-foreground">
            No telemetry yet
          </div>
        )}
      </div>
    </div>
  );
}

TelemetryChart.propTypes = {
  title: PropTypes.string.isRequired,
  series: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      color: PropTypes.string.isRequired,
      data: PropTypes.arrayOf(
        PropTypes.shape({
          x: PropTypes.number,
          y: PropTypes.number,
        }),
      ).isRequired,
    }),
  ).isRequired,
  height: PropTypes.number,
};
