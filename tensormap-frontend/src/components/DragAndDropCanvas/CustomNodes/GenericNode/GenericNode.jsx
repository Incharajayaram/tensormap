import PropTypes from "prop-types";
import { Handle, Position } from "reactflow";
import { getLayerByType } from "@/registry/layers";

const formatValue = (value) => {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return value;
};

function GenericNode({ data, id }) {
  const layer = getLayerByType(data.type);
  const label = layer?.label || "Layer";
  const headerClass = layer?.headerClass || "bg-slate-700";

  const params = data.params || {};
  const paramSummary = layer?.params
    ? Object.entries(layer.params)
        .map(([key, config]) => {
          const value = params[key];
          if (value === undefined || value === null || value === "") return null;
          return `${config.label}: ${formatValue(value)}`;
        })
        .filter(Boolean)
        .slice(0, 3)
        .join(", ")
    : "";

  return (
    <div className="w-44 rounded-lg border bg-white shadow-sm">
      <Handle type="target" position={Position.Left} isConnectable id={`${id}_in`} />
      <div className={`rounded-t-lg ${headerClass} px-3 py-1.5 text-xs font-bold text-white`}>
        {label}
      </div>
      <div className="px-3 py-2 text-xs text-muted-foreground">
        {paramSummary || "Not configured"}
      </div>
      <Handle type="source" position={Position.Right} isConnectable id={`${id}_out`} />
    </div>
  );
}

GenericNode.propTypes = {
  data: PropTypes.shape({
    type: PropTypes.string,
    params: PropTypes.object,
  }).isRequired,
  id: PropTypes.string.isRequired,
};

export default GenericNode;
