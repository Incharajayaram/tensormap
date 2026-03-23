import PropTypes from "prop-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getLayerByType, validateLayerParams } from "../../registry/layers";

function NodePropertiesPanel({
  selectedNode,
  modelName,
  onModelNameChange,
  onSave,
  canSave,
  onNodeUpdate,
}) {
  if (!selectedNode) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Save Model</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Model Name</Label>
            <Input
              placeholder="Enter model name"
              value={modelName}
              onChange={(e) => onModelNameChange(e.target.value)}
            />
          </div>
          <Button className="w-full" onClick={onSave} disabled={!canSave}>
            Validate &amp; Save
          </Button>
        </CardContent>
      </Card>
    );
  }

  const { type, data, id } = selectedNode;
  const params = data.params;
  const layer = getLayerByType(type);
  const paramEntries = Object.entries(layer?.params || {});
  const errors = validateLayerParams(layer, params);

  const updateParam = (name, value) => {
    onNodeUpdate(id, { ...params, [name]: value });
  };

  if (!layer) {
    return (
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-sm">Layer Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Unknown layer type</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="text-sm">{layer.label} Layer</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {paramEntries.length === 0 ? (
          <p className="text-sm text-muted-foreground">No configurable parameters</p>
        ) : (
          paramEntries.map(([key, config]) => {
            const value = params[key];
            const error = errors[key];
            const isNumber = config.type === "number";
            const label = `${config.label}${config.required ? " *" : ""}`;

            if (config.type === "select") {
              return (
                <div key={key} className="space-y-1">
                  <Label>{label}</Label>
                  <Select value={value} onValueChange={(v) => updateParam(key, v)}>
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${config.label.toLowerCase()}`} />
                    </SelectTrigger>
                    <SelectContent>
                      {(config.options || []).map((option) => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {error ? <p className="text-xs text-destructive">{error}</p> : null}
                </div>
              );
            }

            if (config.type === "boolean") {
              const boolValue = value === true ? "true" : "false";
              return (
                <div key={key} className="space-y-1">
                  <Label>{label}</Label>
                  <Select
                    value={boolValue}
                    onValueChange={(v) => updateParam(key, v === "true")}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${config.label.toLowerCase()}`} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="true">True</SelectItem>
                      <SelectItem value="false">False</SelectItem>
                    </SelectContent>
                  </Select>
                  {error ? <p className="text-xs text-destructive">{error}</p> : null}
                </div>
              );
            }

            return (
              <div key={key} className="space-y-1">
                <Label>{label}</Label>
                <Input
                  type={isNumber ? "number" : "text"}
                  min={isNumber && config.min !== undefined ? config.min : undefined}
                  max={isNumber && config.max !== undefined ? config.max : undefined}
                  step={isNumber && config.step !== undefined ? config.step : undefined}
                  placeholder={config.placeholder || ""}
                  value={value}
                  onChange={(e) => {
                    const nextValue =
                      isNumber && e.target.value !== "" ? Number(e.target.value) : e.target.value;
                    updateParam(key, nextValue);
                  }}
                />
                {error ? <p className="text-xs text-destructive">{error}</p> : null}
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

NodePropertiesPanel.propTypes = {
  selectedNode: PropTypes.object,
  modelName: PropTypes.string.isRequired,
  onModelNameChange: PropTypes.func.isRequired,
  onSave: PropTypes.func.isRequired,
  canSave: PropTypes.bool.isRequired,
  onNodeUpdate: PropTypes.func.isRequired,
};

export default NodePropertiesPanel;
