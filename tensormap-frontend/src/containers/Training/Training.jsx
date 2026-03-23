import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import { Trash2 } from "lucide-react";
import io from "socket.io-client";
import { useRecoilState } from "recoil";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as urls from "../../constants/Urls";
import * as strings from "../../constants/Strings";
import logger from "../../shared/logger";
import FeedbackDialog from "../../components/shared/FeedbackDialog";
import Result from "../../components/ResultPanel/Result/Result";
import TelemetryChart from "../../components/Training/TelemetryChart";
import RunHistoryPanel from "../../components/Training/RunHistoryPanel";
import {
  download_code,
  runModel,
  getAllModels,
  updateTrainingConfig,
  deleteModel,
  getRunHistory,
  getRunMetrics,
  startExport,
  getExportStatus,
  downloadExport,
  generateInterpretability,
  getInterpretabilityReport,
  startTuning,
  getTuningStatus,
  getTuningResults,
  applyBestTuning,
} from "../../services/ModelServices";
import { getAllFiles } from "../../services/FileServices";
import { models as modelListAtom } from "../../shared/atoms";

const optimizerOptions = [
  { key: "opt_1", label: "Adam", value: "adam" },
  { key: "opt_2", label: "SGD", value: "sgd" },
  { key: "opt_3", label: "RMSprop", value: "rmsprop" },
  { key: "opt_4", label: "Adagrad", value: "adagrad" },
  { key: "opt_5", label: "AdamW", value: "adamw" },
];

const metricOptions = [
  { key: "acc_1", label: "Accuracy", value: "accuracy" },
  { key: "acc_2", label: "MSE", value: "mse" },
];

const problemTypeOptions = [
  { key: "prob_type_1", label: "Multi class classification", value: "1" },
  { key: "prob_type_2", label: "Linear Regression", value: "2" },
];

const colors = ["#2563eb", "#16a34a", "#f97316", "#9333ea", "#0891b2"];

export default function Training() {
  const { projectId } = useParams();
  const [modelList, setModelList] = useRecoilState(modelListAtom);

  const [selectedModel, setSelectedModel] = useState(null);
  const [resultValues, setResultValues] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState({ open: false, model: null });
  const [deleteFeedback, setDeleteFeedback] = useState({
    open: false,
    success: false,
    message: "",
  });
  const socketRef = useRef(null);
  const timeoutRef = useRef(null);
  const [runs, setRuns] = useState({});
  const [runOrder, setRunOrder] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [comparedRunIds, setComparedRunIds] = useState([]);
  const [exports, setExports] = useState([]);
  const [report, setReport] = useState(null);
  const [reportStatus, setReportStatus] = useState("idle");
  const [tuningConfig, setTuningConfig] = useState({
    strategy: "random",
    objective: "val_accuracy",
    max_trials: 6,
    optimizers: "adam,rmsprop,sgd",
    epochs_min: 5,
    epochs_max: 20,
    epochs_step: 5,
    batch_sizes: "16,32,64",
  });
  const [tuningJobId, setTuningJobId] = useState(null);
  const [tuningStatus, setTuningStatus] = useState(null);
  const [tuningResults, setTuningResults] = useState([]);

  // Training config state
  const [fileList, setFileList] = useState([]);
  const [fileDetails, setFileDetails] = useState([]);
  const [fieldsList, setFieldsList] = useState([]);
  const [configSaved, setConfigSaved] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    file_id: "",
    target_field: "",
    problem_type_id: "",
    optimizer: "adam",
    metric: "",
    epochs: "",
    batch_size: "",
    training_split: "",
  });
  // Validation state
  const [validationErrors, setValidationErrors] = useState({
    model: "",
    file_id: "",
    problem_type_id: "",
    optimizer: "",
    metric: "",
    target_field: "",
    epochs: "",
    batch_size: "",
    training_split: "",
  });

  const ensureRun = useCallback((runId) => {
    setRuns((prev) => {
      if (prev[runId]) return prev;
      return {
        ...prev,
        [runId]: {
          run_id: runId,
          status: "queued",
          startedAt: null,
          updatedAt: null,
          metrics: { train: [], val: [] },
        },
      };
    });
    setRunOrder((prev) => (prev.includes(runId) ? prev : [runId, ...prev]));
    setSelectedRunId((prev) => prev || runId);
  }, []);

  const appendMetricPoint = useCallback((runId, phase, point) => {
    setRuns((prev) => {
      const current = prev[runId];
      if (!current) return prev;
      const nextMetrics = { ...current.metrics };
      const list = nextMetrics[phase] ? [...nextMetrics[phase]] : [];
      const last = list[list.length - 1];
      if (last && last.epoch === point.epoch && last.step === point.step) {
        return prev;
      }
      list.push(point);
      nextMetrics[phase] = list;
      return {
        ...prev,
        [runId]: { ...current, metrics: nextMetrics, updatedAt: point.timestamp || current.updatedAt },
      };
    });
  }, []);

  const handleTelemetryEvent = useCallback(
    (raw) => {
      const event = raw && typeof raw === "object" ? raw : null;
      if (!event) return false;
      if (!event.event_type && !event.schema_version && !event.run_id) return false;

      const runId = event.run_id || event.payload?.run_id;
      if (!runId) return false;
      ensureRun(runId);

      const payload = event.payload || event;
      const eventType = event.event_type || payload.event_type || "epoch_completed";
      const timestamp = event.timestamp || payload.timestamp || Date.now() / 1000;

      setRuns((prev) => {
        const current = prev[runId];
        if (!current) return prev;
        const next = {
          ...current,
          updatedAt: timestamp,
          status:
            eventType === "run_completed"
              ? "completed"
              : eventType === "run_failed"
                ? "failed"
                : "running",
        };
        if (eventType === "run_started") {
          next.startedAt = timestamp;
        }
        return { ...prev, [runId]: next };
      });

      const epoch = payload.epoch ?? payload.step ?? null;
      const step = payload.step ?? null;
      const phase = payload.phase || "train";
      const loss = payload.loss ?? payload.metrics?.loss ?? null;
      const primaryMetric =
        payload.metrics?.accuracy ??
        payload.metrics?.mse ??
        payload.metric ??
        payload.metrics?.metric ??
        null;
      const valLoss = payload.metrics?.val_loss ?? null;
      const valMetric =
        payload.metrics?.val_accuracy ??
        payload.metrics?.val_mse ??
        payload.metrics?.val_metric ??
        null;

      if (epoch !== null && loss !== null) {
        appendMetricPoint(runId, phase, {
          epoch,
          step,
          loss,
          metric: primaryMetric,
          timestamp,
        });
      }

      if (epoch !== null && (valLoss !== null || valMetric !== null)) {
        appendMetricPoint(runId, "val", {
          epoch,
          step,
          loss: valLoss ?? null,
          metric: valMetric ?? null,
          timestamp,
        });
      }

      return true;
    },
    [appendMetricPoint, ensureRun],
  );

  useEffect(() => {
    const socket = io(urls.WS_DL_RESULTS, {
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 10000,
    });
    socketRef.current = socket;

    const dlResultListener = (resp) => {
      if (handleTelemetryEvent(resp)) return;
      clearTimeout(timeoutRef.current);
      if (resp.message && resp.message.includes("Starting")) {
        setResultValues([]);
        setIsLoading(true);
      } else if (resp.message && resp.message.includes("Finish")) {
        setIsLoading(false);
      } else {
        setResultValues((prev) => {
          let newValues = [...prev];
          newValues[parseInt(resp.test)] = resp.message;
          return newValues;
        });
      }
    };

    socket.on(strings.DL_RESULT_LISTENER, dlResultListener);

    socket.on("connect_error", (err) => {
      logger.warn("Socket connection error:", err);
      setConnectionError("Lost connection to server");
    });

    socket.on("connect", () => {
      setConnectionError(null);
    });

    socket.on("disconnect", (reason) => {
      if (reason === "io server disconnect") {
        socket.connect();
      }
    });

    getAllModels(projectId)
      .then((response) => {
        const models = response.map((item, index) => ({
          label: item.model_name + strings.MODEL_EXTENSION,
          value: item.model_name,
          id: item.id,
          key: index,
        }));
        setModelList(models);
      })
      .catch((error) => {
        logger.error("Error loading models:", error);
        setModelList([]);
      });

    getAllFiles(projectId)
      .then((response) => {
        const files = response.map((file, index) => ({
          label: `${file.file_name}.${file.file_type}`,
          value: String(file.file_id),
          key: index,
        }));
        setFileList(files);
        setFileDetails(response);
      })
      .catch((error) => {
        logger.error("Error loading files:", error);
      });

    getRunHistory(projectId)
      .then((resp) => {
        if (!resp?.success || !Array.isArray(resp.data)) return;
        const nextRuns = {};
        const nextOrder = [];
        resp.data.forEach((run) => {
          nextRuns[run.run_id] = {
            run_id: run.run_id,
            status: run.status || "completed",
            startedAt: run.started_at,
            updatedAt: run.updated_at,
            metrics: { train: [], val: [] },
          };
          nextOrder.push(run.run_id);
        });
        if (nextOrder.length > 0) {
          setRuns(nextRuns);
          setRunOrder(nextOrder);
          setSelectedRunId((prev) => prev || nextOrder[0]);
        }
      })
      .catch((error) => {
        logger.error("Error loading run history:", error);
      });

    return () => {
      clearTimeout(timeoutRef.current);
      socket.off(strings.DL_RESULT_LISTENER, dlResultListener);
      socket.disconnect();
    };
  }, [projectId, setModelList, handleTelemetryEvent]);

  // Validation functions
  const validateEpochs = (value) => {
    if (!value || value.trim() === "") {
      return "Epochs is required";
    }
    const trimmed = value.trim();
    if (!/^\d+$/.test(trimmed)) {
      return "Epochs must be a positive integer";
    }
    const num = Number(trimmed);
    if (num <= 0) {
      return "Epochs must be a positive integer";
    }
    return "";
  };

  const validateBatchSize = (value) => {
    if (!value || value.trim() === "") {
      return "Batch size is required";
    }
    const trimmed = value.trim();
    if (!/^\d+$/.test(trimmed)) {
      return "Batch size must be a positive integer";
    }
    const num = Number(trimmed);
    if (num <= 0) {
      return "Batch size must be a positive integer";
    }
    return "";
  };

  const validateTrainingSplit = (value) => {
    const num = parseFloat(value);
    if (!value || value.trim() === "") {
      return "Training split is required";
    }
    if (isNaN(num) || num <= 0 || num >= 1) {
      return "Training split must be between 0 and 1 (exclusive)";
    }
    return "";
  };

  const validateModel = (value) => {
    if (!value) {
      return "A model must be selected";
    }
    return "";
  };

  const validateFile = (value) => {
    if (!value) {
      return "A dataset file must be selected";
    }
    return "";
  };

  const validateProblemType = (value) => {
    if (!value) {
      return "Problem type must be selected";
    }
    return "";
  };

  const validateOptimizer = (value) => {
    if (!value) {
      return "Optimizer must be selected";
    }
    return "";
  };

  const validateMetric = (value) => {
    if (!value) {
      return "Result metric must be selected";
    }
    return "";
  };

  const validateTargetField = (value) => {
    if (!value || value.trim() === "") {
      return "Target field must be specified";
    }
    return "";
  };

  // Real-time validation handler
  const updateValidationErrors = useCallback((field, value) => {
    let error = "";
    switch (field) {
      case "epochs":
        error = validateEpochs(value);
        break;
      case "batch_size":
        error = validateBatchSize(value);
        break;
      case "training_split":
        error = validateTrainingSplit(value);
        break;
      case "model":
        error = validateModel(value);
        break;
      case "file_id":
        error = validateFile(value);
        break;
      case "problem_type_id":
        error = validateProblemType(value);
        break;
      case "optimizer":
        error = validateOptimizer(value);
        break;
      case "metric":
        error = validateMetric(value);
        break;
      case "target_field":
        error = validateTargetField(value);
        break;
      default:
        break;
    }
    setValidationErrors((prev) => ({ ...prev, [field]: error }));
  }, []); // Empty deps since validation functions are stable within component

  // Check if form has any validation errors
  const hasValidationErrors = () => {
    return Object.values(validationErrors).some((error) => error !== "");
  };

  // Validate all fields
  const validateAllFields = () => {
    const errors = {
      model: validateModel(selectedModel),
      file_id: validateFile(trainingConfig.file_id),
      problem_type_id: validateProblemType(trainingConfig.problem_type_id),
      optimizer: validateOptimizer(trainingConfig.optimizer),
      metric: validateMetric(trainingConfig.metric),
      target_field: validateTargetField(trainingConfig.target_field),
      epochs: validateEpochs(trainingConfig.epochs),
      batch_size: validateBatchSize(trainingConfig.batch_size),
      training_split: validateTrainingSplit(trainingConfig.training_split),
    };
    setValidationErrors(errors);
    return !Object.values(errors).some((error) => error !== "");
  };

  const handleFileSelect = useCallback(
    (value) => {
      const normalizedValue = String(value);
      const selected = fileDetails.find((item) => String(item.file_id) === normalizedValue);
      if (selected && selected.fields && selected.fields.length > 0) {
        const fields = selected.fields.map((item, index) => ({
          label: item,
          value: item,
          key: index,
        }));
        setFieldsList(fields);
      } else {
        setFieldsList([]);
      }
      setTrainingConfig((prev) => ({ ...prev, file_id: value, target_field: "" }));
      updateValidationErrors("file_id", value);
      updateValidationErrors("target_field", "");
    },
    [fileDetails, updateValidationErrors],
  );

  const handleModelSelect = (value) => {
    setSelectedModel(value);
    setConfigSaved(false);
    updateValidationErrors("model", value);
  };

  const handleSaveConfig = () => {
    if (!selectedModel) return;

    // Final validation check before submitting
    if (!validateAllFields()) {
      return;
    }

    const data = {
      model_name: selectedModel,
      file_id: trainingConfig.file_id,
      target_field: trainingConfig.target_field || null,
      training_split: Number(trainingConfig.training_split) * 100,
      problem_type_id: Number(trainingConfig.problem_type_id),
      optimizer: trainingConfig.optimizer,
      metric: trainingConfig.metric,
      epochs: Number(trainingConfig.epochs),
      batch_size: trainingConfig.batch_size ? Number(trainingConfig.batch_size) : 32,
      project_id: projectId || null,
    };

    updateTrainingConfig(data)
      .then((resp) => {
        if (resp.success) {
          setConfigSaved(true);
        } else {
          logger.error("Failed to save training config:", resp.message);
        }
      })
      .catch((error) => {
        logger.error("Error saving training config:", error);
      });
  };

  const isConfigValid =
    selectedModel &&
    trainingConfig.file_id &&
    trainingConfig.problem_type_id &&
    trainingConfig.optimizer &&
    trainingConfig.metric &&
    trainingConfig.epochs &&
    trainingConfig.batch_size &&
    trainingConfig.training_split &&
    trainingConfig.target_field &&
    !hasValidationErrors();

  const handleDownload = () => {
    if (selectedModel) {
      download_code(selectedModel, projectId).catch((error) => logger.error(error));
    }
  };

  const handleRun = () => {
    if (!selectedModel) return;

    // Final validation check before training
    if (!validateAllFields()) {
      return;
    }

    if (!socketRef.current?.connected) {
      setResultValues([
        "Cannot start training: not connected to server. Please wait and try again.",
      ]);
      return;
    }
    setResultValues([]);
    setIsLoading(true);
    setRuns({});
    setRunOrder([]);
    setSelectedRunId(null);
    setComparedRunIds([]);
    timeoutRef.current = setTimeout(() => {
      setIsLoading(false);
      setResultValues(["Training timed out. The model may still be running on the server."]);
    }, 300000);
    runModel(selectedModel, projectId)
      .then(() => {})
      .catch((error) => {
        clearTimeout(timeoutRef.current);
        logger.error(error.response?.data);
        setResultValues([error.response?.data?.message ?? "An error occurred"]);
        setIsLoading(false);
      });
  };

  const handleClear = () => {
    setResultValues([]);
    setIsLoading(false);
    setRuns({});
    setRunOrder([]);
    setSelectedRunId(null);
    setComparedRunIds([]);
  };

  const handleDeleteClick = (model, e) => {
    e.stopPropagation();
    setDeleteConfirm({ open: true, model });
  };

  const handleDeleteConfirm = () => {
    const { model } = deleteConfirm;
    setDeleteConfirm({ open: false, model: null });
    deleteModel(model.id)
      .then((resp) => {
        if (resp.success) {
          setModelList((prev) => prev.filter((m) => m.id !== model.id));
          if (selectedModel === model.value) {
            setSelectedModel(null);
            setConfigSaved(false);
          }
        } else {
          logger.error("Failed to delete model:", resp.message);
          setDeleteFeedback({
            open: true,
            success: false,
            message: resp.message || "Failed to delete model",
          });
        }
      })
      .catch((error) => {
        logger.error("Error deleting model:", error);
        setDeleteFeedback({
          open: true,
          success: false,
          message: error.message || "An unexpected error occurred",
        });
      });
  };

  const selectedRun = selectedRunId ? runs[selectedRunId] : null;
  const compareActive = comparedRunIds.length > 1;
  const selectedRuns = compareActive
    ? comparedRunIds.map((id) => runs[id]).filter(Boolean)
    : selectedRun
      ? [selectedRun]
      : [];

  const buildSeries = (metricKey, useVal = false) =>
    selectedRuns.map((run, index) => {
      const source = useVal ? run.metrics.val : run.metrics.train;
      const data = source
        .filter((point) => point[metricKey] !== null && point[metricKey] !== undefined)
        .map((point) => ({ x: point.epoch ?? point.step ?? 0, y: point[metricKey] }));
      return {
        label: compareActive ? run.run_id : useVal ? "Validation" : "Training",
        color: colors[index % colors.length],
        data,
      };
    });

  const toggleCompareRun = (runId) => {
    setComparedRunIds((prev) => {
      if (prev.includes(runId)) return prev.filter((id) => id !== runId);
      return [...prev, runId].slice(-3);
    });
  };

  const beginExport = (format) => {
    if (!selectedModel) return;
    startExport(selectedModel, format, selectedRunId)
      .then((resp) => {
        if (!resp?.success) {
          logger.error(resp?.message || "Export failed");
          return;
        }
        const exportId = resp.data?.export_id;
        if (!exportId) return;
        setExports((prev) => [
          { export_id: exportId, format, status: "running", error: null },
          ...prev,
        ]);
      })
      .catch((error) => {
        logger.error("Export error:", error);
      });
  };

  useEffect(() => {
    if (exports.length === 0) return;
    const running = exports.filter((e) => e.status === "running");
    if (running.length === 0) return;

    const timer = setInterval(() => {
      running.forEach((job) => {
        getExportStatus(job.export_id)
          .then((resp) => {
            if (!resp?.success) return;
            const status = resp.data?.status;
            setExports((prev) =>
              prev.map((item) =>
                item.export_id === job.export_id
                  ? {
                      ...item,
                      status: status || item.status,
                      error: resp.data?.error_text || null,
                    }
                  : item,
              ),
            );
          })
          .catch(() => {});
      });
    }, 2000);

    return () => clearInterval(timer);
  }, [exports, getExportStatus]);

  const handleDownloadExport = (job) => {
    const suffix =
      job.format === "onnx" ? "onnx" : job.format === "tflite" ? "tflite" : "zip";
    downloadExport(job.export_id, `${selectedModel || "model"}_${job.format}.${suffix}`);
  };

  const handleGenerateReport = () => {
    if (!selectedModel) return;
    setReportStatus("running");
    generateInterpretability(selectedModel, selectedRunId)
      .then((resp) => {
        if (!resp?.success) {
          setReportStatus("failed");
          return;
        }
        setReport(resp.data);
        setReportStatus(resp.data?.status || "completed");
      })
      .catch(() => {
        setReportStatus("failed");
      });
  };

  const handleStartTuning = () => {
    if (!selectedModel) return;
    const payload = {
      model_name: selectedModel,
      strategy: tuningConfig.strategy,
      objective: tuningConfig.objective,
      max_trials: Number(tuningConfig.max_trials),
      search_space: {
        optimizer: tuningConfig.optimizers
          .split(",")
          .map((o) => o.trim())
          .filter(Boolean),
        epochs: {
          min: Number(tuningConfig.epochs_min),
          max: Number(tuningConfig.epochs_max),
          step: Number(tuningConfig.epochs_step),
        },
        batch_size: {
          values: tuningConfig.batch_sizes
            .split(",")
            .map((b) => Number(b.trim()))
            .filter((b) => !Number.isNaN(b)),
        },
      },
    };
    startTuning(payload)
      .then((resp) => {
        if (!resp?.success) return;
        setTuningJobId(resp.data?.job_id || null);
        setTuningResults([]);
      })
      .catch((error) => {
        logger.error("Tuning start failed:", error);
      });
  };

  useEffect(() => {
    if (!tuningJobId) return;
    const timer = setInterval(() => {
      getTuningStatus(tuningJobId)
        .then((resp) => {
          if (!resp?.success) return;
          setTuningStatus(resp.data);
          if (resp.data?.status === "completed") {
            getTuningResults(tuningJobId)
              .then((resultsResp) => {
                if (resultsResp?.success) {
                  setTuningResults(resultsResp.data || []);
                }
              })
              .catch(() => {});
          }
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, [tuningJobId, getTuningStatus, getTuningResults]);

  const handleApplyBest = () => {
    if (!tuningJobId) return;
    applyBestTuning(tuningJobId)
      .then((resp) => {
        if (!resp?.success) return;
        const params = resp.data?.params;
        if (!params) return;
        setTrainingConfig((prev) => ({
          ...prev,
          optimizer: params.optimizer || prev.optimizer,
          epochs: String(params.epochs ?? prev.epochs),
          batch_size: String(params.batch_size ?? prev.batch_size),
        }));
        setConfigSaved(false);
      })
      .catch((error) => {
        logger.error("Apply best failed:", error);
      });
  };

  useEffect(() => {
    if (!selectedRunId) return;
    const run = runs[selectedRunId];
    if (!run || run.metrics.train.length > 0 || run.metrics.val.length > 0) return;

    getRunMetrics(selectedRunId)
      .then((resp) => {
        if (!resp?.success || !resp.data?.metrics) return;
        const train = [];
        const val = [];
        resp.data.metrics.forEach((point) => {
          const target = point.phase === "val" ? val : train;
          target.push({
            epoch: point.epoch,
            step: point.step,
            loss: point.loss,
            metric: point.metric,
            timestamp: point.timestamp,
          });
        });
        setRuns((prev) => ({
          ...prev,
          [selectedRunId]: {
            ...prev[selectedRunId],
            metrics: { train, val },
          },
        }));
      })
      .catch((error) => {
        logger.error("Error loading run metrics:", error);
      });
  }, [selectedRunId, runs, getRunMetrics]);

  return (
    <div className="space-y-6">
      <FeedbackDialog
        open={deleteFeedback.open}
        onClose={() => setDeleteFeedback((prev) => ({ ...prev, open: false }))}
        success={deleteFeedback.success}
        message={deleteFeedback.message}
      />
      <Dialog
        open={deleteConfirm.open}
        onOpenChange={(open) => !open && setDeleteConfirm({ open: false, model: null })}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete model</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete <strong>{deleteConfirm.model?.label}</strong>? This
              action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteConfirm({ open: false, model: null })}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeleteConfirm}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Card>
        <CardHeader>
          <CardTitle>Model Training</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <div className="space-y-1">
              <Select
                onValueChange={handleModelSelect}
                value={selectedModel ?? ""}
                disabled={modelList.length === 0}
              >
                <SelectTrigger className={`w-64 ${validationErrors.model ? "border-red-500" : ""}`}>
                  <SelectValue
                    placeholder={modelList.length === 0 ? "No models created" : "Select a model"}
                  />
                </SelectTrigger>
                <SelectContent>
                  {modelList.map((model) => (
                    <SelectItem key={model.key} value={model.value}>
                      <span className="flex items-center justify-between gap-2 w-full">
                        <span>{model.label}</span>
                        <button
                          type="button"
                          className="ml-auto text-destructive hover:text-destructive/80"
                          onClick={(e) => handleDeleteClick(model, e)}
                          aria-label={`Delete ${model.label}`}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {validationErrors.model && (
                <p className="text-sm text-red-500">{validationErrors.model}</p>
              )}
            </div>
            <Button
              onClick={handleDownload}
              disabled={!selectedModel || !configSaved}
              variant="outline"
            >
              Download Code
            </Button>
            <Button
              onClick={handleRun}
              disabled={!selectedModel || !configSaved || isLoading || hasValidationErrors()}
            >
              {isLoading ? "Training..." : "Train"}
            </Button>
            <Button
              onClick={handleClear}
              disabled={resultValues.length === 0 && !isLoading}
              variant="secondary"
            >
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      {selectedModel && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-1">
                <Label>Dataset File</Label>
                <Select onValueChange={handleFileSelect}>
                  <SelectTrigger className={validationErrors.file_id ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select a file" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {fileList.map((f) => (
                      <SelectItem key={f.key} value={f.value}>
                        {f.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.file_id && (
                  <p className="text-sm text-red-500">{validationErrors.file_id}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Problem Type</Label>
                <Select
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, problem_type_id: v }));
                    updateValidationErrors("problem_type_id", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger
                    className={validationErrors.problem_type_id ? "border-red-500" : ""}
                  >
                    <SelectValue placeholder="Select problem type" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {problemTypeOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.problem_type_id && (
                  <p className="text-sm text-red-500">{validationErrors.problem_type_id}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Target Field</Label>
                <div className="relative">
                  <Input
                    type="text"
                    placeholder={
                      fieldsList.length === 0 && trainingConfig.file_id
                        ? "Enter target field name (e.g. species, class, target)"
                        : fieldsList.length === 0
                          ? "Select a dataset file first or enter field name"
                          : "Select from list or enter field name"
                    }
                    value={trainingConfig.target_field}
                    className={validationErrors.target_field ? "border-red-500" : ""}
                    onChange={(e) => {
                      const value = e.target.value;
                      setTrainingConfig((prev) => ({ ...prev, target_field: value }));
                      updateValidationErrors("target_field", value);
                      setConfigSaved(false);
                    }}
                    list={fieldsList.length > 0 ? "target-fields" : undefined}
                  />
                  {fieldsList.length > 0 && (
                    <datalist id="target-fields">
                      {fieldsList.map((f) => (
                        <option key={f.key} value={f.value}>
                          {f.label}
                        </option>
                      ))}
                    </datalist>
                  )}
                </div>
                {fieldsList.length > 0 && (
                  <p className="text-xs text-gray-500">
                    Available fields: {fieldsList.map((f) => f.label).join(", ")}
                  </p>
                )}
                {validationErrors.target_field && (
                  <p className="text-sm text-red-500">{validationErrors.target_field}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Optimizer</Label>
                <Select
                  value={trainingConfig.optimizer}
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, optimizer: v }));
                    updateValidationErrors("optimizer", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger className={validationErrors.optimizer ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select optimizer" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {optimizerOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.optimizer && (
                  <p className="text-sm text-red-500">{validationErrors.optimizer}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Result Metrics</Label>
                <Select
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, metric: v }));
                    updateValidationErrors("metric", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger className={validationErrors.metric ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select metric" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {metricOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.metric && (
                  <p className="text-sm text-red-500">{validationErrors.metric}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Epochs</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Number of epochs"
                  value={trainingConfig.epochs}
                  className={validationErrors.epochs ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, epochs: value }));
                    updateValidationErrors("epochs", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.epochs && (
                  <p className="text-sm text-red-500">{validationErrors.epochs}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Batch size"
                  value={trainingConfig.batch_size}
                  className={validationErrors.batch_size ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, batch_size: value }));
                    updateValidationErrors("batch_size", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.batch_size && (
                  <p className="text-sm text-red-500">{validationErrors.batch_size}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Train:Test Ratio</Label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  placeholder="e.g. 0.8"
                  value={trainingConfig.training_split}
                  className={validationErrors.training_split ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, training_split: value }));
                    updateValidationErrors("training_split", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.training_split && (
                  <p className="text-sm text-red-500">{validationErrors.training_split}</p>
                )}
              </div>
            </div>

            <Button className="mt-4" onClick={handleSaveConfig} disabled={!isConfigValid}>
              {configSaved ? "Configuration Saved" : "Save Configuration"}
            </Button>
          </CardContent>
        </Card>
      )}

      {connectionError && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {connectionError}
        </div>
      )}

      {isLoading && resultValues.length === 0 && (
        <Card>
          <CardContent className="flex items-center justify-center py-16 text-muted-foreground">
            Model is running, please wait...
          </CardContent>
        </Card>
      )}

      {resultValues.length > 0 && (
        <div className="space-y-2">
          {resultValues.map((item, index) => (
            <Result result={item} key={index} />
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Export</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <Button
              variant="outline"
              onClick={() => beginExport("savedmodel")}
              disabled={!selectedModel}
            >
              Export SavedModel
            </Button>
            <Button
              variant="outline"
              onClick={() => beginExport("onnx")}
              disabled={!selectedModel}
            >
              Export ONNX
            </Button>
            <Button
              variant="outline"
              onClick={() => beginExport("tflite")}
              disabled={!selectedModel}
            >
              Export TFLite
            </Button>
          </div>
          {exports.length === 0 ? (
            <p className="text-sm text-muted-foreground">No exports yet.</p>
          ) : (
            <div className="space-y-2">
              {exports.map((job) => (
                <div
                  key={job.export_id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-md border px-3 py-2 text-xs"
                >
                  <div>
                    <div className="font-semibold">{job.export_id}</div>
                    <div className="text-muted-foreground">{job.format}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] ${
                        job.status === "completed"
                          ? "bg-emerald-50 text-emerald-700"
                          : job.status === "failed"
                            ? "bg-red-50 text-red-700"
                            : "bg-slate-50 text-slate-700"
                      }`}
                    >
                      {job.status}
                    </span>
                    {job.status === "completed" && (
                      <Button size="sm" onClick={() => handleDownloadExport(job)}>
                        Download
                      </Button>
                    )}
                  </div>
                  {job.error ? (
                    <div className="w-full text-xs text-destructive">{job.error}</div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {runOrder.length > 0 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <RunHistoryPanel
            runs={runs}
            runOrder={runOrder}
            selectedRunId={selectedRunId}
            onSelect={setSelectedRunId}
            comparedRunIds={comparedRunIds}
            onToggleCompare={toggleCompareRun}
          />
          <div className="space-y-4 lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {compareActive ? "Run Comparison" : "Run Metrics"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <TelemetryChart
                  title="Loss"
                  series={
                    compareActive
                      ? buildSeries("loss", false)
                      : [...buildSeries("loss", false), ...buildSeries("loss", true)]
                  }
                />
                <TelemetryChart
                  title="Primary Metric"
                  series={
                    compareActive
                      ? buildSeries("metric", false)
                      : [...buildSeries("metric", false), ...buildSeries("metric", true)]
                  }
                />
              </CardContent>
            </Card>
            {selectedRun && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Run Summary</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <div className="text-xs text-muted-foreground">Run ID</div>
                    <div className="text-sm font-semibold">{selectedRun.run_id}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Status</div>
                    <div className="text-sm font-semibold">{selectedRun.status}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Train Points</div>
                    <div className="text-sm font-semibold">
                      {selectedRun.metrics.train.length}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Validation Points</div>
                    <div className="text-sm font-semibold">
                      {selectedRun.metrics.val.length}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Interpretability</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap items-center gap-3">
                  <Button onClick={handleGenerateReport} disabled={!selectedModel || reportStatus === "running"}>
                    {reportStatus === "running" ? "Generating..." : "Generate Report"}
                  </Button>
                  {reportStatus === "failed" && (
                    <span className="text-xs text-destructive">Report generation failed</span>
                  )}
                </div>
                {report?.summary && (
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    {Object.entries(report.summary).map(([key, value]) => (
                      <div key={key} className="rounded-md border bg-white px-3 py-2 text-xs">
                        <div className="text-muted-foreground">{key}</div>
                        <div className="text-sm font-semibold">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                )}
                {report?.artifacts?.confusion_matrix && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold">Confusion Matrix</div>
                    <div className="overflow-auto">
                      <table className="w-full border text-xs">
                        <tbody>
                          {report.artifacts.confusion_matrix.map((row, rowIdx) => (
                            <tr key={`row-${rowIdx}`}>
                              {row.map((cell, cellIdx) => (
                                <td
                                  key={`cell-${rowIdx}-${cellIdx}`}
                                  className="border px-2 py-1 text-center"
                                >
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                {report?.artifacts?.classification_report && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold">Classification Report</div>
                    <div className="overflow-auto">
                      <table className="w-full border text-xs">
                        <thead>
                          <tr>
                            <th className="border px-2 py-1 text-left">Class</th>
                            <th className="border px-2 py-1 text-left">Precision</th>
                            <th className="border px-2 py-1 text-left">Recall</th>
                            <th className="border px-2 py-1 text-left">F1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(report.artifacts.classification_report).map(([label, metrics]) => {
                            if (typeof metrics !== "object") return null;
                            return (
                              <tr key={label}>
                                <td className="border px-2 py-1">{label}</td>
                                <td className="border px-2 py-1">{metrics["precision"]}</td>
                                <td className="border px-2 py-1">{metrics["recall"]}</td>
                                <td className="border px-2 py-1">{metrics["f1-score"]}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
                {report?.artifacts?.predictions && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold">Prediction Explorer (sample)</div>
                    <div className="overflow-auto">
                      <table className="w-full border text-xs">
                        <thead>
                          <tr>
                            <th className="border px-2 py-1 text-left">Actual</th>
                            <th className="border px-2 py-1 text-left">Predicted</th>
                          </tr>
                        </thead>
                        <tbody>
                          {report.artifacts.predictions.map((row, index) => (
                            <tr key={`pred-${index}`}>
                              <td className="border px-2 py-1">{row.actual}</td>
                              <td className="border px-2 py-1">{row.predicted}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Hyperparameter Tuning</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <div className="space-y-1">
                    <Label>Strategy</Label>
                    <Select
                      value={tuningConfig.strategy}
                      onValueChange={(v) => setTuningConfig((prev) => ({ ...prev, strategy: v }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select strategy" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="grid">Grid</SelectItem>
                        <SelectItem value="random">Random</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label>Objective</Label>
                    <Select
                      value={tuningConfig.objective}
                      onValueChange={(v) => setTuningConfig((prev) => ({ ...prev, objective: v }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select objective" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="val_accuracy">val_accuracy</SelectItem>
                        <SelectItem value="val_loss">val_loss</SelectItem>
                        <SelectItem value="accuracy">accuracy</SelectItem>
                        <SelectItem value="mse">mse</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label>Max Trials</Label>
                    <Input
                      type="number"
                      min="1"
                      value={tuningConfig.max_trials}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, max_trials: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Optimizers (comma)</Label>
                    <Input
                      value={tuningConfig.optimizers}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, optimizers: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Epochs Min</Label>
                    <Input
                      type="number"
                      value={tuningConfig.epochs_min}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, epochs_min: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Epochs Max</Label>
                    <Input
                      type="number"
                      value={tuningConfig.epochs_max}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, epochs_max: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Epochs Step</Label>
                    <Input
                      type="number"
                      value={tuningConfig.epochs_step}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, epochs_step: e.target.value }))
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Batch Sizes (comma)</Label>
                    <Input
                      value={tuningConfig.batch_sizes}
                      onChange={(e) =>
                        setTuningConfig((prev) => ({ ...prev, batch_sizes: e.target.value }))
                      }
                    />
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <Button onClick={handleStartTuning} disabled={!selectedModel}>
                    Start Tuning
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleApplyBest}
                    disabled={!tuningJobId || tuningResults.length === 0}
                  >
                    Apply Best Config
                  </Button>
                  {tuningStatus && (
                    <span className="text-xs text-muted-foreground">
                      {tuningStatus.completed_trials}/{tuningStatus.total_trials} trials •{" "}
                      {tuningStatus.status}
                    </span>
                  )}
                </div>
                {tuningResults.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold">Trial Results</div>
                    <div className="overflow-auto">
                      <table className="w-full border text-xs">
                        <thead>
                          <tr>
                            <th className="border px-2 py-1 text-left">Trial</th>
                            <th className="border px-2 py-1 text-left">Score</th>
                            <th className="border px-2 py-1 text-left">Params</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tuningResults.map((trial) => (
                            <tr key={trial.trial_id}>
                              <td className="border px-2 py-1">{trial.trial_id}</td>
                              <td className="border px-2 py-1">{trial.score}</td>
                              <td className="border px-2 py-1">
                                {JSON.stringify(trial.params)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
