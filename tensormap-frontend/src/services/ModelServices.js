import axios from "../shared/Axios";
import * as urls from "../constants/Urls";
import * as strings from "../constants/Strings";
import logger from "../shared/logger";

/**
 * Sends a model graph and training config to the backend for validation.
 *
 * On network error, returns a fallback `{ success: false }` object
 * instead of throwing, so callers always receive a response shape.
 *
 * @param {object} data - Payload with `code`, `model`, and `project_id`.
 * @returns {Promise<{ success: boolean, message: string }>}
 */
export const validateModel = async (data) =>
  axios
    .post(urls.BACKEND_VALIDATE_MODEL, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occured" };
    });

/**
 * Fetches the list of validated model objects, optionally scoped to a project.
 *
 * @param {string} [projectId]
 * @returns {Promise<Array<{ id: number, model_name: string }>>} Array of model objects.
 */
export const getAllModels = async (projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(urls.BACKEND_GET_ALL_MODELS, { params })
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.data;
      }
      return [];
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

/**
 * Deletes a saved model by its database ID.
 *
 * @param {number} modelId
 * @returns {Promise<{ success: boolean, message: string }>}
 */
export const deleteModel = async (modelId) =>
  axios
    .delete(`${urls.BACKEND_DELETE_MODEL}/${modelId}`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

/**
 * Downloads the generated Python training script for a model.
 *
 * Creates a temporary download link, triggers the browser download,
 * and cleans up the object URL afterwards.
 *
 * @param {string} model_name
 * @param {string} [projectId]
 * @returns {Promise<void>}
 */
export const download_code = async (model_name, projectId) => {
  const data = { model_name, ...(projectId && { project_id: projectId }) };
  return axios
    .post(urls.BACKEND_DOWNLOAD_CODE, data)
    .then((resp) => {
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(
        new Blob([resp.data], { type: "application/octet-stream" }),
      );
      link.download = data.model_name + strings.MODEL_EXTENSION;

      document.body.appendChild(link);

      link.click();

      setTimeout(() => {
        window.URL.revokeObjectURL(link.href);
        document.body.removeChild(link);
      }, 200);
    })
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

/**
 * Starts a training run for a validated model on the backend.
 *
 * Training progress is delivered separately via the Socket.IO
 * `/dl-result` namespace.
 *
 * @param {string} modelName
 * @param {string} [projectId]
 * @returns {Promise<string>} Success message from the API.
 */
export const saveModel = async (data) =>
  axios
    .post(urls.BACKEND_SAVE_MODEL, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

export const updateTrainingConfig = async (data) =>
  axios
    .patch(urls.BACKEND_UPDATE_TRAINING_CONFIG, data)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      if (err.response && err.response.data) {
        return err.response.data;
      }
      return { success: false, message: "Unknown error occurred" };
    });

/**
 * Fetches the full ReactFlow graph for a saved model.
 *
 * @param {string} modelName
 * @param {string} [projectId]
 * @returns {Promise<{ success: boolean, data?: { model_name: string, graph: object } }>}
 */
export const getModelGraph = async (modelName, projectId) => {
  const params = projectId ? { project_id: projectId } : {};
  return axios
    .get(`${urls.BACKEND_GET_MODEL_GRAPH}/${encodeURIComponent(modelName)}/graph`, { params })
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      return { success: false };
    });
};

export const runModel = async (modelName, projectId) => {
  const data = {
    model_name: modelName,
    ...(projectId && { project_id: projectId }),
  };

  return axios
    .post(urls.BACKEND_RUN_MODEL, data)
    .then((resp) => {
      if (resp.data.success === true) {
        return resp.data.message;
      }
      throw new Error(resp.data);
    })
    .catch((err) => {
      throw err;
    });
};

export const getRunHistory = async (projectId, modelName) => {
  const params = {};
  if (projectId) params.project_id = projectId;
  if (modelName) params.model_name = modelName;
  return axios
    .get("/model/runs", { params })
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });
};

export const getRunMetrics = async (runId) =>
  axios
    .get(`/model/runs/${encodeURIComponent(runId)}/metrics`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const startExport = async (model_name, format, run_id) =>
  axios
    .post("/model/export/start", { model_name, format, run_id })
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const getExportStatus = async (exportId) =>
  axios
    .get(`/model/export/${encodeURIComponent(exportId)}/status`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const downloadExport = async (exportId, filename) =>
  axios
    .get(`/model/export/${encodeURIComponent(exportId)}/download`, {
      responseType: "blob",
    })
    .then((resp) => {
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(resp.data);
      link.download = filename || `export_${exportId}`;
      document.body.appendChild(link);
      link.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(link.href);
        document.body.removeChild(link);
      }, 200);
    });

export const generateInterpretability = async (model_name, run_id) =>
  axios
    .post("/model/interpretability/generate", { model_name, run_id })
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const getInterpretabilityReport = async (reportId) =>
  axios
    .get(`/model/interpretability/${encodeURIComponent(reportId)}`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const getInterpretabilityStatus = async (reportId) =>
  axios
    .get(`/model/interpretability/${encodeURIComponent(reportId)}/status`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const startTuning = async (payload) =>
  axios
    .post("/model/tuning/start", payload)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const getTuningStatus = async (jobId) =>
  axios
    .get(`/model/tuning/${encodeURIComponent(jobId)}/status`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const getTuningResults = async (jobId) =>
  axios
    .get(`/model/tuning/${encodeURIComponent(jobId)}/results`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });

export const applyBestTuning = async (jobId) =>
  axios
    .post(`/model/tuning/${encodeURIComponent(jobId)}/apply-best`)
    .then((resp) => resp.data)
    .catch((err) => {
      logger.error(err);
      throw err;
    });
