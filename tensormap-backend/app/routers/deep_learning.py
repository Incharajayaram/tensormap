import asyncio
import io
import os
import uuid as uuid_pkg

from fastapi import APIRouter, Depends, Query
import zipfile

from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session

from app.database import get_db
from app.schemas.deep_learning import ModelNameRequest, ModelSaveRequest, ModelValidateRequest, TrainingConfigRequest
from app.services.deep_learning import (
    delete_model_service,
    get_available_model_list,
    get_code_service,
    get_run_history_service,
    get_run_metrics_service,
    get_model_graph_service,
    model_save_service,
    model_validate_service,
    run_code_service,
    update_training_config_service,
)
from app.services.export import get_export_download, get_export_status, start_export
from app.services.interpretability import generate_report, get_report, get_report_status
from app.services.tuning import apply_best_config, get_tuning_results, get_tuning_status, start_tuning
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["deep-learning"])


@router.post("/model/validate")
def validate_model(request: ModelValidateRequest, db: Session = Depends(get_db)):
    """Validate a ReactFlow graph as a Keras model and persist the configuration."""
    logger.debug("Validating model for project_id=%s", request.project_id)
    body, status_code = model_validate_service(db, incoming=request.model_dump(), project_id=request.project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/save")
def save_model(request: ModelSaveRequest, db: Session = Depends(get_db)):
    """Save a model architecture from the canvas (no training config)."""
    logger.debug("Saving model architecture: model_name=%s", request.model_name)
    body, status_code = model_save_service(
        db, incoming=request.model.model_dump(), model_name=request.model_name, project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.patch("/model/training-config")
def update_training_config(request: TrainingConfigRequest, db: Session = Depends(get_db)):
    """Set training configuration on a previously saved model."""
    logger.debug("Updating training config for model_name=%s", request.model_name)
    body, status_code = update_training_config_service(
        db, model_name=request.model_name, config=request.model_dump(), project_id=request.project_id
    )
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/code")
def get_code(request: ModelNameRequest, db: Session = Depends(get_db)):
    """Generate and download a Python training script for a saved model."""
    logger.debug("Generating code for model_name=%s", request.model_name)
    result, status_code = get_code_service(db, model_name=request.model_name, project_id=request.project_id)
    if status_code == 200:
        temp_file = io.BytesIO(result["content"].encode())
        return StreamingResponse(
            temp_file,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={result['file_name']}"},
        )
    return JSONResponse(status_code=status_code, content=result)


@router.post("/model/run")
async def run_model(request: ModelNameRequest, db: Session = Depends(get_db)):
    """Train a saved model in a background thread and stream progress via Socket.IO."""
    logger.info("Starting model training: model_name=%s", request.model_name)
    loop = asyncio.get_running_loop()
    body, status_code = await asyncio.to_thread(
        run_code_service, db, model_name=request.model_name, project_id=request.project_id, loop=loop
    )
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/{model_name}/graph")
def get_model_graph(
    model_name: str,
    project_id: uuid_pkg.UUID | None = Query(None),
    db: Session = Depends(get_db),
):
    """Retrieve the full ReactFlow graph for a saved model."""
    body, status_code = get_model_graph_service(db, model_name=model_name, project_id=project_id)
    return JSONResponse(status_code=status_code, content=body)


@router.delete("/model/{model_id}")
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
):
    """Delete a saved model and its associated configuration records."""
    logger.info("Deleting model id=%s", model_id)
    body, status_code = delete_model_service(db, model_id=model_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/model-list")
def get_model_list(
    project_id: uuid_pkg.UUID | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return a paginated list of saved model names, optionally filtered by project."""
    body, status_code = get_available_model_list(db, project_id=project_id, offset=offset, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/runs")
def get_run_history(
    model_name: str | None = Query(None),
    project_id: uuid_pkg.UUID | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Return recent training runs (optionally filtered by model or project)."""
    body, status_code = get_run_history_service(db, model_name=model_name, project_id=project_id, limit=limit)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/runs/{run_id}/metrics")
def get_run_metrics(run_id: str, db: Session = Depends(get_db)):
    """Return persisted metrics for a training run."""
    body, status_code = get_run_metrics_service(db, run_id=run_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/export/start")
def start_model_export(request: dict, db: Session = Depends(get_db)):
    """Start an export job for a model (SavedModel, ONNX, TFLite)."""
    model_name = request.get("model_name")
    fmt = request.get("format")
    run_id = request.get("run_id")
    if not model_name or not fmt:
        return JSONResponse(status_code=400, content={"success": False, "message": "model_name and format required"})
    body, status_code = start_export(db, model_name=model_name, fmt=fmt, run_id=run_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/export/{export_id}/status")
def get_model_export_status(export_id: str, db: Session = Depends(get_db)):
    body, status_code = get_export_status(db, export_id=export_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/export/{export_id}/download")
def download_model_export(export_id: str, db: Session = Depends(get_db)):
    result, status_code = get_export_download(db, export_id=export_id)
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=result)

    path = result["data"]["path"]
    fmt = result["data"]["format"]

    if fmt == "savedmodel" and os.path.isdir(path):
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, path)
                    zf.write(full_path, rel_path)
        mem_zip.seek(0)
        return StreamingResponse(
            mem_zip,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={export_id}_savedmodel.zip"},
        )

    if fmt == "onnx":
        filename = f"{export_id}.onnx"
        media_type = "application/octet-stream"
    elif fmt == "tflite":
        filename = f"{export_id}.tflite"
        media_type = "application/octet-stream"
    else:
        filename = f"{export_id}_artifact"
        media_type = "application/octet-stream"

    return StreamingResponse(
        open(path, "rb"),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/model/interpretability/generate")
def generate_interpretability(request: dict, db: Session = Depends(get_db)):
    model_name = request.get("model_name")
    run_id = request.get("run_id")
    if not model_name:
        return JSONResponse(status_code=400, content={"success": False, "message": "model_name required"})
    body, status_code = generate_report(db, model_name=model_name, run_id=run_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/interpretability/{report_id}")
def get_interpretability_report(report_id: str, db: Session = Depends(get_db)):
    body, status_code = get_report(db, report_id=report_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/interpretability/{report_id}/status")
def get_interpretability_status(report_id: str, db: Session = Depends(get_db)):
    body, status_code = get_report_status(db, report_id=report_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/tuning/start")
def start_tuning_job(request: dict, db: Session = Depends(get_db)):
    body, status_code = start_tuning(db, request)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/tuning/{job_id}/status")
def tuning_status(job_id: str, db: Session = Depends(get_db)):
    body, status_code = get_tuning_status(db, job_id)
    return JSONResponse(status_code=status_code, content=body)


@router.get("/model/tuning/{job_id}/results")
def tuning_results(job_id: str, db: Session = Depends(get_db)):
    body, status_code = get_tuning_results(db, job_id)
    return JSONResponse(status_code=status_code, content=body)


@router.post("/model/tuning/{job_id}/apply-best")
def tuning_apply_best(job_id: str, db: Session = Depends(get_db)):
    body, status_code = apply_best_config(db, job_id)
    return JSONResponse(status_code=status_code, content=body)
