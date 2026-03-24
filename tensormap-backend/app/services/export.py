"""Model export service (SavedModel, ONNX, TFLite)."""

from __future__ import annotations

import os
import uuid as uuid_pkg
from datetime import datetime, timezone

import tensorflow as tf
from sqlmodel import Session, select

from app.models import ExportJob, ModelBasic
from app.shared.constants import MODEL_GENERATION_LOCATION, MODEL_GENERATION_TYPE
from app.shared.logging_config import get_logger

logger = get_logger(__name__)

EXPORT_ROOT = "./exports"
EXPORT_FORMATS = {"savedmodel", "onnx", "tflite"}


def _ensure_export_root() -> None:
    os.makedirs(EXPORT_ROOT, exist_ok=True)


def _model_json_path(model_name: str) -> str:
    return os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE)


def _create_job(db: Session, model: ModelBasic, fmt: str, run_id: str | None) -> ExportJob:
    export_id = f"export_{uuid_pkg.uuid4().hex[:10]}"
    job = ExportJob(
        export_id=export_id,
        model_id=model.id,
        model_name=model.model_name,
        run_id=run_id,
        format=fmt,
        status="running",
        created_on=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _update_job(db: Session, job: ExportJob, status: str, path: str | None = None, error: str | None = None):
    job.status = status
    job.updated_on = datetime.now(timezone.utc)
    if path:
        job.path = path
    if error:
        job.error_text = error
    db.add(job)
    db.commit()


def start_export(db: Session, model_name: str, fmt: str, run_id: str | None = None) -> tuple:
    if fmt not in EXPORT_FORMATS:
        return {"success": False, "message": "Unsupported export format"}, 400

    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return {"success": False, "message": "Model not found"}, 404

    _ensure_export_root()
    job = _create_job(db, model, fmt, run_id)
    export_dir = os.path.join(EXPORT_ROOT, job.export_id)
    os.makedirs(export_dir, exist_ok=True)

    try:
        with open(_model_json_path(model_name)) as f:
            json_string = f.read()
        keras_model = tf.keras.models.model_from_json(json_string, custom_objects=None)

        if fmt == "savedmodel":
            out_path = os.path.join(export_dir, "savedmodel")
            # Newer Keras requires model.export for SavedModel directories.
            keras_model.export(out_path)
            _update_job(db, job, status="completed", path=out_path)
        elif fmt == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            tflite_model = converter.convert()
            out_path = os.path.join(export_dir, "model.tflite")
            with open(out_path, "wb") as f:
                f.write(tflite_model)
            _update_job(db, job, status="completed", path=out_path)
        elif fmt == "onnx":
            try:
                import tf2onnx  # type: ignore
            except Exception:
                raise RuntimeError("tf2onnx not installed on server")
            out_path = os.path.join(export_dir, "model.onnx")
            tf2onnx.convert.from_keras(keras_model, output_path=out_path)
            _update_job(db, job, status="completed", path=out_path)

        return {
            "success": True,
            "message": "Export completed",
            "data": {"export_id": job.export_id, "status": "completed", "format": fmt},
        }, 200
    except Exception as e:
        logger.exception("Export failed: %s", str(e))
        _update_job(db, job, status="failed", error=str(e))
        return {"success": False, "message": f"Export failed: {e}"}, 400


def get_export_status(db: Session, export_id: str) -> tuple:
    job = db.exec(select(ExportJob).where(ExportJob.export_id == export_id)).first()
    if not job:
        return {"success": False, "message": "Export not found"}, 404
    return {
        "success": True,
        "message": "Export status",
        "data": {
            "export_id": job.export_id,
            "format": job.format,
            "status": job.status,
            "path": job.path,
            "error_text": job.error_text,
            "updated_on": job.updated_on.timestamp() if job.updated_on else None,
        },
    }, 200


def get_export_download(db: Session, export_id: str) -> tuple:
    job = db.exec(select(ExportJob).where(ExportJob.export_id == export_id)).first()
    if not job:
        return {"success": False, "message": "Export not found"}, 404
    if job.status != "completed" or not job.path:
        return {"success": False, "message": "Export not ready"}, 400
    return {"success": True, "data": {"path": job.path, "format": job.format}}, 200
