"""Interpretability report generation."""

from __future__ import annotations

import uuid as uuid_pkg
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
from sqlmodel import Session, select

from app.models import InterpretabilityReport, ModelBasic
from app.shared.constants import MODEL_GENERATION_LOCATION, MODEL_GENERATION_TYPE
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger
from app.services.model_run import _helper_generate_file_location  # type: ignore

logger = get_logger(__name__)


def _load_model(model_name: str):
    path = f"{MODEL_GENERATION_LOCATION}/{model_name}{MODEL_GENERATION_TYPE}"
    with open(path) as f:
        json_string = f.read()
    return tf.keras.models.model_from_json(json_string, custom_objects=None)


def _get_file_location(db: Session, model: ModelBasic) -> str:
    return _helper_generate_file_location(db, model.file_id)


def _prepare_tabular(db: Session, model: ModelBasic):
    file_path = _get_file_location(db, model)
    df = pd.read_csv(file_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = df.drop(model.target_field, axis=1)
    y = df[model.target_field]
    split_index = int(len(X) * model.training_split / 100)
    x_test = X[split_index:]
    y_test = y[split_index:]
    return x_test, y_test


def _problem_type(model: ModelBasic) -> str:
    if model.model_type == ProblemType.REGRESSION:
        return "regression"
    if model.model_type == ProblemType.IMAGE_CLASSIFICATION:
        return "image_classification"
    return "classification"


def _create_report(db: Session, model: ModelBasic, run_id: str | None) -> InterpretabilityReport:
    report_id = f"rep_{uuid_pkg.uuid4().hex[:10]}"
    report = InterpretabilityReport(
        report_id=report_id,
        run_id=run_id,
        model_id=model.id,
        model_name=model.model_name,
        problem_type=_problem_type(model),
        status="running",
        created_on=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def _update_report(
    db: Session,
    report: InterpretabilityReport,
    status: str,
    summary: dict | None = None,
    artifacts: dict | None = None,
    error_text: str | None = None,
):
    report.status = status
    report.updated_on = datetime.now(timezone.utc)
    if summary is not None:
        report.summary = summary
    if artifacts is not None:
        report.artifacts = artifacts
    if error_text:
        report.error_text = error_text
    db.add(report)
    db.commit()


def generate_report(db: Session, model_name: str, run_id: str | None = None) -> tuple:
    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model or model.file_id is None:
        return {"success": False, "message": "Model not found or not configured"}, 404

    report = _create_report(db, model, run_id)
    try:
        keras_model = _load_model(model_name)
        problem_type = _problem_type(model)

        if problem_type == "regression":
            x_test, y_test = _prepare_tabular(db, model)
            preds = keras_model.predict(x_test, verbose=0).flatten()
            y_true = y_test.to_numpy().astype(float)
            mae = float(mean_absolute_error(y_true, preds))
            mse = float(mean_squared_error(y_true, preds))
            rmse = float(np.sqrt(mse))
            residuals = (y_true - preds).tolist()
            artifacts = {
                "residuals": residuals[:200],
                "predictions": [
                    {"actual": float(a), "predicted": float(p)}
                    for a, p in list(zip(y_true[:50], preds[:50]))
                ],
            }
            summary = {"mae": mae, "mse": mse, "rmse": rmse}
        else:
            x_test, y_test = _prepare_tabular(db, model)
            preds = keras_model.predict(x_test, verbose=0)
            if preds.ndim > 1 and preds.shape[1] > 1:
                y_pred = preds.argmax(axis=1)
            else:
                y_pred = np.round(preds).astype(int).flatten()
            y_true = y_test.to_numpy().astype(int)
            cm = confusion_matrix(y_true, y_pred).tolist()
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            summary = {
                "accuracy": report_dict.get("accuracy"),
                "macro_f1": report_dict.get("macro avg", {}).get("f1-score"),
            }
            artifacts = {
                "confusion_matrix": cm,
                "classification_report": report_dict,
                "predictions": [
                    {"actual": int(a), "predicted": int(p)}
                    for a, p in list(zip(y_true[:50], y_pred[:50]))
                ],
            }

        _update_report(db, report, status="completed", summary=summary, artifacts=artifacts)
        return {
            "success": True,
            "message": "Report generated",
            "data": {
                "report_id": report.report_id,
                "status": "completed",
                "summary": summary,
                "artifacts": artifacts,
            },
        }, 200
    except Exception as e:
        logger.exception("Interpretability failed: %s", str(e))
        _update_report(db, report, status="failed", error_text=str(e))
        return {"success": False, "message": f"Report failed: {e}"}, 400


def get_report(db: Session, report_id: str) -> tuple:
    report = db.exec(select(InterpretabilityReport).where(InterpretabilityReport.report_id == report_id)).first()
    if not report:
        return {"success": False, "message": "Report not found"}, 404
    return {
        "success": True,
        "message": "Report retrieved",
        "data": {
            "report_id": report.report_id,
            "status": report.status,
            "summary": report.summary,
            "artifacts": report.artifacts,
            "error_text": report.error_text,
        },
    }, 200


def get_report_status(db: Session, report_id: str) -> tuple:
    report = db.exec(select(InterpretabilityReport).where(InterpretabilityReport.report_id == report_id)).first()
    if not report:
        return {"success": False, "message": "Report not found"}, 404
    return {
        "success": True,
        "message": "Report status",
        "data": {
            "report_id": report.report_id,
            "status": report.status,
            "error_text": report.error_text,
        },
    }, 200
