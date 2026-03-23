"""Hyperparameter tuning service (grid + random)."""

from __future__ import annotations

import itertools
import random
import uuid as uuid_pkg
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.models import ModelBasic, TuningJob, TuningTrial
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger
from app.socketio_instance import sio
from app.shared.constants import SOCKETIO_DL_NAMESPACE, SOCKETIO_LISTENER
from app.services.model_run import _helper_generate_file_location  # type: ignore

logger = get_logger(__name__)


def _emit_tuning(event_type: str, payload: dict):
    data = {
        "schema_version": 1,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "payload": payload,
    }
    try:
        sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE)
    except Exception:
        logger.warning("Failed to emit tuning event: %s", event_type)


def _load_tabular_data(db: Session, model: ModelBasic):
    file_path = _helper_generate_file_location(db, model.file_id)
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X = df.drop(model.target_field, axis=1)
    y = df[model.target_field]
    split_index = int(len(X) * model.training_split / 100)
    x_train = X[:split_index]
    y_train = y[:split_index]
    x_test = X[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test


def _load_model_json(model_name: str):
    from app.shared.constants import MODEL_GENERATION_LOCATION, MODEL_GENERATION_TYPE

    with open(f"{MODEL_GENERATION_LOCATION}/{model_name}{MODEL_GENERATION_TYPE}") as f:
        json_string = f.read()
    return tf.keras.models.model_from_json(json_string, custom_objects=None)


def _objective_direction(name: str) -> str:
    if name in {"loss", "val_loss", "mse"}:
        return "min"
    return "max"


def _build_candidates(search_space: dict, strategy: str, max_trials: int, seed: int | None):
    rng = random.Random(seed)
    optimizers = search_space.get("optimizer", ["adam"])
    epochs = search_space.get("epochs", {}).get("values")
    if not epochs:
        min_e = search_space.get("epochs", {}).get("min", 5)
        max_e = search_space.get("epochs", {}).get("max", 20)
        step = search_space.get("epochs", {}).get("step", 5)
        epochs = list(range(min_e, max_e + 1, step))
    batch_sizes = search_space.get("batch_size", {}).get("values", [32])

    grid = list(itertools.product(optimizers, epochs, batch_sizes))
    if strategy == "grid":
        return grid[:max_trials]
    return [rng.choice(grid) for _ in range(max_trials)]


def _run_trial(db: Session, model: ModelBasic, params: dict, objective: str):
    keras_model = _load_model_json(model.model_name)
    if model.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()

    keras_model.compile(optimizer=params["optimizer"], loss=loss, metrics=[model.metric])

    if model.model_type == ProblemType.IMAGE_CLASSIFICATION:
        raise ValueError("Tuning for image classification is not supported in this prototype.")

    x_train, y_train, x_test, y_test = _load_tabular_data(db, model)
    keras_model.fit(
        x_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
    )
    eval_results = keras_model.evaluate(x_test, y_test, verbose=0)
    loss_val = float(eval_results[0])
    metric_val = float(eval_results[1]) if len(eval_results) > 1 else None
    if objective in {"loss", "val_loss"}:
        return loss_val
    return metric_val if metric_val is not None else loss_val


def start_tuning(db: Session, payload: dict) -> tuple:
    model_name = payload.get("model_name")
    strategy = payload.get("strategy", "random")
    objective = payload.get("objective", "val_accuracy")
    max_trials = int(payload.get("max_trials", 10))
    search_space = payload.get("search_space", {})
    seed = payload.get("seed")

    if not model_name:
        return {"success": False, "message": "model_name required"}, 400

    model = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if not model:
        return {"success": False, "message": "Model not found"}, 404
    if model.file_id is None or model.epochs is None:
        return {"success": False, "message": "Training configuration not set"}, 400

    job_id = f"tune_{uuid_pkg.uuid4().hex[:10]}"
    job = TuningJob(
        job_id=job_id,
        model_id=model.id,
        model_name=model.model_name,
        strategy=strategy,
        objective=objective,
        max_trials=max_trials,
        status="running",
        created_on=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()

    candidates = _build_candidates(search_space, strategy, max_trials, seed)
    direction = _objective_direction(objective)
    best_score = None
    best_trial = None

    _emit_tuning("tuning_started", {"job_id": job_id, "total_trials": len(candidates)})

    for idx, (optimizer, epochs, batch_size) in enumerate(candidates, start=1):
        trial_id = f"{job_id}_trial_{idx}"
        params = {"optimizer": optimizer, "epochs": int(epochs), "batch_size": int(batch_size)}
        _emit_tuning("trial_started", {"job_id": job_id, "trial_id": trial_id, "params": params})
        try:
            score = _run_trial(db, model, params, objective)
            trial = TuningTrial(
                job_id=job_id,
                trial_id=trial_id,
                status="completed",
                params=params,
                score=score,
            )
            db.add(trial)
            db.commit()
            _emit_tuning("trial_completed", {"job_id": job_id, "trial_id": trial_id, "score": score})
            if best_score is None or (
                direction == "max" and score > best_score
            ) or (direction == "min" and score < best_score):
                best_score = score
                best_trial = trial
                _emit_tuning("best_updated", {"job_id": job_id, "trial_id": trial_id, "score": score})
        except Exception as e:
            trial = TuningTrial(
                job_id=job_id,
                trial_id=trial_id,
                status="failed",
                params=params,
                error_text=str(e),
            )
            db.add(trial)
            db.commit()
            _emit_tuning("trial_failed", {"job_id": job_id, "trial_id": trial_id, "error": str(e)})

    job.status = "completed"
    job.updated_on = datetime.now(timezone.utc)
    db.add(job)
    db.commit()

    _emit_tuning(
        "tuning_completed",
        {
            "job_id": job_id,
            "best_trial": {
                "trial_id": best_trial.trial_id if best_trial else None,
                "score": best_score,
                "params": best_trial.params if best_trial else None,
            },
        },
    )

    return {
        "success": True,
        "message": "Tuning completed",
        "data": {"job_id": job_id},
    }, 200


def get_tuning_status(db: Session, job_id: str) -> tuple:
    job = db.exec(select(TuningJob).where(TuningJob.job_id == job_id)).first()
    if not job:
        return {"success": False, "message": "Job not found"}, 404
    trials = db.exec(select(TuningTrial).where(TuningTrial.job_id == job_id)).all()
    completed = len([t for t in trials if t.status == "completed"])
    return {
        "success": True,
        "message": "Tuning status",
        "data": {
            "job_id": job.job_id,
            "status": job.status,
            "completed_trials": completed,
            "total_trials": job.max_trials,
        },
    }, 200


def get_tuning_results(db: Session, job_id: str) -> tuple:
    trials = db.exec(select(TuningTrial).where(TuningTrial.job_id == job_id)).all()
    data = [
        {"trial_id": t.trial_id, "status": t.status, "params": t.params, "score": t.score}
        for t in trials
    ]
    return {"success": True, "message": "Tuning results", "data": data}, 200


def apply_best_config(db: Session, job_id: str) -> tuple:
    job = db.exec(select(TuningJob).where(TuningJob.job_id == job_id)).first()
    if not job:
        return {"success": False, "message": "Job not found"}, 404
    trials = db.exec(select(TuningTrial).where(TuningTrial.job_id == job_id)).all()
    completed = [t for t in trials if t.status == "completed" and t.score is not None]
    if not completed:
        return {"success": False, "message": "No completed trials"}, 400
    direction = _objective_direction(job.objective)
    best = max(completed, key=lambda t: t.score) if direction == "max" else min(completed, key=lambda t: t.score)
    model = db.exec(select(ModelBasic).where(ModelBasic.id == job.model_id)).first()
    if not model:
        return {"success": False, "message": "Model not found"}, 404
    model.optimizer = best.params.get("optimizer", model.optimizer)
    model.epochs = best.params.get("epochs", model.epochs)
    model.batch_size = best.params.get("batch_size", model.batch_size)
    db.add(model)
    db.commit()
    return {
        "success": True,
        "message": "Best config applied",
        "data": {"params": best.params, "score": best.score},
    }, 200
