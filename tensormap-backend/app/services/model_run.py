import asyncio
import os
import time
import uuid as uuid_pkg
from datetime import datetime, timezone

import pandas as pd
import tensorflow as tf
from sqlmodel import Session, select

from app.config import get_settings
from app.models.data import DataFile, ImageProperties
from app.models.ml import ModelBasic, RunHistory, RunMetric
from app.database import engine
from app.shared.constants import (
    MODEL_GENERATION_LOCATION,
    MODEL_GENERATION_TYPE,
    SOCKETIO_DL_NAMESPACE,
    SOCKETIO_LISTENER,
)
from app.shared.enums import ProblemType
from app.shared.logging_config import get_logger
from app.socketio_instance import sio

logger = get_logger(__name__)

_main_loop: asyncio.AbstractEventLoop | None = None


class CustomProgressBar(tf.keras.callbacks.Callback):
    """Keras callback that emits training progress to the frontend via Socket.IO."""

    def __init__(self, run_id: str, model_name: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.model_name = model_name

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        """Emit the start of a new epoch."""
        _model_result(f"Epoch {epoch + 1}/{self.params['epochs']}", 0)
        _emit_event(
            run_id=self.run_id,
            event_type="epoch_started",
            payload={"phase": "train", "epoch": epoch + 1},
        )

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        """Emit batch-level progress with loss and metric values."""
        progress = batch / self.params["steps"] if self.params["steps"] else 0
        progress_bar_width = 50
        arrow = ">" * int(progress * progress_bar_width)
        spaces = "=" * (progress_bar_width - len(arrow))
        if "mse" in logs:
            metric = f"MSE: {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy: {logs['accuracy']:.4f}"
        else:
            metric = ""

        _model_result(
            f"{batch + 1}/{self.params['steps']}  [{arrow}{spaces}] "
            f"{int(progress * 100)}% - Loss: {logs['loss']:.4f} - {metric}",
            1,
        )
        if batch % max(1, int(self.params.get("steps", 1) / 10)) == 0:
            _emit_event(
                run_id=self.run_id,
                event_type="batch_progress",
                payload={
                    "phase": "train",
                    "epoch": (self.params.get("epoch", 0) or 0) + 1,
                    "step": batch + 1,
                    "loss": float(logs.get("loss")) if logs else None,
                    "metrics": {
                        "accuracy": float(logs.get("accuracy")) if logs and "accuracy" in logs else None,
                        "mse": float(logs.get("mse")) if logs and "mse" in logs else None,
                    },
                },
            )

    def on_test_begin(self, logs: dict = None) -> None:
        """Emit the start of model evaluation."""
        _model_result("Evaluating...", 2)
        _emit_event(
            run_id=self.run_id,
            event_type="evaluation_started",
            payload={"phase": "eval"},
        )

    def on_test_end(self, logs: dict = None) -> None:
        """Emit final evaluation metrics."""
        if "mse" in logs:
            metric = f"MSE Loss- {logs['mse']:.4f}"
        elif "accuracy" in logs:
            metric = f"Accuracy- {logs['accuracy']:.4f}"
        else:
            metric = ""
        _model_result(f"Evaluation Results: {metric} Loss - {logs['loss']:.4f}", 3)
        _model_result("Finish", 4)
        _emit_event(
            run_id=self.run_id,
            event_type="evaluation_completed",
            payload={
                "phase": "eval",
                "loss": float(logs.get("loss")) if logs else None,
                "metrics": {
                    "accuracy": float(logs.get("accuracy")) if logs and "accuracy" in logs else None,
                    "mse": float(logs.get("mse")) if logs and "mse" in logs else None,
                },
            },
        )

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if not logs:
            return
        _emit_event(
            run_id=self.run_id,
            event_type="epoch_completed",
            payload={
                "phase": "train",
                "epoch": epoch + 1,
                "loss": float(logs.get("loss")) if logs.get("loss") is not None else None,
                "metrics": {
                    "accuracy": float(logs.get("accuracy")) if logs.get("accuracy") is not None else None,
                    "mse": float(logs.get("mse")) if logs.get("mse") is not None else None,
                    "val_loss": float(logs.get("val_loss")) if logs.get("val_loss") is not None else None,
                    "val_accuracy": float(logs.get("val_accuracy"))
                    if logs.get("val_accuracy") is not None
                    else None,
                    "val_mse": float(logs.get("val_mse")) if logs.get("val_mse") is not None else None,
                },
            },
        )
        _persist_epoch_metrics(self.run_id, epoch + 1, logs)


def _model_result(message: str, test: int) -> None:
    """Emit a Socket.IO event with a training progress message."""
    data = {"message": message, "test": test}
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE),
            _main_loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            logger.warning("Failed to emit Socket.IO event: %s", message)
    else:
        logger.warning("No running event loop for Socket.IO emit: %s", message)


def _emit_event(run_id: str, event_type: str, payload: dict | None = None) -> None:
    data = {
        "schema_version": 1,
        "run_id": run_id,
        "event_type": event_type,
        "timestamp": time.time(),
        "payload": payload or {},
    }
    if _main_loop is not None and _main_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            sio.emit(SOCKETIO_LISTENER, data, namespace=SOCKETIO_DL_NAMESPACE),
            _main_loop,
        )
        try:
            future.result(timeout=5)
        except Exception:
            logger.warning("Failed to emit Socket.IO structured event: %s", event_type)
    else:
        logger.warning("No running event loop for Socket.IO structured event: %s", event_type)


def _create_run_history(run_id: str, model: ModelBasic) -> None:
    try:
        with Session(engine) as session:
            session.add(
                RunHistory(
                    run_id=run_id,
                    model_id=model.id,
                    model_name=model.model_name,
                    status="running",
                    started_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
    except Exception as e:
        logger.warning("Failed to create run history for %s: %s", run_id, str(e))


def _update_run_history(run_id: str, status: str, summary: dict | None = None, error_text: str | None = None) -> None:
    try:
        with Session(engine) as session:
            run = session.exec(select(RunHistory).where(RunHistory.run_id == run_id)).first()
            if not run:
                return
            run.status = status
            run.updated_on = datetime.now(timezone.utc)
            if status in {"completed", "failed"}:
                run.completed_at = datetime.now(timezone.utc)
            if summary is not None:
                run.summary = summary
            if error_text:
                run.error_text = error_text
            session.add(run)
            session.commit()
    except Exception as e:
        logger.warning("Failed to update run history for %s: %s", run_id, str(e))


def _persist_epoch_metrics(run_id: str, epoch: int, logs: dict) -> None:
    try:
        with Session(engine) as session:
            timestamp = datetime.now(timezone.utc)
            primary_metric = logs.get("accuracy") if "accuracy" in logs else logs.get("mse")
            session.add(
                RunMetric(
                    run_id=run_id,
                    phase="train",
                    epoch=epoch,
                    loss=float(logs.get("loss")) if logs.get("loss") is not None else None,
                    metric=float(primary_metric) if primary_metric is not None else None,
                    metrics={
                        "accuracy": float(logs.get("accuracy")) if logs.get("accuracy") is not None else None,
                        "mse": float(logs.get("mse")) if logs.get("mse") is not None else None,
                    },
                    timestamp=timestamp,
                )
            )
            if logs.get("val_loss") is not None or logs.get("val_accuracy") is not None or logs.get("val_mse") is not None:
                val_metric = logs.get("val_accuracy") if "val_accuracy" in logs else logs.get("val_mse")
                session.add(
                    RunMetric(
                        run_id=run_id,
                        phase="val",
                        epoch=epoch,
                        loss=float(logs.get("val_loss")) if logs.get("val_loss") is not None else None,
                        metric=float(val_metric) if val_metric is not None else None,
                        metrics={
                            "val_accuracy": float(logs.get("val_accuracy"))
                            if logs.get("val_accuracy") is not None
                            else None,
                            "val_mse": float(logs.get("val_mse")) if logs.get("val_mse") is not None else None,
                        },
                        timestamp=timestamp,
                    )
                )
            session.commit()
    except Exception as e:
        logger.warning("Failed to persist metrics for run %s: %s", run_id, str(e))


def _helper_generate_file_location(db: Session, file_id) -> str:
    """Resolve the on-disk path for a dataset file by its DB ID."""
    upload_folder = get_settings().upload_folder
    file = db.exec(select(DataFile).where(DataFile.id == file_id)).first()
    if file.file_type == "zip":
        return upload_folder + "/" + file.file_name
    return upload_folder + "/" + file.file_name + "." + file.file_type


def _helper_generate_json_model_file_location(model_name: str) -> str:
    """Construct the path to the model's JSON file, validating against path traversal."""
    path = os.path.realpath(os.path.join(MODEL_GENERATION_LOCATION, model_name + MODEL_GENERATION_TYPE))
    base_dir = os.path.realpath(MODEL_GENERATION_LOCATION)
    if not path.startswith(base_dir + os.sep) and path != base_dir:
        raise ValueError("Invalid model path: escapes model directory")
    return path


def model_run(
    model_name: str,
    db: Session,
    loop: asyncio.AbstractEventLoop | None = None,
    run_id: str | None = None,
) -> None:
    """Load, compile, and train a Keras model, emitting progress via Socket.IO."""
    global _main_loop
    _main_loop = loop
    run_id = run_id or f"run_{uuid_pkg.uuid4().hex[:10]}"
    _emit_event(run_id=run_id, event_type="run_started", payload={"model_name": model_name})
    try:
        _run(model_name, db, run_id=run_id)
        _emit_event(run_id=run_id, event_type="run_completed", payload={"model_name": model_name})
    except Exception as e:
        logger.exception("Training failed for model '%s': %s", model_name, str(e))
        _model_result(f"Training failed: {e}", -1)
        _emit_event(run_id=run_id, event_type="run_failed", payload={"error": str(e)})
        _update_run_history(run_id, status="failed", error_text=str(e))
        raise


def _run(model_name: str, db: Session, run_id: str) -> None:
    model_configs = db.exec(select(ModelBasic).where(ModelBasic.model_name == model_name)).first()
    if model_configs:
        _create_run_history(run_id, model_configs)

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        image_properties = db.exec(select(ImageProperties).where(ImageProperties.id == model_configs.file_id)).first()
        image_size = (image_properties.image_size, image_properties.image_size)
        batch_size = image_properties.batch_size
        color_mode = image_properties.color_mode
        label_mode = image_properties.label_mode

        directory = _helper_generate_file_location(db, file_id=model_configs.file_id)
        logger.debug(
            "Image classification params - directory: %s, image_size: %s, "
            "batch_size: %s, color_mode: %s, label_mode: %s",
            directory,
            image_size,
            batch_size,
            color_mode,
            label_mode,
        )
        validation_split = 1 - (model_configs.training_split / 100)
        train_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
        test_data = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode=label_mode,
        )
    else:
        file_location = _helper_generate_file_location(db, file_id=model_configs.file_id)
        features = pd.read_csv(file_location)
        # Ensure all columns are numeric; non-numeric values become NaN and are dropped.
        features = features.apply(pd.to_numeric, errors="coerce")
        features.dropna(inplace=True)
        # Shuffle data to prevent issues with ordered datasets
        features = features.sample(frac=1, random_state=42).reset_index(drop=True)

        X = features.drop(model_configs.target_field, axis=1)
        y = features[model_configs.target_field]
        if model_configs.model_type == ProblemType.REGRESSION:
            y = y.astype(float)
        else:
            y = y.astype(int)

        split_index = int(len(X) * model_configs.training_split / 100)
        x_training = X[:split_index]
        y_training = y[:split_index]
        x_testing = X[split_index:]
        y_testing = y[split_index:]

        batch_size = model_configs.batch_size if model_configs.batch_size is not None else 32

    with open(_helper_generate_json_model_file_location(model_name=model_name)) as f:
        json_string = f.read()
    model = tf.keras.models.model_from_json(json_string, custom_objects=None)
    model.summary()
    if model_configs.loss == "sparse_categorical_crossentropy":
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        loss = tf.keras.losses.MeanSquaredError()
    model.compile(
        optimizer=model_configs.optimizer,
        loss=loss,
        metrics=[model_configs.metric],
    )

    if model_configs.model_type == ProblemType.IMAGE_CLASSIFICATION:
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=model_configs.epochs,
            callbacks=[CustomProgressBar(run_id=run_id, model_name=model_name)],
            verbose=0,
        )
        model.evaluate(test_data, callbacks=[CustomProgressBar(run_id=run_id, model_name=model_name)], verbose=0)
    else:
        model.fit(
            x_training,
            y_training,
            epochs=model_configs.epochs,
            batch_size=batch_size,
            callbacks=[CustomProgressBar(run_id=run_id, model_name=model_name)],
            verbose=0,
        )
        model.evaluate(
            x_testing,
            y_testing,
            callbacks=[CustomProgressBar(run_id=run_id, model_name=model_name)],
            verbose=0,
        )

    _update_run_history(run_id, status="completed")
