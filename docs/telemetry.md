# Training Telemetry

Training emits structured Socket.IO events. The frontend consumes these for charts and run history.

## Socket Payload
```
{
  "schema_version": 1,
  "run_id": "run_abc123",
  "event_type": "epoch_completed",
  "timestamp": 1773350400.12,
  "payload": {
    "phase": "train",
    "epoch": 3,
    "step": 120,
    "loss": 0.2451,
    "metrics": {
      "accuracy": 0.9123,
      "val_loss": 0.3012,
      "val_accuracy": 0.8874
    }
  }
}
```

## Endpoints
- `GET /api/v1/model/runs`
- `GET /api/v1/model/runs/{run_id}/metrics`

## Frontend
- Training page renders live charts and run history.
