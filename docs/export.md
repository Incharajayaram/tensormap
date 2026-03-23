# Model Export

Supported formats:
- SavedModel
- ONNX (requires `tf2onnx`)
- TFLite

## Endpoints
- `POST /api/v1/model/export/start`
- `GET /api/v1/model/export/{export_id}/status`
- `GET /api/v1/model/export/{export_id}/download`

## Frontend
Training page includes an Export panel with buttons and status tracking.
