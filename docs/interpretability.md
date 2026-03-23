# Interpretability

Reports are generated from trained model + test split.

## Outputs
- Classification: confusion matrix + classification report
- Regression: residual summary + MAE/MSE/RMSE
- Prediction explorer (sample rows)

## Endpoints
- `POST /api/v1/model/interpretability/generate`
- `GET /api/v1/model/interpretability/{report_id}`
- `GET /api/v1/model/interpretability/{report_id}/status`

## Frontend
Training page includes an Interpretability panel with report tables.
