# Hyperparameter Tuning

Built-in grid and random search for tabular models.

## Endpoints
- `POST /api/v1/model/tuning/start`
- `GET /api/v1/model/tuning/{job_id}/status`
- `GET /api/v1/model/tuning/{job_id}/results`
- `POST /api/v1/model/tuning/{job_id}/apply-best`

## Frontend
Training page includes a Tuning panel with:
- Search-space form
- Trial table
- "Apply Best Config" button
