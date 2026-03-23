# Layer Registry

The layer registry is the single source of truth for layer types, defaults, and param validation.

## Frontend
- File: `tensormap-frontend/src/registry/layers.js`
- Each layer defines:
  - `type`, `label`, `category`
  - `params` schema (type, required, default, options)

Adding a new layer requires only:
1. Add a new entry to `LAYER_REGISTRY`
2. Ensure backend registry has a matching entry (see below)

## Backend
- File: `tensormap-backend/app/services/layer_registry.py`
- Each layer defines:
  - `params` schema
  - `builder` that returns a Keras layer

## Success Criteria
- Sidebar renders from registry
- Node properties panel auto-generates fields
- Validation is schema-based
- Model generation uses registry (no if/elif blocks)
