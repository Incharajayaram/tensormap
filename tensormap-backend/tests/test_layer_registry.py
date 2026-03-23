"""Unit tests for layer registry validation."""

import pytest

from app.services.layer_registry import normalize_params


def test_dense_requires_units():
    with pytest.raises(ValueError, match="Missing required param"):
        normalize_params("customdense", {"activation": "relu"})


def test_dense_activation_enum():
    with pytest.raises(ValueError, match="Value must be one of"):
        normalize_params("customdense", {"units": 8, "activation": "not-a-real-activation"})


def test_dense_defaults_activation():
    params = normalize_params("customdense", {"units": 8})
    assert params["activation"] == "relu"
