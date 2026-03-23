"""Layer registry and parameter validation for model generation."""

from __future__ import annotations

import math
from typing import Any, Callable

import tensorflow as tf

Activation = str | None


def _activation(name: str | None) -> Activation:
    if name is None:
        return None
    return "linear" if name == "none" else name


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _as_int(value: Any) -> int:
    if value is None or value == "":
        raise ValueError("Value is required")
    return int(value)


def _as_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("Value is required")
    return float(value)


def _validate_range(name: str, value: float, min_val: float | None, max_val: float | None):
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}")


def _enum(value: Any, allowed: set[str]) -> str:
    if value is None or value == "":
        raise ValueError("Value is required")
    if value not in allowed:
        raise ValueError(f"Value must be one of {sorted(allowed)}")
    return value


ParamSpec = dict[str, Any]
Builder = Callable[[dict[str, Any], Any, str], Any]


def _spec(
    *,
    param_type: str,
    required: bool = False,
    default: Any | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    allowed: set[str] | None = None,
):
    return {
        "type": param_type,
        "required": required,
        "default": default,
        "min": min_val,
        "max": max_val,
        "allowed": allowed,
    }


_ACTIVATIONS = {
    "none",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "elu",
    "selu",
    "linear",
}

_PADDINGS = {"valid", "same"}
_INITIALIZERS = {"glorot_uniform", "he_normal", "lecun_normal"}


LAYER_REGISTRY: dict[str, dict[str, Any]] = {
    "customdense": {
        "params": {
            "units": _spec(param_type="int", required=True, min_val=1),
            "activation": _spec(param_type="enum", default="relu", allowed=_ACTIVATIONS),
        },
        "builder": lambda p, x, name: tf.keras.layers.Dense(
            units=p["units"],
            activation=_activation(p.get("activation")),
            name=name,
        )(x),
    },
    "customdenseadvanced": {
        "params": {
            "units": _spec(param_type="int", required=True, min_val=1),
            "activation": _spec(param_type="enum", default="relu", allowed=_ACTIVATIONS),
            "use_bias": _spec(param_type="bool", default=True),
            "kernel_initializer": _spec(param_type="enum", default="glorot_uniform", allowed=_INITIALIZERS),
        },
        "builder": lambda p, x, name: tf.keras.layers.Dense(
            units=p["units"],
            activation=_activation(p.get("activation")),
            use_bias=p.get("use_bias", True),
            kernel_initializer=p.get("kernel_initializer"),
            name=name,
        )(x),
    },
    "customactivation": {
        "params": {
            "activation": _spec(param_type="enum", required=True, allowed=_ACTIVATIONS),
        },
        "builder": lambda p, x, name: tf.keras.layers.Activation(
            _activation(p.get("activation")), name=name
        )(x),
    },
    "customflatten": {
        "params": {},
        "builder": lambda p, x, name: tf.keras.layers.Flatten(name=name)(x),
    },
    "customconv": {
        "params": {
            "filter": _spec(param_type="int", required=True, min_val=1),
            "kernelX": _spec(param_type="int", required=True, min_val=1),
            "kernelY": _spec(param_type="int", required=True, min_val=1),
            "strideX": _spec(param_type="int", required=True, min_val=1),
            "strideY": _spec(param_type="int", required=True, min_val=1),
            "padding": _spec(param_type="enum", default="valid", allowed=_PADDINGS),
            "activation": _spec(param_type="enum", default="none", allowed=_ACTIVATIONS),
        },
        "builder": lambda p, x, name: tf.keras.layers.Conv2D(
            filters=p["filter"],
            kernel_size=(p["kernelX"], p["kernelY"]),
            strides=(p["strideX"], p["strideY"]),
            padding=p.get("padding", "valid"),
            activation=_activation(p.get("activation")),
            name=name,
        )(x),
    },
    "custommaxpool2d": {
        "params": {
            "poolX": _spec(param_type="int", required=True, min_val=1),
            "poolY": _spec(param_type="int", required=True, min_val=1),
            "strideX": _spec(param_type="int", required=False, min_val=1),
            "strideY": _spec(param_type="int", required=False, min_val=1),
            "padding": _spec(param_type="enum", default="valid", allowed=_PADDINGS),
        },
        "builder": lambda p, x, name: tf.keras.layers.MaxPooling2D(
            pool_size=(p["poolX"], p["poolY"]),
            strides=(p.get("strideX"), p.get("strideY"))
            if p.get("strideX") and p.get("strideY")
            else None,
            padding=p.get("padding", "valid"),
            name=name,
        )(x),
    },
    "customavgpool2d": {
        "params": {
            "poolX": _spec(param_type="int", required=True, min_val=1),
            "poolY": _spec(param_type="int", required=True, min_val=1),
            "strideX": _spec(param_type="int", required=False, min_val=1),
            "strideY": _spec(param_type="int", required=False, min_val=1),
            "padding": _spec(param_type="enum", default="valid", allowed=_PADDINGS),
        },
        "builder": lambda p, x, name: tf.keras.layers.AveragePooling2D(
            pool_size=(p["poolX"], p["poolY"]),
            strides=(p.get("strideX"), p.get("strideY"))
            if p.get("strideX") and p.get("strideY")
            else None,
            padding=p.get("padding", "valid"),
            name=name,
        )(x),
    },
    "customglobalavgpool2d": {
        "params": {},
        "builder": lambda p, x, name: tf.keras.layers.GlobalAveragePooling2D(name=name)(x),
    },
    "customsepconv2d": {
        "params": {
            "filter": _spec(param_type="int", required=True, min_val=1),
            "kernelX": _spec(param_type="int", required=True, min_val=1),
            "kernelY": _spec(param_type="int", required=True, min_val=1),
            "strideX": _spec(param_type="int", required=False, min_val=1),
            "strideY": _spec(param_type="int", required=False, min_val=1),
            "depthMultiplier": _spec(param_type="int", required=False, min_val=1),
            "padding": _spec(param_type="enum", default="valid", allowed=_PADDINGS),
            "activation": _spec(param_type="enum", default="none", allowed=_ACTIVATIONS),
        },
        "builder": lambda p, x, name: tf.keras.layers.SeparableConv2D(
            filters=p["filter"],
            kernel_size=(p["kernelX"], p["kernelY"]),
            strides=(p.get("strideX", 1), p.get("strideY", 1)),
            depth_multiplier=p.get("depthMultiplier", 1),
            padding=p.get("padding", "valid"),
            activation=_activation(p.get("activation")),
            name=name,
        )(x),
    },
    "customembedding": {
        "params": {
            "input_dim": _spec(param_type="int", required=True, min_val=1),
            "output_dim": _spec(param_type="int", required=True, min_val=1),
            "input_length": _spec(param_type="int", required=False, min_val=1),
        },
        "builder": lambda p, x, name: tf.keras.layers.Embedding(
            input_dim=p["input_dim"],
            output_dim=p["output_dim"],
            input_length=p.get("input_length"),
            name=name,
        )(x),
    },
    "customlstm": {
        "params": {
            "units": _spec(param_type="int", required=True, min_val=1),
            "activation": _spec(param_type="enum", default="tanh", allowed=_ACTIVATIONS),
            "recurrent_activation": _spec(param_type="enum", default="sigmoid", allowed=_ACTIVATIONS),
            "return_sequences": _spec(param_type="bool", default=False),
            "dropout": _spec(param_type="float", required=False, min_val=0.0, max_val=1.0),
            "recurrent_dropout": _spec(param_type="float", required=False, min_val=0.0, max_val=1.0),
        },
        "builder": lambda p, x, name: tf.keras.layers.LSTM(
            units=p["units"],
            activation=_activation(p.get("activation")),
            recurrent_activation=_activation(p.get("recurrent_activation")),
            return_sequences=p.get("return_sequences", False),
            dropout=p.get("dropout", 0.0),
            recurrent_dropout=p.get("recurrent_dropout", 0.0),
            name=name,
        )(x),
    },
    "customgru": {
        "params": {
            "units": _spec(param_type="int", required=True, min_val=1),
            "activation": _spec(param_type="enum", default="tanh", allowed=_ACTIVATIONS),
            "recurrent_activation": _spec(param_type="enum", default="sigmoid", allowed=_ACTIVATIONS),
            "return_sequences": _spec(param_type="bool", default=False),
            "dropout": _spec(param_type="float", required=False, min_val=0.0, max_val=1.0),
            "recurrent_dropout": _spec(param_type="float", required=False, min_val=0.0, max_val=1.0),
        },
        "builder": lambda p, x, name: tf.keras.layers.GRU(
            units=p["units"],
            activation=_activation(p.get("activation")),
            recurrent_activation=_activation(p.get("recurrent_activation")),
            return_sequences=p.get("return_sequences", False),
            dropout=p.get("dropout", 0.0),
            recurrent_dropout=p.get("recurrent_dropout", 0.0),
            name=name,
        )(x),
    },
    "customdropout": {
        "params": {
            "rate": _spec(param_type="float", required=True, min_val=0.0, max_val=1.0),
        },
        "builder": lambda p, x, name: tf.keras.layers.Dropout(rate=p["rate"], name=name)(x),
    },
    "custombatchnorm": {
        "params": {
            "momentum": _spec(param_type="float", required=False, min_val=0.0, max_val=1.0),
            "epsilon": _spec(param_type="float", required=False, min_val=0.0),
        },
        "builder": lambda p, x, name: tf.keras.layers.BatchNormalization(
            momentum=p.get("momentum", 0.99),
            epsilon=p.get("epsilon", 0.001),
            name=name,
        )(x),
    },
}


def normalize_params(node_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize params for a layer. Raises ValueError on failure."""
    if node_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")
    specs: dict[str, ParamSpec] = LAYER_REGISTRY[node_type]["params"]
    normalized: dict[str, Any] = {}

    for key, spec in specs.items():
        raw_value = params.get(key, spec.get("default"))
        if raw_value is None or raw_value == "":
            if spec.get("required"):
                raise ValueError(f"Missing required param: {key}")
            continue

        param_type = spec["type"]
        if param_type == "int":
            value = _as_int(raw_value)
            _validate_range(key, value, spec.get("min"), spec.get("max"))
        elif param_type == "float":
            value = _as_float(raw_value)
            if math.isnan(value):
                raise ValueError(f"{key} must be a valid number")
            _validate_range(key, value, spec.get("min"), spec.get("max"))
        elif param_type == "bool":
            value = _as_bool(raw_value)
        elif param_type == "enum":
            value = _enum(raw_value, spec.get("allowed") or set())
        else:
            value = raw_value

        normalized[key] = value

    return normalized


def build_layer(node_type: str, params: dict[str, Any], input_tensor, name: str):
    """Build a Keras layer from registry specs."""
    if node_type not in LAYER_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")
    normalized = normalize_params(node_type, params)
    builder: Builder = LAYER_REGISTRY[node_type]["builder"]
    return builder(normalized, input_tensor, name)
