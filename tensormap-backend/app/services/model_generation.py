"""Convert a ReactFlow graph into a Keras-compatible JSON model definition."""

import json
from collections import defaultdict

import tensorflow as tf

from app.services.layer_registry import build_layer
from app.shared.logging_config import get_logger

logger = get_logger(__name__)


def model_generation(model_params: dict) -> dict:
    """Transform ReactFlow nodes and edges into a Keras functional-API JSON structure.

    Builds the model programmatically using the Keras API, then serializes
    it via ``model.to_json()`` so the output always matches the installed
    Keras version's expected format.
    """
    logger.debug("Generating model from %d nodes, %d edges", len(model_params["nodes"]), len(model_params["edges"]))

    # Build adjacency maps
    source_to_targets = defaultdict(list)
    target_to_sources = defaultdict(list)
    for edge in model_params["edges"]:
        source_to_targets[edge["source"]].append(edge["target"])
        target_to_sources[edge["target"]].append(edge["source"])

    nodes_by_id = {node["id"]: node for node in model_params["nodes"]}

    # BFS from input nodes to build Keras layers in topological order
    keras_tensors = {}
    visited = set()
    queue = []

    for node in model_params["nodes"]:
        if node["type"] == "custominput":
            raw_params = node["data"]["params"]
            dims = [int(raw_params.get(f"dim-{i + 1}", 0) or 0) for i in range(3)]
            dims = [d for d in dims if d != 0]
            if not dims or dims[0] <= 0:
                raise ValueError("Input layer requires at least one positive dimension")
            keras_tensors[node["id"]] = tf.keras.Input(shape=dims, name=node["id"])
            visited.add(node["id"])
            queue.append(node["id"])

    while queue:
        current_id = queue.pop(0)
        for target_id in source_to_targets.get(current_id, []):
            if target_id in visited:
                continue
            # Check if all sources of this target have been visited
            all_sources = target_to_sources.get(target_id, [])
            if not all(src in visited for src in all_sources):
                continue

            visited.add(target_id)
            queue.append(target_id)

            # Collect input tensors for this node
            source_tensors = [keras_tensors[src] for src in all_sources]
            if len(source_tensors) > 1:
                input_tensor = tf.keras.layers.Concatenate(axis=-1)(source_tensors)
            else:
                input_tensor = source_tensors[0]

            node = nodes_by_id[target_id]
            keras_tensors[target_id] = _build_layer(node, input_tensor)

    # Identify input and output tensors
    inputs = [keras_tensors[n["id"]] for n in model_params["nodes"] if n["type"] == "custominput"]
    output_ids = [n["id"] for n in model_params["nodes"] if n["id"] not in source_to_targets]
    outputs = [keras_tensors[oid] for oid in output_ids]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return json.loads(model.to_json())


def _build_layer(node: dict, input_tensor):
    """Instantiate a single Keras layer from a ReactFlow node and apply it to the input tensor."""
    params = node["data"]["params"]
    node_type = node["type"]
    name = node["id"]
    return build_layer(node_type, params, input_tensor, name)
