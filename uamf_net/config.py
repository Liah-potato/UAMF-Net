"""Explicit reference settings; clinical data and checkpoints stay external."""

import json
import math
from pathlib import Path

MODALITIES = ("pathology", "ct", "gastro")


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if set(config["models"]) != set(MODALITIES):
        raise ValueError("Configure exactly pathology, ct, and gastro models")
    if len(config["classes"]) != 3 or len(set(config["classes"])) != 3:
        raise ValueError("Exactly three distinct class names are required")
    policy = config["policy"]
    for key in ("entropy_threshold", "recall_threshold", "benign_threshold", "margin"):
        if not math.isfinite(policy[key]) or policy[key] < 0:
            raise ValueError(f"Invalid policy setting: {key}")
    if policy["entropy_threshold"] > math.log(3):
        raise ValueError("Entropy is measured in natural-log units, in [0, ln(3)]")
    if any(policy[k] > 1 for k in ("recall_threshold", "benign_threshold", "margin")):
        raise ValueError("Probability thresholds and margin must be in [0, 1]")
    if policy["tie_priority"] not in ("ct", "gastro"):
        raise ValueError("tie_priority must be ct or gastro")
    if config["pathology_views"] not in (1, 5):
        raise ValueError("pathology_views must be 1 or 5")
    for name, model in config["models"].items():
        if not isinstance(model["input_size"], int) or model["input_size"] < 1:
            raise ValueError(f"Invalid input size for {name}")
        if model["resize_size"] < model["input_size"]:
            raise ValueError(f"resize_size is smaller than input_size for {name}")
    preprocessing = config["normalization"]
    if len(preprocessing["mean"]) != 3 or len(preprocessing["std"]) != 3:
        raise ValueError("RGB normalization requires three means and standard deviations")
    if any(not math.isfinite(v) for v in preprocessing["mean"] + preprocessing["std"]):
        raise ValueError("Normalization values must be finite")
    if any(v <= 0 for v in preprocessing["std"]):
        raise ValueError("Normalization standard deviations must be positive")
    train = config["training"]
    if train["mode"] not in ("head_only", "full"):
        raise ValueError("training.mode must be head_only or full")
    if train["loss"] not in ("poly1", "cross_entropy"):
        raise ValueError("training.loss must be poly1 or cross_entropy")
    if train["loss"] == "poly1" and train["label_smoothing"] != 0:
        raise ValueError("This Poly-1 implementation uses hard labels; set label_smoothing to 0")
    if not 0 <= train["label_smoothing"] < 1 or train["poly_epsilon"] < 0:
        raise ValueError("Invalid loss configuration")
    if any(train[k] < 1 for k in ("epochs", "batch_size", "patience")):
        raise ValueError("epochs, batch_size, and patience must be positive")
    if train["workers"] < 0 or train["learning_rate"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Invalid optimizer or worker settings")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
