"""Select recall and benign thresholds using validation traces only."""

import copy
import json
from pathlib import Path

from .fusion import Prediction, arbitrate
from .metrics import classification_metrics


def tune_policy(config, path):
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or any(r["split"] != "val" for r in records):
        raise ValueError("Threshold selection accepts nonempty validation-only traces")
    if any(r.get("auxiliary_collection") != "all_available" for r in records):
        raise ValueError("Validation traces must collect every available auxiliary expert")
    prepared = []
    for row in records:
        experts = row["experts"]
        values = [Prediction(tuple(experts[name]["probabilities"])) if experts[name] is not None else None
                  for name in ("pathology", "ct", "gastro")]
        if values[0] is None:
            raise ValueError("Every trace requires pathology probabilities")
        prepared.append((row["label"], values))
    candidates, best_key, best_policy = [], None, None
    for recall in (i / 100 for i in range(50, 91, 5)):
        for benign in (i / 100 for i in range(60, 96, 5)):
            policy = {**config["policy"], "recall_threshold": recall, "benign_threshold": benign}
            decisions = [arbitrate(*values, policy) for _, values in prepared]
            labels = [label for label, _ in prepared]
            predictions = [d["final"] for d in decisions]
            accuracy = classification_metrics(labels, predictions)["accuracy"]
            changes = sum(d["changed"] for d in decisions)
            # Predeclared tie break: fewer overrides, then higher thresholds.
            key = (accuracy, -changes, recall, benign)
            candidates.append({"recall_threshold": recall, "benign_threshold": benign,
                               "validation_accuracy": accuracy, "overrides": changes})
            if best_key is None or key > best_key:
                best_key, best_policy = key, policy
    selected = copy.deepcopy(config)
    selected["policy"] = best_policy
    return selected, {"selection_split": "val", "objective": "image_accuracy",
                      "tie_break": "fewer overrides, then higher recall and benign thresholds",
                      "candidates": candidates}
