"""Auditable decision rules, independent of image models and ground-truth labels."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    probabilities: tuple

    def __post_init__(self):
        values = tuple(float(v) for v in self.probabilities)
        if len(values) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError("Expected three finite probabilities in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        object.__setattr__(self, "probabilities", values)

    @property
    def label(self):
        # A probability tie within an expert chooses the lowest class index.
        return max(range(3), key=lambda i: self.probabilities[i])

    @property
    def confidence(self):
        return max(self.probabilities)

    @property
    def entropy(self):
        return -sum(p * math.log(p) for p in self.probabilities if p > 0)

    def to_dict(self):
        return {"probabilities": list(self.probabilities), "label": self.label,
                "confidence": self.confidence, "entropy": self.entropy}


def needs_auxiliary(pathology, policy):
    return pathology.entropy >= policy["entropy_threshold"]


def arbitrate(pathology, ct, gastro, policy):
    """Retain pathology unless an available auxiliary expert meets an override rule.

    Gate: H(pathology) >= tau_H. Select the most confident available auxiliary.
    Recall rule: base != 1, auxiliary == 1, confidence > tau_rec, margin > delta.
    Benign rule: base == 1, auxiliary == 2, confidence > tau_benign.
    Class 0 may therefore change to Class 1 after the entropy gate opens.
    """
    result = {"baseline": pathology.label, "final": pathology.label,
              "pathology_entropy": pathology.entropy, "gate_open": False,
              "selected_auxiliary": None, "margin": None,
              "reason": "low_entropy", "changed": False}
    if not needs_auxiliary(pathology, policy):
        return result
    result["gate_open"] = True
    available = [(name, pred) for name, pred in (("ct", ct), ("gastro", gastro))
                 if pred is not None]
    if not available:
        result["reason"] = "no_auxiliary_available"
        return result
    source, auxiliary = max(available, key=lambda pair: (
        pair[1].confidence, pair[0] == policy["tie_priority"]))
    margin = auxiliary.confidence - pathology.confidence
    result.update(selected_auxiliary=source, margin=margin, reason="retain_pathology")
    if auxiliary.label == 1 and pathology.label != 1:
        if auxiliary.confidence > policy["recall_threshold"] and margin > policy["margin"]:
            result.update(final=1, reason="recall_override")
        else:
            result["reason"] = "recall_conditions_not_met"
    elif pathology.label == 1 and auxiliary.label == 2:
        if auxiliary.confidence > policy["benign_threshold"]:
            result.update(final=2, reason="benign_override")
        else:
            result["reason"] = "benign_threshold_not_met"
    result["changed"] = result["final"] != result["baseline"]
    return result
