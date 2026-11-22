"""Hand-constructed probability examples; no images, labels, or clinical metrics."""

from .fusion import Prediction, arbitrate


def run_demo(config):
    examples = [
        ("confident_pathology", (.98, .01, .01), None, None),
        ("ct_recall_override", (.42, .20, .38), (.05, .90, .05), (.25, .35, .40)),
        ("gastro_benign_override", (.20, .60, .20), (.20, .30, .50), (.03, .05, .92)),
        ("insufficient_margin", (.25, .20, .55), None, (.16, .68, .16)),
        ("missing_auxiliary", (.40, .35, .25), None, None),
        ("ct_wins_confidence_tie", (.25, .50, .25), (.05, .05, .90), (.05, .90, .05)),
    ]
    records = []
    for name, pathology, ct, gastro in examples:
        predictions = [Prediction(tuple(p)) if p is not None else None for p in (pathology, ct, gastro)]
        records.append({"example": name, "synthetic": True,
                        "experts": {key: value.to_dict() if value else None
                                    for key, value in zip(("pathology", "ct", "gastro"), predictions)},
                        **arbitrate(*predictions, config["policy"])})
    return {"kind": "synthetic_probability_demo", "clinical_metrics": None, "decisions": records}
