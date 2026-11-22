"""Three-class image-level metrics and counts of beneficial/harmful overrides."""

from collections import Counter


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Nonempty, equally sized labels and predictions are required")
    matrix = [[0] * 3 for _ in range(3)]
    for truth, prediction in zip(labels, predictions):
        if truth not in range(3) or prediction not in range(3):
            raise ValueError("Labels and predictions must be in {0, 1, 2}")
        matrix[truth][prediction] += 1
    classes = []
    for c in range(3):
        tp = matrix[c][c]
        fn = sum(matrix[c]) - tp
        fp = sum(row[c] for row in matrix) - tp
        tn = len(labels) - tp - fn - fp
        precision, recall = divide(tp, tp + fp), divide(tp, tp + fn)
        classes.append({"class": c, "support": tp + fn, "precision": precision,
                        "recall": recall, "specificity": divide(tn, tn + fp),
                        "f1": divide(2 * precision * recall, precision + recall)})
    return {"n": len(labels), "accuracy": sum(matrix[c][c] for c in range(3)) / len(labels),
            "macro_f1": sum(c["f1"] for c in classes) / 3,
            "per_class": classes, "confusion_matrix": matrix}


def summarize(records):
    labels = [r["label"] for r in records]
    baseline = classification_metrics(labels, [r["baseline"] for r in records])
    final = classification_metrics(labels, [r["final"] for r in records])
    corrected = sum(r["baseline"] != r["label"] and r["final"] == r["label"] for r in records)
    harmed = sum(r["baseline"] == r["label"] and r["final"] != r["label"] for r in records)
    counts = Counter(r["selected_auxiliary"] for r in records if r["selected_auxiliary"])
    contributions = {}
    for source in ("ct", "gastro"):
        selected = [r for r in records if r["selected_auxiliary"] == source]
        contributions[source] = {
            "selected": counts[source],
            "corrected": sum(r["baseline"] != r["label"] and r["final"] == r["label"] for r in selected),
            "harmed": sum(r["baseline"] == r["label"] and r["final"] != r["label"] for r in selected),
        }
    return {"unit": "pathology_image", "patients": len({r["patient_id"] for r in records}),
            "baseline": baseline, "final": final, "corrected": corrected, "harmed": harmed,
            "net_correct": corrected - harmed,
            "accuracy_gain_percentage_points": 100 * (final["accuracy"] - baseline["accuracy"]),
            "gate_open_count": sum(r["gate_open"] for r in records),
            "override_count": sum(r["changed"] for r in records),
            "auxiliary_calls": {name: sum(r["experts"][name] is not None for r in records)
                                for name in ("ct", "gastro")},
            "auxiliary_contributions": contributions}
