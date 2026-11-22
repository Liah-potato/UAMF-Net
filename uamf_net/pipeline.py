"""Gate auxiliary image inference and export one complete trace per input row."""

import json
from pathlib import Path

from .config import write_json
from .fusion import arbitrate, needs_auxiliary
from .metrics import summarize


def evaluate(config, rows, split, checkpoint_dir, output_dir, device, backend=None):
    selected = [r for r in rows if r["split"] == split]
    if not selected:
        raise ValueError(f"No manifest rows in split {split}")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    if backend is None:
        from .models import ImageExperts
        backend = ImageExperts(config, checkpoint_dir, device)
    experts = backend
    experts.load("pathology")
    records = []
    for row in selected:
        pathology = experts.predict("pathology", row["pathology"])
        ct, gastro = None, None
        # Validation exports all available expert outputs so policies can be tuned
        # without re-running image models. Test inference remains gated.
        if split == "val" or needs_auxiliary(pathology, config["policy"]):
            ct = experts.predict("ct", row["ct"])
            gastro = experts.predict("gastro", row["gastro"])
        decision = arbitrate(pathology, ct, gastro, config["policy"])
        records.append({"sample_id": row["sample_id"], "patient_id": row["patient_id"],
                        "split": split, "label": row["label"], **decision,
                        "auxiliary_collection": "all_available" if split == "val" else "gated",
                        "experts": {name: pred.to_dict() if pred is not None else None
                                    for name, pred in (("pathology", pathology), ("ct", ct), ("gastro", gastro))}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    summary = summarize(records)
    summary["split"] = split
    summary["auxiliary_call_mode"] = "all_available_for_validation" if split == "val" else "entropy_gated"
    write_json(output / "metrics.json", summary)
    write_json(output / "config.json", config)
    return summary
