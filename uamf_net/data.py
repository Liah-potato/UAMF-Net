"""Manifest-based pairing with explicit image and patient split validation."""

import csv
from pathlib import Path

REQUIRED = {"sample_id", "patient_id", "split", "label", "pathology", "ct", "gastro"}


def read_manifest(path, data_root, check_files=True):
    root = Path(data_root).resolve()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no samples")
    samples, patient_splits, patient_labels, image_owners = set(), {}, {}, {}
    pathology_images = set()
    for row in rows:
        for key in REQUIRED:
            row[key] = (row[key] or "").strip()
        sample, patient, split = row["sample_id"], row["patient_id"], row["split"]
        if not sample or sample in samples or not patient:
            raise ValueError("sample_id must be unique and patient_id must be nonempty")
        samples.add(sample)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split for {sample}: {split}")
        if patient in patient_splits and patient_splits[patient] != split:
            raise ValueError(f"Patient {patient} occurs in multiple splits")
        patient_splits[patient] = split
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1, 2):
            raise ValueError(f"Invalid label for {sample}")
        if patient in patient_labels and patient_labels[patient] != row["label"]:
            raise ValueError(f"Conflicting reference labels for patient {patient}")
        patient_labels[patient] = row["label"]
        if not row["pathology"]:
            raise ValueError(f"Pathology image is required for {sample}")
        for modality in ("pathology", "ct", "gastro"):
            raw = row[modality]
            if not raw:
                row[modality] = None
                continue
            image = (root / raw).resolve()
            if modality == "pathology":
                if image in pathology_images:
                    raise ValueError(f"Repeated pathology image would duplicate evaluation: {image}")
                pathology_images.add(image)
            owner = image_owners.get(image)
            if owner is not None and owner != (patient, split, modality):
                raise ValueError(f"Image reused across patients, splits, or modalities: {image}")
            image_owners[image] = (patient, split, modality)
            if check_files and not image.is_file():
                raise FileNotFoundError(f"Missing {modality} image for {sample}: {image}")
            row[modality] = str(image)
    return rows


def modality_records(rows, split, modality):
    # Deduplicate auxiliary images paired to several pathology patches.
    seen, selected = set(), []
    for row in rows:
        path = row[modality]
        if row["split"] == split and path and path not in seen:
            selected.append(row)
            seen.add(path)
    return selected
