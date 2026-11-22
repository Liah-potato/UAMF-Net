"""Synthetic checks of routing, evaluation accounting, and data isolation."""

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from uamf_net.config import load_config
from uamf_net.data import modality_records, read_manifest
from uamf_net.demo import run_demo
from uamf_net.fusion import Prediction, arbitrate, needs_auxiliary
from uamf_net.metrics import classification_metrics, summarize
from uamf_net.pipeline import evaluate
from uamf_net.tuning import tune_policy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/reference.json")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = copy.deepcopy(CONFIG["policy"])

    def test_invalid_probabilities_fail(self):
        for values in ((.33, .33, .33), (float("nan"), 0, 1), (-.1, .2, .9)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Prediction(values)

    def test_entropy_and_boundary(self):
        uniform = Prediction((1 / 3, 1 / 3, 1 / 3))
        self.assertAlmostEqual(uniform.entropy, math.log(3))
        self.policy["entropy_threshold"] = uniform.entropy
        self.assertTrue(needs_auxiliary(uniform, self.policy))

    def test_low_entropy_bypasses_auxiliary(self):
        result = arbitrate(Prediction((.98, .01, .01)), None, Prediction((.01, .98, .01)), self.policy)
        self.assertFalse(result["gate_open"])
        self.assertEqual(result["final"], 0)

    def test_recall_can_override_class_zero(self):
        result = arbitrate(Prediction((.42, .20, .38)), Prediction((.05, .90, .05)), None, self.policy)
        self.assertEqual(result["final"], 1)
        self.assertEqual(result["selected_auxiliary"], "ct")

    def test_margin_is_required(self):
        result = arbitrate(Prediction((.25, .20, .55)), None, Prediction((.16, .68, .16)), self.policy)
        self.assertEqual(result["final"], 2)
        self.assertEqual(result["reason"], "recall_conditions_not_met")

    def test_benign_threshold_is_strict(self):
        base = Prediction((.20, .60, .20))
        at_boundary = arbitrate(base, None, Prediction((.1, .1, .8)), self.policy)
        above = arbitrate(base, None, Prediction((.05, .05, .9)), self.policy)
        self.assertEqual(at_boundary["final"], 1)
        self.assertEqual(above["final"], 2)

    def test_no_direct_class_zero_to_benign(self):
        result = arbitrate(Prediction((.45, .30, .25)), None, Prediction((.02, .03, .95)), self.policy)
        self.assertEqual(result["final"], 0)

    def test_missing_auxiliary_is_not_a_uniform_expert(self):
        result = arbitrate(Prediction((.40, .35, .25)), None, None, self.policy)
        self.assertEqual(result["reason"], "no_auxiliary_available")
        self.assertIsNone(result["selected_auxiliary"])

    def test_confidence_tie_has_explicit_priority(self):
        result = arbitrate(Prediction((.25, .50, .25)), Prediction((.05, .05, .90)),
                           Prediction((.05, .90, .05)), self.policy)
        self.assertEqual(result["selected_auxiliary"], "ct")
        self.assertEqual(result["final"], 2)

    def test_demo_does_not_report_clinical_accuracy(self):
        demo = run_demo(CONFIG)
        self.assertIsNone(demo["clinical_metrics"])
        self.assertEqual(len(demo["decisions"]), 6)
        self.assertTrue(all(r["synthetic"] for r in demo["decisions"]))


class ManifestTests(unittest.TestCase):
    def manifest(self, root, rows):
        path = root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "patient_id", "split", "label", "pathology", "ct", "gastro"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, **changes):
        return {"sample_id": "a", "patient_id": "p1", "split": "train", "label": "1",
                "pathology": "a.png", "ct": "shared.png", "gastro": "", **changes}

    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.manifest(root, [self.row(), self.row(sample_id="b", split="test", pathology="b.png")])
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_manifest(path, root, check_files=False)

    def test_image_reused_by_different_patients_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.manifest(root, [self.row(), self.row(sample_id="b", patient_id="p2", pathology="b.png")])
            with self.assertRaisesRegex(ValueError, "Image reused"):
                read_manifest(path, root, check_files=False)

    def test_repeated_pathology_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.manifest(root, [self.row(), self.row(sample_id="b")])
            with self.assertRaisesRegex(ValueError, "Repeated pathology"):
                read_manifest(path, root, check_files=False)

    def test_auxiliary_image_is_deduplicated_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.manifest(root, [self.row(), self.row(sample_id="b", pathology="b.png")])
            rows = read_manifest(path, root, check_files=False)
            self.assertEqual(len(modality_records(rows, "train", "ct")), 1)
            self.assertEqual(len(modality_records(rows, "train", "pathology")), 2)

    def test_declared_file_missing_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.manifest(root, [self.row()])
            with self.assertRaises(FileNotFoundError):
                read_manifest(path, root)


class FakeExperts:
    def __init__(self):
        self.calls = []

    def load(self, modality):
        pass

    def predict(self, modality, path):
        self.calls.append((modality, path))
        if path is None:
            return None
        values = {"easy": (.98, .01, .01), "hard": (.42, .20, .38), "aux": (.05, .90, .05)}
        return Prediction(values[path])


class PipelineTests(unittest.TestCase):
    def rows(self, split):
        return [{"sample_id": "a", "patient_id": "p1", "split": split, "label": 0,
                 "pathology": "easy", "ct": "aux", "gastro": None},
                {"sample_id": "b", "patient_id": "p2", "split": split, "label": 1,
                 "pathology": "hard", "ct": "aux", "gastro": None}]

    def test_test_routing_and_complete_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeExperts()
            result = evaluate(CONFIG, self.rows("test"), "test", tmp, Path(tmp) / "out", "cpu", backend)
            self.assertEqual(backend.calls.count(("ct", "aux")), 1)
            self.assertEqual(result["corrected"], 1)
            self.assertEqual(result["net_correct"], 1)
            self.assertEqual(result["override_count"], 1)
            records = (Path(tmp) / "out" / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 2)

    def test_validation_exports_auxiliaries_and_tuning_rejects_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = FakeExperts()
            evaluate(CONFIG, self.rows("val"), "val", tmp, root / "val", "cpu", backend)
            self.assertEqual(backend.calls.count(("ct", "aux")), 2)
            selected, report = tune_policy(CONFIG, root / "val" / "predictions.jsonl")
            self.assertEqual(len(report["candidates"]), 72)
            self.assertIn("policy", selected)
            evaluate(CONFIG, self.rows("test"), "test", tmp, root / "test", "cpu", FakeExperts())
            with self.assertRaisesRegex(ValueError, "validation-only"):
                tune_policy(CONFIG, root / "test" / "predictions.jsonl")

    def test_metrics_count_harm_separately(self):
        records = []
        for base, final, truth in ((0, 1, 1), (1, 2, 1), (2, 1, 0)):
            records.append({"baseline": base, "final": final, "label": truth, "patient_id": str(len(records)),
                            "selected_auxiliary": "ct", "gate_open": True, "changed": True,
                            "experts": {"ct": {}, "gastro": None}})
        summary = summarize(records)
        self.assertEqual((summary["corrected"], summary["harmed"], summary["net_correct"]), (1, 1, 0))
        metric = classification_metrics([0, 1, 2], [0, 1, 1])
        self.assertEqual(metric["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])


if __name__ == "__main__":
    unittest.main()
