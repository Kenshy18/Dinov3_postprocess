from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from backend.pipeline import pipeline_defaults
from backend.pipeline import pipeline_commands
from backend.pipeline.cli import run_postprocess_only


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_detector_runtime_defaults_live_under_backend_detectors(self) -> None:
        dinov3_runtime = pipeline_defaults.DEFAULT_DINOV3_RUNTIME
        eva02_runtime = pipeline_defaults.DEFAULT_EVA02_RUNTIME

        self.assertEqual(
            dinov3_runtime,
            ROOT / "backend" / "detectors" / "dinov3" / "runtime",
        )
        self.assertEqual(
            eva02_runtime,
            ROOT / "backend" / "detectors" / "eva02" / "runtime",
        )
        self.assertTrue((dinov3_runtime / "infer_video_dinov3_jsonl.py").is_file())
        self.assertTrue((eva02_runtime / "infer_video_eva02_jsonl.py").is_file())

    def test_legacy_inference_paths_are_wrappers_only(self) -> None:
        wrappers = [
            ROOT / "inference" / "dinov3_video_jsonl_runtime" / "infer_video_dinov3_jsonl.py",
            ROOT / "inference" / "dinov3_video_jsonl_runtime" / "infer_images_singleclass.py",
            ROOT / "inference" / "eva02_video_jsonl_runtime" / "infer_video_eva02_jsonl.py",
        ]

        for wrapper in wrappers:
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn("Compatibility wrapper", text)
            self.assertIn("backend.detectors", text)
            self.assertLess(len(text.splitlines()), 30)

    def test_pipeline_uses_backend_postprocess_adapter(self) -> None:
        pipeline_source = inspect.getsource(pipeline_commands)
        postprocess_only_source = inspect.getsource(run_postprocess_only)

        self.assertIn("from backend.postprocess.commands import", pipeline_source)
        self.assertIn("from backend.postprocess.commands import", postprocess_only_source)
        self.assertNotIn('"atosyori_postprocess",\n        "run"', pipeline_source)
        self.assertNotIn('"atosyori_postprocess",\n        "run"', postprocess_only_source)

    def test_stable_compatibility_roots_are_documented(self) -> None:
        architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        for expected in (
            "backend/detectors/dinov3",
            "backend/detectors/eva02",
            "backend/classifiers",
            "backend/schemas",
            "backend/postprocess",
            ".runtime",
            "inference/dinov3_video_jsonl_runtime/* -> backend.detectors.dinov3.runtime.*",
            "inference/eva02_video_jsonl_runtime/* -> backend.detectors.eva02.runtime.*",
        ):
            self.assertIn(expected, architecture)

    def test_classifier_implementations_live_under_backend_classifiers(self) -> None:
        dinov3_classifier = ROOT / "backend" / "classifiers" / "dinov3_roi" / "runtime" / "two_stage_roi_classifier.py"
        eva02_classifier = ROOT / "backend" / "classifiers" / "eva02_roi" / "runtime" / "two_stage_roi_classifier.py"
        self.assertGreater(len(dinov3_classifier.read_text(encoding="utf-8").splitlines()), 1000)
        self.assertGreater(len(eva02_classifier.read_text(encoding="utf-8").splitlines()), 1000)

        for wrapper in (
            ROOT
            / "backend"
            / "detectors"
            / "dinov3"
            / "runtime"
            / "two_stage_multiclass_20260426"
            / "scripts"
            / "two_stage_roi_classifier.py",
            ROOT / "backend" / "detectors" / "eva02" / "runtime" / "two_stage_roi_classifier.py",
            ROOT
            / "inference"
            / "dinov3_video_jsonl_runtime"
            / "two_stage_multiclass_20260426"
            / "scripts"
            / "two_stage_roi_classifier.py",
            ROOT / "inference" / "eva02_video_jsonl_runtime" / "two_stage_roi_classifier.py",
        ):
            text = wrapper.read_text(encoding="utf-8")
            self.assertIn("Compatibility wrapper", text)
            self.assertIn("backend.classifiers", text)
            self.assertLess(len(text.splitlines()), 35)

    def test_local_runtime_state_is_separate_from_configs(self) -> None:
        self.assertEqual(pipeline_defaults.DEFAULT_RUNTIME_PROFILE, ROOT / ".runtime" / "runtime_profile.json")
        self.assertEqual(pipeline_defaults.LEGACY_RUNTIME_PROFILE, ROOT / "configs" / "runtime_profile.json")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".runtime/*", gitignore)
        self.assertTrue((ROOT / "configs" / "runtime_profile.example.json").is_file())


if __name__ == "__main__":
    unittest.main()
