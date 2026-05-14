from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATOSYORI = ROOT / "external" / "atosyori-pipeline-dev"


class PostprocessVendorContractTests(unittest.TestCase):
    def test_vendored_engine_uses_validated_legacy_entrypoint(self) -> None:
        settings = ATOSYORI / "src" / "atosyori_postprocess" / "settings.py"
        text = settings.read_text(encoding="utf-8")

        self.assertIn('LEGACY_ENGINE = PACKAGE_ROOT / "legacy" / "run_standalone.py"', text)

    def test_raw_jsonl_audit_contract_is_present(self) -> None:
        engine = ATOSYORI / "src" / "atosyori_postprocess" / "legacy" / "run_standalone.py"
        text = engine.read_text(encoding="utf-8")

        for expected in (
            "CREATE TABLE raw_tracked_masks",
            "CREATE TABLE raw_tracks",
            "bbox_xyxy_json",
            "raw_tracked_rows",
            "raw_tracks",
            "raw_removed_rows",
            "raw_det_score_min",
        ):
            self.assertIn(expected, text)

    def test_local_vendor_patch_notes_exist(self) -> None:
        notes = ATOSYORI / "docs" / "local_patches.md"
        text = notes.read_text(encoding="utf-8")

        self.assertIn("raw_tracked_masks", text)
        self.assertIn("frame_index", text)
        self.assertIn("frame_idx", text)


if __name__ == "__main__":
    unittest.main()
