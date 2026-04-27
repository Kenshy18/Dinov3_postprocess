from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atosyori_postprocess.stages.common import with_ellipse_defaults, with_polygon_defaults


class StageDefaultTests(unittest.TestCase):
    def test_ellipse_defaults_add_k2_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = with_ellipse_defaults(["--input-sqlite", "x.sqlite"], root)
            self.assertIn("--k2-run-dir", args)
            self.assertIn(str(root.resolve() / "k2_v5"), args)

    def test_polygon_defaults_add_predictor_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = with_polygon_defaults(["--input-sqlite", "x.sqlite"], root)
            self.assertIn("--point-predictor-model-dir", args)
            self.assertIn(str(root.resolve() / "polygon_point_predictor"), args)

    def test_existing_options_are_preserved(self) -> None:
        args = with_polygon_defaults(
            ["--point-predictor-model-dir", "custom", "--predictor-device", "cpu"],
            None,
        )
        self.assertEqual(args.count("--point-predictor-model-dir"), 1)
        self.assertIn("custom", args)
        self.assertEqual(args.count("--predictor-device"), 1)


if __name__ == "__main__":
    unittest.main()
