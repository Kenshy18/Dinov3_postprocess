from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from atosyori_postprocess.pipeline import build_command


class PipelineCommandTests(unittest.TestCase):
    def test_full_pipeline_command_adds_model_paths(self) -> None:
        args = Namespace(
            input_sqlite=Path("input/a.sqlite"),
            input_jsonl=None,
            input_video=None,
            output_dir=Path("output/a"),
            intervals="3,6",
            default_shape_mode="polygon",
            class_policy_json=None,
            model_root=Path("models-local"),
            k2_run_dir=None,
            polygon_point_predictor_model_dir=None,
            k2_device="cpu",
            polygon_predictor_device="cpu",
            render_overlays=False,
            force=True,
            engine_args=[],
        )

        command = build_command(args)

        self.assertIn("--input-sqlite", command)
        self.assertIn("input/a.sqlite", command)
        self.assertIn("--k2-run-dir", command)
        self.assertIn(str(Path("models-local").resolve() / "k2_v5"), command)
        self.assertIn("--polygon-point-predictor-model-dir", command)
        self.assertIn(str(Path("models-local").resolve() / "polygon_point_predictor"), command)
        self.assertIn("--default-shape-mode", command)
        self.assertIn("polygon", command)


if __name__ == "__main__":
    unittest.main()
