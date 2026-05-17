from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tools.setup import benchmark_runtime_batches as bench


class RuntimeBatchBenchmarkTests(unittest.TestCase):
    def test_larger_batch_wins_when_metric_is_within_tie_ratio(self) -> None:
        selected = {"batch_size": 4, "metric_fps": 100.0}
        candidate = {"batch_size": 8, "metric_fps": 98.0}

        should_select, reason = bench.should_select_candidate(candidate, selected, tie_fps_ratio=0.03)

        self.assertTrue(should_select)
        self.assertIn("larger_batch", reason)

    def test_larger_batch_does_not_win_when_metric_drop_is_not_close(self) -> None:
        selected = {"batch_size": 4, "metric_fps": 100.0}
        candidate = {"batch_size": 8, "metric_fps": 94.0}

        should_select, reason = bench.should_select_candidate(candidate, selected, tie_fps_ratio=0.03)

        self.assertFalse(should_select)
        self.assertEqual(reason, "lower_metric_fps")

    def test_eva02_benchmark_command_matches_production_runtime_shape(self) -> None:
        command = bench.command_for_candidate(
            detector="eva02",
            python=Path("/venv/bin/python"),
            input_video=Path("/tmp/input.mp4"),
            output_dir=Path("/tmp/out"),
            batch=8,
            frames=240,
            engine=Path("/tmp/dinov3.engine"),
            classifier_batch_size=4096,
            eva02_compile_backbone="max-autotune",
        )

        self.assertEqual(command[command.index("--compile-backbone") + 1], "max-autotune")
        self.assertEqual(command[command.index("--json-backend") + 1], "json")
        self.assertIn("--no-async-writer", command)
        self.assertNotIn("--async-writer", command)

    def test_dinov3_benchmark_uses_production_mask_approximation(self) -> None:
        command = bench.command_for_candidate(
            detector="dinov3",
            python=Path("/venv/bin/python"),
            input_video=Path("/tmp/input.mp4"),
            output_dir=Path("/tmp/out"),
            batch=8,
            frames=240,
            engine=Path("/tmp/dinov3.engine"),
            classifier_batch_size=4096,
            eva02_compile_backbone="max-autotune",
        )

        self.assertEqual(command[command.index("--mask-approx") + 1], "none")

    def test_parser_defaults_are_longer_and_match_eva02_production_compile(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "BATCH_BENCHMARK_FRAMES": "240",
                "BATCH_BENCHMARK_INPUT": "auto",
                "EVA02_BENCHMARK_COMPILE_BACKBONE": "max-autotune",
                "BATCH_BENCHMARK_TIE_FPS_RATIO": "0.03",
            },
        ):
            parser = bench.build_parser()
            args = parser.parse_args(["--detectors", "eva02"])

        self.assertEqual(args.frames, 240)
        self.assertEqual(args.input_video, "auto")
        self.assertEqual(args.eva02_compile_backbone, "max-autotune")
        self.assertGreater(args.tie_fps_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
