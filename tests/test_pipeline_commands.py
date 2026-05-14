from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import infer_video_postprocess  # noqa: E402
import flow_cli_common  # noqa: E402
import run_full_flow  # noqa: E402
import pipeline_commands  # noqa: E402
import run_postprocess_only  # noqa: E402
import run_integrated_pipeline  # noqa: E402
from backend.detectors.dinov3.commands import build_command as build_dinov3_adapter_command  # noqa: E402
from backend.detectors.eva02.commands import build_command as build_eva02_adapter_command  # noqa: E402
from backend.postprocess.commands import build_run_command as build_atosyori_adapter_command  # noqa: E402
from pipeline_defaults import (  # noqa: E402
    DINO_DEFAULT_BATCH_SIZE,
    DINO_DEFAULT_WARMUP_FRAMES,
    EVA02_DEFAULT_BATCH_SIZE,
    EVA02_DEFAULT_WARMUP_FRAMES,
)


class PipelineCommandTests(unittest.TestCase):
    def test_thin_wrapper_keeps_dinov3_production_defaults(self) -> None:
        args = argparse.Namespace(
            input="input/sample.mp4",
            output_root=ROOT / "output" / "runs",
            run_name="unit",
            recursive=False,
            force=True,
            overlay=False,
            detector="dinov3",
            classifier=True,
            ellipse_only=False,
            max_frames=12,
            warmup_frames=None,
            batch_size=None,
            extra_args=[],
        )

        command = infer_video_postprocess.build_command(args)

        self.assertIn("--postprocess", command)
        self.assertIn("--no-render-overlays", command)
        self.assertEqual(command[command.index("--warmup-frames") + 1], str(DINO_DEFAULT_WARMUP_FRAMES))
        self.assertEqual(command[command.index("--batch-size") + 1], str(DINO_DEFAULT_BATCH_SIZE))
        self.assertEqual(command[command.index("--detector") + 1], "dinov3")
        self.assertEqual(command[command.index("--max-frames") + 1], "12")

    def test_thin_wrapper_maps_eva02_short_options(self) -> None:
        args = argparse.Namespace(
            input="input/sample.mp4",
            output_root=ROOT / "output" / "runs",
            run_name=None,
            recursive=True,
            force=False,
            overlay=True,
            detector="eva02",
            classifier=False,
            ellipse_only=True,
            max_frames=None,
            warmup_frames=3,
            batch_size=4,
            extra_args=["--", "--eva02-score-thresh", "0.25"],
        )

        command = infer_video_postprocess.build_command(args)

        self.assertIn("--no-classifier", command)
        self.assertIn("--render-overlays", command)
        self.assertIn("--recursive", command)
        self.assertEqual(command[command.index("--eva02-warmup-frames") + 1], "3")
        self.assertEqual(command[command.index("--eva02-batch-size") + 1], "4")
        self.assertEqual(command[command.index("--eva02-score-thresh") + 1], "0.25")
        self.assertIn(str(ROOT / "configs" / "class_policy_ellipse_only.json"), command)

    def test_detailed_entrypoint_defaults_match_expected_detector_commands(self) -> None:
        args = run_integrated_pipeline.normalize_args(
            run_integrated_pipeline.build_parser().parse_args(
                [
                    "--input",
                    "input/sample.mp4",
                    "--detector",
                    "eva02",
                    "--max-frames",
                    "1",
                    "--no-postprocess",
                ]
            )
        )

        command = pipeline_commands.build_eva02_command(args, args.input, ROOT / "out" / "eva02")

        self.assertEqual(command[0], str(args.python))
        self.assertIn("infer_video_eva02_jsonl.py", command[1])
        self.assertEqual(command[command.index("--target-size") + 1], "1280")
        self.assertEqual(command[command.index("--score-thresh") + 1], "0.1")
        self.assertEqual(command[command.index("--batch-size") + 1], str(EVA02_DEFAULT_BATCH_SIZE))
        self.assertEqual(command[command.index("--warmup-frames") + 1], str(EVA02_DEFAULT_WARMUP_FRAMES))
        self.assertEqual(command[command.index("--json-backend") + 1], "json")
        self.assertEqual(command[command.index("--mask-approx") + 1], "simple")
        self.assertIn("--no-async-writer", command)
        self.assertEqual(command[command.index("--max-frames") + 1], "1")

    def test_detector_adapters_are_pipeline_command_source(self) -> None:
        args = run_integrated_pipeline.normalize_args(
            run_integrated_pipeline.build_parser().parse_args(
                [
                    "--input",
                    "input/sample.mp4",
                    "--max-frames",
                    "1",
                    "--no-postprocess",
                ]
            )
        )

        out_dir = ROOT / "out" / "dinov3"
        self.assertEqual(
            pipeline_commands.build_dinov3_command(args, args.input, out_dir),
            build_dinov3_adapter_command(args, args.input, out_dir),
        )

        args.detector = "eva02"
        out_dir = ROOT / "out" / "eva02"
        self.assertEqual(
            pipeline_commands.build_eva02_command(args, args.input, out_dir),
            build_eva02_adapter_command(args, args.input, out_dir),
        )

    def test_generated_policy_supports_class_overrides(self) -> None:
        policy = flow_cli_common.build_policy_dict(
            shape_mode="ellipse",
            keyframe_interval=3,
            recall_target=0.96,
            class_policy=["female:polygon:5:0.97", "junction:polygon:4:0.98"],
        )

        self.assertEqual(policy["default"]["shape_mode"], "ellipse")
        self.assertEqual(policy["classes"]["女性器"]["shape_mode"], "polygon")
        self.assertEqual(policy["classes"]["女性器"]["target_interval"], 5)
        self.assertEqual(policy["classes"]["女性器"]["polygon_recall_min"], 0.97)
        self.assertEqual(policy["classes"]["結合部分"]["shape_mode"], "polygon")
        self.assertEqual(policy["classes"]["結合"]["target_interval"], 4)

    def test_full_flow_wrapper_exposes_clean_postprocess_knobs(self) -> None:
        args = argparse.Namespace(
            input="input/sample.mp4",
            output_root=ROOT / "output" / "runs",
            run_name="unit_full",
            recursive=False,
            force=True,
            dry_run=False,
            detector="eva02",
            postprocess=True,
            overlay=True,
            overlay_encoder="cpu",
            shape_mode="polygon",
            keyframe_interval=5,
            recall_target=0.97,
            class_policy=[],
            class_policy_json=None,
            max_frames=10,
            batch_size=1,
            warmup_frames=0,
            score_thresh=0.2,
            raw_cut_detect=True,
            short_track_max_frames=10,
            extra_args=[],
        )

        command = run_full_flow.build_command(
            args,
            ROOT / "output" / "runs" / "unit_full",
            ROOT / "output" / "runs" / "unit_full" / "config" / "class_policy.generated.json",
        )

        self.assertIn("--postprocess", command)
        self.assertIn("--render-overlays", command)
        self.assertEqual(command[command.index("--detector") + 1], "eva02")
        self.assertEqual(command[command.index("--eva02-batch-size") + 1], "1")
        self.assertEqual(command[command.index("--eva02-score-thresh") + 1], "0.2")
        self.assertEqual(command[command.index("--intervals") + 1], "5")
        self.assertEqual(command[command.index("--default-shape-mode") + 1], "polygon")
        self.assertEqual(command[command.index("--polygon-recall-min") + 1], "0.97")

    def test_postprocess_only_wrapper_builds_atosyori_command(self) -> None:
        args = argparse.Namespace(
            input_jsonl=ROOT / "out" / "ai.jsonl",
            input_sqlite=None,
            input_video=ROOT / "input" / "sample.mp4",
            output_root=ROOT / "output" / "postprocess_runs",
            run_name="unit_post",
            output_dir=ROOT / "output" / "postprocess_runs" / "unit_post" / "postprocess",
            model_root=ROOT / "checkpoints" / "postprocess",
            atosyori_repo=ROOT / "external" / "atosyori-pipeline-dev",
            python=Path(sys.executable),
            force=True,
            dry_run=False,
            overlay=False,
            overlay_encoder="cpu",
            shape_mode="ellipse",
            keyframe_interval=3,
            recall_target=0.96,
            class_policy=[],
            class_policy_json=None,
            raw_cut_detect=True,
            short_track_max_frames=10,
            raw_det_score_min=0.35,
            k2_device="auto",
            polygon_predictor_device="auto",
            extra_args=[],
        )

        command = run_postprocess_only.build_command(
            args,
            args.output_dir,
            ROOT / "output" / "postprocess_runs" / "unit_post" / "config" / "class_policy.generated.json",
        )

        self.assertEqual(command[1:4], ["-m", "atosyori_postprocess", "run"])
        self.assertIn("--input-jsonl", command)
        self.assertIn("--input-video", command)
        self.assertIn("--no-render-overlays", command)
        self.assertEqual(command[command.index("--intervals") + 1], "3")
        self.assertIn("--", command)
        self.assertEqual(command[command.index("--dense-recall-target") + 1], "0.96")
        self.assertEqual(command[command.index("--raw-det-score-min") + 1], "0.35")

        adapter_command = build_atosyori_adapter_command(
            python=args.python,
            atosyori_repo=args.atosyori_repo,
            input_jsonl=args.input_jsonl,
            input_video=args.input_video,
            output_dir=args.output_dir,
            model_root=args.model_root,
            intervals=args.keyframe_interval,
            default_shape_mode=args.shape_mode,
            class_policy_json=ROOT / "output" / "postprocess_runs" / "unit_post" / "config" / "class_policy.generated.json",
            k2_device=args.k2_device,
            polygon_predictor_device=args.polygon_predictor_device,
            render_overlays=args.overlay,
            force=args.force,
            engine_args=[
                "--overlay-encoder",
                "cpu",
                "--raw-remove-short-tracks-max-frames",
                "10",
                "--raw-cut-detect",
                "--raw-det-score-min",
                "0.35",
                "--dense-recall-target",
                "0.96",
                "--polygon-recall-min",
                "0.96",
            ],
        )
        self.assertEqual(command, adapter_command)


if __name__ == "__main__":
    unittest.main()
