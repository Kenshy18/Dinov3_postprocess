from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.pipeline.pipeline_outputs import summarize_detector
from backend.schemas.detection_jsonl import (
    DetectionJsonlContractError,
    normalize_frame_record,
    summarize_detection_jsonl,
)


class DetectionJsonlContractTests(unittest.TestCase):
    def test_dinov3_record_normalizes_to_common_contract(self) -> None:
        record = normalize_frame_record(
            {
                "frame_index": 7,
                "time_sec": 0.25,
                "width": 1920,
                "height": 1080,
                "detections": [
                    {
                        "label": "女性器",
                        "class_name": "女性器",
                        "category_id": 1,
                        "category_index": 0,
                        "score": 0.91,
                        "detector_score": 0.86,
                        "class_score": 0.97,
                        "bbox_xyxy": [10, 20, 40, 60],
                        "polygons": [[10, 20, 40, 20, 40, 60, 10, 60]],
                    }
                ],
            }
        )

        self.assertEqual(record["frame_idx"], 7)
        self.assertEqual(record["detections"][0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(record["instances"], record["detections"])

    def test_eva02_record_normalizes_to_common_contract(self) -> None:
        record = normalize_frame_record(
            {
                "frame_idx": 3,
                "width": 1280,
                "height": 720,
                "instances": [
                    {
                        "class_name": "男性器",
                        "category_id": 2,
                        "score": 0.74,
                        "cls_score": 0.88,
                        "bbox": [100, 120, 50, 70],
                        "segmentation": [[100, 120, 150, 120, 150, 190, 100, 190]],
                    }
                ],
            }
        )

        det = record["detections"][0]
        self.assertEqual(record["frame_index"], 3)
        self.assertEqual(det["class_score"], 0.88)
        self.assertEqual(det["bbox_xyxy"], [100.0, 120.0, 150.0, 190.0])

    def test_codino_record_normalizes_to_common_contract(self) -> None:
        record = normalize_frame_record(
            {
                "frame_index": 11,
                "width": 1280,
                "height": 720,
                "detections": [
                    {
                        "class_name": "結合部分",
                        "category_id": 3,
                        "category_index": 2,
                        "score": 0.67,
                        "detector_score": 0.67,
                        "class_score": 0.91,
                        "class_probs": [0.03, 0.06, 0.91],
                        "bbox_xyxy": [25.0, 30.0, 75.0, 90.0],
                        "polygons": [[25, 30, 75, 30, 75, 90, 25, 90]],
                    }
                ],
            }
        )

        det = record["detections"][0]
        self.assertEqual(record["frame_idx"], 11)
        self.assertEqual(det["category_index"], 2)
        self.assertEqual(det["bbox"], [25.0, 30.0, 50.0, 60.0])
        self.assertEqual(det["segmentation"], [[25, 30, 75, 30, 75, 90, 25, 90]])

    def test_streaming_summary_counts_masks_without_loading_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            rows = [
                {"frame_index": 0, "width": 640, "height": 360, "detections": []},
                {
                    "frame_idx": 1,
                    "width": 640,
                    "height": 360,
                    "instances": [
                        {
                            "class_name": "結合部分",
                            "score": 0.8,
                            "bbox_xyxy": [1, 2, 11, 22],
                            "segmentation": [[1, 2, 11, 2, 11, 22, 1, 22]],
                        }
                    ],
                },
            ]
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            stats = summarize_detection_jsonl(path)

            self.assertEqual(stats.frame_records, 2)
            self.assertEqual(stats.empty_frames, 1)
            self.assertEqual(stats.detections, 1)
            self.assertEqual(stats.detections_with_mask, 1)

    def test_bad_record_fails_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"detections": []}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(DetectionJsonlContractError, r"bad\.jsonl:1"):
                summarize_detection_jsonl(path)

    def test_pipeline_summary_includes_jsonl_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            detector_out = Path(tmp) / "dinov3"
            jsonl_dir = detector_out / "jsonl"
            jsonl_dir.mkdir(parents=True)
            video = Path(tmp) / "video.mp4"
            (detector_out / "summary.json").write_text(
                json.dumps({"runs": [{"output_jsonl": str(jsonl_dir / "video.jsonl")}]}),
                encoding="utf-8",
            )
            (jsonl_dir / "video.jsonl").write_text(
                json.dumps({"frame_index": 0, "detections": []}) + "\n",
                encoding="utf-8",
            )

            _jsonl_path, summary = summarize_detector(detector_out, video)

            self.assertEqual(summary["jsonl_contract"]["frame_records"], 1)


if __name__ == "__main__":
    unittest.main()
