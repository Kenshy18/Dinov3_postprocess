from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.detectors.jsonl_writer import AsyncJsonlWriter, JsonlWriter, make_jsonl_writer


class JsonlWriterTests(unittest.TestCase):
    def test_sync_writer_writes_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            writer = JsonlWriter(path, "json")
            writer.write({"frame_index": 1, "label": "女性器"})
            writer.close()

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"frame_index": 1, "label": "女性器"}])

    def test_async_writer_writes_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            writer = AsyncJsonlWriter(output_path=path, backend="json", max_queue=2)
            writer.write({"idx": 1})
            writer.write({"idx": 2})
            writer.close()

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"idx": 1}, {"idx": 2}])

    def test_factory_selects_async_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_jsonl_writer(Path(tmp) / "out.jsonl", "json", async_writer=True)
            try:
                self.assertIsInstance(writer, AsyncJsonlWriter)
            finally:
                writer.close()

    def test_async_writer_reports_open_errors_without_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = AsyncJsonlWriter(output_path=Path(tmp), backend="json", max_queue=1)
            with self.assertRaises(RuntimeError):
                try:
                    writer.write({"idx": 1})
                finally:
                    writer.close()


if __name__ == "__main__":
    unittest.main()
