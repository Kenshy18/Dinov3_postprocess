import unittest

from backend.pipeline.progress import parse_progress_line


class ProgressLineTests(unittest.TestCase):
    def test_key_value_progress_line(self) -> None:
        fields = parse_progress_line("[phase-progress] phase=detector current=10 total=100 percent=10 fps=42.5 eta=0:02")
        self.assertEqual(fields["phase"], "detector")
        self.assertEqual(fields["current"], "10")
        self.assertEqual(fields["fps"], "42.5")

    def test_labelled_postprocess_progress_line(self) -> None:
        fields = parse_progress_line("[phase-progress] preprocess: elapsed=12.0s cpu=18.2% rss=1.25GiB gpu=40%")
        self.assertEqual(fields["phase"], "preprocess")
        self.assertEqual(fields["elapsed"], "12.0s")
        self.assertEqual(fields["rss"], "1.25GiB")


if __name__ == "__main__":
    unittest.main()
