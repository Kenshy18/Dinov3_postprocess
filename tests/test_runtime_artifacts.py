from __future__ import annotations

import unittest

from tools.artifacts import runtime_artifacts as artifacts


class RuntimeArtifactLayoutTests(unittest.TestCase):
    def test_rtdetr_checkpoint_is_required_portable_artifact(self) -> None:
        self.assertIn(artifacts.RTDETR_CHECKPOINT, artifacts.PORTABLE_REQUIRED_ARTIFACTS)
        self.assertIn(artifacts.RTDETR_CHECKPOINT.mapping, artifacts.RUNTIME_MAPPINGS)
        self.assertEqual(
            artifacts.RTDETR_CHECKPOINT.dest,
            "checkpoints/rtdetr/head_face_best_stg1.pth",
        )

    def test_repo_local_rtdetr_runtime_source_is_present(self) -> None:
        self.assertTrue(artifacts.RTDETR_RUNTIME_SCRIPT.is_file())
        self.assertTrue(artifacts.RTDETR_DEFAULT_CONFIG.is_file())


if __name__ == "__main__":
    unittest.main()
