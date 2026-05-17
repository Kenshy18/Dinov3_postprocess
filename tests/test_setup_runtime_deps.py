import os
import unittest
from unittest import mock

from tools.setup.ensure_runtime_dependencies import (
    TorchInfo,
    capability_major,
    cuda_tag,
    should_repin_torch_for_mmcv,
    torch_tag_candidates,
)


class SetupRuntimeDependencyTests(unittest.TestCase):
    def test_cuda_tag_from_torch_cuda(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cuda_tag("12.1"), "cu121")
            self.assertEqual(cuda_tag("11.8"), "cu118")
            self.assertEqual(cuda_tag(None), "cpu")

    def test_cuda_tag_override(self) -> None:
        with mock.patch.dict(os.environ, {"MMCV_CUDA_TAG": "cu121"}, clear=True):
            self.assertEqual(cuda_tag("12.9"), "cu121")

    def test_torch_tag_candidates(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(torch_tag_candidates("2.1.2+cu121"), ["torch2.1", "torch2.1.0"])

    def test_capability_major(self) -> None:
        self.assertEqual(capability_major("8.9"), 8)
        self.assertEqual(capability_major("12.0"), 12)
        self.assertIsNone(capability_major(None))

    def test_repin_torch_for_ada_but_not_blackwell(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                should_repin_torch_for_mmcv(
                    TorchInfo(version="2.5.0+cu124", cuda_version="12.4", capability="8.9", cuda_usable=True)
                )
            )
            self.assertFalse(
                should_repin_torch_for_mmcv(
                    TorchInfo(version="2.11.0+cu129", cuda_version="12.9", capability="12.0", cuda_usable=True)
                )
            )


if __name__ == "__main__":
    unittest.main()
