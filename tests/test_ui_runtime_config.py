from __future__ import annotations

import unittest
from unittest.mock import patch

from apps.qt_ui import runtime_config


class UiRuntimeConfigTests(unittest.TestCase):
    def test_selected_dinov3_engine_ignores_stale_portrait_profile_engine(self) -> None:
        old_engine = (
            runtime_config.ROOT
            / "checkpoints"
            / "trt"
            / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
        )
        with patch.dict("os.environ", {"DINOV3_TRT_BACKBONE_ENGINE": str(old_engine)}):
            selected = runtime_config.selected_trt_engine()

        self.assertEqual(selected.name, "dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine")


if __name__ == "__main__":
    unittest.main()
