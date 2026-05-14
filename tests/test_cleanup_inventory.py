from __future__ import annotations

import unittest
from pathlib import Path

from tools.maintenance.inventory_cleanup_candidates import inventory, to_markdown


ROOT = Path(__file__).resolve().parents[1]


class CleanupInventoryTests(unittest.TestCase):
    def test_inventory_finds_wrappers_without_approving_deletion(self) -> None:
        candidates = inventory(ROOT)
        paths = {item.path for item in candidates}

        self.assertIn("UI/run_app.sh", paths)
        self.assertIn("scripts/run_integrated_pipeline.py", paths)
        self.assertTrue(all(item.action != "delete" for item in candidates))

    def test_markdown_inventory_is_human_reviewable(self) -> None:
        text = to_markdown(inventory(ROOT))

        self.assertIn("This is an inventory only", text)
        self.assertIn("confirm_before_delete", text)


if __name__ == "__main__":
    unittest.main()
