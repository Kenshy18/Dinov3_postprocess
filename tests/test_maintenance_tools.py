from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.maintenance.clean_generated import collect_targets


class CleanGeneratedTests(unittest.TestCase):
    def test_collect_targets_preserves_output_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            (output / ".gitkeep").write_text("", encoding="utf-8")
            (output / "README.md").write_text("keep", encoding="utf-8")
            (output / "run").mkdir()
            (root / ".serena").mkdir()
            (root / "pkg" / "__pycache__").mkdir(parents=True)
            (root / ".venv_integrated" / "lib" / "__pycache__").mkdir(parents=True)

            targets = collect_targets(
                root,
                include_caches=True,
                include_output=True,
                include_serena=True,
            )

        paths = {target.path.relative_to(root).as_posix() for target in targets}
        self.assertIn("output/run", paths)
        self.assertIn(".serena", paths)
        self.assertIn("pkg/__pycache__", paths)
        self.assertNotIn("output/.gitkeep", paths)
        self.assertNotIn("output/README.md", paths)
        self.assertNotIn(".venv_integrated/lib/__pycache__", paths)


if __name__ == "__main__":
    unittest.main()
