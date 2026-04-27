from __future__ import annotations

import pickle
import unittest

from atosyori_postprocess.engine import standalone_runtime_fst
from atosyori_postprocess.engine.registry import import_all


class EngineImportTests(unittest.TestCase):
    def test_all_extracted_engine_modules_import(self) -> None:
        imported = import_all()
        self.assertIn("atosyori_postprocess.engine.polygon_v22", imported)
        self.assertIn("atosyori_postprocess.engine.ellipse_inference", imported)

    def test_registered_fst_worker_is_pickleable(self) -> None:
        worker = standalone_runtime_fst.fst._solve_k1_row_worker
        self.assertIs(pickle.loads(pickle.dumps(worker)), worker)


if __name__ == "__main__":
    unittest.main()
