"""Registry of extracted engine modules."""

from __future__ import annotations

import importlib

MODULES = [
    "atosyori_postprocess.engine.standalone_runtime_fst",
    "atosyori_postprocess.engine.standalone_runtime_k2v5",
    "atosyori_postprocess.engine.ellipse_inference",
    "atosyori_postprocess.engine.optimize_keyframes_standalone",
    "atosyori_postprocess.engine.optimize_keyframes_dense_recall_standalone",
    "atosyori_postprocess.engine.optimize_keyframes_trackk_dense_recall_standalone",
    "atosyori_postprocess.engine.evaluate_keyframes_exact",
    "atosyori_postprocess.engine.fill_trackk_union_gaps",
    "atosyori_postprocess.engine.union_json_to_pred_sqlite",
    "atosyori_postprocess.engine.render_overlay",
    "atosyori_postprocess.engine.polygon_v22",
    "atosyori_postprocess.engine.polygon_runtime",
]


def import_all() -> list[str]:
    imported: list[str] = []
    for module_name in MODULES:
        importlib.import_module(module_name)
        imported.append(module_name)
    return imported
