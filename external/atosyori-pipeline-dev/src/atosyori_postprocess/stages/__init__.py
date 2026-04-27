"""Feature-stage wrappers around the legacy engine hidden subcommands."""

from __future__ import annotations

from . import ellipse_inference, ellipse_keyframes, evaluation, polygon_keyframes, preprocess, render

STAGES = {
    "preprocess": preprocess,
    "ellipse-inference": ellipse_inference,
    "ellipse-keyframes": ellipse_keyframes,
    "polygon-keyframes": polygon_keyframes,
    "evaluation": evaluation,
    "render": render,
}
