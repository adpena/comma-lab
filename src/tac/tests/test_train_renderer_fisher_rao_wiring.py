# SPDX-License-Identifier: MIT
# LOSS_CONVERGENCE_NOT_REQUIRED: CLI/static wiring test for the opt-in
# Fisher-Rao SegNet training surrogate.
"""Train-renderer wiring tests for the Fisher-Rao SegNet surrogate."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN_PATH = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


def _train_src() -> str:
    return TRAIN_PATH.read_text(encoding="utf-8")


def test_parse_args_exposes_fisher_rao_surrogate_flags() -> None:
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args

    default = parse_args(["--tag", "smoke"])
    assert default.segmentation_surrogate == "soft_cosine"
    assert default.segmentation_temperature == 1.0
    assert default.fisher_rao_eps == 1e-6

    fisher = parse_args(
        [
            "--tag",
            "smoke",
            "--segmentation-surrogate",
            "fisher_rao",
            "--segmentation-temperature",
            "0.75",
            "--fisher-rao-eps",
            "1e-7",
        ]
    )
    assert fisher.segmentation_surrogate == "fisher_rao"
    assert fisher.segmentation_temperature == 0.75
    assert fisher.fisher_rao_eps == 1e-7


def test_train_renderer_validates_surrogate_config_at_boot() -> None:
    src = _train_src()

    assert "segnet_surrogate_per_pixel as _segnet_surrogate_probe" in src
    assert "_segnet_surrogate_probe(" in src
    assert "surrogate=args.segmentation_surrogate" in src
    assert "temperature=args.segmentation_temperature" in src
    assert "fisher_rao_eps=args.fisher_rao_eps" in src


def test_train_renderer_threads_surrogate_kwargs_to_all_scorer_paths() -> None:
    src = _train_src()

    assert re.search(
        r"_seg_kwargs\s*=\s*\{.*?segmentation_surrogate.*?"
        r"segmentation_temperature.*?fisher_rao_eps.*?\}",
        src,
        re.DOTALL,
    )
    for call_name in (
        "scorer_loss_cached_with_aux",
        "scorer_loss_cached",
        "scorer_loss_with_aux",
        "scorer_loss",
    ):
        call = re.search(rf"(?<![A-Za-z0-9_]){call_name}\(.*?\)", src, re.DOTALL)
        assert call is not None, f"{call_name} call not found"
        assert "**_seg_kwargs" in call.group(0), f"{call_name} lost Fisher-Rao kwargs"
