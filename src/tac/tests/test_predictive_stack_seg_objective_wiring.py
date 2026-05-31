# SPDX-License-Identifier: MIT
"""Predictive-stack trainer SegNet objective wiring guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(rel_path: str):
    path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_predictive_stack_full_trainers_default_to_argmax_hinge() -> None:
    """Dreamer, Z7, and Z8 should default to the argmax-correct SegNet loss."""
    cases = [
        (
            "experiments/train_substrate_dreamer_v3_rssm.py",
            ["--output-dir", ".omx/research/_parser_guard_dreamer"],
        ),
        (
            "experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py",
            ["--full", "--output-dir", ".omx/research/_parser_guard_z7"],
        ),
        (
            "experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py",
            ["--output-dir", ".omx/research/_parser_guard_z8"],
        ),
    ]
    for rel_path, argv in cases:
        module = _load_script(rel_path)
        ns = module._build_parser().parse_args(argv)
        assert ns.seg_distill_objective == "boundary_argmax_hinge"
        assert ns.seg_tau_boundary == 1.0
        assert ns.seg_hinge_margin == 1.0


def test_z7_full_replay_argv_preserves_hinge_and_ssd_flags() -> None:
    """Z7 replay bundles must not lose the selected objective or SSD lineage."""
    module = _load_script(
        "experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py"
    )
    ns = module._build_parser().parse_args(
        [
            "--full",
            "--output-dir",
            ".omx/research/_parser_guard_z7",
            "--seg-distill-objective",
            "boundary_argmax_hinge",
            "--seg-tau-boundary",
            "2.0",
            "--seg-hinge-margin",
            "1.5",
            "--use-canonical-ssd-mlx-backend",
            "--ssd-nheads",
            "2",
            "--ssd-headdim",
            "8",
        ]
    )
    argv = module._full_replay_argv(ns)
    assert argv[argv.index("--seg-distill-objective") + 1] == "boundary_argmax_hinge"
    assert argv[argv.index("--seg-tau-boundary") + 1] == "2.0"
    assert argv[argv.index("--seg-hinge-margin") + 1] == "1.5"
    assert "--use-canonical-ssd-mlx-backend" in argv
    assert argv[argv.index("--ssd-nheads") + 1] == "2"
    assert argv[argv.index("--ssd-headdim") + 1] == "8"
