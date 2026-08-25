# SPDX-License-Identifier: MIT
"""WD3 crash-resume config identity: `resume_from` masked, all else strict.

2026-08-25 defect: `load_checkpoint` demanded FULL config equality against
the checkpoint's stored as-run config, but a crash resume must repoint
`resume_from` at the checkpoint itself — a file that did not exist when the
config was stored. Full-dict equality was therefore self-referentially
unsatisfiable and the trainer's `resumable_from_disk: true` receipts were
false for WD3-checkpoint resume (surfaced by the s1a seed-2 external
SIGKILL, the first real crash on this trainer). These tests pin the masked
comparison: the ONE self-referential field passes, any OTHER drift refuses.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TRAINER_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "ddm_wd3_scorer_aware_width_distillation.py"
)


def _load_module():
    name = "_wd3_resume_config_identity_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TRAINER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_module()
_identity = _MOD._resume_config_identity


_BASE = {
    "schema": "ddm_wd3_compiled_config.v1",
    "seed": 20260816,
    "arm": "W96_flattened",
    "epochs": 139,
    "output": "/tmp/out",
    "resume_from": "/tmp/births/W96_flattened_birth.pt",
    "expected_builder_sha256": "a" * 64,
    "expected_receiver_sha256": "b" * 64,
}


def test_resume_from_repoint_passes_identity() -> None:
    resumed = dict(_BASE, resume_from="/tmp/out/checkpoints/wd3_epoch_0030.pt")

    assert _identity(_BASE) == _identity(resumed)


def test_builder_sha_repin_passes_identity() -> None:
    # The trainer's SELF-hash necessarily changes when a crash's cure edits
    # this file; _verify_launch_sources still pins the live trainer.
    repinned = dict(_BASE, expected_builder_sha256="c" * 64)

    assert _identity(_BASE) == _identity(repinned)


def test_receiver_sha_drift_breaks_identity() -> None:
    drifted = dict(_BASE, expected_receiver_sha256="d" * 64)

    assert _identity(_BASE) != _identity(drifted)


def test_any_other_field_drift_breaks_identity() -> None:
    for key, drifted in (
        ("seed", 20260817),
        ("epochs", 140),
        ("arm", "W72_flattened"),
        ("output", "/tmp/elsewhere"),
    ):
        assert _identity(_BASE) != _identity(dict(_BASE, **{key: drifted}))


def test_added_field_breaks_identity() -> None:
    assert _identity(_BASE) != _identity(dict(_BASE, extra_knob=1))


def test_removed_field_breaks_identity() -> None:
    shrunk = {k: v for k, v in _BASE.items() if k != "epochs"}

    assert _identity(_BASE) != _identity(shrunk)


def test_identity_does_not_mutate_input() -> None:
    config = dict(_BASE)
    _identity(config)

    assert config == _BASE
