"""Tests for the #304-item-4 per-term loss telemetry (STEP-INSTAB-DIAG landing, 2026-07-05).

Covers the pure helpers (`LOSS_TERM_KEYS`, `_loss_terms_row`, `_loss_term_log_stride`) added to
``experiments/train_levelset_witness_realized_through_R_mlx.py`` plus the additive ``terms_out``
hook threaded through the base trainer's ``make_loss_fn`` and the levelset ``total_loss_fn``.

The telemetry contract under test:
  * row schema is STABLE (every LOSS_TERM_KEYS key present; missing terms -> 0.0);
  * the row is self-checking (sum_terms / sum_minus_total);
  * cadence resolution: TAC_LOSS_TERM_PROBE=1 -> per-batch; --loss-term-log-every N>0 -> every N
    chunks; N<0 -> fully OFF; default 0 -> per-epoch summary (first chunk);
  * the hook is ADDITIVE: terms_out defaults to None everywhere (byte-identical off-path). The
    bitwise trajectory identity (flag off vs on) is proven by the runtime n1 CPU A/B recorded in
    .omx/research/stepping_instability_diagnostic_20260705.md (this file guards the pure surface).

The trainer module imports mlx/torch at top; skip cleanly where unavailable."""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import re

import pytest

pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")

_REPO = pathlib.Path(__file__).resolve().parents[3]
_MODPATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_BASEPATH = _REPO / "experiments" / "train_witness_realized_through_R_mlx.py"


def _load(path: pathlib.Path, name: str):
    if not path.exists():
        pytest.skip(f"trainer not found at {path}", allow_module_level=True)
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass introspection needs the module registered pre-exec
    try:
        spec.loader.exec_module(mod)  # heavy imports only; no GPU ops
    except Exception as exc:  # pragma: no cover - env-dependent
        sys.modules.pop(name, None)
        pytest.skip(f"could not import {path.name}: {type(exc).__name__}: {exc}",
                    allow_module_level=True)
    return mod


MOD = _load(_MODPATH, "_levelset_trainer_loss_term_test")
BASE = _load(_BASEPATH, "_base_trainer_loss_term_test")


# ---------------------------------------------------------------- LOSS_TERM_KEYS schema
def test_loss_term_keys_cover_the_composed_terms():
    keys = set(MOD.LOSS_TERM_KEYS)
    # base terms + every stacked lever term named in total_loss_fn.
    for expected in ("seg", "pose", "eikonal", "length", "boundary_distance", "lane_edge",
                     "margin_saliency", "subpix", "chroma_boundary", "island_amplify",
                     "persistence", "rankfloor", "code_spectral", "thin_lane",
                     "margin_field_head", "code_nuclear"):
        assert expected in keys, expected


def test_loss_term_keys_unique():
    assert len(MOD.LOSS_TERM_KEYS) == len(set(MOD.LOSS_TERM_KEYS))


# ---------------------------------------------------------------- _loss_terms_row
def test_row_schema_stable_with_missing_terms():
    row = MOD._loss_terms_row({"seg": 1.5}, total=1.5, ep=101, accum_batch=0)
    assert row["stage"] == "loss_terms"
    assert row["ep"] == 101 and row["accum_batch"] == 0
    assert set(row["terms"].keys()) == set(MOD.LOSS_TERM_KEYS)
    assert row["terms"]["seg"] == 1.5
    assert row["terms"]["persistence"] == 0.0  # missing -> 0.0 (stable schema)


def test_row_sum_terms_matches_total_within_tolerance():
    terms = {"seg": 30.0, "pose": 5.0, "boundary_distance": 3.5, "persistence": 1.25}
    row = MOD._loss_terms_row(terms, total=39.75, ep=1, accum_batch=2)
    assert row["sum_terms"] == pytest.approx(39.75)
    assert abs(row["sum_minus_total"]) < 1e-6


def test_row_sum_minus_total_flags_mismatch():
    row = MOD._loss_terms_row({"seg": 10.0}, total=40.0, ep=1, accum_batch=0)
    assert row["sum_minus_total"] == pytest.approx(-30.0)


def test_row_gnorm_and_skipped_flags():
    row = MOD._loss_terms_row({"seg": 1.0}, total=1.0, ep=5, accum_batch=3,
                              gnorm=12.34567, skipped=True)
    assert row["gnorm"] == pytest.approx(12.3457)
    assert row["spike_skipped"] is True
    row2 = MOD._loss_terms_row({"seg": 1.0}, total=1.0, ep=5, accum_batch=3,
                               gnorm=float("inf"), skipped=False)
    assert row2["gnorm"] == "nonfinite"
    assert row2["spike_skipped"] is False
    row3 = MOD._loss_terms_row({"seg": 1.0}, total=1.0, ep=5, accum_batch=3)
    assert "gnorm" not in row3 and "spike_skipped" not in row3


def test_row_is_json_serializable():
    import json
    row = MOD._loss_terms_row({"seg": 1.0}, total=1.0, ep=1, accum_batch=0, gnorm=2.0, skipped=False)
    json.dumps(row)


# ---------------------------------------------------------------- _loss_term_log_stride
def test_stride_env_probe_wins():
    assert MOD._loss_term_log_stride(True, 0, 75) == 1
    assert MOD._loss_term_log_stride(True, 10, 75) == 1
    assert MOD._loss_term_log_stride(True, -1, 75) == 1  # probe env overrides even OFF


def test_stride_explicit_every_n():
    assert MOD._loss_term_log_stride(False, 5, 75) == 5
    assert MOD._loss_term_log_stride(False, 1, 75) == 1


def test_stride_negative_is_off():
    assert MOD._loss_term_log_stride(False, -1, 75) == 0


def test_stride_default_is_per_epoch():
    assert MOD._loss_term_log_stride(False, 0, 75) == 75
    assert MOD._loss_term_log_stride(False, 0, 1) == 1
    assert MOD._loss_term_log_stride(False, 0, 0) == 1  # degenerate floor


# ---------------------------------------------------------------- additive hook wiring
def test_base_make_loss_fn_accepts_terms_out_default_none():
    loss_fn = BASE.make_loss_fn(adapter=object(), render_h=4, render_w=4)
    sig = inspect.signature(loss_fn)
    assert "terms_out" in sig.parameters
    assert sig.parameters["terms_out"].default is None


def test_levelset_total_loss_fn_source_threads_terms_out():
    src = _MODPATH.read_text()
    m = re.search(r"def total_loss_fn\((.*?)\):", src)
    assert m is not None
    assert "terms_out=None" in m.group(1)
    # base_loss must receive the same dict (seg/pose split comes from the base trainer).
    assert re.search(r"base_loss\([^)]*terms_out=terms_out", src, re.S)


def test_levelset_argparse_has_loss_term_log_every_flag():
    src = _MODPATH.read_text()
    assert '"--loss-term-log-every"' in src  # NEVER-invent-CLI-flags guard for probe callers


def test_value_and_grad_path_never_passes_terms_out():
    """The grad path must stay byte-identical: no terms_out inside any value_and_grad call site."""
    src = _MODPATH.read_text()
    for m in re.finditer(r"value_and_grad\(\s*model,[^)]*\)", src):
        assert "terms_out" not in m.group(0)
