# SPDX-License-Identifier: MIT
"""Tests for Catalog #388 — `check_torch_ema_uses_warmup`.

R2 of the 2026-06-11 negative-results resurrection sweep
(`.omx/research/negative_results_resurrection_ledger_20260611.md`). The
EMA-shadow-LAG bug class: a training path whose exported EMA shadow uses a
CONSTANT decay with no warmup FREEZES near init on a SHORT run (few
optimizer steps), so any metric rendered from the shadow (exact d_seg) or
the exported archive reads a stale near-init value EVEN THOUGH the live
weights solved the task. This produced the capstone "d_seg 0.505 seg-wall"
AND the lever-C "moved by zero" false-negatives.

The MLX fix (commit f771e6e00) is ported to the canonical
`tac.training.EMA` (warmup default True); this gate refuses re-introduction
of the constant-decay freeze at TWO surfaces:
  (1) an inline EMA class with a constant-decay `update()` and no warmup,
  (2) an explicit `tac.training.EMA(..., warmup=False)` construction,
both in a training-shaped script (one that calls `optimizer.step()`).

NO-FAKE discipline: these tests assert the gate's actual DETECTION
behavior (positive catches the freeze, negative allows the warmup-correct
form, waiver respected, non-training scripts ignored) — not constants.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_torch_ema_uses_warmup,
    _scan_script_for_torch_ema_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ── Inline-class fragments used to build synthetic training scripts ─────────

_CONSTANT_DECAY_INLINE_EMA = '''\
import torch
import torch.nn as nn


class EMA:
    """Inline EMA — CONSTANT decay, no warmup (the bug class)."""

    def __init__(self, model, decay=0.997):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model)
    for _ in range(10):
        opt.step()
        ema.update(model)
    return ema.state_dict()
'''

_WARMUP_INLINE_EMA = '''\
import torch
import torch.nn as nn


class EMA:
    """Inline EMA WITH warmup ramp (the FIX)."""

    def __init__(self, model, decay=0.997):
        self.decay = decay
        self._num_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        self._num_updates += 1
        d = min(self.decay, (1.0 + self._num_updates) / (10.0 + self._num_updates))
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model)
    for _ in range(10):
        opt.step()
        ema.update(model)
    return ema.state_dict()
'''

_CANONICAL_WARMUP_FALSE = '''\
import torch
from tac.training import EMA


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.997, warmup=False)
    for _ in range(10):
        opt.step()
        ema.update(model)
    return ema.state_dict()
'''

_CANONICAL_WARMUP_DEFAULT = '''\
import torch
from tac.training import EMA


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.997)
    for _ in range(10):
        opt.step()
        ema.update(model)
    return ema.state_dict()
'''


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src)
    return p


# ── POSITIVE: the gate CATCHES the constant-decay freeze ────────────────────

def test_positive_inline_constant_decay_is_flagged(tmp_path):
    p = _write(tmp_path, "train_smoke.py", _CONSTANT_DECAY_INLINE_EMA)
    v = _scan_script_for_torch_ema_warmup(p, tmp_path)
    assert len(v) == 1
    assert "inline EMA class" in v[0]
    assert "CONSTANT-decay" in v[0]


def test_positive_canonical_warmup_false_is_flagged(tmp_path):
    p = _write(tmp_path, "train_smoke.py", _CANONICAL_WARMUP_FALSE)
    v = _scan_script_for_torch_ema_warmup(p, tmp_path)
    assert len(v) == 1
    assert "warmup=False" in v[0]


# ── NEGATIVE: the gate ALLOWS the warmup-correct forms ──────────────────────

def test_negative_inline_warmup_ramp_is_allowed(tmp_path):
    p = _write(tmp_path, "train_smoke.py", _WARMUP_INLINE_EMA)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


def test_negative_canonical_default_warmup_is_allowed(tmp_path):
    p = _write(tmp_path, "train_smoke.py", _CANONICAL_WARMUP_DEFAULT)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


def test_negative_non_training_script_ignored(tmp_path):
    # A constant-decay inline EMA in a script that does NOT call
    # optimizer.step() is not a training path → not in scope.
    src = _CONSTANT_DECAY_INLINE_EMA.replace("opt.step()", "pass")
    src = src.replace(
        "opt = torch.optim.AdamW(model.parameters(), lr=1e-3)", "opt = None"
    )
    p = _write(tmp_path, "calibrate.py", src)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


# ── WAIVER respected (and placeholder rejected) ─────────────────────────────

def test_waiver_real_rationale_suppresses(tmp_path):
    src = (
        "# TORCH_EMA_WARMUP_WAIVED: deliberate constant-decay ablation, "
        "never exports a short-run shadow\n" + _CONSTANT_DECAY_INLINE_EMA
    )
    p = _write(tmp_path, "train_ablation.py", src)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


def test_waiver_placeholder_rationale_rejected(tmp_path):
    src = "# TORCH_EMA_WARMUP_WAIVED: <reason>\n" + _CONSTANT_DECAY_INLINE_EMA
    p = _write(tmp_path, "train_smoke.py", src)
    v = _scan_script_for_torch_ema_warmup(p, tmp_path)
    assert len(v) == 1  # placeholder does NOT waive


def test_waiver_empty_rationale_rejected(tmp_path):
    src = "# TORCH_EMA_WARMUP_WAIVED:\n" + _CONSTANT_DECAY_INLINE_EMA
    p = _write(tmp_path, "train_smoke.py", src)
    assert len(_scan_script_for_torch_ema_warmup(p, tmp_path)) == 1


def test_waiver_must_be_in_head(tmp_path):
    # A waiver placed deep in the file (line >8) does NOT suppress.
    body_lines = "\n".join(["# pad"] * 12)
    src = (
        body_lines
        + "\n# TORCH_EMA_WARMUP_WAIVED: too late, deep in file\n"
        + _CONSTANT_DECAY_INLINE_EMA
    )
    p = _write(tmp_path, "train_smoke.py", src)
    assert len(_scan_script_for_torch_ema_warmup(p, tmp_path)) == 1


# ── EDGE: malformed / empty / non-EMA scripts do not crash ──────────────────

def test_edge_syntax_error_returns_empty(tmp_path):
    p = _write(tmp_path, "train_broken.py", "def f(: pass\n")
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


def test_edge_no_ema_no_violation(tmp_path):
    src = textwrap.dedent(
        """
        import torch
        def train(model):
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(10):
                opt.step()
        """
    )
    p = _write(tmp_path, "train_noema.py", src)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


def test_edge_warmup_via_effective_decay_method_allowed(tmp_path):
    # A class that ramps decay via an effective_decay() method (the
    # canonical tac.training.EMA shape) must be allowed.
    src = '''\
import torch


class WeightEMA:
    def __init__(self, model, decay=0.997):
        self.decay = decay
        self._num_updates = 0
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def effective_decay(self):
        warm = (1.0 + self._num_updates) / (10.0 + self._num_updates)
        return min(self.decay, warm)

    def update(self, model):
        self._num_updates += 1
        d = self.effective_decay()
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(d).add_(v, alpha=1 - d)


def train(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = WeightEMA(model)
    for _ in range(10):
        opt.step()
        ema.update(model)
'''
    p = _write(tmp_path, "train_eff.py", src)
    assert _scan_script_for_torch_ema_warmup(p, tmp_path) == []


# ── STRICT behavior + live-codebase clean ───────────────────────────────────

def test_strict_raises_on_violation(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "train_smoke.py").write_text(_CONSTANT_DECAY_INLINE_EMA)
    with pytest.raises(MetaBugViolation):
        check_torch_ema_uses_warmup(repo_root=tmp_path, strict=True, verbose=False)


def test_non_strict_returns_list_not_raise(tmp_path):
    exp = tmp_path / "experiments"
    exp.mkdir()
    (exp / "train_smoke.py").write_text(_CONSTANT_DECAY_INLINE_EMA)
    v = check_torch_ema_uses_warmup(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1


def test_live_codebase_passes_strict():
    # The fix (canonical warmup EMA + 5 inline-EMA fleet fixes) drove the
    # live count to 0; this anchors the strict-flip.
    v = check_torch_ema_uses_warmup(repo_root=REPO_ROOT, strict=False, verbose=False)
    assert v == [], f"live torch-ema-warmup violations: {v}"
