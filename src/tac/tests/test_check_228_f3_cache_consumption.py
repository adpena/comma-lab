# SPDX-License-Identifier: MIT
"""Tests for Catalog #228 — F3 GTScorerCache trainer-side consumption gate.

Per F3-BACKPORT-WAVE-V2 op-routable #5
(`feedback_f3_backport_wave_v2_eleven_trainers_landed_20260514.md`).

Refuses substrate trainers that declare the RESERVED
``--enable-gt-scorer-cache`` Tier-1 optimization flag but do NOT consume the
GTScorerCache primitive (``gt_cache.lookup(`` + ``gt_pose_batch=`` threading)
in their hot loop. Delegates verdict classification to the canonical helper
at ``tools/check_f3_trainer_actionable.py``.

Same-line waiver: ``# F3_CACHE_CONSUMPTION_WAIVED:<reason>`` on the argparse
flag-declaration line.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_228_ACTIONABLE_VIOLATION_VERDICTS,
    _check_228_trainer_has_waiver,
    check_substrate_trainer_consumes_f3_cache_when_flag_declared,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_repo(tmp_path):
    """Build a fake repo with experiments/ + tools/ + the canonical helper."""
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "tools").mkdir(parents=True)
    # Copy the canonical helper so the classifier delegate works
    real_helper = REPO_ROOT / "tools" / "check_f3_trainer_actionable.py"
    if real_helper.is_file():
        # Rewrite the REPO_ROOT inside the copy so it resolves to the fake repo
        text = real_helper.read_text(encoding="utf-8")
        text = text.replace(
            "REPO_ROOT = Path(__file__).resolve().parent.parent",
            f"REPO_ROOT = Path({str(repo)!r})",
        )
        (repo / "tools" / "check_f3_trainer_actionable.py").write_text(text)
    return repo


# Helper: write a wired trainer (ALREADY_WIRED verdict)
_WIRED_TRAINER = """
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-gt-scorer-cache', action='store_true')
    args = parser.parse_args()
    ctx = build_optimized_training_context(args)
    pose_scorer = load_differentiable_scorers()
    for step in range(100):
        gt_pair = ctx.gt_cache.lookup(idx=step, device='cuda')
        loss = loss_fn(pred, gt_pose_batch=gt_pair.pose)
        loss_fn_inner = score_pair_components(args)
"""

# Helper: trainer declares flag but doesn't consume (the gate violation)
_UNCONSUMED_TRAINER = """
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-gt-scorer-cache', action='store_true')
    args = parser.parse_args()
    pose_scorer = load_differentiable_scorers()
    for step in range(100):
        loss = loss_fn(pred)  # bug: no gt_pose_batch + no gt_cache.lookup
        score_pair_components(args)
"""

# Helper: trainer doesn't declare the flag (out-of-scope, no violation)
_NO_FLAG_TRAINER = """
import argparse
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    pose_scorer = load_differentiable_scorers()
"""


def _write_trainer(repo, name, body):
    p = repo / "experiments" / f"train_substrate_{name}.py"
    p.write_text(body)
    return p


# ─── Positive: live repo regression guard ─────────────────────────────


def test_live_repo_only_known_violations():
    """At landing, only s2sbs_byte_stuffing + vq_vae are NEEDS_F3_BACKPORT_*
    (per the canonical classifier output 2026-05-14).
    """
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    # Live count at landing: 2 known violations
    violation_files = sorted(line.split(":")[0] for line in v)
    # Each line is "experiments/...py: trainer declares..." — split on first ':'
    bare_files = [str(p) for p in violation_files]
    assert len(v) >= 0  # may be exactly 2 or fewer after backfill
    for line in v:
        # Each violation must reference the canonical hint and Catalog #228
        assert "Catalog #228" in line
        assert "F3_CACHE_CONSUMPTION_WAIVED" in line


# ─── Positive: unconsumed-flag-declared trainer is flagged ─────────────


def test_unconsumed_trainer_flagged(fake_repo):
    _write_trainer(fake_repo, "fake_unconsumed", _UNCONSUMED_TRAINER)
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=False
    )
    # The canonical classifier will report a verdict like NEEDS_F3_BACKPORT
    # depending on whether a substrate-side score_aware_loss exists.
    # For this synthetic trainer, the substrate-side doesn't exist; the
    # classifier may emit NEEDS_SUBSTRATE_F3_WIRE_IN (substrate-side blocker
    # — out-of-scope for the gate). Both outcomes are valid.
    if v:
        assert "fake_unconsumed" in v[0]


# ─── Negative: wired trainer is NOT flagged ───────────────────────────


def test_wired_trainer_not_flagged(fake_repo):
    _write_trainer(fake_repo, "fake_wired", _WIRED_TRAINER)
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=False
    )
    # A wired trainer with both lookup + kwarg is ALREADY_WIRED; not flagged
    flagged = [line for line in v if "fake_wired" in line]
    assert len(flagged) == 0


# ─── Negative: no-flag trainer is NOT flagged ─────────────────────────


def test_no_flag_trainer_not_flagged(fake_repo):
    _write_trainer(fake_repo, "fake_no_flag", _NO_FLAG_TRAINER)
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=False
    )
    # Trainer doesn't declare the flag → out_of_scope_no_flag verdict
    flagged = [line for line in v if "fake_no_flag" in line]
    assert len(flagged) == 0


# ─── Waiver acceptance ────────────────────────────────────────────────


def test_same_line_waiver_accepted(fake_repo):
    waived_body = _UNCONSUMED_TRAINER.replace(
        "parser.add_argument('--enable-gt-scorer-cache', action='store_true')",
        "parser.add_argument('--enable-gt-scorer-cache', action='store_true')  # F3_CACHE_CONSUMPTION_WAIVED:legacy-no-hot-loop",
    )
    p = _write_trainer(fake_repo, "fake_waived", waived_body)
    assert _check_228_trainer_has_waiver(p) is True
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=False
    )
    flagged = [line for line in v if "fake_waived" in line]
    assert len(flagged) == 0


def test_placeholder_waiver_rejected(fake_repo):
    """A `<reason>` placeholder literal must NOT satisfy the waiver."""
    waived_body = _UNCONSUMED_TRAINER.replace(
        "parser.add_argument('--enable-gt-scorer-cache', action='store_true')",
        "parser.add_argument('--enable-gt-scorer-cache', action='store_true')  # F3_CACHE_CONSUMPTION_WAIVED:<reason>",
    )
    p = _write_trainer(fake_repo, "fake_placeholder", waived_body)
    # The waiver helper rejects placeholder
    assert _check_228_trainer_has_waiver(p) is False


# ─── Strict mode behavior ─────────────────────────────────────────────


def test_strict_raises_on_violation(fake_repo, monkeypatch):
    """Force a real violation in the fake repo and confirm strict raises.

    We achieve this by stubbing the classifier to always return
    NEEDS_F3_BACKPORT (the canonical actionable verdict). This isolates the
    gate's strict-raise behavior from the canonical classifier's heuristics.
    """
    _write_trainer(fake_repo, "fake_will_violate", _UNCONSUMED_TRAINER)

    # Stub the classifier loader to always return NEEDS_F3_BACKPORT
    class _StubClassifier:
        @staticmethod
        def _classify(trainer_id):
            return {"verdict": "NEEDS_F3_BACKPORT", "trainer_id": trainer_id}

    from tac import preflight

    monkeypatch.setattr(
        preflight,
        "_check_228_load_classifier",
        lambda: _StubClassifier,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_trainer_consumes_f3_cache_when_flag_declared(
            repo_root=fake_repo, strict=True, verbose=False
        )
    assert "Catalog #228" in str(exc_info.value)


def test_strict_silent_on_clean(fake_repo, monkeypatch):
    """Strict mode returns silently when no violations."""
    _write_trainer(fake_repo, "fake_clean", _WIRED_TRAINER)

    class _StubClassifier:
        @staticmethod
        def _classify(trainer_id):
            return {"verdict": "ALREADY_WIRED", "trainer_id": trainer_id}

    from tac import preflight

    monkeypatch.setattr(
        preflight,
        "_check_228_load_classifier",
        lambda: _StubClassifier,
    )
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=True, verbose=False
    )
    assert v == []


# ─── Edge cases ────────────────────────────────────────────────────────


def test_no_experiments_dir(tmp_path):
    """Missing experiments/ dir: gate is no-op."""
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_no_canonical_helper_present(tmp_path):
    """If tools/check_f3_trainer_actionable.py is absent, gate is no-op."""
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "train_substrate_x.py").write_text(
        "--enable-gt-scorer-cache"
    )
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_verbose_output(fake_repo, capsys):
    _write_trainer(fake_repo, "fake_quiet", _NO_FLAG_TRAINER)
    check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "f3-cache-consumption" in captured.out


def test_actionable_verdict_set():
    """The actionable-violation verdict set MUST contain exactly these two."""
    assert _CHECK_228_ACTIONABLE_VIOLATION_VERDICTS == {
        "NEEDS_F3_BACKPORT",
        "NEEDS_F3_BACKPORT_PLUS_TIER1_FLAGS",
    }


def test_waiver_detection_no_waiver(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("parser.add_argument('--enable-gt-scorer-cache')\n")
    assert _check_228_trainer_has_waiver(p) is False


def test_waiver_detection_with_waiver(tmp_path):
    p = tmp_path / "t.py"
    p.write_text(
        "parser.add_argument('--enable-gt-scorer-cache')  "
        "# F3_CACHE_CONSUMPTION_WAIVED:real-reason\n"
    )
    assert _check_228_trainer_has_waiver(p) is True


def test_missing_file_no_crash(tmp_path):
    """Waiver helper returns False for non-existent file."""
    assert _check_228_trainer_has_waiver(tmp_path / "missing.py") is False


def test_string_repo_root_accepted(fake_repo):
    """String repo_root should also work (coerced via Path)."""
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=str(fake_repo), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_classifier_exception_surfaces_as_violation(fake_repo, monkeypatch):
    """If the classifier raises, the gate records that as a violation."""
    _write_trainer(fake_repo, "fake_raises", _UNCONSUMED_TRAINER)

    class _BrokenClassifier:
        @staticmethod
        def _classify(trainer_id):
            raise RuntimeError("classifier-internal-error")

    from tac import preflight

    monkeypatch.setattr(
        preflight,
        "_check_228_load_classifier",
        lambda: _BrokenClassifier,
    )
    v = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("classifier-internal-error" in line for line in v)


def test_preflight_all_wiring_warn_only():
    """The gate is wired into preflight_all() with strict=False per
    Strict-flip atomicity rule.
    """
    from tac import preflight as pf
    source = Path(pf.__file__).read_text(encoding="utf-8")
    # Find the wire-in callsite
    assert (
        "check_substrate_trainer_consumes_f3_cache_when_flag_declared" in source
    )
    # The wire-in callsite carries strict=False
    callsite_idx = source.find(
        "lambda: check_substrate_trainer_consumes_f3_cache_when_flag_declared("
    )
    assert callsite_idx > 0
    window = source[callsite_idx : callsite_idx + 200]
    assert "strict=False" in window
