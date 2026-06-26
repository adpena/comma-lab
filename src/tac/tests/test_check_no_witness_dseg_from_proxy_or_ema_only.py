# SPDX-License-Identifier: MIT
"""Tests for Catalog #392 measurement-integrity gate + the helper.

check_no_witness_dseg_from_proxy_or_ema_only_harness refuses any file claiming a
witness d_seg SCORE from a proxy (generator-argmax / no-R / no-SegNet-re-seg) or
EMA-only (no live) path. Companion: tac.measurement_integrity.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tac.measurement_integrity import (
    CONTEST_FAITHFUL_R_FN,
    FEASIBILITY_ONLY_MARKER,
    FEASIBILITY_ONLY_TAG,
    STALE_SWAP_R_FN,
    warn_ema_only_dseg,
    warn_feasibility_only_dseg,
    warn_stale_swap_roundtrip,
)
from tac.preflight import (
    PreflightError,
    check_no_witness_dseg_from_proxy_or_ema_only_harness as chk,
)

CLAIM = "the d_seg here is the EXACT score-native quantity"  # canonical FAKE phrasing


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# POSITIVE: violations are caught
# --------------------------------------------------------------------------
def test_proxy_generator_argmax_with_score_claim_no_marker_is_caught(tmp_path):
    _write(tmp_path, "tools/bad_proxy.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    v = chk(strict=False, repo_root=tmp_path)
    assert any("bad_proxy.py" in x for x in v), v


def test_proxy_argmax_vs_gt_lstar_with_claim_is_caught(tmp_path):
    body = f'"""{CLAIM}"""\nimport numpy as np\nflips = argmax_out != lstar\nd_seg = flips.mean()\n'
    _write(tmp_path, "experiments/bad_lstar.py", body)
    v = chk(strict=False, repo_root=tmp_path)
    assert any("bad_lstar.py" in x for x in v), v


def test_swap_fn_call_with_claim_no_faithful_is_caught(tmp_path):
    body = f'"""{CLAIM}"""\nr = {STALE_SWAP_R_FN}(rgb)\nd_seg = score(r)\n'
    _write(tmp_path, "tools/bad_swap.py", body)
    v = chk(strict=False, repo_root=tmp_path)
    assert any("bad_swap.py" in x for x in v), v


def test_ema_only_with_claim_no_live_is_caught(tmp_path):
    body = f'"""{CLAIM}"""\nema_d_seg = shadow_eval()\nbest = ema_d_seg\n'
    _write(tmp_path, "experiments/bad_ema.py", body)
    v = chk(strict=False, repo_root=tmp_path)
    assert any("bad_ema.py" in x for x in v), v


def test_strict_mode_raises_on_violation(tmp_path):
    _write(tmp_path, "tools/bad_proxy.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    with pytest.raises(PreflightError):
        chk(strict=True, repo_root=tmp_path)


# --------------------------------------------------------------------------
# NEGATIVE: legitimate files are allowed
# --------------------------------------------------------------------------
def test_proxy_with_feasibility_marker_is_allowed(tmp_path):
    body = f'"""{CLAIM}\n{FEASIBILITY_ONLY_MARKER}: tagged feasibility-only"""\nd_seg = generator_argmax(x)\n'
    _write(tmp_path, "tools/ok_marked.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


def test_realized_harness_token_is_allowed(tmp_path):
    body = f'"""{CLAIM}"""\nd_seg = cpu_verdict_d_seg(seg, frame1, gt_argmax)\n'
    _write(tmp_path, "experiments/ok_realized.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


def test_realized_via_vendored_segnet_pipeline_is_allowed(tmp_path):
    # mirrors experiments/probe_yousfi_detector_cost_blindspot_b.py: own real pipeline
    body = (
        '"""runs the EXACT vendored SegNet pipeline on rendered frames"""\n'
        "out = segnet.preprocess_input(x)\nd_seg = (argmax_out != gt_argmax).mean()\n"
        f"# note: {CLAIM}\n"
    )
    _write(tmp_path, "experiments/ok_vendored.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


def test_proxy_without_score_claim_is_allowed(tmp_path):
    # computes a proxy d_seg but NEVER claims it is a score -> not a violation
    body = '"""feasibility descent smoke (no score claim)"""\nd_seg = generator_argmax(x)\n'
    _write(tmp_path, "tools/ok_no_claim.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


def test_swap_call_with_contest_faithful_present_is_allowed(tmp_path):
    body = (
        f'"""{CLAIM}"""\n'
        f"# training recipe uses {STALE_SWAP_R_FN} but verdict uses {CONTEST_FAITHFUL_R_FN}\n"
        f"r = {STALE_SWAP_R_FN}(rgb)\nverdict = {CONTEST_FAITHFUL_R_FN}(rgb)\nd_seg = 0.0\n"
    )
    _write(tmp_path, "tools/ok_both_r.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


def test_ema_with_live_companion_is_allowed(tmp_path):
    body = f'"""{CLAIM}"""\nema_d_seg = e()\nd_seg_live = live()\n'
    _write(tmp_path, "experiments/ok_ema_live.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# WAIVER discipline
# --------------------------------------------------------------------------
def test_valid_waiver_is_respected(tmp_path):
    body = (
        f'"""{CLAIM}"""\n'
        "d_seg = generator_argmax(x)  # WITNESS_DSEG_FEASIBILITY_ONLY_OK: explicit research upper-bound probe\n"
    )
    _write(tmp_path, "tools/ok_waived.py", body)
    assert chk(strict=False, repo_root=tmp_path) == []


@pytest.mark.parametrize("ph", ["<rationale>", "<reason>", "tbd", "todo", ""])
def test_placeholder_waiver_is_rejected(tmp_path, ph):
    body = (
        f'"""{CLAIM}"""\n'
        f"d_seg = generator_argmax(x)  # WITNESS_DSEG_FEASIBILITY_ONLY_OK:{ph}\n"
    )
    _write(tmp_path, "tools/bad_placeholder.py", body)
    v = chk(strict=False, repo_root=tmp_path)
    assert any("bad_placeholder.py" in x for x in v), (ph, v)


# --------------------------------------------------------------------------
# SCOPE / EDGE
# --------------------------------------------------------------------------
def test_files_outside_scan_dirs_are_not_scanned(tmp_path):
    _write(tmp_path, "scripts/elsewhere.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    assert chk(strict=False, repo_root=tmp_path) == []


def test_test_files_are_excluded(tmp_path):
    _write(tmp_path, "tools/test_thing.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    _write(tmp_path, "src/tac/tests/test_x.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    assert chk(strict=False, repo_root=tmp_path) == []


def test_results_and_worktree_dirs_excluded(tmp_path):
    _write(tmp_path, "experiments/results/old/x.py", f'"""{CLAIM}"""\nd_seg = generator_argmax(x)\n')
    assert chk(strict=False, repo_root=tmp_path) == []


def test_empty_repo_is_clean(tmp_path):
    assert chk(strict=False, repo_root=tmp_path) == []


# --------------------------------------------------------------------------
# Helper: tac.measurement_integrity
# --------------------------------------------------------------------------
def test_warn_feasibility_only_dseg_returns_tag_and_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tag = warn_feasibility_only_dseg("unit_test_harness")
    assert tag == FEASIBILITY_ONLY_TAG
    assert "NOT realized" in tag
    assert any("NOT a contest score" in str(x.message) for x in w)


def test_warn_ema_only_returns_tag():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tag = warn_ema_only_dseg("unit_test_harness")
    assert tag == FEASIBILITY_ONLY_TAG
    assert any("EMA" in str(x.message) for x in w)


def test_warn_stale_swap_emits_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_stale_swap_roundtrip("unit_test_caller")
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


# --------------------------------------------------------------------------
# Regression: the live repo proxy harnesses + realized harness state
# --------------------------------------------------------------------------
def test_live_repo_gate_is_clean():
    # The whole point of the foundation: no proxy/EMA d_seg masquerades as a score.
    assert chk(strict=True) == []


def test_known_proxy_harnesses_carry_the_marker():
    repo = Path(__file__).resolve().parents[3]
    for rel in (
        "tools/witness_capstone_deepmath_smoke.py",
        "tools/lever_b_score_native_argmax_smoke.py",
        "tools/score_native_build_byte_closed_candidate.py",
    ):
        p = repo / rel
        if p.is_file():
            assert FEASIBILITY_ONLY_MARKER in p.read_text(encoding="utf-8"), rel
