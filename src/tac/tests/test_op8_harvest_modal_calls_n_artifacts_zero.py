# SPDX-License-Identifier: MIT
"""OP-8 fix tests (codex chunk 5/10, 2026-05-15): ``n_artifacts==0``
mislabeling in ``tools/harvest_modal_calls.py``.

Pre-fix: when a Modal dispatch returned rc=0 + not timed_out but produced
ZERO artifacts, the canonical ``append_platform_training_anchor``
(cost_band_calibration.py) derived ``outcome=successful_dispatch`` from
the rc=0 signal alone. The cost-band posterior was poisoned with anchors
that look successful but produced no usable evidence (no archive, no
auth-eval JSON, no checkpoint). Catalog #185 live-count drift detection
was poisoned by false-positive successful_dispatch counts.

Post-fix: ``tools/harvest_modal_calls.py`` injects
``cost_band_anchor.outcome="harvested_partial"`` into the metadata
copy passed to the canonical helper for the rc=0+zero-artifacts case.
The harvest summary + summary entry also carry an explicit
``outcome_classification`` field (``partial_harvest_zero_artifacts``,
``successful_dispatch``, ``timed_out``, or ``failed_rc_<rc>``).

Sister of Catalog #221 (`check_auth_eval_result_artifacts_fail_closed_for_score_claims`)
+ Catalog #245 (Modal call_id ledger taxonomy) + Catalog #185 (live-count
drift detection).

Memory: feedback_op4_7_8_batch_paired_env_lock_depth_n_artifacts_landed_20260515.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from tac.cost_band_calibration import (
    HARVESTED_PARTIAL,
    SUCCESSFUL_DISPATCH,
    append_platform_training_anchor,
)

# ──────────────────────────────────────────────────────────────────────────
# Direct-call regression on the canonical helper: confirm the override
# path is honored. This is the contract OP-8 relies on.
# ──────────────────────────────────────────────────────────────────────────


def _build_metadata(*, with_outcome_override: str | None = None) -> dict:
    cost_band_anchor = {
        "trainer": "test_trainer",
        "epochs": 100,
        "batch_size": 8,
    }
    if with_outcome_override is not None:
        cost_band_anchor["outcome"] = with_outcome_override
    return {
        "label": "op8_test",
        "gpu": "T4",
        "cost_band_anchor": cost_band_anchor,
    }


def test_op8_canonical_helper_honors_outcome_override(tmp_path):
    """``append_platform_training_anchor`` MUST honor
    ``cost_band_anchor.outcome="harvested_partial"`` even when rc=0 and
    not timed_out. This is the contract OP-8 relies on."""

    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = {"returncode": 0, "elapsed_seconds": 100.0, "timed_out": False}
    metadata = _build_metadata(with_outcome_override="harvested_partial")

    manifest = append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )
    assert manifest["appended"] is True
    rows = [json.loads(line) for line in posterior.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["outcome"] == HARVESTED_PARTIAL


def test_op8_canonical_helper_default_outcome_is_successful_dispatch(tmp_path):
    """Without override, rc=0 + not timed_out → outcome=successful_dispatch.
    This is the pre-OP-8 behavior that the OP-8 fix overrides for the
    zero-artifacts case."""

    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = {"returncode": 0, "elapsed_seconds": 100.0, "timed_out": False}
    metadata = _build_metadata()

    append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=metadata,
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )
    rows = [json.loads(line) for line in posterior.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["outcome"] == SUCCESSFUL_DISPATCH


# ──────────────────────────────────────────────────────────────────────────
# OP-8 classification logic: verify the harvest_modal_calls.py
# outcome_classification matches the canonical taxonomy.
# ──────────────────────────────────────────────────────────────────────────


def _compute_outcome_classification(*, rc, timed_out, n_artifacts):
    """Mirror the OP-8 classification logic from
    ``tools/harvest_modal_calls.py`` for unit testing without needing
    a full Modal harness."""

    if n_artifacts == 0 and rc == 0 and not timed_out:
        return "partial_harvest_zero_artifacts"
    if rc != 0:
        return f"failed_rc_{rc}"
    if timed_out:
        return "timed_out"
    return "successful_dispatch"


def test_op8_outcome_classification_zero_artifacts_rc_0():
    """rc=0 + n_artifacts==0 + not timed_out → partial_harvest_zero_artifacts."""

    assert (
        _compute_outcome_classification(rc=0, timed_out=False, n_artifacts=0)
        == "partial_harvest_zero_artifacts"
    )


def test_op8_outcome_classification_nonzero_artifacts_rc_0():
    """rc=0 + n_artifacts>0 + not timed_out → successful_dispatch."""

    assert (
        _compute_outcome_classification(rc=0, timed_out=False, n_artifacts=3)
        == "successful_dispatch"
    )
    assert (
        _compute_outcome_classification(rc=0, timed_out=False, n_artifacts=1)
        == "successful_dispatch"
    )


def test_op8_outcome_classification_rc_nonzero():
    """rc != 0 → failed_rc_<rc>."""

    assert (
        _compute_outcome_classification(rc=1, timed_out=False, n_artifacts=0)
        == "failed_rc_1"
    )
    assert (
        _compute_outcome_classification(rc=139, timed_out=False, n_artifacts=2)
        == "failed_rc_139"
    )


def test_op8_outcome_classification_timed_out():
    """timed_out=True (and rc=0 — common case) → timed_out."""

    assert (
        _compute_outcome_classification(rc=0, timed_out=True, n_artifacts=0)
        == "timed_out"
    )
    # rc!=0 takes precedence over timed_out (caller decides).
    assert (
        _compute_outcome_classification(rc=124, timed_out=True, n_artifacts=0)
        == "failed_rc_124"
    )


# ──────────────────────────────────────────────────────────────────────────
# Integration: simulate the OP-8 metadata-copy logic from
# tools/harvest_modal_calls.py and verify cost-band outcome override
# is correctly threaded through.
# ──────────────────────────────────────────────────────────────────────────


def test_op8_zero_artifacts_outcome_override_survives_to_posterior(tmp_path):
    """Mirror the OP-8 metadata-copy path inline and verify the
    canonical posterior receives outcome=harvested_partial for the
    rc=0+n_artifacts=0 case."""

    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    meta = _build_metadata()
    rc = 0
    n_artifacts = 0
    timed_out = False

    # OP-8 logic from harvest_modal_calls.py
    if n_artifacts == 0 and rc == 0 and not timed_out:
        base_cost_meta = dict(meta.get("cost_band_anchor") or {})
        base_cost_meta["outcome"] = "harvested_partial"
        cost_meta_for_anchor = {**meta, "cost_band_anchor": base_cost_meta}
    else:
        cost_meta_for_anchor = meta

    result = {"returncode": rc, "elapsed_seconds": 100.0, "timed_out": timed_out}
    append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=cost_meta_for_anchor,
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )
    rows = [json.loads(line) for line in posterior.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["outcome"] == HARVESTED_PARTIAL


def test_op8_nonzero_artifacts_outcome_stays_successful(tmp_path):
    """OP-8 must NOT override when n_artifacts>0; outcome stays
    successful_dispatch (the pre-OP-8 behavior is correct here)."""

    posterior = tmp_path / "posterior.jsonl"
    lock = tmp_path / "posterior.lock"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    meta = _build_metadata()
    rc = 0
    n_artifacts = 5
    timed_out = False

    if n_artifacts == 0 and rc == 0 and not timed_out:
        base_cost_meta = dict(meta.get("cost_band_anchor") or {})
        base_cost_meta["outcome"] = "harvested_partial"
        cost_meta_for_anchor = {**meta, "cost_band_anchor": base_cost_meta}
    else:
        cost_meta_for_anchor = meta

    result = {"returncode": rc, "elapsed_seconds": 100.0, "timed_out": timed_out}
    append_platform_training_anchor(
        "modal",
        out_dir=out_dir,
        metadata=cost_meta_for_anchor,
        result=result,
        posterior_path=posterior,
        lock_path=lock,
    )
    rows = [json.loads(line) for line in posterior.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["outcome"] == SUCCESSFUL_DISPATCH


def test_op8_meta_dict_not_mutated_in_place(tmp_path):
    """OP-8 metadata-copy logic must NOT mutate the original meta dict.
    Pre-fix, mutating ``meta["cost_band_anchor"]["outcome"]`` could leak
    state across harvest iterations."""

    meta = _build_metadata()
    original_cost_meta = dict(meta["cost_band_anchor"])
    rc = 0
    n_artifacts = 0
    timed_out = False

    if n_artifacts == 0 and rc == 0 and not timed_out:
        base_cost_meta = dict(meta.get("cost_band_anchor") or {})
        base_cost_meta["outcome"] = "harvested_partial"
        cost_meta_for_anchor = {**meta, "cost_band_anchor": base_cost_meta}

    # Original meta dict MUST be untouched.
    assert meta["cost_band_anchor"] == original_cost_meta
    assert "outcome" not in meta["cost_band_anchor"]
    # The copy used for the canonical helper has the override.
    assert cost_meta_for_anchor["cost_band_anchor"]["outcome"] == "harvested_partial"


# ──────────────────────────────────────────────────────────────────────────
# Source-code regression guard: refuse silent revert of the OP-8 fix.
# ──────────────────────────────────────────────────────────────────────────


def test_op8_harvest_modal_calls_source_carries_n_artifacts_override():
    """Refuse silent revert: ``tools/harvest_modal_calls.py`` source MUST
    carry the OP-8 outcome override + outcome_classification field."""

    repo_root = Path(__file__).resolve().parents[3]
    src = (repo_root / "tools" / "harvest_modal_calls.py").read_text()
    assert (
        'n_artifacts == 0 and rc == 0 and not timed_out' in src
    ), "OP-8 outcome-override condition missing from harvest_modal_calls.py"
    assert (
        '"outcome"' in src and '"harvested_partial"' in src
    ), "OP-8 outcome=harvested_partial override missing"
    assert (
        "outcome_classification" in src
    ), "OP-8 outcome_classification field missing from harvest_modal_calls.py"
    assert (
        "partial_harvest_zero_artifacts" in src
    ), "OP-8 partial_harvest_zero_artifacts taxonomy missing"
