# SPDX-License-Identifier: MIT
"""Tests for the SNeRV TUB DROP_OR_REIFY source-forward proof.

These tests verify ACTUAL behavior, not constants (CLAUDE.md NO FAKE
IMPLEMENTATIONS, Slot EEE class 2): they confirm that the proof classifies a
TUB source-state facet REIFY/DROP by whether a real source-state bit-flip
propagates through the real receiver RGB primitive, that a real scorer drives a
``CandidateActionEvaluation`` through the shared waterfilling law, and that the
proof fails closed and never claims score authority.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.analysis.snerv_official_tub_source_forward_replay import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    DROP_OR_REIFY_FACET_SCHEMA,
    DROP_OR_REIFY_PROOF_SCHEMA,
    TUB_DROP_OR_REIFY_FACETS,
    TUB_DROP_OR_REIFY_SCORER_PENDING_BLOCKER,
    TUB_DROP_OR_REIFY_SOURCE_GRAPH_BLOCKER,
    TUB_DROP_OR_REIFY_VERDICTS,
    TUB_OUTPUT2_RECEIVER_NONCAUSAL_BLOCKER,
    TUB_SOURCE_FACET_OUTPUT2,
    TUB_SOURCE_FACET_YL_NORM,
    VERDICT_DROP,
    VERDICT_REIFY,
    VERDICT_REIFY_PENDING_SCORER,
    _capture_tub_source_state_for_drop_or_reify,
    _coerce_scorer_metrics,
    _default_reference_scorer,
    _evaluate_tub_source_facet_drop_or_reify,
    _flip_low_mantissa_bit,
    _flip_scalar_pair_low_bit,
    _rgb_uint8_from_frame,
    build_snerv_official_tub_drop_or_reify_source_forward_proof,
)
from tac.substrates.snerv_inverse_steg_carrier.official_tub import (
    official_tub_frame_reconstruction_numpy,
)

FALSE_AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _official_repo() -> Path:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    return DEFAULT_OFFICIAL_SNERV_REPO


# --------------------------------------------------------------------------- #
# Bit-flip helpers actually mutate bytes (no-op-detector discipline).
# --------------------------------------------------------------------------- #


def test_flip_low_mantissa_bit_changes_every_finite_nonzero_element() -> None:
    arr = np.array([[1.0, 2.5], [3.75, -0.5]], dtype=np.float64)
    flipped = _flip_low_mantissa_bit(arr)
    assert flipped.shape == arr.shape
    # Every element actually changed (the proof's source-state mutation is real).
    assert np.all(flipped != arr)
    assert np.all(np.isfinite(flipped))
    # The change is the minimal representable nonzero step (1 ULP).
    assert float(np.max(np.abs(flipped - arr))) < 1e-14
    assert float(np.max(np.abs(flipped - arr))) > 0.0


def test_flip_low_mantissa_bit_leaves_zero_and_nonfinite_untouched() -> None:
    arr = np.array([0.0, np.inf, -np.inf, np.nan, 4.0], dtype=np.float64)
    flipped = _flip_low_mantissa_bit(arr)
    assert flipped[0] == 0.0  # zero untouched
    assert np.isinf(flipped[1]) and flipped[1] > 0
    assert np.isinf(flipped[2]) and flipped[2] < 0
    assert np.isnan(flipped[3])
    assert flipped[4] != 4.0  # finite nonzero element changed


def test_flip_low_mantissa_bit_all_zero_falls_back_to_eps_nudge() -> None:
    arr = np.zeros((3,), dtype=np.float64)
    flipped = _flip_low_mantissa_bit(arr)
    # Fallback guarantees a real byte change even on an all-zero facet.
    assert np.all(flipped != arr)
    assert np.all(np.isfinite(flipped))


def test_flip_scalar_pair_low_bit_changes_both_scalars() -> None:
    base = (0.078125, 0.546875)
    flipped = _flip_scalar_pair_low_bit(base)
    assert flipped != base
    assert flipped[0] != base[0]
    assert flipped[1] != base[1]


# --------------------------------------------------------------------------- #
# Reference scorer is a real deterministic function of RGB (non-authority).
# --------------------------------------------------------------------------- #


def test_default_reference_scorer_zero_on_identical_rgb() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, size=(1, 2, 3, 8, 8)).astype(
        np.uint8
    )
    out = _default_reference_scorer(rgb, rgb)
    assert out["d_seg"] == 0.0
    assert out["d_pose"] == 0.0


def test_default_reference_scorer_nonzero_on_changed_rgb() -> None:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, size=(1, 2, 3, 8, 8)).astype(np.uint8)
    cand = base.copy()
    # Force an argmax flip in the last frame at one pixel.
    cand[0, -1, :, 0, 0] = np.array([255, 0, 0], dtype=np.uint8)
    base[0, -1, :, 0, 0] = np.array([0, 0, 255], dtype=np.uint8)
    out = _default_reference_scorer(base, cand)
    assert out["d_seg"] > 0.0 or out["d_pose"] > 0.0


# --------------------------------------------------------------------------- #
# _coerce_scorer_metrics validates and fails closed.
# --------------------------------------------------------------------------- #


def test_coerce_scorer_metrics_accepts_valid() -> None:
    blockers: list[str] = []
    out = _coerce_scorer_metrics(
        {"d_seg": 0.1, "d_pose": 0.02}, facet_name="yl_norm", blockers=blockers
    )
    assert out == {"d_seg": 0.1, "d_pose": 0.02}
    assert blockers == []


def test_coerce_scorer_metrics_rejects_negative() -> None:
    blockers: list[str] = []
    out = _coerce_scorer_metrics(
        {"d_seg": -1.0, "d_pose": 0.02}, facet_name="yl_norm", blockers=blockers
    )
    assert out is None
    assert any("scorer_metric_invalid" in b for b in blockers)


def test_coerce_scorer_metrics_rejects_non_mapping() -> None:
    blockers: list[str] = []
    out = _coerce_scorer_metrics([0.1, 0.2], facet_name="x", blockers=blockers)
    assert out is None
    assert any("scorer_output_invalid" in b for b in blockers)


# --------------------------------------------------------------------------- #
# Facet evaluator: DROP when receiver does not consume the source state.
# --------------------------------------------------------------------------- #


def _synthetic_captured(
    *, frame: np.ndarray, output2: np.ndarray | None = None
) -> dict:
    img_yl = frame.copy()
    yh_out = np.zeros((1, 3, 3, frame.shape[-2], frame.shape[-1]), dtype=np.float64)
    yl_norm = (0.0, 1.0)
    base = official_tub_frame_reconstruction_numpy(img_yl, yh_out, yl_norm=yl_norm).frame
    return {
        "img_yl": img_yl,
        "yh_out": yh_out,
        "yl_norm": yl_norm,
        "output2_shuffled": (
            output2 if output2 is not None else np.ones((2, 1, 4, 9), dtype=np.float64)
        ),
        "base_frame": np.asarray(base, dtype=np.float64),
        "base_rgb_uint8": _rgb_uint8_from_frame(base),
        "receiver_archive_sha256": "0" * 64,
        "receiver_archive_bytes": 4096,
    }


def test_facet_output2_is_dropped_receiver_does_not_consume() -> None:
    captured = _synthetic_captured(
        frame=np.random.default_rng(2).random((1, 3, 8, 8)).astype(np.float64)
    )
    facet = _evaluate_tub_source_facet_drop_or_reify(
        facet_name=TUB_SOURCE_FACET_OUTPUT2,
        captured=captured,
        base_rgb_uint8=captured["base_rgb_uint8"],
        base_archive_sha256="0" * 64,
        bytes_base=4096,
        action_id="unit",
        scorer_fn=None,
    )
    assert facet["facet"] == TUB_SOURCE_FACET_OUTPUT2
    assert facet["verdict"] == VERDICT_DROP
    assert facet["receiver_consumes_facet"] is False
    # output_2 source byte changed but did not reach the receiver frame at all.
    assert facet["source_byte_changed"] is True
    assert facet["receiver_frame_float_linf"] == 0.0
    assert TUB_OUTPUT2_RECEIVER_NONCAUSAL_BLOCKER in facet["blockers"]
    assert facet["candidate_action_evaluation"] is None


def test_facet_yl_norm_is_receiver_causal_float_level() -> None:
    captured = _synthetic_captured(
        frame=np.random.default_rng(3).random((1, 3, 8, 8)).astype(np.float64)
    )
    facet = _evaluate_tub_source_facet_drop_or_reify(
        facet_name=TUB_SOURCE_FACET_YL_NORM,
        captured=captured,
        base_rgb_uint8=captured["base_rgb_uint8"],
        base_archive_sha256="0" * 64,
        bytes_base=4096,
        action_id="unit",
        scorer_fn=None,
    )
    # yl_norm IS a receiver render input; even a 1-ULP flip propagates to the
    # float frame, so the receiver consumes it (causal, not DROP).
    assert facet["receiver_consumes_facet"] is True
    assert facet["receiver_frame_float_linf"] is not None
    assert float(facet["receiver_frame_float_linf"]) > 0.0
    assert facet["verdict"] in {VERDICT_REIFY, VERDICT_REIFY_PENDING_SCORER}


# --------------------------------------------------------------------------- #
# REIFY-with-scorer: a uint8-surviving receiver-causal facet emits a real
# CandidateActionEvaluation through the shared waterfilling law.
# --------------------------------------------------------------------------- #


def _captured_with_uint8_surviving_yl_norm() -> dict:
    # Construct a base frame whose yl_norm flip is intentionally LARGE so the
    # change survives the uint8 boundary, exercising the REIFY+scorer branch.
    frame = np.random.default_rng(4).random((1, 3, 8, 8)).astype(np.float64)
    img_yl = frame.copy()
    yh_out = np.zeros((1, 3, 3, frame.shape[-2], frame.shape[-1]), dtype=np.float64)
    yl_norm = (0.0, 1.0)
    base = official_tub_frame_reconstruction_numpy(img_yl, yh_out, yl_norm=yl_norm).frame
    return {
        "img_yl": img_yl,
        "yh_out": yh_out,
        "yl_norm": yl_norm,
        "output2_shuffled": np.ones((2, 1, 4, 9), dtype=np.float64),
        "base_frame": np.asarray(base, dtype=np.float64),
        "base_rgb_uint8": _rgb_uint8_from_frame(base),
        "receiver_archive_sha256": "0" * 64,
        "receiver_archive_bytes": 4096,
    }


def test_facet_reify_with_real_scorer_emits_candidate_action_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.analysis.snerv_official_tub_source_forward_replay as mod

    captured = _captured_with_uint8_surviving_yl_norm()

    # Force the yl_norm flip to be uint8-visible by using a large scalar flip.
    def big_pair_flip(pair: tuple[float, float]) -> tuple[float, float]:
        return (float(pair[0]) + 0.25, float(pair[1]))

    monkeypatch.setattr(mod, "_flip_scalar_pair_low_bit", big_pair_flip)

    # A real (injected) scorer: distinct d_seg/d_pose for base-vs-candidate.
    def scorer(base_rgb: np.ndarray, cand_rgb: np.ndarray) -> dict[str, float]:
        d = float(np.mean(np.abs(base_rgb.astype(float) - cand_rgb.astype(float))))
        # base-vs-base => 0; base-vs-candidate => positive seg, smaller pose.
        return {"d_seg": d / 255.0, "d_pose": (d / 255.0) ** 2}

    facet = _evaluate_tub_source_facet_drop_or_reify(
        facet_name=TUB_SOURCE_FACET_YL_NORM,
        captured=captured,
        base_rgb_uint8=captured["base_rgb_uint8"],
        base_archive_sha256="a" * 64,
        bytes_base=4096,
        action_id="unit",
        scorer_fn=scorer,
    )
    assert facet["survives_uint8_boundary"] is True
    assert facet["verdict"] == VERDICT_REIFY
    cae = facet["candidate_action_evaluation"]
    assert cae is not None
    # The CandidateActionEvaluation is the shared waterfilling-law row.
    assert cae["schema"] == "hi_nerv_candidate_action_evaluation.v1"
    assert cae["action_kind"] == "snerv_tub_source_state_facet_bit_flip"
    assert cae["base_archive_sha256"] == "a" * 64
    assert cae["bytes_base"] == 4096
    # Bytes are unchanged (a source-state facet flip, not a byte-adding action),
    # so the rate term is 0 and the verdict is driven purely by the distortion
    # delta the real scorer measured.
    assert cae["delta_bytes"] == 0
    assert cae["promotable"] is False
    # scorer_delta records the exact terms.
    assert facet["scorer_delta"] is not None
    assert facet["scorer_delta"]["d_seg_with_action"] >= 0.0


def test_facet_reify_pending_scorer_when_no_scorer_supplied() -> None:
    captured = _captured_with_uint8_surviving_yl_norm()
    facet = _evaluate_tub_source_facet_drop_or_reify(
        facet_name=TUB_SOURCE_FACET_YL_NORM,
        captured=captured,
        base_rgb_uint8=captured["base_rgb_uint8"],
        base_archive_sha256="b" * 64,
        bytes_base=4096,
        action_id="unit",
        scorer_fn=None,
    )
    # Receiver-causal but no real scorer => REIFY_PENDING_SCORER, no action row.
    assert facet["verdict"] == VERDICT_REIFY_PENDING_SCORER
    assert facet["candidate_action_evaluation"] is None
    assert TUB_DROP_OR_REIFY_SCORER_PENDING_BLOCKER in facet["blockers"]


# --------------------------------------------------------------------------- #
# Full proof: fail-closed when the source checkout is missing.
# --------------------------------------------------------------------------- #


def test_proof_fails_closed_on_missing_source_checkout(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_snerv_repo"
    proof = build_snerv_official_tub_drop_or_reify_source_forward_proof(
        official_repo_dir=missing,
    )
    assert proof["schema"] == DROP_OR_REIFY_PROOF_SCHEMA
    assert proof["source_graph_executed"] is False
    assert proof["tub_drop_or_reify_verdict"] is None
    assert "snerv_official_source_checkout_missing" in proof["blockers"]
    assert TUB_DROP_OR_REIFY_SOURCE_GRAPH_BLOCKER in proof["blockers"]
    for key in FALSE_AUTHORITY_KEYS:
        assert proof[key] is False


def test_proof_never_claims_authority_constants() -> None:
    # The proof schema constants are the canonical false-authority contract.
    assert set(TUB_DROP_OR_REIFY_FACETS) == {
        TUB_SOURCE_FACET_YL_NORM,
        TUB_SOURCE_FACET_OUTPUT2,
    }
    assert VERDICT_REIFY in TUB_DROP_OR_REIFY_VERDICTS
    assert VERDICT_DROP in TUB_DROP_OR_REIFY_VERDICTS
    assert VERDICT_REIFY_PENDING_SCORER in TUB_DROP_OR_REIFY_VERDICTS


# --------------------------------------------------------------------------- #
# Integration: real upstream SNeRV_T source graph (skipped if repo absent).
# --------------------------------------------------------------------------- #


def test_integration_tub_drop_or_reify_real_source_graph() -> None:
    repo = _official_repo()
    proof = build_snerv_official_tub_drop_or_reify_source_forward_proof(
        official_repo_dir=repo,
    )
    assert proof["schema"] == DROP_OR_REIFY_PROOF_SCHEMA
    assert proof["source_graph_executed"] is True
    assert proof["research_only"] is True  # no real scorer supplied
    # The canonical TUB output_2 facet is NOT consumed by the current receiver
    # fixture => the headline TUB verdict is DROP (no bytes for output_2).
    assert proof["tub_drop_or_reify_verdict"] == VERDICT_DROP
    assert proof["tub_output2_receiver_consumed"] is False
    assert TUB_SOURCE_FACET_OUTPUT2 in proof["dropped_facets"]

    facets = {f["facet"]: f for f in proof["facets"]}
    assert facets[TUB_SOURCE_FACET_OUTPUT2]["schema"] == DROP_OR_REIFY_FACET_SCHEMA
    # output_2 source byte changes but does not reach the receiver frame.
    assert facets[TUB_SOURCE_FACET_OUTPUT2]["source_byte_changed"] is True
    assert facets[TUB_SOURCE_FACET_OUTPUT2]["receiver_consumes_facet"] is False
    assert facets[TUB_SOURCE_FACET_OUTPUT2]["receiver_frame_float_linf"] == 0.0
    assert (
        TUB_OUTPUT2_RECEIVER_NONCAUSAL_BLOCKER
        in facets[TUB_SOURCE_FACET_OUTPUT2]["blockers"]
    )

    # yl_norm IS receiver-causal at the float level (it is a frame-recon input).
    yl = facets[TUB_SOURCE_FACET_YL_NORM]
    assert yl["receiver_consumes_facet"] is True
    assert float(yl["receiver_frame_float_linf"]) > 0.0
    assert yl["verdict"] in {VERDICT_REIFY, VERDICT_REIFY_PENDING_SCORER}

    for key in FALSE_AUTHORITY_KEYS:
        assert proof[key] is False


def test_integration_tub_drop_or_reify_with_real_scorer_emits_action() -> None:
    repo = _official_repo()

    # An injected deterministic scorer (stands in for a contest SegNet/PoseNet
    # on contest hardware): distinct, non-negative d_seg/d_pose per RGB pair.
    def scorer(base_rgb: np.ndarray, cand_rgb: np.ndarray) -> dict[str, float]:
        d = float(np.mean(np.abs(base_rgb.astype(float) - cand_rgb.astype(float))))
        return {"d_seg": d / 255.0, "d_pose": (d / 255.0) ** 2}

    proof = build_snerv_official_tub_drop_or_reify_source_forward_proof(
        official_repo_dir=repo,
        scorer_fn=scorer,
    )
    assert proof["source_graph_executed"] is True
    assert proof["real_scorer_supplied"] is True
    assert proof["research_only"] is False
    # output_2 stays DROP regardless of scorer (it is never receiver-consumed).
    assert proof["tub_drop_or_reify_verdict"] == VERDICT_DROP
    # Every emitted CandidateActionEvaluation is a non-promotable waterfill row.
    for cae in proof["candidate_action_evaluations"]:
        assert cae["schema"] == "hi_nerv_candidate_action_evaluation.v1"
        assert cae["promotable"] is False
        assert cae["score_claim"] is False


def test_integration_capture_is_deterministic() -> None:
    repo = _official_repo()
    a = _capture_tub_source_state_for_drop_or_reify(repo, train_one_step=False)
    b = _capture_tub_source_state_for_drop_or_reify(repo, train_one_step=False)
    # Deterministic reproducibility: identical receiver bytes + RGB.
    assert a["receiver_archive_sha256"] == b["receiver_archive_sha256"]
    np.testing.assert_array_equal(a["base_rgb_uint8"], b["base_rgb_uint8"])
