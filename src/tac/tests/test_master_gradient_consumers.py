# SPDX-License-Identifier: MIT
"""Smoke tests for tac.master_gradient_consumers.

Per audit gate lane `lane_post_landing_audit_gate_per_pair_master_gradient_namespace_wave_20260517`
AXIS 4 deliverable. Covers >=15 tests across 5 consumers + loader helpers +
sidecar JSON emit semantics.

Test taxonomy (per brief):
- 3 tests per consumer x 5 consumers = 15 (contract validation + happy path +
  dtype handling + sidecar JSON emit + axis-tag preservation).
- + loader and helper smoke tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac import master_gradient_consumers as mgc

# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def per_pair_grad():
    """Synthetic per-pair gradient of shape (N_bytes=20, N_pairs=10, 3)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, size=(20, 10, 3)).astype(np.float64)


@pytest.fixture
def aggregate_grad():
    """Synthetic aggregate gradient of shape (N_bytes=20, 3)."""
    rng = np.random.default_rng(43)
    return rng.normal(0.0, 1.0, size=(20, 3)).astype(np.float64)


@pytest.fixture
def archive_sha256():
    return "a" * 64


@pytest.fixture
def axis_meta():
    return {
        "measurement_axis": "contest_cpu",
        "measurement_hardware": "linux_x86_64_cpu",
    }


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Redirect CONSUMER_OUTPUT_ROOT to tmp_path so sidecars do not pollute repo state."""
    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", tmp_path / "master_gradient_consumers")
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 1 — Venn classification (3 tests)                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_venn_classification_contract_rejects_2d(archive_sha256, axis_meta):
    """Contract validation: per_pair_gradient with wrong shape rejected."""
    bad = np.zeros((20, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
        mgc.classify_bytes_by_pair_variance(
            bad, archive_sha256=archive_sha256, write_sidecar=False, **axis_meta
        )


def test_venn_classification_happy_path_returns_typed_result(
    per_pair_grad, archive_sha256, axis_meta
):
    """Happy path: returns PerByteVennClassification with all required fields."""
    result = mgc.classify_bytes_by_pair_variance(
        per_pair_grad, archive_sha256=archive_sha256, write_sidecar=False, **axis_meta
    )
    assert isinstance(result, mgc.PerByteVennClassification)
    assert result.n_bytes == 20
    assert result.n_pairs == 10
    assert result.classes.shape == (20,)
    # All bytes assigned to one of the canonical classes
    assert set(result.classes) <= set(mgc.PerByteVennClass.ALL)
    # class_counts sums to n_bytes
    assert sum(result.class_counts.values()) == 20


def test_venn_classification_sidecar_carries_compliance_tags(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar JSON: compliance tags injected per Apples-to-apples discipline."""
    mgc.classify_bytes_by_pair_variance(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=True,
        **axis_meta,
    )
    sidecars = list((tmp_root / "master_gradient_consumers").glob("venn_classification_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    # Compliance tags injected by write_consumer_sidecar_json
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"].startswith("[diagnostic")
    # Axis tag preserved verbatim
    assert payload["measurement_axis"] == "contest_cpu"
    assert payload["measurement_hardware"] == "linux_x86_64_cpu"


def test_venn_classification_sidecar_can_emit_score_weighted_custody(
    per_pair_grad, archive_sha256, tmp_root
):
    """Canonical Cathedral path consumes score-weighted counts only with custody."""
    gradient_path = tmp_root / "venn_gradient.npy"
    result = mgc.classify_bytes_by_pair_variance(
        per_pair_grad,
        archive_sha256=archive_sha256,
        measurement_axis="[contest-CPU]",
        measurement_hardware="linux_x86_64_modal_cpu",
        operating_point=mgc.OperatingPoint(d_seg=0.1, d_pose=0.1, rate=0.1, score=0.1),
        measurement_method="autograd_full_pairs",
        measurement_call_id="modal-call-venn",
        scored_archive_sha256=archive_sha256,
        scored_archive_bytes=12345,
        gradient_array_sha256="b" * 64,
        gradient_array_path=str(gradient_path),
        gradient_tensor_kind="per_pair_per_byte_v1",
        n_pairs_used=10,
        n_pairs_total=10,
        write_sidecar=True,
    )

    sidecars = list((tmp_root / "master_gradient_consumers").glob("venn_classification_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    assert result.score_weighted_class_counts is not None
    assert payload["score_weighted_class_counts"] == result.score_weighted_class_counts
    assert payload["score_axis_weighting"]["available"] is True
    assert payload["scored_archive_sha256"] == archive_sha256
    assert payload["gradient_array_sha256"] == "b" * 64
    assert payload["source_custody"]["custody_required_for_cathedral_reward"] is True
    assert payload["n_pairs"] == per_pair_grad.shape[1]
    assert payload["n_pairs_used"] == 10
    assert payload["n_pairs_total"] == 10


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 2 — fec6 selector marginal matrix (3 tests)                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_fec6_marginal_matrix_contract_rejects_mismatched_modes(
    per_pair_grad, archive_sha256, axis_meta
):
    """Contract validation: current_modes shape must match (n_pairs,)."""
    bad_modes = np.zeros((5,), dtype=np.int64)
    with pytest.raises(ValueError, match="current_modes"):
        mgc.fec6_selector_marginal_matrix(
            per_pair_grad,
            archive_sha256=archive_sha256,
            selector_byte_indices=list(range(10)),
            current_modes=bad_modes,
            n_modes=16,
            write_sidecar=False,
            **axis_meta,
        )


def test_fec6_marginal_matrix_happy_path(per_pair_grad, archive_sha256, axis_meta):
    """Happy path: per (pair, candidate_mode) cell ΔS estimate populated."""
    rng = np.random.default_rng(44)
    n_pairs = per_pair_grad.shape[1]
    current_modes = rng.integers(0, 16, size=n_pairs).astype(np.int64)
    sel_idx = list(range(n_pairs))  # one selector byte per pair, all distinct
    result = mgc.fec6_selector_marginal_matrix(
        per_pair_grad,
        archive_sha256=archive_sha256,
        selector_byte_indices=sel_idx,
        current_modes=current_modes,
        n_modes=16,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(result, mgc.Fec6SelectorMarginalMatrix)
    assert result.n_pairs == n_pairs
    assert result.n_modes == 16
    # cells = n_pairs * (n_modes - 1) (skip k == current_mode)
    assert len(result.cells) == n_pairs * 15


def test_fec6_marginal_matrix_sidecar_emits_top_100(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar JSON: emits top 100 score-lowering candidates only."""
    rng = np.random.default_rng(44)
    n_pairs = per_pair_grad.shape[1]
    current_modes = rng.integers(0, 16, size=n_pairs).astype(np.int64)
    sel_idx = list(range(n_pairs))
    mgc.fec6_selector_marginal_matrix(
        per_pair_grad,
        archive_sha256=archive_sha256,
        selector_byte_indices=sel_idx,
        current_modes=current_modes,
        n_modes=16,
        write_sidecar=True,
        **axis_meta,
    )
    sidecars = list((tmp_root / "master_gradient_consumers").glob("fec6_selector_marginal_matrix_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    assert payload["consumer_id"] == "fec6_selector_marginal_matrix"
    assert len(payload["top_100_score_lowering_swaps"]) <= 100
    # Sorted ascending by predicted_delta_s
    deltas = [c["predicted_delta_s"] for c in payload["top_100_score_lowering_swaps"]]
    assert deltas == sorted(deltas)


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 3 — NSCS01 nullspace audit (3 tests)                                 #
# ──────────────────────────────────────────────────────────────────────────── #


def test_nscs01_nullspace_audit_insufficient_data(per_pair_grad, archive_sha256, axis_meta):
    """No frame_0 indices supplied → INSUFFICIENT_DATA verdict."""
    result = mgc.nscs01_nullspace_empirical_audit(
        per_pair_grad,
        archive_sha256=archive_sha256,
        frame_0_only_byte_indices=[],
        write_sidecar=False,
        **axis_meta,
    )
    assert result.verdict == mgc.Nscs01NullspaceVerdict.INSUFFICIENT_DATA
    assert result.n_frame_0_only_bytes == 0


def test_nscs01_nullspace_audit_confirms_when_seg_zero():
    """When seg axis (0) is exactly zero on all frame_0 bytes → CONFIRMED."""
    # Construct gradient where bytes 0..4 have seg=0 always, pose/rate nonzero
    n_bytes, n_pairs = 10, 5
    grad = np.random.default_rng(0).normal(0.0, 1.0, size=(n_bytes, n_pairs, 3))
    grad[0:5, :, 0] = 0.0  # frame_0 bytes have zero seg gradient
    result = mgc.nscs01_nullspace_empirical_audit(
        grad,
        archive_sha256="b" * 64,
        measurement_axis="contest_cpu",
        measurement_hardware="linux_x86_64_cpu",
        frame_0_only_byte_indices=[0, 1, 2, 3, 4],
        write_sidecar=False,
    )
    assert result.verdict == mgc.Nscs01NullspaceVerdict.CONFIRMED
    assert result.fraction_pairs_nonzero == 0.0
    assert result.max_seg_gradient_magnitude_on_frame_0_bytes == 0.0


def test_nscs01_nullspace_audit_falsifies_when_seg_nonzero():
    """When seg axis NON-zero on frame_0 bytes → FALSIFIED."""
    n_bytes, n_pairs = 10, 5
    grad = np.ones((n_bytes, n_pairs, 3), dtype=np.float64)  # all 1s; seg is nonzero everywhere
    result = mgc.nscs01_nullspace_empirical_audit(
        grad,
        archive_sha256="c" * 64,
        measurement_axis="contest_cpu",
        measurement_hardware="linux_x86_64_cpu",
        frame_0_only_byte_indices=[0, 1, 2, 3, 4],
        write_sidecar=False,
    )
    assert result.verdict == mgc.Nscs01NullspaceVerdict.FALSIFIED
    assert result.fraction_pairs_nonzero == 1.0


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 4 — Wyner-Ziv side-info covariance (3 tests)                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_wyner_ziv_contract_rejects_bad_sample_axis(per_pair_grad, archive_sha256, axis_meta):
    """Contract validation: sample_axis must be 0, 1, or 2."""
    with pytest.raises(ValueError, match="sample_axis"):
        mgc.wyner_ziv_side_info_covariance(
            per_pair_grad,
            archive_sha256=archive_sha256,
            sample_axis=5,
            write_sidecar=False,
            **axis_meta,
        )


def test_wyner_ziv_happy_path_classifies_bytes(per_pair_grad, archive_sha256, axis_meta):
    """Happy path: returns three disjoint byte index sets."""
    result = mgc.wyner_ziv_side_info_covariance(
        per_pair_grad,
        archive_sha256=archive_sha256,
        sample_axis=1,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(result, mgc.WynerZivSideInfoClassification)
    # Three classes partition: shared_prior + pair_specific + mixed = n_bytes
    total = (
        len(result.candidate_shared_prior_byte_indices)
        + len(result.pair_specific_byte_indices)
        + len(result.mixed_byte_indices)
    )
    assert total == result.n_bytes
    # No overlap
    sp = set(result.candidate_shared_prior_byte_indices)
    ps = set(result.pair_specific_byte_indices)
    mx = set(result.mixed_byte_indices)
    assert sp.isdisjoint(ps) and sp.isdisjoint(mx) and ps.isdisjoint(mx)


def test_wyner_ziv_sidecar_preserves_axis(per_pair_grad, archive_sha256, axis_meta, tmp_root):
    """Sidecar: axis_name field reflects sample_axis."""
    mgc.wyner_ziv_side_info_covariance(
        per_pair_grad,
        archive_sha256=archive_sha256,
        sample_axis=0,  # seg
        write_sidecar=True,
        **axis_meta,
    )
    sidecars = list((tmp_root / "master_gradient_consumers").glob("wyner_ziv_side_info_covariance_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    assert payload["sample_axis"] == 0
    assert payload["axis_name"] == "seg"


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 5 — Per-pair difficulty atlas (3 tests)                              #
# ──────────────────────────────────────────────────────────────────────────── #


def test_per_pair_difficulty_contract_rejects_wrong_shape(archive_sha256, axis_meta):
    """Contract validation: shape (N_bytes, N_pairs, 3) required."""
    bad = np.zeros((20, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
        mgc.per_pair_difficulty_atlas(
            bad, archive_sha256=archive_sha256, write_sidecar=False, **axis_meta
        )


def test_per_pair_difficulty_atlas_ranks_pairs(per_pair_grad, archive_sha256, axis_meta):
    """Happy path: all pairs ranked; top-K hardest + bottom-K easiest emitted."""
    result = mgc.per_pair_difficulty_atlas(
        per_pair_grad,
        archive_sha256=archive_sha256,
        top_k=3,
        bottom_k=3,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(result, mgc.PerPairDifficultyAtlas)
    assert result.n_pairs == 10
    assert len(result.entries) == 10
    assert len(result.top_k_hardest_pair_indices) == 3
    assert len(result.bottom_k_easiest_pair_indices) == 3
    # Ranks are consecutive 0..n_pairs-1
    ranks = sorted(e.difficulty_rank for e in result.entries)
    assert ranks == list(range(10))
    # Top-K is harder than bottom-K
    top_norms = [
        e.gradient_norm_l2
        for e in result.entries
        if e.pair_index in result.top_k_hardest_pair_indices
    ]
    bottom_norms = [
        e.gradient_norm_l2
        for e in result.entries
        if e.pair_index in result.bottom_k_easiest_pair_indices
    ]
    assert min(top_norms) >= max(bottom_norms)


def test_per_pair_difficulty_atlas_sidecar_emits_axis_breakdown(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar: per-pair raw + score-weighted axis contributions emitted."""
    gradient_path = tmp_root / "per_pair_gradient.npy"
    mgc.per_pair_difficulty_atlas(
        per_pair_grad,
        archive_sha256=archive_sha256,
        top_k=5,
        bottom_k=5,
        operating_point=mgc.OperatingPoint(d_seg=0.1, d_pose=0.1, rate=0.1, score=0.1),
        measurement_method="autograd_full_pairs",
        measurement_call_id="modal-call-123",
        scored_archive_sha256=archive_sha256,
        scored_archive_bytes=12345,
        gradient_array_sha256="a" * 64,
        gradient_array_path=str(gradient_path),
        gradient_tensor_kind="per_pair_per_byte_v1",
        n_pairs_used=10,
        n_pairs_total=10,
        write_sidecar=True,
        **axis_meta,
    )
    sidecars = list((tmp_root / "master_gradient_consumers").glob("per_pair_difficulty_atlas_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text())
    assert payload["consumer_id"] == "per_pair_difficulty_atlas"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["scored_archive_sha256"] == archive_sha256
    assert payload["scored_archive_bytes"] == 12345
    assert payload["gradient_array_sha256"] == "a" * 64
    assert payload["gradient_array_path"] == str(gradient_path)
    assert payload["gradient_tensor_kind"] == "per_pair_per_byte_v1"
    assert payload["measurement_method"] == "autograd_full_pairs"
    assert payload["measurement_call_id"] == "modal-call-123"
    assert payload["n_pairs"] == per_pair_grad.shape[1]
    assert payload["n_pairs_used"] == 10
    assert payload["n_pairs_total"] == 10
    assert payload["source_custody"]["n_pairs"] == per_pair_grad.shape[1]
    assert payload["source_custody"]["custody_required_for_cathedral_reward"] is True
    assert payload["score_axis_weighting"]["available"] is True
    assert len(payload["top_k_hardest_with_axis_breakdown"]) == 5
    assert len(payload["top_k_score_weighted_pair_indices"]) == 5
    assert len(payload["top_k_score_weighted_with_axis_breakdown"]) == 5
    for entry in payload["top_k_hardest_with_axis_breakdown"]:
        # Per-axis L1 contributions present
        for key in (
            "seg_axis_l1",
            "pose_axis_l1",
            "rate_axis_l1",
            "seg_axis_score_l1",
            "pose_axis_score_l1",
            "rate_axis_score_l1",
            "pose_score_axis_share",
        ):
            assert key in entry
            assert isinstance(entry[key], (int, float))
    for entry in payload["top_k_score_weighted_with_axis_breakdown"]:
        assert "pose_score_axis_share" in entry
        assert isinstance(entry["pose_score_axis_share"], (int, float))


# ──────────────────────────────────────────────────────────────────────────── #
# Loader + helper smoke tests                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_consumer_output_path_canonical_no_tmp(archive_sha256):
    """consumer_output_path never produces /tmp paths."""
    path = mgc.consumer_output_path("venn_classification", archive_sha256=archive_sha256)
    s = str(path)
    assert "/tmp/" not in s
    assert not s.startswith("/tmp")
    assert ".omx/state/master_gradient_consumers" in s
    assert archive_sha256[:12] in s


def test_consumer_output_path_includes_utc(archive_sha256):
    """consumer_output_path filename includes a UTC timestamp segment."""
    path = mgc.consumer_output_path(
        "venn_classification", archive_sha256=archive_sha256, utc_iso="2026-05-17T20:00:00"
    )
    assert "20260517T200000" in str(path)


def test_write_consumer_sidecar_json_injects_compliance_tags(tmp_path):
    """write_consumer_sidecar_json injects the canonical compliance tags
    (score_claim=False, promotion_eligible=False, ready_for_exact_eval_dispatch=False,
    evidence_grade starts with [diagnostic)."""
    target = tmp_path / "out" / "sidecar.json"
    mgc.write_consumer_sidecar_json(target, {"foo": "bar"})
    payload = json.loads(target.read_text())
    assert payload["foo"] == "bar"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"].startswith("[diagnostic")


def test_load_per_pair_gradient_from_anchor_missing_raises():
    """No matching anchor in ledger → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no per-pair master gradient anchor"):
        mgc.load_per_pair_gradient_from_anchor(archive_sha256="z" * 64)


def test_load_aggregate_gradient_from_anchor_missing_raises():
    """No matching aggregate anchor in ledger → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="no aggregate master gradient anchor"):
        mgc.load_aggregate_gradient_from_anchor(archive_sha256="z" * 64)


def test_load_per_pair_gradient_accepts_diagnostic_advisory_anchor(tmp_path: Path):
    """Per-pair gradients stay usable for training/compress/inflate planning when tagged advisory."""
    archive = "f" * 64
    arr = np.zeros((5, 3, 3), dtype=np.float32)
    npy_path = tmp_path / "per_pair.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "per_pair_per_byte_v1",
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_pair_subset_axis_corrected",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 5,
        "n_pairs": 3,
        "n_pairs_used": 3,
        "n_pairs_total": 600,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")

    loaded, loaded_anchor = mgc.load_per_pair_gradient_from_anchor(
        archive_sha256=archive,
        anchor_path=ledger,
    )

    assert loaded.shape == (5, 3, 3)
    assert loaded_anchor["measurement_axis"] == "[macOS-CPU advisory]"


def test_load_per_pair_gradient_uses_tensor_kind_over_method_name(tmp_path: Path):
    """Per-pair anchor selection must not depend on fragile measurement_method text."""
    archive = "e" * 64
    arr = np.zeros((5, 3, 3), dtype=np.float32)
    npy_path = tmp_path / "per_pair.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "per_pair_per_byte_v1",
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_axis_corrected_no_literal_kind_token",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 5,
        "n_pairs": 3,
        "n_pairs_used": 3,
        "n_pairs_total": 600,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")

    loaded, loaded_anchor = mgc.load_per_pair_gradient_from_anchor(
        archive_sha256=archive,
        anchor_path=ledger,
    )

    assert loaded.shape == (5, 3, 3)
    assert loaded_anchor["gradient_tensor_kind"] == "per_pair_per_byte_v1"


def test_load_per_pair_gradient_rejects_false_contest_axis_subset_anchor(
    tmp_path: Path,
):
    """Mislabeled contest-axis per-pair subsets must not feed downstream consumers."""
    archive = "f" * 64
    arr = np.zeros((5, 3, 3), dtype=np.float32)
    npy_path = tmp_path / "per_pair.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "per_pair_per_byte_v1",
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_pair_8pair_subset",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 5,
        "n_pairs": 3,
        "n_pairs_used": 3,
        "n_pairs_total": 600,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")

    with pytest.raises(FileNotFoundError, match="no per-pair master gradient anchor"):
        mgc.load_per_pair_gradient_from_anchor(
            archive_sha256=archive,
            anchor_path=ledger,
        )


def test_load_aggregate_gradient_uses_axis_correction_instead_of_stale_contest_row(
    tmp_path: Path,
):
    """Aggregate consumers use the latest advisory correction row, not stale contest authority."""
    archive = "a" * 64
    old_arr = np.zeros((4, 3), dtype=np.float32)
    new_arr = np.ones((4, 3), dtype=np.float32)
    old_path = tmp_path / "old.npy"
    new_path = tmp_path / "new.npy"
    np.save(old_path, old_arr)
    np.save(new_path, new_arr)
    base = {
        "archive_sha256": archive,
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "n_bytes": 4,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    stale = {
        **base,
        "gradient_array_path": str(old_path),
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_parameter_projected_8pair_subset",
        "measurement_utc": "2026-05-18T00:00:00Z",
    }
    correction = {
        **base,
        "gradient_array_path": str(new_path),
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_parameter_projected_8pair_subset_axis_correction",
        "measurement_utc": "2026-05-18T01:00:00Z",
    }
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(stale) + "\n" + json.dumps(correction) + "\n")

    loaded, loaded_anchor = mgc.load_aggregate_gradient_from_anchor(
        archive_sha256=archive,
        anchor_path=ledger,
    )

    assert float(loaded.sum()) == pytest.approx(12.0)
    assert loaded_anchor["measurement_axis"] == "[macOS-CPU advisory]"


def test_load_aggregate_gradient_prefers_later_append_correction_for_same_measurement(
    tmp_path: Path,
):
    """Append-only metadata fixes with same measurement time must become effective."""
    archive = "b" * 64
    arr = np.ones((4, 3), dtype=np.float32)
    npy_path = tmp_path / "aggregate.npy"
    np.save(npy_path, arr)
    base = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 4,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    stale = {**base, "written_at_utc": "2026-05-18T01:00:01Z"}
    corrected = {
        **base,
        "written_at_utc": "2026-05-18T01:00:02Z",
        "score_axis_dominance": {"schema": "master_gradient_score_axis_dominance_v1"},
    }
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(stale) + "\n" + json.dumps(corrected) + "\n")

    _loaded, loaded_anchor = mgc.load_aggregate_gradient_from_anchor(
        archive_sha256=archive,
        anchor_path=ledger,
    )

    assert loaded_anchor["score_axis_dominance"]["schema"] == "master_gradient_score_axis_dominance_v1"


def test_load_aggregate_gradient_uses_tensor_kind_over_per_pair_method_text(
    tmp_path: Path,
):
    """Aggregate anchor selection must honor gradient_tensor_kind first."""
    archive = "b" * 64
    arr = np.ones((4, 3), dtype=np.float32)
    npy_path = tmp_path / "aggregate.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection_regressed_from_per_pair_basis",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 4,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")

    loaded, loaded_anchor = mgc.load_aggregate_gradient_from_anchor(
        archive_sha256=archive,
        anchor_path=ledger,
    )

    assert loaded.shape == (4, 3)
    assert loaded_anchor["gradient_tensor_kind"] == "aggregate_per_byte_v1"


def test_select_pose_axis_dominant_bytes_emits_typed_specs_and_sidecar(tmp_path: Path):
    archive = "c" * 64
    # d_pose=0.1 gives pose marginal ~=15.811, so row 1 is strongly pose-dominant.
    arr = np.array(
        [
            [10.0, 0.01, 0.0],
            [0.01, 10.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.01, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    npy_path = tmp_path / "aggregate.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "scored_archive_sha256": archive,
        "scored_archive_bytes": 12345,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "gradient_byte_domain": "zip_inner_member_payload",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 4,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")
    output_root = tmp_path / "consumers"

    specs = mgc.select_pose_axis_dominant_bytes(
        archive,
        top_k=2,
        axis_dominance_threshold=0.7,
        anchor_path=ledger,
        output_root=output_root,
    )

    assert len(specs) == 2
    assert all(spec.mutation_grain == "grammar_aware_operator" for spec in specs)
    assert all(spec.raw_archive_byte_coordinates_allowed is False for spec in specs)
    assert all(spec.score_claim is False for spec in specs)
    assert specs[0].axis_label == "pose"
    assert "diagnostic_gradient_subject_byte_index=1" in specs[0].rationale
    sidecars = list(output_root.glob(f"pose_axis_dominant_bytes_{archive[:12]}_*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["consumer_id"] == "select_pose_axis_dominant_bytes"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["score_axis_dominance"]["selected_count"] == 2
    assert payload["score_axis_dominance"]["selected"][0]["diagnostic_gradient_subject_byte_index"] == 1
    assert payload["candidate_modification_specs"][0]["coordinate_system"] == "grammar_aware_operator_response"


def test_select_pose_axis_dominant_bytes_does_not_invent_scored_custody(tmp_path: Path):
    archive = "c" * 64
    arr = np.array(
        [
            [0.01, 10.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    npy_path = tmp_path / "aggregate.npy"
    np.save(npy_path, arr)
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    anchor = {
        "archive_sha256": archive,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "gradient_byte_domain": "scored_archive_bytes",
        "measurement_axis": "[diagnostic-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "measurement_method": "aggregate_projection",
        "measurement_utc": "2026-05-18T01:00:00Z",
        "n_bytes": 2,
        "operating_point": {"d_pose": 0.1, "d_seg": 0.1, "rate": 0.1, "score": 0.1},
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(anchor) + "\n")
    output_root = tmp_path / "consumers"
    sidecar_path = output_root / "pose_axis_dominant_bytes_custody.json"

    specs = mgc.select_pose_axis_dominant_bytes(
        archive,
        top_k=1,
        axis_dominance_threshold=0.7,
        anchor_path=ledger,
        output_root=output_root,
        sidecar_path=sidecar_path,
    )

    assert len(specs) == 1
    assert specs[0].source_archive_sha256 is None
    assert specs[0].source_archive_bytes is None
    assert specs[0].section_name == "diagnostic_uncustodied_gradient_subject_bytes"
    assert "scored_archive_custody_missing" in specs[0].blockers
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["scored_archive_sha256"] is None
    assert payload["scored_archive_bytes"] is None
    assert payload["scored_archive_custody_available"] is False
    assert "scored_archive_custody_missing" in payload["blockers"]


def test_select_pose_axis_dominant_bytes_validates_thresholds(tmp_path: Path):
    with pytest.raises(ValueError, match="top_k"):
        mgc.select_pose_axis_dominant_bytes("d" * 64, top_k=0, anchor_path=tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match="axis_dominance_threshold"):
        mgc.select_pose_axis_dominant_bytes(
            "d" * 64,
            axis_dominance_threshold=1.2,
            anchor_path=tmp_path / "missing.jsonl",
        )


def test_public_api_completeness():
    """Verify all public symbols are exported via __all__ and importable.

    The module implements 6 consumers in v1:
      1. Venn classification
      2. fec6 selector marginal matrix
      3. NSCS01 nullspace empirical audit
      4. Wyner-Ziv side-info covariance
      5. Per-pair difficulty atlas
      6. Rashomon disagreement queue
    Plus loader + sidecar helpers.
    """
    expected = {
        # Consumer 1
        "PerByteVennClass",
        "PerByteVennClassification",
        "classify_bytes_by_pair_variance",
        # Consumer 2
        "Fec6SelectorMarginalCell",
        "Fec6SelectorMarginalMatrix",
        "fec6_selector_marginal_matrix",
        # Consumer 3
        "Nscs01NullspaceVerdict",
        "Nscs01NullspaceAudit",
        "nscs01_nullspace_empirical_audit",
        # Consumer 4
        "WynerZivSideInfoClassification",
        "wyner_ziv_side_info_covariance",
        # Consumer 5
        "PerPairDifficultyEntry",
        "PerPairDifficultyAtlas",
        "per_pair_difficulty_atlas",
        # OP-7 cheap-probe bridge
        "select_pose_axis_dominant_bytes",
        # Consumer 6
        "RashomonDisagreementEntry",
        "RashomonDisagreementQueue",
        "rashomon_disagreement_queue",
        # Consumer 15 — operator-binding Lagrangian-dual per-pair treatment planner
        "Treatment",
        "TreatmentCatalog",
        "Budget",
        "PairTreatmentAssignment",
        "OptimalPerPairTreatmentPlan",
        "OptimalPerPairTreatmentPlanError",
        "DEFAULT_TREATMENT_CATALOG",
        "DEFAULT_BUDGET",
        "DEFAULT_ARCHIVE_BUDGET_BYTES",
        "DEFAULT_COMPUTE_BUDGET_USD",
        "DEFAULT_INFLATE_BUDGET_SECONDS",
        "TREATMENT_NONE",
        "TREATMENT_LORA_RANK_8",
        "TREATMENT_LAMBDA_R_BUMP",
        "TREATMENT_PER_PAIR_PARETO_ENVELOPE",
        "TREATMENT_KKT_RESIDUAL_CORRECTION",
        "TREATMENT_VOLTERRA_CROSS_TERM",
        "TREATMENT_DECODER_PRUNING",
        "TREATMENT_WYNER_ZIV_HOIST",
        "build_default_treatment_catalog",
        "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "optimal_plan_to_candidate_row",
        "optimal_plan_payload_to_candidate_row",
        # Loaders + helpers
        "load_per_pair_gradient_from_anchor",
        "load_aggregate_gradient_from_anchor",
        "load_optimal_plan_for_archive",
        "consumer_output_path",
        "write_consumer_sidecar_json",
    }
    # Cable D D3 v3 wave 2026-05-19 added consumers 7-14 to __all__; use
    # superset semantics so the original v1+v15 contract is preserved while
    # the v3 expansion is additive. The new consumer 7-14 names are exercised
    # by their dedicated test file `test_master_gradient_consumers_7_to_14.py`.
    assert expected.issubset(set(mgc.__all__))
    for name in expected:
        assert hasattr(mgc, name), f"missing public symbol {name}"


# ──────────────────────────────────────────────────────────────────────────── #
# load_optimal_plan_for_archive — Q3 v2 cascade canonical loader (5 tests)      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_load_optimal_plan_for_archive_returns_none_when_missing(tmp_path):
    """Missing root dir returns None (no plan available)."""
    nonexistent = tmp_path / "nope"
    sha = "a" * 64
    result = mgc.load_optimal_plan_for_archive(sha, root=nonexistent)
    assert result is None


def test_load_optimal_plan_for_archive_returns_none_when_no_match(tmp_path):
    """Existing root dir but no matching file => None."""
    root = tmp_path / "consumers"
    root.mkdir()
    # Write a sister consumer's sidecar — not optimal_plan_*
    (root / "venn_classification_aaa_2026.json").write_text("{}", encoding="utf-8")

    sha = "a" * 64
    result = mgc.load_optimal_plan_for_archive(sha, root=root)
    assert result is None


def test_load_optimal_plan_for_archive_short_sha_raises():
    """archive_sha256 shorter than 12 chars raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="12\\+ char hex"):
        mgc.load_optimal_plan_for_archive("short")


def test_load_optimal_plan_for_archive_finds_most_recent(tmp_path):
    """Multiple plan sidecars for same archive: most recent (lex-max) wins."""
    root = tmp_path / "consumers"
    root.mkdir()
    sha = "b" * 64
    base_payload = {
        "archive_sha256": sha,
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": mgc.OPTIMAL_PLAN_CONSUMER_ID,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "schema": "v1",
    }
    # Older
    (root / f"optimal_plan_{sha[:12]}_20260517T100000.json").write_text(
        json.dumps({**base_payload, "predicted_score_delta": -0.001}), encoding="utf-8"
    )
    # Newer
    (root / f"optimal_plan_{sha[:12]}_20260517T200000.json").write_text(
        json.dumps({**base_payload, "predicted_score_delta": -0.099}), encoding="utf-8"
    )

    payload = mgc.load_optimal_plan_for_archive(sha, root=root)
    assert payload is not None
    assert payload["predicted_score_delta"] == -0.099


def test_load_optimal_plan_for_archive_skips_corrupt(tmp_path):
    """Corrupt JSON file skipped; falls through to older valid file."""
    root = tmp_path / "consumers"
    root.mkdir()
    sha = "c" * 64
    valid_payload = {
        "archive_sha256": sha,
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": mgc.OPTIMAL_PLAN_CONSUMER_ID,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "predicted_score_delta": -0.042,
        "schema": "v1",
    }
    # Older valid plan
    (root / f"optimal_plan_{sha[:12]}_20260517T100000.json").write_text(
        json.dumps(valid_payload), encoding="utf-8"
    )
    # Newer CORRUPT plan
    (root / f"optimal_plan_{sha[:12]}_20260517T200000.json").write_text(
        "NOT VALID JSON", encoding="utf-8"
    )

    payload = mgc.load_optimal_plan_for_archive(sha, root=root)
    # Most-recent valid wins; corrupt skipped
    assert payload is not None
    assert payload["predicted_score_delta"] == -0.042


def test_load_optimal_plan_for_archive_skips_payload_archive_mismatch(tmp_path):
    """Filename prefix alone is not authority; payload archive_sha256 must match."""
    root = tmp_path / "consumers"
    root.mkdir()
    sha = "d" * 64
    wrong_sha = sha[:12] + ("e" * 52)
    (root / f"optimal_plan_{sha[:12]}_20260517T100000.json").write_text(
        json.dumps(
            {
                "archive_sha256": wrong_sha,
                "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
                "catalog_consumer_id": mgc.OPTIMAL_PLAN_CONSUMER_ID,
                "evidence_grade": "predicted",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "predicted_score_delta": -999.0,
                "schema": "v1",
            }
        ),
        encoding="utf-8",
    )

    assert mgc.load_optimal_plan_for_archive(sha, root=root) is None


def test_load_optimal_plan_for_archive_skips_authority_bearing_payload(tmp_path):
    """Predicted plans must not carry score or promotion authority into ranking."""
    root = tmp_path / "consumers"
    root.mkdir()
    sha = "e" * 64
    base_payload = {
        "archive_sha256": sha,
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": mgc.OPTIMAL_PLAN_CONSUMER_ID,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "predicted_score_delta": -0.042,
        "schema": "v1",
    }
    (root / f"optimal_plan_{sha[:12]}_20260517T100000.json").write_text(
        json.dumps({**base_payload, "score_claim": True}),
        encoding="utf-8",
    )
    (root / f"optimal_plan_{sha[:12]}_20260517T200000.json").write_text(
        json.dumps({**base_payload, "evidence_grade": "contest-CUDA"}),
        encoding="utf-8",
    )

    assert mgc.load_optimal_plan_for_archive(sha, root=root) is None


def _minimal_optimal_plan_payload(sha: str) -> dict:
    return {
        "archive_sha256": sha,
        "consumer_id": "per_pair_optimal_treatment_plan_via_lagrangian_dual",
        "catalog_consumer_id": mgc.OPTIMAL_PLAN_CONSUMER_ID,
        "evidence_grade": "predicted",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64_cpu",
        "treatment_catalog_sha": "abc123",
        "predicted_score_delta": -0.012,
        "predicted_score_delta_confidence_interval": [-0.014, -0.010],
        "budget": {"compute_usd": 1.25},
        "kkt_residual": 0.001,
        "feasibility_certificate": {"archive_bytes": True},
        "is_pareto_feasible": True,
    }


def test_optimal_plan_payload_to_candidate_row_is_planning_only(tmp_path):
    sha = "f" * 64
    row = mgc.optimal_plan_payload_to_candidate_row(
        _minimal_optimal_plan_payload(sha),
        sidecar_path=tmp_path / "optimal_plan.json",
    )
    assert row.archive_sha256 == sha
    assert row.predicted_score_delta == -0.012
    assert row.estimated_dispatch_cost_usd == 1.25
    assert row.score_claim is False
    assert row.promotion_eligible is False
    assert row.ready_for_exact_eval_dispatch is False
    assert "planning_only_master_gradient_optimal_plan_no_dispatch_packet" in row.blockers


def test_optimal_plan_payload_to_candidate_row_rejects_authority_flags():
    payload = _minimal_optimal_plan_payload("1" * 64)
    payload["ready_for_exact_eval_dispatch"] = True
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError):
        mgc.optimal_plan_payload_to_candidate_row(payload)


def test_optimal_plan_to_candidate_row_live_dataclass_is_planning_only():
    sha = "2" * 64
    plan = mgc.OptimalPerPairTreatmentPlan(
        plan=(
            mgc.PairTreatmentAssignment(
                pair_idx=0,
                treatment_id=mgc.TREATMENT_LORA_RANK_8,
                theta=0.5,
                predicted_delta_seg=-0.001,
                predicted_delta_pose=-0.0001,
                predicted_delta_rate_bytes=3,
                predicted_delta_s_contribution=-0.002,
            ),
        ),
        lambda_archive=0.1,
        lambda_compute=0.2,
        lambda_inflate=0.3,
        nu_per_pair=(0.4,),
        kkt_residual=0.001,
        feasibility_certificate={"archive": True},
        predicted_score_delta=-0.002,
        predicted_score_delta_confidence_interval=(-0.003, -0.001),
        operating_point={"d_seg": 0.0, "d_pose": 1e-4, "R": 100.0},
        treatment_catalog_sha="liveabc",
        archive_sha256_anchor=sha,
        n_admm_iterations=3,
        warm_start_heuristic_used=False,
        measurement_axis="[diagnostic]",
        measurement_hardware="unit-test",
        is_pareto_feasible=True,
    )

    row = mgc.optimal_plan_to_candidate_row(plan)
    assert row.archive_sha256 == sha
    assert row.predicted_score_delta == pytest.approx(-0.002)
    assert row.dispatch_packet_ready is False
    assert row.target_modes == []
    assert row.score_claim is False
    assert row.promotion_eligible is False
    assert row.ready_for_exact_eval_dispatch is False


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 6 — Rashomon disagreement queue (3 tests)                            #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_disagreement_queue_smoke(per_pair_grad, archive_sha256, axis_meta):
    """Happy path: queue computes per-byte disagreement across K members."""
    result = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(result, mgc.RashomonDisagreementQueue)
    # One entry per byte
    assert len(result.entries_tuple) == per_pair_grad.shape[0]


def test_rashomon_disagreement_queue_entries_have_required_fields(
    per_pair_grad, archive_sha256, axis_meta
):
    """Every entry has the required fields per the contract."""
    result = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=False,
        **axis_meta,
    )
    for entry in result.entries_tuple:
        assert isinstance(entry, mgc.RashomonDisagreementEntry)
        assert entry.byte_index >= 0
        assert entry.std_across_k_members >= 0.0
        assert entry.k_members_count >= 1
        # Axis tag preserved per Apples-to-apples discipline
        assert entry.axis == "contest_cpu"


def test_rashomon_disagreement_queue_dtype_handling(archive_sha256, axis_meta):
    """Accepts float32 input without crashing (dtype handling)."""
    rng = np.random.default_rng(99)
    grad_f32 = rng.normal(0.0, 1.0, size=(15, 8, 3)).astype(np.float32)
    result = mgc.rashomon_disagreement_queue(
        grad_f32,
        archive_sha256=archive_sha256,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(result, mgc.RashomonDisagreementQueue)
    assert len(result.entries_tuple) == 15
