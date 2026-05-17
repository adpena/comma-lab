# SPDX-License-Identifier: MIT
"""Tests for master gradient consumer 6 (Rashomon disagreement queue).

Lane: lane_master_gradient_consumer_6_rashomon_disagreement_queue_20260517

Covers >=15 tests across:
  - Contract validation (dataclass fields + constructor invariants + shape guards)
  - Determinism (same seed = same K=8 sample IDs)
  - Disagreement signal correctness (synthetic input where K=8 members agree
    => low std; where they disagree => high std)
  - Sidecar JSON emit + compliance tags
  - Wire-in: end-to-end RashomonEnsembleRanker.update_all_from_master_gradient
    => persisted per-byte disagreement sister JSONL

[verified-against: feedback_per_pair_master_gradient_consumer_integration_design_20260517.md]
[verified-against: CLAUDE.md Catalog #252 (RashomonEnsembleRanker continual update locked)]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac import master_gradient_consumers as mgc
from tac.autopilot_rudin_daubechies.rashomon_ensemble import (
    RashomonEnsembleRanker,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def per_pair_grad():
    """Synthetic per-pair gradient with both pair-variant and pair-invariant bytes.

    Shape: (N_bytes=40, N_pairs=60, 3=seg/pose/rate). Mix of:
      - Bytes [0:10]: pair-invariant (small variance across pairs)
      - Bytes [10:30]: pair-variant (high variance)
      - Bytes [30:40]: dead (near zero on all axes)
    """
    rng = np.random.default_rng(42)
    grad = np.zeros((40, 60, 3), dtype=np.float64)
    # Pair-invariant
    grad[0:10, :, :] = rng.normal(0.5, 0.05, size=(10, 60, 3))
    # Pair-variant
    grad[10:30, :, :] = rng.normal(0.0, 1.0, size=(20, 60, 3))
    # Dead
    grad[30:40, :, :] = rng.normal(0.0, 1e-12, size=(10, 60, 3))
    return grad


@pytest.fixture
def archive_sha256():
    return "c" * 64


@pytest.fixture
def axis_meta():
    return {
        "measurement_axis": "contest_cpu",
        "measurement_hardware": "linux_x86_64_cpu",
    }


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Redirect CONSUMER_OUTPUT_ROOT to tmp_path so sidecars do not pollute repo state."""
    monkeypatch.setattr(
        mgc, "CONSUMER_OUTPUT_ROOT", tmp_path / "master_gradient_consumers"
    )
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────── #
# Contract validation (5 tests)                                                 #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_dataclass_fields_pinned():
    """RashomonDisagreementQueue / RashomonDisagreementEntry expose canonical fields."""
    queue_fields = set(mgc.RashomonDisagreementQueue.__dataclass_fields__.keys())
    expected_queue = {
        "entries_tuple",
        "k_members",
        "pair_subsample_size",
        "pair_subsample_fraction",
        "n_bytes",
        "n_pairs",
        "archive_sha256",
        "top_k_disagreement_indices",
        "aggregate_disagreement_score",
        "random_seed",
        "measurement_axis",
        "measurement_hardware",
    }
    assert queue_fields == expected_queue, (
        f"queue field drift: missing={expected_queue - queue_fields} "
        f"extra={queue_fields - expected_queue}"
    )

    entry_fields = set(mgc.RashomonDisagreementEntry.__dataclass_fields__.keys())
    expected_entry = {
        "byte_index",
        "mean_aggregate_gradient_magnitude",
        "std_across_k_members",
        "k_members_count",
        "axis",
    }
    assert entry_fields == expected_entry


def test_rashomon_contract_rejects_2d_input(archive_sha256, axis_meta):
    """Contract validation: per_pair_gradient with wrong shape (no pairs axis) rejected."""
    bad = np.zeros((10, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
        mgc.rashomon_disagreement_queue(
            bad,
            archive_sha256=archive_sha256,
            write_sidecar=False,
            **axis_meta,
        )


def test_rashomon_contract_rejects_wrong_axis_dim(archive_sha256, axis_meta):
    """Contract validation: last dim must be 3 (seg/pose/rate)."""
    bad = np.zeros((10, 20, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
        mgc.rashomon_disagreement_queue(
            bad,
            archive_sha256=archive_sha256,
            write_sidecar=False,
            **axis_meta,
        )


def test_rashomon_rejects_k_members_below_two(per_pair_grad, archive_sha256, axis_meta):
    """K=1 yields zero disagreement by construction; gate refuses."""
    with pytest.raises(ValueError, match=r"k_members must be >= 2"):
        mgc.rashomon_disagreement_queue(
            per_pair_grad,
            archive_sha256=archive_sha256,
            k_members=1,
            write_sidecar=False,
            **axis_meta,
        )


def test_rashomon_rejects_bad_subsample_fraction(per_pair_grad, archive_sha256, axis_meta):
    """Subsample fraction must be in (0.0, 1.0]."""
    for bad_frac in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="pair_subsample_fraction"):
            mgc.rashomon_disagreement_queue(
                per_pair_grad,
                archive_sha256=archive_sha256,
                pair_subsample_fraction=bad_frac,
                write_sidecar=False,
                **axis_meta,
            )


# ──────────────────────────────────────────────────────────────────────────── #
# Happy path + return contract (3 tests)                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_happy_path_returns_typed_queue(per_pair_grad, archive_sha256, axis_meta):
    """Canonical happy path returns RashomonDisagreementQueue with all fields populated."""
    queue = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        k_members=8,
        pair_subsample_fraction=0.8,
        random_seed=42,
        top_k=10,
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(queue, mgc.RashomonDisagreementQueue)
    assert queue.k_members == 8
    assert queue.pair_subsample_size == 48  # floor(60 * 0.8)
    assert queue.pair_subsample_fraction == 0.8
    assert queue.n_bytes == 40
    assert queue.n_pairs == 60
    assert queue.archive_sha256 == archive_sha256
    assert queue.random_seed == 42
    assert queue.measurement_axis == "contest_cpu"
    assert queue.measurement_hardware == "linux_x86_64_cpu"
    assert len(queue.entries_tuple) == 40
    assert len(queue.top_k_disagreement_indices) == 10


def test_rashomon_entries_preserve_byte_order(per_pair_grad, archive_sha256, axis_meta):
    """entries_tuple[i].byte_index == i for all bytes (canonical ordering)."""
    queue = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=False,
        **axis_meta,
    )
    for i, entry in enumerate(queue.entries_tuple):
        assert entry.byte_index == i
        assert entry.k_members_count == queue.k_members
        assert entry.axis == "contest_cpu"
        assert entry.std_across_k_members >= 0.0


def test_rashomon_top_k_descending_by_stddev(per_pair_grad, archive_sha256, axis_meta):
    """top_k_disagreement_indices are sorted by descending stddev."""
    queue = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        top_k=20,
        write_sidecar=False,
        **axis_meta,
    )
    top_stds = [queue.entries_tuple[i].std_across_k_members for i in queue.top_k_disagreement_indices]
    # Non-strict descending (ties allowed)
    for prev, curr in zip(top_stds, top_stds[1:]):
        assert prev >= curr, f"top-K ordering violation: {prev} < {curr}"


# ──────────────────────────────────────────────────────────────────────────── #
# Determinism (2 tests)                                                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_determinism_same_seed_same_output(per_pair_grad, archive_sha256, axis_meta):
    """Same seed => byte-identical queue (entries, top_k, aggregate score)."""
    queue1 = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=42,
        write_sidecar=False,
        **axis_meta,
    )
    queue2 = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=42,
        write_sidecar=False,
        **axis_meta,
    )
    assert queue1.top_k_disagreement_indices == queue2.top_k_disagreement_indices
    assert queue1.aggregate_disagreement_score == queue2.aggregate_disagreement_score
    for e1, e2 in zip(queue1.entries_tuple, queue2.entries_tuple):
        assert e1.mean_aggregate_gradient_magnitude == e2.mean_aggregate_gradient_magnitude
        assert e1.std_across_k_members == e2.std_across_k_members


def test_rashomon_determinism_different_seed_different_output(
    per_pair_grad, archive_sha256, axis_meta
):
    """Different seeds => different per-byte stddev values (different subsamples)."""
    queue1 = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=42,
        write_sidecar=False,
        **axis_meta,
    )
    queue2 = mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=99,
        write_sidecar=False,
        **axis_meta,
    )
    # At least one per-byte stddev should differ across seeds
    deltas = [
        abs(e1.std_across_k_members - e2.std_across_k_members)
        for e1, e2 in zip(queue1.entries_tuple, queue2.entries_tuple)
    ]
    assert max(deltas) > 1e-9, (
        "Different seeds produced identical stddev for every byte; "
        "subsampling is broken"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Disagreement signal correctness (3 tests)                                     #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_disagreement_low_for_pair_invariant_bytes(archive_sha256, axis_meta):
    """Synthetic input where ALL bytes are pair-invariant => K=8 members agree => low std."""
    rng = np.random.default_rng(0)
    # All bytes: identical mean across pairs, tiny noise
    grad = np.full((30, 80, 3), 0.5, dtype=np.float64)
    grad += rng.normal(0.0, 1e-6, size=grad.shape)
    queue = mgc.rashomon_disagreement_queue(
        grad,
        archive_sha256=archive_sha256,
        k_members=8,
        write_sidecar=False,
        **axis_meta,
    )
    # The aggregate disagreement score should be << 1.0 (CV of pair-invariant
    # signal). The exact threshold is loose because K=8 finite bootstrap has
    # some sampling variance.
    assert queue.aggregate_disagreement_score < 1e-3, (
        f"pair-invariant gradient produced aggregate CV "
        f"{queue.aggregate_disagreement_score:.6e}; expected < 1e-3"
    )


def test_rashomon_disagreement_high_for_pair_variant_bytes(archive_sha256, axis_meta):
    """Pair-variant bytes produce HIGHER per-byte stddev than pair-invariant bytes."""
    rng = np.random.default_rng(7)
    grad = np.zeros((40, 60, 3), dtype=np.float64)
    # Bytes [0:20]: pair-invariant (small variance)
    grad[0:20, :, :] = rng.normal(0.5, 0.001, size=(20, 60, 3))
    # Bytes [20:40]: pair-variant (high variance)
    grad[20:40, :, :] = rng.normal(0.0, 2.0, size=(20, 60, 3))
    queue = mgc.rashomon_disagreement_queue(
        grad,
        archive_sha256=archive_sha256,
        k_members=8,
        write_sidecar=False,
        **axis_meta,
    )
    invariant_stds = [
        queue.entries_tuple[i].std_across_k_members for i in range(0, 20)
    ]
    variant_stds = [
        queue.entries_tuple[i].std_across_k_members for i in range(20, 40)
    ]
    avg_invariant_std = sum(invariant_stds) / len(invariant_stds)
    avg_variant_std = sum(variant_stds) / len(variant_stds)
    assert avg_variant_std > 10.0 * avg_invariant_std, (
        f"pair-variant bytes mean std={avg_variant_std:.6f} not significantly "
        f"larger than pair-invariant mean std={avg_invariant_std:.6f}"
    )


def test_rashomon_disagreement_subsample_full_yields_zero_stddev(
    archive_sha256, axis_meta
):
    """Edge: subsample fraction 1.0 + permutation = identical aggregates across K members.

    With pair_subsample_fraction=1.0 each member sees ALL pairs (just in a
    different ORDER). The mean over all pairs is order-invariant, so per-byte
    stddev across K members MUST be exactly zero.
    """
    rng = np.random.default_rng(13)
    grad = rng.standard_normal((30, 40, 3)).astype(np.float64)
    queue = mgc.rashomon_disagreement_queue(
        grad,
        archive_sha256=archive_sha256,
        k_members=8,
        pair_subsample_fraction=1.0,
        write_sidecar=False,
        **axis_meta,
    )
    assert queue.pair_subsample_size == 40
    # All per-byte stddev must be ~0 (floating-point eps allowed)
    max_std = max(e.std_across_k_members for e in queue.entries_tuple)
    assert max_std < 1e-10, (
        f"full-permutation subsample produced nonzero stddev {max_std}; "
        "order-invariance is broken"
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Sidecar emit + compliance tag preservation (3 tests)                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_rashomon_sidecar_emit_creates_canonical_json(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar JSON lands under canonical path with canonical schema marker."""
    mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=True,
        **axis_meta,
    )
    # Find emitted sidecar
    matches = list((tmp_root / "master_gradient_consumers").glob(
        "rashomon_disagreement_queue_*.json"
    ))
    assert len(matches) == 1, f"expected exactly one sidecar; found {matches}"
    payload = json.loads(matches[0].read_text())
    assert payload["schema"] == "master_gradient_consumer_rashomon_disagreement_queue_v1"
    assert payload["consumer_id"] == "rashomon_disagreement_queue"
    assert payload["archive_sha256"] == archive_sha256
    assert payload["k_members"] == 8
    assert "top_k_disagreement_indices" in payload
    assert "aggregate_disagreement_score" in payload


def test_rashomon_sidecar_carries_compliance_tags(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar JSON carries score_claim=false + evidence_grade per Apples-to-apples."""
    mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=True,
        **axis_meta,
    )
    matches = list((tmp_root / "master_gradient_consumers").glob(
        "rashomon_disagreement_queue_*.json"
    ))
    payload = json.loads(matches[0].read_text())
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "[diagnostic" in payload["evidence_grade"]


def test_rashomon_sidecar_preserves_axis_and_hardware(
    per_pair_grad, archive_sha256, axis_meta, tmp_root
):
    """Sidecar JSON preserves measurement_axis + measurement_hardware as-is."""
    mgc.rashomon_disagreement_queue(
        per_pair_grad,
        archive_sha256=archive_sha256,
        write_sidecar=True,
        **axis_meta,
    )
    matches = list((tmp_root / "master_gradient_consumers").glob(
        "rashomon_disagreement_queue_*.json"
    ))
    payload = json.loads(matches[0].read_text())
    assert payload["measurement_axis"] == "contest_cpu"
    assert payload["measurement_hardware"] == "linux_x86_64_cpu"


# ──────────────────────────────────────────────────────────────────────────── #
# Wire-in: RashomonEnsembleRanker.update_all_from_master_gradient (4 tests)     #
# ──────────────────────────────────────────────────────────────────────────── #


def test_wire_in_method_returns_disagreement_queue(per_pair_grad, archive_sha256, tmp_path):
    """update_all_from_master_gradient returns the canonical disagreement queue."""
    store = tmp_path / "rashomon_anchors.jsonl"
    ranker = RashomonEnsembleRanker(ensemble_size=8, store_path=store)
    queue = ranker.update_all_from_master_gradient(
        per_pair_grad,
        archive_sha256=archive_sha256,
        measurement_axis="contest_cpu",
        measurement_hardware="linux_x86_64_cpu",
    )
    assert isinstance(queue, mgc.RashomonDisagreementQueue)
    assert queue.k_members == 8


def test_wire_in_persists_to_sister_jsonl(per_pair_grad, archive_sha256, tmp_path):
    """update_all_from_master_gradient persists a row to <store>.per_byte_disagreement.jsonl."""
    store = tmp_path / "rashomon_anchors.jsonl"
    sister = store.with_name(store.name + ".per_byte_disagreement.jsonl")
    ranker = RashomonEnsembleRanker(ensemble_size=8, store_path=store)
    assert not sister.exists()
    queue = ranker.update_all_from_master_gradient(
        per_pair_grad,
        archive_sha256=archive_sha256,
        measurement_axis="contest_cpu",
        measurement_hardware="linux_x86_64_cpu",
        random_seed=11,
    )
    assert sister.exists()
    raw_rows = [
        json.loads(line) for line in sister.read_text().splitlines() if line.strip()
    ]
    assert len(raw_rows) == 1
    row = raw_rows[0]
    assert row["schema"] == "rashomon_per_pair_disagreement_v1"
    assert row["archive_sha256"] == archive_sha256
    assert row["measurement_axis"] == "contest_cpu"
    assert row["measurement_hardware"] == "linux_x86_64_cpu"
    assert row["k_members"] == 8
    assert row["random_seed"] == 11
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["aggregate_disagreement_score"] == queue.aggregate_disagreement_score


def test_wire_in_refuses_non_persisted_ensemble(per_pair_grad, archive_sha256):
    """Catalog #252: non-persisted ensemble forbidden for master-gradient update."""
    ranker = RashomonEnsembleRanker(ensemble_size=8, store_path=None)
    with pytest.raises(RuntimeError, match=r"Catalog #252"):
        ranker.update_all_from_master_gradient(
            per_pair_grad,
            archive_sha256=archive_sha256,
        )


def test_wire_in_appends_across_invocations(per_pair_grad, archive_sha256, tmp_path):
    """Two calls => two rows in the sister JSONL (APPEND-ONLY per Catalog #110/#113)."""
    store = tmp_path / "rashomon_anchors.jsonl"
    sister = store.with_name(store.name + ".per_byte_disagreement.jsonl")
    ranker = RashomonEnsembleRanker(ensemble_size=8, store_path=store)
    ranker.update_all_from_master_gradient(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=1,
    )
    ranker.update_all_from_master_gradient(
        per_pair_grad,
        archive_sha256=archive_sha256,
        random_seed=2,
    )
    raw_rows = [
        json.loads(line) for line in sister.read_text().splitlines() if line.strip()
    ]
    assert len(raw_rows) == 2
    assert raw_rows[0]["random_seed"] == 1
    assert raw_rows[1]["random_seed"] == 2


# ──────────────────────────────────────────────────────────────────────────── #
# __all__ export validation (1 test)                                            #
# ──────────────────────────────────────────────────────────────────────────── #


def test_consumer_6_symbols_exported_via_all():
    """Consumer 6 symbols are exported via tac.master_gradient_consumers.__all__."""
    expected = {
        "RashomonDisagreementEntry",
        "RashomonDisagreementQueue",
        "rashomon_disagreement_queue",
    }
    actual = set(mgc.__all__)
    assert expected <= actual, f"missing exports: {expected - actual}"
