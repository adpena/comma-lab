# SPDX-License-Identifier: MIT
"""Tests for the Q1 deliverability_proof_builder canonical helper.

Covers:
  - DeliverabilityTier enum membership + canonical string values
  - DeliverabilityProof frozen-dataclass construction + __post_init__ invariants
  - build_deliverability_proof_from_wyner_ziv_classification on synthetic
    WynerZivSideInfoClassification (3 candidate bytes)
  - Tier 1 detector flags torch.zeros patterns + math constants
  - Tier 2 budget enforcement (cumulative <= 5120 bytes)
  - Tier 3 waiver-required flag fires when cumulative exceeds Tier-2 budget
  - Tier 4 detector flags pose-axis-dominant / seg-axis-dominant bytes
  - verify_deliverability_proof_contest_compliance returns
    (False, ["strict-scorer-rule violated", ...]) when Tier 4 > 0
  - verify_deliverability_proof_contest_compliance returns (True, []) when
    only Tier 1 + Tier 2 present
  - HNeRV parity L4 violation flagged for Tier 3 > 0 without operator approval
  - Catalog #213 Comma2k19LocalCache invocation cited for Comma2k19-derived
    Tier 2
  - Score-savings estimate matches canonical formula 25 * N / 37_545_489
  - Sidecar persistence to
    .omx/state/wyner_ziv_deliverability/proof_<sha[:12]>_<utc>.json
  - Fcntl-locked write per Catalog #131 (4-proc spawn-pool stress test)
  - load_deliverability_proof_for_archive returns most-recent proof
  - load_deliverability_proof_for_archive returns None when no proof exists
  - Schema version pinned at "deliverability_proof_v1"
  - Integration: build from REAL WZ result on the 8-pair fp64 anchor
    (skipped when fixture absent — does not block CI)
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from tac.master_gradient_consumers import WynerZivSideInfoClassification
from tac.wyner_ziv_deliverability import (
    DELIVERABILITY_PROOF_SCHEMA_VERSION,
    DeliverabilityProof,
    DeliverabilityTier,
    build_deliverability_proof_from_wyner_ziv_classification,
    load_deliverability_proof_for_archive,
    verify_deliverability_proof_contest_compliance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SHA_FIXTURE = "a" * 64
_SHA_FIXTURE_B = "b" * 64
_SHA_FIXTURE_C = "c" * 64


def _make_wz_classification(
    *, candidates: tuple[int, ...] = (), archive_sha256: str = _SHA_FIXTURE
) -> WynerZivSideInfoClassification:
    """Build a synthetic WynerZivSideInfoClassification for tests."""
    return WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=tuple(int(i) for i in candidates),
        pair_specific_byte_indices=(),
        mixed_byte_indices=(),
        aggregate_byte_pair_correlation_mean=0.5,
        n_bytes=max(1, max(candidates, default=0) + 1),
        n_pairs=8,
        correlation_threshold_high=0.8,
        correlation_threshold_low=0.3,
        archive_sha256=archive_sha256,
        estimated_wyner_ziv_gain_bytes=len(candidates),
    )


# ---------------------------------------------------------------------------
# Test 1: DeliverabilityTier enum membership + canonical string values
# ---------------------------------------------------------------------------


def test_t1_deliverability_tier_enum_canonical_strings() -> None:
    assert DeliverabilityTier.TIER_1_ZERO_COST.value == "tier_1_zero_cost"
    assert DeliverabilityTier.TIER_2_CONSTANTS.value == "tier_2_constants"
    assert DeliverabilityTier.TIER_3_WAIVER_REQUIRED.value == "tier_3_waiver_required"
    assert DeliverabilityTier.TIER_4_FORBIDDEN.value == "tier_4_forbidden"
    # Exhaustive: 4 members exactly
    assert len(list(DeliverabilityTier)) == 4


# ---------------------------------------------------------------------------
# Test 2: DeliverabilityProof __post_init__ invariants
# ---------------------------------------------------------------------------


def test_t2_proof_post_init_rejects_invalid_sha() -> None:
    with pytest.raises(ValueError, match="64-char hex"):
        DeliverabilityProof(
            archive_sha256="not_hex",
            candidate_shared_prior_byte_count=0,
            tier_1_byte_count=0,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
        )


def test_t3_proof_post_init_per_tier_sum_must_match() -> None:
    with pytest.raises(ValueError, match="per-tier byte counts must sum"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=10,
            tier_1_byte_count=2,
            tier_2_byte_count=3,
            tier_3_byte_count=0,
            tier_4_byte_count=0,  # sum=5, not 10
        )


def test_t4_proof_post_init_indices_must_be_sorted_ascending() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=3,
            tier_1_byte_count=3,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            tier_1_byte_indices=(5, 2, 9),  # unsorted
        )


def test_t5_proof_post_init_rejects_illegal_codec() -> None:
    with pytest.raises(ValueError, match="compression_codec must be in"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=0,
            tier_1_byte_count=0,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            compression_codec="snappy",
        )


def test_t6_proof_post_init_rejects_promotion_without_compliant_verdict() -> None:
    with pytest.raises(ValueError, match="promotion_eligible=True requires"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=0,
            tier_1_byte_count=0,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            contest_compliance_verdict="pending",
            promotion_eligible=True,
        )


# ---------------------------------------------------------------------------
# Test 7: build on synthetic 3-byte WZ result
# ---------------------------------------------------------------------------


def test_t7_build_on_synthetic_three_byte_wz_result(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0, 1, 2))
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        archive_bytes=b"\x00\x00\x00",  # all-zero -> Tier 1
        persist=True,
        proofs_dir=tmp_path,
    )
    assert proof.candidate_shared_prior_byte_count == 3
    assert proof.tier_1_byte_count == 3
    assert proof.tier_2_byte_count == 0
    assert proof.tier_3_byte_count == 0
    assert proof.tier_4_byte_count == 0
    assert proof.contest_compliance_verdict == "compliant"
    # Sidecar persisted
    files = sorted(tmp_path.glob(f"proof_{_SHA_FIXTURE[:12]}_*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Test 8: Tier 1 detector flags torch.zeros patterns (all-zero contiguous)
# ---------------------------------------------------------------------------


def test_t8_tier_1_detector_flags_all_zero_bytes(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=tuple(range(10)))
    archive_bytes = b"\x00" * 10  # entirely zero
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        archive_bytes=archive_bytes,
        persist=False,
    )
    assert proof.tier_1_byte_count == 10
    assert proof.tier_1_byte_indices == tuple(range(10))


# ---------------------------------------------------------------------------
# Test 9: Tier 2 budget enforcement (cumulative <= 5120 bytes)
# ---------------------------------------------------------------------------


def test_t9_tier_2_budget_enforcement_via_prober(tmp_path: Path) -> None:
    # 6 candidate bytes; prober assigns 1000-byte compressed cost each
    # tier_2_budget=5120 -> 5 fit in tier 2 (5*1000=5000<=5120),
    # 6th lands in tier 3 (5000+1000=6000 <= 5120+204800).
    candidates = tuple(range(6))
    wz = _make_wz_classification(candidates=candidates)
    prober = {
        "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=0)",
        "per_byte_classification": {
            str(i): {
                "source_class": "comma2k19",
                "compressed_size_bytes": 1000,
                "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=0)",
            }
            for i in candidates
        },
    }
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        deliverability_prober_result=prober,
        tier_2_budget_bytes=5120,
        tier_3_budget_bytes=10000,
        persist=False,
    )
    assert proof.tier_2_byte_count == 5  # 5 * 1000 = 5000 <= 5120
    assert proof.tier_3_byte_count == 1  # 6th byte
    assert proof.tier_4_byte_count == 0


# ---------------------------------------------------------------------------
# Test 10: Tier 3 waiver flag + operator approval gating
# ---------------------------------------------------------------------------


def test_t10_tier_3_waiver_required_without_operator_approval(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0, 1))
    prober = {
        "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
        "per_byte_classification": {
            "0": {
                "source_class": "comma2k19",
                "compressed_size_bytes": 6000,  # exceeds tier_2_budget=5120
                "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
            },
            "1": {
                "source_class": "comma2k19",
                "compressed_size_bytes": 6000,
                "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
            },
        },
    }
    proof_unapproved = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        deliverability_prober_result=prober,
        operator_approved_tier_3=False,
        persist=False,
    )
    assert proof_unapproved.tier_3_byte_count == 2
    assert proof_unapproved.waiver_required_for_tier_3 is True
    assert proof_unapproved.operator_review_status_for_tier_3 == "pending"
    assert proof_unapproved.contest_compliance_verdict == "partial"

    proof_approved = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE_B,
        deliverability_prober_result=prober,
        operator_approved_tier_3=True,
        persist=False,
    )
    assert proof_approved.tier_3_byte_count == 2
    assert proof_approved.operator_review_status_for_tier_3 == "approved"
    assert proof_approved.contest_compliance_verdict == "compliant"
    # Deliverable savings INCLUDES tier 3 when approved
    assert proof_approved.deliverable_score_savings_estimate > 0.0


# ---------------------------------------------------------------------------
# Test 11: Tier 4 detector flags pose-axis-dominant bytes
# ---------------------------------------------------------------------------


def test_t11_tier_4_detector_flags_pose_seg_dominant_bytes(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0, 1, 2, 3))
    pose_dominant = (1, 3)  # bytes 1 and 3 are scorer-dependent
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        archive_bytes=b"\x00\x00\x00\x00",
        pose_seg_dominant_byte_indices=pose_dominant,
        persist=False,
    )
    assert proof.tier_4_byte_count == 2
    assert proof.tier_4_byte_indices == (1, 3)
    assert proof.contest_compliance_verdict == "non_compliant"


# ---------------------------------------------------------------------------
# Test 12: verify_deliverability_proof_contest_compliance flags Tier 4
# ---------------------------------------------------------------------------


def test_t12_verifier_flags_tier_4_strict_scorer_rule() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=3,
        tier_1_byte_count=1,
        tier_2_byte_count=0,
        tier_3_byte_count=0,
        tier_4_byte_count=2,
        tier_1_byte_indices=(0,),
        tier_4_byte_indices=(1, 2),
        canonical_helper_invocation="tac.wyner_ziv_deliverability (zero-cost only; no external helper required)",
        contest_compliance_verdict="non_compliant",
    )
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    assert ok is False
    assert any("strict-scorer-rule" in b for b in blockers)


def test_t13_verifier_clean_when_only_tier_1_2_present() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=2,
        tier_1_byte_count=1,
        tier_2_byte_count=1,
        tier_3_byte_count=0,
        tier_4_byte_count=0,
        tier_1_byte_indices=(0,),
        tier_2_byte_indices=(1,),
        canonical_helper_invocation="Comma2k19LocalCache.fetch_chunk(idx=0)",
        contest_compliance_verdict="compliant",
    )
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    assert ok is True
    assert blockers == []


# ---------------------------------------------------------------------------
# Test 14: HNeRV parity L4 violation flagged for Tier 3 > 0 unapproved
# ---------------------------------------------------------------------------


def test_t14_verifier_flags_hnerv_l4_tier_3_unapproved() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=1,
        tier_1_byte_count=0,
        tier_2_byte_count=0,
        tier_3_byte_count=1,
        tier_4_byte_count=0,
        tier_3_byte_indices=(0,),
        waiver_required_for_tier_3=True,
        operator_review_status_for_tier_3="pending",
        canonical_helper_invocation="DashcamDomainPrior",
        contest_compliance_verdict="partial",
    )
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    assert ok is False
    assert any("HNeRV parity L4" in b for b in blockers)


# ---------------------------------------------------------------------------
# Test 15: Catalog #213 Comma2k19LocalCache citation enforcement
# ---------------------------------------------------------------------------


def test_t15_verifier_catalog_213_rejects_raw_comma2k19_url() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=1,
        tier_1_byte_count=0,
        tier_2_byte_count=1,
        tier_3_byte_count=0,
        tier_4_byte_count=0,
        tier_2_byte_indices=(0,),
        canonical_helper_invocation="https://academictorrents.com/details/comma2k19_chunk_0",
        contest_compliance_verdict="compliant",
    )
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    assert ok is False
    assert any("Catalog #213" in b for b in blockers)


# ---------------------------------------------------------------------------
# Test 16: Score-savings estimate matches canonical formula
# ---------------------------------------------------------------------------


def test_t16_score_savings_matches_canonical_formula(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=tuple(range(100)))
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        archive_bytes=b"\x00" * 100,  # all-zero -> Tier 1
        persist=False,
    )
    expected = 25.0 * 100.0 / 37_545_489.0
    assert abs(proof.tier_1_score_savings_estimate - expected) < 1e-12


# ---------------------------------------------------------------------------
# Test 17: sidecar persistence + load_deliverability_proof_for_archive
# ---------------------------------------------------------------------------


def test_t17_persistence_and_load_roundtrip(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0, 1, 2))
    proof_written = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        archive_bytes=b"\x00\x00\x00",
        persist=True,
        proofs_dir=tmp_path,
    )
    loaded = load_deliverability_proof_for_archive(_SHA_FIXTURE, proofs_dir=tmp_path)
    assert loaded is not None
    assert loaded.archive_sha256 == _SHA_FIXTURE
    assert loaded.tier_1_byte_count == proof_written.tier_1_byte_count
    assert loaded.proof_sha256 == proof_written.proof_sha256
    assert loaded.contest_compliance_rationale == proof_written.contest_compliance_rationale
    assert (
        loaded.contest_compliance_citation_chain
        == proof_written.contest_compliance_citation_chain
    )


def test_t18_load_returns_none_when_no_proof_exists(tmp_path: Path) -> None:
    result = load_deliverability_proof_for_archive(_SHA_FIXTURE_C, proofs_dir=tmp_path)
    assert result is None


def test_t18b_load_refuses_legacy_proof_missing_compliance_authority(
    tmp_path: Path,
) -> None:
    legacy_payload = {
        "archive_sha256": _SHA_FIXTURE,
        "candidate_shared_prior_byte_count": 0,
        "tier_1_byte_count": 0,
        "tier_2_byte_count": 0,
        "tier_3_byte_count": 0,
        "tier_4_byte_count": 0,
        "deliverable_score_savings_estimate": 0.0,
        "canonical_helper_invocation": "tac.wyner_ziv_deliverability",
        "contest_compliance_verdict": "pending",
        # Intentionally omit contest_compliance_rationale and
        # contest_compliance_citation_chain. Loader authority must not
        # silently backfill these from dataclass defaults.
    }
    (tmp_path / f"proof_{_SHA_FIXTURE[:12]}_legacy.json").write_text(
        json.dumps(legacy_payload),
        encoding="utf-8",
    )

    loaded = load_deliverability_proof_for_archive(_SHA_FIXTURE, proofs_dir=tmp_path)
    assert loaded is None


# ---------------------------------------------------------------------------
# Test 19: schema version pinned
# ---------------------------------------------------------------------------


def test_t19_schema_version_pinned() -> None:
    assert DELIVERABILITY_PROOF_SCHEMA_VERSION == "deliverability_proof_v1"


# ---------------------------------------------------------------------------
# Test 20: fcntl-locked write 4-proc spawn-pool stress
# ---------------------------------------------------------------------------


def _worker_write_proof(args: tuple[str, str, int]) -> str:
    proofs_dir_str, sha, n_bytes = args
    proofs_dir = Path(proofs_dir_str)
    candidates = tuple(range(n_bytes))
    wz = WynerZivSideInfoClassification(
        candidate_shared_prior_byte_indices=candidates,
        pair_specific_byte_indices=(),
        mixed_byte_indices=(),
        aggregate_byte_pair_correlation_mean=0.5,
        n_bytes=max(1, n_bytes),
        n_pairs=8,
        correlation_threshold_high=0.8,
        correlation_threshold_low=0.3,
        archive_sha256=sha,
        estimated_wyner_ziv_gain_bytes=n_bytes,
    )
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=sha,
        archive_bytes=b"\x00" * n_bytes,
        persist=True,
        proofs_dir=proofs_dir,
    )
    return proof.proof_sha256


def test_t20_fcntl_locked_4_proc_spawn_pool_stress(tmp_path: Path) -> None:
    # 4 processes, each writing 5 distinct sha proofs concurrently. Sha
    # prefixes must differ in the first 12 chars so the per-file paths
    # ``proof_<sha[:12]>_<utc>.json`` do not collide on os.replace.
    args = []
    expected_files: set[str] = set()
    for proc_idx in range(4):
        for sub in range(5):
            # Use a prefix that fills the first 12 chars uniquely:
            # 12-char-unique = f"{proc_idx:06d}{sub:06d}"
            prefix12 = f"{proc_idx:06d}{sub:06d}"
            sha_seed = prefix12 + "0" * 52  # total 64 chars
            args.append((str(tmp_path), sha_seed, 1))
            expected_files.add(prefix12)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_worker_write_proof, args)
    assert len(results) == 20
    # Every distinct sha produced at least one proof file
    files = list(tmp_path.glob("proof_*.json"))
    assert len(files) >= 20
    found_prefixes = {f.name.split("_")[1] for f in files}
    assert expected_files.issubset(found_prefixes)


# ---------------------------------------------------------------------------
# Test 21: as_dict serializes tuples -> lists for JSON
# ---------------------------------------------------------------------------


def test_t21_as_dict_serializes_byte_indices_as_lists() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=2,
        tier_1_byte_count=2,
        tier_2_byte_count=0,
        tier_3_byte_count=0,
        tier_4_byte_count=0,
        tier_1_byte_indices=(0, 1),
        canonical_helper_invocation="tac.wyner_ziv_deliverability (zero-cost only; no external helper required)",
    )
    d = proof.as_dict()
    assert isinstance(d["tier_1_byte_indices"], list)
    # JSON-serializable
    json.dumps(d)


# ---------------------------------------------------------------------------
# Test 22: Comma2k19 prober output cited via canonical helper string
# ---------------------------------------------------------------------------


def test_t22_comma2k19_prober_helper_propagated_to_proof(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0,))
    prober = {
        "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
        "per_byte_classification": {
            "0": {
                "source_class": "comma2k19",
                "compressed_size_bytes": 100,
                "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
            }
        },
    }
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        deliverability_prober_result=prober,
        persist=False,
    )
    assert "Comma2k19LocalCache.fetch_chunk" in proof.canonical_helper_invocation
    ok, blockers = verify_deliverability_proof_contest_compliance(proof)
    assert ok is True
    assert blockers == []


# ---------------------------------------------------------------------------
# Test 22b: canonical procedural-generation compliance rationale
# ---------------------------------------------------------------------------


def test_t22b_compliant_tier_2_proof_carries_loophole_boundary_rationale(
    tmp_path: Path,
) -> None:
    wz = _make_wz_classification(candidates=(0,))
    prober = {
        "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
        "per_byte_classification": {
            "0": {
                "source_class": "comma2k19",
                "compressed_size_bytes": 100,
                "canonical_helper_invocation": "Comma2k19LocalCache.fetch_chunk(idx=42)",
            }
        },
    }
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        deliverability_prober_result=prober,
        persist=False,
    )
    rationale = proof.contest_compliance_rationale
    for anchor in (
        "PR #68 loophole_v2",
        "upstream/evaluate.py:63",
        "Catalog #213",
        "Comma2k19LocalCache",
        "INSIDE archive.zip",
        "OUTSIDE archive.zip",
        "score_claim=False",
        "promotion_eligible=False",
    ):
        assert anchor in rationale


def test_t22c_proof_rejects_blank_rationale() -> None:
    with pytest.raises(ValueError, match="contest_compliance_rationale"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=1,
            tier_1_byte_count=0,
            tier_2_byte_count=1,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            tier_2_byte_indices=(0,),
            canonical_helper_invocation="Comma2k19LocalCache.fetch_chunk(idx=0)",
            contest_compliance_verdict="compliant",
            contest_compliance_rationale="",
        )


def test_t22d_proof_requires_compliance_citation_chain_route() -> None:
    with pytest.raises(ValueError, match="contest_compliance_citation_chain"):
        DeliverabilityProof(
            archive_sha256=_SHA_FIXTURE,
            candidate_shared_prior_byte_count=0,
            tier_1_byte_count=0,
            tier_2_byte_count=0,
            tier_3_byte_count=0,
            tier_4_byte_count=0,
            contest_compliance_citation_chain=("PR #68 loophole_v2",),
        )


def test_t22e_as_dict_serializes_compliance_citation_chain() -> None:
    proof = DeliverabilityProof(
        archive_sha256=_SHA_FIXTURE,
        candidate_shared_prior_byte_count=0,
        tier_1_byte_count=0,
        tier_2_byte_count=0,
        tier_3_byte_count=0,
        tier_4_byte_count=0,
    )
    payload = proof.as_dict()
    assert isinstance(payload["contest_compliance_citation_chain"], list)
    assert "archive.zip seed inclusion" in payload["contest_compliance_citation_chain"]


# ---------------------------------------------------------------------------
# Test 23: integration against REAL 8-pair fp64 anchor (skipped if absent)
# ---------------------------------------------------------------------------


def test_t23_integration_against_real_8pair_fp64_anchor(tmp_path: Path) -> None:
    anchor_path = Path(__file__).resolve().parents[3] / ".omx" / "tmp" / "master_gradient_per_pair_8pair_fp64_validate.npy"
    if not anchor_path.exists():
        pytest.skip(f"Real 8-pair fp64 anchor not present at {anchor_path}")
    import numpy as np

    from tac.master_gradient_consumers import wyner_ziv_side_info_covariance

    per_pair = np.load(str(anchor_path))
    if per_pair.ndim != 3 or per_pair.shape[-1] != 3:
        pytest.skip(
            f"Anchor shape {per_pair.shape} not (N_bytes, N_pairs, 3); skipping"
        )
    archive_sha = hashlib.sha256(per_pair.tobytes()).hexdigest()
    wz = wyner_ziv_side_info_covariance(
        per_pair,
        archive_sha256=archive_sha,
        measurement_axis="cpu",
        measurement_hardware="macos_arm64",
        write_sidecar=False,
    )
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=archive_sha,
        persist=False,
    )
    assert proof.candidate_shared_prior_byte_count == len(
        wz.candidate_shared_prior_byte_indices
    )
    # Proof is at minimum well-formed; verdict may be any of the legal values
    assert proof.contest_compliance_verdict in {
        "pending",
        "compliant",
        "partial",
        "non_compliant",
    }


# ---------------------------------------------------------------------------
# Test 24: contest_compliance_verdict "unknown" helper => non_compliant
# ---------------------------------------------------------------------------


def test_t24_unknown_helper_marks_non_compliant(tmp_path: Path) -> None:
    wz = _make_wz_classification(candidates=(0,))
    # No prober, no archive_bytes -> Tier 2 by exhaustion, helper="unknown"
    proof = build_deliverability_proof_from_wyner_ziv_classification(
        wyner_ziv_result=wz,
        archive_sha256=_SHA_FIXTURE,
        persist=False,
    )
    assert proof.canonical_helper_invocation == "unknown"
    assert proof.contest_compliance_verdict == "non_compliant"


# ---------------------------------------------------------------------------
# Test 25: Catalog #319 preflight gate exists in preflight.py
# ---------------------------------------------------------------------------


def test_t25_catalog_319_gate_exists_in_preflight() -> None:
    from tac.preflight import check_substrate_wyner_ziv_reweight_has_deliverability_proof

    violations = check_substrate_wyner_ziv_reweight_has_deliverability_proof()
    assert isinstance(violations, list)
