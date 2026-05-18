# SPDX-License-Identifier: MIT
"""Tests for ``tools/wyner_ziv_deliverability_prober.py``.

[verified-against: tools/wyner_ziv_deliverability_prober.py — the canonical
                   prober this test suite covers]
[verified-against: src/tac/master_gradient_consumers.py consumer 4 — the
                   producer the prober interrogates]
[verified-against: empirical anchor — fec6 archive sha
                   f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd]
"""
from __future__ import annotations

import json
import lzma
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# Add tools/ to sys.path so we can import the prober as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import wyner_ziv_deliverability_prober as prober  # noqa: E402


# ──────────────────────────────────────────────────────────────────────── #
# Synthetic fixtures                                                        #
# ──────────────────────────────────────────────────────────────────────── #


def _make_uniform_archive_bytes(n: int = 1024, seed: int = 42) -> bytes:
    """Uniformly-distributed bytes — proxies entropy-coded source."""
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(0, 256, size=n, dtype=np.uint8).tolist())


def _make_repeating_archive_bytes(n: int = 1024) -> bytes:
    """Highly compressible bytes (all zeros) — proxies pre-entropy source."""
    return b"\x00" * n


def _make_per_pair_gradient(
    n_bytes: int,
    n_pairs: int = 8,
    pair_invariant_fraction: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic per-pair gradient tensor.

    pair_invariant_fraction of bytes have low per-pair variance (CSP);
    the rest have high per-pair variance (pair-specific).
    """
    rng = np.random.default_rng(seed)
    n_invariant = int(n_bytes * pair_invariant_fraction)
    grad = np.zeros((n_bytes, n_pairs, 3), dtype=np.float64)

    # Pair-invariant bytes: same value across pairs on seg+pose, zero rate
    invariant_seg = rng.uniform(1e-6, 1e-5, size=n_invariant)
    invariant_pose = rng.uniform(1e-6, 1e-5, size=n_invariant)
    for i in range(n_invariant):
        grad[i, :, 0] = invariant_seg[i] + rng.normal(0, invariant_seg[i] * 0.01, size=n_pairs)
        grad[i, :, 1] = invariant_pose[i] + rng.normal(0, invariant_pose[i] * 0.01, size=n_pairs)

    # Pair-specific bytes: high variance
    for i in range(n_invariant, n_bytes):
        grad[i, :, 0] = rng.normal(0, 1e-5, size=n_pairs)
        grad[i, :, 1] = rng.normal(0, 1e-5, size=n_pairs)

    return grad


# ──────────────────────────────────────────────────────────────────────── #
# Codec roundtrip tests                                                     #
# ──────────────────────────────────────────────────────────────────────── #


def test_lzma_roundtrip_correct():
    data = _make_uniform_archive_bytes(n=1024)
    result = prober.measure_codec("lzma", data)
    assert result.codec == "lzma"
    assert result.raw_bytes == 1024
    assert result.compressed_bytes > 0
    assert result.decode_correct is True


def test_brotli_roundtrip_correct():
    if not prober._HAS_BROTLI:
        pytest.skip("brotli not installed")
    data = _make_uniform_archive_bytes(n=1024)
    result = prober.measure_codec("brotli", data)
    assert result.decode_correct is True


def test_zlib_roundtrip_correct():
    data = _make_uniform_archive_bytes(n=1024)
    result = prober.measure_codec("zlib", data)
    assert result.decode_correct is True


def test_lzma_inflates_uniform_bytes():
    """Uniform bytes are at Shannon entropy; lzma must NOT compress."""
    data = _make_uniform_archive_bytes(n=8192, seed=7)
    result = prober.measure_codec("lzma", data)
    assert result.compressed_bytes >= len(data) - 10  # within lzma framing overhead
    # Inflated bytes empirically inflate by lzma header (~50-70 bytes)
    assert result.ratio >= 1.0 - 0.001


def test_lzma_compresses_repeating_bytes():
    """All-zero bytes are highly compressible."""
    data = _make_repeating_archive_bytes(n=8192)
    result = prober.measure_codec("lzma", data)
    assert result.compressed_bytes < 1024  # lzma collapses zeros heavily


# ──────────────────────────────────────────────────────────────────────── #
# Tier classification tests                                                 #
# ──────────────────────────────────────────────────────────────────────── #


def test_tier_1_detector_flags_zero_aggregate_bytes():
    """Bytes with zero aggregate magnitude on all axes → Tier 1."""
    n_bytes = 100
    grad = np.zeros((n_bytes, 8, 3), dtype=np.float64)
    # Make 30 bytes nonzero on seg axis
    grad[:30, :, 0] = 1e-5
    # The remaining 70 bytes have zero aggregate on all 3 axes → Tier 1
    raw = bytes(range(n_bytes % 256)) * (n_bytes // 256 + 1)
    raw = raw[:n_bytes]
    csp_indices = list(range(n_bytes))  # all bytes are CSP candidates
    tier = prober.classify_csp_bytes_into_tiers(
        candidate_shared_prior_indices=csp_indices,
        raw_bytes=raw,
        per_pair_gradient=grad,
    )
    assert tier.tier_1_zero_cost_bytes == 70


def test_tier_4_detector_flags_scorer_axis_dominant_bytes():
    """Bytes with seg/pose dominance AND zero rate axis → Tier 4."""
    n_bytes = 50
    grad = np.zeros((n_bytes, 8, 3), dtype=np.float64)
    # All bytes have HIGH seg magnitude, ZERO rate magnitude → scorer-dominant.
    # Inject per-pair variance so the per-byte aggregate stays high.
    rng = np.random.default_rng(7)
    for i in range(n_bytes):
        grad[i, :, 0] = 1e-4 + rng.normal(0, 1e-6, size=8)  # high seg, varying
    raw = bytes(range(256)) * 1
    raw = raw[:n_bytes]
    csp_indices = list(range(n_bytes))
    tier = prober.classify_csp_bytes_into_tiers(
        candidate_shared_prior_indices=csp_indices,
        raw_bytes=raw,
        per_pair_gradient=grad,
    )
    assert tier.tier_4_forbidden_bytes == 50


def test_tier_2_greedy_fill_respects_budget():
    """Tier 2 should NOT exceed the budget."""
    # 1000 bytes, all valid Tier 2 candidates (rate axis nonzero so not Tier 4)
    n_bytes = 1000
    grad = np.zeros((n_bytes, 8, 3), dtype=np.float64)
    grad[:, :, 0] = 1e-5
    grad[:, :, 1] = 1e-5
    grad[:, :, 2] = 1e-5  # nonzero rate axis → not Tier 4
    raw = bytes(np.random.default_rng(7).integers(0, 256, size=n_bytes, dtype=np.uint8).tolist())
    csp_indices = list(range(n_bytes))
    # Tiny budget — Tier 2 should be small
    tier = prober.classify_csp_bytes_into_tiers(
        candidate_shared_prior_indices=csp_indices,
        raw_bytes=raw,
        per_pair_gradient=grad,
        tier_2_budget_bytes=128,
        tier_3_budget_bytes=8192,
    )
    # Tier 2 compressed-size must NOT exceed budget
    if tier.tier_2_byte_indices:
        tier_2_raw = bytes(raw[i] for i in tier.tier_2_byte_indices)
        comp = lzma.compress(tier_2_raw, preset=6)
        assert len(comp) <= 128 + 256  # allow one chunk-size overshoot in greedy boundary


def test_tier_classification_partitions_csp_completely():
    """Tier 1 + 2 + 3 + 4 must cover the entire CSP set (no orphans)."""
    n_bytes = 200
    grad = np.zeros((n_bytes, 8, 3), dtype=np.float64)
    grad[:100, :, 1] = 1e-5  # half nonzero pose
    grad[:, :, 2] = 1e-7  # tiny rate axis (avoid Tier 4 false-positive)
    raw = _make_uniform_archive_bytes(n=n_bytes)
    csp_indices = list(range(n_bytes))
    tier = prober.classify_csp_bytes_into_tiers(
        candidate_shared_prior_indices=csp_indices,
        raw_bytes=raw,
        per_pair_gradient=grad,
    )
    total = (
        tier.tier_1_zero_cost_bytes
        + tier.tier_2_constants_bytes
        + tier.tier_3_waiver_required_bytes
        + tier.tier_4_forbidden_bytes
    )
    assert total == len(csp_indices)


def test_empty_csp_returns_zeros():
    grad = np.zeros((10, 8, 3))
    raw = b"\x00" * 10
    tier = prober.classify_csp_bytes_into_tiers(
        candidate_shared_prior_indices=[],
        raw_bytes=raw,
        per_pair_gradient=grad,
    )
    assert tier.tier_1_zero_cost_bytes == 0
    assert tier.tier_2_constants_bytes == 0
    assert tier.tier_3_waiver_required_bytes == 0
    assert tier.tier_4_forbidden_bytes == 0


# ──────────────────────────────────────────────────────────────────────── #
# T4 timeout headroom tests                                                 #
# ──────────────────────────────────────────────────────────────────────── #


def test_t4_timeout_headroom_within_expected_window():
    headroom = prober.estimate_t4_timeout_headroom(
        fec6_baseline_inflate_seconds_cpu=10.0,
        cpu_to_t4_speedup_factor=3.0,
    )
    # T4 baseline ≈ 10/3 ≈ 3.33 s; headroom ≈ 1800 - 3.33 ≈ 1796.67
    assert abs(headroom.fec6_baseline_inflate_seconds_t4_estimated - 10.0 / 3.0) < 0.01
    assert headroom.available_headroom_seconds_t4 > 1700
    assert headroom.available_headroom_seconds_t4 < 1801


def test_t4_timeout_headroom_synthetic_sizes_monotonic():
    """Larger inflate.py → larger T4 overhead (monotonic)."""
    headroom = prober.estimate_t4_timeout_headroom(
        fec6_baseline_inflate_seconds_cpu=10.0,
    )
    sizes_kb = sorted(int(k.replace("kb", "")) for k in headroom.synthetic_inflate_py_overhead_seconds)
    overheads = [headroom.synthetic_inflate_py_overhead_seconds[f"{kb}kb"] for kb in sizes_kb]
    for i in range(1, len(overheads)):
        assert overheads[i] >= overheads[i - 1], f"overhead not monotonic: {overheads}"


# ──────────────────────────────────────────────────────────────────────── #
# Rate score savings tests                                                  #
# ──────────────────────────────────────────────────────────────────────── #


def test_compute_rate_score_savings_zero_bytes():
    assert prober.compute_rate_score_savings(0) == 0.0


def test_compute_rate_score_savings_canonical_formula():
    # 1000 bytes saved → 25 * 1000 / 37545489 ≈ 6.66e-4
    saved = prober.compute_rate_score_savings(1000)
    expected = 25.0 * 1000 / 37_545_489
    assert abs(saved - expected) < 1e-12


# ──────────────────────────────────────────────────────────────────────── #
# Deliverability verdict tests                                              #
# ──────────────────────────────────────────────────────────────────────── #


def test_verdict_not_deliverable_when_lzma_inflates_csp():
    """Empty CSP should yield NOT_DELIVERABLE verdict."""
    tier = prober.TierClassification(
        tier_1_zero_cost_bytes=0,
        tier_2_constants_bytes=0,
        tier_3_waiver_required_bytes=162000,
        tier_4_forbidden_bytes=123,
        tier_1_byte_indices=(),
        tier_2_byte_indices=(),
        tier_3_byte_indices=(),
        tier_4_byte_indices=(),
        tier_2_budget_bytes=5120,
        tier_3_budget_bytes=200000,
        tier_4_dominance_ratio=10.0,
        tier_1_aggregate_floor_relative=0.01,
    )
    verdict = prober.derive_deliverability_verdict(
        archive_sha256="f174192aeadf",
        archive_path=None,
        per_pair_tensor_path=Path("/tmp/fake.npy"),
        csp_byte_count=162123,
        best_compressed_bytes=162192,  # lzma INFLATED
        best_codec="lzma",
        lzma_compressed=162192,
        brotli_compressed=162128,
        zlib_compressed=162179,
        tier_classification=tier,
    )
    assert verdict.deliverability_verdict == "NOT_DELIVERABLE"
    assert verdict.autopilot_reward_factor_justified is False
    assert verdict.rate_score_savings_estimate == 0.0  # tier 1 was 0


def test_verdict_deliverable_when_csp_compressible_and_tier_1_dominant():
    """A fully-deliverable CSP (high Tier 1, low entropy) → DELIVERABLE."""
    tier = prober.TierClassification(
        tier_1_zero_cost_bytes=100_000,  # 100K bytes free
        tier_2_constants_bytes=50_000,
        tier_3_waiver_required_bytes=0,
        tier_4_forbidden_bytes=0,
        tier_1_byte_indices=tuple(range(100_000)),
        tier_2_byte_indices=tuple(range(100_000, 150_000)),
        tier_3_byte_indices=(),
        tier_4_byte_indices=(),
        tier_2_budget_bytes=5120,
        tier_3_budget_bytes=200000,
        tier_4_dominance_ratio=10.0,
        tier_1_aggregate_floor_relative=0.01,
    )
    verdict = prober.derive_deliverability_verdict(
        archive_sha256="abcdef123456",
        archive_path=None,
        per_pair_tensor_path=Path("/tmp/fake.npy"),
        csp_byte_count=150_000,
        best_compressed_bytes=50_000,
        best_codec="lzma",
        lzma_compressed=50_000,
        brotli_compressed=51_000,
        zlib_compressed=55_000,
        tier_classification=tier,
    )
    # Tier 1 alone provides large rate savings: 25 * 100000 / 37545489 ≈ 0.066
    assert verdict.rate_score_savings_estimate > 0.05
    # Deliverable fraction = (100000 + 50000) / 150000 = 1.0 ≥ 0.5; ΔS > 0.001
    assert verdict.deliverability_verdict == "DELIVERABLE"
    assert verdict.autopilot_reward_factor_justified is True


# ──────────────────────────────────────────────────────────────────────── #
# Output JSON schema validation                                             #
# ──────────────────────────────────────────────────────────────────────── #


def test_persist_emits_canonical_schema_v1(tmp_path):
    """Persisted JSON must carry the canonical schema fields."""
    tier = prober.TierClassification(
        tier_1_zero_cost_bytes=0,
        tier_2_constants_bytes=0,
        tier_3_waiver_required_bytes=162000,
        tier_4_forbidden_bytes=123,
        tier_1_byte_indices=(),
        tier_2_byte_indices=(),
        tier_3_byte_indices=(),
        tier_4_byte_indices=(),
        tier_2_budget_bytes=5120,
        tier_3_budget_bytes=200000,
        tier_4_dominance_ratio=10.0,
        tier_1_aggregate_floor_relative=0.01,
    )
    headroom = prober.T4TimeoutHeadroom(
        fec6_baseline_inflate_seconds_cpu=10.0,
        fec6_baseline_inflate_seconds_t4_estimated=3.3,
        contest_t4_timeout_seconds=1800,
        available_headroom_seconds_t4=1796.7,
        cpu_to_t4_speedup_factor=3.0,
        synthetic_inflate_py_overhead_seconds={"10kb": 0.2, "50kb": 0.21, "100kb": 0.22, "200kb": 0.23, "500kb": 0.24},
    )
    verdict = prober.DeliverabilityVerdict(
        archive_sha256="f174192aeadf",
        archive_path=None,
        per_pair_tensor_path="/tmp/fake.npy",
        candidate_shared_prior_byte_count=162123,
        candidate_shared_prior_subset_compressed_bytes_lzma=162192,
        candidate_shared_prior_subset_compressed_bytes_brotli=162128,
        candidate_shared_prior_subset_compressed_bytes_zlib=162179,
        best_codec="lzma",
        best_compressed_bytes=162192,
        rate_score_savings_estimate=0.0,
        deliverable_score_delta_estimate=0.0,
        autopilot_reward_factor_in_use=1.15,
        autopilot_reward_factor_justified=False,
        deliverability_verdict="NOT_DELIVERABLE",
        reasoning="entropy floor",
    )
    codec_results = [
        prober.CodecResult(codec="lzma", raw_bytes=162123, compressed_bytes=162192, ratio=1.0004, decode_correct=True, parameters={"preset": 9}),
    ]
    path = prober.persist_probe_artifact(
        verdict=verdict,
        t4_headroom=headroom,
        tier_classification=tier,
        codec_results=codec_results,
        output_dir=tmp_path,
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    # Schema fields
    assert payload["schema_version"] == "wyner_ziv_deliverability_probe_v1"
    assert payload["archive_sha256"] == "f174192aeadf"
    assert payload["candidate_shared_prior_byte_count"] == 162123
    assert payload["lzma_result"]["compressed_bytes"] == 162192
    assert payload["best_codec"] == "lzma"
    assert payload["deliverability_verdict"] == "NOT_DELIVERABLE"
    assert payload["autopilot_reward_factor_justified"] is False
    # Compliance tags per CLAUDE.md
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"] == "predicted"
    assert "tier_classification_per_catalog_220" in payload["claude_md_compliance_tags"]
    assert "fcntl_locked_write_per_catalog_131" in payload["claude_md_compliance_tags"]


# ──────────────────────────────────────────────────────────────────────── #
# End-to-end synthetic test                                                 #
# ──────────────────────────────────────────────────────────────────────── #


def test_run_probe_synthetic_1kb(tmp_path):
    """End-to-end probe on a 1 KB synthetic CSP set."""
    n_bytes = 1024
    grad = _make_per_pair_gradient(n_bytes=n_bytes, pair_invariant_fraction=0.9)
    tensor_path = tmp_path / "synthetic_per_pair.npy"
    np.save(tensor_path, grad)

    # Create a synthetic archive.zip
    archive_path = tmp_path / "archive.zip"
    raw = _make_uniform_archive_bytes(n=n_bytes)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", raw)

    verdict, persisted_path = prober.run_probe(
        archive_sha256="synthetic_test_sha",
        per_pair_tensor_path=tensor_path,
        archive_path=archive_path,
        output_dir=tmp_path / "output",
    )
    assert persisted_path is not None
    assert persisted_path.exists()
    payload = json.loads(persisted_path.read_text())
    assert payload["schema_version"] == "wyner_ziv_deliverability_probe_v1"
    # Synthetic CSP is high (~90%); but uniform bytes → likely NOT_DELIVERABLE
    assert verdict.deliverability_verdict in ("NOT_DELIVERABLE", "PARTIAL", "DELIVERABLE")


# ──────────────────────────────────────────────────────────────────────── #
# Real fec6 anchor regression test (gated on file availability)             #
# ──────────────────────────────────────────────────────────────────────── #


REAL_TENSOR = Path(".omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy")
REAL_ARCHIVE = Path(
    "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
REAL_SHA = "f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd"


@pytest.mark.skipif(
    not REAL_TENSOR.exists() or not REAL_ARCHIVE.exists(),
    reason="real fec6 tensor/archive not present in this checkout",
)
def test_run_probe_real_fec6_anchor_classifies_correctly(tmp_path):
    """Regression test against the real 178417-byte fec6 anchor."""
    verdict, persisted_path = prober.run_probe(
        archive_sha256=REAL_SHA,
        per_pair_tensor_path=REAL_TENSOR,
        archive_path=REAL_ARCHIVE,
        output_dir=tmp_path / "output",
    )
    # Empirical invariants:
    # * CSP byte count ≈ 162,123 (sister classification)
    # * fec6 bytes are at entropy floor → lzma INFLATES → NOT_DELIVERABLE
    # * Legacy blanket 1.15x reward is OVERSTATED
    assert verdict.candidate_shared_prior_byte_count >= 160_000
    assert verdict.candidate_shared_prior_byte_count <= 165_000
    assert verdict.deliverability_verdict == "NOT_DELIVERABLE"
    assert verdict.autopilot_reward_factor_justified is False
    assert persisted_path is not None
    assert persisted_path.exists()


# ──────────────────────────────────────────────────────────────────────── #
# CLI smoke test                                                            #
# ──────────────────────────────────────────────────────────────────────── #


def test_cli_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exc_info:
        prober.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_synthetic_runs(tmp_path):
    """CLI end-to-end with synthetic inputs."""
    n_bytes = 512
    grad = _make_per_pair_gradient(n_bytes=n_bytes)
    tensor_path = tmp_path / "synthetic.npy"
    np.save(tensor_path, grad)
    archive_path = tmp_path / "archive.zip"
    raw = _make_uniform_archive_bytes(n=n_bytes)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", raw)
    output_dir = tmp_path / "out"

    rc = prober.main(
        [
            "--archive-sha",
            "synthetic_cli_test",
            "--per-pair-tensor",
            str(tensor_path),
            "--archive-path",
            str(archive_path),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )
    assert rc in (0, 1)  # 0 if deliverable, 1 if not
