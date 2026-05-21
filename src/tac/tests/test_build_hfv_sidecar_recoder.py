# SPDX-License-Identifier: MIT
"""Tests for ``tools/build_hfv_sidecar_recoder.py`` (OVERNIGHT-X2).

Round-trip + size-reduction + canonical-Provenance discipline tests
per Catalog #229 PV + Catalog #287/#323 + Carmack MVP-first.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


# Dynamically load the recoder module (it's in tools/, not src/tac/)
_RECODER_PATH = Path(__file__).resolve().parents[3] / "tools" / "build_hfv_sidecar_recoder.py"
_spec = importlib.util.spec_from_file_location("build_hfv_sidecar_recoder", _RECODER_PATH)
assert _spec is not None and _spec.loader is not None
recoder = importlib.util.module_from_spec(_spec)
# Register the module in sys.modules BEFORE exec so frozen dataclass invariants
# can introspect via cls.__module__ lookup.
sys.modules["build_hfv_sidecar_recoder"] = recoder
_spec.loader.exec_module(recoder)


# ---------------------------------------------------------------------------
# Test 1: bit-identical round-trip across all 3 canonical strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["entropy_brotli", "sparse_delta", "combined"])
def test_round_trip_bit_identical_synthetic_identity(strategy: str) -> None:
    """Identity foveation (1 unique row * 1200) must round-trip bit-identically."""
    fixture = recoder.build_synthetic_smoke_fixture(
        n_frames=1200, sparse_pair_indices=[]
    )
    encoded = recoder.recode_foveation_params(fixture, strategy=strategy)
    decoded = recoder.decode_recoded_sidecar(encoded)
    assert decoded == fixture, f"strategy={strategy} round-trip FAILED"
    assert hashlib.sha256(decoded).hexdigest() == hashlib.sha256(fixture).hexdigest()


@pytest.mark.parametrize("strategy", ["entropy_brotli", "sparse_delta", "combined"])
def test_round_trip_bit_identical_synthetic_sparse(strategy: str) -> None:
    """1-sparse-pair foveation must round-trip bit-identically."""
    fixture = recoder.build_synthetic_smoke_fixture(
        n_frames=1200, sparse_pair_indices=[0, 5, 100, 599]
    )
    encoded = recoder.recode_foveation_params(fixture, strategy=strategy)
    decoded = recoder.decode_recoded_sidecar(encoded)
    assert decoded == fixture
    assert hashlib.sha256(decoded).hexdigest() == hashlib.sha256(fixture).hexdigest()


# ---------------------------------------------------------------------------
# Test 2: size reduction (output < input for all canonical strategies)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["entropy_brotli", "sparse_delta", "combined"])
def test_size_reduction_identity(strategy: str) -> None:
    """Identity foveation must compress LOSSLESSLY to less than dense size."""
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=1200)
    encoded = recoder.recode_foveation_params(fixture, strategy=strategy)
    assert len(encoded) < len(fixture), (
        f"strategy={strategy} OUTPUT_BYTES={len(encoded)} >= "
        f"INPUT_BYTES={len(fixture)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Strategy ``combined`` achieves target ≤ 10,000 bytes
# ---------------------------------------------------------------------------


def test_combined_strategy_meets_target_byte_ceiling() -> None:
    """OVERNIGHT-S Path 3 spec: combined strategy MUST shrink 24KB → ≤10KB."""
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=1200)
    encoded = recoder.recode_foveation_params(fixture, strategy="combined")
    target_bytes = 10_000
    assert len(encoded) <= target_bytes, (
        f"combined strategy output={len(encoded)} > target={target_bytes}"
    )


def test_combined_strategy_meets_target_on_live_sparse_fixture(tmp_path: Path) -> None:
    """Combined strategy on synthetic 16-sparse-pair (mirrors live seed_top16) ≤10KB."""
    fixture = recoder.build_synthetic_smoke_fixture(
        n_frames=1200, sparse_pair_indices=list(range(16))
    )
    encoded = recoder.recode_foveation_params(fixture, strategy="combined")
    assert len(encoded) <= 10_000, (
        f"combined on 16-sparse fixture output={len(encoded)} > 10000"
    )


# ---------------------------------------------------------------------------
# Test 4: graceful failure on malformed input
# ---------------------------------------------------------------------------


def test_parse_hfv1_raises_on_truncated() -> None:
    with pytest.raises(ValueError, match="truncated"):
        recoder.parse_hfv1(b"\x00\x00")


def test_parse_hfv1_raises_on_wrong_magic() -> None:
    bad = b"BADM" + b"\x00" * 12
    with pytest.raises(ValueError, match="magic mismatch"):
        recoder.parse_hfv1(bad)


def test_parse_hfv1_raises_on_size_mismatch() -> None:
    # Header claims 100 frames but only header bytes provided
    bad = recoder.HFV1_HEADER.pack(b"HFV1", 100, 874, 1164)
    with pytest.raises(ValueError, match="size mismatch"):
        recoder.parse_hfv1(bad)


def test_decode_hfrc_raises_on_wrong_magic() -> None:
    with pytest.raises(ValueError, match="magic mismatch"):
        recoder.decode_recoded_sidecar(b"BADM" + b"\x00" * 16)


def test_recode_raises_on_unknown_strategy() -> None:
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=4)
    with pytest.raises(ValueError, match="unknown strategy"):
        recoder.recode_foveation_params(fixture, strategy="bogus_strategy")


# ---------------------------------------------------------------------------
# Test 5: deterministic output (sha256 deterministic for same input + strategy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["entropy_brotli", "sparse_delta", "combined"])
def test_deterministic_output(strategy: str) -> None:
    """Same input + same strategy → same sha256 (no randomness in encoder)."""
    fixture = recoder.build_synthetic_smoke_fixture(
        n_frames=1200, sparse_pair_indices=[0, 5]
    )
    e1 = recoder.recode_foveation_params(fixture, strategy=strategy)
    e2 = recoder.recode_foveation_params(fixture, strategy=strategy)
    assert hashlib.sha256(e1).hexdigest() == hashlib.sha256(e2).hexdigest(), (
        f"strategy={strategy} encoder is NON-DETERMINISTIC"
    )


# ---------------------------------------------------------------------------
# Test 6: Catalog #287 evidence-tag discipline + canonical Provenance
# ---------------------------------------------------------------------------


def test_verdict_carries_prediction_axis_tag_and_non_promotable() -> None:
    """Per Catalog #287/#323: output verdict MUST carry [prediction] tag +
    promotable=False + score_claim=False (canonical Provenance defaults)."""
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=1200)
    encoded = recoder.recode_foveation_params(fixture, strategy="combined")
    # Build a verdict by hand to inspect canonical Provenance
    v = recoder.RecoderVerdict(
        input_path="x",
        output_path="y",
        encoding_strategy="combined",
        input_bytes=len(fixture),
        output_bytes=len(encoded),
        round_trip_verified=True,
    )
    assert v.axis_tag == "[prediction]", "missing canonical [prediction] axis tag"
    assert v.promotable is False, "promotable MUST default False"
    assert v.score_claim is False, "score_claim MUST default False"
    assert v.canonical_equation_reference == "hfv2_sparse_pair_sidecar_replacement_savings_v1"


def test_predicted_rate_savings_matches_canonical_equation_356() -> None:
    """Per canonical equation #356: deltaS = -25*(N_dense - N_recoded)/37_545_489."""
    rate = recoder.predicted_rate_savings(input_bytes=24016, output_bytes=64)
    expected = -25.0 * (24016 - 64) / 37_545_489
    assert abs(rate - expected) < 1e-12, f"rate={rate} expected={expected}"
    # Sanity: input=output produces zero
    assert recoder.predicted_rate_savings(input_bytes=100, output_bytes=100) == 0.0
    # Sanity: output > input produces POSITIVE delta (bad)
    pos = recoder.predicted_rate_savings(input_bytes=100, output_bytes=200)
    assert pos > 0, "output > input MUST produce positive delta (recoder backfired)"


# ---------------------------------------------------------------------------
# Test 7: smoke mode runs without paid GPU (no Modal/Vast invocations)
# ---------------------------------------------------------------------------


def test_smoke_mode_runs_locally_no_paid_gpu(tmp_path: Path) -> None:
    """Carmack MVP-first Step 1: smoke is FREE local CPU only."""
    report = tmp_path / "smoke_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_RECODER_PATH),
            "--smoke",
            "--encoding-strategy", "combined",
            "--verify-roundtrip",
            "--output-recoded-sidecar", str(tmp_path / "recoded.bin"),
            "--report-out-json", str(report),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"smoke exited rc={result.returncode}\nstdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    assert report.is_file()
    data = json.loads(report.read_text())
    assert data["round_trip_verified"] is True
    assert data["under_target"] is True
    assert data["axis_tag"] == "[prediction]"
    assert data["promotable"] is False
    assert data["score_claim"] is False
    assert data["canonical_equation_reference"] == "hfv2_sparse_pair_sidecar_replacement_savings_v1"
    assert data["lane_id"] == "lane_overnight_x2_build_hfv_sidecar_recoder_20260521"


def test_dry_run_does_not_write_output(tmp_path: Path) -> None:
    """--dry-run must compute verdict without touching the output path."""
    out_path = tmp_path / "should_not_exist.bin"
    result = subprocess.run(
        [
            sys.executable,
            str(_RECODER_PATH),
            "--smoke",
            "--encoding-strategy", "combined",
            "--dry-run",
            "--output-recoded-sidecar", str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert not out_path.is_file(), "--dry-run wrote output file (must not)"


def test_no_verify_roundtrip_flag_disables_roundtrip(tmp_path: Path) -> None:
    """--no-verify-roundtrip disables the round-trip check (NOT RECOMMENDED)."""
    out_path = tmp_path / "recoded_no_verify.bin"
    result = subprocess.run(
        [
            sys.executable,
            str(_RECODER_PATH),
            "--smoke",
            "--encoding-strategy", "combined",
            "--no-verify-roundtrip",
            "--output-recoded-sidecar", str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    # Even with no-verify, the output should still be written
    assert out_path.is_file()


# ---------------------------------------------------------------------------
# Additional smoke: live fixture (if exists) integration
# ---------------------------------------------------------------------------


def test_live_seed_top16_fixture_integration(tmp_path: Path) -> None:
    """Live seed_top16 foveation_params.bin (16 sparse pairs) → ≤10KB via combined."""
    live = Path(
        "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
        "official_inflate_control/data_seed_top16_component_hardpairs/foveation_params.bin"
    )
    if not live.is_file():
        pytest.skip("live seed_top16 fixture not present")
    out_path = tmp_path / "seed_top16_recoded.bin"
    result = subprocess.run(
        [
            sys.executable,
            str(_RECODER_PATH),
            "--input-foveation-params-bin", str(live),
            "--output-recoded-sidecar", str(out_path),
            "--encoding-strategy", "combined",
            "--verify-roundtrip",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"live recoder rc={result.returncode}\n{result.stderr}"
    data = json.loads(result.stdout)
    assert data["round_trip_verified"] is True
    assert data["input_bytes"] == 24016
    assert data["output_bytes"] <= 10_000
    assert data["under_target"] is True


# ---------------------------------------------------------------------------
# Strategy-byte dispatch correctness
# ---------------------------------------------------------------------------


def test_strategy_byte_dispatch_distinguishes_outputs() -> None:
    """Each strategy must produce a distinct strategy_byte in the HFRC header."""
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=4)
    e_brotli = recoder.recode_foveation_params(fixture, strategy="entropy_brotli")
    e_sparse = recoder.recode_foveation_params(fixture, strategy="sparse_delta")
    e_combined = recoder.recode_foveation_params(fixture, strategy="combined")
    # Strategy byte is at offset 16 (after magic+n+h+w)
    assert e_brotli[16] == recoder.ENCODING_ENTROPY_BROTLI
    assert e_sparse[16] == recoder.ENCODING_SPARSE_DELTA
    assert e_combined[16] == recoder.ENCODING_COMBINED


def test_decode_recoded_sidecar_auto_dispatches_on_strategy_byte() -> None:
    """The canonical decoder must auto-dispatch on the strategy byte."""
    fixture = recoder.build_synthetic_smoke_fixture(n_frames=8, sparse_pair_indices=[0, 1])
    for strategy in ("entropy_brotli", "sparse_delta", "combined"):
        encoded = recoder.recode_foveation_params(fixture, strategy=strategy)
        decoded = recoder.decode_recoded_sidecar(encoded)
        assert decoded == fixture, f"auto-dispatch failed for {strategy}"


# ---------------------------------------------------------------------------
# Varint helpers (sparse_delta internal correctness)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 127, 128, 16383, 16384, 1_000_000])
def test_varint_round_trip(value: int) -> None:
    encoded = recoder._varint_encode(value)
    decoded, offset = recoder._varint_decode(encoded, 0)
    assert decoded == value
    assert offset == len(encoded)


def test_varint_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        recoder._varint_encode(-1)
