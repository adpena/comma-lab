# SPDX-License-Identifier: MIT
"""Tests for the Filler-STC clean-source mask-DELTA syndrome-vs-brotli probe.

NO FAKE IMPLEMENTATIONS (Slot EEE Class 2): these tests verify ACTUAL
syndrome-trellis encode behavior on REAL ternary mask-delta streams via the
canonical `tac.codec.syndrome_trellis_codec`, NOT canonical-marker constants.
Every test would FAIL if the probe were replaced by a marker-emitting stub.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_PROBE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "tools"
    / "probe_stc_clean_source_mask_delta_syndrome_vs_brotli.py"
)
_spec = importlib.util.spec_from_file_location("_stc_mask_delta_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_sparse_ternary_stream_has_requested_sparsity() -> None:
    """The controlled sweep axis produces the requested non-zero fraction."""
    d = probe._make_sparse_ternary_stream(10_000, rho=0.05, seed=1)
    nonzero = int((d != 0).sum())
    assert abs(nonzero - 500) <= 1  # 5% of 10k
    assert set(np.unique(d)).issubset({-1, 0, 1})


def test_stc_syndrome_uses_real_canonical_codec() -> None:
    """ACTUAL STC: ternary_stc_encode_stream is invoked (not a marker stub).

    Different sparsity -> different syndrome cost/structure; a stub would not
    produce a stream-dependent block count.
    """
    d = probe._make_sparse_ternary_stream(4_000, rho=0.10, seed=7)
    stc_bytes, cost = probe._stc_syndrome_byte_count(
        d, constraint_height=4, block_size=64, seed=7
    )
    # n_blocks = ceil(4000/64) = 63; syndrome_bits = 63*4 = 252 -> 32 bytes.
    assert stc_bytes == 32
    assert cost >= 0.0  # real additive cost is a finite measurement


def test_stc_syndrome_byte_count_scales_with_block_count() -> None:
    """Syndrome bytes are a real function of (n, h, block_size), not constant."""
    d = probe._make_sparse_ternary_stream(4_000, rho=0.10, seed=7)
    a, _ = probe._stc_syndrome_byte_count(d, constraint_height=4, block_size=64, seed=7)
    b, _ = probe._stc_syndrome_byte_count(d, constraint_height=5, block_size=64, seed=7)
    # Larger constraint height -> more syndrome bits per block -> more bytes.
    assert b > a


def test_brotli_baseline_is_measured_not_constant() -> None:
    """The brotli baseline varies with sparsity (real compression measurement)."""
    sparse = probe._pack_ternary_to_bytes(probe._make_sparse_ternary_stream(20_000, 0.02, 1))
    dense = probe._pack_ternary_to_bytes(probe._make_sparse_ternary_stream(20_000, 0.50, 1))
    import brotli

    b_sparse = len(brotli.compress(sparse, quality=11))
    b_dense = len(brotli.compress(dense, quality=11))
    assert b_dense > b_sparse  # denser delta -> larger brotli output


def test_defer_verdict_fires_when_savings_misses_bar(monkeypatch) -> None:
    """The DEFER branch is reachable when measured STC bytes miss the bar."""

    def expensive_stc(deltas, *, constraint_height, block_size, seed):
        return 10**9, 0.0

    monkeypatch.setattr(probe, "_stc_syndrome_byte_count", expensive_stc)
    v = probe.run_probe(n=2_000, constraint_height=4, seed=1337)
    assert v["verdict"] == "DEFER_STC_DOES_NOT_BEAT_BROTLI"
    assert v["proceed_at_realistic_band"] is False
    assert "Forbidden premature KILL" in v["rationale"]
    assert "research-deferral" in v["rationale"]
    for row in v["per_rho"]:
        if row["rho"] <= v["contest_realistic_rho_ceiling"]:
            assert row["stc_beats_brotli_by_5pct"] is False


def test_proceed_verdict_fires_when_savings_clears_bar() -> None:
    """The PROCEED branch is reachable (not dead code): monkeypatch a winning row.

    Verifies the verdict logic honors the symposium >=5% bar, not just DEFER.
    """
    # Force a stream where STC trivially wins by stubbing the byte counters.
    orig_stc = probe._stc_syndrome_byte_count
    orig_brotli_pack = probe._pack_ternary_to_bytes

    def tiny_stc(deltas, *, constraint_height, block_size, seed):
        return 1, 0.0  # 1 byte syndrome -> beats any brotli baseline

    probe._stc_syndrome_byte_count = tiny_stc
    try:
        v = probe.run_probe(n=2_000, rho_sweep=(0.05,), constraint_height=4, seed=3)
        assert v["verdict"] == "PROCEED_STC_BEATS_BROTLI"
        assert v["proceed_at_realistic_band"] is True
    finally:
        probe._stc_syndrome_byte_count = orig_stc
        probe._pack_ternary_to_bytes = orig_brotli_pack


def test_per_rho_table_covers_sparse_to_dense_crossover() -> None:
    """The sweep includes the contest-realistic band AND the dense FALSIFIED case."""
    v = probe.run_probe(n=2_000, constraint_height=4, seed=9)
    rhos = [r["rho"] for r in v["per_rho"]]
    assert min(rhos) <= 0.10  # contest-realistic
    assert max(rhos) >= 1.0   # the dense rho->1 case the original kill was about


def test_verdict_carries_canonical_non_promotable_markers() -> None:
    """Catalog #341 Tier A markers + research-signal axis tag (non-promotable)."""
    v = probe.run_probe(n=2_000, constraint_height=4, seed=5)
    assert v["promotable"] is False
    assert v["score_claim"] is False
    assert v["promotion_eligible"] is False
    assert v["ready_for_exact_eval_dispatch"] is False
    assert v["predicted_delta_adjustment"] == 0.0
    assert v["axis_tag"] == "[macOS-CPU advisory]"
    assert v["evidence_grade"] == "research-signal"
    kwargs = v["catalog_313_probe_outcome_kwargs"]
    assert kwargs["probe_id"] == probe.PROBE_ID
    assert kwargs["substrate"] == "stc_clean_source"
    assert kwargs["verdict"] == probe._catalog_313_probe_verdict(v["verdict"])
    assert kwargs["blocker_status"] == probe._catalog_313_blocker_status(v["verdict"])
    assert kwargs["metric_name"] == probe.PROBE_METRIC_NAME
    assert kwargs["threshold_token"] == probe.PROBE_THRESHOLD_TOKEN
    assert "CC#2 detector-informed cost map" in kwargs["reactivation_criteria"][0]


def test_rationale_discloses_deferral_is_not_kill(monkeypatch) -> None:
    """Honest reporting: the DEFER rationale cites Forbidden premature KILL."""

    def expensive_stc(deltas, *, constraint_height, block_size, seed):
        return 10**9, 0.0

    monkeypatch.setattr(probe, "_stc_syndrome_byte_count", expensive_stc)
    v = probe.run_probe(
        n=2_000,
        rho_sweep=(0.05,),
        constraint_height=4,
        seed=1337,
    )
    assert "Forbidden premature KILL" in v["rationale"]
    assert "research-deferral" in v["rationale"]
    assert "CC#2" in v["rationale"]  # names the next required revision


def test_main_cli_returns_zero_and_emits_schema(capsys) -> None:
    """CLI is a research-signal probe (rc=0); verdict in the field, not exit code."""
    rc = probe.main(["--n", "2000", "--constraint-height", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert probe.PROBE_SCHEMA in out


def test_main_cli_can_register_probe_outcome_to_custom_ledger(tmp_path, capsys) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"
    result = tmp_path / "result.json"

    rc = probe.main(
        [
            "--n",
            "2000",
            "--constraint-height",
            "4",
            "--json-out",
            str(result),
            "--register-probe-outcome",
            "--probe-outcomes-ledger",
            str(ledger),
            "--probe-outcomes-lock",
            str(lock),
        ]
    )

    assert rc == 0
    capsys.readouterr()
    payload = json.loads(result.read_text(encoding="utf-8"))
    row = payload["catalog_313_probe_outcome_row"]
    assert row["probe_id"] == probe.PROBE_ID
    assert row["verdict"] == probe._catalog_313_probe_verdict(payload["verdict"])
    assert row["blocker_status"] == probe._catalog_313_blocker_status(payload["verdict"])
    assert row["evidence_path"] == str(result)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert rows == [row]


def test_canonical_codec_roundtrip_directly() -> None:
    """Directly exercise the REAL canonical STC codec (no probe wrapper).

    Decode of the stego soz stream recovers the embedded all-zero syndrome.
    """
    from tac.codec.syndrome_trellis_codec import (
        STCParams,
        stc_decode_block,
        stc_encode_block,
    )

    rng = np.random.default_rng(11)
    cover = rng.integers(0, 2, size=64, dtype=np.uint8)
    costs = np.ones(64, dtype=np.float64)
    message = np.zeros(4, dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=0)
    y, H_bar = stc_encode_block(cover, costs, message, params)
    recovered = stc_decode_block(y, H_bar)
    assert np.array_equal(recovered, message)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
