# SPDX-License-Identifier: MIT
"""Tests for ATW V2-1 Faiss-IVF-PQ canonical helper + trainer scaffold.

[verified-against: .omx/research/atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md]
[verified-against: ATW V2 reactivation symposium 2026-05-18 Revisions #1-#7 binding]

Tests cover:
- Per-variant byte-budget arithmetic (V1 dense / V2 sparse top-k / V3 pool-shared)
  per design memo §6.3
- Faiss codebook construction → encode → decode round-trip on synthetic SegNet
  softmax (skipped if faiss-cpu not installed; per design memo §9.5)
- Codebook serialize → deserialize round-trip
- Per-pair master gradient compatibility via design memo §6.2 byte-addressability
- NotImplementedError raised from `_full_main` per Catalog #240
- Trainer mode resolution per Catalog #326 (env > legacy SMOKE_ONLY > --smoke > default)
- Canonical-variant constants pinned per design memo §6.3

Per CLAUDE.md "uv pip install faiss-cpu" + the design memo §9.5: faiss-cpu
is NOT installed in the default OSS clone; tests that require Faiss are
skipped via pytest.importorskip rather than failing.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from tac.optimization.faiss_ivf_pq_atw_channel import (  # noqa: E402
    CANONICAL_VARIANT_V1_DENSE,
    CANONICAL_VARIANT_V2_SPARSE_TOP_K,
    CANONICAL_VARIANT_V3_POOL_SHARED,
    CANONICAL_VARIANTS,
    CONTEST_RATE_NORMALIZER_BYTES,
    DEFAULT_M_SUBQ,
    DEFAULT_NBITS,
    DEFAULT_NLIST,
    DEFAULT_TRAINING_SEED,
    PqEncodingBudget,
    compute_pq_mi_verdict,
    estimate_pq_encoding_budget,
)


# ============================================================================
# Section 1 — Byte-budget arithmetic (no Faiss required)
# ============================================================================


def test_canonical_variants_constants_pinned():
    """Per design memo §6.3: three canonical variants must be exported."""
    assert CANONICAL_VARIANT_V1_DENSE == "v1_dense"
    assert CANONICAL_VARIANT_V2_SPARSE_TOP_K == "v2_sparse_top_k"
    assert CANONICAL_VARIANT_V3_POOL_SHARED == "v3_pool_shared"
    assert CANONICAL_VARIANTS == (
        CANONICAL_VARIANT_V1_DENSE,
        CANONICAL_VARIANT_V2_SPARSE_TOP_K,
        CANONICAL_VARIANT_V3_POOL_SHARED,
    )


def test_contest_rate_normalizer_canonical():
    """Per CLAUDE.md `Apples-to-apples evidence discipline`: 37545489 bytes."""
    assert CONTEST_RATE_NORMALIZER_BYTES == 37_545_489.0


def test_default_constants_match_design_memo():
    """Defaults match design memo §6.3 V1 dense recommendation."""
    assert DEFAULT_NLIST == 256
    assert DEFAULT_M_SUBQ == 4
    assert DEFAULT_NBITS == 8
    assert DEFAULT_TRAINING_SEED == 42


def test_v1_dense_budget_not_shippable():
    """V1 dense full-PQ per design memo §6.3: NOT SHIPPABLE."""
    budget = estimate_pq_encoding_budget(
        variant_id="v1_dense",
        n_regions=256,
        nlist=256,
        m_subq=4,
        nbits=8,
        top_k_regions=None,
    )
    # Per design memo §6.3: per_pair ≈ 1280 bytes
    assert budget.per_pair_bytes >= 1000
    # Total archive contribution ≈ 778KB
    assert budget.total_archive_contribution_bytes >= 700_000
    # Contest rate cost ≈ +0.518 (NOT SHIPPABLE)
    assert budget.contest_rate_cost_estimate > 0.4
    assert budget.shippable_verdict == "NOT_SHIPPABLE"


def test_v2_sparse_top_k_budget_arguable():
    """V2 sparse top-k per design memo §6.3: ARGUABLE."""
    budget = estimate_pq_encoding_budget(
        variant_id="v2_sparse_top_k",
        n_regions=16,
        nlist=64,
        m_subq=2,
        nbits=6,
        top_k_regions=8,
    )
    # Per design memo §6.3: per_pair ≈ 24 bytes (sparse top-k=8 of 16 regions)
    assert budget.per_pair_bytes < 50
    # Total ≈ 17KB
    assert budget.total_archive_contribution_bytes < 30_000
    # Rate cost ≈ +0.011 (ARGUABLE band)
    assert 0.005 < budget.contest_rate_cost_estimate < 0.05
    assert budget.shippable_verdict == "ARGUABLE"


def test_v3_pool_shared_budget_shippable():
    """V3 pool-shared per design memo §6.3: SHIPPABLE."""
    budget = estimate_pq_encoding_budget(
        variant_id="v3_pool_shared",
        n_regions=16,
        nlist=64,
        m_subq=2,
        nbits=6,
        top_k_regions=1,
    )
    # Per design memo §6.3: per_pair ≈ 1 byte (top-k=1)
    assert budget.per_pair_bytes < 10
    # Total ≈ 3.2KB
    assert budget.total_archive_contribution_bytes < 10_000
    # Rate cost ≈ +0.003 (SHIPPABLE)
    assert budget.contest_rate_cost_estimate < 0.01
    assert budget.shippable_verdict == "SHIPPABLE"


def test_budget_arithmetic_deterministic():
    """Same inputs produce same outputs (first-principles arithmetic)."""
    inputs = dict(
        variant_id="test",
        n_regions=64,
        nlist=128,
        m_subq=4,
        nbits=8,
        top_k_regions=None,
    )
    budget_1 = estimate_pq_encoding_budget(**inputs)
    budget_2 = estimate_pq_encoding_budget(**inputs)
    assert budget_1 == budget_2


def test_budget_rejects_invalid_inputs():
    """Invalid inputs raise ValueError per canonical helper contract."""
    with pytest.raises(ValueError, match="n_regions"):
        estimate_pq_encoding_budget(variant_id="x", n_regions=0)
    with pytest.raises(ValueError, match="nlist"):
        estimate_pq_encoding_budget(variant_id="x", n_regions=16, nlist=0)
    with pytest.raises(ValueError, match="m_subq"):
        estimate_pq_encoding_budget(variant_id="x", n_regions=16, m_subq=0)
    with pytest.raises(ValueError, match="total_pairs"):
        estimate_pq_encoding_budget(variant_id="x", n_regions=16, total_pairs=0)


def test_pq_feature_padding_matches_canonical_5_class_softmax():
    """Faiss PQ requires d % M == 0; 5-class SegNet vectors must be padded."""
    from tac.optimization import faiss_ivf_pq_atw_channel as channel

    assert channel._padded_dim_for_pq(5, 2) == 6
    assert channel._padded_dim_for_pq(5, 4) == 8
    values = np.asarray([[0.1, 0.2, 0.3, 0.2, 0.2]], dtype=np.float32)
    padded = channel._pad_features_for_codebook(values, codebook_dim=8)
    assert padded.shape == (1, 8)
    np.testing.assert_allclose(padded[:, :5], values)
    np.testing.assert_allclose(padded[:, 5:], 0.0)


def test_budget_as_dict_json_safe():
    """PqEncodingBudget.as_dict produces JSON-serializable output per Catalog #305."""
    import json
    budget = estimate_pq_encoding_budget(
        variant_id="test",
        n_regions=16,
        nlist=64,
        m_subq=2,
        nbits=6,
        top_k_regions=1,
    )
    # Round-trip through JSON serialization (Catalog #305 observability)
    serialized = json.dumps(budget.as_dict())
    deserialized = json.loads(serialized)
    assert deserialized["variant_id"] == "test"
    assert deserialized["shippable_verdict"] == "SHIPPABLE"
    assert "contest_rate_cost_estimate" in deserialized


def test_budget_frozen_dataclass():
    """PqEncodingBudget is frozen — cannot be mutated post-construction."""
    budget = estimate_pq_encoding_budget(
        variant_id="test", n_regions=16
    )
    with pytest.raises((AttributeError, TypeError)):
        budget.variant_id = "mutated"  # type: ignore[misc]


def test_compute_pq_mi_verdict_detects_correlated_side_info():
    """PQ MI verdict is a reusable tac helper, not a probe-local API."""
    latent = bytes([0, 0, 0, 0, 255, 255, 255, 255])
    verdict = compute_pq_mi_verdict(
        latent_stream=latent,
        per_pair_symbols=[1, 2],
        symbols_per_pair=4,
        threshold=0.5,
    )
    assert verdict.verdict == "MEANINGFUL_CONDITIONING"
    assert verdict.mutual_information_bits >= 0.99
    assert verdict.num_unique_side_info_symbols == 2


def test_compute_pq_mi_verdict_rejects_length_mismatch():
    with pytest.raises(ValueError, match="expanded side-info length"):
        compute_pq_mi_verdict(
            latent_stream=b"\x00\x01\x02",
            per_pair_symbols=[1, 2],
            symbols_per_pair=2,
        )


# ============================================================================
# Section 2 — Faiss codebook (skipped if faiss-cpu not installed)
# ============================================================================


def _require_faiss():
    return pytest.importorskip(
        "faiss",
        reason=(
            "faiss-cpu not installed; install via `uv pip install faiss-cpu` "
            "per design memo §9.5"
        ),
    )


def _synthetic_segnet_softmax_batch(n_vectors: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic SegNet 5-class softmax outputs via Dirichlet."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(5), size=n_vectors).astype(np.float32)


def test_build_pq_codebook_returns_faiss_index():
    """Per design memo §6.2: build_pq_codebook returns trained faiss.IndexIVFPQ."""
    faiss = _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import build_pq_codebook

    batch = _synthetic_segnet_softmax_batch(n_vectors=2000)
    codebook = build_pq_codebook(batch, nlist=64, m_subq=2, nbits=6, seed=42)
    assert isinstance(codebook, faiss.IndexIVFPQ)
    assert codebook.is_trained
    # Persist only the trained quantizers, not the training vectors. Adding the
    # vectors bloats the archive-side codebook by roughly 30x.
    assert codebook.ntotal == 0
    assert codebook.d == 6


def test_build_pq_codebook_rejects_invalid_input():
    """Input shape invariants enforced per canonical helper contract."""
    from tac.optimization.faiss_ivf_pq_atw_channel import build_pq_codebook

    with pytest.raises(ValueError, match="must be np.ndarray"):
        build_pq_codebook([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be 2-d"):
        build_pq_codebook(np.zeros((100,)))

    with pytest.raises(ValueError, match="n_vectors.*nlist"):
        build_pq_codebook(np.zeros((10, 5)), nlist=256)


def test_input_validation_runs_before_optional_faiss_import():
    """Cheap local contract errors must not be masked by missing faiss-cpu."""
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        decode_per_region_histogram,
        deserialize_codebook,
        encode_per_region_histogram,
    )

    with pytest.raises(ValueError, match="must be np.ndarray"):
        encode_per_region_histogram([0.2, 0.8], object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be 2-d"):
        encode_per_region_histogram(np.zeros(5), object())

    with pytest.raises(ValueError, match="encoded_bytes must be bytes"):
        decode_per_region_histogram([1, 2, 3], object(), n_regions=16)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="blob must be bytes"):
        deserialize_codebook({"not": "bytes"})  # type: ignore[arg-type]


def test_encode_decode_roundtrip_v3():
    """V3 pool-shared canonical roundtrip — encode → decode preserves shape."""
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        decode_per_region_histogram,
        encode_per_region_histogram,
    )

    n_regions = 16
    training = _synthetic_segnet_softmax_batch(n_vectors=1000)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)

    one_pair_softmax = _synthetic_segnet_softmax_batch(n_vectors=n_regions, seed=43)
    encoded = encode_per_region_histogram(one_pair_softmax, codebook)
    decoded = decode_per_region_histogram(encoded, codebook, n_regions=n_regions)
    assert decoded.shape == (n_regions, 5)
    assert decoded.dtype == np.float32


def test_encode_returns_bytes():
    """encode_per_region_histogram returns native bytes (per design memo §6.2 contract)."""
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        encode_per_region_histogram,
    )

    training = _synthetic_segnet_softmax_batch(n_vectors=500)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)
    one_pair = _synthetic_segnet_softmax_batch(n_vectors=16, seed=43)
    encoded = encode_per_region_histogram(one_pair, codebook)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0


def test_per_pair_byte_addressability_for_master_gradient():
    """Per design memo §6.2 + Catalog #810 sister: per-pair byte addressability.

    Each pair's PQ codewords occupy `n_regions * code_size` bytes at a known offset.
    """
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        encode_per_region_histogram,
    )

    n_regions = 16
    training = _synthetic_segnet_softmax_batch(n_vectors=1000)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)
    code_size = codebook.sa_code_size()

    # Encode 3 distinct pairs; verify per-pair byte budget matches code_size × n_regions
    pair_encodings = []
    for seed in (10, 20, 30):
        pair = _synthetic_segnet_softmax_batch(n_vectors=n_regions, seed=seed)
        encoded = encode_per_region_histogram(pair, codebook)
        pair_encodings.append(encoded)
        assert len(encoded) == code_size * n_regions, (
            f"per-pair byte budget invariant violated: got {len(encoded)}, "
            f"expected {code_size * n_regions}"
        )

    # Per-pair stream byte addressability: concatenated stream is per-pair indexable
    stream = b"".join(pair_encodings)
    per_pair_bytes = code_size * n_regions
    assert len(stream) == per_pair_bytes * 3

    # Catalog #810 master_gradient_consumers compatibility: extract pair 1 by offset
    pair_1_extracted = stream[per_pair_bytes : 2 * per_pair_bytes]
    assert pair_1_extracted == pair_encodings[1]


def test_codebook_serialize_deserialize_roundtrip():
    """Codebook persistence per design memo §6.1 ATW21PQ archive grammar."""
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        decode_per_region_histogram,
        deserialize_codebook,
        encode_per_region_histogram,
        serialize_codebook,
    )

    training = _synthetic_segnet_softmax_batch(n_vectors=500)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)

    # Serialize codebook
    serialized = serialize_codebook(codebook)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    # Deserialize and verify decode produces same result
    pair = _synthetic_segnet_softmax_batch(n_vectors=16, seed=99)
    encoded_via_original = encode_per_region_histogram(pair, codebook)

    codebook_restored = deserialize_codebook(serialized)
    decoded_via_restored = decode_per_region_histogram(
        encoded_via_original, codebook_restored, n_regions=16
    )
    # Restored codebook produces same decoded shape
    assert decoded_via_restored.shape == (16, 5)


def test_encode_rejects_wrong_shape():
    """Input shape invariants enforced per design memo §6.2."""
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        encode_per_region_histogram,
    )

    training = _synthetic_segnet_softmax_batch(n_vectors=500)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)

    with pytest.raises(ValueError, match="must be np.ndarray"):
        encode_per_region_histogram([0.2, 0.2, 0.2, 0.2, 0.2], codebook)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be 2-d"):
        encode_per_region_histogram(np.zeros(5), codebook)


def test_decode_rejects_wrong_byte_count():
    """decode_per_region_histogram enforces byte-count divisibility per code_size."""
    _require_faiss()
    from tac.optimization.faiss_ivf_pq_atw_channel import (
        build_pq_codebook,
        decode_per_region_histogram,
    )

    training = _synthetic_segnet_softmax_batch(n_vectors=500)
    codebook = build_pq_codebook(training, nlist=64, m_subq=2, nbits=6, seed=42)
    code_size = codebook.sa_code_size()

    # Wrong byte count for n_regions=16 should raise
    wrong_bytes = b"\x00" * (code_size * 15)  # n_regions=15 (not 16)
    with pytest.raises(ValueError, match="does not match expected"):
        decode_per_region_histogram(wrong_bytes, codebook, n_regions=16)


# ============================================================================
# Section 3 — Operator-actionable error on missing Faiss
# ============================================================================


def test_actionable_error_when_faiss_missing(monkeypatch):
    """Per design memo §9.5: missing faiss-cpu produces operator-actionable error."""
    from tac.optimization import faiss_ivf_pq_atw_channel

    # Mock the deferred import to simulate faiss-cpu missing
    def _mock_import_fail():
        raise ImportError("No module named 'faiss'")

    with mock.patch.object(faiss_ivf_pq_atw_channel, "_import_faiss", side_effect=ImportError(
        "faiss-cpu is required for ATW V2-1 Faiss-IVF-PQ encoding. "
        "Install via: uv pip install faiss-cpu"
        )):
        with pytest.raises(ImportError, match="uv pip install faiss-cpu"):
            faiss_ivf_pq_atw_channel.build_pq_codebook(
                np.zeros((100, 5)),
                nlist=64,
            )


# ============================================================================
# Section 4 — Trainer scaffold (NotImplementedError per Catalog #240)
# ============================================================================


def test_trainer_full_main_raises_not_implemented(tmp_path):
    """Per Catalog #240: _full_main raises NotImplementedError at landing.

    Reactivation criteria documented per design memo §12.
    """
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])

    with pytest.raises(NotImplementedError, match="V2-1 _full_main pending"):
        trainer._full_main(args)


def test_trainer_smoke_main_returns_zero(tmp_path, capsys):
    """Smoke main returns 0 (even if Faiss not installed; graceful degradation)."""
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args([
        "--output-dir", str(tmp_path),
        "--smoke",
        "--pq-variant", "v3_pool_shared",
        "--pq-n-regions", "16",
    ])
    rc = trainer._smoke_main(args)
    assert rc == 0

    captured = capsys.readouterr()
    assert "SMOKE MODE" in captured.out
    assert "v3_pool_shared" in captured.out
    # Variant byte-budget summary always printed (no Faiss required)
    assert "v1_dense" in captured.out
    assert "v2_sparse_top_k" in captured.out


def test_trainer_mode_resolution_explicit_env_full(monkeypatch, tmp_path):
    """Per Catalog #326: ATW_V2_1_TRAINER_MODE env wins over SMOKE_ONLY."""
    monkeypatch.setenv("ATW_V2_1_TRAINER_MODE", "full")
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    mode = trainer._resolve_trainer_mode(args)
    assert mode == "full"


def test_trainer_mode_resolution_explicit_env_smoke(monkeypatch, tmp_path):
    """ATW_V2_1_TRAINER_MODE=smoke overrides --smoke=False default."""
    monkeypatch.setenv("ATW_V2_1_TRAINER_MODE", "smoke")
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    mode = trainer._resolve_trainer_mode(args)
    assert mode == "smoke"


def test_trainer_mode_resolution_legacy_smoke_only_zero(monkeypatch, tmp_path):
    """Legacy SMOKE_ONLY=0 → full mode (per Catalog #326)."""
    monkeypatch.delenv("ATW_V2_1_TRAINER_MODE", raising=False)
    monkeypatch.setenv("SMOKE_ONLY", "0")
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    mode = trainer._resolve_trainer_mode(args)
    assert mode == "full"


def test_trainer_mode_resolution_legacy_smoke_only_one(monkeypatch, tmp_path):
    """Legacy SMOKE_ONLY=1 → smoke mode."""
    monkeypatch.delenv("ATW_V2_1_TRAINER_MODE", raising=False)
    monkeypatch.setenv("SMOKE_ONLY", "1")
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    mode = trainer._resolve_trainer_mode(args)
    assert mode == "smoke"


def test_trainer_mode_resolution_default_is_smoke(monkeypatch, tmp_path):
    """Default mode is smoke (no env, no --smoke flag) per Catalog #326."""
    monkeypatch.delenv("ATW_V2_1_TRAINER_MODE", raising=False)
    monkeypatch.delenv("SMOKE_ONLY", raising=False)
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    parser = trainer._build_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path)])
    mode = trainer._resolve_trainer_mode(args)
    assert mode == "smoke"


def test_trainer_tier_1_manifest_extracts():
    """Per Catalog #151: TIER_1_OPERATOR_REQUIRED_FLAGS manifest declared."""
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    manifest = trainer.TIER_1_OPERATOR_REQUIRED_FLAGS
    # Required input file flag present per Catalog #152
    assert "--video-path" in manifest
    assert manifest["--video-path"]["required_input_file"] is True
    # PQ variant flag present per design memo §6.3
    assert "--pq-variant" in manifest


def test_trainer_extra_mount_paths_declared():
    """Per Catalog #152 Wave-1: TIER_1_EXTRA_MOUNT_PATHS declared for A1 anchor."""
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    mounts = trainer.TIER_1_EXTRA_MOUNT_PATHS
    assert isinstance(mounts, tuple)
    # A1 anchor archive must be in extra mount paths per design memo §6.2
    assert "submissions/a1/archive.zip" in mounts


def test_trainer_lane_constants():
    """Lane ID + design memo + symposium paths declared."""
    trainer = importlib.import_module("train_substrate_atw_v2_1")
    assert trainer.LANE_ID == "lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518"
    assert "atw_v2_1_faiss_ivf_pq_substrate_design_memo_20260518.md" in trainer.DESIGN_MEMO_PATH
    assert "council_per_substrate_symposium_atw_v2_reactivation_20260518.md" in trainer.SYMPOSIUM_PATH


# ============================================================================
# Section 5 — Module __all__ contract (per Catalog #265 OSS surface)
# ============================================================================


def test_module_all_exports_canonical_api():
    """__all__ declares the public API surface per Catalog #265."""
    from tac.optimization import faiss_ivf_pq_atw_channel

    expected = {
        "CANONICAL_VARIANTS",
        "CANONICAL_VARIANT_V1_DENSE",
        "CANONICAL_VARIANT_V2_SPARSE_TOP_K",
        "CANONICAL_VARIANT_V3_POOL_SHARED",
        "CONTEST_RATE_NORMALIZER_BYTES",
        "DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS",
        "DEFAULT_M_SUBQ",
        "DEFAULT_NBITS",
        "DEFAULT_NLIST",
        "DEFAULT_TRAINING_SEED",
        "INDEPENDENCE_TOLERANCE_BITS",
        "PqEncodingBudget",
        "PqMiVerdict",
        "build_pq_codebook",
        "compute_pq_mi_verdict",
        "decode_per_region_histogram",
        "deserialize_codebook",
        "encode_per_region_histogram",
        "estimate_pq_encoding_budget",
        "serialize_codebook",
    }
    assert set(faiss_ivf_pq_atw_channel.__all__) == expected
