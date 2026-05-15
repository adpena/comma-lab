# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "probe_pr106_format0b_sidecar_compression.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "probe_pr106_format0b_sidecar_compression_under_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_exact_radix_candidate_saves_fourteen_sidecar_bytes() -> None:
    schema = mod.PR106_PR101_RANKED_SCHEMA
    dims = np.asarray([idx % schema.n_dims for idx in range(schema.n_pairs)], dtype=np.int64)
    deltas = np.full(schema.n_pairs, schema.deltas[0], dtype=np.int64)

    candidate = mod.verify_exact_radix_candidate(
        dims=dims,
        deltas=deltas,
        current_rank=b"\x00",
        current_huff=bytes(150),
    )

    assert candidate["lossless_sidecar_equivalence"] is True
    assert candidate["runtime_supported_now"] is True
    assert candidate["candidate_dim_bytes"] == 361
    assert candidate["candidate_payload_bytes"] == 511
    assert candidate["byte_savings_if_runtime_format_lands"] == 14
    assert candidate["score_claim"] is False
    assert candidate["ready_for_exact_eval_dispatch"] is False


def test_fixed_meta_chunks_requires_format0b_sidecar_length() -> None:
    sidecar = bytes(mod.PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES)

    dim, rank, huff = mod.fixed_meta_chunks(sidecar)

    assert len(dim) == 375
    assert len(rank) == 1
    assert len(huff) == 150
    with pytest.raises(mod.ProbeError):
        mod.fixed_meta_chunks(sidecar[:-1])


def test_live_format0b_probe_if_artifact_available() -> None:
    if not mod.DEFAULT_SOURCE_ARCHIVE.exists():
        pytest.skip("format-0x0B artifact is not checked into the repository")

    payload = mod.build_probe(mod.DEFAULT_SOURCE_ARCHIVE)

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["source"]["format_id"] == "0x0B"
    assert payload["summary"]["current_sidecar_payload_bytes"] == 525
    assert (
        payload["candidates"]["exact_radix_runtime_format"][
            "lossless_sidecar_equivalence"
        ]
        is True
    )
    assert payload["decision"]["realized_runtime_supported_byte_savings"] == 14
    assert "format0c_runtime_not_implemented" not in payload["decision"]["dispatch_blockers"]
