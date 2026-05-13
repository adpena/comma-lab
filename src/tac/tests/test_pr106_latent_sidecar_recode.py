from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_NO_OP_DIM,
    PR106SidecarPacket,
    canonicalize_brotli_dim_delta_sidecar_arrays,
    decode_brotli_dim_delta_sidecar_payload,
    decode_pr101_ranked_sidecar_payload_to_dim_delta,
    emit_pr106_sidecar_packet,
    encode_brotli_dim_delta_sidecar_payload,
    encode_pr101_ranked_sidecar_payload,
    lossless_pr106_sidecar_recode_candidates,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "profile_pr106_latent_sidecar_recode.py"


def _sample_arrays(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    dims = np.array([0, 1, 2, 3, 4, 5, 6, PR106_NO_OP_DIM, 7, 8, 9, 10], dtype=np.uint8)[
        :n
    ]
    deltas = np.array([-2, -1, 1, 2, -2, 1, -1, 0, 2, -1, 1, -2], dtype=np.int8)[:n]
    return dims, deltas


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("profile_pr106_latent_sidecar_recode", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_sidecar_encoder_maps_zero_delta_to_noop() -> None:
    dims = np.array([0, 5, 9], dtype=np.uint8)
    deltas = np.array([1, 0, -1], dtype=np.int8)

    payload = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    got_dims, got_deltas = decode_brotli_dim_delta_sidecar_payload(payload)

    assert got_dims.tolist() == [0, PR106_NO_OP_DIM, 9]
    assert got_deltas.tolist() == [1, 0, -1]


def test_canonical_sidecar_rejects_noop_with_nonzero_delta() -> None:
    with pytest.raises(ValueError, match="no-op dim"):
        canonicalize_brotli_dim_delta_sidecar_arrays(
            np.array([PR106_NO_OP_DIM], dtype=np.uint8),
            np.array([1], dtype=np.int8),
        )


def test_lossless_recode_candidates_roundtrip_semantics() -> None:
    dims, deltas = _sample_arrays()
    source_dims, source_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(dims, deltas)

    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    names = {candidate.name for candidate in candidates}

    assert "current_pr100_dim_delta_brotli_q11" in names
    assert "vocab_bitpack_dim_delta_raw" in names
    assert "split_dim_stream_delta_stream_brotli_q11" in names
    for candidate in candidates:
        if not candidate.encoded_bytes:
            continue
        np.testing.assert_array_equal(candidate.decoded_dims, source_dims)
        np.testing.assert_array_equal(candidate.decoded_delta_q, source_deltas)
        assert candidate.charged_bytes > 0


def test_pr101_ranked_sidecar_candidate_roundtrips_600_pair_payload() -> None:
    dims = np.arange(600, dtype=np.uint16) % 28
    deltas = np.resize(np.array([-2, -1, 1, 2], dtype=np.int8), 600)
    payload, framing_meta = encode_pr101_ranked_sidecar_payload(
        dims.astype(np.uint8),
        deltas,
    )

    got_dims, got_deltas = decode_pr101_ranked_sidecar_payload_to_dim_delta(
        payload,
        framing_meta,
    )

    np.testing.assert_array_equal(got_dims, dims.astype(np.uint8))
    np.testing.assert_array_equal(got_deltas, deltas)
    assert len(framing_meta) == 6


def test_recode_profile_tool_writes_nonpromotable_report(tmp_path: Path) -> None:
    dims, deltas = _sample_arrays()
    sidecar = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    sidecar_path = tmp_path / "sidecar.bin"
    sidecar_path.write_bytes(sidecar)
    out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--sidecar-bin",
            str(sidecar_path),
            "--json-out",
            str(out),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr106_latent_sidecar_recode_profile_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["semantic_arrays"]["n_pairs"] == len(dims)
    assert report["best_lossless_candidate"]["lossless_semantic_equivalence_proven"] is True


def test_recode_profile_tool_reads_sidecar_archive(tmp_path: Path) -> None:
    dims, deltas = _sample_arrays()
    sidecar = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    payload = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=0x01,
            pr106_bytes=b"\xfffixture-pr106-payload",
            sidecar_payload=sidecar,
        )
    )
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)
    tool = _load_tool_module()

    report = tool.build_report(
        tool.parse_args(
            [
                "--sidecar-archive",
                str(archive),
                "--json-out",
                str(tmp_path / "unused.json"),
            ]
        )
    )

    assert report["source"]["mode"] == "sidecar_archive"
    assert report["source"]["pr106_inner_payload_bytes"] == len(b"\xfffixture-pr106-payload")
    assert report["score_claim"] is False
