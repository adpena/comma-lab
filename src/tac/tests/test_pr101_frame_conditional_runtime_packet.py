# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import lzma
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tac.codec.frame_conditional_bit_budget import build_frame_conditional_wire_contract
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
)
from tac.repo_io import json_text, sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr101_frame_conditional_runtime_packet.py"


def _load_tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_pr101_frame_conditional_runtime_packet",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_tiny_pr101_contract(monkeypatch: pytest.MonkeyPatch, tool: Any) -> None:
    monkeypatch.setattr(tool, "PR101_DECODER_BLOB_LEN", 4)
    monkeypatch.setattr(tool, "PR101_N_PAIRS", 2)
    monkeypatch.setattr(tool, "PR101_LATENT_DIM", 4)


def _encode_tiny_latent_blob(tool: Any, q_ordered: np.ndarray) -> bytes:
    mins = np.zeros(q_ordered.shape[1], dtype=np.float16)
    scales = np.ones(q_ordered.shape[1], dtype=np.float16)
    q_dim_first = q_ordered.T.copy()
    stored = q_dim_first.copy()
    diffs = (
        (q_dim_first[:, 1:].astype(np.int16) - q_dim_first[:, :-1].astype(np.int16))
        & 255
    ).astype(np.uint8)
    stored[:, 1:] = ((diffs.astype(np.int16) + 128) & 255).astype(np.uint8)
    raw = mins.tobytes() + scales.tobytes() + stored.tobytes()
    return lzma.compress(
        raw, format=lzma.FORMAT_RAW, filters=tool.PR101_LATENT_LZMA_FILTERS
    )


def _write_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)


def _write_runtime(path: Path, *, latent_blob_len: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    inflate = path / "inflate.sh"
    inflate.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$1"
DST="$2"
python "$HERE/inflate.py" "$SRC" "$DST"
""",
        encoding="utf-8",
    )
    inflate.chmod(0o755)
    (path / "inflate.py").write_text("print('unused in unit test')\n", encoding="utf-8")
    src = path / "src"
    src.mkdir()
    (src / "model.py").write_text(
        """class HNeRVDecoder:
    pass
""",
        encoding="utf-8",
    )
    (src / "codec.py").write_text(
        f"""import lzma

import numpy as np
import torch

DECODER_BLOB_LEN = 4
LATENT_BLOB_LEN = {latent_blob_len}
N_PAIRS = 2
LATENT_DIM = 4
BASE_CHANNELS = 1
EVAL_SIZE = (1, 1)
LATENT_DIM_ORDER = (0, 1, 2, 3)
LATENT_LZMA_FILTERS = [
    {{"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}}
]


def decode_decoder_compact(decoder_blob):
    return {{"decoder": torch.tensor([decoder_blob[0]], dtype=torch.float32)}}


def decode_latents_compact(latent_blob):
    raw = lzma.decompress(latent_blob, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    stored = np.frombuffer(raw[LATENT_DIM * 4:], dtype=np.uint8).reshape(LATENT_DIM, N_PAIRS)
    return torch.from_numpy(stored.T.copy().astype(np.float32))


def apply_latent_sidecar(latents, sidecar_blob):
    return latents


def parse_archive(archive_bytes):
    decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
    latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob or not latent_blob:
        raise ValueError("bad compact archive")
    meta = {{
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }}
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
""",
        encoding="utf-8",
    )
    return path


def _a5_manifest(path: Path, *, q_bits: np.ndarray, q_ordered: np.ndarray, source_payload: bytes) -> Path:
    wire = build_frame_conditional_wire_contract(
        q_bits,
        latent_dim=q_ordered.shape[1],
        q_pair_first=q_ordered,
    )
    payload = {
        "schema": "pr101_frame_conditional_bit_anchor.v1",
        "tool": "unit",
        "score_claim": False,
        "byte_proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "input_archive": "source/archive.zip",
        "input_archive_bytes": len(source_payload),
        "input_archive_sha256": sha256_bytes(source_payload),
        "n_pairs": int(q_ordered.shape[0]),
        "latent_dim": int(q_ordered.shape[1]),
        "total_bit_budget": 64.0,
        "rows": [
            {
                "eta": 4.0,
                "floor": 0.25,
                "cap": 3.0,
                "archive_delta_bytes": -1,
                "frame_conditional_wire_contract": wire,
            }
        ],
        "best_eta": 4.0,
        "best_archive_delta_bytes": -1,
        "frame_conditional_wire_contract_status": {
            "schema": "tac_frame_conditional_latent_wire.v1",
            "typed_sideinfo_wire_contract_landed": True,
            "decoder_helper_consumes_sideinfo_bytes": True,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }
    path.write_text(json_text(payload), encoding="utf-8")
    return path


def test_builds_a5_runtime_packet_and_consumption_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    q_ordered = np.array(
        [
            [255, 128, 17, 1],
            [64, 32, 16, 8],
        ],
        dtype=np.uint8,
    )
    q_bits = np.array([4, 8], dtype=np.uint8)
    latent_blob = _encode_tiny_latent_blob(tool, q_ordered)
    monkeypatch.setattr(tool, "PR101_LATENT_BLOB_LEN", len(latent_blob))
    source_payload = b"DECO" + latent_blob + b"SIDE"
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, source_payload)
    runtime = _write_runtime(tmp_path / "runtime", latent_blob_len=len(latent_blob))
    a5_manifest = _a5_manifest(
        tmp_path / "a5.json",
        q_bits=q_bits,
        q_ordered=q_ordered,
        source_payload=source_payload,
    )

    manifest = tool.build_frame_conditional_runtime_packet(
        a5_manifest_path=a5_manifest,
        source_archive=source_archive,
        source_runtime_dir=runtime,
        output_dir=tmp_path / "out",
        q_bits_override=q_bits,
        recorded_at_utc=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
    )

    assert manifest["schema"] == "pr101_frame_conditional_runtime_packet.v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["packet_closure"]["runtime_consumes_changed_archive_bytes"] is True
    assert manifest["packet_local_runtime_patch"]["parse_archive_consumes_q_bits_sideinfo"] is True
    assert (
        manifest["packet_local_runtime_patch"]["decode_latents_consumes_variable_width_payload"]
        is True
    )
    assert manifest["runtime_packet"]["inflate_patch"]["portable_python_fallback"] is True
    assert manifest["parser_section_gate"]["ready"] is True
    assert manifest["parser_section_manifest"]["section_names"] == [
        "a5fc_magic",
        "decoder_len_u32le",
        "latent_meta_len_u32le",
        "q_bits_sideinfo_len_u32le",
        "latent_wire_len_u32le",
        "decoder_blob",
        "latent_min_scale_fp16",
        "q_bits_sideinfo_3bit",
        "latent_wire_variable_width",
        "sidecar_blob",
    ]

    archive_path = REPO_ROOT / manifest["candidate_archive_relpath"]
    inflate_sh = archive_path.parent / "inflate.sh"
    assert '"${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"' in inflate_sh.read_text(
        encoding="utf-8"
    )
    with zipfile.ZipFile(archive_path) as zf:
        candidate_payload = zf.read("x")
    assert candidate_payload.startswith(b"A5FC")
    assert manifest["candidate_archive"]["members"][0]["sha256"] == sha256_bytes(
        candidate_payload
    )

    proof_path = REPO_ROOT / manifest["runtime_consumption_proof"]["path"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["ready_for_exact_eval_runtime"] is True
    assert proof["blockers"] == []
    assert proof["archive_bound_candidate_contract_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    )
    proof_contract = proof["archive_bound_candidate_contract"]
    assert proof_contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert proof_contract["candidate_archive"]["sha256"] == (
        manifest["candidate_archive"]["sha256"]
    )
    assert proof_contract["archive_bound_candidate_ready"] is True
    assert proof_contract["ready_for_exact_eval_dispatch"] is False
    assert proof_contract["score_claim"] is False
    assert manifest["archive_bound_candidate_contract_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    )
    manifest_contract = manifest["archive_bound_candidate_contract"]
    assert manifest_contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert manifest_contract["candidate_archive"]["sha256"] == (
        manifest["candidate_archive"]["sha256"]
    )
    assert manifest_contract["archive_bound_candidate_ready"] is True
    assert manifest_contract["ready_for_exact_eval_dispatch"] is False
    assert manifest_contract["score_claim"] is False
    candidate_smoke = proof["candidate_packet_local_parse_smoke"]
    assert candidate_smoke["zero_sideinfo_negative_control_rejected"] is True
    assert candidate_smoke["latent_wire_mutation_changed_latents"] is True


def test_rejects_materialized_wire_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_pr101_contract(monkeypatch, tool)
    q_ordered = np.array([[255, 128, 17, 1], [64, 32, 16, 8]], dtype=np.uint8)
    q_bits = np.array([4, 8], dtype=np.uint8)
    latent_blob = _encode_tiny_latent_blob(tool, q_ordered)
    monkeypatch.setattr(tool, "PR101_LATENT_BLOB_LEN", len(latent_blob))
    source_payload = b"DECO" + latent_blob + b"SIDE"
    source_archive = tmp_path / "source" / "archive.zip"
    _write_zip(source_archive, source_payload)
    runtime = _write_runtime(tmp_path / "runtime", latent_blob_len=len(latent_blob))
    a5_manifest = _a5_manifest(
        tmp_path / "a5.json",
        q_bits=q_bits,
        q_ordered=q_ordered,
        source_payload=source_payload,
    )
    payload = json.loads(a5_manifest.read_text(encoding="utf-8"))
    payload["rows"][0]["frame_conditional_wire_contract"]["latent_wire_payload"][
        "sha256"
    ] = "0" * 64
    a5_manifest.write_text(json_text(payload), encoding="utf-8")

    with pytest.raises(
        tool.FrameConditionalRuntimePacketError,
        match=r"latent_wire_payload\.sha256 mismatch",
    ):
        tool.build_frame_conditional_runtime_packet(
            a5_manifest_path=a5_manifest,
            source_archive=source_archive,
            source_runtime_dir=runtime,
            output_dir=tmp_path / "out",
            q_bits_override=q_bits,
            recorded_at_utc=datetime(2026, 5, 8, 12, 0, tzinfo=UTC),
        )
