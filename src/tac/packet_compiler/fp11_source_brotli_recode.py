# SPDX-License-Identifier: MIT
"""Recode FP11 source split-Brotli decoder streams with runtime proof.

This materializer operates before the outer ZIP entropy position and inside the
FP11 source payload.  It preserves the decompressed decoder state exactly, then
patches the source runtime's fixed decoder length so latent/sidecar parsing
continues at the right byte offset.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.entropy_position import classify_entropy_position
from tac.optimization.fec6_decoder_mutations import prepare_decoder_blob
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FEC8_MAGIC,
    FECA_MAGIC,
    join_fp11_member,
    split_fp11_member,
)
from tac.pr101_split_brotli_codec import pack_brotli_stream
from tac.repo_io import sha256_bytes, sha256_file, tree_sha256, write_json_artifact

FP11_SOURCE_BROTLI_RECODE_MANIFEST_SCHEMA = "fp11_source_brotli_recode_manifest.v1"
FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA = (
    "fp11_source_brotli_recode_runtime_consumption_proof.v1"
)
FP11_SOURCE_BROTLI_RECODE_TARGET_KIND = "fp11_source_brotli_recode_v1"
FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID = "fp11_source_brotli_recode_adapter"
FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_ID = (
    "fp11_source_brotli_recode_v1.receiver.v1"
)
FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND = (
    "source_runtime_native_fp11_source_brotli_recode"
)
DEFAULT_BROTLI_QUALITIES = tuple(range(1, 12))
DEFAULT_BROTLI_LGWINS: tuple[int | None, ...] = (None, *range(16, 25))
_DECODER_BLOB_LEN_RE = re.compile(r"(?m)^(DECODER_BLOB_LEN\s*=\s*)([0-9_]+)(\s*)$")


class Fp11SourceBrotliRecodeError(ValueError):
    """Raised when an FP11 source Brotli recode candidate is unsafe."""


@dataclass(frozen=True)
class _StreamChoice:
    stream_index: int
    source_bytes: int
    candidate_bytes: int
    source_sha256: str
    candidate_sha256: str
    selected_quality: int | None
    selected_lgwin: int | None
    saved_bytes: int
    kept_source_payload: bool
    attempts: int
    payload: bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "source_bytes": self.source_bytes,
            "candidate_bytes": self.candidate_bytes,
            "source_sha256": self.source_sha256,
            "candidate_sha256": self.candidate_sha256,
            "selected_quality": self.selected_quality,
            "selected_lgwin": self.selected_lgwin,
            "saved_bytes": self.saved_bytes,
            "kept_source_payload": self.kept_source_payload,
            "attempts": self.attempts,
        }


def _read_single_member(archive_path: Path) -> tuple[zipfile.ZipInfo, bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise Fp11SourceBrotliRecodeError(
                f"expected one archive member, found {len(infos)}"
            )
        info = infos[0]
        return info, archive.read(info.filename)


def _write_stored_archive(archive_path: Path, *, member_name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, payload)


def _archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_decoder_blob_len(text: str) -> int:
    match = _DECODER_BLOB_LEN_RE.search(text)
    if match is None:
        raise Fp11SourceBrotliRecodeError("DECODER_BLOB_LEN runtime constant not found")
    return int(match.group(2).replace("_", ""))


def patch_decoder_blob_len_text(text: str, *, new_len: int) -> str:
    if new_len < 1:
        raise Fp11SourceBrotliRecodeError("new decoder length must be positive")
    matches = list(_DECODER_BLOB_LEN_RE.finditer(text))
    if len(matches) != 1:
        raise Fp11SourceBrotliRecodeError(
            f"expected one DECODER_BLOB_LEN constant, found {len(matches)}"
        )
    return _DECODER_BLOB_LEN_RE.sub(rf"\g<1>{int(new_len)}\g<3>", text, count=1)


def _patch_decoder_blob_len(path: Path, *, expected_old_len: int, new_len: int) -> None:
    text = path.read_text(encoding="utf-8")
    old_len = parse_decoder_blob_len(text)
    if old_len != expected_old_len:
        raise Fp11SourceBrotliRecodeError(
            f"runtime DECODER_BLOB_LEN mismatch expected={expected_old_len} actual={old_len}"
        )
    path.write_text(patch_decoder_blob_len_text(text, new_len=new_len), encoding="utf-8")


def _patch_inflate_no_bytecode(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "PYTHONDONTWRITEBYTECODE" in text:
        return
    lines = text.splitlines(keepends=True)
    export_line = "export PYTHONDONTWRITEBYTECODE=1\n"
    if lines and lines[0].startswith("#!"):
        lines.insert(1, export_line)
        patched = "".join(lines)
    else:
        patched = export_line + text
    path.write_text(patched, encoding="utf-8")


def _load_full_frame_parity_proof(
    path: str | Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    proof_path = Path(path)
    if not proof_path.is_file():
        raise Fp11SourceBrotliRecodeError(
            f"full-frame inflate parity proof missing: {proof_path}"
        )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Fp11SourceBrotliRecodeError(
            f"full-frame inflate parity proof must be a JSON object: {proof_path}"
        )
    if payload.get("full_frame_inflate_output_parity_claim") is not True:
        raise Fp11SourceBrotliRecodeError(
            "full-frame inflate parity proof does not claim full-frame parity"
        )
    if payload.get("cmp_equal") is not True or payload.get("output_sha256_match") is not True:
        raise Fp11SourceBrotliRecodeError(
            "full-frame inflate parity proof is not byte-identical"
        )
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        raise Fp11SourceBrotliRecodeError(
            "full-frame inflate parity proof carries blockers: "
            + ",".join(str(item) for item in blockers)
        )
    return payload, proof_path


def _validated_qualities(values: Sequence[int]) -> tuple[int, ...]:
    out = tuple(dict.fromkeys(int(value) for value in values))
    if not out:
        raise Fp11SourceBrotliRecodeError("at least one Brotli quality is required")
    bad = [value for value in out if value < 0 or value > 11]
    if bad:
        raise Fp11SourceBrotliRecodeError(f"Brotli qualities out of range 0..11: {bad}")
    return out


def _validated_lgwins(values: Sequence[int | None]) -> tuple[int | None, ...]:
    out = tuple(dict.fromkeys(values))
    if not out:
        raise Fp11SourceBrotliRecodeError("at least one Brotli lgwin is required")
    bad = [value for value in out if value is not None and (value < 10 or value > 24)]
    if bad:
        raise Fp11SourceBrotliRecodeError(f"Brotli lgwins out of range 10..24: {bad}")
    return out


def _choose_stream_payload(
    *,
    stream_index: int,
    source_payload: bytes,
    raw_payload: bytes,
    qualities: Sequence[int],
    lgwins: Sequence[int | None],
) -> _StreamChoice:
    best_payload = source_payload
    best_quality: int | None = None
    best_lgwin: int | None = None
    attempts = 0
    for quality in qualities:
        for lgwin in lgwins:
            attempts += 1
            payload = pack_brotli_stream(raw_payload, quality=int(quality), lgwin=lgwin)
            if (len(payload), payload) < (len(best_payload), best_payload):
                best_payload = payload
                best_quality = int(quality)
                best_lgwin = lgwin
    return _StreamChoice(
        stream_index=stream_index,
        source_bytes=len(source_payload),
        candidate_bytes=len(best_payload),
        source_sha256=sha256_bytes(source_payload),
        candidate_sha256=sha256_bytes(best_payload),
        selected_quality=best_quality,
        selected_lgwin=best_lgwin,
        saved_bytes=len(source_payload) - len(best_payload),
        kept_source_payload=best_payload == source_payload,
        attempts=attempts,
        payload=best_payload,
    )


def build_fp11_source_brotli_recode_candidate(
    *,
    source_submission_dir: str | Path,
    output_dir: str | Path,
    qualities: Sequence[int] = DEFAULT_BROTLI_QUALITIES,
    lgwins: Sequence[int | None] = DEFAULT_BROTLI_LGWINS,
    full_frame_inflate_parity_proof: str | Path | None = None,
    allow_nonpositive_candidate: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Build a byte-closed FP11 split-Brotli recode candidate."""

    source_dir = Path(source_submission_dir)
    output = Path(output_dir)
    if not source_dir.is_dir():
        raise Fp11SourceBrotliRecodeError(f"missing source submission dir: {source_dir}")
    source_archive = source_dir / "archive.zip"
    runtime_codec = source_dir / "src" / "codec.py"
    if not source_archive.is_file():
        raise Fp11SourceBrotliRecodeError(f"missing source archive: {source_archive}")
    if not runtime_codec.is_file():
        raise Fp11SourceBrotliRecodeError(f"missing source runtime codec: {runtime_codec}")
    qualities = _validated_qualities(qualities)
    lgwins = _validated_lgwins(lgwins)
    parity_payload, parity_source_path = _load_full_frame_parity_proof(
        full_frame_inflate_parity_proof
    )
    if output.exists():
        if not allow_overwrite:
            raise Fp11SourceBrotliRecodeError(f"output exists: {output}")
        shutil.rmtree(output)

    candidate_dir = output / "submission_dir"
    shutil.copytree(
        source_dir,
        candidate_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _patch_inflate_no_bytecode(candidate_dir / "inflate.sh")

    info, member_payload = _read_single_member(source_archive)
    parts = split_fp11_member(
        member_payload,
        allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC),
    )
    decoder_len = parse_decoder_blob_len(runtime_codec.read_text(encoding="utf-8"))
    source_payload = parts["source_payload"]
    if decoder_len >= len(source_payload):
        raise Fp11SourceBrotliRecodeError(
            "runtime decoder length does not leave latent/sidecar source tail"
        )
    decoder_blob = source_payload[:decoder_len]
    source_tail = source_payload[decoder_len:]
    prepared = prepare_decoder_blob(decoder_blob)

    choices: list[_StreamChoice] = []
    for span in prepared.stream_spans:
        raw = prepared.raw[span.raw_range.start : span.raw_range.end]
        compressed = decoder_blob[
            span.compressed_range.start : span.compressed_range.end
        ]
        choices.append(
            _choose_stream_payload(
                stream_index=int(span.stream_index),
                source_payload=compressed,
                raw_payload=raw,
                qualities=qualities,
                lgwins=lgwins,
            )
        )
    candidate_decoder = b"".join(choice.payload for choice in choices)
    candidate_prepared = prepare_decoder_blob(candidate_decoder)
    if candidate_prepared.raw != prepared.raw:
        raise Fp11SourceBrotliRecodeError("candidate decoder raw bytes changed")
    decoder_saved_bytes = len(decoder_blob) - len(candidate_decoder)
    if decoder_saved_bytes <= 0 and not allow_nonpositive_candidate:
        raise Fp11SourceBrotliRecodeError("no rate-positive split-Brotli recode found")

    candidate_source_payload = candidate_decoder + source_tail
    candidate_member = join_fp11_member(
        source_payload=candidate_source_payload,
        selector_payload=parts["selector_payload"],
        dqs1_tail=parts["dqs1_tail"],
    )
    candidate_archive = candidate_dir / "archive.zip"
    _write_stored_archive(candidate_archive, member_name=info.filename, payload=candidate_member)
    _patch_decoder_blob_len(
        candidate_dir / "src" / "codec.py",
        expected_old_len=decoder_len,
        new_len=len(candidate_decoder),
    )

    candidate_parts = split_fp11_member(
        _read_single_member(candidate_archive)[1],
        allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC),
    )
    if candidate_parts["selector_payload"] != parts["selector_payload"]:
        raise Fp11SourceBrotliRecodeError("candidate changed selector payload bytes")
    if candidate_parts["dqs1_tail"] != parts["dqs1_tail"]:
        raise Fp11SourceBrotliRecodeError("candidate changed DQS1 tail bytes")
    if candidate_parts["source_payload"][len(candidate_decoder) :] != source_tail:
        raise Fp11SourceBrotliRecodeError("candidate changed source tail bytes")
    if prepare_decoder_blob(candidate_parts["source_payload"][: len(candidate_decoder)]).raw != prepared.raw:
        raise Fp11SourceBrotliRecodeError("candidate archive decoder raw bytes changed")

    source_archive_record = _archive_record(source_archive)
    candidate_archive_record = _archive_record(candidate_archive)
    archive_saved_bytes = source_archive_record["bytes"] - candidate_archive_record["bytes"]
    serialized_archive_delta = build_serialized_archive_delta_contract(
        source_archive=source_archive_record,
        candidate_archive=candidate_archive_record,
        modeled_saved_bytes=archive_saved_bytes,
        require_realized_saving=False,
    )
    if parity_payload is not None:
        right = parity_payload.get("right") if isinstance(parity_payload.get("right"), dict) else {}
        left = parity_payload.get("left") if isinstance(parity_payload.get("left"), dict) else {}
        if right.get("archive_sha256") != candidate_archive_record["sha256"]:
            raise Fp11SourceBrotliRecodeError(
                "full-frame parity proof right archive SHA does not match candidate"
            )
        if left.get("archive_sha256") != source_archive_record["sha256"]:
            raise Fp11SourceBrotliRecodeError(
                "full-frame parity proof left archive SHA does not match source"
            )

    candidate_runtime_tree_sha = tree_sha256(candidate_dir)
    inflate_parity_satisfied = parity_payload is not None
    readiness_blockers = [
        *(
            []
            if inflate_parity_satisfied
            else ["candidate_requires_full_frame_inflate_output_parity_before_score_claim"]
        ),
        "candidate_requires_exact_auth_eval_before_promotion",
    ]
    entropy_position = classify_entropy_position(
        FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
        operation_family="source_brotli_recode",
        payload_context={"entropy_position_id": "P5/P8/P15-cascade"},
    )
    proof_path = output / "fp11_source_brotli_recode_runtime_consumption_proof.json"
    parity_copy_path = output / "fp11_source_brotli_recode_full_frame_inflate_parity_proof.json"
    parity_copy_record: dict[str, Any] | None = None
    if parity_payload is not None:
        write_json_artifact(parity_copy_path, parity_payload)
        parity_copy_record = _archive_record(parity_copy_path)

    stream_rows = [choice.as_dict() for choice in choices]
    selected_payload = {
        "source_archive_bytes": source_archive_record["bytes"],
        "candidate_archive_bytes": candidate_archive_record["bytes"],
        "saved_bytes": archive_saved_bytes,
        "source_payload_bytes": len(decoder_blob),
        "candidate_payload_bytes": len(candidate_decoder),
        "payload_saved_bytes": decoder_saved_bytes,
        "status": "realized_saving" if archive_saved_bytes > 0 else "zero_delta",
        "savings_realized": archive_saved_bytes > 0,
    }
    common = {
        "target_kind": FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
        "materializer_id": FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID,
        "operation_family": "source_brotli_recode",
        "entropy_position_id": "P5/P8/P15-cascade",
        "entropy_position_classification": entropy_position,
        "receiver_contract_id": FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_ID,
        "receiver_contract_kind": FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND,
        "source_archive": source_archive_record,
        "candidate_archive": candidate_archive_record,
        "candidate_runtime_dir": str(candidate_dir),
        "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
        "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
        "runtime_adapter_ready": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "receiver_contract_satisfied": True,
        "source_member": {
            "name": info.filename,
            "bytes": len(member_payload),
            "sha256": sha256_bytes(member_payload),
        },
        "candidate_member": {
            "name": info.filename,
            "bytes": len(candidate_member),
            "sha256": sha256_bytes(candidate_member),
        },
        "selected_member_name": info.filename,
        "selected_member_names": [info.filename],
        "source_decoder_bytes": len(decoder_blob),
        "candidate_decoder_bytes": len(candidate_decoder),
        "decoder_saved_bytes": decoder_saved_bytes,
        "archive_saved_bytes": archive_saved_bytes,
        "decoder_raw_sha256": prepared.raw_sha256,
        "candidate_decoder_raw_sha256": candidate_prepared.raw_sha256,
        "decoder_raw_roundtrip_equal": True,
        "source_tail_unchanged": True,
        "selector_payload_unchanged": True,
        "dqs1_tail_unchanged": True,
        "source_payload_len_before": len(source_payload),
        "source_payload_len_after": len(candidate_source_payload),
        "runtime_decoder_blob_len_before": decoder_len,
        "runtime_decoder_blob_len_after": len(candidate_decoder),
        "requested_brotli_qualities": list(qualities),
        "requested_brotli_lgwins": [value if value is not None else "default" for value in lgwins],
        "stream_recode_rows": stream_rows,
        "selected_payload": selected_payload,
        "serialized_archive_delta": serialized_archive_delta,
    }
    proof = apply_proxy_evidence_boundary(
        {
            "schema": FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **common,
            "runtime_consumption_proof_passed": True,
            "passed": True,
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(
        proof,
        context="fp11_source_brotli_recode_runtime_consumption_proof",
    )
    write_json_artifact(proof_path, proof)
    proof_record = _archive_record(proof_path)

    manifest = apply_proxy_evidence_boundary(
        {
            "schema": FP11_SOURCE_BROTLI_RECODE_MANIFEST_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **common,
            "selection_scope": "fp11_source_split_brotli_decoder_streams",
            "portability_contract": "source_runtime_native_python_packet_decoder",
            "source_submission_dir": str(source_dir),
            "candidate_submission_dir": str(candidate_dir),
            "source_archive_path": str(source_archive),
            "source_archive_bytes": source_archive_record["bytes"],
            "source_archive_sha256": source_archive_record["sha256"],
            "candidate_archive_path": str(candidate_archive),
            "candidate_archive_bytes": candidate_archive_record["bytes"],
            "candidate_archive_sha256": candidate_archive_record["sha256"],
            "member_name": info.filename,
            "byte_closed_candidate_emitted": True,
            "receiver_proof_ready": True,
            "inflate_parity_satisfied": inflate_parity_satisfied,
            "full_frame_inflate_parity_proven": inflate_parity_satisfied,
            "full_frame_inflate_parity_source_path": (
                str(parity_source_path) if parity_source_path is not None else None
            ),
            "full_frame_inflate_parity_proof_path": (
                str(parity_copy_path) if parity_copy_record is not None else None
            ),
            "full_frame_inflate_parity_proof_bytes": (
                parity_copy_record["bytes"] if parity_copy_record is not None else None
            ),
            "full_frame_inflate_parity_proof_sha256": (
                parity_copy_record["sha256"] if parity_copy_record is not None else None
            ),
            "runtime_consumption_proof_path": str(proof_path),
            "runtime_consumption_proof_bytes": proof_record["bytes"],
            "runtime_consumption_proof_sha256": proof_record["sha256"],
            "receiver_verification": {
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "candidate_runtime_dir": str(candidate_dir),
                "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
                "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
                "proof_path": str(proof_path),
                "blockers": [],
            },
            "readiness_blockers": readiness_blockers,
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(manifest, context="fp11_source_brotli_recode")
    write_json_artifact(output / "fp11_source_brotli_recode_manifest.json", manifest)
    return manifest


__all__ = [
    "DEFAULT_BROTLI_LGWINS",
    "DEFAULT_BROTLI_QUALITIES",
    "FP11_SOURCE_BROTLI_RECODE_MANIFEST_SCHEMA",
    "FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID",
    "FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA",
    "FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_ID",
    "FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND",
    "FP11_SOURCE_BROTLI_RECODE_TARGET_KIND",
    "Fp11SourceBrotliRecodeError",
    "build_fp11_source_brotli_recode_candidate",
    "parse_decoder_blob_len",
    "patch_decoder_blob_len_text",
]
