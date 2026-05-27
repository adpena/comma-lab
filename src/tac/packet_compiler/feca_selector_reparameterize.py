# SPDX-License-Identifier: MIT
"""Build exact-roundtrip FECa selector reparameterization candidates.

The current PR110-family packet stores a FECa/FEC10 arithmetic-coded selector
inside the FP11 member.  This module only changes the selector codec parameters
and the matching runtime constant, then proves the decoded selector codes,
source payload, and DQS1 tail are unchanged.  It is a rate-only packet
transform; it does not claim score authority.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
import time
import zipfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.repo_io import sha256_bytes, sha256_file, tree_sha256, write_json_artifact

FECA_REPARAMETERIZATION_MANIFEST_SCHEMA = "feca_selector_reparameterization_manifest.v1"
FECA_REPARAMETERIZATION_PROOF_SCHEMA = "feca_selector_runtime_consumption_proof.v1"
FECA_SELECTOR_TARGET_KIND = "selector_stream_context_recode_v1"
FECA_SELECTOR_MATERIALIZER_ID = "feca_selector_reparameterize_adapter"
FECA_SELECTOR_RECEIVER_CONTRACT_ID = "selector_stream_context_recode_v1.receiver.v1"
FECA_SELECTOR_RECEIVER_CONTRACT_KIND = "source_runtime_native_feca_selector_recode"
FP11_MAGIC = b"FP11"
FECA_MAGIC = b"FECa"


class FecaSelectorReparameterizationError(ValueError):
    """Raised when a FECa selector reparameterization candidate is unsafe."""


@contextmanager
def _prepended_sys_path(path: Path):
    value = str(path)
    inserted = False
    if value not in sys.path:
        sys.path.insert(0, value)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _load_feca_module(encoder_dir: Path, *, module_suffix: str) -> ModuleType:
    module_path = encoder_dir / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py"
    if not module_path.is_file():
        raise FecaSelectorReparameterizationError(f"missing FECa encoder module: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"feca_selector_reparameterize_{module_suffix}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise FecaSelectorReparameterizationError(f"could not load FECa module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _prepended_sys_path(encoder_dir):
        spec.loader.exec_module(module)
    return module


def _read_single_member(archive_path: Path) -> tuple[zipfile.ZipInfo, bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise FecaSelectorReparameterizationError(
                f"expected one archive member, found {len(infos)}"
            )
        info = infos[0]
        return info, archive.read(info.filename)


def split_fp11_member(member_payload: bytes) -> dict[str, Any]:
    if len(member_payload) < 10 or member_payload[:4] != FP11_MAGIC:
        raise FecaSelectorReparameterizationError("archive member is not an FP11 packet")
    source_len = struct.unpack_from("<I", member_payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(member_payload):
        raise FecaSelectorReparameterizationError("FP11 packet truncated before selector length")
    selector_len = struct.unpack_from("<H", member_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end > len(member_payload):
        raise FecaSelectorReparameterizationError("FP11 selector payload is truncated")
    selector_payload = member_payload[selector_start:selector_end]
    if not selector_payload.startswith(FECA_MAGIC):
        raise FecaSelectorReparameterizationError(
            f"expected FECa selector payload, got {selector_payload[:4]!r}"
        )
    return {
        "source_payload": member_payload[8:selector_len_offset],
        "selector_payload": selector_payload,
        "dqs1_tail": member_payload[selector_end:],
        "source_len": source_len,
        "selector_len": selector_len,
    }


def join_fp11_member(*, source_payload: bytes, selector_payload: bytes, dqs1_tail: bytes) -> bytes:
    if len(source_payload) > 0xFFFFFFFF:
        raise FecaSelectorReparameterizationError("source payload too large for FP11")
    if len(selector_payload) > 0xFFFF:
        raise FecaSelectorReparameterizationError("selector payload too large for FP11")
    return (
        FP11_MAGIC
        + struct.pack("<I", len(source_payload))
        + source_payload
        + struct.pack("<H", len(selector_payload))
        + selector_payload
        + dqs1_tail
    )


def _encode_with_params(
    module: ModuleType,
    codes: Sequence[int],
    *,
    scale: int,
    alpha: int,
) -> bytes:
    original = (
        module.ALPHA_DEFAULT,
        module._BlendContextModel.SCALE,
        module._PRIOR_MODEL,
        module._CTX1_MODELS,
        module._CTX_BLEND_MODELS,
        module._CTX2_ROW_SUMS,
    )
    try:
        module.ALPHA_DEFAULT = int(alpha)
        module._BlendContextModel.SCALE = int(scale)
        (
            module._PRIOR_MODEL,
            module._CTX1_MODELS,
            module._CTX_BLEND_MODELS,
            module._CTX2_ROW_SUMS,
        ) = module._build_priors()
        payload = module.encode_fec10_hybrid_adaptive_blend(codes, n_pairs=len(codes))
        decoded = module.decode_fec10_hybrid_selector(payload)
        if list(decoded) != list(codes):
            raise FecaSelectorReparameterizationError(
                f"FECa scale={scale} alpha={alpha} failed selector-code roundtrip"
            )
        return payload
    finally:
        (
            module.ALPHA_DEFAULT,
            module._BlendContextModel.SCALE,
            module._PRIOR_MODEL,
            module._CTX1_MODELS,
            module._CTX_BLEND_MODELS,
            module._CTX2_ROW_SUMS,
        ) = original


def _patch_runtime_module(path: Path, *, scale: int, alpha: int) -> None:
    text = path.read_text(encoding="utf-8")
    if "ALPHA_DEFAULT = 2" not in text:
        raise FecaSelectorReparameterizationError("FECa runtime alpha constant not found")
    if "SCALE = 1 << 14" not in text:
        raise FecaSelectorReparameterizationError("FECa runtime scale constant not found")
    text = text.replace("ALPHA_DEFAULT = 2", f"ALPHA_DEFAULT = {int(alpha)}", 1)
    text = text.replace("SCALE = 1 << 14", f"SCALE = {int(scale)}", 1)
    path.write_text(text, encoding="utf-8")


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


def _load_full_frame_parity_proof(path: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    proof_path = Path(path)
    if not proof_path.is_file():
        raise FecaSelectorReparameterizationError(
            f"full-frame inflate parity proof missing: {proof_path}"
        )
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FecaSelectorReparameterizationError(
            f"full-frame inflate parity proof must be a JSON object: {proof_path}"
        )
    if payload.get("full_frame_inflate_output_parity_claim") is not True:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof does not claim full-frame parity"
        )
    if payload.get("cmp_equal") is not True or payload.get("output_sha256_match") is not True:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof is not byte-identical"
        )
    blockers = payload.get("blockers")
    if isinstance(blockers, list) and blockers:
        raise FecaSelectorReparameterizationError(
            "full-frame inflate parity proof carries blockers: "
            + ",".join(str(item) for item in blockers)
        )
    return payload, proof_path


def build_feca_selector_reparameterized_candidate(
    *,
    source_submission_dir: str | Path,
    output_dir: str | Path,
    scales: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    alphas: Sequence[int] = tuple(range(1, 17)),
    full_frame_inflate_parity_proof: str | Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_submission_dir)
    output = Path(output_dir)
    if not source_dir.is_dir():
        raise FecaSelectorReparameterizationError(f"missing source submission dir: {source_dir}")
    parity_payload, parity_source_path = _load_full_frame_parity_proof(
        full_frame_inflate_parity_proof
    )
    if output.exists():
        if not allow_overwrite:
            raise FecaSelectorReparameterizationError(f"output exists: {output}")
        shutil.rmtree(output)
    candidate_dir = output / "submission_dir"
    shutil.copytree(source_dir, candidate_dir)

    source_archive = source_dir / "archive.zip"
    candidate_archive = candidate_dir / "archive.zip"
    info, member_payload = _read_single_member(source_archive)
    parts = split_fp11_member(member_payload)
    source_module = _load_feca_module(source_dir / "encoder", module_suffix="source")
    source_codes = source_module.decode_fec10_hybrid_selector(parts["selector_payload"])

    rows: list[dict[str, Any]] = []
    best_payload = parts["selector_payload"]
    best_scale = int(source_module._BlendContextModel.SCALE)
    best_alpha = int(source_module.ALPHA_DEFAULT)
    for scale in scales:
        for alpha in alphas:
            try:
                payload = _encode_with_params(
                    source_module,
                    source_codes,
                    scale=int(scale),
                    alpha=int(alpha),
                )
            except FecaSelectorReparameterizationError as exc:
                rows.append(
                    {
                        "scale": int(scale),
                        "alpha": int(alpha),
                        "status": "roundtrip_failed",
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "scale": int(scale),
                    "alpha": int(alpha),
                    "status": "roundtrip_equal",
                    "selector_payload_bytes": len(payload),
                    "selector_payload_sha256": sha256_bytes(payload),
                    "selector_saved_bytes": len(parts["selector_payload"]) - len(payload),
                }
            )
            if (len(payload), int(scale), int(alpha)) < (len(best_payload), best_scale, best_alpha):
                best_payload = payload
                best_scale = int(scale)
                best_alpha = int(alpha)

    if len(best_payload) >= len(parts["selector_payload"]):
        raise FecaSelectorReparameterizationError("no rate-positive FECa reparameterization found")

    _patch_runtime_module(
        candidate_dir / "encoder" / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py",
        scale=best_scale,
        alpha=best_alpha,
    )
    candidate_member = join_fp11_member(
        source_payload=parts["source_payload"],
        selector_payload=best_payload,
        dqs1_tail=parts["dqs1_tail"],
    )
    _write_stored_archive(candidate_archive, member_name=info.filename, payload=candidate_member)

    candidate_module = _load_feca_module(candidate_dir / "encoder", module_suffix="candidate")
    candidate_parts = split_fp11_member(_read_single_member(candidate_archive)[1])
    candidate_codes = candidate_module.decode_fec10_hybrid_selector(
        candidate_parts["selector_payload"]
    )
    if list(candidate_codes) != list(source_codes):
        raise FecaSelectorReparameterizationError("candidate runtime selector decode changed codes")
    if candidate_parts["source_payload"] != parts["source_payload"]:
        raise FecaSelectorReparameterizationError("candidate changed source payload bytes")
    if candidate_parts["dqs1_tail"] != parts["dqs1_tail"]:
        raise FecaSelectorReparameterizationError("candidate changed DQS1 tail bytes")

    archive_saved_bytes = source_archive.stat().st_size - candidate_archive.stat().st_size
    source_archive_record = _archive_record(source_archive)
    candidate_archive_record = _archive_record(candidate_archive)
    if parity_payload is not None:
        right = parity_payload.get("right") if isinstance(parity_payload.get("right"), dict) else {}
        left = parity_payload.get("left") if isinstance(parity_payload.get("left"), dict) else {}
        if right.get("archive_sha256") != candidate_archive_record["sha256"]:
            raise FecaSelectorReparameterizationError(
                "full-frame parity proof right archive SHA does not match candidate"
            )
        if left.get("archive_sha256") != source_archive_record["sha256"]:
            raise FecaSelectorReparameterizationError(
                "full-frame parity proof left archive SHA does not match source"
            )
    candidate_runtime_tree_sha = tree_sha256(candidate_dir)
    proof_path = output / "feca_selector_runtime_consumption_proof.json"
    parity_copy_path = output / "feca_selector_full_frame_inflate_parity_proof.json"
    parity_copy_record: dict[str, Any] | None = None
    if parity_payload is not None:
        write_json_artifact(parity_copy_path, parity_payload)
        parity_copy_record = _archive_record(parity_copy_path)
    inflate_parity_satisfied = parity_payload is not None
    readiness_blockers = [
        *(
            []
            if inflate_parity_satisfied
            else ["candidate_requires_full_frame_inflate_output_parity_before_score_claim"]
        ),
        "candidate_requires_exact_auth_eval_before_promotion",
    ]
    proof = apply_proxy_evidence_boundary(
        {
            "schema": FECA_REPARAMETERIZATION_PROOF_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_kind": FECA_SELECTOR_TARGET_KIND,
            "materializer_id": FECA_SELECTOR_MATERIALIZER_ID,
            "receiver_contract_id": FECA_SELECTOR_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": FECA_SELECTOR_RECEIVER_CONTRACT_KIND,
            "source_archive": source_archive_record,
            "candidate_archive": candidate_archive_record,
            "candidate_runtime_dir": str(candidate_dir),
            "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
            "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
            "runtime_adapter_ready": True,
            "candidate_runtime_adapter_blocker_cleared": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "receiver_contract_satisfied": True,
            "selector_code_roundtrip_equal": True,
            "source_payload_unchanged": True,
            "dqs1_tail_unchanged": True,
            "selected_payload": {
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
                "source_payload_bytes": len(parts["selector_payload"]),
                "candidate_payload_bytes": len(best_payload),
                "payload_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
                "status": "realized_saving",
                "savings_realized": archive_saved_bytes > 0,
            },
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(proof, context="feca_selector_runtime_consumption_proof")
    write_json_artifact(proof_path, proof)
    proof_record = _archive_record(proof_path)
    manifest = apply_proxy_evidence_boundary(
        {
            "schema": FECA_REPARAMETERIZATION_MANIFEST_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_kind": FECA_SELECTOR_TARGET_KIND,
            "materializer_id": FECA_SELECTOR_MATERIALIZER_ID,
            "operation_family": "selector_context_recode",
            "entropy_position_id": "P11",
            "selection_scope": "fp11_feca_selector_stream",
            "receiver_contract_id": FECA_SELECTOR_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": FECA_SELECTOR_RECEIVER_CONTRACT_KIND,
            "portability_contract": "source_runtime_native_python_packet_decoder",
            "source_submission_dir": str(source_dir),
            "candidate_submission_dir": str(candidate_dir),
            "candidate_runtime_dir": str(candidate_dir),
            "candidate_runtime_tree_sha256": candidate_runtime_tree_sha,
            "expected_runtime_tree_sha256": candidate_runtime_tree_sha,
            "runtime_adapter_ready": True,
            "source_archive_path": str(source_archive),
            "source_archive": source_archive_record,
            "source_archive_bytes": source_archive_record["bytes"],
            "source_archive_sha256": source_archive_record["sha256"],
            "candidate_archive_path": str(candidate_archive),
            "candidate_archive": candidate_archive_record,
            "candidate_archive_bytes": candidate_archive_record["bytes"],
            "candidate_archive_sha256": candidate_archive_record["sha256"],
            "member_name": info.filename,
            "selected_member_name": info.filename,
            "selected_member_names": [info.filename],
            "source_member_bytes": len(member_payload),
            "candidate_member_bytes": len(candidate_member),
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
            "selector_source_bytes": len(parts["selector_payload"]),
            "selector_candidate_bytes": len(best_payload),
            "selector_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
            "archive_saved_bytes": archive_saved_bytes,
            "selected_scale": best_scale,
            "selected_alpha": best_alpha,
            "selector_code_count": len(source_codes),
            "selector_code_roundtrip_equal": True,
            "source_payload_unchanged": True,
            "dqs1_tail_unchanged": True,
            "candidate_runtime_patched": True,
            "byte_closed_candidate_emitted": True,
            "receiver_proof_ready": True,
            "receiver_contract_satisfied": True,
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
                "blockers": [],
            },
            "candidate_runtime_patch": {
                "alpha_default": best_alpha,
                "blend_context_model_scale": best_scale,
            },
            "selected_payload": {
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
                "source_payload_bytes": len(parts["selector_payload"]),
                "candidate_payload_bytes": len(best_payload),
                "payload_saved_bytes": len(parts["selector_payload"]) - len(best_payload),
                "status": "realized_saving",
                "savings_realized": archive_saved_bytes > 0,
            },
            "serialized_archive_delta": {
                "schema": "serialized_archive_delta_contract.v1",
                "source_archive_bytes": source_archive_record["bytes"],
                "candidate_archive_bytes": candidate_archive_record["bytes"],
                "saved_bytes": archive_saved_bytes,
            },
            "readiness_blockers": readiness_blockers,
            "sweep_rows": sorted(
                rows,
                key=lambda row: (
                    int(row.get("selector_payload_bytes") or 1 << 60),
                    int(row["scale"]),
                    int(row["alpha"]),
                ),
            ),
        },
        dispatch_blockers=tuple(readiness_blockers),
    )
    require_no_truthy_authority_fields(manifest, context="feca_selector_reparameterization")
    write_json_artifact(output / "feca_selector_reparameterization_manifest.json", manifest)
    return manifest


__all__ = [
    "FECA_REPARAMETERIZATION_MANIFEST_SCHEMA",
    "FECA_REPARAMETERIZATION_PROOF_SCHEMA",
    "FECA_SELECTOR_MATERIALIZER_ID",
    "FECA_SELECTOR_RECEIVER_CONTRACT_ID",
    "FECA_SELECTOR_RECEIVER_CONTRACT_KIND",
    "FECA_SELECTOR_TARGET_KIND",
    "FecaSelectorReparameterizationError",
    "build_feca_selector_reparameterized_candidate",
    "join_fp11_member",
    "split_fp11_member",
]
