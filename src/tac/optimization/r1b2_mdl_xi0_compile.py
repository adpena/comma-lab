# SPDX-License-Identifier: MIT
"""Fail-closed custody gate and counted archive assembler for R1b2.

This module never manufactures scorer derivatives, secants, full-kernel search
results, or pose coordinates.  It audits those producer-owned artifacts, emits
an exact decomposed blocker while any production edge is absent, and only then
assembles their already-realized payloads into a deterministic counted archive.
The offline full-kernel search is deliberately outside the receiver contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tac.boundary_math.integer_plane_emitter import SCORER_HEIGHT, SCORER_WIDTH
from tac.boundary_math.shared_receiver_admission import (
    MAX_ARCHIVE_BYTES,
    MAX_D_SEG,
    SCORE_BYTES_NORMALIZER,
)
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryJointSolveError,
    decode_boundary_packet,
)

RECEIPT_SCHEMA: Final = "r1b2_mdl_xi0_compile_receipt.v1"
ARCHIVE_SCHEMA: Final = "r1b2_counted_archive.v1"
RANK4_SCHEMA: Final = "r1b2_rank4_secant_custody.v2"
FULL_KERNEL_SCHEMA: Final = "r1b2_full_kernel_mdl_selection.v1"
XI0_SCHEMA: Final = "r1b2_xi0_custody.v1"
VJP_CAMPAIGN_SCHEMA: Final = "vjp_custody_n600_extension.v1"

PAIR_COUNT: Final = 600
BATCH_SIZE: Final = 16
SEED: Final = 1234
CONTROL_ARCHIVE_BYTES: Final = 94_344
CONTROL_ARCHIVE_SHA256: Final = "d633e6bfbbb5963b638f6f469ed1298ac86dbe3e04e5eae1b06b08cf64397539"
CONTROL_D_SEG: Final = 0.003515794640406966
CONTROL_D_POSE: Final = 127.36588287353516
CONTROL_SCORE: Final = 36.10275630841103
# Main inbox 2026-07-20T19:22:34Z: use the one-byte-conservative
# pointer-crossing floor until the registered equation input precision is reconciled.
FIXED_C1_CAP_BYTES: Final = 216_222
TASK_CAP_BYTES_PER_PAIR: Final = 477.8
DECODE_LIMIT_SECONDS: Final = 1_800.0
CONDITIONAL_CARRIER_LIMIT_BYTES: Final = 1_852
CONSUMED_REALIZATION_FRACTION: Final = 0.09462
MODERATE_MARGIN_FLIPS: Final = 16_319
MODERATE_MARGIN_SCORE_DEBT: Final = 0.01383
TIE_TIGHT_FLIPS: Final = 1_607
TIE_TIGHT_SCORE_DEBT: Final = 0.00136
GAP_FLIPS: Final = 17_926

MANIFEST_NAME: Final = "r1b2_manifest.json"
BOUNDARY_NAME: Final = "boundary_coordinate.bgj"
REPLAY_NAME: Final = "full_kernel_replay.r1k"
XI0_NAME: Final = "xi0.xi0"
EXTENSION_NAMES: Final = (MANIFEST_NAME, BOUNDARY_NAME, REPLAY_NAME, XI0_NAME)


class R1B2CompileError(ValueError):
    """Malformed, incomplete, or non-production R1b2 custody."""


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_finite_real(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return float(value) > 0.0 if positive else float(value) >= 0.0


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise R1B2CompileError("artifact is not canonical-JSON encodable") from exc


def _read_json(path: Path) -> dict[str, Any]:
    value, _custody_row = _read_json_snapshot(path)
    return value


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B2CompileError(f"invalid JSON artifact: {resolved}") from exc
    if not isinstance(value, dict):
        raise R1B2CompileError(f"JSON artifact must contain an object: {resolved}")
    return value, {
        "path": str(resolved),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R1B2CompileError(f"receipt overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    )
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _custody(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise R1B2CompileError(f"custody path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def audit_control_receipt(path: Path) -> dict[str, Any]:
    """Re-derive the settled control's small-file custody without re-scoring."""

    resolved = path.expanduser().resolve(strict=True)
    receipt, receipt_custody = _read_json_snapshot(resolved)
    row = receipt.get("row")
    authority = receipt.get("authority")
    archive = receipt.get("archive")
    custody = receipt.get("custody")
    if not all(isinstance(value, dict) for value in (row, authority, archive, custody)):
        raise R1B2CompileError("control receipt lacks row/authority/archive/custody objects")
    expected_row = {
        "archive_bytes": CONTROL_ARCHIVE_BYTES,
        "archive_sha256": CONTROL_ARCHIVE_SHA256,
        "pair_count": PAIR_COUNT,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "d_seg": CONTROL_D_SEG,
        "d_pose": CONTROL_D_POSE,
        "score": CONTROL_SCORE,
    }
    drift = {
        key: {"expected": expected, "actual": row.get(key)}
        for key, expected in expected_row.items()
        if row.get(key) != expected
    }
    if drift:
        raise R1B2CompileError(f"settled control row drift: {drift}")
    if authority.get("axis") != "[macOS-CPU advisory]" or authority.get("contest_score_claim") is not False:
        raise R1B2CompileError("control authority axis or score-claim boundary drifted")

    archive_path = Path(str(archive.get("archive", ""))).expanduser().resolve(strict=True)
    if archive_path.stat().st_size != CONTROL_ARCHIVE_BYTES:
        raise R1B2CompileError("control archive byte count drifted")
    if sha256_file(archive_path) != CONTROL_ARCHIVE_SHA256:
        raise R1B2CompileError("control archive SHA-256 drifted")
    decoder_row = custody.get("base_decoder")
    scorer_rows = custody.get("scorer_hashes")
    upstream = Path(str(custody.get("upstream", ""))).expanduser().resolve(strict=True)
    if not isinstance(decoder_row, dict) or not isinstance(scorer_rows, dict):
        raise R1B2CompileError("control decoder/scorer hash custody is absent")
    decoder = Path(str(decoder_row.get("path", ""))).expanduser().resolve(strict=True)
    if sha256_file(decoder) != decoder_row.get("sha256"):
        raise R1B2CompileError("control decoder SHA-256 drifted")
    scorer_paths = {
        "modules.py": upstream / "modules.py",
        "frame_utils.py": upstream / "frame_utils.py",
        "posenet.safetensors": upstream / "models/posenet.safetensors",
        "segnet.safetensors": upstream / "models/segnet.safetensors",
    }
    actual_scorers = {name: sha256_file(source) for name, source in scorer_paths.items()}
    if actual_scorers != scorer_rows:
        raise R1B2CompileError("control scorer source hashes drifted")
    return {
        "receipt": receipt_custody,
        "archive": _custody(archive_path),
        "base_decoder": _custody(decoder),
        "scorer_hashes": actual_scorers,
        "row": {key: row[key] for key in expected_row},
        "authority": authority,
    }


def _sidecars_from_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str | None,
) -> tuple[list[dict[str, Any]], set[int], set[int], dict[str, Any]]:
    resolved = manifest_path.expanduser().resolve(strict=True)
    manifest, manifest_custody = _read_json_snapshot(resolved)
    observed_sha = manifest_custody["sha256"]
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise R1B2CompileError(f"VJP manifest SHA drift: {resolved}")
    rows = manifest.get("sidecars")
    if not isinstance(rows, list):
        raise R1B2CompileError(f"VJP manifest sidecars are absent: {resolved}")
    completed: set[int] = set()
    declared: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise R1B2CompileError("VJP sidecar row is not an object")
        pair_id = row.get("pair_id")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int) or not 0 <= pair_id < PAIR_COUNT:
            raise R1B2CompileError("VJP sidecar pair_id is outside [0,600)")
        if pair_id in completed:
            raise R1B2CompileError(f"VJP manifest duplicates pair {pair_id}")
        sidecar = Path(str(row.get("path", ""))).expanduser().resolve(strict=True)
        declared_bytes = row.get("bytes")
        declared_sha = row.get("sha256")
        tensor_hashes = row.get("tensor_hashes")
        if (
            sidecar.stat().st_size != declared_bytes
            or not isinstance(declared_sha, str)
            or len(declared_sha) != 64
            or not isinstance(tensor_hashes, dict)
        ):
            raise R1B2CompileError(f"VJP sidecar declared custody is malformed: pair {pair_id}")
        completed.add(pair_id)
        declared.append(
            {
                "pair_id": pair_id,
                "path": str(sidecar),
                "bytes": declared_bytes,
                "declared_sha256": declared_sha,
                "tensor_hashes": tensor_hashes,
                "sidecar_bytes_rehashed_by_r1b2": False,
            }
        )
    refused: set[int] = set()
    for row in manifest.get("refusals", []):
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), int):
            raise R1B2CompileError("VJP refusal row is malformed")
        refused.add(int(row["pair_id"]))
    return (
        declared,
        completed,
        refused,
        manifest_custody,
    )


def audit_vjp_campaign(path: Path, *, rehash_sidecars: bool = True) -> dict[str, Any]:
    """Audit campaign custody, optionally deferring the expensive bulk rehash.

    ``rehash_sidecars=False`` is only manifest-level custody.  It exists so a
    partial compile can report cheaper missing predecessor artifacts without
    rereading roughly 90 GB of VJP tensors.  A terminal candidate compile must
    still call the default strict path and rehash every sidecar.
    """

    resolved = path.expanduser().resolve(strict=True)
    campaign, campaign_custody = _read_json_snapshot(resolved)
    if campaign.get("schema") != VJP_CAMPAIGN_SCHEMA:
        raise R1B2CompileError("VJP campaign schema mismatch")
    declared_rows: dict[int, dict[str, Any]] = {}
    refused: set[int] = set()
    manifest_custody: list[dict[str, Any]] = []

    sources = campaign.get("source_manifests", [])
    chunks = campaign.get("chunks", [])
    if not isinstance(sources, list) or not isinstance(chunks, list):
        raise R1B2CompileError("VJP campaign source/chunk rows are malformed")
    manifest_specs: list[tuple[Path, str | None]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise R1B2CompileError("VJP source manifest row is malformed")
        manifest_specs.append((Path(str(source.get("path", ""))), source.get("sha256")))
        refused.update(int(value) for value in source.get("refused_pair_ids", []))
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise R1B2CompileError("VJP chunk row is malformed")
        manifest_path = Path(str(chunk.get("path", ""))) / "manifest.json"
        if manifest_path.is_file():
            manifest_specs.append((manifest_path, chunk.get("manifest_sha256")))
        refused.update(int(value) for value in chunk.get("refused_pair_ids", []))

    seen_manifests: set[Path] = set()
    for manifest_path, expected_sha in manifest_specs:
        resolved_manifest = manifest_path.expanduser().resolve(strict=True)
        if resolved_manifest in seen_manifests:
            continue
        seen_manifests.add(resolved_manifest)
        rows, completed, manifest_refused, custody = _sidecars_from_manifest(
            resolved_manifest, expected_sha256=expected_sha
        )
        duplicate = set(declared_rows) & completed
        if duplicate:
            raise R1B2CompileError(f"VJP campaign duplicates completed pairs: {sorted(duplicate)}")
        declared_rows.update({int(row["pair_id"]): row for row in rows})
        refused.update(manifest_refused)
        manifest_custody.append(custody)

    completed_ids = sorted(declared_rows)
    missing_ids = sorted(set(range(PAIR_COUNT)) - set(completed_ids))
    refused.difference_update(completed_ids)
    blockers: list[str] = []
    if campaign.get("status") != "COMPLETE_N600":
        blockers.append("VJP_CAMPAIGN_NOT_TERMINAL_COMPLETE_N600")
    if completed_ids != list(range(PAIR_COUNT)):
        blockers.append(f"VJP_COMPLETED_PAIR_COUNT_{len(completed_ids)}_NOT_600")
    if missing_ids:
        blockers.append("VJP_MISSING_PAIR_IDS_PRESENT")
    if refused:
        blockers.append("VJP_REFUSED_PAIR_IDS_PRESENT")
    if campaign.get("final_completed_count") not in (None, len(completed_ids)):
        blockers.append("VJP_CAMPAIGN_FINAL_COUNT_DISAGREES_WITH_MANIFESTS")
    if campaign.get("status") == "COMPLETE_N600":
        if campaign.get("final_completed_count") != PAIR_COUNT:
            blockers.append("VJP_TERMINAL_FINAL_COUNT_NOT_EXACTLY_600")
        if campaign.get("still_missing_pair_ids") != []:
            blockers.append("VJP_TERMINAL_STILL_MISSING_FIELD_NOT_EMPTY")
        if campaign.get("refused_pair_ids") != []:
            blockers.append("VJP_TERMINAL_REFUSED_FIELD_NOT_EMPTY")
    bytes_rehashed = False
    if not blockers and rehash_sidecars:
        for pair_id in completed_ids:
            row = declared_rows[pair_id]
            if sha256_file(Path(row["path"])) != row["declared_sha256"]:
                raise R1B2CompileError(f"VJP sidecar SHA drift: pair {pair_id}")
            row["sidecar_bytes_rehashed_by_r1b2"] = True
        bytes_rehashed = True
    return {
        "campaign": campaign_custody,
        "updated_at_utc": campaign.get("updated_at_utc"),
        "status": campaign.get("status"),
        "completed_pair_count": len(completed_ids),
        "completed_pair_ids": completed_ids,
        "missing_pair_count": len(missing_ids),
        "missing_pair_ids": missing_ids,
        "refused_pair_ids": sorted(refused),
        "manifest_custody": manifest_custody,
        "per_pair_declared_custody": [declared_rows[index] for index in completed_ids],
        "sidecar_bytes_rehashed_by_r1b2": bytes_rehashed,
        "sidecar_rehash_requested": rehash_sidecars,
        "blockers": blockers,
    }


def _audit_payload_manifest(
    path: Path | None,
    *,
    schema: str,
    blocker: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, [blocker]
    resolved = path.expanduser().resolve(strict=True)
    manifest, manifest_custody = _read_json_snapshot(resolved)
    if manifest.get("schema") != schema:
        raise R1B2CompileError(f"{schema} manifest schema mismatch")
    if manifest.get("pair_count") != PAIR_COUNT or manifest.get("score_claim") is not False:
        raise R1B2CompileError(f"{schema} manifest pair-count/authority mismatch")
    return {"custody": manifest_custody, "manifest": manifest}, []


def audit_rank4_secants(path: Path | None, *, vjp_campaign_sha256: str) -> tuple[dict[str, Any] | None, list[str]]:
    audited, blockers = _audit_payload_manifest(
        path,
        schema=RANK4_SCHEMA,
        blocker="R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT",
    )
    if audited is None:
        return None, blockers
    manifest = audited["manifest"]
    required = {
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "head_rank": 4,
        "moderate_margin_lower_inclusive": 1e-3,
        "moderate_margin_upper_exclusive": 1.0,
        "moderate_margin_flip_count": MODERATE_MARGIN_FLIPS,
        "moderate_margin_score_debt": MODERATE_MARGIN_SCORE_DEBT,
        "vjp_campaign_sha256": vjp_campaign_sha256,
    }
    drift = {key: value for key, value in required.items() if manifest.get(key) != value}
    rows = manifest.get("per_pair")
    if drift or not isinstance(rows, list) or len(rows) != PAIR_COUNT:
        raise R1B2CompileError(f"rank4/secant production custody mismatch: {drift}")
    if [row.get("pair_index") for row in rows if isinstance(row, dict)] != list(range(PAIR_COUNT)):
        raise R1B2CompileError("rank4/secant per-pair index coverage mismatch")
    moderate_total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise R1B2CompileError("rank4/secant per-pair row is malformed")
        if row.get("batch_size") != BATCH_SIZE or row.get("head_rank") != 4:
            raise R1B2CompileError("rank4/secant per-pair batch/rank custody mismatch")
        strata = row.get("stratum_counts")
        if not isinstance(strata, dict) or set(strata) != {
            "moderate_margin_1e_3_to_1",
            "tie_tight_lt_1e_3",
            "other",
        }:
            raise R1B2CompileError("rank4/secant per-pair stratum custody mismatch")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in strata.values()):
            raise R1B2CompileError("rank4/secant stratum counts must be nonnegative integers")
        moderate_total += int(strata["moderate_margin_1e_3_to_1"])
        blocks = row.get("custody_blocks")
        if not isinstance(blocks, list):
            raise R1B2CompileError("rank4/secant per-pair custody_blocks must be a list")
        block_ids: set[str] = set()
        block_moderate_total = 0
        for block in blocks:
            required_block_keys = {
                "block_id",
                "base_id",
                "base_sha256",
                "delta_sha256",
                "delta_norm",
                "norm_id",
                "evaluation_scale",
                "remainder_bound",
                "quantization_cells_crossed",
                "hard_oracle_endpoint",
                "frechet_first_order",
                "realized_uint8_secant",
            }
            if not isinstance(block, dict) or set(block) != required_block_keys:
                raise R1B2CompileError("rank4/secant block lacks strict typed custody")
            block_id = block["block_id"]
            if not isinstance(block_id, str) or not block_id or block_id in block_ids:
                raise R1B2CompileError("rank4/secant block_id must be a unique nonempty string")
            block_ids.add(block_id)
            if (
                not isinstance(block["base_id"], str)
                or not block["base_id"]
                or not _is_sha256(block["base_sha256"])
                or not _is_sha256(block["delta_sha256"])
                or not _is_finite_real(block["delta_norm"], positive=True)
                or not isinstance(block["norm_id"], str)
                or not block["norm_id"]
                or not _is_finite_real(block["evaluation_scale"], positive=True)
                or not _is_finite_real(block["remainder_bound"])
                or isinstance(block["quantization_cells_crossed"], bool)
                or not isinstance(block["quantization_cells_crossed"], int)
                or block["quantization_cells_crossed"] < 0
            ):
                raise R1B2CompileError("rank4/secant block base/delta/remainder/cell custody mismatch")
            endpoint = block["hard_oracle_endpoint"]
            if not isinstance(endpoint, dict) or set(endpoint) != {
                "schema",
                "endpoint_id",
                "endpoint_sha256",
                "pair_index",
                "batch_size",
                "seed",
                "through_uint8_rounding",
                "realized_flip_count",
            }:
                raise R1B2CompileError("rank4/secant block lacks strict hard-oracle endpoint custody")
            if (
                endpoint["schema"] != "r1b2_hard_oracle_endpoint.v1"
                or not isinstance(endpoint["endpoint_id"], str)
                or not endpoint["endpoint_id"]
                or not _is_sha256(endpoint["endpoint_sha256"])
                or endpoint["pair_index"] != row["pair_index"]
                or endpoint["batch_size"] != BATCH_SIZE
                or endpoint["seed"] != SEED
                or endpoint["through_uint8_rounding"] is not True
                or isinstance(endpoint["realized_flip_count"], bool)
                or not isinstance(endpoint["realized_flip_count"], int)
                or endpoint["realized_flip_count"] < 0
            ):
                raise R1B2CompileError("rank4/secant hard-oracle endpoint custody mismatch")
            tensors: list[dict[str, Any]] = []
            for name, object_type in (
                ("frechet_first_order", "frechet_first_order_tangent.v1"),
                ("realized_uint8_secant", "realized_uint8_endpoint_secant.v1"),
            ):
                tensor = block[name]
                if not isinstance(tensor, dict) or set(tensor) != {
                    "object_type",
                    "path",
                    "bytes",
                    "sha256",
                    "shape",
                    "dtype",
                }:
                    raise R1B2CompileError(f"rank4/secant block lacks strict {name} tensor custody")
                tensor_path = Path(str(tensor["path"])).expanduser().resolve(strict=True)
                if (
                    tensor["object_type"] != object_type
                    or tensor["bytes"] != tensor_path.stat().st_size
                    or tensor["sha256"] != sha256_file(tensor_path)
                    or tensor["dtype"] not in {"float32", "float64"}
                    or not isinstance(tensor["shape"], list)
                    or len(tensor["shape"]) != 2
                    or isinstance(tensor["shape"][0], bool)
                    or not isinstance(tensor["shape"][0], int)
                    or tensor["shape"][0] <= 0
                    or isinstance(tensor["shape"][1], bool)
                    or not isinstance(tensor["shape"][1], int)
                    or tensor["shape"][1] <= 0
                ):
                    raise R1B2CompileError(f"rank4/secant {name} tensor custody mismatch")
                tensors.append(tensor)
            if tensors[0]["shape"] != tensors[1]["shape"]:
                raise R1B2CompileError("Frechet tangent and realized uint8 secant tensor shapes differ")
            if endpoint["realized_flip_count"] > tensors[0]["shape"][0]:
                raise R1B2CompileError("hard-oracle endpoint flip count exceeds block population")
            block_moderate_total += int(tensors[0]["shape"][0])
        if block_moderate_total != strata["moderate_margin_1e_3_to_1"]:
            raise R1B2CompileError("rank4/secant block populations do not match moderate-margin stratum")
    if moderate_total != MODERATE_MARGIN_FLIPS:
        raise R1B2CompileError(f"rank4/secant moderate-margin total {moderate_total} != {MODERATE_MARGIN_FLIPS}")
    boundary_path = Path(str(manifest.get("boundary_packet_path", ""))).expanduser().resolve(strict=True)
    boundary_bytes = boundary_path.read_bytes()
    try:
        packet = decode_boundary_packet(boundary_bytes)
    except BoundaryJointSolveError as exc:
        raise R1B2CompileError("rank4 manifest boundary packet is invalid") from exc
    if packet.pair_count != PAIR_COUNT or sha256_file(boundary_path) != manifest.get("boundary_packet_sha256"):
        raise R1B2CompileError("rank4 boundary packet pair-count/SHA mismatch")
    audited["boundary_packet"] = _custody(boundary_path)
    return audited, []


def audit_full_kernel(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    audited, blockers = _audit_payload_manifest(
        path,
        schema=FULL_KERNEL_SCHEMA,
        blocker="R1B2_FULL_KERNEL_MDL_SELECTION_AND_COMPACT_REPLAY_ABSENT",
    )
    if audited is None:
        return None, blockers
    manifest = audited["manifest"]
    if manifest.get("offline_search") is not True or manifest.get("receiver_search") is not False:
        raise R1B2CompileError("full-kernel search must be offline and absent from receiver")
    if manifest.get("replay_schema") != "r1b2_compact_full_kernel_replay.v1":
        raise R1B2CompileError("full-kernel compact replay schema mismatch")
    preimages = manifest.get("preimage_custody")
    if not isinstance(preimages, dict) or set(preimages) != {
        "schema",
        "dtype",
        "lower_bound",
        "upper_bound",
        "selection_objective",
        "exact_search",
        "preimage_count",
        "path",
        "sha256",
    }:
        raise R1B2CompileError("full-kernel exact uint8 preimage custody is incomplete")
    if (
        preimages["schema"] != "r1b2_full_resize_kernel_uint8_preimages.v1"
        or preimages["dtype"] != "uint8"
        or preimages["lower_bound"] != 0
        or preimages["upper_bound"] != 255
        or preimages["selection_objective"] != "minimum_description_length"
        or preimages["exact_search"] is not True
        or isinstance(preimages["preimage_count"], bool)
        or not isinstance(preimages["preimage_count"], int)
        or preimages["preimage_count"] <= 0
        or manifest.get("hard_oracle_after_preimage_selection") is not True
    ):
        raise R1B2CompileError("full-kernel preimages must be exact bounded uint8 MDL selections")
    preimage_path = Path(str(preimages["path"])).expanduser().resolve(strict=True)
    if not _is_sha256(preimages["sha256"]) or sha256_file(preimage_path) != preimages["sha256"]:
        raise R1B2CompileError("full-kernel exact uint8 preimage SHA mismatch")
    proof = manifest.get("receiver_proof")
    if (
        not isinstance(proof, dict)
        or proof.get("search_invocations") != 0
        or isinstance(proof.get("deterministic_decode_runs"), bool)
        or not isinstance(proof.get("deterministic_decode_runs"), int)
        or proof["deterministic_decode_runs"] < 2
        or proof.get("decoded_sha256_run1") != proof.get("decoded_sha256_run2")
        or not _is_sha256(proof.get("decoded_sha256_run1"))
        or not isinstance(proof.get("receiver_entrypoint"), str)
        or not proof.get("receiver_entrypoint")
        or not _is_sha256(proof.get("receiver_source_sha256"))
    ):
        raise R1B2CompileError("full-kernel compact receiver proof is incomplete")
    replay_path = Path(str(manifest.get("replay_path", ""))).expanduser().resolve(strict=True)
    if sha256_file(replay_path) != manifest.get("replay_sha256"):
        raise R1B2CompileError("full-kernel compact replay SHA mismatch")
    audited["preimages"] = _custody(preimage_path)
    audited["replay"] = _custody(replay_path)
    return audited, []


def audit_xi0(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    audited, blockers = _audit_payload_manifest(
        path,
        schema=XI0_SCHEMA,
        blocker="R1B2_XI0_ONLY_POSE_CUSTODY_ABSENT",
    )
    if audited is None:
        return None, blockers
    manifest = audited["manifest"]
    if manifest.get("coordinate_indices") != [0]:
        raise R1B2CompileError("xi custody must contain coordinate zero only")
    if (
        manifest.get("receiver_schema") != "r1b2_xi0_receiver.v1"
        or manifest.get("other_coordinates_counted") not in (0, None)
        or manifest.get("quantization") not in {"int8_scaled", "int16_scaled", "float16_le"}
    ):
        raise R1B2CompileError("xi0 receiver/quantization custody mismatch")
    payload_path = Path(str(manifest.get("payload_path", ""))).expanduser().resolve(strict=True)
    if sha256_file(payload_path) != manifest.get("payload_sha256"):
        raise R1B2CompileError("xi0 payload SHA mismatch")
    audited["payload"] = _custody(payload_path)
    return audited, []


def _zip_members(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [row.filename for row in infos]
        if (
            len(names) != len(set(names))
            or any(row.is_dir() or row.flag_bits & 1 for row in infos)
            or any(name in EXTENSION_NAMES for name in names)
        ):
            raise R1B2CompileError("control archive members are unsafe or collide with R1b2")
        return [(name, archive.read(name)) for name in names]


def _write_zip(path: Path, members: Sequence[tuple[str, bytes]]) -> None:
    if path.exists():
        raise R1B2CompileError(f"candidate archive overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, payload in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def parse_candidate_archive(path: Path) -> dict[str, Any]:
    from tac.boundary_math.r1b4_section_receiver import (
        APPLICATION_ORDER,
        RECEIVER_SCHEMA,
        R1B4ReceiverError,
        decode_replay_payload,
    )
    from tac.optimization.r1b3_producer_preflight import (
        R1B3ProducerError,
        decode_xi0_payload,
    )

    resolved = path.expanduser().resolve(strict=True)
    members = _zip_members_allow_extensions(resolved)
    names = [name for name, _ in members]
    if len(names) <= len(EXTENSION_NAMES):
        raise R1B2CompileError("R1b2 archive lacks inherited control members")
    if names[-4:] != list(EXTENSION_NAMES):
        raise R1B2CompileError("R1b2 extension member order mismatch")
    payloads = dict(members)
    manifest_raw = payloads[MANIFEST_NAME]
    try:
        manifest = json.loads(manifest_raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R1B2CompileError("R1b2 manifest is invalid") from exc
    if not isinstance(manifest, dict) or canonical_json(manifest) != manifest_raw:
        raise R1B2CompileError("R1b2 manifest is not canonical")
    if (
        manifest.get("schema") != ARCHIVE_SCHEMA
        or manifest.get("pair_count") != PAIR_COUNT
        or manifest.get("artifact_role") != "r1b2_candidate"
        or manifest.get("receiver_search") is not False
        or manifest.get("receiver_schema") != RECEIVER_SCHEMA
        or manifest.get("application_order") != list(APPLICATION_ORDER)
        or manifest.get("score_claim") is not False
        or any(
            isinstance(manifest.get(name), bool) or not isinstance(manifest.get(name), int)
            for name in (
                "base_archive_bytes",
                "base_zip_compressed_bytes",
                "base_zip_overhead_bytes",
            )
        )
        or manifest.get("base_archive_bytes", 0) <= 0
        or manifest.get("base_zip_compressed_bytes", -1) < 0
        or manifest.get("base_zip_overhead_bytes", -1) < 0
    ):
        raise R1B2CompileError("R1b2 manifest sealed values mismatch")
    base_names = names[: -len(EXTENSION_NAMES)]
    expected_base = manifest.get("base_sections")
    actual_base = {name: {"bytes": len(payloads[name]), "sha256": _sha256_bytes(payloads[name])} for name in base_names}
    if expected_base != actual_base:
        raise R1B2CompileError("R1b2 inherited base-section custody mismatch")
    source_hashes = manifest.get("source_manifest_hashes")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        raise R1B2CompileError("R1b2 source-manifest hash custody is malformed")
    for name in (BOUNDARY_NAME, REPLAY_NAME, XI0_NAME):
        expected = manifest.get("sections", {}).get(name)
        actual = {"bytes": len(payloads[name]), "sha256": _sha256_bytes(payloads[name])}
        if expected != actual:
            raise R1B2CompileError(f"R1b2 section custody mismatch: {name}")
    try:
        packet = decode_boundary_packet(payloads[BOUNDARY_NAME])
        decode_replay_payload(payloads[REPLAY_NAME])
        xi0_values = decode_xi0_payload(payloads[XI0_NAME])
    except (BoundaryJointSolveError, R1B4ReceiverError, R1B3ProducerError) as exc:
        raise R1B2CompileError("R1b2 semantic section parse-back failed") from exc
    if packet.pair_count != PAIR_COUNT:
        raise R1B2CompileError("R1b2 boundary packet is not n600")
    if (packet.scorer_height, packet.scorer_width) != (SCORER_HEIGHT, SCORER_WIDTH):
        raise R1B2CompileError("R1b2 boundary packet scorer geometry is not production 384x512")
    if xi0_values.shape != (PAIR_COUNT,):
        raise R1B2CompileError("R1b2 xi0 payload is not n600")
    with zipfile.ZipFile(resolved, "r") as archive:
        infos = {row.filename: row for row in archive.infolist()}
    candidate_compressed_bytes = sum(row.compress_size for row in infos.values())
    candidate_overhead_bytes = resolved.stat().st_size - candidate_compressed_bytes
    extension_compressed_bytes = sum(infos[name].compress_size for name in EXTENSION_NAMES)
    candidate_base_compressed_bytes = sum(infos[name].compress_size for name in base_names)
    base_repack_compressed_delta = candidate_base_compressed_bytes - manifest["base_zip_compressed_bytes"]
    overhead_delta = candidate_overhead_bytes - manifest["base_zip_overhead_bytes"]
    archive_delta = resolved.stat().st_size - manifest["base_archive_bytes"]
    accounted_delta = base_repack_compressed_delta + extension_compressed_bytes + overhead_delta
    if archive_delta != accounted_delta:
        raise R1B2CompileError("R1b2 archive-delta decomposition is inconsistent")
    return {
        "schema": ARCHIVE_SCHEMA,
        "archive": str(resolved),
        "archive_bytes": resolved.stat().st_size,
        "archive_sha256": sha256_file(resolved),
        "manifest": manifest,
        "parse_back": True,
        "carrier_delta_bytes": archive_delta,
        "rate_decomposition": {
            "base_repack_compressed_delta_bytes": base_repack_compressed_delta,
            "extension_compressed_bytes": extension_compressed_bytes,
            "zip_overhead_delta_bytes": overhead_delta,
            "accounted_archive_delta_bytes": accounted_delta,
        },
    }


def _zip_members_allow_extensions(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [row.filename for row in infos]
        if len(names) != len(set(names)) or any(row.is_dir() or row.flag_bits & 1 for row in infos):
            raise R1B2CompileError("candidate archive members are unsafe")
        return [(name, archive.read(name)) for name in names]


def compile_candidate_archive(
    *,
    control_archive: Path,
    boundary_packet: Path,
    replay_payload: Path,
    xi0_payload: Path,
    source_manifest_hashes: Mapping[str, str],
    output: Path,
) -> dict[str, Any]:
    # Imported lazily to keep the compiler/receiver dependency one-way during
    # module initialization while still refusing counted-but-unparseable bytes.
    from tac.boundary_math.r1b4_section_receiver import (
        APPLICATION_ORDER,
        RECEIVER_SCHEMA,
        R1B4ReceiverError,
        decode_replay_payload,
        default_receiver_policy,
    )
    from tac.optimization.r1b3_producer_preflight import (
        R1B3ProducerError,
        decode_xi0_payload,
    )

    base_members = _zip_members(control_archive)
    boundary = boundary_packet.read_bytes()
    replay = replay_payload.read_bytes()
    xi0 = xi0_payload.read_bytes()
    try:
        packet = decode_boundary_packet(boundary)
    except BoundaryJointSolveError as exc:
        raise R1B2CompileError("boundary packet is invalid") from exc
    if packet.pair_count != PAIR_COUNT:
        raise R1B2CompileError("boundary packet must contain exactly n600 pairs")
    if (packet.scorer_height, packet.scorer_width) != (SCORER_HEIGHT, SCORER_WIDTH):
        raise R1B2CompileError("boundary packet scorer geometry must be production 384x512")
    try:
        decode_replay_payload(replay)
        decoded_xi0 = decode_xi0_payload(xi0)
    except (R1B4ReceiverError, R1B3ProducerError) as exc:
        raise R1B2CompileError("receiver replay/xi0 payload is not strict canonical bytes") from exc
    if decoded_xi0.shape != (PAIR_COUNT,):
        raise R1B2CompileError("xi0 payload must contain exactly n600 coordinate-zero values")
    if not source_manifest_hashes or any(not _is_sha256(value) for value in source_manifest_hashes.values()):
        raise R1B2CompileError("source manifest hashes must be nonempty SHA-256 strings")
    with zipfile.ZipFile(control_archive, "r") as archive:
        base_infos = archive.infolist()
    base_zip_compressed_bytes = sum(row.compress_size for row in base_infos)
    base_zip_overhead_bytes = control_archive.stat().st_size - base_zip_compressed_bytes
    base_sections = {name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)} for name, payload in base_members}
    sections = {
        name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for name, payload in (
            (BOUNDARY_NAME, boundary),
            (REPLAY_NAME, replay),
            (XI0_NAME, xi0),
        )
    }
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "pair_count": PAIR_COUNT,
        "artifact_role": "r1b2_candidate",
        "base_archive_sha256": sha256_file(control_archive),
        "base_archive_bytes": control_archive.stat().st_size,
        "base_zip_compressed_bytes": base_zip_compressed_bytes,
        "base_zip_overhead_bytes": base_zip_overhead_bytes,
        "base_sections": base_sections,
        "sections": sections,
        "source_manifest_hashes": dict(source_manifest_hashes),
        "offline_full_kernel_selection": True,
        "receiver_search": False,
        "xi_coordinate_indices": [0],
        "receiver_schema": RECEIVER_SCHEMA,
        "receiver_policy": default_receiver_policy(),
        "application_order": list(APPLICATION_ORDER),
        "final_output_assertion": {
            "status": "unsealed",
            "pair_cap": PAIR_COUNT,
            "decoded_bytes": PAIR_COUNT * 2 * 874 * 1164 * 3,
            "decoded_sha256": None,
        },
        "score_claim": False,
    }
    members = [
        *base_members,
        (MANIFEST_NAME, canonical_json(manifest)),
        (BOUNDARY_NAME, boundary),
        (REPLAY_NAME, replay),
        (XI0_NAME, xi0),
    ]
    if output.exists():
        raise R1B2CompileError(f"candidate archive overwrite refused: {output}")
    staging = output.with_name(f".{output.name}.candidate.{os.getpid()}")
    if staging.exists():
        raise R1B2CompileError(f"candidate staging path already exists: {staging}")
    try:
        _write_zip(staging, members)
        parsed = parse_candidate_archive(staging)
        delta_bytes = parsed["carrier_delta_bytes"]
        parsed["conditional_carrier_limit_bytes"] = CONDITIONAL_CARRIER_LIMIT_BYTES
        parsed["conditional_carrier_limit_pass"] = delta_bytes <= CONDITIONAL_CARRIER_LIMIT_BYTES
        parsed["task_archive_gate_pass"] = parsed["archive_bytes"] <= MAX_ARCHIVE_BYTES
        parsed["fixed_c1_cap_pass"] = parsed["archive_bytes"] <= FIXED_C1_CAP_BYTES
        if not parsed["conditional_carrier_limit_pass"]:
            raise R1B2CompileError(
                f"compiled carrier delta {delta_bytes} exceeds conditional limit {CONDITIONAL_CARRIER_LIMIT_BYTES}"
            )
        if not parsed["task_archive_gate_pass"] or not parsed["fixed_c1_cap_pass"]:
            raise R1B2CompileError("compiled candidate exceeds an archive byte cap")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        final = parse_candidate_archive(output)
        final.update(
            {
                "conditional_carrier_limit_bytes": CONDITIONAL_CARRIER_LIMIT_BYTES,
                "conditional_carrier_limit_pass": parsed["conditional_carrier_limit_pass"],
                "task_archive_gate_pass": parsed["task_archive_gate_pass"],
                "fixed_c1_cap_pass": parsed["fixed_c1_cap_pass"],
            }
        )
        return final
    finally:
        if staging.exists():
            staging.unlink()


def build_receipt(
    *,
    control: dict[str, Any],
    vjp: dict[str, Any],
    rank4: dict[str, Any] | None,
    full_kernel: dict[str, Any] | None,
    xi0: dict[str, Any] | None,
    blockers: Sequence[str],
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unique_blockers = list(dict.fromkeys(str(value) for value in blockers))
    control_row = control["row"]
    verdict = (
        "COMPILED_UNMEASURED_R1B2_CANDIDATE"
        if candidate is not None and not unique_blockers
        else "DECOMPOSED_PARTIAL_R1B2_PRODUCTION_CUSTODY_BLOCKED"
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "verdict": verdict,
        "verdict_scope": (
            "current production custody and exact control instance only; no R1b2 candidate "
            "measurement and no boundary/xi/full-kernel family negative"
        ),
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_mutation": False,
        },
        "control": control,
        "vjp_campaign": vjp,
        "rank4_first_order_realized_secants": rank4,
        "full_kernel_mdl_compact_replay": full_kernel,
        "xi0": xi0,
        "candidate": candidate,
        "headline_decomposition": {
            "gap": {
                "flips": GAP_FLIPS,
                "score_debt": MODERATE_MARGIN_SCORE_DEBT + TIE_TIGHT_SCORE_DEBT,
            },
            "moderate_margin_1e_3_to_1": {
                "flips": MODERATE_MARGIN_FLIPS,
                "score_debt": MODERATE_MARGIN_SCORE_DEBT,
                "share_of_gap_flips": MODERATE_MARGIN_FLIPS / GAP_FLIPS,
            },
            "tie_tight": {
                "flips": TIE_TIGHT_FLIPS,
                "score_debt": TIE_TIGHT_SCORE_DEBT,
                "share_of_gap_flips": TIE_TIGHT_FLIPS / GAP_FLIPS,
            },
            "vjp_custody": {
                "completed_pairs": vjp["completed_pair_count"],
                "missing_pairs": vjp["missing_pair_count"],
                "refused_pairs": vjp["refused_pair_ids"],
            },
        },
        "break_even": {
            "rate_price_score_per_byte": 25.0 / SCORE_BYTES_NORMALIZER,
            "consumed_realization_fraction": CONSUMED_REALIZATION_FRACTION,
            "conditional_carrier_limit_bytes": CONDITIONAL_CARRIER_LIMIT_BYTES,
            "new_realization_fraction": None,
            "recomputed_carrier_limit_bytes": None,
            "status": "OWED_UNTIL_COMPILED_HARD_ORACLE_REALIZATION_EXISTS",
        },
        "gates": {
            "task_archive_gate_bytes": MAX_ARCHIVE_BYTES,
            "fixed_c1_cap_bytes": FIXED_C1_CAP_BYTES,
            "task_cap_bytes_per_pair": TASK_CAP_BYTES_PER_PAIR,
            "d_seg_gate": MAX_D_SEG,
            "decode_limit_seconds": DECODE_LIMIT_SECONDS,
            "control_archive_gate_pass": control_row["archive_bytes"] <= MAX_ARCHIVE_BYTES,
            "control_fixed_c1_cap_pass": control_row["archive_bytes"] <= FIXED_C1_CAP_BYTES,
            "control_d_seg_gate_pass": control_row["d_seg"] <= MAX_D_SEG,
            "candidate_archive_gate_pass": None if candidate is None else candidate["task_archive_gate_pass"],
            "candidate_fixed_c1_cap_pass": None if candidate is None else candidate["fixed_c1_cap_pass"],
            "candidate_d_seg_gate_pass": None,
            "candidate_decode_gate_pass": None,
        },
        "blockers": unique_blockers,
        "next_coordinate": (
            "consume a terminal exact n600 VJP campaign with no refused pairs; materialize "
            "batch16 per-pair rank4 first-order plus realized-secant custody; run full-kernel "
            "MDL selection offline; compile compact replay plus xi[0]; then decode and hard-score n600"
        ),
        "pointer": "0.19108 [contest-CPU] UNMOVED",
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "FULL_KERNEL_SCHEMA",
    "RANK4_SCHEMA",
    "RECEIPT_SCHEMA",
    "XI0_SCHEMA",
    "R1B2CompileError",
    "atomic_json",
    "audit_control_receipt",
    "audit_full_kernel",
    "audit_rank4_secants",
    "audit_vjp_campaign",
    "audit_xi0",
    "build_receipt",
    "compile_candidate_archive",
    "parse_candidate_archive",
    "sha256_file",
]
