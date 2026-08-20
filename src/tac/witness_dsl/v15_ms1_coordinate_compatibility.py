# SPDX-License-Identifier: MIT
"""Strict V15/MS1 coordinate audit and selected-preimage adapter contract.

This module does not solve or compress a selected preimage.  It proves the
types that a real compiler must connect, computes the coupled score byte
budget, and fail-closes if the sealed receipts no longer support the claimed
compatibility.  Large C1 arrays are never loaded into memory.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from tac.contest_score import compute_contest_score

RECEIPT_SCHEMA: Final = "tac.v15_ms1_coordinate_compatibility_receipt.v1"
PROGRAM_CONTRACT_ID: Final = "tac.taskspace_selected_preimage_program.v1"
PAIR_COUNT: Final = 600
CAMERA_SHAPE: Final = (PAIR_COUNT, 2, 874, 1164, 3)
SCORER_PLANE_SHAPE: Final = (PAIR_COUNT, 2, 384, 512, 3)
C1_RAW_BYTES: Final = int(np.prod(CAMERA_SHAPE, dtype=np.int64))


class CompatibilityAuditError(ValueError):
    """A sealed input or claimed coordinate contract differs."""


@dataclass(frozen=True)
class AuditPaths:
    """Canonical custody and source paths consumed by the audit."""

    repo_root: Path
    v15_receipt: Path
    v15_archive: Path
    ms1_ingest_receipt: Path
    ms1_historical_receipt: Path
    c1_root: Path
    frontier_pointer: Path

    @classmethod
    def canonical(cls, repo_root: Path | str) -> AuditPaths:
        root = Path(repo_root).resolve()
        return cls(
            repo_root=root,
            v15_receipt=root / ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
            "ddm_v15_scorer_solved_templates_n600_receipt.json",
            v15_archive=root / ".omx/research/ddm_v15_scorer_solved_templates_n600_20260723T013000Z/"
            "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes",
            ms1_ingest_receipt=root / ".omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json",
            ms1_historical_receipt=root
            / ".omx/research/ddm_ms1_min_description_lattice_solve_20260723T233549Z/receipt.json",
            c1_root=Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719"),
            frontier_pointer=root / ".omx/state/canonical_frontier_pointer.json",
        )


@dataclass(frozen=True)
class SelectedPreimagePair:
    """One exact independent scorer-plane pair at the V10 receiver boundary."""

    pair_index: int
    frame0: np.ndarray
    frame1: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.pair_index, int) or isinstance(self.pair_index, bool):
            raise CompatibilityAuditError("pair_index must be an exact integer")
        if not 0 <= self.pair_index < PAIR_COUNT:
            raise CompatibilityAuditError("pair_index is outside the n600 population")
        frame0 = _validate_plane(self.frame0, "frame0")
        frame1 = _validate_plane(self.frame1, "frame1")
        if np.shares_memory(frame0, frame1):
            raise CompatibilityAuditError("selected-preimage frames must be independently owned")
        object.__setattr__(self, "frame0", frame0)
        object.__setattr__(self, "frame1", frame1)


class TaskspaceSelectedPreimageProgram(Protocol):
    """Decode-only interface owed by the first real V15-to-V10 compiler."""

    contract_id: str
    pair_count: int

    def decode_selected_preimage_pair(self, pair_index: int) -> SelectedPreimagePair:
        """Decode one counted-program state to two independent uint8 planes."""


def _validate_plane(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (384, 512, 3):
        raise CompatibilityAuditError(f"{label} must have shape (384,512,3) and dtype uint8")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompatibilityAuditError("receipt is not canonical-JSON encodable") from exc


def sha256_file(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise CompatibilityAuditError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityAuditError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CompatibilityAuditError(f"{label} must be a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise CompatibilityAuditError(f"{label} differs: expected {expected!r}, got {actual!r}")


def _artifact(path: Path, *, expected_bytes: int, expected_sha256: str, live_hash: bool = True) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CompatibilityAuditError(f"artifact is absent: {path}") from exc
    _require_equal(size, expected_bytes, f"{path} bytes")
    observed_sha = sha256_file(path) if live_hash else None
    if observed_sha is not None:
        _require_equal(observed_sha, expected_sha256, f"{path} SHA-256")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": expected_sha256,
        "live_sha256_verified": live_hash,
    }


def _strict_v15_envelope(archive_payload: bytes) -> dict[str, Any]:
    """Parse the complete small ZIP envelope without materializing n600 masks."""

    expected_members = (
        "manifest.json",
        "predictor.zip",
        "predict/movable_polygon_worldsheet.g1s",
        "render/receiver_realization.ddrp",
        "render/scorer_solved_templates.ddst",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            _require_equal(names, expected_members, "V15 member order")
            for info in infos:
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.flag_bits & 0x1
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.extra
                    or info.comment
                ):
                    raise CompatibilityAuditError("V15 ZIP member framing is unsafe or noncanonical")
            members = {name: archive.read(name) for name in names}
            manifest = json.loads(members["manifest.json"])
    except CompatibilityAuditError:
        raise
    except (KeyError, RuntimeError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityAuditError("V15 archive is not a readable typed envelope") from exc
    if not isinstance(manifest, dict):
        raise CompatibilityAuditError("V15 manifest must be a JSON object")
    _require_equal(manifest.get("schema"), "direct_description_v15_scorer_solved_template_archive.v1", "V15 schema")
    _require_equal(manifest.get("magic"), "DDV15S1", "V15 magic")
    _require_equal(manifest.get("pair_count"), PAIR_COUNT, "V15 envelope pair count")
    _require_equal(manifest.get("source_pair_start"), 0, "V15 source pair start")
    linked = {
        "predict/movable_polygon_worldsheet.g1s": manifest.get("grammar", {})
        .get("movable", {})
        .get("g1_polygon_worldsheet", {}),
        "render/receiver_realization.ddrp": manifest.get("realization_profile", {}),
        "render/scorer_solved_templates.ddst": manifest.get("scorer_solved_templates", {}),
    }
    for name, row in linked.items():
        if not isinstance(row, dict):
            raise CompatibilityAuditError(f"V15 manifest lacks member row {name}")
        _require_equal(row.get("member"), name, f"V15 member link {name}")
        _require_equal(row.get("bytes"), len(members[name]), f"V15 member bytes {name}")
        _require_equal(row.get("sha256"), hashlib.sha256(members[name]).hexdigest(), f"V15 member SHA-256 {name}")
    try:
        with zipfile.ZipFile(io.BytesIO(members["predictor.zip"]), "r") as predictor_archive:
            predictor_manifest = json.loads(predictor_archive.read("manifest.json"))
    except (KeyError, RuntimeError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityAuditError("V15 nested predictor is unreadable") from exc
    _require_equal(
        predictor_manifest.get("schema"), "direct_description_stratum_composed_archive.v1", "V15 predictor schema"
    )
    _require_equal(predictor_manifest.get("pair_count"), PAIR_COUNT, "V15 predictor pair count")
    _require_equal(predictor_manifest.get("source_pair_start"), 0, "V15 predictor source pair start")
    _require_equal(predictor_manifest.get("ground_truth_argmax_present"), False, "V15 GT absence")
    _require_equal(predictor_manifest.get("scorer_weights_present"), False, "V15 scorer-weight absence")
    return {
        "strict_archive_parse": True,
        "member_order": list(expected_members),
        "member_count": len(expected_members),
        "nested_predictor_schema": predictor_manifest["schema"],
        "pair_count": PAIR_COUNT,
        "source_pair_start": 0,
        "scorer_weights_present": False,
        "ground_truth_argmax_present": False,
    }


def byte_ceiling_for_score(target: float, d_seg: float, d_pose: float) -> int:
    """Largest integer archive size whose canonical arithmetic is <= target."""

    for label, value in (("target", target), ("d_seg", d_seg), ("d_pose", d_pose)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise CompatibilityAuditError(f"{label} must be finite and nonnegative")
    low = 0
    high = 1
    while compute_contest_score(d_seg, d_pose, high) <= target:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if compute_contest_score(d_seg, d_pose, middle) <= target:
            low = middle
        else:
            high = middle
    return low


def _frontier(pointer: Mapping[str, Any]) -> dict[str, Any]:
    try:
        effective = pointer["effective_frontier"]
        cpu = float(pointer["our_local_frontier_contest_cpu"]["score"])
        cuda = float(pointer["our_local_frontier_contest_cuda"]["score"])
        upstream = float(pointer["upstream_leaderboard_snapshot"]["best_entry"]["score"])
        score = float(effective["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompatibilityAuditError("frontier pointer lacks its selection inputs") from exc
    _require_equal(score, min(cpu, cuda, upstream), "effective frontier selection")
    _require_equal(
        effective.get("selection_rule"),
        "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, upstream_official_leaderboard.best_entry)",
        "effective frontier rule",
    )
    return {
        "path": None,
        "effective_score_to_beat": score,
        "selection_inputs": {"contest_cpu": cpu, "contest_cuda": cuda, "upstream": upstream},
        "selection_rule_verified": True,
        "source": effective.get("source"),
    }


def audit_coordinate_compatibility(paths: AuditPaths, *, strict_v15_parse: bool = True) -> dict[str, Any]:
    """Audit the real n600 artifacts and return a deterministic typed receipt."""

    v15_receipt = _read_json(paths.v15_receipt, label="V15 receipt")
    _require_equal(v15_receipt.get("schema"), "ddm_v15_scorer_solved_template_receipt.v1", "V15 receipt schema")
    ladder = v15_receipt.get("solved_template_ladder")
    if not isinstance(ladder, list):
        raise CompatibilityAuditError("V15 receipt lacks a candidate ladder")
    selected = next((row for row in ladder if row.get("candidate") == v15_receipt.get("selected_candidate")), None)
    if not isinstance(selected, dict):
        raise CompatibilityAuditError("V15 selected candidate is absent from the ladder")
    _require_equal(selected.get("archive_bytes"), 133_941, "V15 archive bytes")
    _require_equal(
        selected.get("archive_sha256"),
        "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df",
        "V15 archive SHA-256",
    )
    _require_equal(selected.get("batch_size"), 16, "V15 measurement batch size")
    _require_equal(selected.get("full_p_camera_identity", {}).get("pair_count"), PAIR_COUNT, "V15 pair count")
    v15_archive = _artifact(
        paths.v15_archive,
        expected_bytes=133_941,
        expected_sha256="759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df",
    )

    render_contract: dict[str, Any] = {
        "strict_archive_parse": False,
        "output_shape": list(CAMERA_SHAPE),
        "output_dtype": "uint8",
        "output_contract_authority": "sealed full-P camera identity in the V15 receipt",
    }
    if strict_v15_parse:
        render_contract.update(_strict_v15_envelope(paths.v15_archive.read_bytes()))

    ms1_ingest = _read_json(paths.ms1_ingest_receipt, label="MS1 ingest receipt")
    ms1_historical = _read_json(paths.ms1_historical_receipt, label="MS1 historical receipt")
    _require_equal(
        ms1_ingest.get("schema"), "ddm_ms1_min_description_lattice_solve_repo_receipt.v1", "MS1 ingest schema"
    )
    _require_equal(
        ms1_historical.get("schema"), "ddm_min_description_lattice_solve_receipt.v1", "MS1 historical schema"
    )
    historical_identity = ms1_ingest.get("historical_v1_receipt", {})
    _require_equal(historical_identity.get("bytes"), paths.ms1_historical_receipt.stat().st_size, "MS1 receipt bytes")
    _require_equal(
        historical_identity.get("sha256"),
        sha256_file(paths.ms1_historical_receipt),
        "MS1 historical receipt SHA-256",
    )
    raw = ms1_historical.get("inputs", {}).get("raw", {})
    _require_equal(raw.get("bytes"), C1_RAW_BYTES, "MS1/C1 raw bytes")
    _require_equal(
        raw.get("sha256"),
        "31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b",
        "MS1/C1 raw SHA-256",
    )
    changed = ms1_historical.get("lattice", {}).get("changed_pairs", {})
    _require_equal(changed, {"B_previous_frame": 0, "C_xi_motion": 0}, "MS1 changed-pair counts")
    local = ms1_ingest.get("local_member_selection", {})
    for name in ("previous_frame", "pose_xi"):
        _require_equal(local.get(name, {}).get("proposal_wins"), 0, f"MS1 {name} proposal wins")
        _require_equal(local.get(name, {}).get("changed_pairs"), 0, f"MS1 {name} changed pairs")
    diagnostic = ms1_ingest.get("realized_frozen_scorer_diagnostic", {})
    _require_equal(diagnostic.get("pair_count"), PAIR_COUNT, "MS1 diagnostic pair count")
    _require_equal(diagnostic.get("argmax_identical_pairs"), PAIR_COUNT, "MS1 argmax-identical pairs")
    _require_equal(diagnostic.get("pose6_bit_identical_pairs"), PAIR_COUNT, "MS1 pose-identical pairs")
    d_seg = float(diagnostic["realized_d_seg"])
    d_pose = float(diagnostic["realized_d_pose"])

    prepare_path = paths.c1_root / "prepare_receipt.json"
    receiver_path = paths.c1_root / "inflate_serial_receipt.json.receiver.json"
    prepare = _read_json(prepare_path, label="C1 prepare receipt")
    receiver_receipt = _read_json(receiver_path, label="C1 receiver receipt")
    _require_equal(prepare.get("schema"), "v10_two_plane_receiver_prepare.v1", "C1 prepare schema")
    _require_equal(prepare.get("pair_count"), PAIR_COUNT, "C1 pair count")
    _require_equal(prepare.get("camera_hw"), [874, 1164], "C1 camera geometry")
    _require_equal(prepare.get("scorer_hw"), [384, 512], "C1 scorer geometry")
    _require_equal(prepare.get("frame0_policy_id"), "description-frame0.v1", "C1 frame0 policy")
    _require_equal(receiver_receipt.get("pair_count"), PAIR_COUNT, "C1 receiver pair count")
    _require_equal(receiver_receipt.get("both_planes_exact"), True, "C1 exact two-plane verdict")
    _require_equal(receiver_receipt.get("raw_bytes"), C1_RAW_BYTES, "C1 receiver raw bytes")
    _require_equal(receiver_receipt.get("raw_sha256"), raw.get("sha256"), "C1/MS1 raw identity")
    c1_archive = _artifact(
        paths.c1_root / "capstone_submission/archive.zip",
        expected_bytes=409_526_925,
        expected_sha256="e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42",
        live_hash=False,
    )

    pointer_json = _read_json(paths.frontier_pointer, label="frontier pointer")
    frontier = _frontier(pointer_json)
    frontier["path"] = str(paths.frontier_pointer)
    frontier["sha256"] = sha256_file(paths.frontier_pointer)
    target = float(frontier["effective_score_to_beat"])
    target_ceiling = byte_ceiling_for_score(target, d_seg, d_pose)
    goal_ceiling = byte_ceiling_for_score(0.15, d_seg, d_pose)

    source_path = paths.repo_root / "src/tac/optimization/direct_description_carrier_compose.py"
    historical_source = next(
        (
            row
            for row in v15_receipt.get("producer_custody", [])
            if row.get("path") == "src/tac/optimization/direct_description_carrier_compose.py"
        ),
        None,
    )
    if not isinstance(historical_source, dict):
        raise CompatibilityAuditError("V15 receipt lacks historical receiver source custody")
    current_source_sha = sha256_file(source_path)

    c0b_source = paths.repo_root / "src/tac/witness_dsl/c0b_semantic_quotient.py"
    c0b_builder = paths.repo_root / "tools/build_c0b_semantic_quotient_archive.py"
    v10_receiver = paths.repo_root / "src/tac/witness_dsl/v10_production_receiver.py"
    for path in (c0b_source, c0b_builder, v10_receiver):
        if not path.is_file():
            raise CompatibilityAuditError(f"existing exact seam source is absent: {path}")

    distortion_only = compute_contest_score(d_seg, d_pose, 0)
    economics: dict[str, Any] = {
        "authority": "derived canonical score arithmetic only; not an eval row",
        "d_seg": d_seg,
        "d_pose": d_pose,
        "distortion_only_score": distortion_only,
        "effective_frontier_target": target,
        "max_archive_bytes_at_effective_frontier": target_ceiling,
        "score_at_max_archive_bytes": compute_contest_score(d_seg, d_pose, target_ceiling),
        "score_at_next_byte": compute_contest_score(d_seg, d_pose, target_ceiling + 1),
        "max_archive_bytes_at_sub_0_15": goal_ceiling,
        "bases": {},
    }
    for name, size in (("v15_direct", 133_941), ("v15_j2_seeded", 134_211)):
        economics["bases"][name] = {
            "archive_bytes": size,
            "hypothetical_score_if_ms1_distortion_survived": compute_contest_score(d_seg, d_pose, size),
            "headroom_bytes_to_effective_frontier_ceiling": target_ceiling - size,
            "headroom_bytes_to_sub_0_15_ceiling": goal_ceiling - size,
            "score_claim": False,
        }

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "heavy_launch_performed": False,
            "large_raw_live_rehashed": False,
            "purpose": "strict type/coordinate compatibility and coupled byte-budget audit",
        },
        "frontier": frontier,
        "artifacts": {
            "v15": {
                "archive": v15_archive,
                "receipt": {
                    "path": str(paths.v15_receipt),
                    "bytes": paths.v15_receipt.stat().st_size,
                    "sha256": sha256_file(paths.v15_receipt),
                },
                "render_contract": render_contract,
                "historical_receiver_source_sha256": historical_source.get("sha256"),
                "current_receiver_source_sha256": current_source_sha,
                "current_source_matches_historical": current_source_sha == historical_source.get("sha256"),
                "d_seg": float(selected["d_seg"]),
                "d_pose": float(selected["d_pose"]),
            },
            "ms1": {
                "ingest_receipt": {
                    "path": str(paths.ms1_ingest_receipt),
                    "bytes": paths.ms1_ingest_receipt.stat().st_size,
                    "sha256": sha256_file(paths.ms1_ingest_receipt),
                },
                "historical_receipt": {
                    "path": str(paths.ms1_historical_receipt),
                    "bytes": paths.ms1_historical_receipt.stat().st_size,
                    "sha256": sha256_file(paths.ms1_historical_receipt),
                },
                "state_identity": "UNCHANGED_C1_CAMERA_RAW; no winning lattice proposal and no distinct latent",
                "changed_pairs": changed,
                "proposal_wins": {"previous_frame": 0, "pose_xi": 0},
                "selected_raw": raw,
                "diagnostic": diagnostic,
            },
            "c1": {
                "archive": c1_archive,
                "prepare_receipt": {
                    "path": str(prepare_path),
                    "bytes": prepare_path.stat().st_size,
                    "sha256": sha256_file(prepare_path),
                },
                "receiver_receipt": {
                    "path": str(receiver_path),
                    "bytes": receiver_path.stat().st_size,
                    "sha256": sha256_file(receiver_path),
                },
                "packet": {
                    "bytes": prepare.get("packet_bytes"),
                    "sha256": prepare.get("packet_sha256"),
                    "predictor_payload_bytes": prepare.get("predictor_payload_bytes"),
                    "predictor_payload_sha256": prepare.get("predictor_payload_sha256"),
                },
                "raw": raw,
            },
        },
        "compatibility": {
            "outer_camera_coordinate": {
                "verdict": "EXACT_SHARED",
                "shape": list(CAMERA_SHAPE),
                "dtype": "uint8",
                "order": "pair,frame,row,column,channel",
            },
            "scorer_plane_coordinate": {
                "verdict": "SHAPE_SHARED_STATE_NOT_SHARED",
                "shape": list(SCORER_PLANE_SHAPE),
                "dtype": "uint8",
                "v15_state": "semantic role masks, curves/worldsheet, shared templates, Pose6 codes",
                "c1_state": "two independent arbitrary scorer-plane RGB arrays",
            },
            "latent_or_field_cast": {
                "verdict": "FALSIFIED",
                "reason": "MS1 emitted no compact latent; V15 has no field for two independent C1 planes",
            },
            "pose_coordinate": {
                "verdict": "NOT_SHARED",
                "reason": "V15 Pose6 codes and shared photometric templates do not encode C1's independent pose-legible pair",
            },
            "bridge_required": "real semantic renderer plus selected-preimage factor/quotient/residual program",
        },
        "existing_exact_seam": {
            "verdict": "IMPLEMENTED_NONPROMOTABLE_BASELINE",
            "container": "tac.c0b_semantic_quotient_archive.v1",
            "mechanism": "counted semantic program plus exact LZMA-compressed uint8 XOR quotient into V10 factor2 receiver",
            "scientific_label": "NONPROMOTABLE_DENSE_C1_QUOTIENT_SCIENTIFIC_SEAM_BASELINE",
            "sources": [
                {"path": str(c0b_source), "sha256": sha256_file(c0b_source)},
                {"path": str(c0b_builder), "sha256": sha256_file(c0b_builder)},
                {"path": str(v10_receiver), "sha256": sha256_file(v10_receiver)},
            ],
            "current_base_archive_bytes": 339_094,
            "frontier_blocker": "base already exceeds the MS1-distortion byte ceiling; dense quotient is much larger",
        },
        "lawful_candidate_boundary": {
            "free_decode_code": [
                "generic semantic renderer",
                "V10 factor2 integer realization",
                "generic factor/quotient synthesis and deterministic repair algorithms",
            ],
            "counted_video_derived_payload": [
                "fresh own-lineage semantic program",
                "fresh selected-preimage factors/coefficients/selectors",
                "fresh residual or quotient bytes",
                "fresh pair gauge/pose realization state",
            ],
            "forbidden_or_nonpromotable": [
                "scorer weights or target/GT tables in the archive or decoder",
                "historical C1/MS1 planes hidden in inflate code",
                "historical selected arrays copied as a fresh current candidate",
                "dense C1 quotient presented as compact or frontier-ready",
            ],
        },
        "economics": economics,
        "first_executable_adapter_contract": {
            "contract_id": PROGRAM_CONTRACT_ID,
            "population": "exact n600 only",
            "compiler_input": {
                "semantic_program": "fresh counted V15-equivalent typed program",
                "selected_preimages": "fresh encoder-only n600 C1-equivalent teacher planes",
                "obligations": "frozen scorer-recursive Seg argmax and Pose6 cells",
                "config": "typed deterministic factorization/rate allocation config",
            },
            "compiler_output": {
                "counted_program": "semantic base plus factored selected-preimage correction",
                "required_parse_back": "byte-canonical and all payload sections receiver-effective",
            },
            "decode_api": "decode_selected_preimage_pair(program,pair_index)->SelectedPreimagePair",
            "decode_output": {
                "frame0": {"shape": [384, 512, 3], "dtype": "uint8"},
                "frame1": {"shape": [384, 512, 3], "dtype": "uint8"},
                "independent_ownership": True,
            },
            "receiver": "exact V10 factor2 realization to (2,874,1164,3) uint8 per pair",
            "admission": [
                "archive bytes within coupled target ceiling",
                "exact n600 public inflate closure",
                "realized through-R d_seg/d_pose on exact archive bytes",
                "authoritative upstream/evaluate.py row before any score/frontier claim",
            ],
            "missing_implementation": [
                "fresh V15-equivalent semantic packet at or below the direct 133941-byte base",
                "factorized selected-preimage compiler that fits the remaining coupled byte budget",
                "pose-gauge realization state that preserves the low-distortion pair",
            ],
        },
        "verdict": (
            "CONFIRM exact shared outer camera coordinate; FALSIFY native latent/field compatibility. "
            "MS1 is the unchanged C1 raw, and the exact C0B seam already exists. "
            "The frontier-moving missing object is the compact fresh factorized selected-preimage program."
        ),
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    return receipt


__all__ = [
    "PROGRAM_CONTRACT_ID",
    "RECEIPT_SCHEMA",
    "AuditPaths",
    "CompatibilityAuditError",
    "SelectedPreimagePair",
    "TaskspaceSelectedPreimageProgram",
    "audit_coordinate_compatibility",
    "byte_ceiling_for_score",
    "canonical_json",
    "sha256_file",
]
