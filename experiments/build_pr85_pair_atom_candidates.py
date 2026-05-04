#!/usr/bin/env python3
"""Build or fail-close PR85 pair-atom archive candidates.

The scorer-gradient PR85 plan ranks pair indices by first-order opportunity; it
does not contain a stream/value action direction. This tool therefore only
builds archives when an explicit pair-action spec and an explicit local runtime
contract are supplied. Without those, it emits a deterministic readiness JSON
and keeps dispatch locked.

No scorer is imported, no GPU is used, no dispatch state is touched, and every
archive that is built remains eligible for exact eval only after a lane claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import brotli
except ImportError:  # pragma: no cover - environment guard
    brotli = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    FIXED_V5_LENGTHS,
    SEGMENT_ORDER,
    Pr85BundleError,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/build_pr85_pair_atom_candidates.py"
SCHEMA = "pr85_pair_atom_candidate_readiness_v1"
MANIFEST_SCHEMA = "pr85_pair_atom_candidate_v1"
ACTION_SPEC_SCHEMA = "pr85_pair_atom_action_spec_v1"
RUNTIME_CONTRACT_SCHEMA = "pr85_pair_atom_runtime_contract_v1"
SCORER_PLAN_SCHEMA = "pr85_scorer_gradient_atom_opportunity_v1"

DEFAULT_SOURCE_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_SCORER_PLAN = Path("/tmp/pr85_scorer_gradient_atoms_plan_baseline.json")
DEFAULT_OUT_DIR = Path("/tmp/pr85_pair_atom_candidates_worker_20260504")
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr85_pair_atom_candidates_worker_20260504.md"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAIR_COUNT = 600
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
TOP_POLISH_PAIRS = (192, 60, 164, 197, 70, 496, 106, 522)
KNOWN_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
    "score": 0.25806611029397786,
}

CHOICE_STREAM_DEFAULTS = {
    "shift": 40,
    "frac": 4,
    "frac2": 4,
    "frac3": 4,
    "bias": 13,
    "region": 0,
}
CHOICE_STREAMS = tuple(CHOICE_STREAM_DEFAULTS)


class PairAtomBuilderError(ValueError):
    """Raised when explicit inputs are malformed enough to abort the run."""


@dataclass(frozen=True)
class Action:
    pair_index: int
    stream: str
    value: int
    rationale: str | None = None


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    actions: tuple[Action, ...]
    header_mode: str | None


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairAtomBuilderError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PairAtomBuilderError(f"{path} must contain a JSON object")
    return payload


def _stable_digest(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "stable_plan_digest_sha256"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_candidate_id(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise PairAtomBuilderError(f"unsafe candidate_id: {value!r}")
    return value


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    validate_pr85_member_name(name)
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_member_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), payload)


def _read_source_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise PairAtomBuilderError(f"source archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise PairAtomBuilderError(f"PR85 source archive must contain exactly one safe member 'x'; got {names!r}")
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    sha = _sha256_file(path)
    return (
        {
            "path": _rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": sha,
            "member_name": info.filename,
            "member_bytes": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_sha256": _sha256_bytes(raw),
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
            "known_pr85_anchor_match": {
                "matches": (
                    int(path.stat().st_size) == KNOWN_PR85["archive_bytes"]
                    and sha == KNOWN_PR85["archive_sha256"]
                ),
                "expected_archive_bytes": KNOWN_PR85["archive_bytes"],
                "expected_archive_sha256": KNOWN_PR85["archive_sha256"],
                "exact_t4_score_context": KNOWN_PR85["score"],
            },
        },
        raw,
    )


def _archive_info(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise PairAtomBuilderError(f"candidate archive must contain one member; found {len(infos)}")
        info = infos[0]
        raw = zf.read(info)
    return {
        "archive_path": _rel(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_name": info.filename,
        "member_bytes": int(len(raw)),
        "member_sha256": _sha256_bytes(raw),
        "zip_stored": info.compress_type == zipfile.ZIP_STORED,
    }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _top_atoms_from_plan(payload: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    atoms = payload.get("atom_ranking")
    if not isinstance(atoms, list):
        return []
    rows: list[dict[str, Any]] = []
    for atom in atoms[:limit]:
        if not isinstance(atom, dict):
            continue
        pair_index = atom.get("pair_index")
        if not isinstance(pair_index, int) or isinstance(pair_index, bool):
            continue
        break_even = (
            atom.get("byte_break_even", {})
            if isinstance(atom.get("byte_break_even"), dict)
            else {}
        )
        combined = (
            break_even.get("combined", {})
            if isinstance(break_even.get("combined"), dict)
            else {}
        )
        rows.append(
            {
                "atom_id": atom.get("atom_id"),
                "pair_index": pair_index,
                "frame_indices": atom.get("frame_indices"),
                "ranking_score": atom.get("ranking_score"),
                "break_even_bytes": combined.get("max_charged_bytes_for_zero_net_change"),
                "dispatch_gate": atom.get("dispatch_gate"),
            }
        )
    return rows


def _scorer_plan_report(
    path: Path,
    *,
    source_archive: Mapping[str, Any],
    expected_pairs: Sequence[int],
    require_known_pr85_anchor: bool,
    top_limit: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    blockers: list[dict[str, Any]] = []
    if not path.is_file():
        return (
            {
                "path": _rel(path),
                "status": "blocked",
                "blocker_class": "missing_scorer_gradient_plan",
                "blockers": [{"blocker_class": "missing_scorer_gradient_plan", "reason": "plan JSON does not exist"}],
                "top_atoms": [],
            },
            None,
        )
    payload = _load_json(path)
    if payload.get("schema") != SCORER_PLAN_SCHEMA:
        blockers.append({"blocker_class": "stale_scorer_gradient_source", "reason": "unexpected scorer-gradient schema"})
    if payload.get("planning_only") is not True or payload.get("score_claim") is not False:
        blockers.append({"blocker_class": "unsafe_scorer_gradient_source", "reason": "plan must be planning_only=true and score_claim=false"})
    for key in ("dispatch_performed", "remote_jobs_dispatched", "inflate_time_scorer_load_allowed"):
        if payload.get(key) not in (False, None):
            blockers.append({"blocker_class": "unsafe_scorer_gradient_source", "reason": f"{key} must be false"})

    declared_digest = payload.get("stable_plan_digest_sha256")
    computed_digest = _stable_digest(payload)
    if isinstance(declared_digest, str) and declared_digest != computed_digest:
        blockers.append({"blocker_class": "stale_scorer_gradient_source", "reason": "stable_plan_digest_sha256 mismatch"})

    exact_eval = payload.get("exact_eval") if isinstance(payload.get("exact_eval"), dict) else {}
    provenance = exact_eval.get("provenance") if isinstance(exact_eval.get("provenance"), dict) else {}
    plan_sha = provenance.get("archive_sha256")
    plan_bytes = provenance.get("archive_size_bytes", exact_eval.get("archive_size_bytes"))
    if plan_sha != source_archive["archive_sha256"] or int(plan_bytes or -1) != int(source_archive["archive_bytes"]):
        blockers.append(
            {
                "blocker_class": "stale_scorer_gradient_source",
                "reason": "scorer-gradient exact-eval source archive does not match selected source archive",
                "plan_archive_sha256": plan_sha,
                "source_archive_sha256": source_archive["archive_sha256"],
                "plan_archive_bytes": plan_bytes,
                "source_archive_bytes": source_archive["archive_bytes"],
            }
        )
    if require_known_pr85_anchor and not source_archive["known_pr85_anchor_match"]["matches"]:
        blockers.append({"blocker_class": "stale_scorer_gradient_source", "reason": "source archive is not the known PR85 T4 anchor"})
    if require_known_pr85_anchor and exact_eval.get("reported_score") != KNOWN_PR85["score"]:
        blockers.append({"blocker_class": "stale_scorer_gradient_source", "reason": "reported score does not match known PR85 T4 anchor"})

    top_atoms = _top_atoms_from_plan(payload, limit=top_limit)
    top_pair_set = {int(row["pair_index"]) for row in top_atoms}
    missing_pairs = [int(pair) for pair in expected_pairs if int(pair) not in top_pair_set]
    if require_known_pr85_anchor and missing_pairs:
        blockers.append({"blocker_class": "stale_scorer_gradient_source", "reason": "required top polish pairs missing from scorer-gradient top atoms", "missing_pair_indices": missing_pairs})

    status = "passed" if not blockers else "blocked"
    blocker_class = "none" if not blockers else str(blockers[0]["blocker_class"])
    return (
        {
            "path": _rel(path),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
            "status": status,
            "blocker_class": blocker_class,
            "blockers": blockers,
            "stable_plan_digest_sha256": declared_digest,
            "computed_stable_plan_digest_sha256": computed_digest,
            "exact_eval": {
                "archive_size_bytes": exact_eval.get("archive_size_bytes"),
                "reported_score": exact_eval.get("reported_score"),
                "avg_posenet_dist": exact_eval.get("avg_posenet_dist"),
                "avg_segnet_dist": exact_eval.get("avg_segnet_dist"),
                "n_samples": exact_eval.get("n_samples"),
                "provenance": provenance,
            },
            "top_atoms": top_atoms,
        },
        payload,
    )


def _runtime_probe() -> dict[str, Any]:
    paths = {
        "bundle_helper": REPO_ROOT / "src/tac/pr85_bundle.py",
        "fixed_runtime_bridge_builder": REPO_ROOT / "experiments/build_pr85_fixed_runtime_bridge_candidate.py",
        "bridge_sparse_action_builder": REPO_ROOT / "experiments/build_pr85_bridge_sparse_action_candidates.py",
        "final_bias_builder": REPO_ROOT / "experiments/build_pr85_final_bias_stack_candidates.py",
        "runtime_qpost_helper": REPO_ROOT / "submissions/robust_current/apply_qzs3_postprocess.py",
    }
    rows: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        rows[name] = {
            "path": _rel(path),
            "exists": path.is_file(),
            "has_qpost": "QPS1" in text or "qpost.bin" in text,
            "has_qrm1": "QRM1" in text,
            "has_pair_atom_contract": "pr85_pair_atom_runtime_contract_v1" in text,
            "has_scorer_load": "PoseNet" in text or "SegNet" in text,
        }
    return {
        "status": "inspected",
        "capabilities": rows,
        "conclusion": (
            "Existing PR85 helpers expose bundle slicing, qpost/QRM1 group bridges, "
            "and final-bias stacking, but no reviewed pair-action contract maps a "
            "scorer-gradient pair id to legal stream/value deltas."
        ),
    }


def _runtime_contract_report(
    path: Path | None,
    *,
    action_streams: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if path is None:
        return (
            {
                "status": "blocked",
                "blocker_class": "missing_pair_atom_runtime_contract",
                "path": None,
                "blockers": [
                    {
                        "blocker_class": "missing_pair_atom_runtime_contract",
                        "reason": "no explicit pair-atom runtime contract JSON was supplied",
                    }
                ],
            },
            None,
        )
    if not path.is_file():
        return (
            {
                "status": "blocked",
                "blocker_class": "missing_pair_atom_runtime_contract",
                "path": _rel(path),
                "blockers": [
                    {
                        "blocker_class": "missing_pair_atom_runtime_contract",
                        "reason": "runtime contract JSON does not exist",
                    }
                ],
            },
            None,
        )
    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != RUNTIME_CONTRACT_SCHEMA:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "unexpected runtime contract schema"})
    if payload.get("supports_pair_specific_actions") is not True:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "supports_pair_specific_actions must be true"})
    if payload.get("scorer_load_allowed") is not False:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "scorer_load_allowed must be false"})
    if payload.get("sidecars_allowed") is not False:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "sidecars_allowed must be false"})
    if payload.get("archive_member_contract") != "single_member_x":
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "archive_member_contract must be single_member_x"})
    supported_streams = payload.get("supported_streams")
    if not isinstance(supported_streams, list) or not all(isinstance(item, str) for item in supported_streams):
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "supported_streams must be a list of strings"})
        supported_streams = []
    missing = sorted(set(action_streams) - set(supported_streams))
    if missing:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "action streams are not supported by runtime contract", "missing_streams": missing})
    modes = payload.get("supported_header_modes")
    if not isinstance(modes, list) or not all(mode in {"v5", "explicit_30"} for mode in modes):
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "supported_header_modes must list v5 and/or explicit_30"})
    status = "passed" if not blockers else "blocked"
    return (
        {
            "status": status,
            "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
            "path": _rel(path),
            "sha256": _sha256_file(path),
            "blockers": blockers,
            "contract": payload,
        },
        payload,
    )


def _parse_action(raw: Any, *, row_index: int) -> Action:
    if not isinstance(raw, Mapping):
        raise PairAtomBuilderError(f"actions[{row_index}] must be an object")
    pair_index = raw.get("pair_index")
    if not isinstance(pair_index, int) or isinstance(pair_index, bool) or not 0 <= pair_index < PAIR_COUNT:
        raise PairAtomBuilderError(f"actions[{row_index}].pair_index must be an integer in [0,{PAIR_COUNT})")
    stream = raw.get("stream")
    if stream not in CHOICE_STREAMS:
        raise PairAtomBuilderError(f"actions[{row_index}].stream must be one of {CHOICE_STREAMS}")
    value = raw.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
        raise PairAtomBuilderError(f"actions[{row_index}].value must be an integer byte")
    rationale = raw.get("rationale")
    return Action(pair_index=int(pair_index), stream=str(stream), value=int(value), rationale=rationale if isinstance(rationale, str) else None)


def _action_spec_report(path: Path | None) -> tuple[dict[str, Any], list[CandidateSpec]]:
    if path is None:
        return (
            {
                "status": "blocked",
                "blocker_class": "missing_pair_action_spec",
                "path": None,
                "blockers": [
                    {
                        "blocker_class": "missing_pair_action_spec",
                        "reason": "scorer-gradient pairs are rankings only; no stream/value action spec was supplied",
                    }
                ],
                "candidate_specs": [],
            },
            [],
        )
    if not path.is_file():
        return (
            {
                "status": "blocked",
                "blocker_class": "missing_pair_action_spec",
                "path": _rel(path),
                "blockers": [{"blocker_class": "missing_pair_action_spec", "reason": "action spec JSON does not exist"}],
                "candidate_specs": [],
            },
            [],
        )
    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != ACTION_SPEC_SCHEMA:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "unexpected action spec schema"})
    if payload.get("score_claim") is not False:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "score_claim must be false"})
    if payload.get("dispatch_performed") is not False:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "dispatch_performed must be false"})
    if payload.get("inflate_time_scorer_load_allowed") is not False:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "inflate_time_scorer_load_allowed must be false"})

    specs: list[CandidateSpec] = []
    raw_candidates = payload.get("candidates")
    if isinstance(raw_candidates, list):
        candidate_rows = raw_candidates
    else:
        candidate_rows = [payload]
    for candidate_index, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            raise PairAtomBuilderError(f"candidates[{candidate_index}] must be an object")
        candidate_id = _safe_candidate_id(str(row.get("candidate_id") or payload.get("candidate_id") or f"pair_atom_{candidate_index:03d}"))
        raw_actions = row.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} has no actions"})
            continue
        actions = tuple(_parse_action(item, row_index=index) for index, item in enumerate(raw_actions))
        header_mode = row.get("header_mode", payload.get("header_mode"))
        if header_mode is not None and header_mode not in {"v5", "explicit_30"}:
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} has unsupported header_mode {header_mode!r}"})
            continue
        specs.append(CandidateSpec(candidate_id=candidate_id, actions=actions, header_mode=header_mode if isinstance(header_mode, str) else None))
    status = "passed" if not blockers else "blocked"
    return (
        {
            "status": status,
            "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
            "path": _rel(path),
            "sha256": _sha256_file(path),
            "blockers": blockers,
            "candidate_specs": [
                {
                    "candidate_id": spec.candidate_id,
                    "selected_pair_indices": sorted({action.pair_index for action in spec.actions}),
                    "action_count": len(spec.actions),
                    "streams": sorted({action.stream for action in spec.actions}),
                    "header_mode": spec.header_mode,
                }
                for spec in specs
            ],
        },
        specs,
    )


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values: list[int] = []
    for _ in range(count):
        value = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise PairAtomBuilderError("truncated varint stream")
            byte = raw[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if byte < 128:
                values.append(value)
                break
            shift += 7
            if shift > 63:
                raise PairAtomBuilderError("overlong varint stream")
    return values, pos


def _write_varint(value: int) -> bytes:
    if value < 0:
        raise PairAtomBuilderError(f"varint cannot encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_sparse_choice(raw: bytes, *, magic: bytes, default_choice: int) -> bytes:
    if not raw.startswith(magic) or len(raw) < 5:
        raise PairAtomBuilderError(f"bad sparse choice magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    vals = raw[pos : pos + count]
    if len(vals) != count or pos + count != len(raw):
        raise PairAtomBuilderError("sparse choice payload length mismatch")
    out = bytearray([default_choice] * PAIR_COUNT)
    index = -1
    for gap, value in zip(gaps, vals, strict=True):
        index += gap + 1
        if not 0 <= index < PAIR_COUNT:
            raise PairAtomBuilderError(f"sparse choice index out of range: {index}")
        out[index] = int(value) - 1
    return bytes(out)


def _encode_sparse_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    if len(values) != PAIR_COUNT:
        raise PairAtomBuilderError(f"choice stream must contain {PAIR_COUNT} pairs")
    indices = [index for index, value in enumerate(values) if value != default_choice]
    out = bytearray(magic + len(indices).to_bytes(2, "little"))
    previous = -1
    for index in indices:
        out += _write_varint(index - previous - 1)
        previous = index
    out += bytes(values[index] + 1 for index in indices)
    return bytes(out)


def _decode_direct_or_delta(raw: bytes, *, direct_magic: bytes, delta_magic: bytes | None, default_choice: int) -> tuple[bytes, str]:
    if raw.startswith(direct_magic):
        values = raw[len(direct_magic) :]
        if len(values) != PAIR_COUNT:
            raise PairAtomBuilderError(f"{direct_magic!r} stream has {len(values)} pairs")
        return bytes(values), direct_magic.decode("ascii")
    if delta_magic is not None and raw.startswith(delta_magic):
        encoded = raw[len(delta_magic) :]
        if len(encoded) != PAIR_COUNT:
            raise PairAtomBuilderError(f"{delta_magic!r} stream has {len(encoded)} pairs")
        return bytes(default_choice if value == 0 else value - 1 for value in encoded), delta_magic.decode("ascii")
    raise PairAtomBuilderError(f"unsupported choice stream magic {raw[:4]!r}")


def _decode_choice_stream(name: str, segment: bytes) -> tuple[bytearray, dict[str, Any]]:
    if brotli is None:
        raise PairAtomBuilderError("brotli is required for PR85 pair atom builds")
    try:
        raw = brotli.decompress(segment)
    except brotli.error as exc:
        raise PairAtomBuilderError(f"PR85 segment {name!r} is not Brotli-decodable") from exc
    default = CHOICE_STREAM_DEFAULTS[name]
    if name == "shift":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"SH4", delta_magic=b"SD4", default_choice=default)
    elif name == "frac":
        if raw.startswith(b"FV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"FV1", default_choice=default), "FV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH1", delta_magic=None, default_choice=default)
    elif name == "frac2":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH2", delta_magic=None, default_choice=default)
    elif name == "frac3":
        values, codec = _decode_direct_or_delta(raw, direct_magic=b"FH3", delta_magic=b"FD3", default_choice=default)
    elif name == "bias":
        if raw.startswith(b"BV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"BV1", default_choice=default), "BV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"BH1", delta_magic=b"BD1", default_choice=default)
    elif name == "region":
        if raw.startswith(b"RV1"):
            values, codec = _decode_sparse_choice(raw, magic=b"RV1", default_choice=default), "RV1"
        else:
            values, codec = _decode_direct_or_delta(raw, direct_magic=b"RH1", delta_magic=b"RD1", default_choice=default)
    else:  # pragma: no cover - guarded by action parsing
        raise PairAtomBuilderError(f"unsupported stream {name!r}")
    return bytearray(values), {
        "stream": name,
        "codec": codec,
        "source_raw_bytes": len(raw),
        "source_raw_sha256": _sha256_bytes(raw),
        "source_segment_bytes": len(segment),
        "source_segment_sha256": _sha256_bytes(segment),
        "default_symbol": default,
    }


def _encode_choice_stream(name: str, values: bytes, *, codec: str) -> tuple[bytes, dict[str, Any]]:
    if brotli is None:
        raise PairAtomBuilderError("brotli is required for PR85 pair atom builds")
    default = CHOICE_STREAM_DEFAULTS[name]
    if codec in {"SH4", "FH1", "FH2", "FH3", "BH1", "RH1"}:
        raw = codec.encode("ascii") + bytes(values)
    elif codec in {"SD4", "FD3", "BD1", "RD1"}:
        raw = codec.encode("ascii") + bytes(0 if value == default else value + 1 for value in values)
    elif codec in {"FV1", "BV1", "RV1"}:
        raw = _encode_sparse_choice(codec.encode("ascii"), bytes(values), default_choice=default)
    else:
        raise PairAtomBuilderError(f"unsupported re-encode codec {codec!r}")
    encoded = brotli.compress(raw, quality=11, lgwin=24)
    decoded = brotli.decompress(encoded)
    if decoded != raw:
        raise PairAtomBuilderError(f"{name} Brotli roundtrip changed bytes")
    return encoded, {
        "candidate_raw_bytes": len(raw),
        "candidate_raw_sha256": _sha256_bytes(raw),
        "candidate_segment_bytes": len(encoded),
        "candidate_segment_sha256": _sha256_bytes(encoded),
        "brotli_quality": 11,
        "brotli_lgwin": 24,
    }


def _choose_header_mode(
    *,
    source_format: str,
    requested: str | None,
    contract: Mapping[str, Any],
) -> str:
    if requested is not None:
        mode = requested
    elif source_format == "pr85_explicit_30byte_lengths":
        mode = "explicit_30"
    else:
        mode = "v5"
    modes = contract.get("supported_header_modes", [])
    if mode not in modes:
        raise PairAtomBuilderError(f"runtime contract does not support header_mode {mode!r}")
    return mode


def _candidate_blocked(candidate_id: str, blocker_class: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "build_status": "blocked",
        "blocker_class": blocker_class,
        "blockers": [{"blocker_class": blocker_class, "reason": reason, **extra}],
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "candidate_archive": None,
    }


def _build_one_candidate(
    *,
    spec: CandidateSpec,
    source_archive: Mapping[str, Any],
    source_raw: bytes,
    scorer_top_atoms: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    bundle = parse_pr85_bundle(source_raw)
    source_segments = {name: bytes(bundle.segments[name]) for name in SEGMENT_ORDER}
    top_by_pair = {int(row["pair_index"]): row for row in scorer_top_atoms if isinstance(row.get("pair_index"), int)}
    missing_pairs = sorted({action.pair_index for action in spec.actions if action.pair_index not in top_by_pair})
    if missing_pairs:
        return _candidate_blocked(
            spec.candidate_id,
            "pair_not_ranked_by_scorer_gradient_source",
            "action spec selected pair(s) not present in scorer-gradient top atoms",
            missing_pair_indices=missing_pairs,
        )

    streams = sorted({action.stream for action in spec.actions})
    values_by_stream: dict[str, bytearray] = {}
    stream_reports: dict[str, dict[str, Any]] = {}
    action_proofs: list[dict[str, Any]] = []
    for stream in streams:
        values, report = _decode_choice_stream(stream, source_segments[stream])
        values_by_stream[stream] = values
        stream_reports[stream] = report

    changed = False
    for action in spec.actions:
        values = values_by_stream[action.stream]
        before = int(values[action.pair_index])
        values[action.pair_index] = action.value
        changed_here = before != action.value
        changed = changed or changed_here
        top_atom = top_by_pair[action.pair_index]
        action_proofs.append(
            {
                "pair_index": action.pair_index,
                "stream": action.stream,
                "source_value": before,
                "candidate_value": action.value,
                "changed": changed_here,
                "rationale": action.rationale,
                "scorer_gradient_atom": {
                    "atom_id": top_atom.get("atom_id"),
                    "ranking_score": top_atom.get("ranking_score"),
                    "break_even_bytes": top_atom.get("break_even_bytes"),
                    "frame_indices": top_atom.get("frame_indices"),
                },
            }
        )
    if not changed:
        return _candidate_blocked(
            spec.candidate_id,
            "non_noop_proof_failed",
            "every explicit action preserved the source semantic value",
            selected_pair_indices=sorted({action.pair_index for action in spec.actions}),
        )

    candidate_segments = dict(source_segments)
    transform_reports = []
    for stream in streams:
        source_segment = source_segments[stream]
        encoded, encode_meta = _encode_choice_stream(stream, bytes(values_by_stream[stream]), codec=stream_reports[stream]["codec"])
        candidate_segments[stream] = encoded
        transform_reports.append(
            {
                **stream_reports[stream],
                **encode_meta,
                "segment_byte_delta": int(len(encoded) - len(source_segment)),
                "semantic_sha256_before": _sha256_bytes(bytes(_decode_choice_stream(stream, source_segment)[0])),
                "semantic_sha256_after": _sha256_bytes(bytes(values_by_stream[stream])),
                "changed_pair_indices": sorted({action.pair_index for action in spec.actions if action.stream == stream}),
            }
        )

    header_mode = _choose_header_mode(
        source_format=bundle.format,
        requested=spec.header_mode,
        contract=runtime_contract,
    )
    if header_mode == "v5":
        changed_fixed = [
            name
            for name in FIXED_V5_LENGTHS
            if len(candidate_segments[name]) != FIXED_V5_LENGTHS[name]
        ]
        if changed_fixed:
            return _candidate_blocked(
                spec.candidate_id,
                "fixed_v5_segment_length_changed",
                "v5 PR85 header cannot encode changed fixed-length bias/region segment sizes",
                changed_fixed_segments=changed_fixed,
            )
    try:
        payload = pack_pr85_bundle(candidate_segments, header_mode=header_mode)
        parsed = parse_pr85_bundle(payload)
    except Pr85BundleError as exc:
        return _candidate_blocked(spec.candidate_id, "candidate_bundle_validation_failed", str(exc))
    if {name: bytes(parsed.segments[name]) for name in SEGMENT_ORDER} != candidate_segments:
        return _candidate_blocked(spec.candidate_id, "candidate_bundle_validation_failed", "reparsed segments do not match candidate segments")

    candidate_dir = out_dir / spec.candidate_id
    archive_path = candidate_dir / "archive.zip"
    _write_single_member_archive(archive_path, payload)
    candidate_archive = _archive_info(archive_path)
    payload_changed = _sha256_bytes(source_raw) != _sha256_bytes(payload)
    changed_segments = [name for name in SEGMENT_ORDER if candidate_segments[name] != source_segments[name]]
    non_noop = bool(payload_changed and changed_segments and any(row["changed"] for row in action_proofs))
    if not non_noop:
        return _candidate_blocked(spec.candidate_id, "non_noop_proof_failed", "payload or decoded side-channel semantics did not change")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool": TOOL,
        "candidate_id": spec.candidate_id,
        "build_status": "built",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "evidence_grade": "empirical_local_archive_build_only",
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "fixed_length_segments": dict(bundle.fixed_length_segments),
            "member_sha256": _sha256_bytes(source_raw),
        },
        "candidate_bundle": {
            "header_mode": header_mode,
            "member_bytes": len(payload),
            "member_sha256": _sha256_bytes(payload),
            "segment_lengths": {name: len(candidate_segments[name]) for name in SEGMENT_ORDER},
        },
        "selected_pair_indices": sorted({action.pair_index for action in spec.actions}),
        "selected_streams": streams,
        "action_proofs": action_proofs,
        "transforms": transform_reports,
        "changed_segments": changed_segments,
        "charged_bytes": {
            "candidate_archive_bytes": candidate_archive["archive_bytes"],
            "source_archive_bytes": source_archive["archive_bytes"],
            "byte_delta_vs_source_archive": int(candidate_archive["archive_bytes"] - source_archive["archive_bytes"]),
            "formula_only_rate_score_delta_vs_source": (
                int(candidate_archive["archive_bytes"] - source_archive["archive_bytes"]) * RATE_SCORE_PER_BYTE
            ),
        },
        "runtime_contract": {
            "schema": runtime_contract.get("schema"),
            "archive_member_contract": runtime_contract.get("archive_member_contract"),
            "supported_streams": runtime_contract.get("supported_streams"),
            "supported_header_modes": runtime_contract.get("supported_header_modes"),
            "scorer_load_allowed": runtime_contract.get("scorer_load_allowed"),
            "sidecars_allowed": runtime_contract.get("sidecars_allowed"),
        },
        "non_noop_proof": {
            "status": "passed",
            "payload_changed": payload_changed,
            "decoded_sidechannel_semantics_changed": True,
            "changed_segments": changed_segments,
            "source_member_sha256": _sha256_bytes(source_raw),
            "candidate_member_sha256": _sha256_bytes(payload),
        },
        "dispatch_unlocked": True,
        "dispatch_gate": "eligible_for_exact_eval_after_lane_claim",
        "lane_claim_required_before_exact_eval": True,
        "next_gate": "Claim the lane with tools/claim_lane_dispatch.py before any exact CUDA auth eval dispatch.",
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def _overall_blockers(*reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for report in reports:
        raw = report.get("blockers")
        if isinstance(raw, list):
            blockers.extend(item for item in raw if isinstance(item, dict))
    return blockers


def build_pair_atom_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    scorer_plan_json: Path = DEFAULT_SCORER_PLAN,
    action_spec_json: Path | None = None,
    runtime_contract_json: Path | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    top_pairs: Sequence[int] = TOP_POLISH_PAIRS,
    require_known_pr85_anchor: bool = True,
    top_limit: int = 8,
    write_outputs: bool = True,
) -> dict[str, Any]:
    source, raw = _read_source_archive(source_archive)
    scorer_report, _scorer_payload = _scorer_plan_report(
        scorer_plan_json,
        source_archive=source,
        expected_pairs=top_pairs,
        require_known_pr85_anchor=require_known_pr85_anchor,
        top_limit=top_limit,
    )
    action_report, specs = _action_spec_report(action_spec_json)
    action_streams = sorted({action.stream for spec in specs for action in spec.actions})
    runtime_report, runtime_contract = _runtime_contract_report(
        runtime_contract_json,
        action_streams=action_streams,
    )
    runtime_probe = _runtime_probe()
    blockers = _overall_blockers(scorer_report, action_report, runtime_report)
    candidates: list[dict[str, Any]] = []
    if not blockers and runtime_contract is not None:
        for spec in specs:
            candidates.append(
                _build_one_candidate(
                    spec=spec,
                    source_archive=source,
                    source_raw=raw,
                    scorer_top_atoms=scorer_report["top_atoms"],
                    runtime_contract=runtime_contract,
                    out_dir=out_dir,
                )
            )
    elif specs:
        for spec in specs:
            candidates.append(
                _candidate_blocked(
                    spec.candidate_id,
                    str(blockers[0]["blocker_class"]) if blockers else "blocked",
                    str(blockers[0]["reason"]) if blockers else "global preflight blocked",
                )
            )

    candidate_blockers = [
        blocker
        for row in candidates
        for blocker in row.get("blockers", [])
        if isinstance(blocker, dict)
    ]
    all_blockers = blockers + candidate_blockers
    dispatch_unlocked_count = sum(1 for row in candidates if row.get("dispatch_unlocked") is True)
    built = [row for row in candidates if row.get("build_status") == "built"]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "sidecars_required": False,
        "source_archive": source,
        "scorer_gradient_plan": scorer_report,
        "action_spec": action_report,
        "runtime_contract": runtime_report,
        "existing_runtime_investigation": runtime_probe,
        "top_polish_pair_indices": [int(pair) for pair in top_pairs],
        "candidate_attempt_count": len(candidates),
        "candidate_archive_count": len(built),
        "dispatch_unlocked_count": dispatch_unlocked_count,
        "dispatch_unlocked": dispatch_unlocked_count > 0,
        "blocker_class": "none" if not all_blockers else str(all_blockers[0]["blocker_class"]),
        "blockers": all_blockers,
        "candidates": candidates,
        "planning_json_path": _rel(out_dir / "planning.json"),
    }
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "planning.json", summary)
    return summary


def render_ledger(summary: Mapping[str, Any]) -> str:
    plan = summary["scorer_gradient_plan"]
    source = summary["source_archive"]
    top_atoms = plan.get("top_atoms", []) if isinstance(plan, Mapping) else []
    lines = [
        "# PR85 Pair-Atom Candidate Readiness",
        "",
        "- tool: `experiments/build_pr85_pair_atom_candidates.py`",
        "- score_claim: false",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        f"- dispatch_unlocked: {str(summary.get('dispatch_unlocked')).lower()}",
        f"- blocker_class: `{summary.get('blocker_class')}`",
        "",
        "## Source Anchor",
        "",
        f"- archive bytes: {source.get('archive_bytes')}",
        f"- archive sha256: `{source.get('archive_sha256')}`",
        f"- known PR85 T4 anchor match: {source.get('known_pr85_anchor_match', {}).get('matches')}",
        "",
        "## Scorer-Gradient Intake",
        "",
        f"- plan: `{plan.get('path')}`",
        f"- plan status: `{plan.get('status')}`",
        f"- stable digest: `{plan.get('stable_plan_digest_sha256')}`",
        "",
        "## Top Pair Opportunities",
        "",
    ]
    if not top_atoms:
        lines.append("- none available from the scorer-gradient plan")
    for row in top_atoms[:8]:
        lines.append(
            "- "
            f"pair_{int(row['pair_index']):04d}: "
            f"break_even_bytes={row.get('break_even_bytes')} "
            f"ranking_score={row.get('ranking_score')}"
        )
    lines.extend(
        [
            "",
            "## Runtime Investigation",
            "",
            "- Existing PR85 bundle code can slice and repack `x`; existing bridge code can materialize `qpost.bin`/`QRM1` and group-level sparse actions.",
            "- Existing final-bias code stacks a coarse 300-byte `fb` atom, not pair-specific stream actions.",
            "- No reviewed pair-action runtime contract was found in the existing PR85 runtime surfaces, and the scorer-gradient plan does not supply stream/value deltas.",
            "",
            "## Readiness Decision",
            "",
        ]
    )
    if summary.get("dispatch_unlocked"):
        lines.append("- Local byte-closed pair-atom candidates were built. Exact eval remains blocked until a lane claim is recorded.")
    else:
        lines.append("- No candidate archive was unlocked from the default PR85 inputs.")
    for blocker in summary.get("blockers", []):
        if isinstance(blocker, Mapping):
            lines.append(f"- blocker `{blocker.get('blocker_class')}`: {blocker.get('reason')}")
    lines.extend(
        [
            "",
            "## Minimal Implementation Needed",
            "",
            "- A compression-time action source that maps each selected pair to explicit PR85 stream/value deltas or to a decoded-output-parity recode.",
            "- A reviewed runtime contract proving those stream families are consumed without scorer loads or sidecars.",
            "- Non-noop payload or decoded-output proof in the candidate manifest.",
            "- `tools/claim_lane_dispatch.py claim ...` before any exact CUDA auth eval dispatch.",
            "",
        ]
    )
    return "\n".join(lines)


def write_ledger(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ledger(summary), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--scorer-plan-json", type=Path, default=DEFAULT_SCORER_PLAN)
    parser.add_argument("--action-spec-json", type=Path)
    parser.add_argument("--runtime-contract-json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--allow-synthetic-source", action="store_true")
    parser.add_argument("--top-limit", type=int, default=8)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_pair_atom_candidates(
        source_archive=args.source_archive,
        scorer_plan_json=args.scorer_plan_json,
        action_spec_json=args.action_spec_json,
        runtime_contract_json=args.runtime_contract_json,
        out_dir=args.out_dir,
        require_known_pr85_anchor=not args.allow_synthetic_source,
        top_limit=args.top_limit,
    )
    write_ledger(args.ledger_md, summary)
    if args.stdout:
        sys.stdout.write(_json_text(summary))
    else:
        print(_json_text({"planning_json": summary["planning_json_path"], "dispatch_unlocked": summary["dispatch_unlocked"], "blocker_class": summary["blocker_class"]}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
