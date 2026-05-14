#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a CMG3A adaptive nonzero-row-run mask candidate archive.

CMG3A keeps the existing CMG3 ``nonzero_row_runs_topk_v1`` wire mode but
selects row-run atoms from a global deterministic priority queue.  Every row
starts with its best nonzero run.  The builder then spends a global extra-run
or approximate compressed-body budget on second/third/etc. row runs, instead
of forcing the same top-K count onto every row.

The output is a byte-screen candidate only.  It is not score evidence until the
exact archive bytes pass CUDA auth eval through the contest path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg3_nonzero_runs_candidate.py"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"

SCHEMA = "cmg3a_adaptive_nonzero_row_runs_candidate_v1"
POLICY_SCHEMA = "cmg3a_adaptive_global_nonzero_row_run_policy_v1"
FIELD_POLICY_SCHEMA = "yousfi_fridrich_atom_field_allocator_v1"
DEFAULT_TARGET_EXTRA_RUNS = 50_000
DEFAULT_BASE_RUNS_PER_ROW = 1
RAW_BYTES_PER_RUN_ATOM = 5.0

DEFAULT_CLASS_WEIGHTS = {
    1: 1.08,
    2: 1.30,
    3: 1.22,
    4: 1.16,
}
DEFAULT_CLASS_WEIGHT_NOTE = (
    "Explicit byte-screen priors only: class-id preference for likely "
    "vehicle/lane/boundary/source-sensitive mask structure. Not score evidence."
)
DEFAULT_BODY_SEARCH_EXACT_LIMIT = 4096
DEFAULT_BODY_SEARCH_COARSE_STEPS = 192
DEFAULT_BODY_SEARCH_LOCAL_RADIUS = 12
BODY_SEARCH_MODES = ("auto", "exhaustive", "coarse")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(BASE_BUILDER_PATH, "_cmg3a_base_nonzero_runs_builder")

HEIGHT = int(BASE.HEIGHT)
WIDTH = int(BASE.WIDTH)
CLASS_COUNT = int(BASE.CLASS_COUNT)
ORIGINAL_VIDEO_BYTES = int(BASE.ORIGINAL_VIDEO_BYTES)
CMG3_MAGIC = BASE.CMG3_MAGIC
CMG3_VERSION = int(BASE.CMG3_VERSION)
CMG3_HEADER_STRUCT = BASE.CMG3_HEADER_STRUCT


@dataclass(frozen=True)
class RunAtom:
    frame_index: int
    y: int
    flat_row: int
    rank: int
    start: int
    end: int
    class_id: int
    length: int
    priority: float

    @property
    def exclusive_end(self) -> int:
        return int(self.end) + 1

    def field_key(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.frame_index),
            int(self.y),
            int(self.start),
            int(self.exclusive_end),
            int(self.class_id),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _extract_frontier_members(frontier_archive: Path) -> dict[str, bytes]:
    unpacker = _load_module(UNPACKER_PATH, "_cmg3a_adaptive_runs_unpacker")
    payload = BASE._read_single_member(frontier_archive)  # noqa: SLF001
    try:
        _header, members = unpacker._parse_payload(payload)  # noqa: SLF001
    except ValueError as raw_parse_error:
        try:
            import brotli
        except ImportError:
            raise
        try:
            _header, members = unpacker._parse_payload(brotli.decompress(payload))  # noqa: SLF001
        except Exception as compressed_parse_error:
            raise raw_parse_error from compressed_parse_error
    required = {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
    missing = required - set(members)
    if missing:
        raise ValueError(f"frontier archive did not unpack required members: {sorted(missing)}")
    return {name: members[name] for name in sorted(required)}


def _as_str_keyed_int_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(int(key)): int(counter[key]) for key in sorted(counter)}


def _center_prior(position: float, size: int) -> float:
    if size <= 1:
        return 1.0
    normalized = (float(position) + 0.5) / float(size)
    return max(0.0, 1.0 - min(1.0, abs(normalized - 0.5) / 0.5))


def _parse_json_or_text_indices(path: Path) -> set[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    values: Any
    if payload is None:
        tokens = text.replace(",", " ").split()
        values = [int(token) for token in tokens]
    elif isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in (
            "hard_frame_indices",
            "frame_indices",
            "frames",
            "hard_pair_indices",
            "pair_indices",
            "pairs",
            "indices",
        ):
            if key in payload:
                values = payload[key]
                break
        else:
            raise ValueError(f"index JSON object in {path} lacks a known index key")
    else:
        raise ValueError(f"unsupported index payload in {path}: {type(payload).__name__}")

    if not isinstance(values, list):
        raise ValueError(f"indices in {path} must be a list")
    out: set[int] = set()
    for value in values:
        if not isinstance(value, int):
            raise ValueError(f"indices in {path} must be integers, got {value!r}")
        if value < 0:
            raise ValueError(f"indices in {path} must be non-negative, got {value}")
        out.add(int(value))
    return out


def load_hard_frame_indices(
    *,
    hard_frame_indices: Path | None = None,
    hard_pair_indices: Path | None = None,
) -> tuple[set[int], dict[str, Any]]:
    frames: set[int] = set()
    sources: list[dict[str, Any]] = []
    if hard_frame_indices is not None:
        loaded = _parse_json_or_text_indices(hard_frame_indices)
        frames.update(loaded)
        sources.append(
            {
                "kind": "frame_indices",
                "path": str(hard_frame_indices.resolve()),
                "sha256": BASE._sha256_file(hard_frame_indices.resolve()),  # noqa: SLF001
                "count": len(loaded),
            }
        )
    if hard_pair_indices is not None:
        pairs = _parse_json_or_text_indices(hard_pair_indices)
        expanded = {2 * pair for pair in pairs} | {2 * pair + 1 for pair in pairs}
        frames.update(expanded)
        sources.append(
            {
                "kind": "pair_indices_expanded_to_frames_2i_2i_plus_1",
                "path": str(hard_pair_indices.resolve()),
                "sha256": BASE._sha256_file(hard_pair_indices.resolve()),  # noqa: SLF001
                "pair_count": len(pairs),
                "expanded_frame_count": len(expanded),
            }
        )
    return frames, {"sources": sources, "frame_count": len(frames), "frames": sorted(frames)}


def parse_class_weights(raw: str | None) -> dict[int, float]:
    if raw is None:
        return dict(DEFAULT_CLASS_WEIGHTS)
    payload_text = Path(raw[1:]).read_text(encoding="utf-8") if raw.startswith("@") else raw
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("--class-weights-json must decode to an object keyed by class id")
    weights: dict[int, float] = {}
    for key, value in payload.items():
        class_id = int(key)
        if not (1 <= class_id < CLASS_COUNT):
            raise ValueError(f"class weight key must be in [1,{CLASS_COUNT}), got {class_id}")
        weight = float(value)
        if weight <= 0.0:
            raise ValueError(f"class weight for {class_id} must be positive, got {weight}")
        weights[class_id] = weight
    for class_id, weight in DEFAULT_CLASS_WEIGHTS.items():
        weights.setdefault(class_id, weight)
    return weights


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_field_policy(payload: dict[str, Any], policy_id: str | None) -> dict[str, Any]:
    if payload.get("schema") != FIELD_POLICY_SCHEMA:
        raise ValueError(
            f"field policy JSON must have schema={FIELD_POLICY_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    policies = payload.get("candidate_policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("field policy JSON must contain non-empty candidate_policies")
    if policy_id is None:
        return policies[0]
    for policy in policies:
        if isinstance(policy, dict) and policy.get("policy_id") == policy_id:
            return policy
    raise ValueError(f"field policy id {policy_id!r} not found")


def load_field_policy(path: Path | None, *, policy_id: str | None = None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    selected = _select_field_policy(payload, policy_id)
    return {
        "source_path": str(path),
        "source_sha256": BASE._sha256_file(path),  # noqa: SLF001
        "source_schema": payload.get("schema"),
        "source_mode": payload.get("mode"),
        "policy_id": selected.get("policy_id"),
        "policy": selected,
    }


def make_policy(
    *,
    base_runs_per_row: int = DEFAULT_BASE_RUNS_PER_ROW,
    adaptive_max_runs_per_row: int,
    target_extra_runs: int | None,
    target_body_bytes: int | None,
    effective_extra_run_cap: int,
    class_weights: dict[int, float] | None = None,
    hard_frame_indices: set[int] | None = None,
    hard_index_manifest: dict[str, Any] | None = None,
    hard_frame_multiplier: float = 1.35,
    foveal_row_weight: float = 0.20,
    foveal_col_weight: float = 0.20,
    boundary_detail_weight: float = 0.08,
    rank_decay: float = 0.92,
) -> dict[str, Any]:
    if not (0 <= base_runs_per_row <= 255):
        raise ValueError(f"base_runs_per_row must be in [0,255], got {base_runs_per_row}")
    if not (1 <= adaptive_max_runs_per_row <= 255):
        raise ValueError(f"adaptive_max_runs_per_row must be in [1,255], got {adaptive_max_runs_per_row}")
    if base_runs_per_row > adaptive_max_runs_per_row:
        raise ValueError(
            f"base_runs_per_row cannot exceed adaptive_max_runs_per_row: "
            f"{base_runs_per_row} > {adaptive_max_runs_per_row}"
        )
    if target_extra_runs is not None and target_extra_runs < 0:
        raise ValueError(f"target_extra_runs must be non-negative, got {target_extra_runs}")
    if target_body_bytes is not None and target_body_bytes <= 0:
        raise ValueError(f"target_body_bytes must be positive, got {target_body_bytes}")
    if hard_frame_multiplier <= 0.0:
        raise ValueError("--hard-frame-multiplier must be positive")
    if rank_decay <= 0.0:
        raise ValueError("--rank-decay must be positive")

    weights = class_weights if class_weights is not None else dict(DEFAULT_CLASS_WEIGHTS)
    hard_frames = sorted(hard_frame_indices or set())
    return {
        "schema": POLICY_SCHEMA,
        "wire_mode": "nonzero_row_runs_topk_v1",
        "selector": "global_weighted_budget_after_base_runs_per_row",
        "base_runs_per_row": int(base_runs_per_row),
        "adaptive_max_runs_per_row": int(adaptive_max_runs_per_row),
        "target_extra_runs": target_extra_runs,
        "target_body_bytes": target_body_bytes,
        "effective_extra_run_cap": int(effective_extra_run_cap),
        "sort_order": "priority_desc_then_frame_row_rank_start_end_class",
        "row_rank_prefix_required": False,
        "raw_bytes_per_extra_atom": RAW_BYTES_PER_RUN_ATOM,
        "priority_formula": (
            "After base_runs_per_row ranked nonzero row runs are kept, extra atoms are ranked by "
            "length * class_weight * (1 + foveal_row_weight*row_center + "
            "foveal_col_weight*col_center) * hard_frame_multiplier_if_selected "
            "* (1 + boundary_detail_weight*short_run_prior) "
            "* rank_decay**max(0, rank-2) / raw_bytes_per_extra_atom"
        ),
        "weights": {
            "class_weights": {str(key): float(weights[key]) for key in sorted(weights)},
            "class_weight_note": DEFAULT_CLASS_WEIGHT_NOTE,
            "hard_frame_multiplier": float(hard_frame_multiplier),
            "foveal_row_weight": float(foveal_row_weight),
            "foveal_col_weight": float(foveal_col_weight),
            "boundary_detail_weight": float(boundary_detail_weight),
            "boundary_reference_run_length": 64,
            "rank_decay": float(rank_decay),
        },
        "hard_frame_indices": {
            **(hard_index_manifest or {"sources": [], "frame_count": len(hard_frames)}),
            "frames": hard_frames,
        },
        "evidence_boundary": (
            "Policy is a deterministic byte-screen prior. It is not scorer, "
            "promotion, or retirement evidence without exact CUDA auth eval."
        ),
    }


def _atom_priority(
    *,
    frame_index: int,
    y: int,
    start: int,
    end: int,
    class_id: int,
    length: int,
    rank: int,
    policy: dict[str, Any],
    hard_frame_indices: set[int],
) -> float:
    weights = policy["weights"]
    class_weights = {int(key): float(value) for key, value in weights["class_weights"].items()}
    class_weight = class_weights.get(int(class_id), 1.0)
    row_center = _center_prior(float(y), HEIGHT)
    col_center = _center_prior((float(start) + float(end)) / 2.0, WIDTH)
    foveal = 1.0 + float(weights["foveal_row_weight"]) * row_center + float(weights["foveal_col_weight"]) * col_center
    hard = float(weights["hard_frame_multiplier"]) if int(frame_index) in hard_frame_indices else 1.0
    boundary_ref = float(weights["boundary_reference_run_length"])
    short_run_prior = max(0.0, 1.0 - min(1.0, float(length) / boundary_ref))
    boundary = 1.0 + float(weights["boundary_detail_weight"]) * short_run_prior
    rank_term = float(weights["rank_decay"]) ** max(0, int(rank) - 2)
    return float(length) * class_weight * foveal * hard * boundary * rank_term / RAW_BYTES_PER_RUN_ATOM


def _ranked_nonzero_runs(row: np.ndarray) -> list[tuple[int, int, int, int]]:
    row = np.asarray(row)
    if row.size == 0:
        return []
    change_points = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change_points))
    stops = np.concatenate((change_points, np.array([row.size], dtype=np.int64)))
    classes = row[starts].astype(np.int64, copy=False)
    nonzero = classes != 0
    runs = [
        (int(start), int(stop) - 1, int(class_id), int(stop - start))
        for start, stop, class_id in zip(starts[nonzero], stops[nonzero], classes[nonzero], strict=True)
    ]
    return sorted(runs, key=lambda item: (-int(item[3]), int(item[0]), int(item[1]), int(item[2])))


def _enumerate_atoms(
    masks: np.ndarray,
    *,
    policy: dict[str, Any],
    hard_frame_indices: set[int],
) -> tuple[dict[int, list[RunAtom]], list[RunAtom], dict[str, Any]]:
    base_runs_per_row = int(policy.get("base_runs_per_row", DEFAULT_BASE_RUNS_PER_ROW))
    adaptive_max_runs_per_row = int(policy["adaptive_max_runs_per_row"])
    base_by_row: dict[int, list[RunAtom]] = {}
    candidates: list[RunAtom] = []
    rank_histogram: Counter[int] = Counter()
    total_nonzero_runs = 0
    max_available_runs_per_row = 0

    for frame_index, frame in enumerate(masks):
        for y, row in enumerate(frame):
            flat_row = frame_index * HEIGHT + y
            ranked_runs = _ranked_nonzero_runs(row)
            total_nonzero_runs += len(ranked_runs)
            max_available_runs_per_row = max(max_available_runs_per_row, len(ranked_runs))
            for rank, (start, end, class_id, length) in enumerate(ranked_runs, start=1):
                rank_histogram[rank] += 1
                if rank > adaptive_max_runs_per_row:
                    continue
                atom = RunAtom(
                    frame_index=int(frame_index),
                    y=int(y),
                    flat_row=int(flat_row),
                    rank=int(rank),
                    start=int(start),
                    end=int(end),
                    class_id=int(class_id),
                    length=int(length),
                    priority=_atom_priority(
                        frame_index=int(frame_index),
                        y=int(y),
                        start=int(start),
                        end=int(end),
                        class_id=int(class_id),
                        length=int(length),
                        rank=int(rank),
                        policy=policy,
                        hard_frame_indices=hard_frame_indices,
                    ),
                )
                if rank <= base_runs_per_row:
                    base_by_row.setdefault(flat_row, []).append(atom)
                else:
                    candidates.append(atom)

    candidates.sort(
        key=lambda atom: (
            -atom.priority,
            atom.frame_index,
            atom.y,
            atom.rank,
            atom.start,
            atom.end,
            atom.class_id,
        )
    )
    stats = {
        "total_nonzero_runs": int(total_nonzero_runs),
        "base_runs_per_row": int(base_runs_per_row),
        "base_required_runs": int(sum(len(v) for v in base_by_row.values())),
        "base_top1_runs": int(sum(1 for atoms in base_by_row.values() for atom in atoms if atom.rank == 1)),
        "candidate_extra_runs": int(len(candidates)),
        "max_available_runs_per_row": int(max_available_runs_per_row),
        "available_rank_histogram": _as_str_keyed_int_counter(rank_histogram),
    }
    return base_by_row, candidates, stats


def _materialize_selection(
    base_by_row: dict[int, list[RunAtom]],
    extra_atoms: list[RunAtom],
) -> dict[int, list[RunAtom]]:
    selected = {flat_row: list(atoms) for flat_row, atoms in base_by_row.items()}
    for atom in extra_atoms:
        selected.setdefault(atom.flat_row, []).append(atom)
    return selected


def _requested_field_atom_keys(field_atom_policy: dict[str, Any]) -> list[tuple[int, int, int, int, int]]:
    policy = field_atom_policy.get("policy", field_atom_policy)
    raw_atoms = policy.get("selected_row_run_atoms")
    if not isinstance(raw_atoms, list):
        raise ValueError("field atom policy must contain selected_row_run_atoms")
    out: list[tuple[int, int, int, int, int]] = []
    for index, raw in enumerate(raw_atoms):
        if not isinstance(raw, dict):
            raise ValueError(f"selected_row_run_atoms[{index}] must be an object")
        try:
            key = (
                int(raw["frame_index"]),
                int(raw["y"]),
                int(raw["x0"]),
                int(raw["x1_exclusive"]),
                int(raw["class_id"]),
            )
        except KeyError as exc:
            raise ValueError(f"selected_row_run_atoms[{index}] missing key {exc.args[0]!r}") from exc
        frame_index, y, x0, x1_exclusive, class_id = key
        if frame_index < 0 or y < 0 or x0 < 0 or x1_exclusive <= x0:
            raise ValueError(f"selected_row_run_atoms[{index}] has invalid bounds: {raw!r}")
        if class_id <= 0 or class_id >= CLASS_COUNT:
            raise ValueError(f"selected_row_run_atoms[{index}] has invalid class_id: {class_id}")
        out.append(key)
    return out


def _field_policy_required_base_runs_per_row(field_atom_policy: dict[str, Any]) -> int | None:
    policy = field_atom_policy.get("policy", field_atom_policy)
    for key in ("required_base_runs_per_row", "expected_base_runs_per_row"):
        value = policy.get(key)
        if value is not None:
            out = int(value)
            if out < 0:
                raise ValueError(f"field policy {key} must be nonnegative, got {value!r}")
            return out
    return None


def _short_policy_source(path_value: Any) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    return Path(path_value).name


def _field_policy_extra_atoms(
    *,
    base_by_row: dict[int, list[RunAtom]],
    candidates: list[RunAtom],
    field_atom_policy: dict[str, Any],
) -> tuple[list[RunAtom], dict[str, Any]]:
    candidate_by_key = {atom.field_key(): atom for atom in candidates}
    base_keys = {atom.field_key() for atoms in base_by_row.values() for atom in atoms}
    requested = _requested_field_atom_keys(field_atom_policy)
    selected: list[RunAtom] = []
    unmatched: list[dict[str, int]] = []
    already_base: list[dict[str, int]] = []
    duplicate_selected = 0
    selected_keys: set[tuple[int, int, int, int, int]] = set()

    for key in requested:
        if key in selected_keys:
            duplicate_selected += 1
            continue
        if key in base_keys:
            already_base.append(
                {
                    "frame_index": key[0],
                    "y": key[1],
                    "x0": key[2],
                    "x1_exclusive": key[3],
                    "class_id": key[4],
                }
            )
            selected_keys.add(key)
            continue
        atom = candidate_by_key.get(key)
        if atom is None:
            unmatched.append(
                {
                    "frame_index": key[0],
                    "y": key[1],
                    "x0": key[2],
                    "x1_exclusive": key[3],
                    "class_id": key[4],
                }
            )
            continue
        selected.append(atom)
        selected_keys.add(key)

    if duplicate_selected:
        raise ValueError(f"field policy contains {duplicate_selected} duplicate row-run atom selections")
    if unmatched:
        raise ValueError(
            f"field policy contains {len(unmatched)} row-run atoms that do not match the selected base/candidate pool; "
            f"first unmatched={unmatched[0]}"
        )

    selected.sort(
        key=lambda atom: (
            requested.index(atom.field_key()),
            atom.frame_index,
            atom.y,
            atom.start,
            atom.end,
            atom.class_id,
        )
    )
    return selected, {
        "source_basename": _short_policy_source(field_atom_policy.get("source_path")),
        "source_sha256": field_atom_policy.get("source_sha256"),
        "source_schema": field_atom_policy.get("source_schema"),
        "source_mode": field_atom_policy.get("source_mode"),
        "policy_id": field_atom_policy.get("policy_id"),
        "expected_base_runs_per_row": _field_policy_required_base_runs_per_row(field_atom_policy),
        "requested_row_run_atom_count": len(requested),
        "matched_extra_atom_count": len(selected),
        "already_base_atom_count": len(already_base),
        "duplicate_selected_atom_count": int(duplicate_selected),
        "unmatched_atom_count": len(unmatched),
        "already_base_atoms": already_base[:32],
        "unmatched_atoms": unmatched[:64],
        "contract": (
            "selected_row_run_atoms identify source row-run atoms by "
            "frame_index,y,x0,x1_exclusive,class_id. Atoms already present in "
            "the mandatory base_runs_per_row prefix are recorded and not re-encoded."
        ),
    }


def _encode_selected_run_stream(
    *,
    frame_count: int,
    selected_by_row: dict[int, list[RunAtom]],
) -> bytes:
    out = bytearray()
    row_count = int(frame_count) * HEIGHT
    for flat_row in range(row_count):
        selected = sorted(
            selected_by_row.get(flat_row, []),
            key=lambda atom: (atom.start, atom.end, atom.class_id, atom.rank),
        )
        if len(selected) > 255:
            raise ValueError(f"selected row has {len(selected)} runs, exceeds u8 record count")
        out.append(len(selected))
        previous_end = -1
        for atom in selected:
            if atom.start <= previous_end:
                raise ValueError(f"selected runs overlap or are out of order in flat row {flat_row}")
            previous_end = atom.end
            out.append(int(atom.class_id))
            out += int(atom.start).to_bytes(2, "little")
            out += int(atom.end).to_bytes(2, "little")
    return bytes(out)


def _reconstruct_from_selection(
    *,
    masks: np.ndarray,
    selected_by_row: dict[int, list[RunAtom]],
) -> np.ndarray:
    recon = np.zeros_like(masks)
    for flat_row, atoms in selected_by_row.items():
        frame_index = flat_row // HEIGHT
        y = flat_row % HEIGHT
        for atom in atoms:
            recon[frame_index, y, atom.start : atom.end + 1] = np.uint8(atom.class_id)
    return recon


def _selection_stats(
    *,
    masks: np.ndarray,
    recon: np.ndarray,
    selected_by_row: dict[int, list[RunAtom]],
    base_by_row: dict[int, list[RunAtom]],
    selected_extra_atoms: list[RunAtom],
    atom_stats: dict[str, Any],
    body_search: dict[str, Any],
) -> dict[str, Any]:
    row_count_histogram: Counter[int] = Counter()
    rank_histogram: Counter[int] = Counter()
    class_histogram: Counter[int] = Counter()
    extra_rank_histogram: Counter[int] = Counter()
    extra_class_histogram: Counter[int] = Counter()
    rows_with_extra = 0
    max_selected_runs_per_row = 0

    for flat_row in range(int(masks.shape[0]) * HEIGHT):
        atoms = selected_by_row.get(flat_row, [])
        count = len(atoms)
        row_count_histogram[count] += 1
        max_selected_runs_per_row = max(max_selected_runs_per_row, count)
        if count > len(base_by_row.get(flat_row, [])):
            rows_with_extra += 1
        for atom in atoms:
            rank_histogram[atom.rank] += 1
            class_histogram[atom.class_id] += 1
    for atom in selected_extra_atoms:
        extra_rank_histogram[atom.rank] += 1
        extra_class_histogram[atom.class_id] += 1

    diff = recon != masks
    selected_run_count = int(sum(len(v) for v in selected_by_row.values()))
    base_run_count = int(sum(len(v) for v in base_by_row.values()))
    return {
        **atom_stats,
        "selected_runs": selected_run_count,
        "selected_extra_runs": int(len(selected_extra_atoms)),
        "selected_run_fraction": float(selected_run_count / atom_stats["total_nonzero_runs"])
        if atom_stats["total_nonzero_runs"]
        else 1.0,
        "base_required_runs": base_run_count,
        "base_top1_runs": int(atom_stats.get("base_top1_runs", base_run_count)),
        "rows_with_extra_runs": int(rows_with_extra),
        "max_selected_runs_per_row": int(max_selected_runs_per_row),
        "selected_rows_by_run_count": _as_str_keyed_int_counter(row_count_histogram),
        "selected_rank_histogram": _as_str_keyed_int_counter(rank_histogram),
        "selected_class_histogram": _as_str_keyed_int_counter(class_histogram),
        "selected_extra_rank_histogram": _as_str_keyed_int_counter(extra_rank_histogram),
        "selected_extra_class_histogram": _as_str_keyed_int_counter(extra_class_histogram),
        "selected_extra_priority_max": float(selected_extra_atoms[0].priority) if selected_extra_atoms else None,
        "selected_extra_priority_min": float(selected_extra_atoms[-1].priority) if selected_extra_atoms else None,
        "pixel_disagreement": float(diff.mean()),
        "pixel_disagreement_count": int(diff.sum()),
        "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
        "body_search": body_search,
    }


def _compressed_body_bytes(run_stream: bytes, compressor: str) -> int:
    _selected, body, _report = BASE._choose_compressor(run_stream, compressor)  # noqa: SLF001
    return len(body)


def _positive_int(value: int | None, *, name: str, default: int) -> int:
    selected = default if value is None else int(value)
    if selected <= 0:
        raise ValueError(f"{name} must be positive, got {selected}")
    return selected


def _select_extra_prefix_count(
    *,
    masks: np.ndarray,
    base_by_row: dict[int, list[RunAtom]],
    candidates: list[RunAtom],
    effective_extra_run_cap: int,
    target_body_bytes: int | None,
    compressor: str,
    body_search_mode: str = "auto",
    body_search_exact_limit: int | None = None,
    body_search_coarse_steps: int | None = None,
    body_search_local_radius: int | None = None,
) -> tuple[int, dict[str, Any]]:
    capped = min(int(effective_extra_run_cap), len(candidates))
    if target_body_bytes is None:
        return capped, {
            "target_body_bytes": None,
            "selected_by": "target_extra_runs_only",
            "effective_extra_run_cap": capped,
        }

    mode = str(body_search_mode)
    if mode not in BODY_SEARCH_MODES:
        raise ValueError(f"body_search_mode must be one of {BODY_SEARCH_MODES}, got {mode!r}")
    exact_limit = _positive_int(
        body_search_exact_limit,
        name="body_search_exact_limit",
        default=DEFAULT_BODY_SEARCH_EXACT_LIMIT,
    )
    coarse_steps = _positive_int(
        body_search_coarse_steps,
        name="body_search_coarse_steps",
        default=DEFAULT_BODY_SEARCH_COARSE_STEPS,
    )
    local_radius = _positive_int(
        body_search_local_radius,
        name="body_search_local_radius",
        default=DEFAULT_BODY_SEARCH_LOCAL_RADIUS,
    )
    target = int(target_body_bytes)
    evaluations: list[dict[str, int]] = []
    body_cache: dict[int, int] = {}

    def body_for(count: int) -> int:
        count = int(count)
        if count in body_cache:
            return body_cache[count]
        selected_by_row = _materialize_selection(base_by_row, candidates[:count])
        run_stream = _encode_selected_run_stream(frame_count=int(masks.shape[0]), selected_by_row=selected_by_row)
        body_bytes = _compressed_body_bytes(run_stream, compressor)
        body_cache[count] = int(body_bytes)
        evaluations.append({"extra_runs": int(count), "body_bytes": int(body_bytes)})
        return int(body_bytes)

    def evaluate_counts(counts: set[int]) -> None:
        for count in sorted(int(v) for v in counts if 0 <= int(v) <= capped):
            body_for(count)

    if mode == "exhaustive" or (mode == "auto" and capped <= exact_limit):
        evaluate_counts(set(range(capped + 1)))
        selected_by = "largest_exhaustive_priority_prefix_under_target_body_bytes"
        exhaustive = True
        coarse_step = None
    else:
        coarse_step = max(1, int(math.ceil(capped / float(coarse_steps))))
        initial_counts = {0, capped}
        initial_counts.update(range(0, capped + 1, coarse_step))
        evaluate_counts(initial_counts)

        sorted_initial = sorted(initial_counts)
        refinement_centers: set[int] = set()
        ranked_by_target_distance = sorted(
            body_cache,
            key=lambda count: (abs(body_cache[count] - target), -count),
        )
        refinement_centers.update(ranked_by_target_distance[: min(16, len(ranked_by_target_distance))])

        for left, right in zip(sorted_initial, sorted_initial[1:]):
            left_feasible = body_cache[left] <= target
            right_feasible = body_cache[right] <= target
            if left_feasible != right_feasible:
                refinement_centers.update((left, right))

        feasible_initial = [count for count, body in body_cache.items() if body <= target]
        if feasible_initial:
            refinement_centers.add(max(feasible_initial))

        refinement_counts: set[int] = set()
        for center in refinement_centers:
            start = max(0, int(center) - local_radius)
            stop = min(capped, int(center) + local_radius)
            refinement_counts.update(range(start, stop + 1))
        evaluate_counts(refinement_counts)
        selected_by = "largest_sampled_priority_prefix_under_target_body_bytes"
        exhaustive = False

    baseline_body = body_cache.get(0, body_for(0))
    cap_body = body_cache.get(capped, body_for(capped))
    feasible = [(count, body) for count, body in body_cache.items() if body <= target]
    if feasible:
        best, best_body = max(feasible, key=lambda item: (item[0], -item[1]))
        target_met = True
    else:
        best, best_body = min(body_cache.items(), key=lambda item: (abs(item[1] - target), item[0]))
        target_met = False
        selected_by = "no_evaluated_priority_prefix_fits_target_body_bytes"

    return best, {
        "target_body_bytes": int(target),
        "selected_by": selected_by,
        "target_met": bool(target_met),
        "baseline_body_bytes": int(baseline_body),
        "cap_body_bytes": int(cap_body),
        "selected_body_bytes_during_search": int(best_body),
        "effective_extra_run_cap": capped,
        "body_search_mode": mode,
        "exhaustive_prefix_search": bool(exhaustive),
        "evaluated_prefix_count": len(body_cache),
        "unevaluated_prefix_count": int(max(0, capped + 1 - len(body_cache))),
        "coarse_step": coarse_step,
        "exact_limit": int(exact_limit),
        "coarse_steps": int(coarse_steps),
        "local_radius": int(local_radius),
        "evaluations": evaluations,
        "monotonic_binary_search": False,
        "note": (
            "Compressed body size is explicitly treated as nonmonotonic over "
            "priority prefixes. Exhaustive mode is exact over all prefixes; "
            "coarse mode records sampled-prefix coverage and must not be "
            "described as an exact optimizer."
        ),
    }


def encode_adaptive_run_stream(
    masks: np.ndarray,
    *,
    base_runs_per_row: int = DEFAULT_BASE_RUNS_PER_ROW,
    target_extra_runs: int | None = None,
    target_body_bytes: int | None = None,
    adaptive_max_runs_per_row: int = 8,
    compressor: str = "auto",
    class_weights: dict[int, float] | None = None,
    hard_frame_indices: set[int] | None = None,
    hard_index_manifest: dict[str, Any] | None = None,
    hard_frame_multiplier: float = 1.35,
    foveal_row_weight: float = 0.20,
    foveal_col_weight: float = 0.20,
    boundary_detail_weight: float = 0.08,
    rank_decay: float = 0.92,
    field_atom_policy: dict[str, Any] | None = None,
    body_search_mode: str = "auto",
    body_search_exact_limit: int | None = None,
    body_search_coarse_steps: int | None = None,
    body_search_local_radius: int | None = None,
) -> tuple[bytes, np.ndarray, dict[str, Any], dict[str, Any]]:
    if masks.ndim != 3 or masks.dtype != np.uint8:
        raise ValueError(f"masks must be uint8 rank-3, got shape={masks.shape} dtype={masks.dtype}")
    if masks.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"masks must be {HEIGHT}x{WIDTH}, got {masks.shape[1:]}")
    if int(masks.min()) < 0 or int(masks.max()) >= CLASS_COUNT:
        raise ValueError(f"masks classes must be in [0,{CLASS_COUNT})")
    hard_frames = set(hard_frame_indices or set())

    effective_extra_run_cap = (
        DEFAULT_TARGET_EXTRA_RUNS
        if target_extra_runs is None and target_body_bytes is None
        else (10**18 if target_extra_runs is None else int(target_extra_runs))
    )
    policy = make_policy(
        base_runs_per_row=base_runs_per_row,
        adaptive_max_runs_per_row=adaptive_max_runs_per_row,
        target_extra_runs=target_extra_runs,
        target_body_bytes=target_body_bytes,
        effective_extra_run_cap=min(effective_extra_run_cap, 2**63 - 1),
        class_weights=class_weights,
        hard_frame_indices=hard_frames,
        hard_index_manifest=hard_index_manifest,
        hard_frame_multiplier=hard_frame_multiplier,
        foveal_row_weight=foveal_row_weight,
        foveal_col_weight=foveal_col_weight,
        boundary_detail_weight=boundary_detail_weight,
        rank_decay=rank_decay,
    )
    base_by_row, candidates, atom_stats = _enumerate_atoms(masks, policy=policy, hard_frame_indices=hard_frames)
    field_policy_report = None
    candidate_pool = candidates
    if field_atom_policy is not None:
        required_base = _field_policy_required_base_runs_per_row(field_atom_policy)
        if required_base is not None and int(required_base) != int(base_runs_per_row):
            raise ValueError(
                f"field policy requires base_runs_per_row={required_base}, "
                f"but builder was called with base_runs_per_row={base_runs_per_row}"
            )
        candidate_pool, field_policy_report = _field_policy_extra_atoms(
            base_by_row=base_by_row,
            candidates=candidates,
            field_atom_policy=field_atom_policy,
        )
        policy["selector"] = "field_equation_explicit_row_run_policy_after_base_runs_per_row"
        policy["field_atom_policy"] = field_policy_report

    selected_extra_count, body_search = _select_extra_prefix_count(
        masks=masks,
        base_by_row=base_by_row,
        candidates=candidate_pool,
        effective_extra_run_cap=min(int(effective_extra_run_cap), len(candidate_pool)),
        target_body_bytes=target_body_bytes,
        compressor=compressor,
        body_search_mode=body_search_mode,
        body_search_exact_limit=body_search_exact_limit,
        body_search_coarse_steps=body_search_coarse_steps,
        body_search_local_radius=body_search_local_radius,
    )
    selected_extra_atoms = candidate_pool[:selected_extra_count]
    selected_by_row = _materialize_selection(base_by_row, selected_extra_atoms)
    run_stream = _encode_selected_run_stream(frame_count=int(masks.shape[0]), selected_by_row=selected_by_row)
    recon = _reconstruct_from_selection(masks=masks, selected_by_row=selected_by_row)
    stats = _selection_stats(
        masks=masks,
        recon=recon,
        selected_by_row=selected_by_row,
        base_by_row=base_by_row,
        selected_extra_atoms=selected_extra_atoms,
        atom_stats=atom_stats,
        body_search=body_search,
    )
    policy["effective_extra_run_cap"] = int(min(int(effective_extra_run_cap), len(candidates)))
    policy["selected_extra_runs"] = int(selected_extra_count)
    policy["selected_max_runs_per_row"] = int(stats["max_selected_runs_per_row"])
    if field_policy_report is not None:
        policy["effective_field_candidate_pool"] = {
            "candidate_extra_runs_available": int(len(candidate_pool)),
            "selected_extra_runs_from_field_pool": int(selected_extra_count),
        }
    return run_stream, recon, stats, policy


def encode_cmg3a_payload(
    run_stream: bytes,
    *,
    frame_count: int,
    max_runs_per_row: int,
    source_mask_sha256: str,
    reconstructed_mask_sha256: str,
    pixel_disagreement: float,
    pixel_disagreement_count: int,
    policy: dict[str, Any],
    compressor: str = "auto",
) -> tuple[bytes, dict[str, Any]]:
    selected_compressor, body, compression_report = BASE._choose_compressor(run_stream, compressor)  # noqa: SLF001
    header = {
        "schema": SCHEMA,
        "mode": "nonzero_row_runs_topk_v1",
        "selector": "cmg3a_adaptive_global_budget",
        "compressor": selected_compressor,
        "run_stream_bytes": len(run_stream),
        "run_stream_sha256": _sha256_bytes(run_stream),
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "compression_candidates": compression_report,
        "frame_count": int(frame_count),
        "height": HEIGHT,
        "width": WIDTH,
        "class_count": CLASS_COUNT,
        "default_class": 0,
        "max_runs_per_row": int(max_runs_per_row),
        "record_struct": "u8_count_then_u8_class_u16_start_u16_end_le",
        "source_mask_u8_sha256": source_mask_sha256,
        "reconstructed_mask_u8_sha256": reconstructed_mask_sha256,
        "pixel_disagreement_vs_source": float(pixel_disagreement),
        "pixel_disagreement_count_vs_source": int(pixel_disagreement_count),
        "lossless_under_builder": bool(pixel_disagreement_count == 0),
        "selection_policy": policy,
        "score_claim": False,
        "promotion_eligible": False,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload = (
        struct.pack(CMG3_HEADER_STRUCT, CMG3_MAGIC, CMG3_VERSION, len(header_bytes))
        + header_bytes
        + body
    )
    return payload, header


def build_candidate(
    *,
    frontier_archive: Path,
    decoded_mask_array: Path,
    output_dir: Path,
    base_runs_per_row: int = DEFAULT_BASE_RUNS_PER_ROW,
    target_extra_runs: int | None = None,
    target_body_bytes: int | None = None,
    adaptive_max_runs_per_row: int = 8,
    compressor: str = "auto",
    hard_frame_indices: set[int] | None = None,
    hard_index_manifest: dict[str, Any] | None = None,
    class_weights: dict[int, float] | None = None,
    hard_frame_multiplier: float = 1.35,
    foveal_row_weight: float = 0.20,
    foveal_col_weight: float = 0.20,
    boundary_detail_weight: float = 0.08,
    rank_decay: float = 0.92,
    field_atom_policy: dict[str, Any] | None = None,
    body_search_mode: str = "auto",
    body_search_exact_limit: int | None = None,
    body_search_coarse_steps: int | None = None,
    body_search_local_radius: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _extract_frontier_members(frontier_archive.resolve())
    masks = BASE._load_decoded_masks(decoded_mask_array.resolve())  # noqa: SLF001
    source_mask_sha = _sha256_bytes(masks.tobytes(order="C"))
    run_stream, recon, run_stats, policy = encode_adaptive_run_stream(
        masks,
        base_runs_per_row=base_runs_per_row,
        target_extra_runs=target_extra_runs,
        target_body_bytes=target_body_bytes,
        adaptive_max_runs_per_row=adaptive_max_runs_per_row,
        compressor=compressor,
        class_weights=class_weights,
        hard_frame_indices=hard_frame_indices,
        hard_index_manifest=hard_index_manifest,
        hard_frame_multiplier=hard_frame_multiplier,
        foveal_row_weight=foveal_row_weight,
        foveal_col_weight=foveal_col_weight,
        boundary_detail_weight=boundary_detail_weight,
        rank_decay=rank_decay,
        field_atom_policy=field_atom_policy,
        body_search_mode=body_search_mode,
        body_search_exact_limit=body_search_exact_limit,
        body_search_coarse_steps=body_search_coarse_steps,
        body_search_local_radius=body_search_local_radius,
    )
    recon_sha = _sha256_bytes(recon.tobytes(order="C"))
    cmg3_payload, cmg3_header = encode_cmg3a_payload(
        run_stream,
        frame_count=int(masks.shape[0]),
        max_runs_per_row=int(run_stats["max_selected_runs_per_row"]),
        source_mask_sha256=source_mask_sha,
        reconstructed_mask_sha256=recon_sha,
        pixel_disagreement=float(run_stats["pixel_disagreement"]),
        pixel_disagreement_count=int(run_stats["pixel_disagreement_count"]),
        policy=policy,
        compressor=compressor,
    )

    source_archive = output_dir / "cmg3a_adaptive_runs_source_members.zip"
    BASE._write_source_archive(  # noqa: SLF001
        source_archive,
        [
            ("renderer.bin", members["renderer.bin"]),
            ("masks.cmg3", cmg3_payload),
            ("optimized_poses.bin", members["optimized_poses.bin"]),
        ],
    )

    packer = _load_module(PACKER_PATH, "_cmg3a_adaptive_runs_packer")
    archive_path = output_dir / "archive.zip"
    packed_meta = packer.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=packer.POSE_QP1_CODEC,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    archive_size = archive_path.stat().st_size
    frontier_size = frontier_archive.stat().st_size
    delta_bytes = archive_size - frontier_size
    manifest = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_screen_archive_candidate_until_exact_cuda",
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "frontier_archive": {
            "path": str(frontier_archive.resolve()),
            "bytes": frontier_size,
            "sha256": BASE._sha256_file(frontier_archive.resolve()),  # noqa: SLF001
        },
        "decoded_mask_array": {
            "path": str(decoded_mask_array.resolve()),
            "npy_sha256": BASE._sha256_file(decoded_mask_array.resolve()),  # noqa: SLF001
            "tensor_sha256": source_mask_sha,
            "shape": [int(v) for v in masks.shape],
        },
        "policy": policy,
        "cmg3": {
            **cmg3_header,
            "payload_bytes": len(cmg3_payload),
            "payload_sha256": _sha256_bytes(cmg3_payload),
            "run_stats": run_stats,
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": BASE._sha256_file(source_archive),  # noqa: SLF001
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_size,
            "sha256": BASE._sha256_file(archive_path),  # noqa: SLF001
            "delta_bytes_vs_frontier": delta_bytes,
            "formula_only_rate_delta_vs_frontier": 25.0 * delta_bytes / ORIGINAL_VIDEO_BYTES,
        },
        "packed_payload": packed_meta,
        "required_next_steps": [
            "local unpack/inflate smoke to prove masks.cmg3 routes through runtime",
            "exact CUDA diagnostic on the archive bytes",
            "T4 promotion only if component gates are near or below C-067",
        ],
    }
    (output_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-archive", type=Path, required=True)
    parser.add_argument("--decoded-mask-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-extra-runs", type=int, default=None)
    parser.add_argument("--target-body-bytes", type=int, default=None)
    parser.add_argument(
        "--base-runs-per-row",
        type=int,
        default=int(os.environ.get("PACT_CMG3A_BASE_RUNS_PER_ROW", DEFAULT_BASE_RUNS_PER_ROW)),
        help="Ranked nonzero runs kept in every row before adaptive/field-selected extra atoms.",
    )
    parser.add_argument("--adaptive-max-runs-per-row", type=int, default=8)
    parser.add_argument("--compressor", choices=("auto", "bz2", "raw", "zlib", "lzma_xz"), default="auto")
    parser.add_argument("--hard-frame-indices", type=Path, default=None)
    parser.add_argument("--hard-pair-indices", type=Path, default=None)
    parser.add_argument("--class-weights-json", default=None)
    parser.add_argument("--hard-frame-multiplier", type=float, default=1.35)
    parser.add_argument("--foveal-row-weight", type=float, default=0.20)
    parser.add_argument("--foveal-col-weight", type=float, default=0.20)
    parser.add_argument("--boundary-detail-weight", type=float, default=0.08)
    parser.add_argument("--rank-decay", type=float, default=0.92)
    parser.add_argument(
        "--body-search-mode",
        choices=BODY_SEARCH_MODES,
        default=os.environ.get("PACT_CMG3A_BODY_SEARCH_MODE", "auto"),
        help=(
            "Prefix search used with --target-body-bytes. auto uses exhaustive "
            "scan below --body-search-exact-limit and sampled nonmonotonic "
            "coarse search above it."
        ),
    )
    parser.add_argument(
        "--body-search-exact-limit",
        type=int,
        default=int(os.environ.get("PACT_CMG3A_BODY_SEARCH_EXACT_LIMIT", DEFAULT_BODY_SEARCH_EXACT_LIMIT)),
    )
    parser.add_argument(
        "--body-search-coarse-steps",
        type=int,
        default=int(os.environ.get("PACT_CMG3A_BODY_SEARCH_COARSE_STEPS", DEFAULT_BODY_SEARCH_COARSE_STEPS)),
    )
    parser.add_argument(
        "--body-search-local-radius",
        type=int,
        default=int(os.environ.get("PACT_CMG3A_BODY_SEARCH_LOCAL_RADIUS", DEFAULT_BODY_SEARCH_LOCAL_RADIUS)),
    )
    parser.add_argument(
        "--field-policy-json",
        type=Path,
        default=None,
        help="Optional Yousfi-Fridrich field-equation policy JSON selecting explicit row-run atoms.",
    )
    parser.add_argument(
        "--field-policy-id",
        default=None,
        help="Policy id inside --field-policy-json. Defaults to the first candidate policy.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hard_frames, hard_manifest = load_hard_frame_indices(
        hard_frame_indices=args.hard_frame_indices,
        hard_pair_indices=args.hard_pair_indices,
    )
    manifest = build_candidate(
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        output_dir=args.output_dir,
        base_runs_per_row=args.base_runs_per_row,
        target_extra_runs=args.target_extra_runs,
        target_body_bytes=args.target_body_bytes,
        adaptive_max_runs_per_row=args.adaptive_max_runs_per_row,
        compressor=args.compressor,
        hard_frame_indices=hard_frames,
        hard_index_manifest=hard_manifest,
        class_weights=parse_class_weights(args.class_weights_json),
        hard_frame_multiplier=args.hard_frame_multiplier,
        foveal_row_weight=args.foveal_row_weight,
        foveal_col_weight=args.foveal_col_weight,
        boundary_detail_weight=args.boundary_detail_weight,
        rank_decay=args.rank_decay,
        field_atom_policy=load_field_policy(args.field_policy_json, policy_id=args.field_policy_id),
        body_search_mode=args.body_search_mode,
        body_search_exact_limit=args.body_search_exact_limit,
        body_search_coarse_steps=args.body_search_coarse_steps,
        body_search_local_radius=args.body_search_local_radius,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "archive": manifest["output_archive"],
                "pixel_disagreement": manifest["cmg3"]["run_stats"]["pixel_disagreement"],
                "selected_extra_runs": manifest["cmg3"]["run_stats"]["selected_extra_runs"],
                "max_runs_per_row": manifest["cmg3"]["max_runs_per_row"],
                "compressor": manifest["cmg3"]["compressor"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
