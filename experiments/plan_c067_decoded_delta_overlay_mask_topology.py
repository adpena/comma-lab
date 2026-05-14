#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local-only C067 decoded-baseline delta/overlay mask candidates.

This planner keeps the C067 decoded mask tensor as the global geometry source
of truth and considers donor mask tensors only as local overlay proposals.  It
does not build a scoring archive or launch remote work.  A payload spec is
written only when every changed pixel is inside an explicit pair/class trust
region and no selected pair appears in the recent exact-negative catastrophic
pair set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c067_decoded_delta_overlay_mask_topology.py"
SCHEMA = "c067_decoded_delta_overlay_mask_topology_plan_v1"
PAYLOAD_SCHEMA = "c067_decoded_delta_overlay_payload_v1"
PAYLOAD_MAGIC = b"CDO1"
PAYLOAD_VERSION = 1
PAYLOAD_HEADER_STRUCT = struct.Struct("<4sHI")
RUN_STRUCT = struct.Struct("<HHHHB")
RUN_STRUCT_NAME = "u16_frame_u16_y_u16_x0_u16_length_u8_value_le"
PAIR_INDEX_BASIS_AUTO = "auto"
PAIR_INDEX_BASIS_HALF_FRAME = "half_frame_pair_index"
PAIR_INDEX_BASIS_VIDEO_FRAME = "video_frame_pair_index"
PAIR_INDEX_BASIS_CHOICES = (
    PAIR_INDEX_BASIS_AUTO,
    PAIR_INDEX_BASIS_HALF_FRAME,
    PAIR_INDEX_BASIS_VIDEO_FRAME,
)

ORIGINAL_VIDEO_BYTES = 37_545_489
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = (
    "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
)
C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE = 252_760
DEFAULT_MAX_PIXEL_DISAGREEMENT_FRACTION = 0.0010
DEFAULT_MAX_SELECTED_PIXELS = 32_768
DEFAULT_MAX_COMPRESSED_PAYLOAD_BYTES = 16_384
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

DEFAULT_BASE_MASK_ARRAY = (
    REPO_ROOT
    / "experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/"
    "decoded_mask_array.npy"
)
DEFAULT_GEOMETRY_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/c067_geometry_safe_mask_topology_v2_20260502/"
    "c067_geometry_safe_mask_topology_v2_plan.json"
)
DEFAULT_POSTDECODE_TRUST_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "c067_postdecode_mask_repair_waterfill_pair_class_plan.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c067_decoded_delta_overlay_mask_topology_20260502"
)
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "c067_decoded_delta_overlay_mask_topology_plan.json"
DEFAULT_CANDIDATE_MASKS: tuple[tuple[str, Path], ...] = (
    (
        "cmg3a_body200",
        REPO_ROOT
        / "experiments/results/c067_multimask_reconciliation_20260502/"
        "cmg3a_body200.decoded_mask_array.npz",
    ),
    (
        "cmg3_nonzero_top2",
        REPO_ROOT
        / "experiments/results/c067_multimask_reconciliation_20260502/"
        "cmg3_nonzero_top2.decoded_mask_array.npz",
    ),
    (
        "cmg3_nonzero_top1",
        REPO_ROOT
        / "experiments/results/c067_multimask_reconciliation_20260502/"
        "cmg3_nonzero_top1.decoded_mask_array.npz",
    ),
    (
        "cmg3_rowspan_stride1",
        REPO_ROOT
        / "experiments/results/c067_multimask_reconciliation_20260502/"
        "cmg3_rowspan_stride1.decoded_mask_array.npz",
    ),
)


class OverlayPlannerError(ValueError):
    """Raised for malformed overlay planner inputs."""


@dataclass(frozen=True)
class OverlayRun:
    frame_index: int
    y: int
    x0: int
    length: int
    value: int

    def as_list(self) -> list[int]:
        return [self.frame_index, self.y, self.x0, self.length, self.value]


@dataclass(frozen=True)
class DonorMaskInput:
    label: str
    path: Path


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OverlayPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise OverlayPlannerError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path | str | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(repo_root.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def _load_mask_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise OverlayPlannerError(f"mask array missing: {path}")
    loaded = np.load(path, allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.files)
            preferred = [key for key in ("masks", "decoded_masks", "array") if key in keys]
            if preferred:
                array = loaded[preferred[0]]
            elif len(keys) == 1:
                array = loaded[keys[0]]
            else:
                raise OverlayPlannerError(
                    f"{path} must contain one array or a masks/decoded_masks/array key"
                )
        else:
            array = loaded
        array = np.asarray(array)
        if array.ndim != 3 or array.dtype != np.uint8:
            raise OverlayPlannerError(
                f"{path} must be uint8 rank-3, got shape={array.shape} dtype={array.dtype}"
            )
        if array.shape[1:] != (384, 512):
            raise OverlayPlannerError(f"{path} must have 384x512 masks, got {array.shape[1:]}")
        if int(array.min()) < 0 or int(array.max()) >= 5:
            raise OverlayPlannerError(
                f"{path} classes must be in [0,5), got [{int(array.min())},{int(array.max())}]"
            )
        return np.ascontiguousarray(array)
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()


def _mask_tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _parse_int_set(raw: str | None, *, field: str) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return ()
    values: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"{field} range has end before start: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"{field} values must be nonnegative")
    return tuple(sorted(values))


def _parse_string_set(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return ()
    return tuple(sorted({part.strip() for part in raw.split(",") if part.strip()}))


def _candidate_mask_inputs(raw: Iterable[str], repo_root: Path) -> tuple[DonorMaskInput, ...]:
    out: list[DonorMaskInput] = []
    for item in raw:
        if "=" in item:
            label, raw_path = item.split("=", 1)
        else:
            raw_path = item
            label = Path(item).stem
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("candidate mask label must be nonempty")
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = repo_root / path
        out.append(DonorMaskInput(label=label, path=path))
    return tuple(out)


def _derive_trust_from_postdecode_plan(path: Path, *, budget_payload_bytes: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": _display_path(path, REPO_ROOT),
            "available": False,
            "pair_indices": [],
            "class_ids": [],
            "selected_atoms": [],
        }
    payload = _read_json(path)
    policies = payload.get("budget_policies")
    if not isinstance(policies, list):
        raise OverlayPlannerError(f"{path} lacks budget_policies")
    selected_policy: dict[str, Any] | None = None
    for policy in policies:
        if not isinstance(policy, dict):
            continue
        if int(policy.get("budget_payload_bytes", -1)) == int(budget_payload_bytes):
            selected_policy = policy
            break
    if selected_policy is None and policies:
        first = policies[0]
        if isinstance(first, dict):
            selected_policy = first
    if selected_policy is None:
        raise OverlayPlannerError(f"{path} has no usable budget policy")

    pair_indices: set[int] = set()
    class_ids: set[int] = set()
    selected_atoms: list[dict[str, Any]] = []
    for atom in selected_policy.get("selected_atoms") or []:
        if not isinstance(atom, dict):
            continue
        pairs = [int(value) for value in atom.get("pair_indices") or [] if isinstance(value, int)]
        pair_indices.update(pairs)
        class_id = atom.get("class_id")
        if isinstance(class_id, int):
            class_ids.add(int(class_id))
        selected_atoms.append(
            {
                "atom_id": atom.get("atom_id"),
                "pair_indices": pairs,
                "class_id": class_id if isinstance(class_id, int) else None,
                "changed_pixels": atom.get("changed_pixels"),
                "expected_component_score_improvement_first_order": atom.get(
                    "expected_component_score_improvement_first_order"
                ),
            }
        )

    return {
        "source": _display_path(path, REPO_ROOT),
        "available": True,
        "policy_id": selected_policy.get("policy_id"),
        "budget_payload_bytes": selected_policy.get("budget_payload_bytes"),
        "pair_indices": sorted(pair_indices),
        "class_ids": sorted(class_ids),
        "selected_atoms": selected_atoms,
    }


def _catastrophic_pairs_from_geometry_plan(
    path: Path,
    *,
    family_groups: set[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": _display_path(path, REPO_ROOT),
            "available": False,
            "pair_indices": [],
            "by_negative": [],
            "family_group_filter": sorted(family_groups) if family_groups else None,
        }
    payload = _read_json(path)
    exact_negative_inputs = payload.get("exact_negative_inputs")
    if not isinstance(exact_negative_inputs, list):
        raise OverlayPlannerError(f"{path} lacks exact_negative_inputs")
    catastrophic: set[int] = set()
    by_negative: list[dict[str, Any]] = []
    for row in exact_negative_inputs:
        if not isinstance(row, dict):
            continue
        pairs = [int(value) for value in row.get("catastrophic_pair_indices") or [] if isinstance(value, int)]
        family_group = row.get("family_group")
        included = family_groups is None or str(family_group) in family_groups
        if included:
            catastrophic.update(pairs)
        by_negative.append(
            {
                "negative_id": row.get("negative_id"),
                "family_group": family_group,
                "catastrophic_pair_count": len(pairs),
                "catastrophic_pair_indices_sha256": _sha256_bytes(
                    _json_bytes({"pairs": sorted(pairs)})
                ),
                "included_in_veto": included,
            }
        )
    return {
        "source": _display_path(path, REPO_ROOT),
        "available": True,
        "pair_indices": sorted(catastrophic),
        "by_negative": by_negative,
        "family_group_filter": sorted(family_groups) if family_groups else None,
    }


def _resolve_pair_index_basis(frame_count: int, basis: str) -> str:
    if basis == PAIR_INDEX_BASIS_AUTO:
        if frame_count == 600:
            return PAIR_INDEX_BASIS_HALF_FRAME
        return PAIR_INDEX_BASIS_VIDEO_FRAME
    if basis not in {
        PAIR_INDEX_BASIS_HALF_FRAME,
        PAIR_INDEX_BASIS_VIDEO_FRAME,
    }:
        raise OverlayPlannerError(f"unsupported pair_index_basis={basis!r}")
    return basis


def _pair_frame_mask(
    frame_count: int,
    pair_indices: set[int],
    *,
    pair_index_basis: str = PAIR_INDEX_BASIS_AUTO,
) -> np.ndarray:
    resolved_basis = _resolve_pair_index_basis(frame_count, pair_index_basis)
    mask = np.zeros(frame_count, dtype=bool)
    for pair_index in pair_indices:
        pair_index = int(pair_index)
        if resolved_basis == PAIR_INDEX_BASIS_HALF_FRAME:
            if 0 <= pair_index < frame_count:
                mask[pair_index] = True
        else:
            start = 2 * pair_index
            if start >= frame_count:
                continue
            mask[start : min(start + 2, frame_count)] = True
    return mask


def _selected_pair_indices_from_frame_indices(
    frame_indices: np.ndarray,
    *,
    frame_count: int,
    pair_index_basis: str,
) -> list[int]:
    resolved_basis = _resolve_pair_index_basis(frame_count, pair_index_basis)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.size == 0:
        return []
    if resolved_basis == PAIR_INDEX_BASIS_HALF_FRAME:
        pairs = frames
    else:
        pairs = frames // 2
    return sorted(set(int(value) for value in pairs.tolist()))


def _runs_from_selected(base: np.ndarray, donor: np.ndarray, selected: np.ndarray) -> list[OverlayRun]:
    runs: list[OverlayRun] = []
    frames, height, width = [int(value) for value in base.shape]
    for frame_index in range(frames):
        frame_selected = selected[frame_index]
        if not bool(frame_selected.any()):
            continue
        for y in range(height):
            row = frame_selected[y]
            if not bool(row.any()):
                continue
            x = 0
            while x < width:
                if not bool(row[x]):
                    x += 1
                    continue
                value = int(donor[frame_index, y, x])
                x0 = x
                x += 1
                while x < width and bool(row[x]) and int(donor[frame_index, y, x]) == value:
                    x += 1
                runs.append(
                    OverlayRun(
                        frame_index=frame_index,
                        y=y,
                        x0=x0,
                        length=x - x0,
                        value=value,
                    )
                )
    return runs


def _encode_overlay_payload(
    *,
    runs: list[OverlayRun],
    header: dict[str, Any],
) -> bytes:
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += PAYLOAD_HEADER_STRUCT.pack(PAYLOAD_MAGIC, PAYLOAD_VERSION, len(header_bytes))
    out += header_bytes
    for run in runs:
        if not (
            0 <= run.frame_index <= 65535
            and 0 <= run.y <= 65535
            and 0 <= run.x0 <= 65535
            and 0 < run.length <= 65535
            and 0 <= run.value <= 255
        ):
            raise OverlayPlannerError(f"overlay run out of fixed-width range: {run}")
        out += RUN_STRUCT.pack(run.frame_index, run.y, run.x0, run.length, run.value)
    return bytes(out)


def _compressed_payloads(raw_payload: bytes) -> dict[str, dict[str, Any]]:
    zlib_payload = zlib.compress(raw_payload, level=9)
    lzma_payload = lzma.compress(raw_payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    return {
        "raw": {"bytes": len(raw_payload), "sha256": _sha256_bytes(raw_payload)},
        "zlib": {"bytes": len(zlib_payload), "sha256": _sha256_bytes(zlib_payload)},
        "lzma_xz": {"bytes": len(lzma_payload), "sha256": _sha256_bytes(lzma_payload)},
    }


def _best_compressor(payloads: dict[str, dict[str, Any]]) -> str:
    return min(payloads, key=lambda key: (int(payloads[key]["bytes"]), key))


def _histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.uint8, copy=False), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)}


def _summarize_overlay_candidate(
    *,
    label: str,
    donor_path: Path,
    base: np.ndarray,
    donor: np.ndarray,
    base_sha256: str,
    donor_sha256: str,
    allowed_pairs: set[int],
    allowed_classes: set[int],
    catastrophic_pairs: set[int],
    max_pixel_disagreement_fraction: float,
    max_selected_pixels: int,
    max_compressed_payload_bytes: int,
    pair_index_basis: str,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if donor.shape != base.shape:
        raise OverlayPlannerError(f"{label} shape {donor.shape} differs from base {base.shape}")
    diff = base != donor
    total_elements = int(base.size)
    donor_disagreement_count = int(diff.sum())
    resolved_pair_index_basis = _resolve_pair_index_basis(
        int(base.shape[0]),
        pair_index_basis,
    )
    pair_mask = _pair_frame_mask(
        int(base.shape[0]),
        allowed_pairs,
        pair_index_basis=resolved_pair_index_basis,
    )
    selected = diff & pair_mask[:, None, None]
    if allowed_classes:
        selected &= np.isin(base, list(allowed_classes))
    selected_pair_indices = _selected_pair_indices_from_frame_indices(
        np.nonzero(selected)[0],
        frame_count=int(base.shape[0]),
        pair_index_basis=resolved_pair_index_basis,
    )
    selected_class_values = donor[selected]
    source_class_values = base[selected]
    selected_pixel_count = int(selected.sum())
    selected_fraction = float(selected_pixel_count) / float(total_elements)
    catastrophic_overlap = sorted(set(selected_pair_indices) & catastrophic_pairs)
    outside_rejected_count = int(donor_disagreement_count - selected_pixel_count)
    runs = _runs_from_selected(base, donor, selected) if selected_pixel_count else []
    changed = base.copy()
    if selected_pixel_count:
        changed[selected] = donor[selected]
    overlay_tensor_sha256 = _mask_tensor_sha256(changed)
    del changed

    blockers: list[str] = []
    if selected_pixel_count == 0:
        blockers.append("no donor pixels survive the trust-region selector")
    if catastrophic_overlap:
        blockers.append(
            "selected pairs overlap catastrophic exact-negative pairs: "
            + ",".join(str(value) for value in catastrophic_overlap[:24])
        )
    if selected_fraction > max_pixel_disagreement_fraction:
        blockers.append(
            "selected pixel disagreement fraction "
            f"{selected_fraction:.8f} exceeds {max_pixel_disagreement_fraction:.8f}"
        )
    if selected_pixel_count > max_selected_pixels:
        blockers.append(
            f"selected pixels {selected_pixel_count} exceed cap {max_selected_pixels}"
        )

    payload_header = {
        "schema": PAYLOAD_SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "base_mask_tensor_sha256": base_sha256,
        "donor_mask_tensor_sha256": donor_sha256,
        "shape": [int(value) for value in base.shape],
        "pair_index_basis": resolved_pair_index_basis,
        "run_struct": RUN_STRUCT_NAME,
        "run_count": len(runs),
        "selected_pixel_count": selected_pixel_count,
        "selected_pair_indices": selected_pair_indices,
        "allowed_pair_indices": sorted(allowed_pairs),
        "allowed_source_class_ids": sorted(allowed_classes),
        "reconstructed_mask_u8_sha256": overlay_tensor_sha256,
    }
    raw_payload = _encode_overlay_payload(runs=runs, header=payload_header)
    payloads = _compressed_payloads(raw_payload)
    best = _best_compressor(payloads)
    if int(payloads[best]["bytes"]) > max_compressed_payload_bytes:
        blockers.append(
            f"best compressed payload bytes {payloads[best]['bytes']} exceed cap "
            f"{max_compressed_payload_bytes}"
        )

    safe = not blockers
    payload_record: dict[str, Any] | None = None
    if safe:
        safe_dir = output_dir / "safe_specs" / label
        safe_dir.mkdir(parents=True, exist_ok=True)
        raw_path = safe_dir / "overlay_payload.cdo1"
        raw_path.write_bytes(raw_payload)
        spec_path = safe_dir / "overlay_spec.json"
        payload_record = {
            "schema": "c067_decoded_delta_overlay_safe_spec_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "dispatchable": False,
            "reason_not_dispatchable": (
                "local overlay payload spec only; archive runtime decoder and exact CUDA auth eval "
                "are still required"
            ),
            "payload_path": _display_path(raw_path, repo_root),
            "payload_bytes": len(raw_payload),
            "payload_sha256": _sha256_bytes(raw_payload),
            "recommended_compressor": best,
            "compressed_payloads": payloads,
            "payload_header": payload_header,
        }
        _write_json(spec_path, payload_record)
        payload_record["spec_path"] = _display_path(spec_path, repo_root)

    estimated_archive_bytes = C067_FRONTIER_ARCHIVE_BYTES + int(payloads[best]["bytes"])
    byte_gate_passed = estimated_archive_bytes <= C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE
    return {
        "candidate_id": f"c067_decoded_delta_overlay_{label}",
        "donor_label": label,
        "donor_mask_array": {
            "path": _display_path(donor_path, repo_root),
            "file_sha256": _sha256_file(donor_path),
            "tensor_sha256": donor_sha256,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "safe_spec_emitted": safe,
        "payload_spec": payload_record,
        "trust_region": {
            "pair_index_basis": resolved_pair_index_basis,
            "allowed_pair_indices": sorted(allowed_pairs),
            "allowed_source_class_ids": sorted(allowed_classes),
            "catastrophic_selected_pair_overlap": catastrophic_overlap,
            "selected_pair_indices": selected_pair_indices,
        },
        "pixel_disagreement": {
            "donor_vs_base_count": donor_disagreement_count,
            "donor_vs_base_fraction": round(donor_disagreement_count / total_elements, 12),
            "selected_overlay_count": selected_pixel_count,
            "selected_overlay_fraction": round(selected_fraction, 12),
            "outside_trust_region_rejected_count": outside_rejected_count,
            "overlay_vs_base_count": selected_pixel_count,
            "overlay_tensor_sha256": overlay_tensor_sha256,
            "source_class_histogram": _histogram(source_class_values),
            "donor_value_histogram": _histogram(selected_class_values),
        },
        "byte_closure": {
            "payload_run_count": len(runs),
            "payload_run_struct": RUN_STRUCT_NAME,
            "compressed_payloads": payloads,
            "recommended_compressor": best,
            "estimated_archive_bytes_if_runtime_added_without_replacing_mask_stream": estimated_archive_bytes,
            "estimated_archive_delta_vs_c067": int(payloads[best]["bytes"]),
            "unchanged_distortion_sub0300_byte_gate": C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
            "byte_gate_passed": byte_gate_passed,
            "archive_byte_closed": False,
            "archive_byte_closure_note": (
                "The payload bytes are deterministic and charged in the spec, but no "
                "contest archive is emitted until a reviewed inflate runtime consumes CDO1."
            ),
        },
        "safety_gate": {
            "status": "safe_local_spec_only" if safe else "fail_closed",
            "blockers": blockers,
            "max_pixel_disagreement_fraction": max_pixel_disagreement_fraction,
            "max_selected_pixels": max_selected_pixels,
            "max_compressed_payload_bytes": max_compressed_payload_bytes,
        },
    }


def build_plan(
    *,
    repo_root: Path,
    base_mask_array: Path,
    candidate_masks: Iterable[DonorMaskInput],
    geometry_plan_json: Path,
    postdecode_trust_plan_json: Path,
    trust_pairs: Iterable[int] = (),
    trust_classes: Iterable[int] = (),
    trust_budget_payload_bytes: int = 4000,
    max_pixel_disagreement_fraction: float = DEFAULT_MAX_PIXEL_DISAGREEMENT_FRACTION,
    max_selected_pixels: int = DEFAULT_MAX_SELECTED_PIXELS,
    max_compressed_payload_bytes: int = DEFAULT_MAX_COMPRESSED_PAYLOAD_BYTES,
    catastrophic_family_groups: Iterable[str] = (),
    pair_index_basis: str = PAIR_INDEX_BASIS_AUTO,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    if max_pixel_disagreement_fraction <= 0.0:
        raise OverlayPlannerError("max_pixel_disagreement_fraction must be positive")
    if max_selected_pixels < 0:
        raise OverlayPlannerError("max_selected_pixels must be nonnegative")
    if max_compressed_payload_bytes < 0:
        raise OverlayPlannerError("max_compressed_payload_bytes must be nonnegative")

    base = _load_mask_array(base_mask_array)
    resolved_pair_index_basis = _resolve_pair_index_basis(
        int(base.shape[0]),
        pair_index_basis,
    )
    base_sha = _mask_tensor_sha256(base)
    derived_trust = _derive_trust_from_postdecode_plan(
        postdecode_trust_plan_json,
        budget_payload_bytes=trust_budget_payload_bytes,
    )
    explicit_pairs = set(int(value) for value in trust_pairs)
    explicit_classes = set(int(value) for value in trust_classes)
    allowed_pairs = explicit_pairs or set(int(value) for value in derived_trust["pair_indices"])
    allowed_classes = explicit_classes or set(int(value) for value in derived_trust["class_ids"])
    family_group_filter = {str(value) for value in catastrophic_family_groups if str(value)}
    catastrophic = _catastrophic_pairs_from_geometry_plan(
        geometry_plan_json,
        family_groups=family_group_filter or None,
    )
    catastrophic_pairs = set(int(value) for value in catastrophic["pair_indices"])

    candidates: list[dict[str, Any]] = []
    missing_candidate_masks: list[dict[str, str]] = []
    for donor in candidate_masks:
        if not donor.path.exists():
            missing_candidate_masks.append(
                {"label": donor.label, "path": _display_path(donor.path, repo_root) or str(donor.path)}
            )
            continue
        donor_array = _load_mask_array(donor.path)
        donor_sha = _mask_tensor_sha256(donor_array)
        candidates.append(
            _summarize_overlay_candidate(
                label=donor.label,
                donor_path=donor.path,
                base=base,
                donor=donor_array,
                base_sha256=base_sha,
                donor_sha256=donor_sha,
                allowed_pairs=allowed_pairs,
                allowed_classes=allowed_classes,
                catastrophic_pairs=catastrophic_pairs,
                max_pixel_disagreement_fraction=max_pixel_disagreement_fraction,
                max_selected_pixels=max_selected_pixels,
                max_compressed_payload_bytes=max_compressed_payload_bytes,
                pair_index_basis=resolved_pair_index_basis,
                output_dir=output_dir,
                repo_root=repo_root,
            )
        )
    candidates.sort(
        key=lambda row: (
            not bool(row["safe_spec_emitted"]),
            int(row["byte_closure"]["compressed_payloads"][row["byte_closure"]["recommended_compressor"]]["bytes"]),
            str(row["candidate_id"]),
        )
    )
    safe = [row for row in candidates if row["safe_spec_emitted"]]
    dispatchable = [
        row
        for row in safe
        if bool(row["byte_closure"]["archive_byte_closed"])
        and bool(row["byte_closure"]["byte_gate_passed"])
    ]
    trust_pair_catastrophic_overlap = sorted(allowed_pairs & catastrophic_pairs)

    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "frontier": {
            "archive_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "archive_sha256": C067_FRONTIER_ARCHIVE_SHA256,
            "unchanged_distortion_sub0300_byte_gate": C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
        },
        "base_decoded_mask": {
            "path": _display_path(base_mask_array, repo_root),
            "file_sha256": _sha256_file(base_mask_array),
            "tensor_sha256": base_sha,
            "shape": [int(value) for value in base.shape],
            "dtype": str(base.dtype),
        },
        "trust_region": {
            "pair_index_basis": resolved_pair_index_basis,
            "source": "explicit_cli" if explicit_pairs or explicit_classes else "postdecode_pair_class_plan",
            "postdecode_trust_plan": derived_trust,
            "allowed_pair_indices": sorted(allowed_pairs),
            "allowed_source_class_ids": sorted(allowed_classes),
            "catastrophic_pair_source": catastrophic,
            "allowed_pair_catastrophic_overlap": trust_pair_catastrophic_overlap,
            "policy": (
                "Preserve C067 decoded global geometry exactly outside selected overlay runs. "
                "Emit payload specs only when selected pairs have zero catastrophic overlap."
            ),
            "catastrophic_family_groups": sorted(family_group_filter) if family_group_filter else "all",
        },
        "candidate_count": len(candidates),
        "safe_spec_count": len(safe),
        "dispatchable_candidate_count": len(dispatchable),
        "dispatchable_candidates": dispatchable,
        "safe_specs": safe,
        "gated_candidates": candidates,
        "missing_candidate_masks": missing_candidate_masks,
        "decision": "local_specs_only_no_dispatch"
        if safe
        else "fail_closed_no_safe_overlay_specs",
        "required_before_remote_eval": [
            "review and land a contest inflate runtime for CDO1 overlay payloads",
            "emit a deterministic archive.zip with payload closure and runtime tree hash",
            "claim lane with tools/claim_lane_dispatch.py before any remote exact eval",
            "run exact CUDA auth eval; this planner makes no score claim",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--base-mask-array", type=Path, default=DEFAULT_BASE_MASK_ARRAY)
    parser.add_argument("--geometry-plan-json", type=Path, default=DEFAULT_GEOMETRY_PLAN_JSON)
    parser.add_argument("--postdecode-trust-plan-json", type=Path, default=DEFAULT_POSTDECODE_TRUST_PLAN_JSON)
    parser.add_argument("--trust-budget-payload-bytes", type=int, default=4000)
    parser.add_argument("--trust-pairs", type=str, default="")
    parser.add_argument("--trust-classes", type=str, default="")
    parser.add_argument(
        "--candidate-mask",
        action="append",
        default=[],
        help="Donor mask input as label=path. Defaults to local C067 decoded mask candidates.",
    )
    parser.add_argument("--max-pixel-disagreement-fraction", type=float, default=DEFAULT_MAX_PIXEL_DISAGREEMENT_FRACTION)
    parser.add_argument("--max-selected-pixels", type=int, default=DEFAULT_MAX_SELECTED_PIXELS)
    parser.add_argument("--max-compressed-payload-bytes", type=int, default=DEFAULT_MAX_COMPRESSED_PAYLOAD_BYTES)
    parser.add_argument(
        "--pair-index-basis",
        choices=PAIR_INDEX_BASIS_CHOICES,
        default=PAIR_INDEX_BASIS_AUTO,
        help=(
            "How trust-plan pair indices map to mask tensor rows. auto uses "
            "half_frame_pair_index for 600-row half-frame C067 masks and "
            "video_frame_pair_index otherwise."
        ),
    )
    parser.add_argument(
        "--catastrophic-family-groups",
        type=str,
        default="",
        help=(
            "Comma-separated geometry-plan family_group values to include in "
            "the catastrophic-pair veto. Defaults to all families."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json.exists() and not args.force:
        raise SystemExit(f"{output_json} exists; pass --force to overwrite")
    candidate_masks = (
        _candidate_mask_inputs(args.candidate_mask, args.repo_root)
        if args.candidate_mask
        else tuple(DonorMaskInput(label=label, path=path) for label, path in DEFAULT_CANDIDATE_MASKS)
    )
    plan = build_plan(
        repo_root=args.repo_root,
        base_mask_array=args.base_mask_array,
        candidate_masks=candidate_masks,
        geometry_plan_json=args.geometry_plan_json,
        postdecode_trust_plan_json=args.postdecode_trust_plan_json,
        trust_pairs=_parse_int_set(args.trust_pairs, field="trust_pairs"),
        trust_classes=_parse_int_set(args.trust_classes, field="trust_classes"),
        trust_budget_payload_bytes=args.trust_budget_payload_bytes,
        max_pixel_disagreement_fraction=args.max_pixel_disagreement_fraction,
        max_selected_pixels=args.max_selected_pixels,
        max_compressed_payload_bytes=args.max_compressed_payload_bytes,
        catastrophic_family_groups=_parse_string_set(args.catastrophic_family_groups),
        pair_index_basis=args.pair_index_basis,
        output_dir=args.output_dir,
    )
    _write_json(output_json, plan)
    print(
        json.dumps(
            {
                "output_json": _display_path(output_json, args.repo_root),
                "candidate_count": plan["candidate_count"],
                "safe_spec_count": plan["safe_spec_count"],
                "dispatchable_candidate_count": plan["dispatchable_candidate_count"],
                "decision": plan["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
