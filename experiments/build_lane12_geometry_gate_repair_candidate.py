#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build byte-closed Lane 12 geometry-gate CDO1 repair candidates.

This bridge consumes the local-only Lane 12 geometry atom plan and materializes
one selected policy as a charged CDO1 decoded-mask overlay over the measured
``masks.nrv`` archive. It does not run scorers and does not claim score. The
output is an exact-evaluable archive only after the local residual geometry gate
is recorded and the archive bytes/SHA are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.build_c067_decoded_delta_overlay_candidate import (  # noqa: E402
    build_candidate as build_cdo1_candidate,
)
from experiments.plan_c067_decoded_delta_overlay_mask_topology import (  # noqa: E402
    RUN_STRUCT_NAME,
    _encode_overlay_payload,
    _mask_tensor_sha256,
    _runs_from_selected,
)


SCHEMA = "lane12_geometry_gate_repair_candidate_v1"
TOOL = "experiments/build_lane12_geometry_gate_repair_candidate.py"
PLAN_SCHEMA = "lane12_geometry_gate_repair_atom_plan_v1"
DEFAULT_PLAN_JSON = (
    REPO_ROOT
    / "experiments/results/lane12_geometry_gate_repair_atoms_20260503/"
    "lane12_geometry_gate_repair_atoms.json"
)
DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/"
    "archive_lane_12_nerv.zip"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/lane12_geometry_gate_repair_candidate_20260503"
)
RATE_DENOMINATOR_BYTES = 37_545_489
DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION = 0.001
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


class Lane12GeometryRepairBuildError(ValueError):
    """Raised when a Lane 12 repair policy cannot be materialized safely."""


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Lane12GeometryRepairBuildError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise Lane12GeometryRepairBuildError(f"{path} must contain a JSON object")
    return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(raw: str | Path, *, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_mask_tensor(path: Path) -> np.ndarray:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for key in ("masks", "mask", "decoded_masks", "tensor"):
            if key in tensor:
                tensor = tensor[key]
                break
    if not hasattr(tensor, "ndim"):
        raise Lane12GeometryRepairBuildError(f"{path} did not contain a tensor")
    if tensor.ndim == 4 and int(tensor.shape[1]) == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise Lane12GeometryRepairBuildError(
            f"{path} tensor must be THW, got shape={tuple(tensor.shape)}"
        )
    tensor = tensor.detach().cpu().to(dtype=torch.uint8).contiguous()
    array = tensor.numpy()
    if array.ndim != 3 or array.shape[1:] != (384, 512):
        # Unit tests use tiny tensors; production artifacts must stay scorer-shaped.
        if "pytest" not in sys.modules:
            raise Lane12GeometryRepairBuildError(
                f"{path} must be scorer mask shape T x 384 x 512, got {array.shape}"
            )
    if int(array.min()) < 0 or int(array.max()) >= 5:
        raise Lane12GeometryRepairBuildError(
            f"{path} mask classes must be in [0,5), got [{int(array.min())},{int(array.max())}]"
        )
    return np.ascontiguousarray(array)


def _policy_by_id(plan: dict[str, Any], policy_id: str | None) -> dict[str, Any]:
    policies = plan.get("candidate_policies")
    if not isinstance(policies, list):
        raise Lane12GeometryRepairBuildError("plan lacks candidate_policies")
    valid = [policy for policy in policies if isinstance(policy, dict)]
    if not valid:
        raise Lane12GeometryRepairBuildError("plan has no candidate policies")
    if policy_id is None:
        return valid[-1]
    for policy in valid:
        if policy.get("policy_id") == policy_id:
            return policy
    raise Lane12GeometryRepairBuildError(f"policy_id not found: {policy_id}")


def _atoms_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    atoms = plan.get("atoms")
    if not isinstance(atoms, list):
        raise Lane12GeometryRepairBuildError("plan lacks atoms")
    out: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        if not isinstance(atom, dict) or not isinstance(atom.get("atom_id"), str):
            continue
        out[str(atom["atom_id"])] = atom
    return out


def _selected_atom_ids(policy: dict[str, Any]) -> list[str]:
    raw = policy.get("selected_atom_ids")
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise Lane12GeometryRepairBuildError(
            f"policy {policy.get('policy_id')!r} lacks selected_atom_ids"
        )
    if not raw:
        raise Lane12GeometryRepairBuildError(
            f"policy {policy.get('policy_id')!r} selected no atoms"
        )
    return list(raw)


def _mark_box(
    selected: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    frame: int,
    box_xyxy: list[Any],
) -> int:
    if len(box_xyxy) != 4:
        raise Lane12GeometryRepairBuildError(f"invalid box_xyxy={box_xyxy!r}")
    x0, y0, x1, y1 = [int(value) for value in box_xyxy]
    if not (0 <= frame < baseline.shape[0]):
        raise Lane12GeometryRepairBuildError(f"frame out of range: {frame}")
    if not (0 <= x0 <= x1 <= baseline.shape[2] and 0 <= y0 <= y1 <= baseline.shape[1]):
        raise Lane12GeometryRepairBuildError(f"box outside mask tensor: {box_xyxy!r}")
    region_diff = baseline[frame, y0:y1, x0:x1] != candidate[frame, y0:y1, x0:x1]
    before = int(selected[frame, y0:y1, x0:x1].sum())
    selected[frame, y0:y1, x0:x1] |= region_diff
    return int(selected[frame, y0:y1, x0:x1].sum()) - before


def _mark_transition_pair(
    selected: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    frames: list[Any] | None,
    pair_index: int | None,
) -> int:
    if frames is None:
        if pair_index is None:
            raise Lane12GeometryRepairBuildError("transition atom lacks frames and pair_index")
        frames = [2 * int(pair_index), 2 * int(pair_index) + 1]
    changed = 0
    for raw_frame in frames:
        frame = int(raw_frame)
        if not (0 <= frame < baseline.shape[0]):
            continue
        diff = baseline[frame] != candidate[frame]
        before = int(selected[frame].sum())
        selected[frame] |= diff
        changed += int(selected[frame].sum()) - before
    return changed


def _selection_from_atoms(
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    selected_atoms: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if baseline.shape != candidate.shape:
        raise Lane12GeometryRepairBuildError(
            f"baseline shape {baseline.shape} differs from candidate shape {candidate.shape}"
        )
    selected = np.zeros(baseline.shape, dtype=bool)
    atom_rows: list[dict[str, Any]] = []
    for atom in selected_atoms:
        identity = atom.get("identity") if isinstance(atom.get("identity"), dict) else {}
        kind = str(atom.get("atom_kind"))
        before = int(selected.sum())
        if kind in {"residual_region_patch", "critical_component_box_patch"}:
            frame = int(identity.get("frame"))
            box = identity.get("box_xyxy")
            if not isinstance(box, list):
                raise Lane12GeometryRepairBuildError(f"{atom['atom_id']} lacks box_xyxy")
            added = _mark_box(
                selected,
                baseline,
                candidate,
                frame=frame,
                box_xyxy=box,
            )
        elif kind == "transition_pair_focus":
            raw_frames = identity.get("frames")
            frames = raw_frames if isinstance(raw_frames, list) else None
            raw_pair = identity.get("pair_index")
            pair_index = int(raw_pair) if raw_pair is not None else None
            added = _mark_transition_pair(
                selected,
                baseline,
                candidate,
                frames=frames,
                pair_index=pair_index,
            )
        else:
            raise Lane12GeometryRepairBuildError(f"unsupported atom kind {kind!r}")
        after = int(selected.sum())
        atom_rows.append(
            {
                "atom_id": atom.get("atom_id"),
                "atom_kind": kind,
                "identity": identity,
                "new_selected_pixels": int(added),
                "cumulative_selected_pixels": after,
                "overlap_pixels": max(0, int((before + added) - after)),
            }
        )
    return selected, atom_rows


def _residual_metrics(
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    repaired = candidate.copy()
    repaired[selected] = baseline[selected]
    before = candidate != baseline
    after = repaired != baseline
    changed_before = int(before.sum())
    changed_after = int(after.sum())
    total = int(before.size)
    selected_frames = np.flatnonzero(selected.reshape(selected.shape[0], -1).any(axis=1))
    residual_frames = np.flatnonzero(after.reshape(after.shape[0], -1).any(axis=1))
    selected_pairs = sorted({int(frame) // 2 for frame in selected_frames.tolist()})
    residual_pairs = sorted({int(frame) // 2 for frame in residual_frames.tolist()})
    return repaired, {
        "total_pixels": total,
        "candidate_disagreement_pixels_before": changed_before,
        "residual_disagreement_pixels_after": changed_after,
        "selected_repair_pixels": int(selected.sum()),
        "repaired_disagreement_pixels": changed_before - changed_after,
        "global_disagreement_before": float(changed_before) / float(total),
        "global_disagreement_after": float(changed_after) / float(total),
        "selected_frame_count": int(len(selected_frames)),
        "selected_frames": [int(v) for v in selected_frames[:512].tolist()],
        "selected_pair_indices": selected_pairs,
        "residual_frame_count": int(len(residual_frames)),
        "residual_pair_count": int(len(residual_pairs)),
    }


def _rate_cost(byte_count: int) -> float:
    return 25.0 * float(byte_count) / float(RATE_DENOMINATOR_BYTES)


def _archive_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path, "r") as zf:
        return [info.filename for info in zf.infolist()]


def _decompress_cdo1_member(archive: Path) -> bytes:
    with zipfile.ZipFile(archive, "r") as zf:
        names = [name for name in zf.namelist() if name.startswith("masks.cdo1")]
        if len(names) != 1:
            raise Lane12GeometryRepairBuildError(
                f"expected exactly one CDO1 member, found {names}"
            )
        data = zf.read(names[0])
        if names[0].endswith(".xz"):
            return lzma.decompress(data, format=lzma.FORMAT_XZ)
        if names[0].endswith(".zlib"):
            import zlib

            return zlib.decompress(data)
        if names[0].endswith(".br"):
            import brotli

            return brotli.decompress(data)
        return data


def build_lane12_geometry_gate_repair_candidate(
    *,
    plan_json: Path = DEFAULT_PLAN_JSON,
    policy_id: str | None = None,
    base_archive: Path = DEFAULT_BASE_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    overlay_compressor: str = "lzma_xz",
    max_residual_disagreement_fraction: float = DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    if plan.get("schema") != PLAN_SCHEMA:
        raise Lane12GeometryRepairBuildError(f"plan schema must be {PLAN_SCHEMA}")
    if plan.get("score_claim") is not False or plan.get("promotion_eligible") is not False:
        raise Lane12GeometryRepairBuildError("input plan must remain non-promotable")
    tensor_status = (
        plan.get("inputs", {}).get("tensor_status")
        if isinstance(plan.get("inputs"), dict)
        else None
    )
    if not isinstance(tensor_status, dict) or tensor_status.get("loaded") is not True:
        raise Lane12GeometryRepairBuildError("plan must contain loaded baseline/candidate tensors")
    baseline_info = tensor_status.get("baseline")
    candidate_info = tensor_status.get("candidate")
    if not isinstance(baseline_info, dict) or not isinstance(candidate_info, dict):
        raise Lane12GeometryRepairBuildError("tensor status lacks baseline/candidate records")
    baseline_path = _resolve_path(str(baseline_info.get("tensor_path")), repo_root=repo_root)
    candidate_path = _resolve_path(str(candidate_info.get("tensor_path")), repo_root=repo_root)
    baseline = _load_mask_tensor(baseline_path)
    candidate = _load_mask_tensor(candidate_path)

    policy = _policy_by_id(plan, policy_id)
    atoms = _atoms_by_id(plan)
    selected_ids = _selected_atom_ids(policy)
    missing = [atom_id for atom_id in selected_ids if atom_id not in atoms]
    if missing:
        raise Lane12GeometryRepairBuildError(f"policy references missing atom ids: {missing}")
    selected_atoms = [atoms[atom_id] for atom_id in selected_ids]
    selected, atom_rows = _selection_from_atoms(
        baseline=baseline,
        candidate=candidate,
        selected_atoms=selected_atoms,
    )
    if not bool(selected.any()):
        raise Lane12GeometryRepairBuildError(f"policy {policy.get('policy_id')} selects no changed pixels")
    repaired, metrics = _residual_metrics(baseline=baseline, candidate=candidate, selected=selected)
    runs = _runs_from_selected(candidate, baseline, selected)
    selected_pairs = metrics["selected_pair_indices"]
    header = {
        "schema": "c067_decoded_delta_overlay_payload_v1",
        "producer": TOOL,
        "score_claim": False,
        "base_mask_tensor_sha256": _mask_tensor_sha256(candidate),
        "target_mask_tensor_sha256": _mask_tensor_sha256(baseline),
        "reconstructed_mask_u8_sha256": _mask_tensor_sha256(repaired),
        "shape": [int(value) for value in candidate.shape],
        "pair_index_basis": "video_frame_pair_index",
        "run_struct": RUN_STRUCT_NAME,
        "run_count": len(runs),
        "selected_pixel_count": int(selected.sum()),
        "selected_pair_indices": selected_pairs,
        "source_plan": {
            "path": str(plan_json),
            "sha256": _sha256_file(plan_json),
            "schema": plan.get("schema"),
        },
        "policy_id": policy.get("policy_id"),
        "selected_atom_count": len(selected_atoms),
    }
    raw_payload = _encode_overlay_payload(runs=runs, header=header)
    policy_slug = str(policy.get("policy_id", "policy")).replace("/", "_")
    out = output_dir / policy_slug
    payload_path = out / "masks.cdo1"
    archive_path = out / "archive.zip"
    cdo1_manifest = out / "cdo1_builder_manifest.json"
    manifest_path = out / "lane12_geometry_gate_repair_candidate_manifest.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(raw_payload)
    cdo1_report = build_cdo1_candidate(
        base_archive=base_archive,
        overlay_payload=payload_path,
        output_archive=archive_path,
        manifest_json=cdo1_manifest,
        overlay_compressor=overlay_compressor,
        pack_output_payload=False,
        repo_root=repo_root,
    )
    raw_from_archive = _decompress_cdo1_member(archive_path)
    if raw_from_archive != raw_payload:
        raise Lane12GeometryRepairBuildError("archive CDO1 member does not roundtrip to raw payload")
    archive_bytes = archive_path.stat().st_size
    base_bytes = base_archive.stat().st_size
    geometry_gate_passed = (
        float(metrics["global_disagreement_after"]) <= float(max_residual_disagreement_fraction)
    )
    manifest = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_claim": False,
        "dispatch_allowed": bool(geometry_gate_passed),
        "dispatch_gate": {
            "gate": "local_decoded_mask_global_residual_disagreement",
            "passed": bool(geometry_gate_passed),
            "max_residual_disagreement_fraction": float(max_residual_disagreement_fraction),
            "exact_cuda_auth_eval_still_required": True,
        },
        "auth_eval_required": True,
        "exact_score_path": CUDA_AUTH_EVAL_PATH,
        "plan": {
            "path": str(plan_json),
            "sha256": _sha256_file(plan_json),
            "policy_id": policy.get("policy_id"),
            "policy": policy,
            "selected_atom_ids": selected_ids,
        },
        "selection": {
            "selected_atoms": atom_rows,
            "selected_pixel_count": int(selected.sum()),
            "run_count": len(runs),
        },
        "geometry_metrics": metrics,
        "base_archive": {
            "path": str(base_archive),
            "bytes": base_bytes,
            "sha256": _sha256_file(base_archive),
            "members": _archive_members(base_archive),
        },
        "overlay_payload": {
            "path": str(payload_path),
            "bytes": len(raw_payload),
            "sha256": _sha256_bytes(raw_payload),
            "compressor": overlay_compressor,
            "header": header,
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_bytes,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_base": archive_bytes - base_bytes,
            "rate_score_cost": _rate_cost(archive_bytes),
            "delta_rate_score_vs_base": _rate_cost(archive_bytes - base_bytes),
            "members": _archive_members(archive_path),
        },
        "cdo1_builder_report": cdo1_report,
    }
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--policy-id")
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overlay-compressor",
        choices=("raw", "zlib", "lzma_xz", "brotli"),
        default="lzma_xz",
    )
    parser.add_argument(
        "--max-residual-disagreement-fraction",
        type=float,
        default=DEFAULT_MAX_RESIDUAL_DISAGREEMENT_FRACTION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_lane12_geometry_gate_repair_candidate(
        plan_json=args.plan_json,
        policy_id=args.policy_id,
        base_archive=args.base_archive,
        output_dir=args.output_dir,
        overlay_compressor=args.overlay_compressor,
        max_residual_disagreement_fraction=args.max_residual_disagreement_fraction,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "archive": report["output_archive"],
                "dispatch_allowed": report["dispatch_allowed"],
                "geometry_metrics": report["geometry_metrics"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
