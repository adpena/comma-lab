#!/usr/bin/env python3
"""Plan compressed-byte water-fill policies for C067 postdecode mask repair.

This is an offline selector for charged AMR1 postdecode mask-repair atoms. It
consumes diagnostic component traces plus C067 postdecode repair manifests and
emits deterministic, explicitly non-promotable budget policies. It does not
decode masks, build archives, run scorers, or dispatch remote jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c067_postdecode_mask_repair_waterfill.py"
SCHEMA = "c067_postdecode_mask_repair_waterfill_plan_v1"
EXPECTED_SAMPLES = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NON_PROMOTABLE_WARNING = (
    "This artifact is planning signal only. It cannot promote, rank, kill, "
    "retire, or support a score claim until a closed archive is evaluated "
    "through exact CUDA auth eval on identical archive bytes."
)

C067_FRONTIER_SCORE = 0.31561703078448233
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/c067_postdecode_mask_repair_candidate_20260502/"
    "c067_postdecode_mask_repair_waterfill_plan.json"
)
DEFAULT_BUDGETS = (4_000, 8_000, 12_000)


class WaterfillPlanError(ValueError):
    """Raised when planner inputs fail schema or custody checks."""


@dataclass(frozen=True)
class TracePairSignal:
    pair_index: int
    pose_score_term: float
    seg_score_term: float
    combined_score_term: float
    sources: tuple[str, ...]


@dataclass
class AtomRecord:
    atom_id: str
    frames: tuple[int, ...]
    pair_indices: tuple[int, ...]
    class_id: int
    changed_pixels: int
    manifest_paths: set[str] = field(default_factory=set)
    byte_observations: list[float] = field(default_factory=list)
    fallback_byte_observations: list[float] = field(default_factory=list)
    trace_pair_indices: tuple[int, ...] = ()
    pose_score_term: float = 0.0
    seg_score_term: float = 0.0
    combined_score_term: float = 0.0
    estimated_payload_bytes: float = 0.0
    byte_estimate_method: str = "unestimated"

    def freeze_json(self) -> dict[str, Any]:
        rate_cost = self.estimated_payload_bytes * RATE_SCORE_PER_BYTE
        return {
            "atom_id": self.atom_id,
            "frames": list(self.frames),
            "pair_indices": list(self.pair_indices),
            "trace_pair_indices": list(self.trace_pair_indices),
            "class_id": self.class_id,
            "changed_pixels": self.changed_pixels,
            "estimated_payload_bytes": round(self.estimated_payload_bytes, 6),
            "byte_estimate_method": self.byte_estimate_method,
            "expected_pose_score_improvement_first_order": round(self.pose_score_term, 12),
            "expected_seg_score_improvement_proxy": round(self.seg_score_term, 12),
            "expected_component_score_improvement_first_order": round(self.combined_score_term, 12),
            "expected_rate_score_cost": round(rate_cost, 12),
            "expected_net_score_delta_if_prior_realizes": round(rate_cost - self.combined_score_term, 12),
            "benefit_per_payload_byte": (
                round(self.combined_score_term / self.estimated_payload_bytes, 12)
                if self.estimated_payload_bytes > 0.0
                else None
            ),
            "source_manifest_count": len(self.manifest_paths),
        }


@dataclass
class PolicyUnit:
    unit_id: str
    policy_kind: str
    policy_values: tuple[int, ...]
    atom_granularity: str
    atoms: list[AtomRecord]

    @property
    def estimated_payload_bytes(self) -> float:
        return sum(atom.estimated_payload_bytes for atom in self.atoms)

    @property
    def pose_score_term(self) -> float:
        return sum(atom.pose_score_term for atom in self.atoms)

    @property
    def seg_score_term(self) -> float:
        return sum(atom.seg_score_term for atom in self.atoms)

    @property
    def combined_score_term(self) -> float:
        return sum(atom.combined_score_term for atom in self.atoms)

    @property
    def changed_pixels(self) -> int:
        return sum(atom.changed_pixels for atom in self.atoms)

    def freeze_json(self) -> dict[str, Any]:
        bytes_ = self.estimated_payload_bytes
        rate_cost = bytes_ * RATE_SCORE_PER_BYTE
        return {
            "unit_id": self.unit_id,
            "policy_kind": self.policy_kind,
            "policy_values": list(self.policy_values),
            "atom_granularity": self.atom_granularity,
            "atom_count": len(self.atoms),
            "atom_ids": [atom.atom_id for atom in self.atoms],
            "changed_pixels": self.changed_pixels,
            "estimated_payload_bytes": round(bytes_, 6),
            "expected_pose_score_improvement_first_order": round(self.pose_score_term, 12),
            "expected_seg_score_improvement_proxy": round(self.seg_score_term, 12),
            "expected_component_score_improvement_first_order": round(self.combined_score_term, 12),
            "expected_rate_score_cost": round(rate_cost, 12),
            "expected_net_score_delta_if_prior_realizes": round(rate_cost - self.combined_score_term, 12),
            "benefit_per_payload_byte": round(self.combined_score_term / bytes_, 12) if bytes_ > 0.0 else None,
        }


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WaterfillPlanError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise WaterfillPlanError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WaterfillPlanError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise WaterfillPlanError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WaterfillPlanError(f"{field} must be an integer")
    return int(value)


def _int_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise WaterfillPlanError(f"{field} must be a list of integers")
    if any(item < 0 for item in value):
        raise WaterfillPlanError(f"{field} contains negative values")
    return tuple(int(item) for item in value)


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError("label before '=' must be non-empty")
        return label, Path(path_text)
    path = Path(value)
    return path.stem, path


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values: set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("expected comma-separated positive integers")
        values.add(parsed)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one positive integer")
    return tuple(sorted(values))


def _trace_pose_term(sample: Mapping[str, Any], *, avg_pose: float, n_samples: int, field_prefix: str) -> float:
    explicit = sample.get("score_pose_contribution_first_order")
    if explicit is not None:
        return _finite_float(explicit, field=f"{field_prefix}.score_pose_contribution_first_order")
    pose = _finite_float(sample.get("posenet_dist"), field=f"{field_prefix}.posenet_dist")
    if avg_pose <= 0.0:
        return 0.0
    return (5.0 / math.sqrt(10.0 * avg_pose)) * (pose / n_samples)


def _trace_seg_term(sample: Mapping[str, Any], *, n_samples: int, field_prefix: str) -> float:
    explicit = sample.get("score_seg_contribution_exact")
    if explicit is not None:
        return _finite_float(explicit, field=f"{field_prefix}.score_seg_contribution_exact")
    seg = _finite_float(sample.get("segnet_dist"), field=f"{field_prefix}.segnet_dist")
    return 100.0 * seg / n_samples


def _load_component_trace(
    path: Path,
    *,
    label: str,
    expected_samples: int | None,
    require_cross_check: bool,
) -> dict[str, Any]:
    payload = _read_json_object(path, label=f"component trace {label}")
    if payload.get("score_claim") is not False:
        raise WaterfillPlanError(f"{path}: component trace score_claim must be false")
    if payload.get("evidence_grade") != "diagnostic_component_trace":
        raise WaterfillPlanError(f"{path}: evidence_grade must be diagnostic_component_trace")
    n_samples = _int_value(payload.get("n_samples"), field=f"{label}.n_samples")
    if expected_samples is not None and n_samples != expected_samples:
        raise WaterfillPlanError(f"{path}: n_samples must be {expected_samples}, got {n_samples}")
    cross = payload.get("contest_auth_eval_cross_check")
    if require_cross_check and (not isinstance(cross, Mapping) or cross.get("all_match") is not True):
        raise WaterfillPlanError(f"{path}: component trace must cross-check contest_auth_eval.json")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != n_samples:
        raise WaterfillPlanError(f"{path}: samples must contain n_samples rows")
    avg_pose = _finite_float(payload.get("avg_posenet_dist"), field=f"{label}.avg_posenet_dist")
    by_pair: dict[int, TracePairSignal] = {}
    seen: set[int] = set()
    for row, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise WaterfillPlanError(f"{path}: samples[{row}] must be an object")
        pair = _int_value(sample.get("pair_index"), field=f"{label}.samples[{row}].pair_index")
        if pair < 0 or pair >= n_samples:
            raise WaterfillPlanError(f"{path}: samples[{row}].pair_index out of range")
        if pair in seen:
            raise WaterfillPlanError(f"{path}: duplicate pair_index={pair}")
        seen.add(pair)
        prefix = f"{label}.samples[{row}]"
        pose_term = max(0.0, _trace_pose_term(sample, avg_pose=avg_pose, n_samples=n_samples, field_prefix=prefix))
        seg_term = max(0.0, _trace_seg_term(sample, n_samples=n_samples, field_prefix=prefix))
        combined = sample.get("score_combined_contribution_first_order")
        if combined is None:
            combined_term = pose_term + seg_term
        else:
            combined_term = max(
                0.0,
                _finite_float(combined, field=f"{prefix}.score_combined_contribution_first_order"),
            )
        by_pair[pair] = TracePairSignal(
            pair_index=pair,
            pose_score_term=pose_term,
            seg_score_term=seg_term,
            combined_score_term=combined_term,
            sources=(label,),
        )
    if seen != set(range(n_samples)):
        missing = sorted(set(range(n_samples)) - seen)[:8]
        raise WaterfillPlanError(f"{path}: missing component trace pair indices: {missing}")
    return {
        "label": label,
        "path": path,
        "file": _file_meta(path),
        "n_samples": n_samples,
        "archive_size_bytes": int(
            _finite_float(payload.get("archive_size_bytes"), field=f"{label}.archive_size_bytes")
        ),
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": _finite_float(payload.get("avg_segnet_dist"), field=f"{label}.avg_segnet_dist"),
        "score_recomputed_from_components": _finite_float(
            payload.get("score_recomputed_from_components"),
            field=f"{label}.score_recomputed_from_components",
        ),
        "archive_sha256": (payload.get("trace_inputs") or {}).get("archive_sha256"),
        "hardware": {
            "device": (payload.get("trace_inputs") or {}).get("device"),
            "cuda_device_name": (payload.get("trace_inputs") or {}).get("cuda_device_name"),
            "torch_version": (payload.get("trace_inputs") or {}).get("torch_version"),
            "torch_cuda_version": (payload.get("trace_inputs") or {}).get("torch_cuda_version"),
        },
        "pairs": by_pair,
    }


def _aggregate_traces(traces: list[dict[str, Any]], *, mode: str) -> dict[int, TracePairSignal]:
    if not traces:
        raise WaterfillPlanError("at least one component trace is required")
    pair_sets = [set(trace["pairs"]) for trace in traces]
    if any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
        raise WaterfillPlanError("component traces must cover identical pair indices")
    out: dict[int, TracePairSignal] = {}
    for pair in sorted(pair_sets[0]):
        signals = [trace["pairs"][pair] for trace in traces]
        sources = tuple(signal.sources[0] for signal in signals)
        if mode == "max":
            selected = max(
                signals,
                key=lambda signal: (
                    signal.combined_score_term,
                    signal.pose_score_term,
                    signal.seg_score_term,
                    -signal.pair_index,
                ),
            )
            out[pair] = TracePairSignal(
                pair_index=pair,
                pose_score_term=selected.pose_score_term,
                seg_score_term=selected.seg_score_term,
                combined_score_term=selected.combined_score_term,
                sources=sources,
            )
        elif mode == "mean":
            denom = float(len(signals))
            out[pair] = TracePairSignal(
                pair_index=pair,
                pose_score_term=sum(signal.pose_score_term for signal in signals) / denom,
                seg_score_term=sum(signal.seg_score_term for signal in signals) / denom,
                combined_score_term=sum(signal.combined_score_term for signal in signals) / denom,
                sources=sources,
            )
        else:
            raise WaterfillPlanError(f"unsupported trace aggregation mode {mode!r}")
    return out


def _load_repair_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path, label="repair manifest")
    if payload.get("schema") != "c067_postdecode_mask_repair_candidate_v1":
        raise WaterfillPlanError(f"{path}: unsupported schema={payload.get('schema')!r}")
    if payload.get("score_claim") is not False:
        raise WaterfillPlanError(f"{path}: manifest score_claim must be false")
    if payload.get("promotion_eligible") is not False:
        raise WaterfillPlanError(f"{path}: manifest promotion_eligible must be false")
    selector = payload.get("repair_selector")
    repair = payload.get("repair_payload")
    archive = payload.get("archive")
    if not isinstance(selector, Mapping) or not isinstance(repair, Mapping) or not isinstance(archive, Mapping):
        raise WaterfillPlanError(f"{path}: missing repair_selector, repair_payload, or archive object")
    atoms = selector.get("selected_atoms")
    if not isinstance(atoms, list):
        raise WaterfillPlanError(f"{path}: repair_selector.selected_atoms must be a list")
    compressed_bytes = int(
        _finite_float(repair.get("compressed_size_bytes"), field=f"{path}.repair_payload.compressed_size_bytes")
    )
    policy = selector.get("policy")
    if not isinstance(policy, Mapping):
        raise WaterfillPlanError(f"{path}: repair_selector.policy must be an object")
    atom_granularity = str(policy.get("atom_granularity") or selector.get("atom_granularity") or "")
    if atom_granularity not in {"frame_class", "pair_class"}:
        raise WaterfillPlanError(f"{path}: atom_granularity must be frame_class or pair_class")
    sequence: list[AtomRecord] = []
    for idx, raw in enumerate(atoms):
        if not isinstance(raw, Mapping):
            raise WaterfillPlanError(f"{path}: selected_atoms[{idx}] must be an object")
        atom_id = raw.get("atom_id")
        if not isinstance(atom_id, str) or not atom_id:
            raise WaterfillPlanError(f"{path}: selected_atoms[{idx}].atom_id must be non-empty")
        changed = _int_value(raw.get("changed_pixels"), field=f"{path}.selected_atoms[{idx}].changed_pixels")
        if changed <= 0:
            raise WaterfillPlanError(f"{path}: selected_atoms[{idx}].changed_pixels must be positive")
        class_id = _int_value(raw.get("class_id"), field=f"{path}.selected_atoms[{idx}].class_id")
        sequence.append(
            AtomRecord(
                atom_id=atom_id,
                frames=_int_tuple(raw.get("frames"), field=f"{path}.selected_atoms[{idx}].frames"),
                pair_indices=_int_tuple(raw.get("pair_indices"), field=f"{path}.selected_atoms[{idx}].pair_indices"),
                class_id=class_id,
                changed_pixels=changed,
                manifest_paths={str(path)},
            )
        )
    return {
        "path": path,
        "file": _file_meta(path),
        "payload": payload,
        "sequence": sequence,
        "atom_ids": [atom.atom_id for atom in sequence],
        "atom_granularity": atom_granularity,
        "compressor": str(repair.get("compressor")),
        "repair_member": repair.get("archive_member"),
        "compressed_payload_bytes": compressed_bytes,
        "selected_repair_pixels": int(
            _finite_float(selector.get("selected_repair_pixels"), field=f"{path}.selected_repair_pixels")
        ),
        "archive": {
            "path": (archive.get("path")),
            "size_bytes": int(_finite_float(archive.get("size_bytes"), field=f"{path}.archive.size_bytes")),
            "sha256": archive.get("sha256"),
            "delta_vs_base_bytes": archive.get("delta_vs_base_bytes"),
            "rate_term_delta_vs_base": archive.get("rate_term_delta_vs_base"),
        },
        "policy": dict(policy),
    }


def _same_contract(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["atom_granularity"] == right["atom_granularity"]
        and left["compressor"] == right["compressor"]
        and left["repair_member"] == right["repair_member"]
    )


def _add_weighted_observation(atoms: Iterable[AtomRecord], *, total_bytes: float) -> None:
    rows = list(atoms)
    pixels = sum(atom.changed_pixels for atom in rows)
    if pixels <= 0:
        return
    for atom in rows:
        atom.byte_observations.append(float(total_bytes) * atom.changed_pixels / pixels)


def _add_fallback_observation(atoms: Iterable[AtomRecord], *, total_bytes: float) -> None:
    rows = list(atoms)
    pixels = sum(atom.changed_pixels for atom in rows)
    if pixels <= 0:
        return
    for atom in rows:
        atom.fallback_byte_observations.append(float(total_bytes) * atom.changed_pixels / pixels)


def _collect_atoms_and_bytes(manifests: list[dict[str, Any]]) -> dict[str, AtomRecord]:
    atoms: dict[str, AtomRecord] = {}
    for manifest in manifests:
        for atom in manifest["sequence"]:
            existing = atoms.get(atom.atom_id)
            if existing is None:
                atoms[atom.atom_id] = atom
            else:
                if (
                    existing.frames != atom.frames
                    or existing.pair_indices != atom.pair_indices
                    or existing.class_id != atom.class_id
                ):
                    raise WaterfillPlanError(f"conflicting atom identity for {atom.atom_id}")
                existing.manifest_paths.update(atom.manifest_paths)

    for manifest in manifests:
        _add_fallback_observation(
            [atoms[atom.atom_id] for atom in manifest["sequence"]],
            total_bytes=manifest["compressed_payload_bytes"],
        )

    sorted_manifests = sorted(
        manifests,
        key=lambda item: (
            item["atom_granularity"],
            item["compressor"],
            item["repair_member"] or "",
            len(item["sequence"]),
            item["compressed_payload_bytes"],
            str(item["path"]),
        ),
    )
    for current in sorted_manifests:
        prefixes = [
            previous
            for previous in sorted_manifests
            if previous is not current
            and _same_contract(previous, current)
            and len(previous["atom_ids"]) < len(current["atom_ids"])
            and current["atom_ids"][: len(previous["atom_ids"])] == previous["atom_ids"]
        ]
        previous = max(prefixes, key=lambda item: len(item["atom_ids"]), default=None)
        if previous is None:
            segment = current["sequence"]
            delta_bytes = current["compressed_payload_bytes"]
        else:
            segment = current["sequence"][len(previous["atom_ids"]) :]
            delta_bytes = current["compressed_payload_bytes"] - previous["compressed_payload_bytes"]
        if delta_bytes > 0 and segment:
            _add_weighted_observation(
                [atoms[atom.atom_id] for atom in segment],
                total_bytes=float(delta_bytes),
            )

    for atom in atoms.values():
        if atom.byte_observations:
            atom.estimated_payload_bytes = float(statistics.median(atom.byte_observations))
            atom.byte_estimate_method = "nested_prefix_marginal_median"
        elif atom.fallback_byte_observations:
            atom.estimated_payload_bytes = float(statistics.median(atom.fallback_byte_observations))
            atom.byte_estimate_method = "policy_weighted_median"
        else:
            raise WaterfillPlanError(f"no byte observation for atom {atom.atom_id}")
    return atoms


def _trace_pairs_for_atom(atom: AtomRecord, *, atom_trace_map: str, atom_granularity: str) -> tuple[int, ...]:
    if atom_trace_map == "atom-pair-indices":
        pairs = atom.pair_indices
    elif atom_trace_map == "mask-frame-is-pair-index":
        pairs = atom.frames
    elif atom_trace_map == "auto":
        pairs = atom.frames if atom_granularity == "frame_class" else atom.pair_indices
    else:
        raise WaterfillPlanError(f"unsupported atom_trace_map {atom_trace_map!r}")
    return tuple(sorted(set(int(pair) for pair in pairs if pair >= 0)))


def _assign_trace_signal(
    atoms: dict[str, AtomRecord],
    pair_signals: Mapping[int, TracePairSignal],
    *,
    atom_trace_map: str,
    atom_granularity: str,
) -> None:
    pair_pixel_totals: dict[int, int] = {}
    atom_pairs: dict[str, tuple[int, ...]] = {}
    for atom in atoms.values():
        pairs = _trace_pairs_for_atom(atom, atom_trace_map=atom_trace_map, atom_granularity=atom_granularity)
        pairs = tuple(pair for pair in pairs if pair in pair_signals)
        atom_pairs[atom.atom_id] = pairs
        for pair in pairs:
            pair_pixel_totals[pair] = pair_pixel_totals.get(pair, 0) + atom.changed_pixels

    for atom in atoms.values():
        pairs = atom_pairs[atom.atom_id]
        atom.trace_pair_indices = pairs
        pose = 0.0
        seg = 0.0
        combined = 0.0
        for pair in pairs:
            total_pixels = pair_pixel_totals.get(pair, 0)
            if total_pixels <= 0:
                continue
            fraction = atom.changed_pixels / total_pixels
            signal = pair_signals[pair]
            pose += signal.pose_score_term * fraction
            seg += signal.seg_score_term * fraction
            combined += signal.combined_score_term * fraction
        atom.pose_score_term = pose
        atom.seg_score_term = seg
        atom.combined_score_term = combined


def _policy_units(atoms: Iterable[AtomRecord], *, atom_granularity: str) -> list[PolicyUnit]:
    grouped: dict[tuple[str, tuple[int, ...]], list[AtomRecord]] = {}
    if atom_granularity == "frame_class":
        policy_kind = "frame_indices"
        for atom in atoms:
            key = (policy_kind, tuple(sorted(atom.frames)))
            grouped.setdefault(key, []).append(atom)
    elif atom_granularity == "pair_class":
        policy_kind = "pair_indices"
        for atom in atoms:
            key = (policy_kind, tuple(sorted(atom.pair_indices)))
            grouped.setdefault(key, []).append(atom)
    else:
        raise WaterfillPlanError(f"unsupported atom_granularity {atom_granularity!r}")
    units: list[PolicyUnit] = []
    for (policy_kind, values), rows in grouped.items():
        if policy_kind == "frame_indices":
            unit_id = "frame" + "_".join(f"{value:04d}" for value in values)
        else:
            unit_id = "pair" + "_".join(f"{value:04d}" for value in values)
        units.append(
            PolicyUnit(
                unit_id=unit_id,
                policy_kind=policy_kind,
                policy_values=values,
                atom_granularity=atom_granularity,
                atoms=sorted(rows, key=lambda atom: atom.atom_id),
            )
        )
    units.sort(key=_rank_unit_key)
    return units


def _rank_atom_key(atom: AtomRecord) -> tuple[float, float, float, int, str]:
    benefit_per_byte = atom.combined_score_term / atom.estimated_payload_bytes if atom.estimated_payload_bytes > 0 else 0.0
    net = atom.estimated_payload_bytes * RATE_SCORE_PER_BYTE - atom.combined_score_term
    return (-benefit_per_byte, net, -atom.combined_score_term, -atom.changed_pixels, atom.atom_id)


def _rank_unit_key(unit: PolicyUnit) -> tuple[float, float, float, int, str]:
    bytes_ = unit.estimated_payload_bytes
    benefit_per_byte = unit.combined_score_term / bytes_ if bytes_ > 0 else 0.0
    net = bytes_ * RATE_SCORE_PER_BYTE - unit.combined_score_term
    return (-benefit_per_byte, net, -unit.combined_score_term, -unit.changed_pixels, unit.unit_id)


def _policy_for_budget(
    *,
    budget: int,
    ranked_units: list[PolicyUnit],
    label_prefix: str,
    atom_granularity: str,
) -> dict[str, Any]:
    selected: list[PolicyUnit] = []
    used = 0.0
    for unit in ranked_units:
        unit_bytes = unit.estimated_payload_bytes
        if unit.combined_score_term <= 0.0:
            continue
        if used + unit_bytes <= float(budget):
            selected.append(unit)
            used += unit_bytes
    atoms = [atom for unit in selected for atom in unit.atoms]
    pose = sum(unit.pose_score_term for unit in selected)
    seg = sum(unit.seg_score_term for unit in selected)
    combined = sum(unit.combined_score_term for unit in selected)
    rate = used * RATE_SCORE_PER_BYTE
    frame_values = sorted({value for unit in selected if unit.policy_kind == "frame_indices" for value in unit.policy_values})
    pair_values = sorted({value for unit in selected if unit.policy_kind == "pair_indices" for value in unit.policy_values})
    if atom_granularity == "frame_class":
        policy_kind = "frame_indices"
        builder_args = ["--policy", "frame_indices", "--frame-indices", ",".join(str(value) for value in frame_values)]
    else:
        policy_kind = "pair_indices"
        builder_args = ["--policy", "pair_indices", "--pair-indices", ",".join(str(value) for value in pair_values)]
    policy_json = {
        "label": f"{label_prefix}_budget{budget}",
        "policy": policy_kind,
        "atom_granularity": atom_granularity,
        "max_atoms": None,
        "max_repair_payload_bytes": budget,
        "pair_indices": pair_values,
        "frame_indices": frame_values,
        "class_ids": [],
    }
    return {
        "policy_id": f"{label_prefix}_budget{budget}",
        "score_claim": False,
        "promotion_eligible": False,
        "budget_payload_bytes": int(budget),
        "estimated_payload_bytes": round(used, 6),
        "budget_slack_bytes": round(float(budget) - used, 6),
        "selected_unit_count": len(selected),
        "selected_atom_count": len(atoms),
        "selected_repair_pixels": sum(atom.changed_pixels for atom in atoms),
        "selected_units": [unit.freeze_json() for unit in selected],
        "selected_atoms": [atom.freeze_json() for atom in sorted(atoms, key=lambda atom: atom.atom_id)],
        "expected_marginal_score_terms": {
            "pose_score_improvement_first_order": round(pose, 12),
            "seg_score_improvement_proxy": round(seg, 12),
            "component_score_improvement_first_order": round(combined, 12),
            "rate_score_cost": round(rate, 12),
            "net_score_delta_if_component_prior_realizes": round(rate - combined, 12),
            "break_even_under_first_order_prior": combined > rate,
        },
        "builder_contract": {
            "tool": "experiments/build_c067_postdecode_mask_repair_candidate.py",
            "policy_json": policy_json,
            "cli_args_fragment": builder_args + ["--max-repair-payload-bytes", str(budget)],
            "compatibility_note": (
                "The current builder selects frame or pair policy units. selected_atoms "
                "records the planning atom set inside those units."
            ),
        },
    }


def build_waterfill_plan(
    *,
    component_trace_specs: list[tuple[str, Path]],
    manifest_paths: list[Path],
    output_json: Path | None = None,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    expected_samples: int | None = EXPECTED_SAMPLES,
    require_cross_check: bool = True,
    trace_aggregation: str = "max",
    atom_trace_map: str = "auto",
    label_prefix: str = "c067_postdecode_waterfill",
    top_k: int = 64,
) -> dict[str, Any]:
    if not component_trace_specs:
        raise WaterfillPlanError("component_trace_specs must be non-empty")
    if not manifest_paths:
        raise WaterfillPlanError("manifest_paths must be non-empty")
    if any(budget <= 0 for budget in budgets):
        raise WaterfillPlanError("budgets must be positive")
    if top_k <= 0:
        raise WaterfillPlanError("top_k must be positive")

    traces = [
        _load_component_trace(
            path,
            label=label,
            expected_samples=expected_samples,
            require_cross_check=require_cross_check,
        )
        for label, path in component_trace_specs
    ]
    pair_signals = _aggregate_traces(traces, mode=trace_aggregation)
    manifests = [_load_repair_manifest(path) for path in manifest_paths]
    granularities = {manifest["atom_granularity"] for manifest in manifests}
    if len(granularities) != 1:
        raise WaterfillPlanError(f"all manifests must use one atom_granularity, got {sorted(granularities)}")
    atom_granularity = next(iter(granularities))
    atoms = _collect_atoms_and_bytes(manifests)
    _assign_trace_signal(
        atoms,
        pair_signals,
        atom_trace_map=atom_trace_map,
        atom_granularity=atom_granularity,
    )
    ranked_atoms = sorted(atoms.values(), key=_rank_atom_key)
    ranked_units = _policy_units(ranked_atoms, atom_granularity=atom_granularity)
    policies = [
        _policy_for_budget(
            budget=budget,
            ranked_units=ranked_units,
            label_prefix=label_prefix,
            atom_granularity=atom_granularity,
        )
        for budget in sorted(set(budgets))
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_planning_non_promotable",
        "cuda_jobs_launched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
        "frontier_reference": {
            "label": "C067_A++_frontier",
            "score": C067_FRONTIER_SCORE,
            "archive_size_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "archive_sha256": C067_FRONTIER_ARCHIVE_SHA256,
        },
        "configuration": {
            "budgets_payload_bytes": list(sorted(set(budgets))),
            "trace_aggregation": trace_aggregation,
            "atom_trace_map": atom_trace_map,
            "atom_granularity": atom_granularity,
            "rate_score_per_payload_byte_proxy": RATE_SCORE_PER_BYTE,
            "label_prefix": label_prefix,
            "top_k": top_k,
        },
        "inputs": {
            "component_traces": [
                {
                    "label": trace["label"],
                    **trace["file"],
                    "n_samples": trace["n_samples"],
                    "archive_size_bytes": trace["archive_size_bytes"],
                    "archive_sha256": trace["archive_sha256"],
                    "score_recomputed_from_components": trace["score_recomputed_from_components"],
                    "avg_posenet_dist": trace["avg_posenet_dist"],
                    "avg_segnet_dist": trace["avg_segnet_dist"],
                    "hardware": trace["hardware"],
                }
                for trace in traces
            ],
            "repair_manifests": [
                {
                    **manifest["file"],
                    "atom_granularity": manifest["atom_granularity"],
                    "compressor": manifest["compressor"],
                    "repair_member": manifest["repair_member"],
                    "selected_atom_count": len(manifest["sequence"]),
                    "selected_repair_pixels": manifest["selected_repair_pixels"],
                    "compressed_payload_bytes": manifest["compressed_payload_bytes"],
                    "archive": manifest["archive"],
                    "policy": manifest["policy"],
                }
                for manifest in manifests
            ],
        },
        "atom_count": len(ranked_atoms),
        "policy_unit_count": len(ranked_units),
        "top_atoms": [atom.freeze_json() for atom in ranked_atoms[:top_k]],
        "top_policy_units": [unit.freeze_json() for unit in ranked_units[:top_k]],
        "budget_policies": policies,
        "next_step": (
            "Inspect budget_policies, build any selected policy with "
            "experiments/build_c067_postdecode_mask_repair_candidate.py, then require "
            "exact CUDA auth eval before making any score or promotion claim."
        ),
    }
    if output_json is not None:
        _write_json(output_json, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-trace",
        action="append",
        type=_parse_label_path,
        required=True,
        help="Diagnostic component trace as LABEL=PATH or PATH. May be repeated.",
    )
    parser.add_argument(
        "--archive-manifest",
        action="append",
        type=Path,
        required=True,
        help="C067 postdecode repair manifest. May be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget-bytes", type=_parse_positive_ints, default=DEFAULT_BUDGETS)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    parser.add_argument(
        "--allow-uncrosschecked-traces",
        action="store_true",
        help="Allow diagnostic traces without contest_auth_eval_cross_check.all_match=true.",
    )
    parser.add_argument("--trace-aggregation", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--atom-trace-map",
        choices=("auto", "mask-frame-is-pair-index", "atom-pair-indices"),
        default="auto",
    )
    parser.add_argument("--label-prefix", default="c067_postdecode_waterfill")
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_waterfill_plan(
        component_trace_specs=args.component_trace,
        manifest_paths=args.archive_manifest,
        output_json=args.output_json,
        budgets=args.budget_bytes,
        expected_samples=args.expected_samples,
        require_cross_check=not args.allow_uncrosschecked_traces,
        trace_aggregation=args.trace_aggregation,
        atom_trace_map=args.atom_trace_map,
        label_prefix=args.label_prefix,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "atom_count": payload["atom_count"],
                "policy_unit_count": payload["policy_unit_count"],
                "budget_policy_count": len(payload["budget_policies"]),
                "score_claim": payload["score_claim"],
                "promotion_eligible": payload["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
