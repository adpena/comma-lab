#!/usr/bin/env python3
"""Resumable local-CPU equivalence receipt for the default-OFF yhat-native arm.

This harness is deliberately not a trainer, archive scorer, launch surface, or score claim.
It retains only one pair's reconstructed frames and scorer tensors at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402

SCHEMA = "yhat_native_equivalence_receipt.v1"
STATE_SCHEMA = "yhat_native_equivalence_state.v1"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
SEED = 20260719
POSE_SCORING_HEADS = frozenset(("pose",))
NONFINITE_SENTINEL = "NONFINITE_COMPARISON"
PREIMAGE_POLICY = "target_derived_minimum_norm_preimage_exact_uint8_block_solver"
F32_ADMISSIBILITY = "f32_receiver_arithmetic_exactness_admissibility_v1"
DEFAULT_SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
DEFAULT_CKPT = DEFAULT_SACRED / "levelset_witness_ema_BEST.npz"
DEFAULT_FRAMES = Path(
    "/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/selected_n24_packet/inflated/0.raw"
)
DEFAULT_PAIRS = "0,10,50,60,100,110,150,160,200,210,250,260,300,310,350,360,400,410,450,460,500,510,550,560"
DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_EVIDENCE = Path("/Volumes/VertigoDataTier/pact/evidence/yhat_native_20260719")
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


class YhatNativeMeasurementError(RuntimeError):
    """A descriptive fail-closed custody, resume, or equivalence error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _package_versions() -> dict[str, str]:
    distributions = (
        "torch",
        "timm",
        "segmentation-models-pytorch",
        "einops",
        "safetensors",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "MISSING"
    return versions


def validate_evidence_root(path: Path) -> tuple[Path, Path]:
    """Require the preflighted SSD evidence root named by this measurement."""

    resolved = path.expanduser().resolve()
    if not any(root.is_dir() and _is_relative_to(resolved, root.resolve()) for root in SSD_ROOTS):
        raise YhatNativeMeasurementError(f"evidence root must use the SSD waterfall: {resolved}")
    if _is_relative_to(resolved, DEFAULT_SACRED.resolve()):
        raise YhatNativeMeasurementError(f"evidence root may not be inside the sacred donor tree: {resolved}")
    preflight = resolved / "storage_preflight.json"
    if not preflight.is_file():
        raise YhatNativeMeasurementError(f"storage preflight is required before measurement: {preflight}")
    try:
        payload = json.loads(preflight.read_text())
    except (OSError, ValueError) as exc:
        raise YhatNativeMeasurementError(f"storage preflight is unreadable: {preflight}") from exc
    selected = Path(str(payload.get("selected_workload_root", ""))).expanduser().resolve()
    if (
        payload.get("blockers")
        or selected != resolved
        or payload.get("selected_workload_root_matches_expected") is not True
    ):
        raise YhatNativeMeasurementError(f"storage preflight does not authorize this evidence root: {preflight}")
    return resolved, preflight


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_pair_ids(raw: str) -> tuple[int, ...]:
    try:
        pairs = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise YhatNativeMeasurementError("pair ids must be comma-separated integers") from exc
    if len(pairs) < 24 or len(set(pairs)) != len(pairs) or any(pair < 0 or pair >= 600 for pair in pairs):
        raise YhatNativeMeasurementError("require at least 24 unique real pair ids in [0,600)")
    return pairs


def tree_snapshot(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    if not root.is_dir():
        return {"exists": False, "entries": 0, "metadata_sha256": None}
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        base = Path(current)
        for name in (*directories, *files):
            path = base / name
            stat = path.lstat()
            digest.update(
                f"{path.relative_to(root).as_posix()}\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
            entries += 1
    return {"exists": True, "entries": entries, "metadata_sha256": digest.hexdigest()}


def _ordered_int32(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32).astype(np.uint64)
    magnitude = bits & np.uint64(0x7FFFFFFF)
    sign = (bits & np.uint64(0x80000000)) != 0
    ordered = np.where(sign, np.uint64(0x80000000) - magnitude, np.uint64(0x80000000) + magnitude)
    return ordered.astype(np.int64)


def float32_ulp_distance(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    """Native-f32 ULP distance; nonfinite values are equal only when bit-identical."""

    a, b = np.broadcast_arrays(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32))
    equal_bits = a.view(np.uint32) == b.view(np.uint32)
    finite = np.isfinite(a) & np.isfinite(b)
    distance = np.full(a.shape, np.iinfo(np.int64).max, dtype=np.int64)
    distance[finite] = np.abs(_ordered_int32(a[finite]) - _ordered_int32(b[finite]))
    distance[equal_bits] = 0
    distance[(a == 0.0) & (b == 0.0)] = 0
    return distance


def classify_equivalence(
    *,
    rational_plane_exact: bool,
    oracle_bit_identical: bool,
    metrics_bit_identical: bool,
    native_f32_deltas_described: bool,
) -> str:
    if oracle_bit_identical and metrics_bit_identical:
        return "BIT_IDENTICAL"
    if rational_plane_exact and native_f32_deltas_described:
        return "EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS"
    return "NOT_EQUIVALENT_UNDER_NARROW_SCOPE"


def _tensor_comparison(left: Any, right: Any) -> dict[str, Any]:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            raise YhatNativeMeasurementError("frozen-oracle output mapping keys differ")
        leaves = {str(key): _tensor_comparison(left[key], right[key]) for key in sorted(left)}
        return {
            "bit_identical": all(value["bit_identical"] for value in leaves.values()),
            "max_abs_delta": max((value["max_abs_delta"] for value in leaves.values()), default=0.0),
            "max_native_f32_ulp": max((value["max_native_f32_ulp"] for value in leaves.values()), default=0),
            "has_nonfinite": any(value["has_nonfinite"] for value in leaves.values()),
            "leaves": leaves,
        }
    a = left.detach().cpu().numpy().astype(np.float32, copy=False)
    b = right.detach().cpu().numpy().astype(np.float32, copy=False)
    if a.shape != b.shape:
        raise YhatNativeMeasurementError(f"frozen-oracle tensor shapes differ: {a.shape} != {b.shape}")
    ulps = float32_ulp_distance(a, b)
    nonfinite = not (np.isfinite(a).all() and np.isfinite(b).all())
    finite = np.isfinite(a) & np.isfinite(b)
    finite_abs = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    return {
        "bit_identical": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
        "max_abs_delta": float(np.max(finite_abs, initial=0.0)),
        "max_native_f32_ulp": int(np.max(ulps, initial=0)),
        "has_nonfinite": bool(nonfinite),
        "sentinel": NONFINITE_SENTINEL if nonfinite else None,
    }


def scored_posenet_output(model: Any, pose_outputs: Mapping[str, Any], torch: Any) -> Any:
    """Concatenate only declared scored PoseNet heads in Hydra declaration order."""

    if not isinstance(pose_outputs, Mapping):
        raise YhatNativeMeasurementError("PoseNet output must be a head mapping")
    heads = getattr(getattr(getattr(model, "posenet", None), "hydra", None), "heads", None)
    if heads is None:
        raise YhatNativeMeasurementError("PoseNet Hydra head declaration is unavailable")
    selected = []
    for head in heads:
        if head.name not in POSE_SCORING_HEADS:
            continue
        if head.name not in pose_outputs:
            raise YhatNativeMeasurementError(f"missing scored PoseNet head: {head.name}")
        selected.append(pose_outputs[head.name][..., : head.out // 2])
    if not selected:
        raise YhatNativeMeasurementError("PoseNet has no declared scored heads")
    return torch.cat(selected, dim=-1)


def _finite_summary(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_summary(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_summary(item) for item in value)
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def native_f32_deltas_described(
    *,
    rational_plane_exact: bool,
    direct: Mapping[str, Any],
    native: Mapping[str, Any],
    metric_deltas: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> bool:
    """The narrow ULP class requires proofs and finite, fully recorded summaries."""

    has_nonfinite = any(
        bool(value.get("has_nonfinite")) or value.get("sentinel") == NONFINITE_SENTINEL
        for value in comparisons.values()
    )
    return bool(
        rational_plane_exact
        and _finite_summary(direct)
        and _finite_summary(native)
        and _finite_summary(metric_deltas)
        and _finite_summary(comparisons)
        and not has_nonfinite
    )


def _scorer_inputs(model: Any, torch: Any, f0: np.ndarray, f1: np.ndarray) -> tuple[Any, Any]:
    pair = torch.from_numpy(np.stack((f0, f1), axis=0)[None]).float()
    with torch.inference_mode():
        return model.preprocess_input(pair)


def _load_ladder_helpers() -> tuple[Any, Any, Any]:
    try:
        from tools.measure_yhat_rd_ladder import _distortion_from_outputs, _distortion_outputs, _load_distortion_net
    except ImportError as exc:
        raise YhatNativeMeasurementError(
            "cannot import frozen-oracle helpers from tools/measure_yhat_rd_ladder.py"
        ) from exc
    return _load_distortion_net, _distortion_outputs, _distortion_from_outputs


def validate_donor_raw_size(path: Path, pair_count: int) -> None:
    expected = 2 * pair_count * CAMERA_HW[0] * CAMERA_HW[1] * 3
    if path.stat().st_size != expected:
        raise YhatNativeMeasurementError(
            f"donor raw bytes must equal {expected} for {pair_count} pairs, got {path.stat().st_size}"
        )


def validate_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            epoch = int(stored["__epoch"].item())
            render_hw = np.asarray(stored["__render_hw"], dtype=np.int64).tolist()
    except (KeyError, OSError, ValueError) as exc:
        raise YhatNativeMeasurementError(f"cannot read sacred checkpoint metadata: {path}") from exc
    if epoch != 725 or render_hw != list(SCORER_HW):
        raise YhatNativeMeasurementError(
            f"sacred donor must be ep725 render={SCORER_HW}, got epoch={epoch} render={render_hw}"
        )
    return {"epoch": epoch, "render_hw": render_hw}


def normalized_scientific_binding(
    *,
    pair_ids: Sequence[int],
    cpu_threads: int,
    max_nodes_per_block: int,
    files: Mapping[str, Mapping[str, Any]],
    sacred: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    git_commit: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Stable scientific binding; transport-only ``--resume`` is intentionally absent."""

    return {
        "schema": STATE_SCHEMA,
        "pair_ids": list(pair_ids),
        "seed": seed,
        "cpu_threads": cpu_threads,
        "max_nodes_per_block": max_nodes_per_block,
        "dtype_order": "uint8_camera -> exact rational numerator -> numpy-fp32 frozen CPU oracle",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "package_versions": _package_versions(),
        "git_commit": git_commit,
        "files": dict(files),
        "sacred": dict(sacred),
        "checkpoint_metadata": dict(checkpoint_metadata),
    }


def _binding(args: argparse.Namespace, pairs: Sequence[int], sacred_before: Mapping[str, Any]) -> dict[str, Any]:
    try:
        args.donor_checkpoint.resolve().relative_to(args.sacred_run.resolve())
    except ValueError as exc:
        raise YhatNativeMeasurementError("donor checkpoint must remain inside the declared sacred run") from exc
    donor_packet = args.donor_frames.resolve().parent.parent
    paths = {
        "donor_checkpoint": args.donor_checkpoint,
        "donor_frames": args.donor_frames,
        "donor_archive": donor_packet / "archive.zip",
        "donor_inflate_py": donor_packet / "inflate.py",
        "donor_inflate_sh": donor_packet / "inflate.sh",
        "gt_cache": args.gt_cache,
        "upstream_modules": args.upstream / "modules.py",
        "upstream_frame_utils": args.upstream / "frame_utils.py",
        "upstream_posenet_weights": args.upstream / "models/posenet.safetensors",
        "upstream_segnet_weights": args.upstream / "models/segnet.safetensors",
        "measurement_tool": Path(__file__).resolve(),
        "lattice_module": REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        "oracle_helper": REPO / "tools/measure_yhat_rd_ladder.py",
        "storage_preflight": args.storage_preflight,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise YhatNativeMeasurementError(f"required {name} is missing: {path}")
    if args.cpu_threads < 1 or args.max_nodes_per_block < 1:
        raise YhatNativeMeasurementError("cpu threads and max nodes per block must be positive")
    if not sacred_before["exists"]:
        raise YhatNativeMeasurementError(f"sacred donor run is missing: {args.sacred_run}")
    validate_donor_raw_size(args.donor_frames, len(pairs))
    checkpoint_metadata = validate_checkpoint_metadata(args.donor_checkpoint)
    files = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()
    return normalized_scientific_binding(
        pair_ids=pairs,
        cpu_threads=args.cpu_threads,
        max_nodes_per_block=args.max_nodes_per_block,
        files=files,
        sacred={"root": str(args.sacred_run), "before": dict(sacred_before)},
        checkpoint_metadata=checkpoint_metadata,
        git_commit=git_commit,
    )


def verify_bound_files(binding: Mapping[str, Any]) -> None:
    """Re-hash every bound input/source after the run to detect in-flight drift."""

    for name, expected in binding["files"].items():
        path = Path(expected["path"])
        if not path.is_file():
            raise YhatNativeMeasurementError(f"bound file disappeared during measurement: {name}={path}")
        actual_bytes = path.stat().st_size
        actual_sha = _sha256_file(path)
        if actual_bytes != expected["bytes"] or actual_sha != expected["sha256"]:
            raise YhatNativeMeasurementError(f"bound file changed during measurement: {name}={path}")


def _scientific_stage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove first-observed environmental timing before resume-stage comparison."""

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key not in {"timing", "runtime"} and not key.endswith("_runtime_seconds")
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(payload)


def resume_binding_matches(state: Mapping[str, Any], binding: Mapping[str, Any], binding_sha256: str) -> bool:
    """True only when an existing state binds exactly the current science, not transport."""

    return (
        state.get("schema") == STATE_SCHEMA
        and state.get("binding_sha256") == binding_sha256
        and state.get("binding") == binding
    )


def _stage_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _load_completed_rows(
    *,
    state: Mapping[str, Any],
    pairs: Sequence[int],
    stages: Path,
) -> dict[int, dict[str, Any]]:
    """Load only state-bound prefix stages; an unbound crash-tail is re-derived."""

    raw_completed = state.get("completed_pairs", [])
    if not isinstance(raw_completed, list) or raw_completed != list(pairs[: len(raw_completed)]):
        raise YhatNativeMeasurementError("resume completed-pair state must be an ordered prefix")
    hashes = state.get("completed_stage_sha256", {})
    expected_keys = {str(pair_id) for pair_id in raw_completed}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_keys:
        raise YhatNativeMeasurementError("resume state lacks exact per-stage hash custody")
    expected_names = {f"pair_{pair_id:04d}.json" for pair_id in pairs}
    unexpected = sorted(path.name for path in stages.glob("pair_*.json") if path.name not in expected_names)
    if unexpected:
        raise YhatNativeMeasurementError(f"resume stages contain pair ids outside the binding: {unexpected}")
    loaded: dict[int, dict[str, Any]] = {}
    for pair_id in raw_completed:
        path = stages / f"pair_{pair_id:04d}.json"
        if not path.is_file():
            raise YhatNativeMeasurementError(f"resume state names a missing pair stage: {path}")
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise YhatNativeMeasurementError(f"resume pair stage is unreadable: {path}") from exc
        if (
            row.get("schema") != SCHEMA
            or row.get("pair_id") != pair_id
            or row.get("preimage_policy") != PREIMAGE_POLICY
            or row.get("f32_receiver_arithmetic_admissibility") != F32_ADMISSIBILITY
        ):
            raise YhatNativeMeasurementError(f"resume pair stage contract mismatch: {path}")
        if _stage_sha256(row) != hashes[str(pair_id)]:
            raise YhatNativeMeasurementError(f"resume pair stage hash mismatch: {path}")
        loaded[pair_id] = row
    return loaded


def _write_stage_once(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        stored = json.loads(path.read_text())
        if _scientific_stage_payload(stored) != _scientific_stage_payload(payload):
            raise YhatNativeMeasurementError(f"preserved stage differs from deterministic rebuild: {path}")
        return stored
    _atomic_json(path, payload)
    return dict(payload)


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise YhatNativeMeasurementError("cannot aggregate zero completed pairs")
    count = len(rows)
    exact_blocks = exact_samples = failures = disagreements = 0
    classification_counts: dict[str, int] = {}
    surfaces: dict[str, dict[str, float | int]] = {}
    for row in rows:
        for proof in row["exact_rational_planes"]:
            exact_blocks += int(proof["exact_blocks"])
            exact_samples += int(proof["exact_samples"])
            failures += int(proof["failures"])
        disagreements += int(row["segnet_argmax_disagreement"])
        classification = str(row["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        for surface, comparison in row["comparisons"].items():
            current = surfaces.setdefault(surface, {"max_abs_delta": 0.0, "max_native_f32_ulp": 0})
            current["max_abs_delta"] = max(float(current["max_abs_delta"]), float(comparison["max_abs_delta"]))
            current["max_native_f32_ulp"] = max(
                int(current["max_native_f32_ulp"]), int(comparison["max_native_f32_ulp"])
            )
    direct_mean = {key: float(sum(float(row["direct"][key]) for row in rows) / count) for key in ("d_seg", "d_pose")}
    native_mean = {
        key: float(sum(float(row["yhat_native"][key]) for row in rows) / count) for key in ("d_seg", "d_pose")
    }
    deltas = {key: native_mean[key] - direct_mean[key] for key in direct_mean}
    all_bit_identical = all(row["classification"] == "BIT_IDENTICAL" for row in rows)
    all_nonbit_described = all(
        row["classification"] == "BIT_IDENTICAL"
        or (
            row["classification"] == "EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS"
            and bool(row["native_f32_deltas_described"])
        )
        for row in rows
    )
    classification = classify_equivalence(
        rational_plane_exact=failures == 0,
        oracle_bit_identical=all_bit_identical,
        metrics_bit_identical=all_bit_identical,
        native_f32_deltas_described=all_nonbit_described,
    )
    pair_seconds = [float(row["timing"]["pair_runtime_seconds"]) for row in rows]
    solve_seconds = [float(proof["solve_runtime_seconds"]) for row in rows for proof in row["exact_rational_planes"]]
    observed_solve_per_pair = sum(solve_seconds) / count
    derived_n600_preimage_seconds = observed_solve_per_pair * 600
    return {
        "n_pairs": count,
        "exact_blocks": exact_blocks,
        "exact_samples": exact_samples,
        "failures": failures,
        "classification_counts": classification_counts,
        "classification": classification,
        "segnet_argmax_disagreements": disagreements,
        "direct_mean": direct_mean,
        "yhat_native_mean": native_mean,
        "mean_deltas": deltas,
        "surface_maxima": surfaces,
        "timing": {
            "observed_pair_seconds_sum": sum(pair_seconds),
            "observed_pair_seconds_mean": sum(pair_seconds) / count,
            "observed_pair_seconds_max": max(pair_seconds),
            "observed_solve_seconds_sum": sum(solve_seconds),
            "observed_solve_seconds_mean_per_plane": sum(solve_seconds) / len(solve_seconds),
            "derived_n600_preimage_seconds": derived_n600_preimage_seconds,
            "derived_n600_preimage_minutes": derived_n600_preimage_seconds / 60.0,
            "derived_preimage_within_30_minutes": derived_n600_preimage_seconds <= 1800.0,
            "decoder_30_minute_boundary": "NOT_PROVEN_GENERIC_EXPANDER_PACKAGING_AND_IO_OWED",
            "projection_provenance": "DERIVED_FROM_N24_MEAN_TWO_PLANE_SOLVE_RUNTIME",
        },
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    pairs = parse_pair_ids(args.pair_ids)
    args.evidence_root, args.storage_preflight = validate_evidence_root(args.evidence_root)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    state_path = args.evidence_root / "state.json"
    stages = args.evidence_root / "stages"
    sacred_before = tree_snapshot(args.sacred_run)
    binding = _binding(args, pairs, sacred_before)
    binding_sha = hashlib.sha256(_canonical(binding)).hexdigest()
    input_stage = {"schema": SCHEMA, "kind": "input_custody", "binding": binding, "binding_sha256": binding_sha}
    completed_rows: dict[int, dict[str, Any]] = {}
    if args.resume:
        if not state_path.is_file():
            raise YhatNativeMeasurementError("--resume requires an existing state.json")
        state = json.loads(state_path.read_text())
        if not resume_binding_matches(state, binding, binding_sha):
            raise YhatNativeMeasurementError("resume binding mismatch; refusing to reuse stages")
        _write_stage_once(stages / "input_custody.json", input_stage)
        completed_rows = _load_completed_rows(state=state, pairs=pairs, stages=stages)
    elif state_path.exists() or stages.exists():
        raise YhatNativeMeasurementError("evidence already exists; pass --resume or choose a new evidence root")
    else:
        _atomic_json(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "binding": binding,
                "binding_sha256": binding_sha,
                "completed_pairs": [],
                "completed_stage_sha256": {},
            },
        )
        _write_stage_once(stages / "input_custody.json", input_stage)

    from tools.measure_uint8_lattice_feasibility import stored_npy_memmap

    cache = {key: stored_npy_memmap(args.gt_cache, key) for key in ("n_pairs", "gt_f0", "gt_f1")}
    if int(np.asarray(cache["n_pairs"]).reshape(())) != 600:
        raise YhatNativeMeasurementError("GT cache must bind real n600 pairs")
    for key in ("gt_f0", "gt_f1"):
        value = cache[key]
        if value.dtype != np.uint8 or value.shape != (600, *CAMERA_HW, 3):
            raise YhatNativeMeasurementError(f"{key} must be uint8[600,874,1164,3], got {value.dtype}{value.shape}")
    donor = np.memmap(args.donor_frames, mode="r", dtype=np.uint8, shape=(2 * len(pairs), *CAMERA_HW, 3))
    load_oracle, forward, distortion = _load_ladder_helpers()
    model, torch, scorer_hashes = load_oracle(args.upstream, args.cpu_threads)
    expected_scorer_hashes = {
        "modules.py": binding["files"]["upstream_modules"]["sha256"],
        "posenet.safetensors": binding["files"]["upstream_posenet_weights"]["sha256"],
        "segnet.safetensors": binding["files"]["upstream_segnet_weights"]["sha256"],
    }
    if scorer_hashes != expected_scorer_hashes:
        raise YhatNativeMeasurementError("loaded frozen scorer hashes differ from the scientific binding")
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    rows: list[dict[str, Any]] = []
    for index, pair_id in enumerate(pairs):
        if pair_id in completed_rows:
            rows.append(completed_rows[pair_id])
            print(f"[yhat-native] pair={pair_id} resumed ({len(rows)}/{len(pairs)})", flush=True)
            continue
        pair_started = time.monotonic()
        direct0, direct1 = np.asarray(donor[2 * index]).copy(), np.asarray(donor[2 * index + 1]).copy()
        native_frames: list[np.ndarray] = []
        proofs: list[dict[str, Any]] = []
        for frame in (direct0, direct1):
            solve_started = time.monotonic()
            numerators, denominator = operator.apply_numerators(frame)
            solved = operator.solve_uint8(
                numerators / denominator,
                target_numerators=numerators,
                max_nodes_per_block=args.max_nodes_per_block,
            )
            replayed, replay_denominator = operator.apply_numerators(solved.frame)
            if (
                replay_denominator != denominator
                or not np.array_equal(replayed, numerators)
                or not solved.certified_exact
            ):
                raise YhatNativeMeasurementError(f"pair {pair_id}: exact uint8/rational realization failed")
            native_frames.append(solved.frame)
            proofs.append(
                {
                    "exact_blocks": solved.diagnostics.exact_blocks,
                    "exact_samples": int(numerators.size),
                    "failures": 0,
                    "denominator": denominator,
                    "nodes_visited": solved.diagnostics.nodes_visited,
                    "solve_runtime_seconds": time.monotonic() - solve_started,
                }
            )
        direct_outputs = forward(model, torch, direct0, direct1)
        native_outputs = forward(model, torch, native_frames[0], native_frames[1])
        direct_inputs = _scorer_inputs(model, torch, direct0, direct1)
        native_inputs = _scorer_inputs(model, torch, native_frames[0], native_frames[1])
        source_outputs = forward(
            model, torch, np.asarray(cache["gt_f0"][pair_id]).copy(), np.asarray(cache["gt_f1"][pair_id]).copy()
        )
        direct_metrics = distortion(model, direct_outputs, source_outputs)
        native_metrics = distortion(model, native_outputs, source_outputs)
        comparisons = {
            "scorer_input_posenet_yuv6": _tensor_comparison(direct_inputs[0], native_inputs[0]),
            "scorer_input_segnet_rgb": _tensor_comparison(direct_inputs[1], native_inputs[1]),
            "posenet_scored_output": _tensor_comparison(
                scored_posenet_output(model, direct_outputs[0], torch),
                scored_posenet_output(model, native_outputs[0], torch),
            ),
            "segnet_logits": _tensor_comparison(direct_outputs[1], native_outputs[1]),
        }
        metric_deltas = {key: native_metrics[key] - direct_metrics[key] for key in direct_metrics}
        rational_plane_exact = all(proof["failures"] == 0 for proof in proofs)
        described = native_f32_deltas_described(
            rational_plane_exact=rational_plane_exact,
            direct=direct_metrics,
            native=native_metrics,
            metric_deltas=metric_deltas,
            comparisons=comparisons,
        )
        row = {
            "schema": SCHEMA,
            "pair_id": pair_id,
            "exact_rational_planes": proofs,
            "preimage_policy": PREIMAGE_POLICY,
            "f32_receiver_arithmetic_admissibility": F32_ADMISSIBILITY,
            "direct": direct_metrics,
            "yhat_native": native_metrics,
            "metric_deltas": metric_deltas,
            "segnet_argmax_disagreement": int(
                np.count_nonzero(
                    np.argmax(direct_outputs[1].detach().cpu().numpy(), axis=1)
                    != np.argmax(native_outputs[1].detach().cpu().numpy(), axis=1)
                )
            ),
            "comparisons": comparisons,
            "native_f32_deltas_described": described,
            "classification": classify_equivalence(
                rational_plane_exact=rational_plane_exact,
                oracle_bit_identical=all(
                    value["bit_identical"] and not value["has_nonfinite"] for value in comparisons.values()
                ),
                metrics_bit_identical=all(
                    np.float64(direct_metrics[key]).view(np.uint64) == np.float64(native_metrics[key]).view(np.uint64)
                    for key in direct_metrics
                ),
                native_f32_deltas_described=described,
            ),
            "timing": {"pair_runtime_seconds": time.monotonic() - pair_started},
        }
        row = _write_stage_once(stages / f"pair_{pair_id:04d}.json", row)
        rows.append(row)
        _atomic_json(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "binding": binding,
                "binding_sha256": binding_sha,
                "completed_pairs": [item["pair_id"] for item in rows],
                "completed_stage_sha256": {str(item["pair_id"]): _stage_sha256(item) for item in rows},
            },
        )
        print(f"[yhat-native] pair={pair_id} complete ({len(rows)}/{len(pairs)})", flush=True)
    verify_bound_files(binding)
    sacred_after = tree_snapshot(args.sacred_run)
    if sacred_after != sacred_before:
        raise YhatNativeMeasurementError("sacred donor run changed during measurement")
    receipt = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        "verdict_scope": "n>=24 supplied real pairs; frozen CPU oracle only; no archive bytes, other receiver, hardware, batch, score, promotion, launch, or pointer authority",
        "binding": binding,
        "scorer_hashes": scorer_hashes,
        "runtime": {
            "total_runtime_seconds": time.monotonic() - started,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seed": SEED,
        },
        "pairs": rows,
        "aggregate": aggregate_rows(rows),
        "authority": {"score_claim": False, "promotion": False, "launch": False, "pointer_movement": False},
        "sacred_after": sacred_after,
    }
    receipt = _write_stage_once(stages / "aggregate_verification.json", receipt)
    _atomic_json(args.evidence_root / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--sacred-run", type=Path, default=DEFAULT_SACRED)
    parser.add_argument("--donor-frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--pair-ids", default=DEFAULT_PAIRS)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-nodes-per-block", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = _run(args)
    except YhatNativeMeasurementError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"receipt": str(args.evidence_root / "receipt.json"), "pairs": receipt["aggregate"]["n_pairs"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
