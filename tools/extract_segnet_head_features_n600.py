#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Governed, resumable n600 frozen-SegNet head-feature extraction.

Large arrays are admitted only on the approved SSD waterfall and a real
governed-launch marker is mandatory outside tests.  Each canonical frame is a
direct frozen-forward capture that is flushed, fsynced, hash-chained, and
bitwise serialized to cache.  Only frame 195 receives an independent fresh
re-forward at completion; there is no all-n600 independent-replay claim.
``--max-frames`` may stop after a tiny honest prefix, which remains explicitly
partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as torch_functional

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.governed_profile_admission import (  # noqa: E402
    GovernedAdmissionError,
    attest_safe_run_parent,
)
from tac.witness_control.segnet_head_feature_cache import (  # noqa: E402
    APPROVED_SSD_WATERFALL,
    FeatureCacheError,
    SegnetHeadFeatureCache,
    atomic_json,
    build_immutable_identity,
    canonical_json_bytes,
    expected_cache_bytes,
    extractor_receipt_path,
    open_gt_f1_stored_memmap,
    read_bound_file,
    source_file_row,
    validate_feature_cache,
)

EXPECTED_PAIRS: Final = 600
EXPECTED_CLASSES: Final = 5
EXPECTED_HEAD_RANK: Final = 4
EXPECTED_CAMERA_SHAPE: Final = (EXPECTED_PAIRS, 874, 1164, 3)
LOCAL_TEST_MAX_FRAMES: Final = 2
SSD_ROOTS: Final = tuple(Path(root) for root in APPROVED_SSD_WATERFALL)


class ExtractionError(RuntimeError):
    """Fail-closed scorer, storage, geometry, or source-binding error."""


@dataclass(frozen=True)
class _FrozenScorerSnapshot:
    rows: dict[str, dict[str, Any]]
    segnet_payload: bytes


def _assert_real_governed_admission(
    *,
    allow_local_output_for_tests: bool,
    exact_child_argv: list[str],
    rss_cap_mb: int,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Require direct safe-run parent custody for every extraction mode."""

    if type(allow_local_output_for_tests) is not bool:
        raise ExtractionError("allow_local_output_for_tests must be boolean")
    try:
        return attest_safe_run_parent(
            exact_child_argv=exact_child_argv,
            rss_cap_mb=rss_cap_mb,
            timeout_seconds=timeout_seconds,
            repo_root=REPO_ROOT,
            env=os.environ if env is None else env,
        )
    except GovernedAdmissionError as exc:
        raise ExtractionError("extraction requires direct safe_run parent custody") from exc


def _require_local_test_scope(args: argparse.Namespace, *, env: dict[str, str] | None = None) -> None:
    if not args.allow_local_output_for_tests:
        return
    environment = os.environ if env is None else env
    # ``python -m pytest`` legitimately exposes ``__main__.py`` as argv[0].
    # Parent safe-run custody is a separate mandatory production gate.
    if not environment.get("PYTEST_CURRENT_TEST") or "pytest" not in sys.modules:
        raise ExtractionError("local output is permitted only inside an actual pytest test")
    if args.max_frames > LOCAL_TEST_MAX_FRAMES or args.max_frames >= EXPECTED_PAIRS:
        raise ExtractionError(f"local pytest output is limited to an absolute prefix of {LOCAL_TEST_MAX_FRAMES} frames")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_without_symlink_components(path: Path) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ExtractionError(f"output path custody is unavailable: {cursor}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ExtractionError(f"output path may not traverse a symlink: {cursor}")
        if cursor != absolute and not stat.S_ISDIR(metadata.st_mode):
            raise ExtractionError(f"output path traverses a non-directory: {cursor}")
    return absolute.resolve()


def _require_exact_module_file(module: Any, expected: Path, *, role: str) -> Path:
    loaded_name = getattr(module, "__file__", None)
    if not isinstance(loaded_name, str):
        raise ExtractionError(f"executed {role} module has no source path")
    loaded = Path(loaded_name).resolve(strict=True)
    exact = expected.resolve(strict=True)
    if loaded != exact:
        raise ExtractionError(f"executed {role} path {loaded} != manifest source {exact}")
    return loaded


def storage_preflight(
    output_root: Path,
    *,
    required_bytes: int,
    allow_local_output_for_tests: bool,
) -> dict[str, Any]:
    resolved = _resolve_without_symlink_components(output_root)
    existing_approved_roots: list[Path] = []
    for root in SSD_ROOTS:
        try:
            metadata = root.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ExtractionError(f"approved SSD root custody is unavailable: {root}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ExtractionError(f"approved SSD root is not a non-symlink directory: {root}")
        existing_approved_roots.append(root.resolve())
    if not allow_local_output_for_tests:
        if not existing_approved_roots:
            raise ExtractionError("no approved SSD root exists for the feature cache")
        selected_approved_root = existing_approved_roots[0]
        if not _is_relative_to(resolved, selected_approved_root):
            raise ExtractionError(
                f"feature cache must use first existing SSD root {selected_approved_root}; lower tiers are refused"
            )
    anchor = resolved
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    if usage.free < required_bytes:
        raise ExtractionError(f"storage preflight refused: free={usage.free} < required={required_bytes}")
    return {
        "waterfall_order": [str(root) for root in SSD_ROOTS],
        "existing_approved_roots": [str(root) for root in existing_approved_roots],
        "selected_root": str(resolved),
        "filesystem_anchor": str(anchor),
        "free_bytes_before": int(usage.free),
        "required_free_bytes": int(required_bytes),
        "allow_local_output_for_tests": allow_local_output_for_tests,
        "PASS": True,
    }


def _native_f32_power_assign(points: np.ndarray, target: Any) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    sites = np.asarray(target.sites, dtype=np.float32)
    weights = np.asarray(target.weights, dtype=np.float32)
    dot = np.sum(values[:, None, :] * sites[None, :, :], axis=2, dtype=np.float32)
    norm = np.sum(sites * sites, axis=1, dtype=np.float32)
    scores = np.asarray(np.float32(2.0) * dot + weights[None, :] - norm[None, :], dtype=np.float32)
    return np.asarray(target.class_ids[np.argmax(scores, axis=1)], dtype=np.int64)


def extract_frame_arrays(
    model: torch.nn.Module,
    final_head: torch.nn.Conv2d,
    quotient_basis: np.ndarray,
    frame_hwc_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one direct scorer forward and one separate quotient convolution."""

    frame = np.asarray(frame_hwc_u8)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ExtractionError("frame must be HxWx3 uint8")
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1 or inputs[0].ndim != 4:
            raise ExtractionError("final-head pre-hook geometry drift")
        captured.append(inputs[0].detach())

    hook = final_head.register_forward_pre_hook(capture)
    try:
        batch = (
            torch.from_numpy(np.array(frame, copy=True, order="C")).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        )
        with torch.inference_mode():
            logits = model(model.preprocess_input(batch))
        if len(captured) != 1:
            raise ExtractionError("expected exactly one final-head input capture")
        basis = np.asarray(quotient_basis, dtype=np.float64)
        expected_flat = int(np.prod(final_head.weight.shape[1:]))
        if basis.shape != (expected_flat, EXPECTED_HEAD_RANK):
            raise ExtractionError(f"quotient basis shape {basis.shape} != {(expected_flat, EXPECTED_HEAD_RANK)}")
        filters = torch.from_numpy(basis.T.reshape(EXPECTED_HEAD_RANK, *final_head.weight.shape[1:])).to(
            dtype=captured[0].dtype
        )
        with torch.inference_mode():
            quotient = torch_functional.conv2d(
                captured[0],
                filters,
                stride=final_head.stride,
                padding=final_head.padding,
                dilation=final_head.dilation,
                groups=final_head.groups,
            )
        live = np.ascontiguousarray(logits[0].cpu().numpy(), dtype=np.float32)
        algebraic = np.ascontiguousarray(quotient[0].cpu().numpy(), dtype=np.float32)
        if not np.isfinite(live).all() or not np.isfinite(algebraic).all():
            raise ExtractionError("SegNet extraction produced non-finite values")
        return live, algebraic
    finally:
        hook.remove()


def algebraic_diagnostic(
    live_logits: np.ndarray,
    quotient_features: np.ndarray,
    target: Any,
    *,
    power_assign_fn: Any,
) -> dict[str, Any]:
    live = np.asarray(live_logits, dtype=np.float32)
    quotient = np.asarray(quotient_features, dtype=np.float32)
    points = np.moveaxis(quotient, 0, -1).reshape(-1, quotient.shape[0])
    generic = np.asarray(power_assign_fn(points.astype(np.float64), target), dtype=np.int64)
    native = _native_f32_power_assign(points, target)
    direct = np.argmax(live, axis=0).reshape(-1)
    return {
        "scope": "ALGEBRAIC_DIAGNOSTIC_NOT_EXTRACTION_BLOCKER",
        "generic_f64_vs_live_argmax_disagreements": int(np.count_nonzero(generic != direct)),
        "native_f32_vs_live_argmax_disagreements": int(np.count_nonzero(native != direct)),
        "generic_f64_vs_native_f32_disagreements": int(np.count_nonzero(generic != native)),
    }


def _source_bindings(upstream_root: Path, gt_cache: Path) -> _FrozenScorerSnapshot:
    paths = {
        "gt_n600_npz": gt_cache,
        "executed_modules_py": upstream_root / "modules.py",
        "executed_frame_utils_py": upstream_root / "frame_utils.py",
        "executed_tac_scorer_py": REPO_ROOT / "src/tac/scorer.py",
        "executed_factorization_module_py": REPO_ROOT / "src/tac/boundary_math/power_diagram_witness.py",
        "extractor_tool": Path(__file__),
        "cache_module": REPO_ROOT / "src/tac/witness_control/segnet_head_feature_cache.py",
    }
    rows = {role: source_file_row(path) for role, path in paths.items()}
    weights_path = upstream_root / "models" / "segnet.safetensors"
    admitted = read_bound_file(weights_path, role="admitted SegNet weights")
    rows["segnet_weights"] = {
        "path": str(weights_path),
        "bytes": len(admitted.payload),
        "sha256": hashlib.sha256(admitted.payload).hexdigest(),
    }
    return _FrozenScorerSnapshot(rows=rows, segnet_payload=admitted.payload)


def _load_segnet_from_admitted_payload(modules_module: ModuleType, payload: bytes) -> torch.nn.Module:
    """Construct frozen CPU SegNet from the exact descriptor-admitted bytes."""

    if type(payload) is not bytes:
        raise ExtractionError("admitted SegNet weight payload must be immutable bytes")
    try:
        from safetensors.torch import load

        admitted_state = load(payload)
        constructor = getattr(modules_module, "SegNet", None)
        if not callable(constructor):
            raise ExtractionError("admitted modules.py has no callable SegNet constructor")
        model = constructor()
        if not isinstance(model, torch.nn.Module):
            raise ExtractionError("admitted SegNet constructor returned a non-module")
        model.load_state_dict(admitted_state, strict=True)
    except ExtractionError:
        raise
    except BaseException as exc:
        raise ExtractionError("admitted SegNet payload cannot construct the frozen scorer") from exc
    model = model.to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    realized = model.state_dict()
    if set(realized) != set(admitted_state):
        raise ExtractionError("SegNet realized state keys differ from admitted payload")
    for name, expected in admitted_state.items():
        actual = realized[name].detach().cpu()
        expected_cpu = expected.detach().cpu()
        if (
            actual.dtype != expected_cpu.dtype
            or actual.shape != expected_cpu.shape
            or not torch.equal(actual, expected_cpu)
        ):
            raise ExtractionError(f"SegNet realized state differs from admitted payload: {name}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ExtractionError("byte-fed SegNet did not preserve eval/frozen semantics")
    return model


def _load_source_module_from_snapshot(
    module_name: str,
    source_row: Mapping[str, Any],
    *,
    role: str,
) -> ModuleType:
    """Execute exactly the descriptor-read bytes admitted by the first snapshot."""

    path_value = source_row.get("path")
    expected_bytes = source_row.get("bytes")
    expected_sha256 = source_row.get("sha256")
    if not isinstance(path_value, str) or type(expected_bytes) is not int or not isinstance(expected_sha256, str):
        raise ExtractionError(f"{role} first source snapshot row is malformed")
    snapshot = read_bound_file(Path(path_value), role=f"executed {role} source")
    if len(snapshot.payload) != expected_bytes or hashlib.sha256(snapshot.payload).hexdigest() != expected_sha256:
        raise ExtractionError(f"{role} source changed before admitted execution")
    module = ModuleType(module_name)
    module.__file__ = path_value
    module.__package__ = module_name.rpartition(".")[0]
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(snapshot.payload, path_value, "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _require_equal_source_snapshots(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Admit only a complete scorer/source snapshot stable across loading."""

    normalized_before = json.loads(canonical_json_bytes({key: dict(value) for key, value in before.items()}))
    normalized_after = json.loads(canonical_json_bytes({key: dict(value) for key, value in after.items()}))
    if normalized_before != normalized_after:
        changed = sorted(
            set(normalized_before) | set(normalized_after),
            key=str,
        )
        changed = [role for role in changed if normalized_before.get(role) != normalized_after.get(role)]
        raise ExtractionError(f"frozen scorer source snapshot changed during load/factorization: {changed}")
    return normalized_before


def _source_bindings_with_admission(
    source_bindings: Mapping[str, Mapping[str, Any]],
    parent_attestation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind stable launcher source bytes without volatile invocation fields."""

    source_custody = parent_attestation.get("source_custody")
    if not isinstance(source_custody, Mapping) or set(source_custody) != {
        "governed_profile_admission",
        "safe_run",
        "admission_guard",
    }:
        raise ExtractionError("safe-run source custody is malformed")
    combined = {role: dict(row) for role, row in source_bindings.items()}
    for role, row in source_custody.items():
        if not isinstance(row, Mapping):
            raise ExtractionError("safe-run source custody row is malformed")
        combined[f"operational_custody_{role}"] = dict(row)
    return json.loads(canonical_json_bytes(combined))


def _build_cache_identity(
    *,
    args: argparse.Namespace,
    source_bindings: Mapping[str, Mapping[str, Any]],
    rebuild_argv: Sequence[str],
    camera_hw: tuple[int, int],
    seg_hw: tuple[int, int],
    frame_count: int = EXPECTED_PAIRS,
) -> dict[str, Any]:
    """Build the immutable cache identity from resume-stable inputs only."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ExtractionError("cache identity frame_count must be a positive integer")
    return build_immutable_identity(
        source_files=source_bindings,
        config={
            "authority_mode": "deterministic_cpu_float32_batch_one",
            "batch_size": args.batch_size,
            "chunk_frames": args.chunk_frames,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "requested_resource_caps": {
                "rss_cap_mb": args.rss_cap_mb,
                "timeout_seconds": args.timeout_seconds,
                "extractor_self_enforced": False,
                "scope": "REQUESTED_CAP_METADATA_ONLY_NOT_ENFORCEMENT_RECEIPT",
            },
            "canonical_fresh_rebuild_argv": list(rebuild_argv),
            "camera_hw": list(camera_hw),
            "seg_hw": list(seg_hw),
            "expected_pairs": frame_count,
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_executable": str(Path(sys.executable).resolve()),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "determinism": {
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "torch_threads_effective": torch.get_num_threads(),
                "torch_interop_threads_effective": torch.get_num_interop_threads(),
            },
        },
        frame_count=frame_count,
        live_slice_shape=(EXPECTED_CLASSES, *seg_hw),
        quotient_slice_shape=(EXPECTED_HEAD_RANK, *seg_hw),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int, default=EXPECTED_PAIRS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--rss-cap-mb", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--allow-local-output-for-tests", action="store_true")
    return parser.parse_args(argv)


def _attest_exact_argv(args: argparse.Namespace, exact_argv: list[str]) -> list[str]:
    if not isinstance(exact_argv, list) or len(exact_argv) < 2:
        raise ExtractionError("exact argv must name the Python executable and extractor tool")
    if any(not isinstance(token, str) or not token for token in exact_argv):
        raise ExtractionError("exact argv contains an invalid token")
    try:
        executable = Path(exact_argv[0]).expanduser().resolve(strict=True)
        tool = Path(exact_argv[1]).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExtractionError("exact argv executable/tool custody is unavailable") from exc
    if executable != Path(sys.executable).resolve(strict=True) or tool != Path(__file__).resolve(strict=True):
        raise ExtractionError("exact argv does not name this Python runtime and extractor tool")
    try:
        reparsed = _parse_args(exact_argv[2:])
    except (argparse.ArgumentError, SystemExit) as exc:
        raise ExtractionError("exact argv does not parse as an extractor invocation") from exc
    if set(vars(reparsed)) != set(vars(args)) or any(
        getattr(reparsed, name) != getattr(args, name) for name in vars(args)
    ):
        raise ExtractionError("exact argv does not reproduce the effective extractor arguments")
    return list(exact_argv)


def _canonical_rebuild_argv(exact_argv: list[str], *, resume: bool) -> list[str]:
    rebuilt = list(exact_argv)
    resume_count = rebuilt.count("--resume")
    if resume:
        if resume_count != 1:
            raise ExtractionError("resume argv must contain exactly one --resume token")
        rebuilt.remove("--resume")
    elif resume_count:
        raise ExtractionError("fresh extractor argv unexpectedly contains --resume")
    return rebuilt


def _emit_extraction_receipt(
    cache_root: Path,
    receipt: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    require_complete: bool,
) -> Path:
    """Commit operational custody beside, never inside, the exact cache."""

    path = extractor_receipt_path(cache_root)
    prior_payloads: tuple[bytes, ...] = ()
    if os.path.lexists(path):
        prior = read_bound_file(path, role="prior extraction receipt")
        desired_payload = canonical_json_bytes(dict(receipt)) + b"\n"
        if prior.payload == desired_payload:
            prior_payloads = (prior.payload,)
        else:
            try:
                prior_value = json.loads(prior.payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExtractionError("prior extraction receipt is malformed; preserving bytes") from exc
            if (
                not isinstance(prior_value, dict)
                or prior.payload != canonical_json_bytes(prior_value) + b"\n"
                or set(prior_value) != set(receipt)
                or prior_value.get("schema") != "extract_segnet_head_features_n600_receipt.v1"
                or prior_value.get("cache_root") != receipt.get("cache_root")
                or prior_value.get("expected_frames") != receipt.get("expected_frames")
                or prior_value.get("canonical_fresh_rebuild_argv") != receipt.get("canonical_fresh_rebuild_argv")
                or prior_value.get("requested_resource_caps") != receipt.get("requested_resource_caps")
                or type(prior_value.get("next_frame")) is not int
                or type(receipt.get("next_frame")) is not int
                or not 0 <= prior_value["next_frame"] <= receipt["next_frame"] <= receipt["expected_frames"]
                or not isinstance(prior_value.get("authority"), dict)
                or prior_value["authority"].get("direct_frozen_forward_capture_frames") != prior_value["next_frame"]
            ):
                raise ExtractionError(
                    "prior extraction receipt is not an exact reachable cache state; preserving bytes"
                )
            prior_payloads = (prior.payload,)
    atomic_json(path, receipt, expected_prior_payloads=prior_payloads)
    validate_feature_cache(
        cache_root,
        expected_identity=identity,
        require_complete=require_complete,
    )
    return path


def run_extraction(args: argparse.Namespace, *, exact_argv: list[str]) -> dict[str, Any]:
    if args.batch_size != 1 or args.chunk_frames != 1:
        raise ExtractionError("canonical deterministic CPU authority requires batch/chunk one")
    for name in (
        "max_frames",
        "torch_threads",
        "torch_interop_threads",
        "rss_cap_mb",
        "timeout_seconds",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ExtractionError(f"{name} must be a positive integer")
    if args.max_frames > EXPECTED_PAIRS:
        raise ExtractionError("max_frames exceeds canonical n600 geometry")
    attested_argv = _attest_exact_argv(args, exact_argv)
    rebuild_argv = _canonical_rebuild_argv(attested_argv, resume=args.resume)
    _require_local_test_scope(args)
    parent_attestation = _assert_real_governed_admission(
        allow_local_output_for_tests=args.allow_local_output_for_tests,
        exact_child_argv=attested_argv,
        rss_cap_mb=args.rss_cap_mb,
        timeout_seconds=args.timeout_seconds,
    )
    gt_cache = Path(source_file_row(args.gt_cache)["path"])
    upstream_root = _resolve_without_symlink_components(args.upstream_root)
    if not upstream_root.is_dir():
        raise ExtractionError("upstream root must be an existing non-symlink directory")
    output_root = _resolve_without_symlink_components(args.output_root)
    frozen_source_before = _source_bindings(upstream_root, gt_cache)
    source_snapshot_before = frozen_source_before.rows
    factorization_module = _load_source_module_from_snapshot(
        "tac.boundary_math.power_diagram_witness",
        source_snapshot_before["executed_factorization_module_py"],
        role="factorization",
    )
    gt_f1 = open_gt_f1_stored_memmap(gt_cache)
    if gt_f1.shape != EXPECTED_CAMERA_SHAPE:
        raise ExtractionError(f"gt_f1 canonical geometry {gt_f1.shape} != {EXPECTED_CAMERA_SHAPE}")

    prepend_paths(upstream_root)
    expected_upstream_files = {
        "modules": upstream_root / "modules.py",
        "frame_utils": upstream_root / "frame_utils.py",
    }
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    torch.use_deterministic_algorithms(True)
    frame_utils = _load_source_module_from_snapshot(
        "frame_utils",
        source_snapshot_before["executed_frame_utils_py"],
        role="frame_utils",
    )
    modules = _load_source_module_from_snapshot(
        "modules",
        source_snapshot_before["executed_modules_py"],
        role="modules",
    )
    scorer_module = _load_source_module_from_snapshot(
        "tac.scorer",
        source_snapshot_before["executed_tac_scorer_py"],
        role="tac.scorer",
    )
    model = _load_segnet_from_admitted_payload(modules, frozen_source_before.segnet_payload)
    _require_exact_module_file(modules, expected_upstream_files["modules"], role="modules")
    _require_exact_module_file(
        frame_utils,
        expected_upstream_files["frame_utils"],
        role="frame_utils",
    )
    _require_exact_module_file(
        scorer_module,
        REPO_ROOT / "src/tac/scorer.py",
        role="tac.scorer",
    )
    _require_exact_module_file(
        factorization_module,
        REPO_ROOT / "src/tac/boundary_math/power_diagram_witness.py",
        role="factorization",
    )
    final_head = model.segmentation_head[0]
    if not isinstance(final_head, torch.nn.Conv2d):
        raise ExtractionError("frozen SegNet final head is not Conv2d")
    frozen_head = factorization_module.affine_head_to_power_diagram(
        final_head.weight.detach().cpu().numpy(),
        final_head.bias.detach().cpu().numpy(),
    )
    if frozen_head.target.rank != EXPECTED_HEAD_RANK:
        raise ExtractionError("frozen SegNet quotient rank drift")
    seg_hw = tuple(int(value) for value in frame_utils.segnet_model_input_size[::-1])
    camera_hw = tuple(int(value) for value in gt_f1.shape[1:3])
    frozen_source_after = _source_bindings(upstream_root, gt_cache)
    stable_source_snapshot = _require_equal_source_snapshots(source_snapshot_before, frozen_source_after.rows)
    if frozen_source_before.segnet_payload != frozen_source_after.segnet_payload:
        raise ExtractionError("frozen SegNet weight payload changed during byte-fed construction")
    source_bindings = _source_bindings_with_admission(
        stable_source_snapshot,
        parent_attestation,
    )
    identity = _build_cache_identity(
        args=args,
        source_bindings=source_bindings,
        rebuild_argv=rebuild_argv,
        camera_hw=camera_hw,
        seg_hw=seg_hw,
    )
    preflight = storage_preflight(
        output_root,
        required_bytes=expected_cache_bytes(identity),
        allow_local_output_for_tests=args.allow_local_output_for_tests,
    )
    if args.resume:
        cache = SegnetHeadFeatureCache.resume(output_root, expected_identity=identity)
    else:
        cache = SegnetHeadFeatureCache.create(
            output_root,
            identity=identity,
            rebuild_command=rebuild_argv,
            storage_preflight=preflight,
        )

    if args.allow_local_output_for_tests and (
        cache.next_frame > LOCAL_TEST_MAX_FRAMES or cache.progress["status"] == "complete"
    ):
        raise ExtractionError("local pytest mode may never open or produce a complete n600 cache")

    for frame_index in range(cache.next_frame, args.max_frames):
        live, quotient = extract_frame_arrays(
            model,
            final_head,
            frozen_head.quotient_basis,
            np.asarray(gt_f1[frame_index]),
        )
        diagnostics: dict[str, Any] = {}
        if frame_index == 195:
            diagnostics["frame195"] = algebraic_diagnostic(
                live,
                quotient,
                frozen_head.target,
                power_assign_fn=factorization_module.power_assign,
            )
        cache.commit_frame(
            frame_index,
            live,
            quotient,
            diagnostics=diagnostics,
        )

    completion_control: dict[str, Any] | None = None
    if cache.next_frame == EXPECTED_PAIRS:
        positive_frame = 195
        fresh_live, fresh_quotient = extract_frame_arrays(
            model,
            final_head,
            frozen_head.quotient_basis,
            np.asarray(gt_f1[positive_frame]),
        )
        points = np.moveaxis(fresh_quotient, 0, -1).reshape(-1, EXPECTED_HEAD_RANK)
        generic = factorization_module.power_assign(points.astype(np.float64), frozen_head.target).reshape(seg_hw)
        completion_control = cache.mark_complete(
            positive_frame=positive_frame,
            fresh_live_logits=fresh_live,
            algebraic_argmax=generic,
        ).__dict__
    receipt = {
        "schema": "extract_segnet_head_features_n600_receipt.v1",
        "status": "complete" if cache.progress["status"] == "complete" else "partial",
        "next_frame": cache.next_frame,
        "expected_frames": EXPECTED_PAIRS,
        "exact_argv": attested_argv,
        "canonical_fresh_rebuild_argv": rebuild_argv,
        "cache_root": str(output_root),
        "storage_preflight": preflight,
        "requested_resource_caps": identity["config"]["requested_resource_caps"],
        "resource_custody": {
            "parent_command_attestation": parent_attestation,
            "completed_safe_run_status_receipt": None,
            "status_receipt_scope": "PARENT_EMITS_ONLY_AFTER_CHILD_EXIT_SEPARATE_FROM_THIS_RECEIPT",
        },
        "completion_positive_control": completion_control,
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_authority": False,
            "promotion_eligible": False,
            "direct_frozen_forward_capture_frames": cache.next_frame,
            "all_committed_frames_are_direct_frozen_forward_captures": True,
            "live_logits_bitwise_serialized_to_cache": True,
            "independently_fresh_reforwarded_bitwise_control_frames": (
                [195] if cache.progress["status"] == "complete" else []
            ),
            "independent_control_scope": "ONLY_FRAME_195_NOT_ALL_N600",
            "all_n600_independent_replay": False,
            "rank4_quotient_scope": "DIAGNOSTIC_ONLY",
            "rank4_quotient_score_authority": False,
            "quotient_features_replay_live_logits_bitwise": False,
        },
    }
    _emit_extraction_receipt(
        output_root,
        receipt,
        identity=identity,
        require_complete=cache.progress["status"] == "complete",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    exact_argv = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        receipt = run_extraction(args, exact_argv=exact_argv)
    except (FeatureCacheError, ExtractionError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
