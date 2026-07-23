# SPDX-License-Identifier: MIT
"""Build and validate the frozen scorer's transitive module inventory.

The inventory deliberately separates three evidence strata:

* immutable evaluator sources and frozen checkpoint bytes;
* the module graph instantiated by the *observed* Python environment;
* versions selected by the upstream ``uv.lock``.

An observed graph with version drift is still useful for checkpoint/name/shape
truthing, but it is not silently promoted to locked evaluator-source authority.
The receipt therefore fails closed with
``BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED`` whenever those strata do not
match.

This is research-only tooling.  It never runs the scorer, decodes video, writes
inside the pinned upstream tree, or claims a score.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "ddm_scorer_module_inventory.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
COMMON_LOCKED_PACKAGES = (
    "av",
    "einops",
    "numpy",
    "safetensors",
    "segmentation-models-pytorch",
    "timm",
)
PLATFORM_LOCKED_PACKAGES = ("torch", "torchvision")
PACKAGE_IMPORT_NAMES = {
    "av": "av",
    "einops": "einops",
    "numpy": "numpy",
    "safetensors": "safetensors",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
}
MECHANISM_CLASSES = {
    "AllNorm",
    "Attention",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNormAct2d",
    "GELUTanh",
    "LayerScale2d",
    "ReparamLargeKernelConv",
    "SEModule",
    "SiLU",
    "SqueezeExcite",
}


class ScorerInventoryError(ValueError):
    """The inventory or its custody contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None and resolved.is_relative_to(root.resolve())
        else str(resolved)
    )
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _source_ref(target: Any, *, upstream_root: Path) -> dict[str, Any]:
    source = inspect.getsourcefile(target)
    if source is None:
        raise ScorerInventoryError(f"no Python source for {target!r}")
    source_path = Path(source).resolve()
    lines, start = inspect.getsourcelines(target)
    row = file_identity(source_path, root=upstream_root)
    row.update(
        {
            "line_start": start,
            "line_stop_inclusive": start + len(lines) - 1,
            "qualified_name": f"{target.__module__}.{target.__qualname__}",
        }
    )
    return row


def _locked_versions(lock_path: Path) -> dict[str, list[dict[str, Any]]]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    wanted = set(COMMON_LOCKED_PACKAGES + PLATFORM_LOCKED_PACKAGES)
    for package in lock["package"]:
        name = package["name"]
        if name not in wanted:
            continue
        result.setdefault(name, []).append(
            {
                "version": package["version"],
                "source": package.get("source"),
                "resolution_markers": package.get("resolution-markers"),
            }
        )
    for rows in result.values():
        rows.sort(key=lambda row: canonical_json_bytes(row))
    return dict(sorted(result.items()))


def _observed_versions() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for package, import_name in PACKAGE_IMPORT_NAMES.items():
        module = importlib.import_module(import_name)
        rows[package] = {
            "version": str(getattr(module, "__version__", "UNKNOWN")),
            "import_path": str(Path(module.__file__).resolve()),
        }
    return rows


def _selected_lock_versions_for_observed_host(
    locked: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, str]:
    """Return the exact common rows and the macOS CPU rows for this host.

    The upstream lock carries multiple Torch/CUDA resolutions.  This delegated
    lane is explicitly macOS-CPU advisory, so its comparison target is the
    Darwin CPU row (plain 2.10.0 / 0.25.0 in the current lock).
    """

    selected: dict[str, str] = {}
    for package in COMMON_LOCKED_PACKAGES:
        rows = locked.get(package, ())
        if len(rows) != 1:
            raise ScorerInventoryError(f"{package}: expected one common lock row")
        selected[package] = str(rows[0]["version"])
    for package in PLATFORM_LOCKED_PACKAGES:
        rows = locked.get(package, ())
        candidates = [
            row
            for row in rows
            if row.get("source", {}).get("registry")
            in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu/"}
            and "+cpu" not in str(row["version"])
            and any(
                "sys_platform == 'darwin'" in marker
                for marker in (row.get("resolution_markers") or ())
            )
        ]
        versions = sorted({str(row["version"]) for row in candidates})
        if len(versions) != 1:
            raise ScorerInventoryError(
                f"{package}: could not select one macOS CPU lock version: {versions}"
            )
        selected[package] = versions[0]
    return dict(sorted(selected.items()))


def _checkpoint_inventory(
    *,
    checkpoint_path: Path,
    state_dict: Mapping[str, Any],
    upstream_root: Path,
) -> dict[str, Any]:
    from safetensors import safe_open

    tensors: list[dict[str, Any]] = []
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            view = handle.get_slice(name)
            tensors.append(
                {
                    "name": name,
                    "shape": list(view.get_shape()),
                    "dtype": view.get_dtype(),
                }
            )
        metadata = handle.metadata()
    checkpoint = {row["name"]: row for row in tensors}
    module_state = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in state_dict.items()
    }
    missing = sorted(set(module_state) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(module_state))
    defaulted_batchnorm_counters = [
        name for name in missing if name.endswith(".num_batches_tracked")
    ]
    hard_missing = sorted(set(missing) - set(defaulted_batchnorm_counters))
    shape_mismatches = [
        {
            "name": name,
            "checkpoint_shape": checkpoint[name]["shape"],
            "module_shape": module_state[name]["shape"],
        }
        for name in sorted(set(checkpoint) & set(module_state))
        if checkpoint[name]["shape"] != module_state[name]["shape"]
    ]
    return {
        **file_identity(checkpoint_path, root=upstream_root),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "module_state_match": {
            "status": (
                "EXACT_NAMES_AND_SHAPES"
                if not missing and not unexpected and not shape_mismatches
                else (
                    "LOAD_COMPATIBLE_BATCHNORM_COUNTERS_DEFAULTED"
                    if not hard_missing
                    and not unexpected
                    and not shape_mismatches
                    else "MISMATCH"
                )
            ),
            "missing_from_checkpoint": missing,
            "hard_missing_from_checkpoint": hard_missing,
            "defaulted_noncheckpoint_batchnorm_counters": (
                defaulted_batchnorm_counters
            ),
            "unexpected_in_checkpoint": unexpected,
            "shape_mismatches": shape_mismatches,
            "module_state_tensor_count": len(module_state),
        },
    }


def _module_inventory(network: Any, *, upstream_root: Path) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    source_cache: dict[type[Any], dict[str, Any]] = {}
    for name, module in network.named_modules():
        module_type = type(module)
        if module_type not in source_cache:
            source_cache[module_type] = _source_ref(
                module_type, upstream_root=upstream_root
            )
        row = {
            "name": name or "<root>",
            "class": module_type.__name__,
            "qualified_class": f"{module_type.__module__}.{module_type.__qualname__}",
            "source": source_cache[module_type],
            "local_parameters": [
                {
                    "name": parameter_name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for parameter_name, parameter in module.named_parameters(recurse=False)
            ],
            "local_buffers": [
                {
                    "name": buffer_name,
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                }
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ],
        }
        modules.append(row)
    counts = Counter(row["class"] for row in modules)
    mechanisms = {
        class_name: {
            "count": counts.get(class_name, 0),
            "instances": [
                row["name"] for row in modules if row["class"] == class_name
            ],
            "sources": sorted(
                {
                    canonical_json_bytes(row["source"]).decode("utf-8")
                    for row in modules
                    if row["class"] == class_name
                }
            ),
        }
        for class_name in sorted(MECHANISM_CLASSES)
        if counts.get(class_name, 0)
    }
    return {
        "module_count": len(modules),
        "class_counts": dict(sorted(counts.items())),
        "mechanisms": mechanisms,
        "modules": modules,
    }


def _load_upstream_modules(upstream_root: Path) -> Any:
    root_text = str(upstream_root.resolve())
    sys.path.insert(0, root_text)
    try:
        # Deliberately import the same top-level module names as evaluate.py.
        for name in ("frame_utils", "modules"):
            sys.modules.pop(name, None)
        return importlib.import_module("modules")
    finally:
        if sys.path[0] == root_text:
            sys.path.pop(0)


def _source_laws(modules: Any, *, upstream_root: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    posenet = modules.PoseNet()
    segnet = modules.SegNet()
    class_index: dict[str, type[Any]] = {
        type(module).__name__: type(module)
        for network in (posenet, segnet)
        for module in network.modules()
    }
    wanted = {
        name: class_index[name]
        for name in (
            "AllNorm",
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNormAct2d",
            "GELUTanh",
            "LayerScale2d",
            "ReparamLargeKernelConv",
            "SEModule",
            "SiLU",
            "SqueezeExcite",
        )
        if name in class_index
    }
    laws = {
        name: _source_ref(target, upstream_root=upstream_root)
        for name, target in sorted(wanted.items())
    }
    laws.update(
        {
            "F.interpolate": _source_ref(
                functional.interpolate, upstream_root=upstream_root
            ),
            "torch.nn.functional.batch_norm": _source_ref(
                functional.batch_norm, upstream_root=upstream_root
            ),
            "torch.nn.BatchNorm2d.forward": _source_ref(
                torch.nn.BatchNorm2d.forward, upstream_root=upstream_root
            ),
            "upstream.AllNorm": _source_ref(
                modules.AllNorm, upstream_root=upstream_root
            ),
            "upstream.PoseNet": _source_ref(
                modules.PoseNet, upstream_root=upstream_root
            ),
            "upstream.SegNet": _source_ref(
                modules.SegNet, upstream_root=upstream_root
            ),
            "upstream.DistortionNet": _source_ref(
                modules.DistortionNet, upstream_root=upstream_root
            ),
        }
    )
    return laws


def build_inventory(*, upstream_root: Path, created_at_utc: str) -> dict[str, Any]:
    root = upstream_root.resolve()
    required = {
        "evaluate": root / "evaluate.py",
        "modules": root / "modules.py",
        "frame_utils": root / "frame_utils.py",
        "pyproject": root / "pyproject.toml",
        "uv_lock": root / "uv.lock",
        "video_names": root / "public_test_video_names.txt",
        "posenet_checkpoint": root / "models" / "posenet.safetensors",
        "segnet_checkpoint": root / "models" / "segnet.safetensors",
    }
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        raise ScorerInventoryError(f"missing upstream inputs: {absent}")

    modules = _load_upstream_modules(root)
    posenet = modules.PoseNet()
    segnet = modules.SegNet()
    locked = _locked_versions(required["uv_lock"])
    selected = _selected_lock_versions_for_observed_host(locked)
    observed = _observed_versions()
    drift = {
        package: {
            "selected_lock_version": selected[package],
            "observed_version": observed[package]["version"],
        }
        for package in selected
        if observed[package]["version"] != selected[package]
    }
    pose_inventory = _module_inventory(posenet, upstream_root=root)
    seg_inventory = _module_inventory(segnet, upstream_root=root)

    seg_attention = [
        module
        for module in segnet.modules()
        if type(module).__name__ == "Attention"
    ]
    active_seg_attention = [
        type(module.attention).__name__
        for module in seg_attention
        if type(module.attention).__name__ != "Identity"
    ]
    video_names = required["video_names"].read_text(encoding="utf-8").splitlines()
    if any(not name for name in video_names):
        raise ScorerInventoryError("video names file contains an empty row")

    body = {
        "schema": SCHEMA,
        "created_at_utc": created_at_utc,
        "research_only": True,
        "score_claim": False,
        "execution_allowed": False,
        "evidence_axis": EVIDENCE_AXIS,
        "first_rung": True,
        "verdict_scope": (
            "frozen upstream source/checkpoint inventory plus the observed local "
            "macOS import graph; locked contest-CPU/CUDA execution remains open"
        ),
        "source_strata": {
            "A_evaluator_composition": {
                "status": "EXACT_IMMUTABLE_SOURCE_BYTES",
                "files": {
                    name: file_identity(path, root=root)
                    for name, path in required.items()
                    if name
                    in {
                        "evaluate",
                        "modules",
                        "frame_utils",
                        "pyproject",
                        "uv_lock",
                        "video_names",
                    }
                },
                "semantics": {
                    "pairing": {
                        "seq_len": 2,
                        "pair_count_required_by_this_atlas": 600,
                        "batch_size_default": 16,
                        "batch_pair_counts_at_n600": [16] * 37 + [8],
                        "nonoverlapping_pairs": True,
                        "odd_trailing_frame": "DISCARDED_BY_FLOOR_FRAME_COUNT_DIV_SEQ_LEN",
                        "source": "frame_utils.py:10,118-143,159-253",
                    },
                    "input_cast": {
                        "law": "BTHWC uint8 -> BTCHW -> torch.float32",
                        "source": "modules.py:143-148",
                    },
                    "resize": {
                        "law": (
                            "torch.nn.functional.interpolate bilinear to 384x512; "
                            "align_corners omitted (False default), antialias omitted "
                            "(False default)"
                        ),
                        "source": "modules.py:70-74,107-109",
                    },
                    "pose": {
                        "both_frames_consumed": True,
                        "rgb_to_yuv6": (
                            "BT.601 coefficients on resized float RGB, clamp [0,255], "
                            "2x2 box chroma, four luma sublattices"
                        ),
                        "normalization": "(x-127.5)/63.75 over 12 YUV6 channels",
                        "head_output_coordinates": 12,
                        "distortion_coordinates": 6,
                        "distortion": "mean squared error over first six coordinates",
                        "source": "modules.py:20-26,61-84; frame_utils.py:50-78",
                    },
                    "seg": {
                        "frame0_influence": "EXACT_ZERO",
                        "frame1_only": True,
                        "distortion": (
                            "mean of last-frame 5-class argmax disagreements over "
                            "uniform 384x512 sites"
                        ),
                        "per_pair_mean_equals_global_pixel_mean_at_n600": True,
                        "source": "modules.py:103-113",
                    },
                    "forward_and_aggregation": {
                        "mode": "torch.inference_mode; models eval before checkpoint load",
                        "four_network_forwards_per_batch": True,
                        "accumulators": "device-local zero-dimensional torch.float32",
                        "reduction_order": (
                            "per-pair distortions -> batch tensor sum -> sequential "
                            "batch accumulator; all_reduce on CUDA distributed"
                        ),
                        "means": "accumulator / fp32 batch-size accumulator",
                        "score": (
                            "100*d_seg + sqrt(10*d_pose) + "
                            "25*archive_bytes/uncompressed_bytes"
                        ),
                        "source": "evaluate.py:52-92; modules.py:130-158",
                    },
                    "ground_truth_decode": {
                        "cpu": (
                            "PyAV yuv420p planes -> manual limited-range BT.601 RGB "
                            "with bilinear chroma align_corners=False -> round uint8"
                        ),
                        "phantom_pose_prevention": (
                            "manual YUV-plane conversion; not PyAV rgb24"
                        ),
                        "source": "frame_utils.py:145-216",
                    },
                    "compressed_decode": {
                        "law": (
                            "raw inflated RGB tensor with exact byte-count check and "
                            "shape [-1,874,1164,3]"
                        ),
                        "source": "frame_utils.py:218-253",
                    },
                },
            },
            "B_imported_library_sources": {
                "status": (
                    "EXACT_OBSERVED_IMPORT_SOURCES_NOT_LOCKED_AUTHORITY"
                    if drift
                    else "EXACT_LOCK_MATCHED_IMPORT_SOURCES"
                ),
                "python": sys.version,
                "observed": observed,
                "selected_macos_cpu_lock_versions": selected,
                "all_relevant_lock_rows": locked,
                "version_drift": drift,
                "source_laws": _source_laws(modules, upstream_root=root),
                "binding_gate": (
                    "BLOCKED_LOCKED_LIBRARY_SOURCE_NOT_MATERIALIZED"
                    if drift
                    else "PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED"
                ),
            },
            "C_loaded_artifacts": {
                "status": "EXACT_BYTES_NAMES_SHAPES_DTYPES",
                "video_names": video_names,
                "posenet": _checkpoint_inventory(
                    checkpoint_path=required["posenet_checkpoint"],
                    state_dict=posenet.state_dict(),
                    upstream_root=root,
                ),
                "segnet": _checkpoint_inventory(
                    checkpoint_path=required["segnet_checkpoint"],
                    state_dict=segnet.state_dict(),
                    upstream_root=root,
                ),
            },
        },
        "networks": {
            "posenet": {
                **pose_inventory,
                "mechanism_summary": {
                    "actual_se_modules": pose_inventory["class_counts"].get(
                        "SEModule", 0
                    ),
                    "batchnorm_act2d": pose_inventory["class_counts"].get(
                        "BatchNormAct2d", 0
                    ),
                    "batchnorm2d": pose_inventory["class_counts"].get(
                        "BatchNorm2d", 0
                    ),
                    "batchnorm1d": pose_inventory["class_counts"].get(
                        "BatchNorm1d", 0
                    ),
                    "allnorm": pose_inventory["class_counts"].get("AllNorm", 0),
                    "layer_scale2d": pose_inventory["class_counts"].get(
                        "LayerScale2d", 0
                    ),
                    "gelu_tanh": pose_inventory["class_counts"].get(
                        "GELUTanh", 0
                    ),
                    "reparam_large_kernel_conv": pose_inventory[
                        "class_counts"
                    ].get("ReparamLargeKernelConv", 0),
                    "attention": 0,
                },
            },
            "segnet": {
                **seg_inventory,
                "mechanism_summary": {
                    "squeeze_excite": seg_inventory["class_counts"].get(
                        "SqueezeExcite", 0
                    ),
                    "silu": seg_inventory["class_counts"].get("SiLU", 0),
                    "batchnorm_act2d": seg_inventory["class_counts"].get(
                        "BatchNormAct2d", 0
                    ),
                    "batchnorm2d": seg_inventory["class_counts"].get(
                        "BatchNorm2d", 0
                    ),
                    "attention_wrappers": len(seg_attention),
                    "active_attention_modules": len(active_seg_attention),
                    "active_attention_classes": active_seg_attention,
                    "attention_verdict": (
                        "NAME_ARTIFACT_ALL_WRAPPERS_CONTAIN_IDENTITY"
                        if not active_seg_attention
                        else "ACTIVE_ATTENTION_PRESENT"
                    ),
                },
            },
        },
        "analytic_binding": {
            "closed_forms": {
                "batchnorm_expected_stats": {
                    "inputs": (
                        "checkpoint running_mean, running_var, weight, bias plus "
                        "layer epsilon"
                    ),
                    "source_laws": [
                        "BatchNorm1d",
                        "BatchNorm2d",
                        "BatchNormAct2d",
                        "torch.nn.functional.batch_norm",
                    ],
                },
                "squeeze_excite_gate": {
                    "inputs": "checkpoint reduce/expand weights and biases",
                    "source_laws": ["SEModule", "SqueezeExcite", "SiLU"],
                },
                "kernel_dft_frequency_phase": {
                    "inputs": "checkpoint OIHW kernel tensors",
                    "source_laws": [
                        "ReparamLargeKernelConv",
                        "upstream.PoseNet",
                        "upstream.SegNet",
                    ],
                },
                "bn_silu_contrast": {
                    "inputs": "frozen BN affine factor composed with SiLU",
                    "source_laws": [
                        "BatchNormAct2d",
                        "SiLU",
                        "torch.nn.functional.batch_norm",
                    ],
                },
            },
            "status": (
                "BLOCKED_LIBRARY_SOURCE_VERSION_DRIFT"
                if drift
                else "READY_LOCKED_SOURCE_BOUND"
            ),
            "consumption_rule": (
                "Every materialized closed form must stamp this receipt hash, "
                "checkpoint hash, and the cited source-law file hash; rederive "
                "on any mismatch."
            ),
        },
    }
    return body


def wrap_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    return {
        "body": payload,
        "body_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != {"body", "body_sha256"}:
        raise ScorerInventoryError("receipt must contain body and body_sha256 only")
    body = receipt["body"]
    if not isinstance(body, Mapping) or body.get("schema") != SCHEMA:
        raise ScorerInventoryError("receipt body schema mismatch")
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if receipt["body_sha256"] != actual:
        raise ScorerInventoryError("receipt body SHA-256 mismatch")
    for network in ("posenet", "segnet"):
        checkpoint = body["source_strata"]["C_loaded_artifacts"][network]
        if checkpoint["module_state_match"]["status"] not in {
            "EXACT_NAMES_AND_SHAPES",
            "LOAD_COMPATIBLE_BATCHNORM_COUNTERS_DEFAULTED",
        }:
            raise ScorerInventoryError(f"{network} checkpoint does not match graph")
    if body["source_strata"]["A_evaluator_composition"]["semantics"]["pairing"][
        "pair_count_required_by_this_atlas"
    ] != 600:
        raise ScorerInventoryError("inventory is not n600")


def write_receipt_once(path: Path, receipt: Mapping[str, Any]) -> None:
    validate_receipt(receipt)
    payload = canonical_json_bytes(receipt) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_and_validate_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    return receipt


__all__ = [
    "EVIDENCE_AXIS",
    "SCHEMA",
    "ScorerInventoryError",
    "build_inventory",
    "canonical_json_bytes",
    "read_and_validate_receipt",
    "validate_receipt",
    "wrap_receipt",
    "write_receipt_once",
]
