#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a custody-checked component_sensitivity_v1 manifest.

This script does not compute scorer gradients. It assembles the artifacts
produced by CUDA sensitivity/profile runs into the strict manifest consumed by
promotion gates. The output is valid only when the referenced maps, response
curves, exact eval JSON, and archive are the actual artifacts from the same
run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.component_sensitivity_artifact import (  # noqa: E402
    COMPONENTS,
    CONTEST_SAMPLE_COUNT,
    ComponentSensitivityArtifactError,
    materialize_component_sensitivity_manifest,
    write_component_sensitivity_manifest,
)
from tac.sensitivity_map import (  # noqa: E402
    SENSITIVITY_MAP_FORMAT,
    SensitivityMapError,
    validate_certified_sensitivity_map_metadata,
)


class ComponentSensitivityManifestBuildError(ValueError):
    """Raised when input artifacts cannot form a promotion-grade manifest."""


_DIAGNOSTIC_TRUE_KEYS = {
    "debug",
    "debug_mode",
    "dummy",
    "fake",
    "fake_sensitivity",
    "is_debug",
    "is_smoke",
    "non_promotable",
    "proxy",
    "proxy_only",
    "random",
    "random_sensitivity",
    "smoke",
    "smoke_test",
    "synthetic_sensitivity",
}
_DIAGNOSTIC_STRING_KEYS = {
    "artifact_type",
    "code",
    "evidence_grade",
    "format",
    "kind",
    "source",
    "status",
    "sensitivity_type",
    "tool",
}
_DIAGNOSTIC_STRING_MARKERS = (
    "diagnostic",
    "fisher_proxy",
    "from_fisher",
    "non_promotable",
    "profile_component_sensitivity.py",
    "proxy_only",
    "random_sensitivity",
    "smoke",
    "synthetic_sensitivity",
)


def _sha256_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentSensitivityManifestBuildError(f"{path}: invalid JSON") from exc


def _load_contest_eval(path: Path, *, expected_n_samples: int) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ComponentSensitivityManifestBuildError(f"{path}: contest eval must be a JSON object")
    n_samples = payload.get("n_samples")
    if n_samples != expected_n_samples:
        raise ComponentSensitivityManifestBuildError(
            f"{path}: n_samples={n_samples!r}, expected {expected_n_samples}"
        )
    provenance = payload.get("provenance")
    device = None
    if isinstance(provenance, dict):
        device = provenance.get("device")
    if device != "cuda":
        raise ComponentSensitivityManifestBuildError(
            f"{path}: provenance.device must be 'cuda', got {device!r}"
        )
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_contest_eval_archive_custody(
    contest_payload: dict[str, Any],
    *,
    archive: Path,
    contest_auth_eval_json: Path,
) -> None:
    archive_bytes = archive.stat().st_size
    archive_sha = _sha256_file(archive)
    payload_bytes = contest_payload.get("archive_size_bytes")
    provenance = contest_payload.get("provenance")
    payload_sha = None
    if isinstance(provenance, dict):
        payload_sha = provenance.get("archive_sha256")
    payload_sha = payload_sha or contest_payload.get("archive_sha256")
    if payload_bytes != archive_bytes:
        raise ComponentSensitivityManifestBuildError(
            f"{contest_auth_eval_json}: archive_size_bytes={payload_bytes!r} "
            f"does not match {archive} bytes={archive_bytes}"
        )
    if payload_sha != archive_sha:
        raise ComponentSensitivityManifestBuildError(
            f"{contest_auth_eval_json}: archive_sha256={payload_sha!r} "
            f"does not match {archive} sha256={archive_sha}"
        )


def _tensor_metadata_for_path(path: Path, *, component: str) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    _reject_diagnostic_source_payload(
        path,
        payload,
        context="component sensitivity map",
    )
    if not isinstance(payload, dict) or payload.get("format") != SENSITIVITY_MAP_FORMAT:
        raise ComponentSensitivityManifestBuildError(
            f"{path}: promotion maps must use certified {SENSITIVITY_MAP_FORMAT!r} artifacts"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ComponentSensitivityManifestBuildError(
            f"{path}: certified sensitivity map requires metadata"
        )
    try:
        certification = validate_certified_sensitivity_map_metadata(
            metadata,
            component=component,
        )
    except SensitivityMapError as exc:
        raise ComponentSensitivityManifestBuildError(
            f"{path}: sensitivity map certification failed: {exc}"
        ) from exc
    tensors: dict[str, torch.Tensor] = {}
    _collect_tensors(payload.get("sensitivities"), tensors, prefix="sensitivities")
    if not tensors:
        raise ComponentSensitivityManifestBuildError(
            f"{path}: no torch tensors found; component maps must expose tensor custody"
        )
    out: dict[str, Any] = {
        "map_format": SENSITIVITY_MAP_FORMAT,
        "certification": certification,
    }
    if len(tensors) == 1:
        _name, tensor = next(iter(tensors.items()))
        out["tensor"] = _single_tensor_metadata(tensor)
        return out
    out["tensor_metadata"] = {
            key: _single_tensor_metadata(tensor)
            for key, tensor in sorted(tensors.items())
        }
    return out


def _collect_tensors(value: Any, out: dict[str, torch.Tensor], *, prefix: str) -> None:
    if torch.is_tensor(value):
        out[prefix] = value.detach().cpu()
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_tensors(child, out, prefix=f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _collect_tensors(child, out, prefix=f"{prefix}[{index}]")


def _single_tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    if not torch.isfinite(tensor.detach()).all():
        raise ComponentSensitivityManifestBuildError(
            "component sensitivity tensor contains NaN/Inf values"
        )
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(dim) for dim in tensor.shape],
        "numel": int(tensor.numel()),
    }


def _response_curve_metadata(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    _reject_diagnostic_source_payload(
        path,
        payload,
        context="component response curve",
    )
    if isinstance(payload, list):
        count = len(payload)
        holdout_error = _max_numeric_field(payload, ("holdout_error", "error", "abs_error"))
        if holdout_error is None:
            raise ComponentSensitivityManifestBuildError(
                f"{path}: response curve list needs holdout_error/error/abs_error values"
            )
        return {"count": count, "holdout_error": holdout_error}
    if not isinstance(payload, dict):
        raise ComponentSensitivityManifestBuildError(f"{path}: response curve must be JSON object or list")

    count = payload.get("count")
    if count is None:
        for key in ("points", "curve", "response_curve", "samples"):
            values = payload.get(key)
            if isinstance(values, list):
                count = len(values)
                break
    holdout_error = _first_numeric(payload, ("holdout_error", "holdout_error_max", "max_holdout_error"))
    if holdout_error is None:
        for key in ("points", "curve", "response_curve", "samples"):
            values = payload.get(key)
            if isinstance(values, list):
                holdout_error = _max_numeric_field(values, ("holdout_error", "error", "abs_error"))
                if holdout_error is not None:
                    break
    if not isinstance(count, int) or count <= 0:
        raise ComponentSensitivityManifestBuildError(f"{path}: response curve count must be positive")
    if holdout_error is None:
        raise ComponentSensitivityManifestBuildError(f"{path}: response curve holdout_error is required")
    metadata: dict[str, Any] = {"count": int(count), "holdout_error": float(holdout_error)}
    for key in (
        "official_component_response",
        "passed",
        "gate_results",
        "gate_spec",
        "promotion_blockers",
        "component_readout",
        "official_readout",
        "readout",
        "response_kind",
        "curve_kind",
        "sensitivity_kind",
        "epsilon_ladder",
        "epsilons",
        "symmetric_epsilon_pairs",
        "central_difference_pairs",
        "directional_action",
        "directional_actions",
        "action_point",
    ):
        if key in payload:
            metadata[key] = payload[key]
    if "epsilon_ladder" not in metadata and "epsilons" not in metadata:
        epsilons = _epsilon_ladder_from_payload(payload)
        if epsilons:
            metadata["epsilon_ladder"] = epsilons
    return metadata


def _first_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out = float(value)
            if out == out and abs(out) != float("inf"):
                return out
    return None


def _max_numeric_field(items: list[Any], keys: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        value = _first_numeric(item, keys)
        if value is not None:
            values.append(abs(value))
    if not values:
        return None
    return max(values)


def _epsilon_ladder_from_payload(payload: dict[str, Any]) -> list[float]:
    for key in ("points", "curve", "response_curve", "samples"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        epsilons: list[float] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            epsilon = item.get("epsilon")
            if isinstance(epsilon, (int, float)) and not isinstance(epsilon, bool):
                eps = float(epsilon)
                if eps == eps and abs(eps) != float("inf"):
                    epsilons.append(eps)
        if epsilons:
            return sorted(set(epsilons))
    return []


def _reject_diagnostic_source_payload(path: Path, payload: Any, *, context: str) -> None:
    markers = _diagnostic_source_markers(payload, path="payload")
    if not markers:
        return
    details = "; ".join(markers[:8])
    suffix = "" if len(markers) <= 8 else f"; ... (+{len(markers) - 8} more)"
    raise ComponentSensitivityManifestBuildError(
        f"{path}: {context} is diagnostic/non-promotable ({details}{suffix}). "
        "Do not assemble component_sensitivity_v1 promotion manifests from "
        "profile_component_sensitivity.py Fisher-proxy outputs."
    )


def _diagnostic_source_markers(value: Any, *, path: str) -> list[str]:
    markers: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            key_l = key_s.lower()
            child_path = f"{path}.{key_s}"
            if key_l == "promotion_eligible" and child is False:
                markers.append(f"{child_path}=False")
            elif key_l == "official_component_response" and child is False:
                markers.append(f"{child_path}=False")
            elif key_l == "promotion_blockers" and bool(child):
                markers.append(f"{child_path} is non-empty")
            elif key_l in _DIAGNOSTIC_TRUE_KEYS and child is True:
                markers.append(f"{child_path}=True")
            elif key_l in _DIAGNOSTIC_STRING_KEYS and isinstance(child, str):
                lowered = child.lower().replace("-", "_")
                if any(marker in lowered for marker in _DIAGNOSTIC_STRING_MARKERS):
                    markers.append(f"{child_path}={child!r}")
            markers.extend(_diagnostic_source_markers(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            markers.extend(_diagnostic_source_markers(child, path=f"{path}[{index}]"))
    return markers


def _load_or_generate_sample_plan(
    path: Path | None,
    *,
    n_pairs: int,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    if path is not None:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ComponentSensitivityManifestBuildError(f"{path}: sample plan must be a JSON object")
        plan = dict(payload)
    else:
        plan = _generate_sample_plan(
            n_pairs=n_pairs,
            split_seed=split_seed,
            holdout_fraction=holdout_fraction,
        )
    plan.setdefault("split_seed", int(split_seed))
    split_hash = _sha256_jsonable(
        {
            "calibration_pairs": plan.get("calibration_pairs"),
            "holdout_pairs": plan.get("holdout_pairs"),
            "split_seed": plan.get("split_seed"),
        }
    )
    existing_hash = plan.get("split_hash")
    if existing_hash is not None and existing_hash != split_hash:
        raise ComponentSensitivityManifestBuildError(
            f"{path}: sample_plan.split_hash={existing_hash!r} does not match "
            f"computed split hash {split_hash!r}"
        )
    plan["split_hash"] = split_hash
    return plan


def _generate_sample_plan(
    *,
    n_pairs: int,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    if n_pairs <= 1:
        raise ComponentSensitivityManifestBuildError("n_pairs must be > 1")
    if not (0.0 < holdout_fraction < 1.0):
        raise ComponentSensitivityManifestBuildError("holdout_fraction must be in (0, 1)")
    indices = list(range(n_pairs))
    # Deterministic split without depending on random module version details.
    indices.sort(key=lambda idx: hashlib.sha256(f"{split_seed}:{idx}".encode("ascii")).hexdigest())
    n_holdout = max(1, min(n_pairs - 1, round(n_pairs * holdout_fraction)))
    holdout = sorted(indices[:n_holdout])
    calibration = sorted(indices[n_holdout:])
    return {
        "calibration_pairs": [_pair_record(idx) for idx in calibration],
        "holdout_pairs": [_pair_record(idx) for idx in holdout],
        "split_seed": int(split_seed),
    }


def _pair_record(pair_index: int) -> dict[str, int]:
    return {
        "video": 0,
        "pair_index": int(pair_index),
        "t": int(2 * pair_index),
        "t1": int(2 * pair_index + 1),
    }


def _load_stability(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ComponentSensitivityManifestBuildError(f"{path}: stability must be a JSON object")
    return payload


def _resolve_input_path(path: str | Path, *, root: str | Path | None) -> Path:
    """Resolve a manifest input for local reads without changing manifest custody paths."""
    p = Path(path)
    if p.is_absolute() or root is None:
        return p
    return Path(root) / p


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    contest_auth_eval_json = _resolve_input_path(
        args.contest_auth_eval_json,
        root=args.root,
    )
    archive = _resolve_input_path(args.archive, root=args.root)
    contest_payload = _load_contest_eval(
        contest_auth_eval_json,
        expected_n_samples=args.n_samples,
    )
    _validate_contest_eval_archive_custody(
        contest_payload,
        archive=archive,
        contest_auth_eval_json=contest_auth_eval_json,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": "component_sensitivity_v1",
        "device": args.device,
        "promotion_eligible": True,
        "evidence_grade": args.evidence_grade,
        "inputs": {
            "checkpoint": {"path": str(args.checkpoint)},
            "video": {"path": str(args.video)},
            "upstream": {"path": str(args.upstream)},
        },
        "sample_plan": _load_or_generate_sample_plan(
            (
                _resolve_input_path(args.sample_plan_json, root=args.root)
                if args.sample_plan_json is not None
                else None
            ),
            n_pairs=args.n_pairs,
            split_seed=args.split_seed,
            holdout_fraction=args.holdout_fraction,
        ),
        "component_maps": {},
        "stability": _load_stability(
            _resolve_input_path(args.stability_json, root=args.root)
        ),
        "response_curves": {},
        "contest_eval": {
            "archive": {"path": str(args.archive)},
            "contest_auth_eval_json": {"path": str(args.contest_auth_eval_json)},
            "device": contest_payload.get("provenance", {}).get("device"),
            "n_samples": int(contest_payload["n_samples"]),
        },
        "promotion_blockers": [],
    }

    for component, map_path in args.component_maps.items():
        entry = {"path": str(map_path), "scorer_target": component}
        entry.update(
            _tensor_metadata_for_path(
                _resolve_input_path(map_path, root=args.root),
                component=component,
            )
        )
        manifest["component_maps"][component] = entry

    for component, curve_path in args.response_curves.items():
        entry = {"path": str(curve_path)}
        entry.update(
            _response_curve_metadata(_resolve_input_path(curve_path, root=args.root))
        )
        manifest["response_curves"][component] = entry

    return materialize_component_sensitivity_manifest(
        manifest,
        root=args.root,
        promotion=True,
    )


def _path_arg(value: str) -> Path:
    return Path(value)


def _validate_existing_input_paths(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    for attr in (
        "checkpoint",
        "video",
        "upstream",
        "archive",
        "contest_auth_eval_json",
        "posenet_map",
        "segnet_map",
        "combined_map",
        "posenet_response_curve",
        "segnet_response_curve",
        "combined_response_curve",
        "stability_json",
        "sample_plan_json",
    ):
        value = getattr(args, attr)
        if value is None:
            continue
        resolved = _resolve_input_path(value, root=args.root)
        if not resolved.exists():
            option = "--" + attr.replace("_", "-")
            parser.error(f"{option}: not found: {value} (resolved as {resolved})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=_path_arg, required=True)
    parser.add_argument("--video", type=_path_arg, required=True)
    parser.add_argument("--upstream", type=_path_arg, required=True)
    parser.add_argument("--archive", type=_path_arg, required=True)
    parser.add_argument("--contest-auth-eval-json", type=_path_arg, required=True)
    parser.add_argument("--posenet-map", type=_path_arg, required=True)
    parser.add_argument("--segnet-map", type=_path_arg, required=True)
    parser.add_argument("--combined-map", type=_path_arg, required=True)
    parser.add_argument("--posenet-response-curve", type=_path_arg, required=True)
    parser.add_argument("--segnet-response-curve", type=_path_arg, required=True)
    parser.add_argument("--combined-response-curve", type=_path_arg, required=True)
    parser.add_argument("--stability-json", type=_path_arg, required=True)
    parser.add_argument("--sample-plan-json", type=_path_arg, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--evidence-grade", default="A", choices=["A", "A++"])
    parser.add_argument("--n-samples", type=int, default=CONTEST_SAMPLE_COUNT)
    parser.add_argument("--n-pairs", type=int, default=CONTEST_SAMPLE_COUNT)
    parser.add_argument("--split-seed", type=int, default=20260430)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    _validate_existing_input_paths(args, parser)
    args.component_maps = {
        "posenet": args.posenet_map,
        "segnet": args.segnet_map,
        "combined": args.combined_map,
    }
    args.response_curves = {
        "posenet": args.posenet_response_curve,
        "segnet": args.segnet_response_curve,
        "combined": args.combined_response_curve,
    }
    missing = [component for component in COMPONENTS if component not in args.component_maps]
    if missing:
        raise ComponentSensitivityManifestBuildError(f"missing component map(s): {missing}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_manifest(args)
    except (ComponentSensitivityArtifactError, ComponentSensitivityManifestBuildError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    write_component_sensitivity_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "format": manifest["format"],
                "device": manifest["device"],
                "evidence_grade": manifest["evidence_grade"],
                "promotion_eligible": manifest["promotion_eligible"],
                "archive_sha256": manifest["contest_eval"]["archive"]["sha256"],
                "archive_bytes": manifest["contest_eval"]["archive"]["bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
