"""Schema checks for component sensitivity artifacts.

This module validates the custody envelope around PoseNet, SegNet, and combined
sensitivity maps. It intentionally stays pure-stdlib so validation can run in
preflight paths without importing torch or scorer code.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


COMPONENT_SENSITIVITY_FORMAT = "component_sensitivity_v1"
COMPONENT_SENSITIVITY_SCHEMA_VERSION = 1
CERTIFIED_COMPONENT_MAP_FORMAT = "tac_score_sensitivity_map_v1"
COMPONENT_MAP_CERTIFICATION_FORMAT = "component_sensitivity_map_certification_v1"
COMPONENTS = ("posenet", "segnet", "combined")
REQUIRED_INPUTS = ("checkpoint", "video", "upstream")
CONTEST_SAMPLE_COUNT = 600

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PROMOTION_EVIDENCE_GRADES = {"a", "a++"}
_TOP_LEVEL_ORDER = (
    "schema_version",
    "format",
    "device",
    "promotion_eligible",
    "evidence_grade",
    "inputs",
    "sample_plan",
    "component_maps",
    "stability",
    "response_curves",
    "contest_eval",
    "promotion_blockers",
)
_PROMOTION_FLAG_KEYS = {
    "debug",
    "debug_mode",
    "dummy",
    "dummy_sensitivity",
    "fake",
    "fake_sensitivity",
    "is_debug",
    "is_smoke",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "proxy",
    "proxy_only",
    "proxy_score",
    "random",
    "random_sensitivity",
    "score_proxy",
    "scorer_proxy",
    "smoke",
    "smoke_test",
    "synthetic_sensitivity",
    "use_proxy",
    "uses_proxy",
}
_DEVICE_FIELD_KEYS = {
    "auth_eval_device",
    "device",
    "eval_device",
    "evaluation_device",
    "profile_device",
    "scorer_device",
    "source_device",
}
_PROMOTION_STRING_MARKERS = {
    "diagnostic",
    "debug",
    "dummy",
    "fake",
    "fake_sensitivity",
    "fisher_proxy",
    "from_fisher",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "profile_component_sensitivity_py",
    "proxy",
    "proxy_only",
    "random",
    "random_sensitivity",
    "score_proxy",
    "scorer_proxy",
    "smoke",
    "smoke_test",
    "synthetic_sensitivity",
}
_PROMOTION_STRING_SUBSTRINGS = (
    "diagnostic",
    "debug",
    "fake_sensitivity",
    "fisher_proxy",
    "from_fisher",
    "profile_component_sensitivity_py",
    "proxy_only",
    "random_sensitivity",
    "smoke",
    "uniform_sensitivity",
)
_RESPONSE_READOUT_ALIASES = {
    "posenet": {
        "official_pose_mse",
        "official_posenet_pose_mse",
        "upstream_posenet_pose_mse",
        "cuda_posenet_pose_mse",
    },
    "segnet": {
        "official_argmax_disagreement",
        "official_segnet_argmax_disagreement",
        "upstream_segnet_argmax_disagreement",
        "cuda_segnet_argmax_disagreement",
    },
    "combined": {
        "official_component_formula",
        "official_contest_component_formula",
        "contest_component_formula",
        "component_formula_from_pose_seg",
    },
}
_RESPONSE_KIND_KEYS = ("response_kind", "curve_kind", "sensitivity_kind")
_RESPONSE_EPSILON_KEYS = ("epsilon_ladder", "epsilons")
_RESPONSE_GATE_RESULT_TRUE_KEYS = (
    "coverage_passed",
    "zero_repro",
    "signal_present",
    "prediction_error_passed",
    "promotion_gate_passed",
)


class ComponentSensitivityArtifactError(ValueError):
    """Raised when a component sensitivity artifact is malformed or non-promotable."""


ComponentSensitivityManifestError = ComponentSensitivityArtifactError


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest for a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: str | Path) -> dict[str, Any]:
    """Return path, byte count, and sha256 metadata for manifest construction."""
    p = Path(path)
    return {
        "path": str(p),
        "bytes": int(p.stat().st_size),
        "sha256": sha256_file(p),
    }


def custody_metadata(path: str | Path) -> dict[str, Any]:
    """Return deterministic custody metadata for a file or directory tree."""
    p = Path(path)
    if p.is_file():
        out = file_metadata(p)
        out["kind"] = "file"
        return out
    if p.is_dir():
        return _directory_custody_metadata(p)
    raise FileNotFoundError(path)


def _directory_custody_metadata(path: Path) -> dict[str, Any]:
    records: list[tuple[str, int, str]] = []
    total_bytes = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_dir():
            continue
        if child.is_symlink():
            raise ComponentSensitivityArtifactError(
                f"{child}: custody tree hashing rejects symlinks"
            )
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        size = int(child.stat().st_size)
        digest = sha256_file(child)
        total_bytes += size
        records.append((rel, size, digest))

    h = hashlib.sha256()
    for rel, size, digest in records:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")

    return {
        "path": str(path),
        "kind": "directory",
        "bytes": total_bytes,
        "sha256": h.hexdigest(),
        "file_count": len(records),
    }


def materialize_component_sensitivity_manifest(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    promotion: bool = True,
) -> dict[str, Any]:
    """Fill SHA/byte custody fields and return a canonical manifest.

    Existing bytes or SHA-256 values are treated as assertions: if a referenced
    path exists and the supplied custody value disagrees, materialization fails
    instead of silently rewriting the claim.
    """
    if not isinstance(manifest, Mapping):
        raise ComponentSensitivityArtifactError(
            f"manifest must be a mapping, got {type(manifest).__name__}"
        )

    base = deepcopy(dict(manifest))
    base.setdefault("schema_version", COMPONENT_SENSITIVITY_SCHEMA_VERSION)
    base.setdefault("format", COMPONENT_SENSITIVITY_FORMAT)
    root_path = Path(root) if root is not None else None

    if isinstance(base.get("inputs"), Mapping):
        base["inputs"] = _materialize_file_section(
            base["inputs"],
            root=root_path,
            preferred_order=REQUIRED_INPUTS,
            context="inputs",
        )
    if isinstance(base.get("component_maps"), Mapping):
        base["component_maps"] = _materialize_file_section(
            base["component_maps"],
            root=root_path,
            preferred_order=COMPONENTS,
            context="component_maps",
        )
    if isinstance(base.get("response_curves"), Mapping):
        base["response_curves"] = _materialize_file_section(
            base["response_curves"],
            root=root_path,
            preferred_order=COMPONENTS,
            context="response_curves",
        )
    if isinstance(base.get("contest_eval"), Mapping):
        base["contest_eval"] = _materialize_contest_eval(
            base["contest_eval"],
            root=root_path,
        )

    out = canonicalize_component_sensitivity_manifest(base)
    validate_component_sensitivity_manifest(out, promotion=promotion)
    return out


def canonicalize_component_sensitivity_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministically ordered manifest mapping."""
    out: dict[str, Any] = {}
    for key in _TOP_LEVEL_ORDER:
        if key in manifest:
            out[key] = _canonicalize_known_section(key, manifest[key])
    for key in sorted((k for k in manifest.keys() if k not in _TOP_LEVEL_ORDER), key=lambda item: str(item)):
        out[str(key)] = _canonicalize_value(manifest[key])
    return out


def dumps_component_sensitivity_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest as deterministic, NaN-rejecting JSON."""
    canonical = canonicalize_component_sensitivity_manifest(manifest)
    return json.dumps(canonical, indent=2, allow_nan=False) + "\n"


def write_component_sensitivity_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a deterministic component sensitivity manifest JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps_component_sensitivity_manifest(manifest))


def _materialize_file_section(
    section: Mapping[str, Any],
    *,
    root: Path | None,
    preferred_order: tuple[str, ...],
    context: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _ordered_mapping_keys(section, preferred_order):
        value = section[key]
        if isinstance(value, Mapping):
            out[str(key)] = _materialize_file_entry(
                value,
                root=root,
                context=f"{context}.{key}",
            )
        else:
            out[str(key)] = deepcopy(value)
    return out


def _materialize_contest_eval(
    contest_eval: Mapping[str, Any],
    *,
    root: Path | None,
) -> dict[str, Any]:
    out = {str(k): deepcopy(v) for k, v in contest_eval.items()}
    for key in ("archive", "contest_auth_eval_json", "contest_auth_eval"):
        value = out.get(key)
        if isinstance(value, Mapping):
            out[key] = _materialize_file_entry(
                value,
                root=root,
                context=f"contest_eval.{key}",
            )
    return out


def _materialize_file_entry(
    entry: Mapping[str, Any],
    *,
    root: Path | None,
    context: str,
) -> dict[str, Any]:
    out = {str(k): deepcopy(v) for k, v in entry.items()}
    path_value = out.get("path")
    if not isinstance(path_value, str) or not path_value:
        return out

    resolved = _resolve_manifest_path(path_value, root=root)
    meta = custody_metadata(resolved)
    _assert_matching_metadata(out, "bytes", meta["bytes"], context)
    _assert_matching_metadata(out, "size_bytes", meta["bytes"], context)
    _assert_matching_metadata(out, "sha256", meta["sha256"], context)
    out["bytes"] = int(meta["bytes"])
    out["sha256"] = str(meta["sha256"])
    if "kind" in meta:
        out.setdefault("kind", meta["kind"])
    if "file_count" in meta:
        out.setdefault("file_count", meta["file_count"])
    return out


def _resolve_manifest_path(path_value: str, *, root: Path | None) -> Path:
    p = Path(path_value)
    if p.is_absolute() or root is None:
        return p
    return root / p


def _assert_matching_metadata(
    entry: Mapping[str, Any],
    key: str,
    expected: Any,
    context: str,
) -> None:
    if key not in entry:
        return
    actual = entry[key]
    if actual != expected:
        raise ComponentSensitivityArtifactError(
            f"{context}.{key}={actual!r} does not match computed custody value {expected!r}"
        )


def _canonicalize_known_section(key: str, value: Any) -> Any:
    if key == "inputs" and isinstance(value, Mapping):
        return _canonicalize_ordered_mapping(value, REQUIRED_INPUTS)
    if key in {"component_maps", "response_curves"} and isinstance(value, Mapping):
        return _canonicalize_ordered_mapping(value, COMPONENTS)
    if key == "stability" and isinstance(value, Mapping):
        return _canonicalize_stability(value)
    if key == "contest_eval" and isinstance(value, Mapping):
        return _canonicalize_ordered_mapping(
            value,
            (
                "archive",
                "archive_bytes",
                "archive_sha256",
                "contest_auth_eval_json",
                "contest_auth_eval_json_bytes",
                "contest_auth_eval_json_sha256",
                "device",
                "n_samples",
            ),
        )
    return _canonicalize_value(value)


def _canonicalize_stability(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section_name in _ordered_mapping_keys(value, ("cv", "rank", "top_k", "thresholds", "passed")):
        section = value[section_name]
        if isinstance(section, Mapping):
            out[str(section_name)] = _canonicalize_ordered_mapping(section, COMPONENTS)
        else:
            out[str(section_name)] = _canonicalize_value(section)
    return out


def _canonicalize_ordered_mapping(
    value: Mapping[str, Any],
    preferred_order: tuple[str, ...],
) -> dict[str, Any]:
    return {
        str(key): _canonicalize_value(value[key])
        for key in _ordered_mapping_keys(value, preferred_order)
    }


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_value(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if _is_nonstring_sequence(value):
        return [_canonicalize_value(item) for item in value]
    return value


def _ordered_mapping_keys(
    value: Mapping[str, Any],
    preferred_order: tuple[str, ...],
) -> list[Any]:
    keys = list(value.keys())
    out: list[Any] = [key for key in preferred_order if key in value]
    out.extend(sorted((key for key in keys if key not in preferred_order), key=lambda item: str(item)))
    return out


def validate_component_sensitivity_manifest(
    manifest: Mapping[str, Any],
    *,
    promotion: bool = True,
) -> None:
    """Validate a ``component_sensitivity_v1`` manifest.

    Promotion validation is intentionally fail-closed: CUDA is mandatory,
    diagnostic/proxy/random/fake markers are rejected, and all custody sections
    must be present. ``promotion=False`` keeps the same format checks but allows
    partial diagnostic artifacts without making them promotable.
    """
    if not isinstance(manifest, Mapping):
        raise ComponentSensitivityArtifactError(
            f"manifest must be a mapping, got {type(manifest).__name__}"
        )

    schema_version = manifest.get("schema_version")
    if schema_version != COMPONENT_SENSITIVITY_SCHEMA_VERSION:
        raise ComponentSensitivityArtifactError(
            "schema_version must be "
            f"{COMPONENT_SENSITIVITY_SCHEMA_VERSION}, got {schema_version!r}"
        )

    fmt = manifest.get("format")
    if fmt != COMPONENT_SENSITIVITY_FORMAT:
        raise ComponentSensitivityArtifactError(
            f"format must be {COMPONENT_SENSITIVITY_FORMAT!r}, got {fmt!r}"
        )

    if promotion:
        _reject_promotional_markers(manifest)
        _reject_non_cuda_device_fields(manifest)
        _require_cuda_device(_require_str(manifest, "device"), "device")
        if manifest.get("promotion_eligible") is not True:
            raise ComponentSensitivityArtifactError(
                "promotion_eligible must be exactly True for promotion artifacts"
            )
        _reject_promotion_blockers(manifest.get("promotion_blockers"))
        _validate_evidence_grade(_require_str(manifest, "evidence_grade"))

        _validate_inputs(_require_mapping(manifest, "inputs"), promotion=True)
        _validate_sample_plan(_require_mapping(manifest, "sample_plan"), promotion=True)
        _validate_component_maps(_require_mapping(manifest, "component_maps"), promotion=True)
        _validate_stability(_require_mapping(manifest, "stability"), promotion=True)
        _validate_response_curves(_require_mapping(manifest, "response_curves"), promotion=True)
        _validate_contest_eval(_require_mapping(manifest, "contest_eval"), promotion=True)
        return

    if "device" in manifest and not isinstance(manifest["device"], str):
        raise ComponentSensitivityArtifactError("device must be a string when present")
    if "evidence_grade" in manifest and not isinstance(manifest["evidence_grade"], str):
        raise ComponentSensitivityArtifactError("evidence_grade must be a string when present")
    if "inputs" in manifest:
        _validate_inputs(_require_mapping(manifest, "inputs"), promotion=False)
    if "sample_plan" in manifest:
        _validate_sample_plan(_require_mapping(manifest, "sample_plan"), promotion=False)
    if "component_maps" in manifest:
        _validate_component_maps(_require_mapping(manifest, "component_maps"), promotion=False)
    if "stability" in manifest:
        _validate_stability(_require_mapping(manifest, "stability"), promotion=False)
    if "response_curves" in manifest:
        _validate_response_curves(_require_mapping(manifest, "response_curves"), promotion=False)
    if "contest_eval" in manifest:
        _validate_contest_eval(_require_mapping(manifest, "contest_eval"), promotion=False)
    missing_requirements = _missing_promotion_requirements(manifest)
    if missing_requirements:
        _validate_promotion_blockers(
            manifest.get("promotion_blockers"),
            require=True,
            missing_requirements=missing_requirements,
        )
    elif "promotion_blockers" in manifest:
        _validate_promotion_blockers(
            manifest.get("promotion_blockers"),
            require=False,
            missing_requirements=(),
        )


def _validate_inputs(inputs: Mapping[str, Any], *, promotion: bool) -> None:
    names = REQUIRED_INPUTS if promotion else tuple(k for k in REQUIRED_INPUTS if k in inputs)
    for name in names:
        _validate_file_like_metadata(
            _require_mapping(inputs, name, context="inputs"),
            f"inputs.{name}",
            require_bytes=True,
            require_sha256=True,
        )

    if not promotion:
        for name, value in inputs.items():
            if name in REQUIRED_INPUTS:
                continue
            if isinstance(value, Mapping):
                _validate_file_like_metadata(
                    value,
                    f"inputs.{name}",
                    require_bytes=False,
                    require_sha256=False,
                )


def _validate_sample_plan(sample_plan: Mapping[str, Any], *, promotion: bool) -> None:
    if promotion or "calibration_pairs" in sample_plan:
        _validate_pairs(sample_plan.get("calibration_pairs"), "sample_plan.calibration_pairs")
    if promotion or "holdout_pairs" in sample_plan:
        _validate_pairs(sample_plan.get("holdout_pairs"), "sample_plan.holdout_pairs")
    if promotion or "split_seed" in sample_plan:
        _require_int(sample_plan, "split_seed", context="sample_plan", minimum=0)
    if promotion or "split_hash" in sample_plan:
        _validate_sha256(_require_str(sample_plan, "split_hash", context="sample_plan"), "sample_plan.split_hash")
    if promotion:
        _validate_full_sample_plan_coverage(sample_plan)


def _validate_component_maps(component_maps: Mapping[str, Any], *, promotion: bool) -> None:
    names = COMPONENTS if promotion else tuple(k for k in COMPONENTS if k in component_maps)
    for name in names:
        context = f"component_maps.{name}"
        entry = _require_mapping(component_maps, name, context="component_maps")
        _validate_file_like_metadata(
            entry,
            context,
            require_bytes=promotion,
            require_sha256=True,
        )
        if promotion or "scorer_target" in entry:
            target = _require_str(entry, "scorer_target", context=context)
            if _normalise_component_target(target) != name:
                raise ComponentSensitivityArtifactError(
                    f"{context}.scorer_target must be {name!r}, got {target!r}"
                )
        if promotion:
            _validate_component_map_certification(entry, component=name, context=context)
        _validate_tensor_metadata(entry, context)


def _validate_component_map_certification(
    entry: Mapping[str, Any],
    *,
    component: str,
    context: str,
) -> None:
    map_format = _require_str(entry, "map_format", context=context)
    if map_format != CERTIFIED_COMPONENT_MAP_FORMAT:
        raise ComponentSensitivityArtifactError(
            f"{context}.map_format must be {CERTIFIED_COMPONENT_MAP_FORMAT!r}, "
            f"got {map_format!r}"
        )
    certification = _require_mapping(entry, "certification", context=context)
    cert_format = _require_str(certification, "format", context=f"{context}.certification")
    if cert_format != COMPONENT_MAP_CERTIFICATION_FORMAT:
        raise ComponentSensitivityArtifactError(
            f"{context}.certification.format must be {COMPONENT_MAP_CERTIFICATION_FORMAT!r}, "
            f"got {cert_format!r}"
        )
    cert_component = _require_str(certification, "component", context=f"{context}.certification")
    if cert_component != component:
        raise ComponentSensitivityArtifactError(
            f"{context}.certification.component must be {component!r}, got {cert_component!r}"
        )
    _require_cuda_device(
        _require_str(certification, "device", context=f"{context}.certification"),
        f"{context}.certification.device",
    )
    for key in (
        "official_component_response",
        "canonical_scorer_path",
        "promotion_eligible",
    ):
        _require_exact_true(certification, key, context=f"{context}.certification")
    for key in (
        "source_map_sha256",
        "official_response_curve_sha256",
        "stability_sha256",
        "sample_plan_sha256",
        "baseline_archive_sha256",
        "contest_auth_eval_json_sha256",
    ):
        _validate_sha256(
            _require_str(certification, key, context=f"{context}.certification"),
            f"{context}.certification.{key}",
        )
    if "prediction_deltas_sha256" in certification:
        _validate_sha256(
            _require_str(certification, "prediction_deltas_sha256", context=f"{context}.certification"),
            f"{context}.certification.prediction_deltas_sha256",
        )
    if "perturbation_basis_sha256" in certification:
        _validate_sha256(
            _require_str(certification, "perturbation_basis_sha256", context=f"{context}.certification"),
            f"{context}.certification.perturbation_basis_sha256",
        )
    if "review_packet_sha256" in certification:
        _validate_sha256(
            _require_str(certification, "review_packet_sha256", context=f"{context}.certification"),
            f"{context}.certification.review_packet_sha256",
        )
    _require_positive_int_value(
        certification.get("baseline_archive_bytes"),
        f"{context}.certification.baseline_archive_bytes",
    )
    clean_passes = _require_int(
        certification,
        "review_clean_passes",
        context=f"{context}.certification",
        minimum=3,
    )
    if clean_passes < 3:
        raise ComponentSensitivityArtifactError(
            f"{context}.certification.review_clean_passes must be at least 3"
        )
    unresolved = certification.get("review_unresolved_blockers", [])
    if unresolved not in ([], None):
        raise ComponentSensitivityArtifactError(
            f"{context}.certification.review_unresolved_blockers must be empty"
        )
    response_gate = _require_mapping(
        certification,
        "response_gate_results",
        context=f"{context}.certification",
    )
    for key in _RESPONSE_GATE_RESULT_TRUE_KEYS:
        _require_exact_true(response_gate, key, context=f"{context}.certification.response_gate_results")
    _require_exact_true(response_gate, "finite_values", context=f"{context}.certification.response_gate_results")
    _require_finite_number(
        response_gate.get("zero_repro_error"),
        f"{context}.certification.response_gate_results.zero_repro_error",
        minimum=0.0,
    )
    _require_finite_number(
        response_gate.get("observed_delta_max"),
        f"{context}.certification.response_gate_results.observed_delta_max",
        minimum=0.0,
    )
    _require_finite_number(
        response_gate.get("max_relative_prediction_error"),
        f"{context}.certification.response_gate_results.max_relative_prediction_error",
        minimum=0.0,
    )
    stability_gate = _require_mapping(
        certification,
        "stability_gate_results",
        context=f"{context}.certification",
    )
    _require_exact_true(stability_gate, "passed", context=f"{context}.certification.stability_gate_results")
    if _validate_finite_numeric_tree(stability_gate, f"{context}.certification.stability_gate_results") == 0:
        raise ComponentSensitivityArtifactError(
            f"{context}.certification.stability_gate_results must contain finite numeric metrics"
        )


def _validate_stability(stability: Mapping[str, Any], *, promotion: bool) -> None:
    if promotion:
        _require_exact_true(stability, "passed", context="stability")
        thresholds = _require_mapping(stability, "thresholds", context="stability")
        if _validate_finite_numeric_tree(thresholds, "stability.thresholds") == 0:
            raise ComponentSensitivityArtifactError(
                "stability.thresholds must contain finite numeric pass thresholds"
            )

    for section_name in ("cv", "rank", "top_k"):
        if promotion or section_name in stability:
            section = _require_mapping(stability, section_name, context="stability")
        else:
            continue

        names = COMPONENTS if promotion else tuple(k for k in COMPONENTS if k in section)
        for component in names:
            field = f"stability.{section_name}.{component}"
            if component not in section:
                raise ComponentSensitivityArtifactError(f"{field} is required")
            if _validate_finite_numeric_tree(section[component], field) == 0:
                raise ComponentSensitivityArtifactError(f"{field} must contain finite numeric values")


def _validate_response_curves(response_curves: Mapping[str, Any], *, promotion: bool) -> None:
    names = COMPONENTS if promotion else tuple(k for k in COMPONENTS if k in response_curves)
    for name in names:
        context = f"response_curves.{name}"
        curve = _require_mapping(response_curves, name, context="response_curves")
        _validate_file_like_metadata(
            curve,
            context,
            require_bytes=promotion,
            require_sha256=True,
        )
        path = _require_str(curve, "path", context=context)
        if not path:
            raise ComponentSensitivityArtifactError(f"{context}.path must be non-empty")
        _require_int(curve, "count", context=context, minimum=1)
        holdout_error = _require_any(curve, ("holdout_error", "holdout_error_max", "max_holdout_error"), context)
        _require_finite_number(holdout_error, f"{context}.holdout_error", minimum=0.0)
        if promotion:
            _validate_official_response_curve_gate(curve, component=name, context=context)


def _validate_official_response_curve_gate(
    curve: Mapping[str, Any],
    *,
    component: str,
    context: str,
) -> None:
    _require_exact_true(curve, "official_component_response", context=context)
    _require_exact_true(curve, "passed", context=context)
    _validate_response_curve_gate_results(curve, context=context)
    _validate_response_curve_promotion_blockers(curve, context=context)

    gate_spec = _require_mapping(curve, "gate_spec", context=context)
    if _validate_finite_numeric_tree(gate_spec, f"{context}.gate_spec") == 0:
        raise ComponentSensitivityArtifactError(
            f"{context}.gate_spec must contain finite numeric pass thresholds"
        )

    _validate_official_component_readout(curve, component=component, context=context)
    _validate_response_curve_coverage(curve, context=context)


def _validate_response_curve_gate_results(curve: Mapping[str, Any], *, context: str) -> None:
    gate_results = _require_mapping(curve, "gate_results", context=context)
    _validate_finite_numeric_tree(gate_results, f"{context}.gate_results")
    for key in _RESPONSE_GATE_RESULT_TRUE_KEYS:
        _require_exact_true(gate_results, key, context=f"{context}.gate_results")
    if "external_baseline_repro" in gate_results:
        _require_exact_true(gate_results, "external_baseline_repro", context=f"{context}.gate_results")


def _validate_response_curve_promotion_blockers(curve: Mapping[str, Any], *, context: str) -> None:
    if "promotion_blockers" not in curve:
        return
    blockers = curve["promotion_blockers"]
    if _is_nonstring_sequence(blockers) and len(blockers) == 0:
        return
    raise ComponentSensitivityArtifactError(
        f"{context}.promotion_blockers must be empty for promotion response curves"
    )


def _validate_official_component_readout(
    curve: Mapping[str, Any],
    *,
    component: str,
    context: str,
) -> None:
    value = _optional_any(curve, ("component_readout", "official_readout", "readout"))
    if not isinstance(value, str) or not value.strip():
        raise ComponentSensitivityArtifactError(
            f"{context}.component_readout must identify the official component readout"
        )
    readout = _normalise_marker(value)
    if readout not in _RESPONSE_READOUT_ALIASES[component]:
        raise ComponentSensitivityArtifactError(
            f"{context}.component_readout={value!r} is not an official {component} readout"
        )


def _validate_response_curve_coverage(curve: Mapping[str, Any], *, context: str) -> None:
    value = _optional_any(curve, _RESPONSE_KIND_KEYS)
    if not isinstance(value, str) or not value.strip():
        raise ComponentSensitivityArtifactError(
            f"{context}.response_kind must be 'symmetric' or 'directional'"
        )
    kind = _normalise_marker(value)
    if kind in {"symmetric", "symmetric_curve", "central_difference", "central_differences"}:
        _validate_symmetric_response_coverage(curve, context=context)
        return
    if kind in {"directional", "directional_action", "one_sided", "one_sided_action"}:
        _validate_directional_response_coverage(curve, context=context)
        return
    raise ComponentSensitivityArtifactError(
        f"{context}.response_kind must be 'symmetric' or 'directional', got {value!r}"
    )


def _validate_symmetric_response_coverage(curve: Mapping[str, Any], *, context: str) -> None:
    pair_count = _optional_any(curve, ("symmetric_epsilon_pairs", "central_difference_pairs"))
    if pair_count is not None:
        _require_positive_int_value(pair_count, f"{context}.symmetric_epsilon_pairs")
        return

    epsilon_values = _optional_any(curve, _RESPONSE_EPSILON_KEYS)
    if epsilon_values is None:
        raise ComponentSensitivityArtifactError(
            f"{context}.epsilon_ladder or symmetric_epsilon_pairs is required for symmetric curves"
        )
    values = _require_finite_number_sequence(epsilon_values, f"{context}.epsilon_ladder")
    if not any(_is_close_to_zero(value) for value in values):
        raise ComponentSensitivityArtifactError(f"{context}.epsilon_ladder must include eps=0")
    positives = [value for value in values if value > 0 and not _is_close_to_zero(value)]
    negatives = [value for value in values if value < 0 and not _is_close_to_zero(value)]
    if not positives or not negatives:
        raise ComponentSensitivityArtifactError(
            f"{context}.epsilon_ladder must include at least one -eps/+eps pair"
        )
    for positive in positives:
        if any(_is_close(value, -positive) for value in negatives):
            return
    raise ComponentSensitivityArtifactError(
        f"{context}.epsilon_ladder must include a matched -eps/+eps pair"
    )


def _validate_directional_response_coverage(curve: Mapping[str, Any], *, context: str) -> None:
    metadata = _optional_any(curve, ("directional_action", "directional_actions", "action_point"))
    if metadata is None:
        raise ComponentSensitivityArtifactError(
            f"{context}.directional_action is required for directional curves"
        )
    if isinstance(metadata, Mapping) and not metadata:
        raise ComponentSensitivityArtifactError(f"{context}.directional_action must not be empty")
    if _is_nonstring_sequence(metadata) and not metadata:
        raise ComponentSensitivityArtifactError(f"{context}.directional_action must not be empty")
    if isinstance(metadata, str) and not metadata.strip():
        raise ComponentSensitivityArtifactError(f"{context}.directional_action must be non-empty")


def _validate_contest_eval(contest_eval: Mapping[str, Any], *, promotion: bool) -> None:
    archive_meta = contest_eval.get("archive")
    if archive_meta is not None and not isinstance(archive_meta, Mapping):
        raise ComponentSensitivityArtifactError("contest_eval.archive must be a mapping when present")

    archive_bytes = _optional_any(contest_eval, ("archive_bytes", "archive_size_bytes"))
    if archive_bytes is None and isinstance(archive_meta, Mapping):
        archive_bytes = _optional_any(archive_meta, ("bytes", "size_bytes"))
    if archive_bytes is None:
        raise ComponentSensitivityArtifactError("contest_eval.archive_bytes is required")
    _require_positive_int_value(archive_bytes, "contest_eval.archive_bytes")

    archive_sha256 = _optional_any(contest_eval, ("archive_sha256",))
    if archive_sha256 is None and isinstance(archive_meta, Mapping):
        archive_sha256 = _optional_any(archive_meta, ("sha256",))
    if archive_sha256 is None:
        raise ComponentSensitivityArtifactError("contest_eval.archive_sha256 is required")
    _validate_sha256(archive_sha256, "contest_eval.archive_sha256")

    json_sha256 = _optional_any(
        contest_eval,
        (
            "contest_auth_eval_json_sha256",
            "contest_auth_eval_sha256",
            "auth_eval_json_sha256",
        ),
    )
    for key in ("contest_auth_eval_json", "contest_auth_eval"):
        nested = contest_eval.get(key)
        if json_sha256 is None and isinstance(nested, Mapping):
            json_sha256 = _optional_any(nested, ("sha256", "json_sha256"))
    if json_sha256 is not None:
        _validate_sha256(json_sha256, "contest_eval.contest_auth_eval_json_sha256")
    json_bytes = _optional_any(
        contest_eval,
        (
            "contest_auth_eval_json_bytes",
            "contest_auth_eval_bytes",
            "auth_eval_json_bytes",
        ),
    )
    json_meta = None
    for key in ("contest_auth_eval_json", "contest_auth_eval"):
        nested = contest_eval.get(key)
        if isinstance(nested, Mapping):
            json_meta = nested
            break
    if isinstance(json_meta, Mapping):
        _validate_file_like_metadata(
            json_meta,
            "contest_eval.contest_auth_eval_json",
            require_bytes=promotion,
            require_sha256=True,
        )
        if json_bytes is None:
            json_bytes = _optional_any(json_meta, ("bytes", "size_bytes"))
    if promotion and (json_sha256 is None or json_bytes is None):
        raise ComponentSensitivityArtifactError(
            "contest_eval.contest_auth_eval_json custody bytes and sha256 are required"
        )
    if json_bytes is not None:
        _require_positive_int_value(json_bytes, "contest_eval.contest_auth_eval_json.bytes")

    if promotion and "device" in contest_eval:
        _require_cuda_device(_require_str(contest_eval, "device", context="contest_eval"), "contest_eval.device")
    if promotion or "n_samples" in contest_eval:
        n_samples = _require_int(contest_eval, "n_samples", context="contest_eval", minimum=1)
        if promotion and n_samples != CONTEST_SAMPLE_COUNT:
            raise ComponentSensitivityArtifactError(
                f"contest_eval.n_samples must be {CONTEST_SAMPLE_COUNT}, got {n_samples}"
            )


def _reject_promotion_blockers(value: Any) -> None:
    if value is None:
        return
    if _is_nonstring_sequence(value) and len(value) == 0:
        return
    raise ComponentSensitivityArtifactError(
        "promotion_blockers must be empty for promotion artifacts"
    )


def _validate_promotion_blockers(
    value: Any,
    *,
    require: bool,
    missing_requirements: Sequence[str],
) -> None:
    if not _is_nonstring_sequence(value):
        if require:
            missing = ", ".join(missing_requirements[:6])
            suffix = "..." if len(missing_requirements) > 6 else ""
            raise ComponentSensitivityArtifactError(
                "promotion_blockers with mathematical_explanation are required "
                f"because the manifest is not promotable; missing: {missing}{suffix}"
            )
        return
    if require and not value:
        raise ComponentSensitivityArtifactError(
            "promotion_blockers must not be empty for non-promotable manifests"
        )
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ComponentSensitivityArtifactError(
                f"promotion_blockers[{index}] must be a mapping"
            )
        code = _require_str(item, "code", context=f"promotion_blockers[{index}]")
        if not code.strip():
            raise ComponentSensitivityArtifactError(
                f"promotion_blockers[{index}].code must be non-empty"
            )
        explanation = _optional_any(item, ("mathematical_explanation", "explanation"))
        if not isinstance(explanation, str) or not explanation.strip():
            raise ComponentSensitivityArtifactError(
                f"promotion_blockers[{index}].mathematical_explanation must be a non-empty string"
            )


def _missing_promotion_requirements(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    missing: list[str] = []
    for key in ("inputs", "sample_plan", "component_maps", "stability", "response_curves", "contest_eval"):
        if key not in manifest:
            missing.append(key)

    component_maps = manifest.get("component_maps")
    if isinstance(component_maps, Mapping):
        for component in COMPONENTS:
            if component not in component_maps:
                missing.append(f"component_maps.{component}")

    response_curves = manifest.get("response_curves")
    if isinstance(response_curves, Mapping):
        for component in COMPONENTS:
            if component not in response_curves:
                missing.append(f"response_curves.{component}")

    stability = manifest.get("stability")
    if isinstance(stability, Mapping):
        for section_name in ("cv", "rank", "top_k"):
            section = stability.get(section_name)
            if not isinstance(section, Mapping):
                missing.append(f"stability.{section_name}")
                continue
            for component in COMPONENTS:
                if component not in section:
                    missing.append(f"stability.{section_name}.{component}")

    return tuple(missing)


def _validate_evidence_grade(value: str) -> None:
    grade = value.strip().lower()
    if grade not in _PROMOTION_EVIDENCE_GRADES:
        raise ComponentSensitivityArtifactError(
            "evidence_grade must be A or A++ for promotion artifacts; "
            f"got {value!r}"
        )


def _validate_file_like_metadata(
    metadata: Mapping[str, Any],
    context: str,
    *,
    require_bytes: bool,
    require_sha256: bool,
) -> None:
    if "path" in metadata and not isinstance(metadata["path"], str):
        raise ComponentSensitivityArtifactError(f"{context}.path must be a string when present")

    byte_count = _optional_any(metadata, ("bytes", "size_bytes"))
    if byte_count is None:
        if require_bytes:
            raise ComponentSensitivityArtifactError(f"{context}.bytes is required")
    else:
        _require_positive_int_value(byte_count, f"{context}.bytes")

    sha256 = _optional_any(metadata, ("sha256",))
    if sha256 is None:
        if require_sha256:
            raise ComponentSensitivityArtifactError(f"{context}.sha256 is required")
    else:
        _validate_sha256(sha256, f"{context}.sha256")


def _validate_tensor_metadata(entry: Mapping[str, Any], context: str) -> None:
    if "tensor" in entry:
        _validate_single_tensor_metadata(_require_mapping(entry, "tensor", context=context), f"{context}.tensor")
        return

    if "tensor_metadata" in entry:
        metadata = _require_mapping(entry, "tensor_metadata", context=context)
        if "dtype" in metadata or "shape" in metadata or "numel" in metadata:
            _validate_single_tensor_metadata(metadata, f"{context}.tensor_metadata")
            return
        _validate_tensor_metadata_collection(metadata, f"{context}.tensor_metadata")
        return

    if "tensors" in entry:
        tensors = entry["tensors"]
        if isinstance(tensors, Mapping):
            _validate_tensor_metadata_collection(tensors, f"{context}.tensors")
            return
        if _is_nonstring_sequence(tensors):
            if not tensors:
                raise ComponentSensitivityArtifactError(f"{context}.tensors must not be empty")
            for index, item in enumerate(tensors):
                if not isinstance(item, Mapping):
                    raise ComponentSensitivityArtifactError(
                        f"{context}.tensors[{index}] must be a mapping"
                    )
                _validate_single_tensor_metadata(item, f"{context}.tensors[{index}]")
            return
        raise ComponentSensitivityArtifactError(f"{context}.tensors must be a mapping or sequence")

    if "tensor_count" in entry and "dtype" in entry:
        _require_positive_int_value(entry["tensor_count"], f"{context}.tensor_count")
        if not isinstance(entry["dtype"], str) or not entry["dtype"]:
            raise ComponentSensitivityArtifactError(f"{context}.dtype must be a non-empty string")
        return

    raise ComponentSensitivityArtifactError(
        f"{context} must include tensor, tensors, tensor_metadata, or tensor_count metadata"
    )


def _validate_tensor_metadata_collection(collection: Mapping[str, Any], context: str) -> None:
    if not collection:
        raise ComponentSensitivityArtifactError(f"{context} must not be empty")
    for name, item in collection.items():
        if not isinstance(item, Mapping):
            raise ComponentSensitivityArtifactError(f"{context}.{name} must be a mapping")
        _validate_single_tensor_metadata(item, f"{context}.{name}")


def _validate_single_tensor_metadata(metadata: Mapping[str, Any], context: str) -> None:
    dtype = metadata.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ComponentSensitivityArtifactError(f"{context}.dtype must be a non-empty string")

    if "shape" in metadata:
        shape = metadata["shape"]
        if not _is_nonstring_sequence(shape) or not shape:
            raise ComponentSensitivityArtifactError(f"{context}.shape must be a non-empty sequence")
        for index, dim in enumerate(shape):
            _require_positive_int_value(dim, f"{context}.shape[{index}]")
        return

    if "numel" in metadata:
        _require_positive_int_value(metadata["numel"], f"{context}.numel")
        return

    raise ComponentSensitivityArtifactError(f"{context} must include shape or numel")


def _validate_pairs(value: Any, field: str) -> None:
    if not _is_nonstring_sequence(value):
        raise ComponentSensitivityArtifactError(f"{field} must be a non-empty sequence")
    if not value:
        raise ComponentSensitivityArtifactError(f"{field} must not be empty")


def _validate_full_sample_plan_coverage(sample_plan: Mapping[str, Any]) -> None:
    calibration = sample_plan.get("calibration_pairs")
    holdout = sample_plan.get("holdout_pairs")
    if not _is_nonstring_sequence(calibration):
        raise ComponentSensitivityArtifactError(
            "sample_plan.calibration_pairs must be a non-empty sequence"
        )
    if not _is_nonstring_sequence(holdout):
        raise ComponentSensitivityArtifactError(
            "sample_plan.holdout_pairs must be a non-empty sequence"
        )

    records: list[tuple[str, int, Mapping[str, Any]]] = []
    for split_name, pairs in (("calibration_pairs", calibration), ("holdout_pairs", holdout)):
        for index, item in enumerate(pairs):
            context = f"sample_plan.{split_name}[{index}]"
            if not isinstance(item, Mapping):
                raise ComponentSensitivityArtifactError(f"{context} must be a mapping")
            pair_index = _require_int(item, "pair_index", context=context, minimum=0)
            if "video" in item:
                _require_int(item, "video", context=context, minimum=0)
            if "t" in item:
                t = _require_int(item, "t", context=context, minimum=0)
                if t != 2 * pair_index:
                    raise ComponentSensitivityArtifactError(
                        f"{context}.t must equal 2 * pair_index for absolute contest pair IDs"
                    )
            if "t1" in item:
                t1 = _require_int(item, "t1", context=context, minimum=0)
                if t1 != 2 * pair_index + 1:
                    raise ComponentSensitivityArtifactError(
                        f"{context}.t1 must equal 2 * pair_index + 1 for absolute contest pair IDs"
                    )
            records.append((split_name, pair_index, item))

    pair_ids = [pair_index for _split, pair_index, _item in records]
    if len(pair_ids) != CONTEST_SAMPLE_COUNT:
        raise ComponentSensitivityArtifactError(
            f"sample_plan must cover exactly {CONTEST_SAMPLE_COUNT} absolute contest pairs, "
            f"got {len(pair_ids)}"
        )
    if len(set(pair_ids)) != len(pair_ids):
        raise ComponentSensitivityArtifactError(
            "sample_plan calibration/holdout pair_index values must be unique"
        )
    expected = list(range(CONTEST_SAMPLE_COUNT))
    if sorted(pair_ids) != expected:
        raise ComponentSensitivityArtifactError(
            "sample_plan pair_index values must be absolute contest pair IDs 0..599"
        )


def _require_mapping(parent: Mapping[str, Any], key: str, *, context: str | None = None) -> Mapping[str, Any]:
    if key not in parent:
        field = f"{context}.{key}" if context else key
        raise ComponentSensitivityArtifactError(f"{field} is required")
    value = parent[key]
    field = f"{context}.{key}" if context else key
    if not isinstance(value, Mapping):
        raise ComponentSensitivityArtifactError(
            f"{field} must be a mapping, got {type(value).__name__}"
        )
    return value


def _require_str(parent: Mapping[str, Any], key: str, *, context: str | None = None) -> str:
    if key not in parent:
        field = f"{context}.{key}" if context else key
        raise ComponentSensitivityArtifactError(f"{field} is required")
    value = parent[key]
    field = f"{context}.{key}" if context else key
    if not isinstance(value, str):
        raise ComponentSensitivityArtifactError(f"{field} must be a string")
    return value


def _require_int(
    parent: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: int,
) -> int:
    if key not in parent:
        raise ComponentSensitivityArtifactError(f"{context}.{key} is required")
    value = parent[key]
    _require_int_value(value, f"{context}.{key}", minimum=minimum)
    return int(value)


def _require_any(parent: Mapping[str, Any], keys: tuple[str, ...], context: str) -> Any:
    value = _optional_any(parent, keys)
    if value is None:
        joined = "|".join(keys)
        raise ComponentSensitivityArtifactError(f"{context}.{joined} is required")
    return value


def _optional_any(parent: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in parent:
            return parent[key]
    return None


def _require_exact_true(parent: Mapping[str, Any], key: str, *, context: str) -> None:
    if parent.get(key) is not True:
        raise ComponentSensitivityArtifactError(f"{context}.{key} must be exactly true")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ComponentSensitivityArtifactError(f"{field} must be a 64-character sha256 hex digest")


def _require_cuda_device(value: str, field: str) -> None:
    device = value.strip().lower()
    if device != "cuda" and not device.startswith("cuda:"):
        raise ComponentSensitivityArtifactError(
            f"{field} must be CUDA for promotion artifacts; got {value!r}"
        )


def _reject_non_cuda_device_fields(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_norm = _normalise_marker(str(key))
            child_path = f"{path}.{key}"
            if key_norm in _DEVICE_FIELD_KEYS and isinstance(child, str):
                _require_cuda_device(child, child_path)
            _reject_non_cuda_device_fields(child, path=child_path)
        return

    if _is_nonstring_sequence(value):
        for index, child in enumerate(value):
            _reject_non_cuda_device_fields(child, path=f"{path}[{index}]")


def _normalise_component_target(value: str) -> str:
    target = _normalise_marker(value)
    aliases = {
        "pose": "posenet",
        "pose_net": "posenet",
        "seg": "segnet",
        "seg_net": "segnet",
        "combined_score": "combined",
        "full_score": "combined",
    }
    return aliases.get(target, target)


def _require_finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ComponentSensitivityArtifactError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ComponentSensitivityArtifactError(f"{field} must be finite")
    if minimum is not None and out < minimum:
        raise ComponentSensitivityArtifactError(f"{field} must be >= {minimum}")
    return out


def _validate_finite_numeric_tree(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, Real):
        _require_finite_number(value, field)
        return 1
    if isinstance(value, Mapping):
        total = 0
        for key, child in value.items():
            total += _validate_finite_numeric_tree(child, f"{field}.{key}")
        return total
    if _is_nonstring_sequence(value):
        total = 0
        for index, child in enumerate(value):
            total += _validate_finite_numeric_tree(child, f"{field}[{index}]")
        return total
    return 0


def _require_finite_number_sequence(value: Any, field: str) -> list[float]:
    if not _is_nonstring_sequence(value) or not value:
        raise ComponentSensitivityArtifactError(f"{field} must be a non-empty numeric sequence")
    out: list[float] = []
    for index, item in enumerate(value):
        out.append(_require_finite_number(item, f"{field}[{index}]"))
    return out


def _is_close_to_zero(value: float) -> bool:
    return abs(value) <= 1e-12


def _is_close(a: float, b: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= 1e-9 * scale


def _require_positive_int_value(value: Any, field: str) -> int:
    return _require_int_value(value, field, minimum=1)


def _require_int_value(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentSensitivityArtifactError(f"{field} must be an integer")
    if value < minimum:
        raise ComponentSensitivityArtifactError(f"{field} must be >= {minimum}")
    return int(value)


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _reject_promotional_markers(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_norm = _normalise_marker(str(key))
            child_path = f"{path}.{key}"
            if key_norm in _PROMOTION_FLAG_KEYS and _marker_truthy(child):
                raise ComponentSensitivityArtifactError(
                    f"{child_path} marks a debug/smoke/fake/random/proxy artifact; "
                    "promotion validation is fail-closed"
                )
            _reject_promotional_markers(child, path=child_path)
        return

    if _is_nonstring_sequence(value):
        for index, child in enumerate(value):
            _reject_promotional_markers(child, path=f"{path}[{index}]")
        return

    if isinstance(value, str):
        marker = _normalise_marker(value)
        if marker in _PROMOTION_STRING_MARKERS or any(token in marker for token in _PROMOTION_STRING_SUBSTRINGS):
            raise ComponentSensitivityArtifactError(
                f"{path}={value!r} marks a debug/smoke/fake/random/proxy artifact; "
                "promotion validation is fail-closed"
            )


def _marker_truthy(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none"}
    return bool(value)


def _normalise_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


__all__ = [
    "COMPONENT_SENSITIVITY_FORMAT",
    "COMPONENT_SENSITIVITY_SCHEMA_VERSION",
    "ComponentSensitivityArtifactError",
    "ComponentSensitivityManifestError",
    "canonicalize_component_sensitivity_manifest",
    "custody_metadata",
    "dumps_component_sensitivity_manifest",
    "file_metadata",
    "materialize_component_sensitivity_manifest",
    "sha256_file",
    "validate_component_sensitivity_manifest",
    "write_component_sensitivity_manifest",
]
