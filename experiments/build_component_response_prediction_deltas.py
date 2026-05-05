#!/usr/bin/env python3
"""Build deterministic map-projected predictions for component response plans.

This tool prepares the ``predicted_delta`` payload consumed by
``build_component_response_perturbation_plan.py``. It intentionally does not
run the scorer and it refuses to consume official response curves or
``contest_auth_eval.json`` artifacts. The output is a pre-response prediction
artifact, not evidence that a component-response curve passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.sensitivity_map import load_sensitivity_map  # noqa: E402


PREDICTION_DELTAS_FORMAT = "official_component_response_prediction_deltas_v1"
PERTURBATION_BASIS_FORMAT = "perturbation_basis_v1"
PRODUCER = "experiments/build_component_response_prediction_deltas.py"
COMPONENTS = ("posenet", "segnet", "combined")
DEFAULT_EPSILONS = (-1.0, 0.0, 1.0)
SCORE_EPS = 1e-12  # [heuristic:numerical-floor for division-safe component-response score ratios]
LEAKAGE_KEYS = {
    "actual_delta",
    "contest_auth_eval",
    "contest_auth_eval_json",
    "holdout_error",
    "measured_delta",
    "observed_delta",
    "official_response",
    "official_response_curve",
    "response_curve",
}


class ComponentResponsePredictionError(ValueError):
    """Raised when prediction inputs cannot form a rigorous artifact."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentResponsePredictionError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ComponentResponsePredictionError(f"{field} must be finite")
    return out


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponsePredictionError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentResponsePredictionError(f"{label} must be a JSON object: {path}")
    return payload


def _reject_observed_response_leakage(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_s = str(key)
            if key_s.lower() in LEAKAGE_KEYS:
                raise ComponentResponsePredictionError(
                    f"{path}.{key_s} is an observed-response field; prediction "
                    "artifacts must be authored before official response evaluation"
                )
            _reject_observed_response_leakage(child, path=f"{path}.{key_s}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_observed_response_leakage(child, path=f"{path}[{index}]")


def _load_basis_atoms(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path, label="perturbation basis")
    _reject_observed_response_leakage(payload)
    if payload.get("format") != PERTURBATION_BASIS_FORMAT:
        raise ComponentResponsePredictionError(
            f"{path}: expected format {PERTURBATION_BASIS_FORMAT!r}"
        )
    raw_atoms = payload.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise ComponentResponsePredictionError(f"{path}: atoms must be a non-empty list")
    atoms: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_atoms):
        if not isinstance(raw, Mapping):
            raise ComponentResponsePredictionError(f"{path}: atoms[{index}] must be an object")
        member = raw.get("member")
        if member != "renderer.bin":
            raise ComponentResponsePredictionError(
                f"{path}: atoms[{index}].member must be 'renderer.bin'"
            )
        offset = raw.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ComponentResponsePredictionError(f"{path}: atoms[{index}].offset invalid")
        delta = raw.get("delta_per_epsilon", raw.get("direction"))
        if isinstance(delta, bool) or not isinstance(delta, int) or delta == 0:
            raise ComponentResponsePredictionError(
                f"{path}: atoms[{index}].delta_per_epsilon must be a nonzero integer"
            )
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        layer = raw.get("layer_name", metadata.get("layer_name"))
        if not isinstance(layer, str) or not layer:
            raise ComponentResponsePredictionError(
                f"{path}: atoms[{index}].layer_name is required for map projection"
            )
        channel = raw.get(
            "channel_index",
            raw.get("channel", metadata.get("channel_index", metadata.get("channel"))),
        )
        if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
            raise ComponentResponsePredictionError(
                f"{path}: atoms[{index}].channel_index must be a nonnegative integer"
            )
        atom_id = raw.get("atom_id") or f"renderer.bin@{offset}:d{delta:+d}"
        if not isinstance(atom_id, str) or not atom_id:
            raise ComponentResponsePredictionError(f"{path}: atoms[{index}].atom_id invalid")
        key = ("renderer.bin", int(offset))
        if key in seen:
            raise ComponentResponsePredictionError(f"{path}: duplicate atom offset {key}")
        seen.add(key)
        atoms.append(
            {
                "atom_id": atom_id,
                "member": "renderer.bin",
                "offset": int(offset),
                "delta_per_epsilon": int(delta),
                "layer_name": layer,
                "channel_index": int(channel),
            }
        )
    return atoms


def _atom_set_sha256(atoms: list[Mapping[str, Any]]) -> str:
    payload = [
        {
            "atom_id": str(atom["atom_id"]),
            "delta_per_epsilon": int(atom["delta_per_epsilon"]),
            "member": str(atom["member"]),
            "offset": int(atom["offset"]),
        }
        for atom in sorted(
            atoms,
            key=lambda item: (
                str(item["member"]),
                int(item["offset"]),
                str(item["atom_id"]),
                int(item["delta_per_epsilon"]),
            ),
        )
    ]
    return _canonical_hash(payload)


def _load_component_maps(paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    for component in COMPONENTS:
        path = Path(paths[component]).resolve()
        sensitivities, meta = load_sensitivity_map(path)
        maps[component] = {key: value.detach().to("cpu").float() for key, value in sensitivities.items()}
        metadata[component] = {
            **_file_meta(path),
            "map_metadata": meta,
        }
    return maps, metadata


def _validate_archive_byte_prediction_contract(metadata: Mapping[str, Any]) -> None:
    """Reject map-unit contracts that cannot predict archive byte mutations."""

    bad_components: list[str] = []
    for component in COMPONENTS:
        meta = metadata.get(component)
        map_meta = meta.get("map_metadata") if isinstance(meta, Mapping) else None
        if not isinstance(map_meta, Mapping):
            continue
        if (
            map_meta.get("sensitivity_source")
            == "direct_renderer_cuda_finite_difference_component_response"
            and map_meta.get("archive_byte_prediction_eligible") is not True
        ):
            bad_components.append(component)
    if bad_components:
        raise ComponentResponsePredictionError(
            "direct finite-difference component sensitivity maps are channel weight-space "
            "response maps, not archive-byte response maps. Refusing to predict byte "
            "mutation deltas for components "
            f"{bad_components} without archive_byte_prediction_eligible=true metadata "
            "from a reviewed byte-basis calibration."
        )


def _atom_sensitivity(
    maps: Mapping[str, Mapping[str, Any]],
    *,
    component: str,
    atom: Mapping[str, Any],
) -> float:
    layer = str(atom["layer_name"])
    key = layer if layer.endswith(".weight") else f"{layer}.weight"
    channel = int(atom["channel_index"])
    component_map = maps[component]
    value = component_map.get(key)
    if value is None:
        raise ComponentResponsePredictionError(
            f"{component}: sensitivity map missing {key} required by atom {atom['atom_id']}"
        )
    if channel >= int(value.numel()):
        raise ComponentResponsePredictionError(
            f"{component}: atom {atom['atom_id']} channel {channel} outside {key} "
            f"with {int(value.numel())} channel(s)"
        )
    sensitivity = float(value.reshape(-1)[channel].item())
    if not math.isfinite(sensitivity) or sensitivity < 0.0:
        raise ComponentResponsePredictionError(
            f"{component}: non-finite or negative sensitivity for atom {atom['atom_id']}"
        )
    return sensitivity


def _integer_scaled_delta(epsilon: float, delta_per_epsilon: int, *, atom_id: str) -> int:
    scaled = float(epsilon) * int(delta_per_epsilon)
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-9:
        raise ComponentResponsePredictionError(
            f"epsilon {epsilon!r} times atom {atom_id!r} delta_per_epsilon "
            f"{delta_per_epsilon} is not an integer byte delta"
        )
    return int(rounded)


def build_component_response_prediction_deltas(
    *,
    baseline_archive: Path,
    perturbation_basis_json: Path,
    output_json: Path,
    component_maps: Mapping[str, Path],
    epsilons: list[float],
) -> dict[str, Any]:
    baseline_archive = baseline_archive.resolve()
    perturbation_basis_json = perturbation_basis_json.resolve()
    output_json = output_json.resolve()
    if not baseline_archive.is_file():
        raise ComponentResponsePredictionError(f"baseline archive not found: {baseline_archive}")
    if not epsilons:
        raise ComponentResponsePredictionError("epsilon ladder must not be empty")
    epsilon_values = sorted({_finite_float(eps, field="epsilon") for eps in epsilons})
    if not any(abs(eps) > SCORE_EPS for eps in epsilon_values):
        raise ComponentResponsePredictionError("epsilon ladder needs a nonzero point")
    if not any(abs(eps) <= SCORE_EPS for eps in epsilon_values):
        epsilon_values.insert(0, 0.0)

    atoms = _load_basis_atoms(perturbation_basis_json)
    maps, map_metadata = _load_component_maps(component_maps)
    _validate_archive_byte_prediction_contract(map_metadata)
    atom_set_sha = _atom_set_sha256(atoms)

    atom_component_sensitivities: list[dict[str, Any]] = []
    for atom in atoms:
        sensitivities = {
            component: _atom_sensitivity(maps, component=component, atom=atom)
            for component in COMPONENTS
        }
        atom_component_sensitivities.append({**atom, "sensitivities": sensitivities})

    points: list[dict[str, Any]] = []
    for epsilon in epsilon_values:
        raw_abs: dict[str, float] = {component: 0.0 for component in COMPONENTS}
        contributions: list[dict[str, Any]] = []
        for atom in atom_component_sensitivities:
            delta = _integer_scaled_delta(
                epsilon,
                int(atom["delta_per_epsilon"]),
                atom_id=str(atom["atom_id"]),
            )
            delta_sq = float(delta * delta)
            atom_contribution = {
                "atom_id": atom["atom_id"],
                "integer_byte_delta": int(delta),
                "component_abs_delta": {},
            }
            for component in COMPONENTS:
                value = delta_sq * float(atom["sensitivities"][component])
                raw_abs[component] += value
                atom_contribution["component_abs_delta"][component] = value
            contributions.append(atom_contribution)
        predicted = {
            component: 0.0 if abs(epsilon) <= SCORE_EPS else float(raw_abs[component])
            for component in COMPONENTS
        }
        points.append(
            {
                "epsilon": float(epsilon),
                "predicted_delta": predicted,
                "component_raw_abs_delta": dict(predicted),
                "atom_contributions": contributions,
            }
        )

    payload = {
        "schema_version": 1,
        "format": PREDICTION_DELTAS_FORMAT,
        "producer": PRODUCER,
        "prediction_source": {
            "source_kind": "component_sensitivity_map_projection",
            "baseline_archive": _file_meta(baseline_archive),
            "perturbation_basis": {
                **_file_meta(perturbation_basis_json),
                "atom_set_sha256": atom_set_sha,
            },
            "component_maps": map_metadata,
        },
        "prediction_model": {
            "model": "quadratic_archive_byte_channel_sensitivity_projection_v1",
            "prediction_delta_semantics": "nonnegative_component_delta_magnitude",
            "prediction_error_mode": "absolute_magnitude",
            "equation": (
                "predicted_delta[component] = sum_atoms "
                "round(epsilon * delta_per_epsilon)^2 * "
                "component_sensitivity[layer_name.weight][channel_index]"
            ),
            "sign_policy": (
                "nonnegative component-delta magnitude from nonnegative sensitivity maps; "
                "signed improvement/degradation claims require a future signed-map artifact"
            ),
            "uses_official_response_observations": False,
        },
        "epsilon_ladder": [float(eps) for eps in epsilon_values],
        "atom_set_sha256": atom_set_sha,
        "points": points,
    }
    _reject_observed_response_leakage(payload)
    _write_json(output_json, payload)
    return {
        "format": "official_component_response_prediction_deltas_summary_v1",
        "prediction_deltas": _file_meta(output_json),
        "baseline_archive": _file_meta(baseline_archive),
        "perturbation_basis": _file_meta(perturbation_basis_json),
        "atom_set_sha256": atom_set_sha,
        "point_count": len(points),
        "epsilon_ladder": [float(eps) for eps in epsilon_values],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--perturbation-basis-json", type=Path, required=True)
    parser.add_argument("--posenet-map", type=Path, required=True)
    parser.add_argument("--segnet-map", type=Path, required=True)
    parser.add_argument("--combined-map", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--epsilon", action="append", type=float, default=None)
    args = parser.parse_args(argv)
    epsilons = args.epsilon if args.epsilon is not None else list(DEFAULT_EPSILONS)
    for index, epsilon in enumerate(epsilons):
        if not math.isfinite(float(epsilon)):
            parser.error(f"--epsilon value {index} must be finite")
    args.epsilons = [float(eps) for eps in epsilons]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_component_response_prediction_deltas(
            baseline_archive=args.baseline_archive,
            perturbation_basis_json=args.perturbation_basis_json,
            output_json=args.output_json,
            component_maps={
                "posenet": args.posenet_map,
                "segnet": args.segnet_map,
                "combined": args.combined_map,
            },
            epsilons=args.epsilons,
        )
    except ComponentResponsePredictionError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
