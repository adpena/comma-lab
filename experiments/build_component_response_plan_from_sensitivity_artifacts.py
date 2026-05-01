#!/usr/bin/env python3
"""Build an official component-response plan from CUDA sensitivity artifacts.

This is a deterministic glue tool for the post-harvest path:

1. Validate a harvested Lightning diagnostic component-sensitivity artifact
   directory.
2. Project the CUDA sensitivity maps onto the perturbation basis before any
   official response evaluation is run.
3. Build deterministic archive perturbation variants and a response plan with
   structured pre-response predictions attached.

The output is still not score evidence. It prepares the next official CUDA
``archive.zip -> inflate.sh -> upstream/evaluate.py`` response job.
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
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT / "src", EXPERIMENTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_component_response_perturbation_plan import (  # noqa: E402
    _load_basis_atoms,
    build_component_response_perturbation_plan,
)
from build_component_response_prediction_deltas import (  # noqa: E402
    build_component_response_prediction_deltas,
)
from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    validate_local_component_sensitivity_artifact_dir,
)


PRODUCER = "experiments/build_component_response_plan_from_sensitivity_artifacts.py"
SUMMARY_FORMAT = "official_component_response_plan_from_sensitivity_artifacts_summary_v1"
DEFAULT_OFFICIAL_EPSILONS = (-2.0, -1.0, 0.0, 1.0, 2.0)
COMPONENTS = ("posenet", "segnet", "combined")


class ComponentResponsePlanFromSensitivityError(ValueError):
    """Raised when harvested sensitivity artifacts cannot build a safe plan."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ComponentResponsePlanFromSensitivityError(f"required file not found: {path}")
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


def _finite_epsilons(values: list[float]) -> list[float]:
    if not values:
        raise ComponentResponsePlanFromSensitivityError("epsilon ladder must not be empty")
    out = sorted({float(value) for value in values})
    for index, value in enumerate(out):
        if not math.isfinite(value):
            raise ComponentResponsePlanFromSensitivityError(
                f"epsilon value {index} must be finite"
            )
    if not any(abs(value) > 1e-12 for value in out):
        raise ComponentResponsePlanFromSensitivityError(
            "epsilon ladder needs at least one nonzero point"
        )
    if not any(abs(value) <= 1e-12 for value in out):
        out.insert(0, 0.0)
        out = sorted(set(out))
    return out


def _required_artifact_file(artifact_dir: Path, name: str) -> Path:
    path = (artifact_dir / name).resolve()
    if not path.is_file():
        raise ComponentResponsePlanFromSensitivityError(
            f"sensitivity artifact missing required file: {path}"
        )
    return path


def build_component_response_plan_from_sensitivity_artifacts(
    *,
    sensitivity_artifact_dir: Path,
    baseline_archive: Path,
    baseline_contest_auth_eval_json: Path,
    perturbation_basis_json: Path | None = None,
    output_dir: Path,
    epsilons: list[float],
    max_mutated_bytes: int = 1024,
    max_abs_byte_delta: int = 32,
    max_raw_l1_delta: int = 4096,
    max_archive_bytes: int | None = None,
    max_archive_byte_delta: int | None = None,
) -> dict[str, Any]:
    """Build prediction deltas and official response plan from harvested maps."""

    sensitivity_artifact_dir = sensitivity_artifact_dir.resolve()
    baseline_archive = baseline_archive.resolve()
    baseline_contest_auth_eval_json = baseline_contest_auth_eval_json.resolve()
    output_dir = output_dir.resolve()
    epsilon_values = _finite_epsilons(epsilons)
    baseline_meta = _file_meta(baseline_archive)
    baseline_eval_meta = _file_meta(baseline_contest_auth_eval_json)

    validation = validate_local_component_sensitivity_artifact_dir(
        sensitivity_artifact_dir,
        expected_baseline_archive_sha256=baseline_meta["sha256"],
        expected_baseline_archive_size_bytes=baseline_meta["bytes"],
    )
    if validation.get("device") != "cuda":
        raise ComponentResponsePlanFromSensitivityError(
            f"sensitivity artifact device={validation.get('device')!r}, expected 'cuda'"
        )
    if validation.get("promotion_eligible") is not False:
        raise ComponentResponsePlanFromSensitivityError(
            "diagnostic sensitivity artifact must remain promotion_eligible=false"
        )

    if perturbation_basis_json is None:
        basis_json = _required_artifact_file(sensitivity_artifact_dir, "perturbation_basis_v1.json")
        basis_source = "sensitivity_artifact_dir"
    else:
        basis_json = perturbation_basis_json.resolve()
        _file_meta(basis_json)
        basis_source = "explicit_perturbation_basis_json"
    map_paths = {
        component: _required_artifact_file(
            sensitivity_artifact_dir,
            f"{component}_sensitivity_map.pt",
        )
        for component in COMPONENTS
    }

    prediction_json = output_dir / "official_component_response_prediction_deltas.json"
    prediction_summary = build_component_response_prediction_deltas(
        baseline_archive=baseline_archive,
        perturbation_basis_json=basis_json,
        output_json=prediction_json,
        component_maps=map_paths,
        epsilons=epsilon_values,
    )

    plan_dir = output_dir / "official_response_plan"
    plan_summary = build_component_response_perturbation_plan(
        baseline_archive=baseline_archive,
        output_dir=plan_dir,
        atoms=_load_basis_atoms(basis_json),
        epsilons=epsilon_values,
        plan_output=plan_dir / "official_component_response_plan.json",
        basis_output=plan_dir / "perturbation_basis_v1.json",
        variants_manifest_output=plan_dir / "archive_variants_manifest.json",
        baseline_contest_auth_eval_json=baseline_contest_auth_eval_json,
        predicted_deltas_json=prediction_json,
        require_predicted_deltas=True,
        max_mutated_bytes=max_mutated_bytes,
        max_abs_byte_delta=max_abs_byte_delta,
        max_raw_l1_delta=max_raw_l1_delta,
        max_archive_bytes=max_archive_bytes,
        max_archive_byte_delta=max_archive_byte_delta,
    )

    summary_path = output_dir / "component_response_plan_from_sensitivity_artifacts_summary.json"
    summary = {
        "schema_version": 1,
        "format": SUMMARY_FORMAT,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "planning:pre_response_prediction_and_archive_variants",
        "baseline_archive": baseline_meta,
        "baseline_contest_auth_eval_json": baseline_eval_meta,
        "sensitivity_artifact": {
            "path": str(sensitivity_artifact_dir),
            "device": validation.get("device"),
            "gpu_model": validation.get("gpu_model"),
            "baseline_archive_sha256": validation.get("baseline_archive_sha256"),
            "baseline_archive_size_bytes": validation.get("baseline_archive_size_bytes"),
            "promotion_eligible": validation.get("promotion_eligible"),
            "score_claim": validation.get("score_claim"),
            "score_source": validation.get("score_source"),
        },
        "epsilon_ladder": [float(value) for value in epsilon_values],
        "component_maps": {component: _file_meta(path) for component, path in map_paths.items()},
        "perturbation_basis": {
            **_file_meta(basis_json),
            "basis_source": basis_source,
        },
        "prediction_deltas": prediction_summary["prediction_deltas"],
        "official_response_plan": plan_summary["plan"],
        "archive_variants_manifest": plan_summary["archive_variants_manifest"],
        "official_response_next_step": {
            "script": "experiments/profile_component_sensitivity_official.py",
            "requires_device": "cuda",
            "requires_same_run_zero": True,
            "requires_require_passed": True,
        },
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-artifact-dir", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, required=True)
    parser.add_argument(
        "--perturbation-basis-json",
        type=Path,
        default=None,
        help=(
            "Optional fresh perturbation_basis_v1 JSON. If omitted, uses "
            "<sensitivity-artifact-dir>/perturbation_basis_v1.json."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", action="append", type=float, default=None)
    parser.add_argument("--max-mutated-bytes", type=int, default=1024)
    parser.add_argument("--max-abs-byte-delta", type=int, default=32)
    parser.add_argument("--max-raw-l1-delta", type=int, default=4096)
    parser.add_argument("--max-archive-bytes", type=int, default=None)
    parser.add_argument("--max-archive-byte-delta", type=int, default=None)
    args = parser.parse_args(argv)
    epsilons = args.epsilon if args.epsilon is not None else list(DEFAULT_OFFICIAL_EPSILONS)
    args.epsilons = _finite_epsilons([float(value) for value in epsilons])
    for field in ("max_mutated_bytes", "max_abs_byte_delta", "max_raw_l1_delta"):
        if getattr(args, field) < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    for field in ("max_archive_bytes", "max_archive_byte_delta"):
        value = getattr(args, field)
        if value is not None and value < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_component_response_plan_from_sensitivity_artifacts(
            sensitivity_artifact_dir=args.sensitivity_artifact_dir,
            baseline_archive=args.baseline_archive,
            baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
            perturbation_basis_json=args.perturbation_basis_json,
            output_dir=args.output_dir,
            epsilons=args.epsilons,
            max_mutated_bytes=args.max_mutated_bytes,
            max_abs_byte_delta=args.max_abs_byte_delta,
            max_raw_l1_delta=args.max_raw_l1_delta,
            max_archive_bytes=args.max_archive_bytes,
            max_archive_byte_delta=args.max_archive_byte_delta,
        )
    except (ComponentResponsePlanFromSensitivityError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
