#!/usr/bin/env python3
"""Certify CUDA component-sensitivity maps after official response gates.

This tool deliberately does not edit source maps. It copies tensor values into
new ``tac_score_sensitivity_map_v1`` artifacts whose metadata cites exact CUDA
archive-response gates, sample coverage, stability, baseline custody, and
review status. Fisher/proxy/debug maps remain non-promotable and are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.sensitivity_map import (  # noqa: E402
    CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT,
    SENSITIVITY_MAP_FORMAT,
    SensitivityMapError,
    load_sensitivity_map,
    save_certified_sensitivity_map,
)


COMPONENTS = ("posenet", "segnet", "combined")
CONTEST_SAMPLE_COUNT = 600
EXPECTED_SOURCE = "direct_renderer_cuda_finite_difference_component_response"
OFFICIAL_RESPONSE_FORMAT = "official_component_response_curves_v1"
SUMMARY_FORMAT = "component_sensitivity_map_certification_summary_v1"


class CertificationError(ValueError):
    """Raised when maps cannot be certified for promotion."""


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


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CertificationError(f"{path}: invalid JSON") from exc


def _finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CertificationError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise CertificationError(f"{field} must be finite")
    if minimum is not None and out < minimum:
        raise CertificationError(f"{field} must be >= {minimum}")
    return out


def _load_baseline_eval(path: Path, *, archive: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CertificationError(f"{path}: baseline eval must be a JSON object")
    if payload.get("n_samples") != CONTEST_SAMPLE_COUNT:
        raise CertificationError(f"{path}: n_samples must be {CONTEST_SAMPLE_COUNT}")
    provenance = payload.get("provenance")
    device = provenance.get("device") if isinstance(provenance, dict) else None
    if device != "cuda":
        raise CertificationError(f"{path}: provenance.device must be 'cuda'")
    archive_meta = _file_meta(archive)
    payload_sha = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    payload_sha = payload_sha or payload.get("archive_sha256")
    if payload_sha != archive_meta["sha256"]:
        raise CertificationError(f"{path}: archive sha256 does not match {archive}")
    if payload.get("archive_size_bytes") != archive_meta["bytes"]:
        raise CertificationError(f"{path}: archive size does not match {archive}")
    pose = _finite(payload.get("avg_posenet_dist"), field=f"{path}.avg_posenet_dist", minimum=0.0)
    seg = _finite(payload.get("avg_segnet_dist"), field=f"{path}.avg_segnet_dist", minimum=0.0)
    recomputed = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_meta["bytes"] / 37_545_489
    recorded = payload.get("score_recomputed_from_components", payload.get("final_score"))
    if recorded is not None and abs(float(recorded) - recomputed) > 1e-9 * max(1.0, abs(recomputed)):
        raise CertificationError(f"{path}: score does not recompute from components and archive bytes")
    return payload


def _assert_full_sample_plan(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CertificationError(f"{path}: sample plan must be a JSON object")
    pair_ids: list[int] = []
    for split_name in ("calibration_pairs", "holdout_pairs"):
        pairs = payload.get(split_name)
        if not isinstance(pairs, list) or not pairs:
            raise CertificationError(f"{path}: {split_name} must be a non-empty list")
        for index, item in enumerate(pairs):
            if not isinstance(item, dict):
                raise CertificationError(f"{path}: {split_name}[{index}] must be an object")
            pair_index = item.get("pair_index")
            if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                raise CertificationError(f"{path}: {split_name}[{index}].pair_index must be an integer")
            if item.get("t", 2 * pair_index) != 2 * pair_index:
                raise CertificationError(f"{path}: {split_name}[{index}].t is not an absolute contest pair id")
            if item.get("t1", 2 * pair_index + 1) != 2 * pair_index + 1:
                raise CertificationError(f"{path}: {split_name}[{index}].t1 is not an absolute contest pair id")
            pair_ids.append(pair_index)
    if sorted(pair_ids) != list(range(CONTEST_SAMPLE_COUNT)):
        raise CertificationError(f"{path}: sample plan must cover absolute pair ids 0..599 exactly once")
    return payload


def _max_numeric(value: Any) -> float | None:
    values: list[float] = []
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, Mapping):
        for child in value.values():
            found = _max_numeric(child)
            if found is not None:
                values.append(found)
    if isinstance(value, list):
        for child in value:
            found = _max_numeric(child)
            if found is not None:
                values.append(found)
    return max(values) if values else None


def _min_numeric(value: Any) -> float | None:
    values: list[float] = []
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, Mapping):
        for child in value.values():
            found = _min_numeric(child)
            if found is not None:
                values.append(found)
    if isinstance(value, list):
        for child in value:
            found = _min_numeric(child)
            if found is not None:
                values.append(found)
    return min(values) if values else None


def _stability_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CertificationError(f"{path}: stability must be a JSON object")
    if payload.get("passed") is not True:
        raise CertificationError(f"{path}: stability.passed must be true")
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    cv_limit = float(thresholds.get("cv_max", 0.35))
    spearman_min = float(thresholds.get("spearman_min", 0.30))
    top_overlap_min = float(thresholds.get("top_decile_overlap_min", 0.50))
    cv_observed = _max_numeric(payload.get("cv"))
    rank_observed = _min_numeric(payload.get("rank"))
    top_observed = _min_numeric(payload.get("top_k"))
    if cv_observed is None or cv_observed > cv_limit:
        raise CertificationError(f"{path}: stability cv gate failed")
    if rank_observed is None or rank_observed < spearman_min:
        raise CertificationError(f"{path}: stability rank/spearman gate failed")
    if top_observed is None or top_observed < top_overlap_min:
        raise CertificationError(f"{path}: stability top-k overlap gate failed")
    return {
        "passed": True,
        "cv_max": cv_observed,
        "spearman_min": rank_observed,
        "top_decile_overlap_min": top_observed,
    }


def _required_json_file(path: Path, *, label: str, expected_format: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise CertificationError(f"missing {label}: {path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CertificationError(f"{path}: {label} must be a JSON object")
    if payload.get("format") != expected_format:
        raise CertificationError(f"{path}: {label} format must be {expected_format!r}")
    return payload, _file_meta(path)


def _prediction_deltas_gate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = _required_json_file(
        path,
        label="prediction deltas",
        expected_format="official_component_response_prediction_deltas_v1",
    )
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise CertificationError(f"{path}: prediction deltas points must be non-empty")
    atom_ids: set[str] = set()
    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            raise CertificationError(f"{path}: points[{point_index}] must be an object")
        contributions = point.get("atom_contributions")
        if not isinstance(contributions, list) or not contributions:
            raise CertificationError(f"{path}: points[{point_index}].atom_contributions must be non-empty")
        for contribution_index, contribution in enumerate(contributions):
            if not isinstance(contribution, dict):
                raise CertificationError(
                    f"{path}: points[{point_index}].atom_contributions[{contribution_index}] must be an object"
                )
            atom_id = contribution.get("atom_id")
            if not isinstance(atom_id, str) or not atom_id:
                raise CertificationError(f"{path}: prediction contribution missing atom_id")
            atom_ids.add(atom_id)
    gate = {
        "format": payload["format"],
        "sha256": meta["sha256"],
        "bytes": meta["bytes"],
        "atom_ids": sorted(atom_ids),
        "epsilon_ladder": payload.get("epsilon_ladder"),
        "atom_set_sha256": payload.get("atom_set_sha256"),
    }
    return gate, meta


def _perturbation_basis_gate(
    path: Path,
    *,
    baseline_archive_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, meta = _required_json_file(
        path,
        label="perturbation basis",
        expected_format="perturbation_basis_v1",
    )
    if payload.get("basis_kind") != "archive_byte_additive":
        raise CertificationError(f"{path}: perturbation basis must be archive_byte_additive")
    if payload.get("canonical_response_eval_path") != "archive.zip -> inflate.sh -> upstream/evaluate.py":
        raise CertificationError(f"{path}: perturbation basis has noncanonical response eval path")
    source_archive = payload.get("source_archive")
    if isinstance(source_archive, dict):
        if source_archive.get("sha256") != baseline_archive_meta["sha256"]:
            raise CertificationError(f"{path}: basis source archive sha256 does not match baseline")
        if source_archive.get("bytes") != baseline_archive_meta["bytes"]:
            raise CertificationError(f"{path}: basis source archive bytes do not match baseline")
    atoms = payload.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise CertificationError(f"{path}: perturbation basis atoms must be non-empty")
    atom_ids: set[str] = set()
    for index, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            raise CertificationError(f"{path}: atoms[{index}] must be an object")
        for key in ("atom_id", "member", "offset", "delta_per_epsilon"):
            if key not in atom:
                raise CertificationError(f"{path}: atoms[{index}] missing {key}")
        if atom["member"] != "renderer.bin":
            raise CertificationError(f"{path}: atoms[{index}] member must be renderer.bin for current certifier")
        if isinstance(atom["offset"], bool) or not isinstance(atom["offset"], int) or atom["offset"] < 0:
            raise CertificationError(f"{path}: atoms[{index}].offset must be a nonnegative integer")
        atom_ids.add(str(atom["atom_id"]))
    gate = {
        "format": payload["format"],
        "sha256": meta["sha256"],
        "bytes": meta["bytes"],
        "atom_ids": sorted(atom_ids),
        "epsilon_ladder": payload.get("epsilon_ladder"),
        "basis_id": payload.get("basis_id"),
    }
    return gate, meta


def _cross_check_prediction_basis(
    *,
    prediction_gate: Mapping[str, Any],
    basis_gate: Mapping[str, Any],
) -> None:
    predicted_atoms = set(prediction_gate.get("atom_ids") or [])
    basis_atoms = set(basis_gate.get("atom_ids") or [])
    missing = sorted(predicted_atoms.difference(basis_atoms))
    if missing:
        raise CertificationError(
            "prediction deltas cite atoms absent from perturbation basis: "
            + ", ".join(missing[:10])
        )
    if prediction_gate.get("epsilon_ladder") != basis_gate.get("epsilon_ladder"):
        raise CertificationError("prediction deltas epsilon_ladder does not match perturbation basis")


def _curve_gate(
    path: Path,
    *,
    component: str,
    external_baseline_required: bool,
    prediction_deltas_sha256: str,
    perturbation_basis_sha256: str,
) -> dict[str, Any]:
    curve = _load_json(path)
    if not isinstance(curve, dict):
        raise CertificationError(f"{path}: response curve must be a JSON object")
    if curve.get("format") != OFFICIAL_RESPONSE_FORMAT:
        raise CertificationError(f"{path}: response curve format must be {OFFICIAL_RESPONSE_FORMAT!r}")
    if curve.get("component") != component:
        raise CertificationError(f"{path}: component mismatch")
    if curve.get("official_component_response") is not True:
        raise CertificationError(f"{path}: official_component_response must be true")
    if curve.get("canonical_scorer_path") is not True:
        raise CertificationError(f"{path}: canonical_scorer_path must be true")
    if curve.get("component_response_path") != "archive_zip_inflate_sh_upstream_evaluate_py":
        raise CertificationError(f"{path}: noncanonical component response path")
    if curve.get("passed") is not True:
        raise CertificationError(f"{path}: response curve did not pass")
    if curve.get("promotion_blockers") not in ([], None):
        raise CertificationError(f"{path}: response curve has promotion blockers")
    gate = curve.get("gate_results")
    if not isinstance(gate, dict):
        raise CertificationError(f"{path}: gate_results is required")
    for key in (
        "finite_values",
        "coverage_passed",
        "zero_repro",
        "signal_present",
        "prediction_error_passed",
        "promotion_gate_passed",
    ):
        if gate.get(key) is not True:
            raise CertificationError(f"{path}: gate_results.{key} must be true")
    if external_baseline_required and gate.get("external_baseline_repro") is not True:
        raise CertificationError(f"{path}: gate_results.external_baseline_repro must be true")
    if "external_baseline_repro" in gate and gate.get("external_baseline_repro") is not True:
        raise CertificationError(f"{path}: gate_results.external_baseline_repro must be true")
    perturbation = curve.get("perturbation")
    if not isinstance(perturbation, dict):
        raise CertificationError(f"{path}: perturbation custody is required")
    predicted_source = perturbation.get("predicted_deltas_source")
    if not isinstance(predicted_source, dict) or predicted_source.get("sha256") != prediction_deltas_sha256:
        raise CertificationError(f"{path}: perturbation predicted_deltas_source sha256 mismatch")
    if perturbation.get("basis_sha256") != perturbation_basis_sha256:
        raise CertificationError(f"{path}: perturbation basis_sha256 mismatch")
    zero_error = _finite(gate.get("zero_repro_error"), field=f"{path}.zero_repro_error", minimum=0.0)
    observed = _finite(gate.get("observed_delta_max"), field=f"{path}.observed_delta_max", minimum=0.0)
    pred_error = _finite(
        gate.get("max_relative_prediction_error"),
        field=f"{path}.max_relative_prediction_error",
        minimum=0.0,
    )
    if zero_error > 1e-7:
        raise CertificationError(f"{path}: zero reproduction error exceeds 1e-7")
    if observed < 1e-12:
        raise CertificationError(f"{path}: observed response signal is below 1e-12")
    if pred_error > 0.35:
        raise CertificationError(f"{path}: prediction error exceeds 0.35")
    return {
        "finite_values": True,
        "coverage_passed": True,
        "zero_repro": True,
        "zero_repro_error": zero_error,
        "signal_present": True,
        "observed_delta_max": observed,
        "prediction_error_passed": True,
        "max_relative_prediction_error": pred_error,
        "promotion_gate_passed": True,
    }


def _review_packet(path: Path | None, *, clean_passes: int) -> tuple[int, list[Any], str | None]:
    if path is None:
        return clean_passes, [], None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise CertificationError(f"{path}: review packet must be a JSON object")
    passes = payload.get("clean_passes", payload.get("grand_council_clean_passes", clean_passes))
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < clean_passes:
        raise CertificationError(f"{path}: review clean passes must be >= {clean_passes}")
    blockers = payload.get("unresolved_blockers", payload.get("blockers", []))
    if blockers not in ([], None):
        raise CertificationError(f"{path}: review packet has unresolved blockers")
    return int(passes), [], _sha256_file(path)


def _source_map(path: Path, *, component: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != SENSITIVITY_MAP_FORMAT:
        raise CertificationError(f"{path}: source map must use {SENSITIVITY_MAP_FORMAT!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CertificationError(f"{path}: source map metadata is required")
    if metadata.get("device") != "cuda":
        raise CertificationError(f"{path}: source map must be CUDA-authored")
    if metadata.get("sensitivity_source") != EXPECTED_SOURCE:
        raise CertificationError(
            f"{path}: sensitivity_source must be {EXPECTED_SOURCE!r}; "
            "Fisher/proxy maps cannot be certified"
        )
    if metadata.get("component") != component and metadata.get("scorer_target") != component:
        raise CertificationError(f"{path}: source map component does not match {component}")
    try:
        tensors, _loaded_meta = load_sensitivity_map(path)
    except SensitivityMapError as exc:
        raise CertificationError(f"{path}: invalid sensitivity tensors: {exc}") from exc
    if not tensors:
        raise CertificationError(f"{path}: source map has no tensors")
    return tensors, metadata


def certify_maps(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = Path(args.candidate_artifact_dir)
    response_dir = Path(args.official_response_artifact_dir)
    output_dir = Path(args.output_dir)
    baseline_archive = Path(args.baseline_archive)
    baseline_eval = Path(args.baseline_contest_auth_eval_json)
    for path in (candidate_dir, response_dir):
        if not path.exists() or not path.is_dir():
            raise CertificationError(f"directory not found: {path}")
    for path in (baseline_archive, baseline_eval):
        if not path.exists() or not path.is_file():
            raise CertificationError(f"file not found: {path}")

    baseline_payload = _load_baseline_eval(baseline_eval, archive=baseline_archive)
    baseline_archive_meta = _file_meta(baseline_archive)
    baseline_eval_meta = _file_meta(baseline_eval)
    prediction_gate, prediction_deltas_meta = _prediction_deltas_gate(args.prediction_deltas_json)
    basis_gate, perturbation_basis_meta = _perturbation_basis_gate(
        args.perturbation_basis_json,
        baseline_archive_meta=baseline_archive_meta,
    )
    _cross_check_prediction_basis(
        prediction_gate=prediction_gate,
        basis_gate=basis_gate,
    )
    sample_plan = candidate_dir / "sample_plan.json"
    stability = candidate_dir / "stability.json"
    _assert_full_sample_plan(sample_plan)
    stability_gate = _stability_gate(stability)
    response_summary = response_dir / "official_component_response_summary.json"
    if not response_summary.exists():
        raise CertificationError(f"missing official response summary: {response_summary}")
    response_summary_meta = _file_meta(response_summary)
    response_summary_payload = _load_json(response_summary)
    if not isinstance(response_summary_payload, dict) or response_summary_payload.get("promotion_eligible") is not True:
        raise CertificationError(f"{response_summary}: official response summary must be promotion_eligible=true")
    external_baseline_required = response_summary_payload.get("external_baseline_contest_auth_eval_json") is not None

    review_passes, review_blockers, review_sha = _review_packet(
        args.review_packet_json,
        clean_passes=args.review_clean_passes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    certified: dict[str, Any] = {}
    for component in COMPONENTS:
        source_path = candidate_dir / f"{component}_sensitivity_map.pt"
        curve_path = response_dir / f"{component}_official_response_curve.json"
        if not source_path.exists():
            raise CertificationError(f"missing source map: {source_path}")
        if not curve_path.exists():
            raise CertificationError(f"missing official curve: {curve_path}")
        tensors, _source_metadata = _source_map(source_path, component=component)
        response_gate = _curve_gate(
            curve_path,
            component=component,
            external_baseline_required=external_baseline_required,
            prediction_deltas_sha256=prediction_deltas_meta["sha256"],
            perturbation_basis_sha256=perturbation_basis_meta["sha256"],
        )
        certification: dict[str, Any] = {
            "format": CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT,
            "component": component,
            "device": "cuda",
            "official_component_response": True,
            "canonical_scorer_path": True,
            "promotion_eligible": True,
            "source_map_sha256": _sha256_file(source_path),
            "official_response_curve_sha256": _sha256_file(curve_path),
            "stability_sha256": _sha256_file(stability),
            "sample_plan_sha256": _sha256_file(sample_plan),
            "baseline_archive_sha256": baseline_archive_meta["sha256"],
            "baseline_archive_bytes": baseline_archive_meta["bytes"],
            "contest_auth_eval_json_sha256": baseline_eval_meta["sha256"],
            "official_response_summary_sha256": response_summary_meta["sha256"],
            "review_clean_passes": review_passes,
            "review_unresolved_blockers": review_blockers,
            "response_gate_results": response_gate,
            "stability_gate_results": stability_gate,
            "prediction_deltas_sha256": prediction_deltas_meta["sha256"],
            "perturbation_basis_sha256": perturbation_basis_meta["sha256"],
        }
        if review_sha is not None:
            certification["review_packet_sha256"] = review_sha
        out_path = output_dir / f"{component}_certified_sensitivity_map.pt"
        save_certified_sensitivity_map(
            out_path,
            tensors,
            component=component,
            certification=certification,
        )
        certified[component] = {
            "path": str(out_path),
            "bytes": out_path.stat().st_size,
            "sha256": _sha256_file(out_path),
            "source_map_sha256": certification["source_map_sha256"],
            "official_response_curve_sha256": certification["official_response_curve_sha256"],
        }

    summary = {
        "format": SUMMARY_FORMAT,
        "schema_version": 1,
        "promotion_eligible": True,
        "score_claim": False,
        "baseline_archive": baseline_archive_meta,
        "baseline_contest_auth_eval_json": baseline_eval_meta,
        "baseline_score_recomputed_from_components": baseline_payload.get("score_recomputed_from_components"),
        "candidate_artifact_dir": str(candidate_dir),
        "sample_plan": _file_meta(sample_plan),
        "stability": _file_meta(stability),
        "official_response_summary": response_summary_meta,
        "prediction_deltas": prediction_deltas_meta,
        "perturbation_basis": perturbation_basis_meta,
        "certified_maps": certified,
        "review_clean_passes": review_passes,
    }
    summary_path = output_dir / "component_sensitivity_map_certification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--official-response-artifact-dir", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-packet-json", type=Path, default=None)
    parser.add_argument("--review-clean-passes", type=int, default=3)
    parser.add_argument("--prediction-deltas-json", type=Path, required=True)
    parser.add_argument("--perturbation-basis-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = certify_maps(args)
    except CertificationError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
