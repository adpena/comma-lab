#!/usr/bin/env python3
"""Build a deterministic contest problem-space manifest.

The output is a machine-readable coordinate system for optimization, not score
evidence. It centralizes contest constants, exact frontier custody, public
archive anatomy, action-functional terms, and compliance constraints so later
optimizers do not rely on scattered prose or stale chat state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TOOL = "experiments/build_problem_space_manifest.py"
SCHEMA = "contest_problem_space_manifest_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
PAIR_COUNT = 600
FRAME_COUNT = 1200
CAMERA_WIDTH = 1164
CAMERA_HEIGHT = 874
SEGNET_WIDTH = 512
SEGNET_HEIGHT = 384
MASK_CLASS_COUNT = 5
RATE_LAMBDA = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_FRONTIER_JSON = (
    Path("experiments/results/lightning_batch")
    / "exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z"
    / "contest_auth_eval.adjudicated.json"
)
DEFAULT_PR85_PROFILE = (
    Path("experiments/results/public_pr85_intake_20260503_codex")
    / "pr85_adaptive_masking_sidechannel_attribution_plan.json"
)
DEFAULT_PR86_PROFILE = (
    Path("experiments/results/public_pr86_intake_20260504_codex")
    / "profile_pr86_archive_static.json"
)
DEFAULT_OUTPUT = (
    Path("experiments/results/problem_space_manifest_20260504_codex")
    / "problem_space_manifest.json"
)


class ProblemSpaceManifestError(ValueError):
    """Raised when a manifest input is missing required structured facts."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProblemSpaceManifestError(f"{path} is not valid JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProblemSpaceManifestError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ProblemSpaceManifestError(f"{name} must be finite")
    return out


def score_terms(*, segnet_dist: float, posenet_dist: float, archive_bytes: int) -> dict[str, float]:
    """Return exact contest score terms from structured components."""
    if archive_bytes < 0:
        raise ProblemSpaceManifestError("archive_bytes must be nonnegative")
    seg_term = 100.0 * segnet_dist
    pose_term = math.sqrt(10.0 * posenet_dist)
    rate_term = 25.0 * float(archive_bytes) / float(ORIGINAL_VIDEO_BYTES)
    return {
        "segnet_term": seg_term,
        "posenet_term": pose_term,
        "rate_term": rate_term,
        "score": seg_term + pose_term + rate_term,
    }


def _frontier(frontier_json: Path) -> dict[str, Any]:
    payload = _read_json(frontier_json)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ProblemSpaceManifestError(f"{frontier_json} lacks provenance object")
    archive_bytes = int(payload["archive_size_bytes"])
    segnet = _finite_number(payload["avg_segnet_dist"], name="avg_segnet_dist")
    posenet = _finite_number(payload["avg_posenet_dist"], name="avg_posenet_dist")
    visible_component_terms = score_terms(
        segnet_dist=segnet,
        posenet_dist=posenet,
        archive_bytes=archive_bytes,
    )
    reported_score = _finite_number(payload["score_recomputed_from_components"], name="score")
    canonical_score = _finite_number(
        payload.get("canonical_score", reported_score),
        name="canonical_score",
    )
    promotion_eligible = payload.get("promotion_eligible")
    if promotion_eligible is not None and not isinstance(promotion_eligible, bool):
        raise ProblemSpaceManifestError("promotion_eligible must be bool when present")
    return {
        "label": "pr85_adaptive_masking_joint_frame_model_t4",
        "evidence_grade": "A++",
        "path": str(frontier_json),
        "json_sha256": _sha256_file(frontier_json),
        "archive_sha256": provenance.get("archive_sha256"),
        "archive_bytes": archive_bytes,
        "n_samples": int(payload["n_samples"]),
        "device": provenance.get("device"),
        "gpu_model": provenance.get("gpu_model"),
        "avg_segnet_dist": segnet,
        "avg_posenet_dist": posenet,
        "score": canonical_score,
        "score_source": payload.get(
            "canonical_score_source",
            "score_recomputed_from_components",
        ),
        "visible_component_terms": visible_component_terms,
        "score_from_visible_components": visible_component_terms["score"],
        "visible_component_score_delta_vs_canonical": (
            visible_component_terms["score"] - canonical_score
        ),
        "reported_score_recomputed_from_components": reported_score,
        "score_formula_matches_reported_from_visible_components": (
            abs(visible_component_terms["score"] - reported_score) < 1.0e-12
        ),
        "promotion_eligible": promotion_eligible,
        "promotion_eligible_source": (
            "recorded_in_source_json"
            if promotion_eligible is not None
            else "absent_in_source_json_not_interpreted_as_false"
        ),
    }


def _pr85_fields(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ProblemSpaceManifestError(f"{path} lacks segments list")
    fields = []
    for row in segments:
        if not isinstance(row, dict):
            raise ProblemSpaceManifestError(f"{path} segment rows must be objects")
        raw_bytes = int(row["raw_bytes"])
        fields.append(
            {
                "name": str(row["name"]),
                "coordinate_kind": _coordinate_kind(str(row["name"])),
                "charged_bytes": raw_bytes,
                "rate_term": 25.0 * raw_bytes / ORIGINAL_VIDEO_BYTES,
                "container_codec": row.get("container_codec"),
                "decoded_bytes": row.get("decoded_bytes"),
                "schema_facts": row.get("schema_facts", {}),
                "optimization_role": _optimization_role(str(row["name"])),
            }
        )
    return {
        "label": "public_pr85",
        "evidence_grade": payload.get("evidence_grade"),
        "path": str(path),
        "json_sha256": _sha256_file(path),
        "source_archive": payload.get("source_archive"),
        "bundle": payload.get("bundle"),
        "fields": fields,
    }


def _coordinate_kind(name: str) -> str:
    return {
        "mask": "semantic_mask_manifold",
        "model": "learned_renderer_weight_chart",
        "pose": "low_dimensional_pose_subspace",
        "post": "pairwise_postprocess_code_field",
        "shift": "ego_motion_micro_choice_field",
        "frac": "sparse_fractional_warp_field",
        "frac2": "dense_fractional_warp_field",
        "frac3": "dense_fractional_warp_field",
        "bias": "pairwise_rgb_bias_field",
        "region": "region_conditioned_bias_field",
        "randmulti": "sparse_multiscale_residual_action_field",
    }.get(name, "unknown_field")


def _optimization_role(name: str) -> str:
    return {
        "mask": "dominant rate and SegNet/PoseNet geometry coordinate",
        "model": "renderer capacity and self-compression coordinate",
        "pose": "PoseNet-sensitive low-byte coordinate",
        "post": "small dense per-pair correction coordinate",
        "shift": "small ego-motion correction coordinate",
        "frac": "sparse foveated warp correction coordinate",
        "frac2": "dense foveated warp correction coordinate",
        "frac3": "dense foveated warp correction coordinate",
        "bias": "fixed-length PR85 micro-bias coordinate",
        "region": "fixed-length PR85 region-bias coordinate",
        "randmulti": "largest explicit correction table and water-fill target",
    }.get(name, "unclassified coordinate")


def _pr86_fields(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    members = payload.get("members")
    if not isinstance(members, list):
        raise ProblemSpaceManifestError(f"{path} lacks members list")
    return {
        "label": "public_pr86",
        "evidence_grade": "external_static_profile",
        "path": str(path),
        "json_sha256": _sha256_file(path),
        "archive_bytes": int(payload["archive_bytes"]),
        "archive_sha256": payload["archive_sha256"],
        "public_claim": {
            "avg_posenet_dist": 0.00045701,
            "avg_segnet_dist": 0.00067815,
            "archive_bytes": 207_579,
            "terms": score_terms(
                segnet_dist=0.00067815,
                posenet_dist=0.00045701,
                archive_bytes=207_579,
            ),
            "evidence_grade": "external_report_not_our_score_claim",
        },
        "fields": [
            {
                "name": str(row["name"]),
                "coordinate_kind": _pr86_coordinate_kind(str(row["name"])),
                "charged_bytes": int(row["bytes"]),
                "rate_term": 25.0 * int(row["bytes"]) / ORIGINAL_VIDEO_BYTES,
                "sha256": row.get("sha256"),
                "decoded_bytes": row.get("decoded_bytes"),
                "optimization_role": _pr86_role(str(row["name"])),
            }
            for row in members
        ],
    }


def _pr86_coordinate_kind(name: str) -> str:
    return {
        "master.pt.gz": "master_neural_renderer_weights",
        "slave.pt.gz": "slave_nerv_refinement_weights",
        "hpac.pt.ppmd": "entropy_model_weights",
        "tokens.bin": "arithmetic_coded_token_field",
        "meta.pt": "runtime_metadata",
    }.get(name, "unknown_pr86_member")


def _pr86_role(name: str) -> str:
    return {
        "master.pt.gz": "learned representation coordinate",
        "slave.pt.gz": "learned refinement coordinate",
        "hpac.pt.ppmd": "self-compressed entropy prior",
        "tokens.bin": "dominant PR86 rate field and HPAC transfer target",
        "meta.pt": "small fixed runtime contract",
    }.get(name, "unclassified coordinate")


def _equation_system() -> dict[str, Any]:
    return {
        "action_functional": "S(A)=100*seg_dist(D_A)+sqrt(10*pose_dist(D_A))+25*|A|/37545489",
        "set_theoretic_view": {
            "archive_space": "A = {byte strings accepted by strict zip/archive validator}",
            "runtime_space": "D = {deterministic inflate programs closed over archive bytes and fixed contest code}",
            "field_space": "Phi = M x P x Theta x Z x H x G",
            "feasible_set": "F = A intersect C_compliance intersect C_runtime intersect C_reproducibility",
            "promotion_set": "P++ = F intersect {exact CUDA T4/equivalent evidence, 600 samples, clean custody}",
            "stackability": "two transforms compose only when their read/write coordinate sets are disjoint or an exact stacked archive proves compatibility",
        },
        "measure_theoretic_view": {
            "rate_measure": "mu_rate(atom)=charged_bytes(atom)",
            "segnet_measure": "mu_seg(pair, pixel, class)=argmax-disagreement contribution after scorer preprocessing",
            "posenet_measure": "mu_pose(pair, dim)=squared pose-coordinate error before outer sqrt",
            "empirical_measure": "uniform measure over the 600 official pair samples",
        },
        "number_theoretic_and_coding_view": {
            "integer_lattice": "archive bytes, quantized symbols, row runs, pose deltas, and entropy-code states live on integer lattices",
            "prefix_codes": "variable-length and arithmetic-coded streams must be uniquely decodable under the charged runtime grammar",
            "modular_constraints": "fixed-width and fixed-length PR85 fields impose congruence/length constraints before any shrink is valid",
            "canonicalization": "member order, timestamps, permissions, and codec parameters are part of the integer byte solution",
        },
        "signal_theoretic_view": {
            "source_signal": "1200-frame 1164x874 driving video sampled at 20 fps",
            "scorer_channels": "SegNet observes second-frame semantic content; PoseNet observes two-frame motion/geometry",
            "bases": [
                "temporal prediction and residuals",
                "mask-boundary finite differences",
                "DCT/Fourier/Walsh low-rank modes",
                "wavelet and multiresolution mask atoms",
                "ego-motion/foveation vector fields",
                "learned token and entropy-model latents",
            ],
            "risks": [
                "aliasing or phase errors can be visually small but PoseNet-catastrophic",
                "argmax semantic cliffs make smooth proxy losses non-promotable",
                "generic entropy wins can be no-ops unless decoded signal identity or exact component deltas are recorded",
            ],
            "actionable_use": "choose low-dimensional proposal bases and anti-aliasing guards before exact CUDA archive tests",
        },
        "calculus_and_linear_algebra_view": {
            "differential": "dS = grad_seg dM + grad_pose dP + lambda_rate d|A|",
            "hessian": "H approximates pair/frame/class/stream interactions and non-additive scorer curvature",
            "active_subspace": "low-rank eigenspaces of empirical response identify high-benefit coordinates under byte constraints",
            "projection": "Dykstra/ADMM-style projections enforce rate, runtime, compliance, and reproducibility constraints",
            "series_expansions": [
                "Taylor/Frechet expansions for local scorer response",
                "Fourier/DCT/Walsh expansions for mask and residual fields",
                "Neumann-style iterative correction when operators are contractive inside a trust region",
            ],
            "matrix_objects": [
                "pair x component trace matrix",
                "atom x byte-cost vector",
                "atom x scorer-benefit matrix",
                "stream x stream interaction matrix",
                "basis x frame coefficient matrix",
            ],
        },
        "optimization_variables": [
            "archive byte string A",
            "deterministic inflate program D_A",
            "semantic mask field M",
            "pose field P",
            "renderer weights theta",
            "correction/action fields z",
            "entropy model/hyperprior h",
            "packer grammar g",
        ],
        "constraints": [
            "all score-affecting bits are charged inside archive.zip or fixed contest code",
            "inflate is deterministic and finishes inside contest budget on T4/equivalent",
            "no scorer patching, sidecars, network fetches, or host-local score bits",
            "archive construction is byte-for-byte reproducible",
            "score claims require exact CUDA auth eval on exact archive bytes",
            "Apogee-owned native codecs and parsers prefer Rust over C++ unless public replay custody requires existing source",
        ],
        "local_variation": "delta S = <grad_seg+grad_pose,dphi> + 0.5<dphi,H dphi> + lambda_rate*delta_bytes + higher_order_terms",
        "field_bases": [
            "pair/frame/class atoms",
            "mask boundaries and connected components",
            "ego-motion and foveation coordinates",
            "pose active subspaces",
            "renderer tensor blocks",
            "sparse residual action groups",
            "entropy-code symbols and packer layout bits",
        ],
        "promotion_oracle": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
    }


def _stacking_rules() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "disjoint_coordinate_safe_until_runtime_coupling",
            "set_condition": "write_set(T1) intersect write_set(T2) = empty",
            "required_evidence": "exact stacked archive before score claim",
            "example": "rate-only packer change plus unchanged decoded PR85 fields",
        },
        {
            "rule_id": "same_coordinate_requires_reconciliation_operator",
            "set_condition": "write_set(T1) intersect write_set(T2) != empty",
            "required_evidence": "typed reconciliation manifest plus exact CUDA eval",
            "example": "two mask repair policies both editing QMA9-derived semantic geometry",
        },
        {
            "rule_id": "runtime_change_invalidates_archive_only_comparison",
            "set_condition": "runtime_tree_sha256 changes",
            "required_evidence": "runtime custody manifest and explicit comparison class",
            "example": "PR85 replay runtime with range_mask_codec.cpp included in hash",
        },
        {
            "rule_id": "proxy_signal_is_not_membership_in_promotion_set",
            "set_condition": "evidence_grade not in {A++, A}",  # CUSTODY_VALIDATOR_OK: literal text in rule catalogue, not executable promotion gate
            "required_evidence": "exact CUDA auth eval on exact archive bytes",
            "example": "entropy profile can rank atoms but cannot promote them",
        },
    ]


def build_manifest(
    *,
    frontier_json: Path,
    pr85_profile: Path | None,
    pr86_profile: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "deterministic": True,
        "score_claim": False,
        "dispatch_performed": False,
        "evidence_grade": "planning_only_manifest",
        "constants": {
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_lambda": RATE_LAMBDA,
            "pair_count": PAIR_COUNT,
            "frame_count": FRAME_COUNT,
            "camera_width": CAMERA_WIDTH,
            "camera_height": CAMERA_HEIGHT,
            "segnet_width": SEGNET_WIDTH,
            "segnet_height": SEGNET_HEIGHT,
            "mask_class_count": MASK_CLASS_COUNT,
        },
        "score_formula": {
            "text": "score = 100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
            "terms": ["segnet_term", "posenet_term", "rate_term"],
        },
        "equation_system": _equation_system(),
        "stacking_rules": _stacking_rules(),
        "current_frontier": _frontier(frontier_json),
        "public_basin_profiles": [],
        "planning_constraints": [
            "This manifest is planning-only and cannot rank or promote by itself.",
            "Structured exact CUDA artifacts override all static and external fields.",
            "Any future dispatch still requires an active lane claim.",
        ],
    }
    if pr85_profile is not None:
        payload["public_basin_profiles"].append(_pr85_fields(pr85_profile))
    if pr86_profile is not None:
        payload["public_basin_profiles"].append(_pr86_fields(pr86_profile))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-json", type=Path, default=DEFAULT_FRONTIER_JSON)
    parser.add_argument("--pr85-profile", type=Path, default=DEFAULT_PR85_PROFILE)
    parser.add_argument("--pr86-profile", type=Path, default=DEFAULT_PR86_PROFILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    pr85_profile = args.pr85_profile if args.pr85_profile.is_file() else None
    pr86_profile = args.pr86_profile if args.pr86_profile.is_file() else None
    payload = build_manifest(
        frontier_json=args.frontier_json,
        pr85_profile=pr85_profile,
        pr86_profile=pr86_profile,
    )
    data = _json_bytes(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(data)
    print(data.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
