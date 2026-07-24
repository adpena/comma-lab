#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the SHA-bound five-type addendum for the settled SN1 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Final

from tac.analysis.ddm_sn1_five_type_addendum import (
    SN1FiveTypeStreamTag,
    build_five_type_addendum,
    sha256_file,
)

SN1_ROOT: Final = Path(".omx/research/ddm_sn1_segnet_telemetry_asymmetry_n600_20260723")
TENSOR_ROOT: Final = Path(".omx/research/ddm_sn1_error_source_tensor_n600_20260723")
SN1_RECEIPT: Final = SN1_ROOT / "ddm_sn1_segnet_telemetry_asymmetry_receipt.json"
TENSOR_RECEIPT: Final = TENSOR_ROOT / "ddm_sn1_error_source_tensor_receipt.json"
SN1_RECEIPT_SHA256: Final = "1f727c00f91a2425f8d40660b93594515c8038c42b802226ff11348791190eb4"
TENSOR_RECEIPT_SHA256: Final = "ecf9f015fa6999b9bb7602c93027da713bb278389b92d5d1bf0b95f4ced19faa"
DEFAULT_OUTPUT: Final = Path(".omx/research/ddm_sn1_five_type_derivation_addendum_20260724.json")


def _tag(
    stream_id: str,
    artifact_path: Path,
    artifact_sha256: str,
    artifact_selector: str,
    representation_type: str,
    layer_home: str,
    evaluate_recursion_level: str,
    derivation: str,
    metric_geometry: str,
    first_rung: str,
    verdict_scope: str,
) -> SN1FiveTypeStreamTag:
    return SN1FiveTypeStreamTag(
        stream_id=stream_id,
        artifact_path=str(artifact_path),
        artifact_sha256=artifact_sha256,
        artifact_selector=artifact_selector,
        representation_type=representation_type,
        layer_home=layer_home,
        evaluate_recursion_level=evaluate_recursion_level,
        derivation=derivation,
        metric_geometry=metric_geometry,
        first_rung=first_rung,
        verdict_scope=verdict_scope,
    )


def _load_pinned_receipt(
    repo_root: Path,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    resolved = repo_root / path
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise RuntimeError(f"pinned receipt SHA drift for {path}: expected {expected_sha256}, observed {observed}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"pinned receipt must be a JSON object: {path}")
    return value


def build_tags(repo_root: Path) -> list[SN1FiveTypeStreamTag]:
    """Declare the complete SN1 stream/finding type inventory."""

    sn1_receipt = _load_pinned_receipt(repo_root, SN1_RECEIPT, SN1_RECEIPT_SHA256)
    tensor_receipt = _load_pinned_receipt(repo_root, TENSOR_RECEIPT, TENSOR_RECEIPT_SHA256)
    sn1_shas = {Path(row["path"]).name: row["sha256"] for row in sn1_receipt["outputs"]}
    tensor_shas = {key: row["sha256"] for key, row in tensor_receipt["artifacts"].items()}

    telemetry = SN1_ROOT / "segnet_internal_telemetry_n600.jsonl"
    sided = SN1_ROOT / "sdwl1_sided_tolerance_n600.jsonl"
    inverse = SN1_ROOT / "inverse_solve_three_segments_receipt.json"
    tensor = TENSOR_ROOT / "error_source_tensor_n600.jsonl.gz"
    menu = TENSOR_ROOT / "error_source_solve_menu.jsonl"
    budget = TENSOR_ROOT / "error_source_budget.json"
    seg_product = TENSOR_ROOT / "segnet_scorer_native_product_n600.jsonl.gz"
    pose_product = TENSOR_ROOT / "posenet_scorer_native_product_n600.jsonl.gz"
    seg_analytic = TENSOR_ROOT / "segnet_analytic_knowledge.jsonl.gz"
    pose_analytic = TENSOR_ROOT / "posenet_analytic_knowledge.jsonl.gz"
    paint_budget = TENSOR_ROOT / "paint_floor_mechanism_budget.json"
    survival_wall = TENSOR_ROOT / "survival_wall_149_n600.json"
    vocabulary_ranking = TENSOR_ROOT / "vocabulary_gap_ranking.json"

    return [
        _tag(
            "telemetry_identity_noop",
            SN1_RECEIPT,
            SN1_RECEIPT_SHA256,
            "/telemetry_identity",
            "GAUGE",
            "L5_VERDICT",
            "L0_SCORE_SIGNATURE",
            (
                "read-only hooks alter the observation trace while exact logits "
                "and argmax remain unchanged, so the evaluator quotient removes "
                "the telemetry direction"
            ),
            "IDENTITY_NOOP_GAUGE",
            ("retain this on/off identity guard for every future scorer-hook consumer"),
            ("frozen SN1 hook path only; this is score neutrality, not a receiver ker(R) dimension or byte saving"),
        ),
        _tag(
            "segnet_internal_activation_values",
            telemetry,
            sn1_shas[telemetry.name],
            "schema=ddm_sn1_segnet_telemetry.frame.v1",
            "FIBER",
            "L3_SCORER_FEATURE",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "continuous per-layer activations, margin evolution, channel "
                "energy, boundary-band response, and ERF response lie inside "
                "the Seg term before argmax"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("measure intermediate Fisher pullbacks only in a separately claimed relay-solve probe"),
            ("n600 frozen-SegNet advisory summaries; no raw activation tensor or full Jacobian spectrum is claimed"),
        ),
        _tag(
            "segnet_temporal_feature_transport",
            seg_product,
            tensor_shas["segnet_scorer_native_product"],
            "/layers/*/(across_frame|across_pair|xi_advected|temporal_spectrum)",
            "CONNECTION",
            "L3_SCORER_FEATURE",
            "L2_TEMPORAL_COMPOSITION",
            (
                "ordered pairs and the clip trajectory induce transport between "
                "feature fibers; the connection owns cross-frame, cross-pair, "
                "and xi-advected comparisons"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("test a receiver-realized relay candidate at the highest-ranked contracting depth"),
            (
                "directional secants and transported feature summaries only; "
                "not a causal Jacobian or receiver intervention"
            ),
        ),
        _tag(
            "posenet_pair_feature_fibers",
            pose_product,
            tensor_shas["posenet_scorer_native_product"],
            "/layers/*",
            "FIBER",
            "L3_SCORER_FEATURE",
            "L2_TEMPORAL_COMPOSITION",
            (
                "PoseNet consumes two frames jointly, so its continuous layer "
                "state is a pair-native fiber generated by temporal recursion"
            ),
            "POSE_EXACT_OUTPUT_QUADRATIC_LE6",
            ("bind any relay proposal to the exact six-output Pose quadratic before admission"),
            ("scorer-native advisory product; the first-six diagnostic is not official d_pose"),
        ),
        _tag(
            "segnet_frozen_weight_analytic_fibers",
            seg_analytic,
            tensor_shas["segnet_analytic_knowledge"],
            "/layers/*/(conv_dft|batchnorm|squeeze_excite|contrast)",
            "FIBER",
            "L3_SCORER_FEATURE",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "frozen convolution, BatchNorm, squeeze-excite, and contrast "
                "operators derive the continuous Seg feature fiber before "
                "empirical validation"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("measure only the analytic-versus-composed residual at the relay depths selected by scorer geometry"),
            (
                "exact frozen-weight factors for recorded local operators; no "
                "single global uint8 resize transfer scalar is inferred"
            ),
        ),
        _tag(
            "posenet_frozen_weight_analytic_fibers",
            pose_analytic,
            tensor_shas["posenet_analytic_knowledge"],
            "/layers/*/(conv_dft|batchnorm1d|layerscale2d|gelutanh|se)",
            "FIBER",
            "L3_SCORER_FEATURE",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "the frozen Pose stack derives continuous pair-feature fibers "
                "with 24 LayerScale2d, 8 BatchNorm1d, 19 GELUTanh, and one SE"
            ),
            "POSE_EXACT_OUTPUT_QUADRATIC_LE6",
            ("compose the analytic factors into the exact six-output Pose quadratic before a relay proposal"),
            ("frozen-weight inventory and local responses only; no fictitious per-block PoseNet SE factors"),
        ),
        _tag(
            "sdwl1_boundary_token_membership",
            sided,
            sn1_shas[sided.name],
            "schema=sdwl1.sided_tolerance.row.v1::orientation",
            "SKELETON",
            "L1_PROGRAM",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "Seg argmax produces discrete ordered cell adjacency; SDWL1 "
                "boundary tokens store that sided skeleton in the grammar"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("compile the same orientation field into the e1 exporter and prove parse-back before byte pricing"),
            (
                "ordered class adjacency on frozen-SegNet n600 boundaries; absent "
                "support remains absent rather than zero"
            ),
        ),
        _tag(
            "sdwl1_sided_margin_fiber",
            sided,
            sn1_shas[sided.name],
            ("schema=sdwl1.sided_tolerance.row.v1::inner_tolerance_d2,outer_tolerance_d2"),
            "FIBER",
            "L4_SCORER_DECISION",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "within each ordered argmax cell, rank-4 winner/rival margins "
                "induce the continuous sided decision-distance fiber"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("price inner and outer fields with SHA-current e2 costates on the same realized candidate"),
            ("head-space boundary distance only; pixel realization and Pose collateral are not inferred"),
        ),
        _tag(
            "receiver_inverse_segment_residual",
            inverse,
            sn1_shas[inverse.name],
            "/rows/*",
            "RESIDUAL",
            "L2_RECEIVER_R",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "bounded camera edits are receiver-side corrections against the "
                "sided target after exact uint8, R, and frozen SegNet"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("replace feasibility deltas with a receiver-closed counted candidate and jointly measure Pose collateral"),
            (
                "three local segment demonstrations with collateral flips "
                "retained; not an archive or ordered-pair family verdict"
            ),
        ),
        _tag(
            "v19c_error_source_residual_tensor",
            tensor,
            tensor_shas["tensor"],
            "schema=ddm_sn1_error_source_tensor.row.v1",
            "RESIDUAL",
            "L4_SCORER_DECISION",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "target-versus-argmax disagreement remaining after current "
                "description and receiver realization is the decision residual"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("route each solvable cluster to vocabulary, chart, or receiver repair before any descent"),
            (
                "exact v19c plus one SHA-pinned DV1 extension; the scoped-hard "
                "bucket is not a family impossibility result"
            ),
        ),
        _tag(
            "record_level_temporal_constancy",
            SN1_RECEIPT,
            SN1_RECEIPT_SHA256,
            "/measurement/record_constancy_correction",
            "CONNECTION",
            "L4_SCORER_DECISION",
            "L2_TEMPORAL_COMPOSITION",
            (
                "record identity is defined along the 600-state trajectory, so "
                "adjacent changes and recurrence are properties of a temporal "
                "connection rather than repeated pixel coordinates"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("test persistent primitives plus sparse innovations; do not retry whole-record static coding"),
            (
                "all eleven current record formulations change at every adjacent "
                "state; this rejects whole-record static coding only"
            ),
        ),
        _tag(
            "paint_floor_mechanism_residual_budget",
            paint_budget,
            tensor_shas["paint_floor_mechanism_budget"],
            "/source_totals",
            "RESIDUAL",
            "L4_SCORER_DECISION",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "boundary distance, curve availability, and margin partition "
                "realization loss into observable decision-residual mechanisms"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("measure a same-candidate receiver intervention to separate association from latent cause"),
            ("deterministic observable-axis partition; not hidden-cause identification"),
        ),
        _tag(
            "boundary_survival_wall_verdict",
            survival_wall,
            tensor_shas["survival_wall_149"],
            "/current_vehicle",
            "RESIDUAL",
            "L5_VERDICT",
            "L0_SCORE_SIGNATURE",
            (
                "the target-boundary residual fraction is a Seg-verdict bound "
                "on placement-only recovery for the current vehicle"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("remeasure on the exact receiver candidate before using the wall as an admission ceiling"),
            (
                "current n600 receiver and historical mp128 reference have "
                "different scopes; their ratio is contextual, not promotable"
            ),
        ),
        _tag(
            "solve_menu_program_skeleton",
            menu,
            tensor_shas["solve_menu"],
            "/menu/solved_move",
            "SKELETON",
            "L1_PROGRAM",
            "L1_TERM_NATIVE_GEOMETRY",
            ("vocabulary, event, and chart moves are discrete program edits that propose a new description skeleton"),
            "RATE_EXACT_BYTES",
            ("measure receiver survival and exact counted bytes for the highest-mass vocabulary cluster"),
            (
                "menu rows rank measured error mass and shared prices; they are "
                "not receiver-closed score-unit values per byte"
            ),
        ),
        _tag(
            "vocabulary_gap_program_ranking",
            vocabulary_ranking,
            tensor_shas["vocabulary_gap_ranking"],
            "/rows/*",
            "SKELETON",
            "L1_PROGRAM",
            "L1_TERM_NATIVE_GEOMETRY",
            (
                "shared residual geometry proposes a missing or misshaped "
                "description primitive in the discrete program skeleton"
            ),
            "RATE_EXACT_BYTES",
            (
                "receiver-close the highest-mass primitive and measure exact "
                "amortized bytes before chart or point correction"
            ),
            (
                "semantic error mass per shared byte only; receiver survival, "
                "Pose value, and exact archive value remain owed"
            ),
        ),
        _tag(
            "three_way_error_budget_verdict",
            budget,
            tensor_shas["budget_json"],
            "/source_totals",
            "RESIDUAL",
            "L5_VERDICT",
            "L0_SCORE_SIGNATURE",
            (
                "the exact residual partition rolls decision errors up to the "
                "Seg verdict term while retaining source-class custody"
            ),
            "SEG_MARGIN_FISHER_RANK4",
            ("use classes i and ii as menu allocations; reserve descent for the measured class-iii leftover only"),
            ("advisory d_seg accounting on the frozen local axis; no contest CPU/CUDA score or promotion claim"),
        ),
    ]


def build(repo_root: Path) -> dict[str, object]:
    source_receipts = [
        {"path": str(SN1_RECEIPT), "sha256": SN1_RECEIPT_SHA256},
        {"path": str(TENSOR_RECEIPT), "sha256": TENSOR_RECEIPT_SHA256},
    ]
    return build_five_type_addendum(
        repo_root=repo_root,
        source_receipts=source_receipts,
        tags=build_tags(repo_root),
        typed_stream_tag_status=(
            "ABSENT_AT_FIRE_TIME_COMPATIBILITY_ROWS_REUSE_CANONICAL_"
            "REPRESENTATION_TYPES_MAIN_MUST_ADAPT_IF_TS1_LANDS_FIRST"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    payload = build(repo_root)
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "typed_stream_count": payload["typed_stream_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
