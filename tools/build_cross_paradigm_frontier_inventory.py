#!/usr/bin/env python3
"""Build a deterministic cross-paradigm frontier inventory.

The output is an orchestration artifact, not a score ledger. It separates exact
evidence from planning/proposal surfaces and makes stack/replacement candidates
discoverable without launching GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InventoryRow:
    key: str
    title: str
    paradigms: tuple[str, ...]
    role: str
    status: str
    evidence_grade: str
    stackability: str
    replacement_potential: str
    code_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    next_patch: str
    blockers: tuple[str, ...]


STATIC_ROWS: tuple[InventoryRow, ...] = (
    InventoryRow(
        key="hnerv_lowlevel_brotli_repack",
        title="HNeRV low-level brotli repack",
        paradigms=("alpha_mask_payload", "entropy_coding", "archive_packing"),
        role="stacker_rate_only",
        status="exact_evaluated_pr106x_rate_frontier",
        evidence_grade="A++ exact CUDA for PR106x; empirical for PR106 q10 rebuild until exact eval",
        stackability="high: raw-equivalent section recode should stack before scorer-changing atoms",
        replacement_potential="low: rate-only improvement, not a representation replacement",
        code_paths=(
            "src/tac/hnerv_lowlevel_packer.py",
            "src/tac/hnerv_section_repack.py",
            "tools/build_hnerv_lowlevel_repack_candidate.py",
            "tools/audit_hnerv_section_candidate_diff.py",
        ),
        evidence_paths=(
            "experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json",
            "experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/manifest.json",
            ".omx/research/hnerv_decoder_recode_pr106_20260506_codex.md",
        ),
        next_patch=(
            "Promote only exact-evaluated archive SHAs; keep PR106 q10 as archive-preflight-ready "
            "until lane claim and exact CUDA auth eval."
        ),
        blockers=("requires exact eval for each rebuilt archive SHA",),
    ),
    InventoryRow(
        key="hnerv_per_tensor_context_entropy",
        title="HNeRV per-tensor/context entropy recode",
        paradigms=("alpha_mask_payload", "entropy_coding"),
        role="stacker_rate_only",
        status="shared_context_fixture_landed_still_byte_negative",
        evidence_grade="empirical entropy-floor plus parity fixture profile",
        stackability="medium-high: could stack with HNeRV raw-equivalent section recodes",
        replacement_potential="low: recodes current decoder representation",
        code_paths=(
            "src/tac/hnerv_decoder_recode.py",
            "src/tac/arithmetic_qint_codec.py",
            "src/tac/lossless/range_coder.py",
            "src/tac/lossless/frequency_coder.py",
        ),
        evidence_paths=("experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json",),
        next_patch=(
            "Cluster or codebook-share HDC2 context tables; HDC2 cut PR106x penalty from "
            "+96,671B to +51,103B but remains byte-negative."
        ),
        blockers=(
            "HDC1/HDC2 parity fixtures are raw-equal but still byte-negative versus source brotli",
            "requires deterministic decoder runtime before any exact eval",
        ),
    ),
    InventoryRow(
        key="hnerv_wavelet_wr01_apply",
        title="HNeRV WR01 wavelet apply transform",
        paradigms=("alpha_mask_payload", "wavelet", "latent_repair"),
        role="stacker_scorer_changing",
        status="archive_candidate_preflighted",
        evidence_grade="empirical archive candidate; exact CUDA pending",
        stackability="medium: stack after raw-equivalent rate recodes, before sidechannel stacks",
        replacement_potential="low-medium: local residual transform, not full representation replacement",
        code_paths=(
            "src/tac/hnerv_wavelet_residual.py",
            "src/tac/hnerv_wavelet_apply_transform.py",
            "src/tac/hnerv_wavelet_apply_gate.py",
            "tools/build_hnerv_wavelet_apply_transform_candidate.py",
            "tools/build_wr01_exact_eval_packet.py",
        ),
        evidence_paths=(
            "experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json",
            "experiments/results/hnerv_wavelet_sidechannel_pr106x_20260506_codex/manifest.json",
        ),
        next_patch="Harvest/finalize incoming custody hardening, then exact CUDA only after lane claim.",
        blockers=("changes decoded output", "tiny byte win means component drift dominates EV"),
    ),
    InventoryRow(
        key="categorical_qma9_clade_spade_openpilot",
        title="Categorical QMA9 plus CLADE/SPADE/openpilot labels",
        paradigms=("alpha_mask_payload", "categorical_masks", "openpilot_priors"),
        role="replacement_or_mask_stacker",
        status="contract_and_candidate_readiness_landed_needs_byte_closed_candidate",
        evidence_grade="external/planning plus deterministic archive-readiness audit",
        stackability=(
            "medium: strongest as mask grammar or class-conditioned residual layer; must not duplicate "
            "HNeRV decoder bytes"
        ),
        replacement_potential="high for mask stream if learned/predictive grammar avoids CMG2 collapse",
        code_paths=(
            "src/tac/categorical_compression_contract.py",
            "src/tac/categorical_candidate_readiness.py",
            "src/tac/qma9_range_mask_contract.py",
            "src/tac/qma9_run_grammar.py",
            "src/tac/qma9_alt_grammar.py",
            "src/tac/mask_grayscale_lut.py",
            "src/tac/semantic_quantization.py",
            "src/tac/contrib/diffusion_renderer.py",
            "src/tac/openpilot_seeding.py",
            "tools/audit_categorical_compression_contract.py",
            "tools/audit_categorical_candidate_readiness.py",
        ),
        evidence_paths=(
            ".omx/research/qma9_range_mask_deconstruction_20260503_codex.md",
            ".omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md",
            "experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json",
        ),
        next_patch=(
            "Build the first real byte-closed categorical candidate and pass "
            "tools/audit_categorical_candidate_readiness.py before any lane claim or exact eval."
        ),
        blockers=(
            "raw lossless class wrapping was byte-regressive",
            "CMG2 exact evals showed PoseNet collapse",
            "CLADE/SPADE/openpilot priors need charged archive consumption",
        ),
    ),
    InventoryRow(
        key="cmg3_predictive_mask_grammar",
        title="CMG3 predictive row-span mask grammar",
        paradigms=("alpha_mask_payload", "categorical_masks", "foveation"),
        role="replacement_mask_stream",
        status="empirical_byte_probe_needs_runtime_decoder",
        evidence_grade="empirical byte probe only",
        stackability="medium: can feed residual/categorical/foveated atoms if decoder is closed",
        replacement_potential="high for C067-style mask stream; unknown for HNeRV frontier",
        code_paths=(
            "experiments/probe_predictive_mask_grammar.py",
            "experiments/build_cmg3_rowspan_candidate.py",
            "experiments/plan_cmg3_pixel_lagrangian_atoms.py",
            "experiments/build_cmg3_adaptive_runs_candidate.py",
        ),
        evidence_paths=(
            ".omx/research/shannon_floor_claim_matrix_20260430_codex.md",
            ".omx/research/charged_mask_grammar_ego_foveation_greenup_20260502_codex.md",
        ),
        next_patch="Close a deterministic runtime decoder and exact archive fixture before ranking.",
        blockers=("existing CMG2 repairs are A-negative", "probe excludes decoder/archive overhead"),
    ),
    InventoryRow(
        key="lapose_motion_atom_allocator",
        title="LA-POSE motion atom allocator",
        paradigms=("la_pose", "openpilot_priors", "meta_lagrangian", "pose"),
        role="proposal_allocator",
        status="planning_chain_hardened_non_rankable_global_allocations",
        evidence_grade="diagnostic CUDA response allocated to pairs; planning only",
        stackability="high as allocator feedback for pose, foveation, categorical, and sidechannel atoms",
        replacement_potential="medium if it produces a charged motion decoder or pose-sidecar replacement",
        code_paths=(
            "src/tac/analysis/lapose_lite_inputs.py",
            "src/tac/analysis/lapose_motion_atoms.py",
            "src/tac/analysis/lapose_motion_evidence.py",
            "tools/build_lapose_lite_inputs_from_pair_metrics.py",
            "tools/build_lapose_motion_atom_manifest.py",
        ),
        evidence_paths=("docs/runbooks/analysis_optimization_package_map.md",),
        next_patch=(
            "Keep labeled as LA-Pose-inspired until a paper-faithful inverse-dynamics encoder and "
            "pose head exist; add class/openpilot manifests, calibrate confidence, and require a "
            "charged archive consumer before dispatch."
        ),
        blockers=("current outputs are planning only", "confidence thresholds need calibration evidence"),
    ),
    InventoryRow(
        key="meta_lagrangian_cross_paradigm_allocator",
        title="Meta-lagrangian cross-paradigm allocator",
        paradigms=("meta_lagrangian", "all_paradigms"),
        role="canonical_ranker",
        status="planning_ranker_landed",
        evidence_grade="derivation/planning",
        stackability="high: common row schema can compare rate, pose, seg, class, hard-pair support",
        replacement_potential="none directly; selects concrete stack/replacement builders",
        code_paths=("src/tac/optimization/meta_lagrangian_allocator.py",),
        evidence_paths=(".omx/research/atom_lagrangian_waterfill_sub03_system_20260501_codex.md",),
        next_patch=(
            "Add cross-paradigm family/conflict fields and refuse dispatchable rows unless a "
            "byte-closed archive manifest is attached."
        ),
        blockers=("planner assumes additive expected deltas until exact stacked archives land",),
    ),
    InventoryRow(
        key="sensitivity_omega_w_v3",
        title="Sensitivity-aware Omega-W-V3",
        paradigms=("beta_sensitivity", "quantization", "waterfill"),
        role="stacker_quantization",
        status="guarded_needs_real_sensitivity",
        evidence_grade="planning/diagnostic until certified sensitivity artifact",
        stackability="medium-high after HNeRV source archive identity is locked",
        replacement_potential="low-medium: quantization policy, not representation replacement",
        code_paths=(
            "src/tac/sensitivity_map.py",
            "src/tac/component_sensitivity_artifact.py",
            "src/tac/owv3_sensitivity_weighted.py",
            "tools/dispatch_dryrun_omega_w_v3.py",
            "experiments/repack_pr106_with_water_filling.py",
        ),
        evidence_paths=(".omx/research/component_sensitivity_map_certification_20260501_codex.md",),
        next_patch="Replace all-ones/stub sensitivity producers with certified CUDA/component artifacts.",
        blockers=("stub sensitivity must fail closed", "component collapse risk"),
    ),
    InventoryRow(
        key="joint_admm_balle_arithmetic_stack",
        title="Joint ADMM plus Balle hyperprior plus arithmetic stack",
        paradigms=("gamma_joint_codec", "entropy_coding", "hyperprior"),
        role="stack_or_replacement_orchestrator",
        status="components_landed_not_end_to_end_stacked",
        evidence_grade="planning/component empirical",
        stackability="high in architecture, unproven in exact archive",
        replacement_potential="medium-high if it can replace HNeRV sections with a smaller scorer-stable stack",
        code_paths=(
            "src/tac/joint_codec_stack_orchestrator.py",
            "src/tac/joint_admm_coordinator.py",
            "src/tac/balle_hyperprior_codec.py",
            "src/tac/arithmetic_qint_codec.py",
            "tools/audit_arithmetic_qint_optimality.py",
            "src/tac/entropy_archive.py",
        ),
        evidence_paths=("docs/score_aware_sidechannel_paradigm_20260504.md",),
        next_patch="Build one end-to-end typed stack manifest and a no-op fixture before optimization.",
        blockers=("individual components do not imply composability", "side information must be charged"),
    ),
    InventoryRow(
        key="telescopic_foveation_field",
        title="Telescopic foveation field",
        paradigms=("foveation", "openpilot_priors", "categorical_masks", "pose"),
        role="scorer_weighted_proposal_or_replacement",
        status="historical_mixed_needs_qfaithful_successor_contract",
        evidence_grade="planning plus historical negatives",
        stackability="medium: best as atom-ranking/foveation prior before charged runtime warp",
        replacement_potential="medium if runtime consumes charged geometry without PoseNet collapse",
        code_paths=(
            "src/tac/hyperbolic_foveation.py",
            "src/tac/foveation_readiness.py",
            "tools/audit_hyperbolic_foveation_readiness.py",
            "experiments/plan_cmg3_pixel_lagrangian_atoms.py",
            "experiments/preflight_qfaithful_successor_geometry_contract.py",
            "src/tac/raft_radial_pose.py",
        ),
        evidence_paths=(
            ".omx/research/lane_g_v3_stacking_skunkworks_20260428.md",
            ".omx/research/all_scores_forensic_audit_20260430.md",
        ),
        next_patch=(
            "Run charged foveation-params readiness audit, then keep foveation as ranking feedback "
            "until a runtime consumer passes geometry preflight and exact component gates."
        ),
        blockers=("configured-but-unconsumed foveation is a known fail-closed preflight class",),
    ),
    InventoryRow(
        key="raft_radial_openpilot_pose",
        title="RAFT/radial/openpilot pose basis",
        paradigms=("pose", "openpilot_priors", "la_pose"),
        role="proposal_or_pose_sidecar_replacement",
        status="prototype_readiness_needed",
        evidence_grade="planning/prototype",
        stackability="medium: can seed la-pose atoms, foveation centers, or pose-sidecar search",
        replacement_potential="medium if charged pose stream beats current pose bytes/components",
        code_paths=(
            "src/tac/raft_pose.py",
            "src/tac/raft_radial_pose.py",
            "src/tac/openpilot_seeding.py",
            "src/tac/openpilot_features.py",
        ),
        evidence_paths=(".omx/research/council_lane_raft_radial_pose_design_20260430.md",),
        next_patch="Emit deterministic pose-disagreement and runtime-consumption readiness artifacts.",
        blockers=("RAFT full-frame jobs historically OOM at naive settings", "small pose errors can dominate score"),
    ),
    InventoryRow(
        key="selfcompress_mdl_fp4_tto",
        title="Self-compress NN plus MDL/Bayesian plus FP4/TTO",
        paradigms=("delta_epsilon_zeta", "self_compress", "mdl", "tto"),
        role="replacement_renderer_or_decoder",
        status="pending_integration",
        evidence_grade="planning/prototype",
        stackability="medium-low until deterministic export/inflate closure exists",
        replacement_potential="high if it can replace HNeRV renderer with smaller scorer-stable decoder",
        code_paths=(
            "src/tac/self_compressing_nn.py",
            "src/tac/mdl_bayesian_codec.py",
            "src/tac/joint_renderer_scorer_finetune.py",
            "src/tac/tto.py",
            "src/tac/fp4_quantize.py",
        ),
        evidence_paths=(".omx/research/council_lane_mdl_bayesian_design_20260430.md",),
        next_patch="Require deterministic export manifest, inflate budget proof, and no scorer load at inflate.",
        blockers=("runtime/export closure risk", "TTO must not rely on sidecars outside archive"),
    ),
)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _path_status(paths: tuple[str, ...], *, repo_root: Path) -> dict[str, Any]:
    existing = [path for path in paths if (repo_root / path).exists()]
    missing = [path for path in paths if not (repo_root / path).exists()]
    return {
        "existing": existing,
        "missing": missing,
        "existing_count": len(existing),
        "missing_count": len(missing),
    }


def _score_snapshot(row: InventoryRow, *, repo_root: Path) -> dict[str, Any] | None:
    for relpath in row.evidence_paths:
        if not relpath.endswith("contest_auth_eval.adjudicated.json"):
            continue
        payload = _load_json(repo_root / relpath)
        if not isinstance(payload, dict):
            continue
        score = payload.get(
            "score_recomputed_from_components",
            payload.get("canonical_score", payload.get("final_score", payload.get("score"))),
        )
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
        return {
            "path": relpath,
            "score": score,
            "archive_bytes": archive.get(
                "bytes",
                payload.get("archive_bytes", payload.get("archive_size_bytes")),
            ),
            "archive_sha256": archive.get("sha256", payload.get("archive_sha256")),
            "seg_dist": components.get("seg_dist", payload.get("seg_dist", payload.get("avg_segnet_dist"))),
            "pose_dist": components.get(
                "pose_dist",
                payload.get("pose_dist", payload.get("avg_posenet_dist")),
            ),
        }
    return None


def build_inventory(*, repo_root: Path) -> dict[str, Any]:
    rows = []
    for row in STATIC_ROWS:
        code_status = _path_status(row.code_paths, repo_root=repo_root)
        evidence_status = _path_status(row.evidence_paths, repo_root=repo_root)
        rows.append(
            {
                "key": row.key,
                "title": row.title,
                "paradigms": list(row.paradigms),
                "role": row.role,
                "status": row.status,
                "evidence_grade": row.evidence_grade,
                "stackability": row.stackability,
                "replacement_potential": row.replacement_potential,
                "code_paths": list(row.code_paths),
                "evidence_paths": list(row.evidence_paths),
                "path_audit": {
                    "code": code_status,
                    "evidence": evidence_status,
                },
                "score_snapshot": _score_snapshot(row, repo_root=repo_root),
                "next_patch": row.next_patch,
                "blockers": list(row.blockers),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    role_counts = Counter(row["role"] for row in rows)
    paradigm_counts = Counter(paradigm for row in rows for paradigm in row["paradigms"])
    missing_code = sum(row["path_audit"]["code"]["missing_count"] for row in rows)
    missing_evidence = sum(row["path_audit"]["evidence"]["missing_count"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/build_cross_paradigm_frontier_inventory.py",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "research_grade_standard": {
            "external_method_rule": (
                "Do not claim 1:1 paper implementation unless the repo has the paper's core "
                "model, training objective, inputs, outputs, and evaluation contract."
            ),
            "contest_native_rule": (
                "Contest-native methods must carry typed contracts, deterministic manifests, "
                "no-op controls, byte-closed archive consumption, and exact CUDA custody before "
                "promotion."
            ),
        },
        "row_count": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "paradigm_counts": dict(sorted(paradigm_counts.items())),
        "missing_code_path_count": missing_code,
        "missing_evidence_path_count": missing_evidence,
        "rows": rows,
        "dispatch_blockers": [
            "inventory_only",
            "requires_candidate_specific_archive_manifest",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Paradigm Frontier Inventory",
        "",
        "Inventory-only orchestration artifact. It does not claim scores or unlock dispatch.",
        "",
        f"- row_count: `{payload['row_count']}`",
        f"- missing_code_path_count: `{payload['missing_code_path_count']}`",
        f"- missing_evidence_path_count: `{payload['missing_evidence_path_count']}`",
        "",
        "| key | role | paradigms | status | stackability | replacement | next patch | blockers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_md(row['key'])}`",
                    f"`{_md(row['role'])}`",
                    "<br>".join(f"`{_md(item)}`" for item in row["paradigms"]),
                    _md(row["status"]),
                    _md(row["stackability"]),
                    _md(row["replacement_potential"]),
                    _md(row["next_patch"]),
                    "<br>".join(_md(item) for item in row["blockers"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_inventory(repo_root=args.repo_root)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        sys.stdout.write(json_text(payload) if args.format == "json" else render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
