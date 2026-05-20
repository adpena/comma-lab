# SPDX-License-Identifier: MIT
"""Source-backed theoretical-floor candidate contracts.

This module is deliberately not a strategy essay. It records which recent
external ideas are allowed to influence compress-time search, what the contest
runtime is permitted to consume, and which local gate must run before a score
claim can exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "theoretical_floor_candidate_matrix_v1"

PR110_ARCHIVE_BYTES = 178_517
CONTEST_NORMALIZER = 37_545_489.0
PR110_SEGNET_DIST = 0.00056029
PR110_POSENET_DIST = 0.00002943


@dataclass(frozen=True)
class SourceReference:
    """Primary source or local evidence used by one or more candidates."""

    source_id: str
    title: str
    url: str
    role: str
    runtime_dependency_allowed: bool


@dataclass(frozen=True)
class RuntimeContract:
    """Contest-runtime boundary for a candidate family."""

    candidate_id: str
    objective: str
    runtime_payload_members: tuple[str, ...]
    runtime_dependencies: tuple[str, ...]
    compress_time_teachers: tuple[str, ...]
    local_surfaces: tuple[str, ...]
    first_gate: str
    dispatch_blockers: tuple[str, ...]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


def pr110_score_decomposition() -> dict[str, float]:
    """Return the verified PR #110 component decomposition used as baseline."""

    seg_term = 100.0 * PR110_SEGNET_DIST
    pose_term = (10.0 * PR110_POSENET_DIST) ** 0.5
    rate_term = 25.0 * PR110_ARCHIVE_BYTES / CONTEST_NORMALIZER
    return {
        "segnet_raw_distortion": PR110_SEGNET_DIST,
        "posenet_raw_distortion": PR110_POSENET_DIST,
        "segnet_weighted_score_term": seg_term,
        "posenet_weighted_score_term": pose_term,
        "rate_weighted_score_term": rate_term,
        "distortion_only_zero_archive_score": seg_term + pose_term,
        "rate_only_perfect_distortion_score": rate_term,
        "total_recomputed_score": seg_term + pose_term + rate_term,
    }


def source_references() -> tuple[SourceReference, ...]:
    """Return source references used in the matrix.

    All external references are compress-time or design references unless a
    byte-closed archive/runtime gate explicitly promotes a small derived
    payload into the contest runtime.
    """

    return (
        SourceReference(
            source_id="telescope_hyperbolic_foveation",
            title="Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection",
            url="https://princeton-computational-imaging.github.io/Telescope/",
            role="Source for compact hyperbolic foveation transforms and far-field scale normalization.",
            runtime_dependency_allowed=False,
        ),
        SourceReference(
            source_id="la_pose",
            title="LA-Pose: Latent Action Pretraining Meets Pose Estimation",
            url="https://arxiv.org/abs/2604.27448",
            role="Compress-time pose/latent-action teacher for selecting geometry-sensitive atoms.",
            runtime_dependency_allowed=False,
        ),
        SourceReference(
            source_id="sea_raft",
            title="SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow",
            url="https://arxiv.org/abs/2405.14793",
            role="Compress-time optical-flow teacher for motion/foveation priors.",
            runtime_dependency_allowed=False,
        ),
        SourceReference(
            source_id="visual_primitives",
            title="Thinking with Visual Primitives",
            url="https://www.k-a.in/Thinking_with_Visual_Primitives.pdf",
            role="Design reference for small point/box primitives as charged spatial hints.",
            runtime_dependency_allowed=False,
        ),
        SourceReference(
            source_id="siren",
            title="Implicit neural representation / SIREN family",
            url="https://arxiv.org/abs/2006.09661",
            role="Candidate full renderer with a small coordinate-network inflate runtime.",
            runtime_dependency_allowed=True,
        ),
        SourceReference(
            source_id="vq_vae",
            title="Neural Discrete Representation Learning / VQ-VAE family",
            url="https://arxiv.org/abs/1711.00937",
            role="Candidate full renderer using charged codebook and token streams.",
            runtime_dependency_allowed=True,
        ),
    )


def runtime_contracts() -> tuple[RuntimeContract, ...]:
    """Return candidate runtime contracts ordered by current local actionability."""

    return (
        RuntimeContract(
            candidate_id="tf_siren_first_anchor",
            objective=(
                "Replace HNeRV-family representation with a byte-closed SIREN "
                "coordinate renderer and exact archive/inflate contract."
            ),
            runtime_payload_members=("0.bin", "inflate.py", "inflate.sh"),
            runtime_dependencies=("stdlib", "torch"),
            compress_time_teachers=("score_aware_loss",),
            local_surfaces=(
                "experiments/train_substrate_siren.py",
                "src/tac/substrates/siren/archive.py",
                "src/tac/substrates/siren/inflate.py",
                "tools/audit_siren_substrate_readiness.py",
            ),
            first_gate=(
                ".venv/bin/python experiments/train_substrate_siren.py "
                "--video-path upstream/videos/0.mkv --output-dir "
                "experiments/results/siren_smoke_<utc> --epochs 3 "
                "--device cpu --smoke --skip-archive-build --skip-auth-eval"
            ),
            dispatch_blockers=(
                "operator_authorization_required_for_gpu_spend",
                "active_lane_dispatch_claim_required",
                "exact_cuda_auth_eval_missing",
            ),
        ),
        RuntimeContract(
            candidate_id="tf_telescope_lfv1_pose_foveation",
            objective=(
                "Use Telescope-style hyperbolic foveation plus LA-Pose/SEA-RAFT/"
                "visual-primitive teachers to emit a tiny charged geometry "
                "payload consumed by a contest runtime."
            ),
            runtime_payload_members=(
                "lapose_foveation_tuples.lfv1",
                "foveation_params.bin",
                "runtime_consumer.py",
                "inflate.sh",
            ),
            runtime_dependencies=("stdlib", "numpy", "torch"),
            compress_time_teachers=(
                "telescope_hyperbolic_foveation",
                "la_pose",
                "sea_raft",
                "visual_primitives",
            ),
            local_surfaces=(
                "src/tac/hyperbolic_foveation.py",
                "src/tac/lapose_foveation_payload_candidate.py",
                "tools/build_lapose_foveation_atom_manifest.py",
                "tools/build_lapose_foveation_tuple_payload.py",
                "tools/build_lapose_foveation_payload_archive.py",
            ),
            first_gate=(
                ".venv/bin/python tools/build_lapose_foveation_payload_archive.py "
                "--out-dir experiments/results/theoretical_floor_lfv1_<utc>/archive_candidate "
                "--lfv1-payload experiments/results/theoretical_floor_lfv1_<utc>/"
                "lapose_foveation_tuples.lfv1 --source-readiness-json "
                "experiments/results/theoretical_floor_lfv1_<utc>/lfv1_payload_readiness.json"
            ),
            dispatch_blockers=(
                "runtime_loader_parity_not_passed",
                "scorer_visible_output_parity_not_proven",
                "runtime_output_parity_not_proven",
                "exact_cuda_auth_eval_missing",
            ),
        ),
        RuntimeContract(
            candidate_id="tf_vqvae_full_renderer",
            objective=(
                "Move from continuous HNeRV weights to discrete codebook tokens "
                "with charged codebook, token stream, and small decoder runtime."
            ),
            runtime_payload_members=("0.bin", "inflate.py", "inflate.sh"),
            runtime_dependencies=("stdlib", "torch", "brotli"),
            compress_time_teachers=("score_aware_loss", "codebook_perplexity_gate"),
            local_surfaces=(
                "src/tac/vqvae_as_full_renderer.py",
                "experiments/train_vqvae_as_renderer.py",
                "src/tac/vqvae_mask_codec.py",
            ),
            first_gate=(
                ".venv/bin/python -m pytest "
                "src/tac/tests/test_train_vqvae_as_renderer.py "
                "src/tac/tests/test_vqvae_mask_codec.py -q"
            ),
            dispatch_blockers=(
                "byte_closed_export_smoke_missing",
                "exact_cuda_auth_eval_missing",
            ),
        ),
        RuntimeContract(
            candidate_id="tf_c3_coolchic_sparse_residual",
            objective=(
                "Attach sparse learned-codec residual bytes to a proven base "
                "runtime, using score-aware sparse Lagrangian instead of empty "
                "sidecars."
            ),
            runtime_payload_members=("archive.zip", "residual_bytes"),
            runtime_dependencies=("stdlib", "numpy", "torch", "brotli"),
            compress_time_teachers=("cool_chic", "c3", "hinton_distilled_scorer"),
            local_surfaces=(
                "tools/materialize_cool_chic_residual_pr106_sidecar.py",
                "tools/materialize_c3_residual_pr106_sidecar.py",
                "src/tac/residual_basis/cool_chic_residual.py",
                "src/tac/residual_basis/c3_residual.py",
            ),
            first_gate=(
                "Run l2_encoded materializer with explicit --decoded-raw, "
                "--gt-raw, --byte-budget, --encoding sparse, and no score claim."
            ),
            dispatch_blockers=(
                "decoded_raw_required",
                "gt_raw_required",
                "explicit_byte_budget_required",
                "exact_cuda_auth_eval_missing",
            ),
        ),
    )


def build_candidate_matrix(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build a machine-readable matrix with optional local surface existence."""

    root = Path(repo_root) if repo_root is not None else None
    sources = [asdict(source) for source in source_references()]
    contracts: list[dict[str, Any]] = []
    for contract in runtime_contracts():
        item = asdict(contract)
        if root is not None:
            item["local_surface_status"] = {
                surface: (root / surface).exists() for surface in contract.local_surfaces
            }
        contracts.append(item)
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "baseline": {
            "id": "pr110_cpu_axis_component_decomposition",
            **pr110_score_decomposition(),
        },
        "sources": sources,
        "runtime_contracts": contracts,
        "policy": {
            "external_models_at_inflate_time": "forbidden_unless_weights_and_runtime_are_charged",
            "teacher_models": "compress_time_only",
            "score_claim_requires": (
                "byte_closed_archive_runtime, no_op_controls, component recompute, "
                "and exact CPU/CUDA auth-eval axis custody"
            ),
        },
    }


def render_candidate_matrix_markdown(matrix: dict[str, Any]) -> str:
    """Render a concise operator-facing markdown summary."""

    lines = [
        "# Source-Backed Theoretical-Floor Candidate Matrix",
        "",
        "Authority: design/runtime routing only; score_claim=false.",
        "",
        "## Baseline Correction",
        "",
    ]
    baseline = matrix["baseline"]
    lines.extend(
        [
            f"- weighted SegNet term: `{baseline['segnet_weighted_score_term']:.6f}`",
            f"- weighted PoseNet term: `{baseline['posenet_weighted_score_term']:.6f}`",
            f"- weighted rate term: `{baseline['rate_weighted_score_term']:.6f}`",
            f"- distortion-only zero-archive score: `{baseline['distortion_only_zero_archive_score']:.6f}`",
            f"- rate-only perfect-distortion score: `{baseline['rate_only_perfect_distortion_score']:.6f}`",
            "",
            "## Candidate Contracts",
            "",
            "| candidate | runtime payload | teachers | first gate | blockers |",
            "|---|---|---|---|---|",
        ]
    )
    for contract in matrix["runtime_contracts"]:
        lines.append(
            "| {candidate} | {payload} | {teachers} | `{gate}` | {blockers} |".format(
                candidate=contract["candidate_id"],
                payload=", ".join(contract["runtime_payload_members"]),
                teachers=", ".join(contract["compress_time_teachers"]),
                gate=contract["first_gate"],
                blockers=", ".join(contract["dispatch_blockers"]),
            )
        )
    lines.extend(["", "## Source References", ""])
    for source in matrix["sources"]:
        lines.append(
            f"- `{source['source_id']}`: {source['title']} ({source['url']}); "
            f"inflate-time dependency allowed: `{str(source['runtime_dependency_allowed']).lower()}`"
        )
    lines.append("")
    return "\n".join(lines)
