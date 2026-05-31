# SPDX-License-Identifier: MIT
"""Z8 joint variational driver primitives.

This module is intentionally small and executable: it supplies the MLX-local
rate/quantizer term that lets the Z8 score-aware harness optimize renderer
weights, categorical allocation, and archive pressure in the same gradient
step. It is not a score authority surface; it is the local driver that prepares
byte-closed candidates for receiver proof and exact-axis evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.joint_p18_p19_waterfill import (
    JOINT_P18_P19_RATE_ATTACK_ROLE,
    JOINT_P18_P19_WEIGHT_FORMULA,
)
from tac.score_composition import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
)

Z8_JOINT_VARIATIONAL_DRIVER_SCHEMA = "z8_joint_variational_driver.v1"
Z8_JOINT_P18_P19_WATERFILL_CONTRACT_SCHEMA = "z8_joint_p18_p19_gradient_waterfill_contract.v1"
Z8_TRAINED_MLX_RENDERED_OUTPUT_EXPORT_SCHEMA = "z8_trained_mlx_rendered_output_export.v1"


@dataclass(frozen=True)
class Z8JointVariationalDriverConfig:
    """Score-domain local driver configuration for Z8 MLX training.

    ``archive_rate_weight`` scales the expected categorical archive-rate term
    in canonical score units. ``argmax_commitment_weight`` sharpens the
    categorical posterior toward the byte-packed argmax path, giving the
    straight-through renderer gradient a matching archive-boundary pressure.
    """

    archive_rate_weight: float = 1.0
    argmax_commitment_weight: float = 1e-3
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.archive_rate_weight < 0.0:
            raise ValueError("archive_rate_weight must be >= 0")
        if self.argmax_commitment_weight < 0.0:
            raise ValueError("argmax_commitment_weight must be >= 0")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")


def expected_categorical_archive_rate_score(
    model: Any,
    pair_indices: Any,
    *,
    eps: float = 1e-8,
) -> Any:
    """Differentiable expected categorical rate in contest score units.

    The archive stores per-level argmax category indices; this proxy uses the
    current categorical posterior entropy as the differentiable pre-coder rate
    pressure. It is the local gradient field for ``25 * bytes / 37_545_489``.
    """

    import mlx.core as mx

    cfg = model.cfg
    log2 = float(np.log(2.0))
    total_bits = mx.zeros(())
    for logits_all in model.logits_per_level:
        logits = mx.take(logits_all, pair_indices, axis=0)
        probs = mx.softmax(logits, axis=-1)
        entropy_bits = -mx.sum(probs * mx.log(probs + float(eps)), axis=-1) / log2
        # Sum groups, then average sampled pairs.
        total_bits = total_bits + mx.mean(mx.sum(entropy_bits, axis=-1))

    expected_archive_bytes = (total_bits / 8.0) * float(cfg.num_pairs)
    return float(CANONICAL_RATE_MULTIPLIER) * expected_archive_bytes / float(CANONICAL_RATE_DENOM_BYTES)


def argmax_commitment_loss(
    model: Any,
    pair_indices: Any,
    *,
    eps: float = 1e-8,
) -> Any:
    """Differentiable confidence pressure toward the byte-packed argmax path."""

    import mlx.core as mx

    total = mx.zeros(())
    for logits_all in model.logits_per_level:
        logits = mx.take(logits_all, pair_indices, axis=0)
        probs = mx.softmax(logits, axis=-1)
        max_prob = mx.max(probs, axis=-1)
        total = total + mx.mean(-mx.log(max_prob + float(eps)))
    return total


def build_z8_joint_variational_extra_loss_terms(
    config: Z8JointVariationalDriverConfig,
) -> Callable[[Any, Any], Mapping[str, Any]]:
    """Build the Z8 extra-loss callback consumed by ``RendererBundle``."""

    def _terms(model: Any, pair_indices: Any) -> Mapping[str, Any]:
        return {
            "joint_archive_rate_score": expected_categorical_archive_rate_score(
                model,
                pair_indices,
                eps=config.eps,
            ),
            "joint_argmax_commitment": argmax_commitment_loss(
                model,
                pair_indices,
                eps=config.eps,
            ),
        }

    return _terms


def build_z8_joint_p18_p19_gradient_waterfill_contract() -> dict[str, Any]:
    """Return the joint scorer-surface contract for Z8 local acquisition.

    SegNet supplies the large argmax-flip boundary field, but PoseNet decides
    where freed bytes are dangerous. The contract makes that coupling explicit
    so downstream water-fill/acquisition code cannot silently run a SegNet-only
    spend policy.
    """

    return {
        "schema": Z8_JOINT_P18_P19_WATERFILL_CONTRACT_SCHEMA,
        "operator_stages": ["P19", "P18"],
        "score_functional": "100*d_seg + sqrt(10*d_pose) + 25*bytes/N",
        "weight_formula": JOINT_P18_P19_WEIGHT_FORMULA,
        "binding_axis_interpretation": (
            "z8_600_pair_advisory_is_rate_bound; use joint scorer weights as "
            "the wavelet detail-band dead-zone allocator"
        ),
        "rate_axis_attack_role": JOINT_P18_P19_RATE_ATTACK_ROLE,
        "executable_materializer": {
            "module": ("tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill"),
            "function": "materialize_joint_p18_p19_deadzone_candidate",
            "relinearized_search_function": ("materialize_joint_p18_p19_relinearized_deadzone_search"),
            "cli": "tools/materialize_z8_joint_p18_p19_deadzone_candidate.py",
            "output_schema": "z8_joint_p18_p19_coefficient_deadzone_candidate.v1",
            "relinearized_search_schema": ("z8_joint_p18_p19_coefficient_relinearized_search.v1"),
            "archive_target": "z8hpc1_wavelet_coeffs_blob",
            "mutation": "deterministic_detail_subband_deadzone_quantization",
            "surface_refresh_contract": ("fresh_joint_p18_p19_surface_per_iteration_from_mlx_scorer_vjp"),
            "full_video_surface_coverage_required": True,
        },
        "segnet_surface": {
            "stage": "P18",
            "role": "large_argmax_flip_boundary_gradient_surface",
            "required_measurement": "boundary_argmax_hinge_marginal_surface",
        },
        "posenet_surface": {
            "stage": "P19",
            "role": ("null_subset_detection_plus_mahalanobis_or_ail_pair_weighting"),
            "required_measurements": [
                "posenet_null_subset_pair_ids",
                "posenet_mahalanobis_or_ail_pair_weights",
            ],
        },
        "rate_spend_guard": (
            "dead_zone_low_joint_weight_wavelet_atoms_only; protect seg_boundary_and_pose_sensitive_atoms"
        ),
        "iterative_search": {
            "ste_boundary": "straight_through_deadzone_quantization_proxy",
            "interaction_penalty": ("penalize_cumulative_mse_increase_between_relinearization_steps"),
            "fresh_surface_required": True,
            "full_video_surface_coverage_required": True,
        },
        "forbidden_policy": "segnet_only_waterfill",
        "allowed_use": "local_mlx_joint_acquisition_routing_only",
        "forbidden_use": "score_claim_or_exact_axis_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_z8_joint_variational_driver_metadata(
    config: Z8JointVariationalDriverConfig,
    *,
    archive_export_enabled: bool,
) -> dict[str, Any]:
    """Return non-authority metadata for the local joint driver."""

    return {
        "schema": Z8_JOINT_VARIATIONAL_DRIVER_SCHEMA,
        "local_axis": "[macOS-MLX research-signal]",
        "gradient_field": (
            "joint renderer + categorical allocation + argmax-boundary commitment + differentiable pre-coder rate proxy"
        ),
        "objective_terms": [
            "reconstruction",
            "real_segnet_teacher_surrogate",
            "real_posenet_teacher_surrogate",
            "joint_p18_p19_gradient_waterfill_guard",
            "expected_categorical_archive_rate_score",
            "argmax_commitment_loss",
        ],
        "joint_p18_p19_gradient_waterfill_contract": (build_z8_joint_p18_p19_gradient_waterfill_contract()),
        "archive_rate_weight": float(config.archive_rate_weight),
        "argmax_commitment_weight": float(config.argmax_commitment_weight),
        "rate_formula": (f"{CANONICAL_RATE_MULTIPLIER} * expected_categorical_bytes / {CANONICAL_RATE_DENOM_BYTES}"),
        "ste_boundary": "gumbel_softmax_argmax_indices_to_archive",
        "iterative_relinearization": ("long_training_canonical reruns value_and_grad at every MLX step"),
        "archive_export_enabled": bool(archive_export_enabled),
        "full_contest_scorer_backprop_status": (
            "not_claimed; MLX-local path uses real scorer teacher caches plus "
            "learnable heads, exact CPU/CUDA remains promotion authority"
        ),
        "implicit_dykstra_allocator_diff_status": "pending_follow_on",
    }


def _flatten_arrays(prefix: str, obj: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_arrays(child, value, out)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child = f"{prefix}.{index}" if prefix else str(index)
            _flatten_arrays(child, value, out)
    elif hasattr(obj, "shape"):
        out[prefix or "value"] = np.asarray(obj, dtype=np.float32)


def _argmax_indices_numpy(model: Any) -> list[np.ndarray]:
    import mlx.core as mx

    indices: list[np.ndarray] = []
    for logits in model.logits_per_level:
        idx = mx.argmax(logits, axis=-1).astype(mx.int32)
        mx.eval(idx)
        indices.append(np.asarray(idx, dtype=np.int32))
    return indices


def _render_pair_arrays_numpy(
    model: Any,
    *,
    chunk_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Render trained MLX output to NHWC ``[0, 1]`` arrays for archive export."""

    import mlx.core as mx

    cfg = model.cfg
    all_indices = _argmax_indices_numpy(model)
    frame0_chunks: list[np.ndarray] = []
    frame1_chunks: list[np.ndarray] = []
    for start in range(0, int(cfg.num_pairs), int(chunk_size)):
        end = min(start + int(chunk_size), int(cfg.num_pairs))
        batch_indices = [mx.array(level_idx[start:end].astype(np.int32)) for level_idx in all_indices]
        pair = model.forward_eval_from_indices(batch_indices)
        mx.eval(pair)
        pair_np = np.asarray(pair, dtype=np.float32)
        frame0 = np.transpose(pair_np[:, 0], (0, 2, 3, 1)) / 255.0
        frame1 = np.transpose(pair_np[:, 1], (0, 2, 3, 1)) / 255.0
        frame0_chunks.append(np.clip(frame0, 0.0, 1.0).astype(np.float16))
        frame1_chunks.append(np.clip(frame1, 0.0, 1.0).astype(np.float16))
    return np.concatenate(frame0_chunks, axis=0), np.concatenate(frame1_chunks, axis=0)


def export_trained_mlx_rendered_z8hpc1_archive(
    model: Any,
    output_dir: Path,
) -> tuple[Path, str, int] | None:
    """Emit a byte-closed Z8HPC1 archive from trained MLX rendered output.

    The renderer state itself is still not a portable inflate runtime. This
    export therefore records the trained rendered frames into the Z8HPC1
    archive grammar, includes the MLX state and argmax indices as consumed
    stack-context bytes, and leaves score authority false until receiver proof
    plus exact-axis evaluation land.
    """

    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
        parse_archive,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
        export_z8hpc1_archive_bytes,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_custody import (
        build_z8_runtime_custody_contract,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame0, frame1 = _render_pair_arrays_numpy(model)
    binding = build_canonical_quadruple_binding_from_z8_config(model.cfg)
    base_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding,
        frame0.astype(np.float32),
        frame1.astype(np.float32),
    )
    parsed = parse_archive(base_bytes)

    decoder_state_dict: dict[str, np.ndarray] = {}
    _flatten_arrays("mlx_state", model.parameters(), decoder_state_dict)
    if not decoder_state_dict:
        digest = hashlib.sha256(base_bytes).digest()
        decoder_state_dict["mlx_state_fallback_digest"] = np.frombuffer(
            digest[:16],
            dtype=np.uint8,
        ).astype(np.float32)

    meta = dict(parsed.meta)
    meta["trained_mlx_rendered_output_export"] = {
        "schema": Z8_TRAINED_MLX_RENDERED_OUTPUT_EXPORT_SCHEMA,
        "rendered_output_archive_bound": True,
        "renderer_state_portable_runtime_ready": False,
        "argmax_indices_from_trained_logits": True,
    }
    meta["runtime_custody_contract"] = build_z8_runtime_custody_contract(
        source="z8_m12a_trained_mlx_rendered_output_export",
        section_name_style="archive_parser",
        archive_bound_candidate_package_emitted=True,
        trained_mlx_renderer_archive_export_ready=False,
    )

    archive_bytes = pack_archive(
        decoder_state_dict=decoder_state_dict,
        per_level_category_indices=_argmax_indices_numpy(model),
        wavelet_coeffs_blob=parsed.wavelet_coeffs_blob,
        wyner_ziv_top_blob=parsed.wyner_ziv_top_blob,
        dreamer_state_dict={
            "trained_rendered_output_digest": np.frombuffer(
                hashlib.sha256(base_bytes).digest()[:16],
                dtype=np.uint8,
            ).astype(np.float32),
        },
        meta=meta,
        num_levels=parsed.num_levels,
        num_groups_per_level=parsed.num_groups_per_level,
        num_categories_per_level=parsed.num_categories_per_level,
        num_pairs=parsed.num_pairs,
        decoder_latent_dim=parsed.decoder_latent_dim,
        base_channels=parsed.base_channels,
        wavelet_basis_id=parsed.wavelet_basis_id,
    )
    return export_z8hpc1_archive_bytes(
        archive_bytes,
        output_dir / "archive_bound_trained_mlx_rendered_output",
        emit_archive_bound_candidate_package=True,
        mlx_triage_argv=None,
    )


__all__ = [
    "Z8_JOINT_P18_P19_WATERFILL_CONTRACT_SCHEMA",
    "Z8_JOINT_VARIATIONAL_DRIVER_SCHEMA",
    "Z8JointVariationalDriverConfig",
    "argmax_commitment_loss",
    "build_z8_joint_p18_p19_gradient_waterfill_contract",
    "build_z8_joint_variational_driver_metadata",
    "build_z8_joint_variational_extra_loss_terms",
    "expected_categorical_archive_rate_score",
    "export_trained_mlx_rendered_z8hpc1_archive",
]
