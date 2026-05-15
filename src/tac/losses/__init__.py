# SPDX-License-Identifier: MIT
"""Reusable loss-function primitives for tac trainers.

The historical single-file ``tac.losses`` implementation now lives in
``tac.losses.core`` so newer loss submodules can coexist without shadowing
legacy imports such as ``from tac.losses import scorer_loss``.

Every name listed below in ``from .core import ( ... )`` is an explicit
top-level re-export so that AST-based dead-import scanners (preflight
``Check 13``) see each name at the package's ``__init__.py`` top level. The
prior ``from .core import *`` + runtime ``globals()`` injection produced the
same runtime symbol table but was invisible to AST scanners and silently
allowed dead-import drift on the 47 legacy callers.

The explicit list is the full public surface of ``tac.losses.core`` plus the
private helper ``_hwc_to_chw`` which is intentionally part of the
cross-trainer API (used by ``experiments/optimize_latent_codes.py`` and
``src/tac/experiments/train_renderer.py``). Adding a new public symbol to
``core.py`` requires adding it here too; ``Check 13`` will fail loudly if it
is missing from a caller's import line.
"""
from __future__ import annotations

from .cat_entropy_v2 import (
    CatEntropyV2Config,
    cat_entropy_v2,
)
from .core import (
    DEFAULT_SEGNET_NUM_CLASSES,
    DEFAULT_SINKHORN_BLUR,
    DEFAULT_SINKHORN_ITERS,
    DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    SEGMENTATION_SURROGATE_FISHER_RAO,
    SEGMENTATION_SURROGATE_SINKHORN,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    SINKHORN_MAX_BLUR,
    SINKHORN_MIN_BLUR,
    ScorerProxyDiscriminator,
    _apply_class_weights,
    _default_categorical_cost_matrix,
    _hwc_to_chw,
    band_orthogonality_loss,
    bhattacharyya_distance,
    boundary_aware_loss,
    boundary_aware_loss_cached,
    compute_boundary_mask,
    dual_saliency_reconstruction_loss,
    eval_scorer_loss,
    feature_matching_loss,
    focal_segnet_ste_loss,
    frequency_aware_loss,
    kl_distill_scorer_loss,
    kl_distill_segnet_only,
    output_decorrelation_loss,
    parse_class_weights_csv,
    per_class_seg_distortion,
    posenet_embedding_loss,
    renderer_scorer_loss,
    saliency_reconstruction_loss,
    saliency_reconstruction_loss_alpha,
    scorer_forward_pair,
    scorer_loss,
    scorer_loss_cached,
    scorer_loss_cached_with_aux,
    scorer_loss_pcgrad,
    scorer_loss_pcgrad_cached,
    scorer_loss_terms_btchw,
    scorer_loss_terms_cached_btchw,
    scorer_loss_with_aux,
    segnet_fisher_rao_per_pixel,
    segnet_kl_divergence_loss,
    segnet_ste_loss,
    segnet_surrogate_per_pixel,
    segnet_uncertainty_weighted_loss,
    sinkhorn_w2_mask_distortion_per_pixel,
    temperature_scorer_loss,
    train_scorer_proxy,
    uniward_quant_noise_loss,
)
from .u_die_kl import (
    DEFAULT_DIE_CACHE_INTERVAL,
    DEFAULT_KL_TEMPERATURE,
    DEFAULT_UNIWARD_EPSILON,
    UDIEKLConfig,
    UDIEKLLoss,
    compute_die_weight_map,
    compute_uniward_weight_map,
    kl_distill_segnet_term,
)

__all__ = [
    "DEFAULT_DIE_CACHE_INTERVAL",
    "DEFAULT_KL_TEMPERATURE",
    "DEFAULT_SEGNET_NUM_CLASSES",
    "DEFAULT_SINKHORN_BLUR",
    "DEFAULT_SINKHORN_ITERS",
    "DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK",
    "DEFAULT_UNIWARD_EPSILON",
    "SEGMENTATION_SURROGATE_FISHER_RAO",
    "SEGMENTATION_SURROGATE_SINKHORN",
    "SEGMENTATION_SURROGATE_SOFT_COSINE",
    "SINKHORN_MAX_BLUR",
    "SINKHORN_MIN_BLUR",
    "CatEntropyV2Config",
    "ScorerProxyDiscriminator",
    "UDIEKLConfig",
    "UDIEKLLoss",
    "_apply_class_weights",
    "_default_categorical_cost_matrix",
    "_hwc_to_chw",
    "band_orthogonality_loss",
    "bhattacharyya_distance",
    "boundary_aware_loss",
    "boundary_aware_loss_cached",
    "cat_entropy_v2",
    "compute_boundary_mask",
    "compute_die_weight_map",
    "compute_uniward_weight_map",
    "dual_saliency_reconstruction_loss",
    "eval_scorer_loss",
    "feature_matching_loss",
    "focal_segnet_ste_loss",
    "frequency_aware_loss",
    "kl_distill_scorer_loss",
    "kl_distill_segnet_only",
    "kl_distill_segnet_term",
    "output_decorrelation_loss",
    "parse_class_weights_csv",
    "per_class_seg_distortion",
    "posenet_embedding_loss",
    "renderer_scorer_loss",
    "saliency_reconstruction_loss",
    "saliency_reconstruction_loss_alpha",
    "scorer_forward_pair",
    "scorer_loss",
    "scorer_loss_cached",
    "scorer_loss_cached_with_aux",
    "scorer_loss_pcgrad",
    "scorer_loss_pcgrad_cached",
    "scorer_loss_terms_btchw",
    "scorer_loss_terms_cached_btchw",
    "scorer_loss_with_aux",
    "segnet_fisher_rao_per_pixel",
    "segnet_kl_divergence_loss",
    "segnet_ste_loss",
    "segnet_surrogate_per_pixel",
    "segnet_uncertainty_weighted_loss",
    "sinkhorn_w2_mask_distortion_per_pixel",
    "temperature_scorer_loss",
    "train_scorer_proxy",
    "uniward_quant_noise_loss",
]
