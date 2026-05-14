# SPDX-License-Identifier: MIT
"""Pre-trained driving-prior world-model scaffold for the 2032 lane.

This is a lightweight, scorer-free L0 substrate contract. It makes the future
cooperative-receiver lane buildable by fixing a typed config, deterministic
placeholder renderer/codebook, charged archive grammar for prior weights plus
residual bytes, and proxy-safe metadata. It is not a trainer, not exact eval,
and never claims score authority.
"""

from .archive import (
    DPWM_HEADER_BYTES,
    DPWM_MAGIC,
    DPWM_PROXY_EVIDENCE_GRADE,
    DPWM_SCHEMA_VERSION,
    ArchiveSection,
    DrivingPriorWorldModelArchive,
    DrivingPriorWorldModelError,
    build_readiness_manifest,
    deterministic_prior_weights,
    deterministic_residual_bytes,
    expected_prior_weight_bytes,
    expected_residual_bytes,
    pack_archive,
    parse_archive,
)
from .config import DEFAULT_LANE_ID, DrivingPriorWorldModelConfig
from .inflate import apply_world_model_archive, inflate_world_model_archive
from .renderer import render_prior_world_model

__all__ = [
    "DEFAULT_LANE_ID",
    "DPWM_HEADER_BYTES",
    "DPWM_MAGIC",
    "DPWM_PROXY_EVIDENCE_GRADE",
    "DPWM_SCHEMA_VERSION",
    "ArchiveSection",
    "DrivingPriorWorldModelArchive",
    "DrivingPriorWorldModelConfig",
    "DrivingPriorWorldModelError",
    "apply_world_model_archive",
    "build_readiness_manifest",
    "deterministic_prior_weights",
    "deterministic_residual_bytes",
    "expected_prior_weight_bytes",
    "expected_residual_bytes",
    "inflate_world_model_archive",
    "pack_archive",
    "parse_archive",
    "render_prior_world_model",
]
