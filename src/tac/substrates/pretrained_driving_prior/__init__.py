# SPDX-License-Identifier: MIT
"""Pre-trained driving prior substrate (DP1).

Distills a TINY (5-10 KB) frozen dashcam-statistical codebook from publicly
available driving datasets (Comma2k19 MIT; BDD100K BSD-3 + dataset-images opt-in;
Waymo SKIPPED per non-commercial restriction), then uses the codebook as a soft
prior during contest-video score-aware overfit. The per-pair int8 residual
encodes the contest-specific delta from the prior.

**Strategic context** (operator 2026-05-13 directive): the substrate is
DUAL-PURPOSE — contest-score side-lane AND production-deployment-shaped
contest entry. The codebook's shape is designed so future Comma edge devices
can compute LOCAL deltas and contribute upstream via federated aggregation;
the inflate-time consumer is identical whether the codebook came from offline
distillation or a federated aggregation update.

**Predicted contest-CPU score: [0.175, 0.190]** ``[time-traveler-prediction]``
(NOT a score claim — score authority requires CUDA + CPU paired auth eval on
1:1 contest-CI hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA"). MEDIUM-EV at the contest; HIGH-EV for production-deployment alignment.

**Composition with the time-traveler substrate**: this substrate registers as
a "prior-class atom" — the codebook can be composed orthogonally with the
time-traveler L5 substrate's world-model + foveation + dynamics + per-pair
residual decomposition. Both substrates share the score-domain Lagrangian
contract (Catalog #164 ``score_pair_components``) so composition is a typed
sum of the two archive grammars under a wrapper.

**Catalog #124 archive-grammar 8 fields** (declared inline so AST walker sees):

* ``archive_grammar``: monolithic single-file ``0.bin`` (HNeRV parity L3)
* ``parser_section_manifest``: DP1 header + 4 length-prefixed sections
  (codebook + renderer state_dict + per-pair int8 residual + meta JSON)
* ``inflate_runtime_loc_budget``: <= 200 LOC (substantive)
* ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 <= 2 deps)
* ``export_format``: DP1 monolithic single-zip-member ``0.bin``
* ``score_aware_loss``: ``DrivingPriorScoreAwareLoss`` runs eval-roundtrip +
  Atick-Redlich cooperative-receiver + soft codebook prior
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity L7)
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte

Cross-references:

* Design memo: ``.omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md``
  §2.3 (NASA Goddard PASS-AI prior) + §6 #2 (Time-Traveler pre-trained prior)
* Sister substrate (composition reference): ``tac.substrates.time_traveler_l5_autonomy``
* Sister substrate (residual-sidecar precedent): ``tac.substrates.a1_plus_wavelet_residual``
* Canonical scorer-input contract: ``tac.substrates.score_aware_common``
* Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``
* Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``

Lane: ``lane_pretrained_driving_prior_lane_scaffold_20260513``
"""

from tac.substrates.pretrained_driving_prior.architecture import (
    EVAL_HW,
    DrivingPriorRenderer,
    DrivingPriorRendererConfig,
)
from tac.substrates.pretrained_driving_prior.archive import (
    DP1_HEADER_FMT,
    DP1_HEADER_SIZE,
    DP1_MAGIC,
    DP1_SCHEMA_VERSION,
    DP1_SECTION_ROLES,
    DrivingPriorArchive,
    build_readiness_manifest,
    compose_procedural_archive,
    pack_archive,
    parse_archive,
    parse_dp1_archive_bytes,
)
from tac.substrates.pretrained_driving_prior.codebook import (
    CODEBOOK_TOTAL_TARGET_BYTES_MAX,
    CODEBOOK_TOTAL_TARGET_BYTES_MIN,
    LANE_CURVATURE_PCA_SHAPE,
    ROAD_PLANE_BASIS_SHAPE,
    SKY_HORIZON_PROFILE_SHAPE,
    VEHICLE_APPEARANCE_BASIS_SHAPE,
    DashcamCodebook,
    codebook_to_torch_tensors,
    deterministic_zero_codebook,
    parse_codebook,
    serialize_codebook,
    validate_codebook,
)
from tac.substrates.pretrained_driving_prior.composition import (
    DPCOMP_HEADER_SIZE,
    DPCOMP_MAGIC,
    DPCOMP_SCHEMA_VERSION,
    ComposedArchive,
    compose_from_files,
    compose_with,
    decompose,
    known_base_substrates,
    verify_composition,
)
from tac.substrates.pretrained_driving_prior.dataset_source import (
    DP1DatasetSource,
    build_dp1_dataset_source,
    collect_local_video_manifest,
    final_step_chunk_ids,
    final_step_chunk_sha256s,
    is_sha256_hex,
)
from tac.substrates.pretrained_driving_prior.distillation import (
    Comma2k19FrameIterator,
    ContestVideoLeakageError,
    DistillationConfig,
    aggregate_local_codebooks,
    check_no_contest_video_leakage,
    distill_codebook,
    write_codebook_to_disk,
)
from tac.substrates.pretrained_driving_prior.distillation_procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    PROCEDURAL_CODEBOOK_DTYPE_DEFAULT,
    PROCEDURAL_CODEBOOK_SHAPE_DEFAULT,
    PROCEDURAL_SEED_SIZE_BYTES,
    ProceduralVariantConfig,
    ProceduralVariantError,
    compose_with_procedural_codebook,
    derive_procedural_codebook_replacement,
    verify_procedural_codebook_in_domain,
    verify_seed_mutation_changes_codebook_bytes,
)

PROCEDURAL_VARIANT_AVAILABLE: bool = True
"""Flag indicating the DP1 procedural-codebook replacement variant is
landed and importable. Sister recipe checkers (operator-authorize harness,
canonical autopilot ranker) may import this flag for capability detection.

Set to ``True`` at scaffold landing 2026-05-20 per
``feedback_dp1_procedural_codebook_paired_smoke_pre_dispatch_design_landed_20260520.md``
OP-ROUTABLE #2 (BUILD the variant trainer extension).
"""
from tac.substrates.pretrained_driving_prior.local_chunk_cache import (
    COMMA2K19_LICENSE_SPDX,
    COMMA2K19_REPO_URL,
    DEFAULT_CHUNK_MANIFEST,
    Comma2k19CacheError,
    Comma2k19ChunkManifestEntry,
    Comma2k19LocalCache,
    DiskBudgetExceededError,
    DownloadIntegrityError,
    default_cache_dir,
    verify_chunk_sha256,
)
from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    CANONICAL_SOURCE_URL as STREAMER_CANONICAL_SOURCE_URL,
)
from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    DATASET_LICENSE as STREAMER_DATASET_LICENSE,
)
from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    DEFAULT_LOG_DIR as STREAMER_DEFAULT_LOG_DIR,
)
from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    DEFAULT_RAM_BUFFER_GB as STREAMER_DEFAULT_RAM_BUFFER_GB,
)
from tac.substrates.pretrained_driving_prior.local_chunk_streamer import (
    ChunkSpec,
    Comma2k19LocalStreamer,
    DynamicChunkingStrategy,
    SHA256MismatchError,
    StreamAccessRecord,
    StreamingError,
    replay_stream_log,
    summarize_stream_log,
)
from tac.substrates.pretrained_driving_prior.log_incremental_feeder import (
    LogIncrementalSchedule,
    ScheduleStepResult,
    codebook_pca_quality_metric,
    log_incremental_distillation,
    log_incremental_distillation_streaming,
)
from tac.substrates.pretrained_driving_prior.prior_application import (
    DashcamPriorLoss,
    PriorApplicationWeights,
)
from tac.substrates.pretrained_driving_prior.score_aware_loss import (
    DrivingPriorLossWeights,
    DrivingPriorScoreAwareLoss,
)

__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "CODEBOOK_TOTAL_TARGET_BYTES_MAX",
    "CODEBOOK_TOTAL_TARGET_BYTES_MIN",
    "COMMA2K19_LICENSE_SPDX",
    "COMMA2K19_REPO_URL",
    "DEFAULT_CHUNK_MANIFEST",
    "DP1_HEADER_FMT",
    "DP1_HEADER_SIZE",
    "DP1_MAGIC",
    "DP1_SCHEMA_VERSION",
    "DP1_SECTION_ROLES",
    "DPCOMP_HEADER_SIZE",
    "DPCOMP_MAGIC",
    "DPCOMP_SCHEMA_VERSION",
    "EVAL_HW",
    "LANE_CURVATURE_PCA_SHAPE",
    "PROCEDURAL_CODEBOOK_DTYPE_DEFAULT",
    "PROCEDURAL_CODEBOOK_SHAPE_DEFAULT",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "PROCEDURAL_VARIANT_AVAILABLE",
    "ROAD_PLANE_BASIS_SHAPE",
    "SKY_HORIZON_PROFILE_SHAPE",
    "STREAMER_CANONICAL_SOURCE_URL",
    "STREAMER_DATASET_LICENSE",
    "STREAMER_DEFAULT_LOG_DIR",
    "STREAMER_DEFAULT_RAM_BUFFER_GB",
    "VEHICLE_APPEARANCE_BASIS_SHAPE",
    "ChunkSpec",
    "Comma2k19CacheError",
    "Comma2k19ChunkManifestEntry",
    "Comma2k19FrameIterator",
    "Comma2k19LocalCache",
    "Comma2k19LocalStreamer",
    "ComposedArchive",
    "ContestVideoLeakageError",
    "DP1DatasetSource",
    "DashcamCodebook",
    "DashcamPriorLoss",
    "DiskBudgetExceededError",
    "DistillationConfig",
    "DownloadIntegrityError",
    "DrivingPriorArchive",
    "DrivingPriorLossWeights",
    "DrivingPriorRenderer",
    "DrivingPriorRendererConfig",
    "DrivingPriorScoreAwareLoss",
    "DynamicChunkingStrategy",
    "LogIncrementalSchedule",
    "PriorApplicationWeights",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "SHA256MismatchError",
    "ScheduleStepResult",
    "StreamAccessRecord",
    "StreamingError",
    "aggregate_local_codebooks",
    "build_dp1_dataset_source",
    "build_readiness_manifest",
    "check_no_contest_video_leakage",
    "codebook_pca_quality_metric",
    "codebook_to_torch_tensors",
    "collect_local_video_manifest",
    "compose_from_files",
    "compose_procedural_archive",
    "compose_with",
    "compose_with_procedural_codebook",
    "decompose",
    "default_cache_dir",
    "derive_procedural_codebook_replacement",
    "deterministic_zero_codebook",
    "distill_codebook",
    "final_step_chunk_ids",
    "final_step_chunk_sha256s",
    "is_sha256_hex",
    "known_base_substrates",
    "log_incremental_distillation",
    "log_incremental_distillation_streaming",
    "pack_archive",
    "parse_archive",
    "parse_codebook",
    "parse_dp1_archive_bytes",
    "replay_stream_log",
    "serialize_codebook",
    "summarize_stream_log",
    "validate_codebook",
    "verify_chunk_sha256",
    "verify_composition",
    "verify_procedural_codebook_in_domain",
    "verify_seed_mutation_changes_codebook_bytes",
    "write_codebook_to_disk",
]
