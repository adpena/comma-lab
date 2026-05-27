# SPDX-License-Identifier: MIT
"""Canonical-automated submission packet pipeline — public API.

Per operator NON-NEGOTIABLE 2026-05-26 verbatim:
*"Remember everything we had to do to clean up and properly bundle our
submission, let's make that canonical and automated moving forward"*
(9th standing directive) and the amendment
*"Remember contest compliance and bundling full compression script and
all and everything"* (FULL lifecycle: compression + compliance +
everything).

Phase 2 lands ``compression_pipeline`` (Layer 0 per Phase 1 audit
specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``).

Per the 11th ORDER-MATTERS standing directive: this Layer 0 is the
canonical FIRST encoder pipeline orchestrator — every future submission
phase (Phase 3 archive_grammar / Phase 4 builder / Phase 5 compliance /
Phase 6 paired_auth_eval / Phase 7 operator runbook CLI / Phase 8
Catalog #362 STRICT preflight gate / Phase 9 cathedral consumer /
Phase 10 PR111-candidate end-to-end regression) consumes
:class:`CompressionPipelineResult` directly.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: this package's API is canonical-frozen-dataclass-
return + canonical-Provenance-routing + 4-layer canonical-helper-pattern
sister of :mod:`tac.deploy.modal.call_id_ledger` (Catalog #245),
:mod:`tac.probe_outcomes_ledger` (Catalog #313), and
:mod:`tac.canonical_equations` (Catalog #344).

Per the 8th MLX-first numpy-portable standing directive: training routes
through MLX-first encoder on Apple Silicon; weights export to portable
``.npz`` for numpy-only inflate per HNeRV parity L4 (≤200 LOC, ≤2 deps).

Quick start::

    from tac.submission_packet import (
        CompressionPipelineResult,
        build_compression_pipeline,
    )

    result = build_compression_pipeline(
        lane_id="lane_pr111_candidate_20260601",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=Path("experiments/train_substrate_nscs06_v8_chroma_lut.py"),
        recipe_path=Path(".omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_local_apple_silicon_dispatch.yaml"),
        hardware_substrate="auto",
        qat_enabled=True,
        output_dir=Path("experiments/results/pr111_candidate"),
    )
    print(result.weights_export_path, result.weights_sha256[:12])

Discipline cross-references:
  * Catalog #245 / #313 / #344 / #354 / #355 canonical 4-layer pattern
  * Catalog #270 dispatch optimization protocol (Tier1 + Tier2 + Tier3 umbrella)
  * Catalog #146 contest-compliant inflate runtime template
  * Catalog #361 Modal artifact filter preserves output/submission/
  * Catalog #205 canonical select_inflate_device routing
  * Catalog #295 PYTHONPATH self-containment
  * Catalog #339 + #360 silent-no-spawn extinction
  * Catalog #190 hardware_substrate detection (NO false precision)
  * Catalog #226 canonical scorer-loss helper routing
  * Catalog #228 GTScorerCache F3 consumption
  * Catalog #323 canonical Provenance umbrella
  * Catalog #340 sister-checkpoint guard
  * Catalog #356 per-axis decomposition (if Tier B available)
  * Catalog #365 + #366 + #367 just-landed STRICT gates respected
  * CLAUDE.md "Beauty, simplicity, and developer experience"
  * CLAUDE.md "Subagent coherence-by-default"
  * CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
"""
from __future__ import annotations

from tac.submission_packet.compression_pipeline import (
    CANONICAL_EQUATION_ID,
    COMPRESSION_PIPELINE_SCHEMA_VERSION,
    HardwareSubstrateClass,
    PHASE_2_LAYER_VERSION,
    CompressionPipelineError,
    CompressionPipelineResult,
    PerAxisPredictedBand,
    build_compression_pipeline,
    classify_hardware_substrate_for_dispatch,
    derive_compression_pipeline_provenance,
    validate_recipe_trainer_pair,
    verify_compression_pipeline_protocol_complete,
)

__all__ = [
    "CANONICAL_EQUATION_ID",
    "COMPRESSION_PIPELINE_SCHEMA_VERSION",
    "HardwareSubstrateClass",
    "PHASE_2_LAYER_VERSION",
    "CompressionPipelineError",
    "CompressionPipelineResult",
    "PerAxisPredictedBand",
    "build_compression_pipeline",
    "classify_hardware_substrate_for_dispatch",
    "derive_compression_pipeline_provenance",
    "validate_recipe_trainer_pair",
    "verify_compression_pipeline_protocol_complete",
]
