"""``tac.paradigm_delta_epsilon_zeta`` — PARADIGM-δεζ Phase 1 scaffold.

This package implements the **Phase 1 Track 1 (T1)** end-to-end Ballé-hyperprior
+ 128K decoder + Lagrangian-ADMM coordinator scaffold. It is the missing module
that blocked T1 dispatch (see
``feedback_t1_balle_hyperprior_endtoend_BLOCKED_scaffold_missing_20260509.md``).

CLAUDE.md compliance — non-negotiable
-------------------------------------

1. **No score claims at module level**: every predicted score must be tagged
   ``[predicted; Phase 1 scaffold; not yet empirical]`` until a contest-CUDA
   anchor lands. Trainers in this package never set ``promotion_eligible=True``
   without ``contest_auth_eval.py`` exit code 0 + provenance.
2. **EMA at 0.997 with snapshot+restore**: every trainer instantiates EMA via
   ``tac.training.EMA`` (decay 0.997) and applies the shadow at evaluation
   only inside a snapshot+restore guard. Inference checkpoints are the
   ``ema.state_dict()`` shadow, never the live weights.
3. **eval_roundtrip mandatory**: every loss path simulates the contest's
   inflate roundtrip (384→874→uint8→384) so the proxy-auth gap stays
   bounded.
4. **Never MPS authoritative**: the trainer raises on ``--device mps`` and any
   MPS forward pass is tagged ``[macOS-CPU advisory only]`` per CLAUDE.md
   ``MPS-NOISE`` rule.
5. **Frozen A1 encoder**: T1 reuses the canonical A1 candidate (128KB-class
   latent table + lightweight encoder); the package exposes
   :func:`load_frozen_a1_encoder` which refuses to load anything other than
   the operator-designated canonical artifact under
   ``experiments/results/A1_canonical/`` (see
   ``.omx/state/canonical_a1_designation.md``).
6. **eval-time CUDA auth eval**: the trainer is a build-only deliverable in
   Phase 1. It MUST emit a deterministic archive that
   ``experiments/contest_auth_eval.py --device cuda`` can score; it MUST NOT
   replace or shadow that scorer.

Public API
----------

.. autosummary::
   :nosignatures:

   FrozenA1Encoder
   load_frozen_a1_encoder
   Decoder128K
   build_decoder_128k
   BalleHyperpriorWrapper
   build_balle_hyperprior
   JointLagrangianADMM
   PARADIGM_DELTA_EPS_ZETA_VERSION
"""
from __future__ import annotations

PARADIGM_DELTA_EPS_ZETA_VERSION = "0.1.0-phase1-scaffold"
"""Semantic version for the package; bumped each public-API mutation.

Phase 1 scaffold (this version): build-only modules + tests + dispatcher.
NO empirical score claim is associated with this version. The first
empirical contest-CUDA anchor will bump this to 0.2.0 with an
``empirical_anchor`` field in :data:`PARADIGM_DELTA_EPS_ZETA_PROVENANCE`.
"""

PARADIGM_DELTA_EPS_ZETA_PROVENANCE = {
    "schema_version": 1,
    "version": PARADIGM_DELTA_EPS_ZETA_VERSION,
    "phase": "1_scaffold_build_only",
    "tracks": ["T1_balle_128k_endtoend"],
    "empirical_anchor": None,
    "predicted_score_band": "[predicted; Phase 1 scaffold; not yet empirical]",
    "council_memo_refs": [
        "feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md",
        "feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md",
        "feedback_t1_balle_hyperprior_endtoend_BLOCKED_scaffold_missing_20260509.md",
    ],
    "compliance_tags": [
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "frozen_a1_encoder_designated",
        "score_tag_predicted_only",
    ],
}

# Lazy imports keep cold-import cheap for callers that only need the version
# string (e.g. preflight gates, lane registry serialization).
from tac.paradigm_delta_epsilon_zeta.frozen_a1_encoder import (  # noqa: E402
    FrozenA1Encoder,
    FrozenA1EncoderError,
    load_frozen_a1_encoder,
    A1_CANONICAL_DIR_NAME,
)
from tac.paradigm_delta_epsilon_zeta.decoder_128k import (  # noqa: E402
    Decoder128K,
    Decoder128KConfig,
    build_decoder_128k,
    decoder_128k_param_count,
)
from tac.paradigm_delta_epsilon_zeta.balle_hyperprior import (  # noqa: E402
    BalleHyperpriorWrapper,
    BalleHyperpriorConfig,
    build_balle_hyperprior,
)
from tac.paradigm_delta_epsilon_zeta.joint_lagrangian_admm import (  # noqa: E402
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    LagrangianStepResult,
)

__all__ = [
    "PARADIGM_DELTA_EPS_ZETA_VERSION",
    "PARADIGM_DELTA_EPS_ZETA_PROVENANCE",
    "FrozenA1Encoder",
    "FrozenA1EncoderError",
    "load_frozen_a1_encoder",
    "A1_CANONICAL_DIR_NAME",
    "Decoder128K",
    "Decoder128KConfig",
    "build_decoder_128k",
    "decoder_128k_param_count",
    "BalleHyperpriorWrapper",
    "BalleHyperpriorConfig",
    "build_balle_hyperprior",
    "JointLagrangianADMM",
    "JointLagrangianADMMConfig",
    "LagrangianStepResult",
]
