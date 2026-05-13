"""Time-Traveler L5 Autonomy substrate (TT5L).

Reverse-engineered from the operator's "time traveler from the future who solved
L5 self-driving on a single comma.ai unit" framing (2026-05-13 PAIR T directive).
The substrate composes five first-principles design moves into a single archive
grammar at 95-110 KB target size:

1. **Cooperative-receiver theorem** (Atick-Redlich 1990): score-aware loss
   maximizes MI(B; S(B)) where S = fixed contest scorer (SegNet + PoseNet),
   not generic image reconstruction.
2. **Predictive coding hierarchy** (Rao-Ballard 1999): per-pair side info is
   only the residual the world model cannot predict.
3. **Foveation matched to ego-motion** (Gibson 1950, Lee 1976, LAPose): a
   log-polar foveation grid centered on the camera focus-of-expansion gives
   5-10x effective resolution gain on score-relevant regions.
4. **Differentiable world model**: a small MLP renderer + Lie-algebra ego-pose
   dynamics prior encodes "physics" once (~25 KB) instead of pixels per frame.
5. **Sub-100K params + Tikhonov regularization**: ~35 KB decoder weights
   beats PR101's ~114 KB at predicted score 0.150-0.170.

**Predicted contest-CPU score: 0.150-0.170** ``[time-traveler-prediction]``
(NOT a score claim — score authority requires CUDA + CPU paired auth eval on
1:1 contest-CI hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA").

Catalog #124 STRICT archive-grammar 8 fields (declared inline so the AST
walker sees them):

- ``archive_grammar``: monolithic single-file ``0.bin`` (HNeRV parity L3)
- ``parser_section_manifest``: TT5L header + 4 length-prefixed sections
  (world model state_dict + per-pair side info + AC state + meta JSON)
- ``inflate_runtime_loc_budget``: <= 200 LOC substantive (HNeRV parity L4
  substrate-engineering waiver — full renderer including pose dynamics +
  foveation grid + differentiable physics)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 <= 2 deps)
- ``export_format``: TT5L monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``TimeTravelerScoreAwareLoss`` runs eval-roundtrip +
  Atick-Redlich cooperative-receiver + predictive-coding hierarchy
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7); the full renderer is substrate engineering not a bolt-on
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes; archive
  payload is structurally consumed by every section of inflate.py

Cross-references
----------------
- Design memo: ``.omx/research/time_traveler_architecture_reverse_engineered_20260513.md``
- Sister substrates (existing foveation precedent): ``tac.substrates.a1_plus_lapose``
- Sister substrates (residual sidecar precedent): ``tac.substrates.a1_plus_wavelet_residual``
- Sister substrates (frozen-base + adapter precedent): ``tac.substrates.pr95_lora_dora``
- Canonical scorer-input contract: ``tac.substrates.score_aware_common``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``

Lane: ``lane_time_traveler_l5_autonomy_substrate_20260513``
"""

from tac.substrates.time_traveler_l5_autonomy.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_SIDE_INFO_TARGET_BYTES,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
    TimeTravelerArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.time_traveler_l5_autonomy.score_aware_loss import (
    TimeTravelerLossWeights,
    TimeTravelerScoreAwareLoss,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_SIDE_INFO_TARGET_BYTES",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "TT5L_MAGIC",
    "TT5L_SCHEMA_VERSION",
    "TimeTravelerArchive",
    "TimeTravelerConfig",
    "TimeTravelerLossWeights",
    "TimeTravelerScoreAwareLoss",
    "TimeTravelerSubstrate",
    "pack_archive",
    "parse_archive",
]
