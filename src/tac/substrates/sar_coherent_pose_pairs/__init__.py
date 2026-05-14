"""L2 SAR coherent pose-pair integration substrate (SARC).

Reverse-engineered from the alien-tech ``feedback_expert_team_signal_processing_
alien_tech_landed_20260513.md`` Lincoln Lab L2 entry (rank #1 EV/$ in the
floor-v3 commit ``27a7950fd``: $1, ΔS -0.0056, 30-60 min wall-clock). The
canonical Lincoln Lab SAR + pose-pair derivation lives at
``.omx/research/expert_team_signal_processing_lincoln_lab_20260513.md`` §2.

The substrate is a self-contained renderer + per-pair pose codec that exploits
**phase coherence across the temporal axis** of the contest's 600 non-overlapping
pose-pair stream. Per Cook & Bernfeld 1967 *Radar Signals* and Carrara, Goodman,
Majewski 1995 *Spotlight Synthetic Aperture Radar*: coherent integration over
N pulses synthesizes an effective aperture sqrt(N) larger than any single pulse
sees. For N=600 contest pairs that is a √600 ≈ 24× coherent SNR gain, which
translates (per the L2 ledger §2.2) to ~0.5·log2(600) ≈ 4.6 fewer bits per
pose-symbol — predicted ΔS -0.0056 vs the PR101 0.193 anchor.

The codec is structured as three first-principles design moves:

1. **SAR coherent pose codec** (Stage 1, ~3-5 KB encoded once): rather than
   storing each pair's 6-D pose independently, we store the per-pair *deltas*
   in a Fourier-coherent representation (sparse top-K rFFT over the 600-pair
   axis × 6 dims). Temporally smooth pose trajectories concentrate energy in
   low-frequency rFFT bins; sparse retention compresses the pose stream
   ~4-5× vs raw int8.
2. **SIREN-style frame renderer** (Stage 2, ~15-25 KB at FP16): a small
   coordinate-MLP that consumes (x, y, t, pose_codes_t) and emits an RGB pair.
   Sub-50K params; Tikhonov-regularized; reuses canonical SIREN init.
3. **Per-pair int8 RGB residual** (Stage 3, ~30-40 KB at 50-65 B/pair): the
   predictive-coding residual the SAR-coherent renderer cannot predict.

Total archive target: 50-70 KB. Predicted contest band [0.187, 0.190]
``[first-principles-bound, literature-prediction]`` — NOT a score claim.

**Catalog #124 archive-grammar 8 fields** declared inline so the AST walker
observes them:

* ``archive_grammar``: monolithic single-file ``0.bin`` (HNeRV parity L3)
* ``parser_section_manifest``: SARC header + 4 length-prefixed sections
  (renderer state_dict + sparse rFFT pose codec + per-pair RGB residual + meta JSON)
* ``inflate_runtime_loc_budget``: ≤ 200 LOC substrate-engineering waiver
  (HNeRV parity L4; full coordinate-MLP renderer + SAR pose decoder)
* ``runtime_dep_closure``: torch + numpy + brotli only (HNeRV parity L4 ≤ 3 deps)
* ``export_format``: SARC monolithic single-zip-member ``0.bin``
* ``score_aware_loss``: ``SARCoherentScoreAwareLoss`` runs eval-roundtrip +
  Atick-Redlich cooperative-receiver via canonical ``score_pair_components``
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity L7);
  full renderer is substrate engineering, not a bolt-on
* ``no_op_detector_planned``: emit/parse roundtrip preserves bytes byte-for-byte;
  archive payload is structurally consumed by every section of inflate.py

Cross-references
----------------
- Design memo §2: ``.omx/research/expert_team_signal_processing_lincoln_lab_20260513.md``
- Master alien-tech synthesis: ``feedback_expert_team_signal_processing_alien_tech_landed_20260513.md``
- Floor-v3: ``.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md``
- Sister substrate (sub-100K-param SIREN renderer precedent): ``tac.substrates.time_traveler_l5_autonomy``
- Canonical scorer-input contract: ``tac.substrates.score_aware_common``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``

Lane: ``lane_sar_coherent_pose_pairs_substrate_20260513``
"""

from tac.substrates.sar_coherent_pose_pairs.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    PER_PAIR_RESIDUAL_TARGET_BYTES,
    POSE_DIM,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    SARCoherentConfig,
    SARCoherentSubstrate,
    SARCoherentRenderer,
    SARCoherentPoseCodec,
)
from tac.substrates.sar_coherent_pose_pairs.archive import (
    SARC_MAGIC,
    SARC_SCHEMA_VERSION,
    SARCoherentArchive,
    pack_archive,
    parse_archive,
    quantize_per_pair_residual_int8,
    dequantize_per_pair_residual,
)
from tac.substrates.sar_coherent_pose_pairs.score_aware_loss import (
    SARCoherentLossWeights,
    SARCoherentScoreAwareLoss,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_RESIDUAL_TARGET_BYTES",
    "POSE_DIM",
    "SARC_MAGIC",
    "SARC_SCHEMA_VERSION",
    "SARCoherentArchive",
    "SARCoherentConfig",
    "SARCoherentLossWeights",
    "SARCoherentPoseCodec",
    "SARCoherentRenderer",
    "SARCoherentScoreAwareLoss",
    "SARCoherentSubstrate",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "dequantize_per_pair_residual",
    "pack_archive",
    "parse_archive",
    "quantize_per_pair_residual_int8",
]
