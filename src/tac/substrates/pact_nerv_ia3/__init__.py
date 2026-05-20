# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_ia3 — Pact-NeRV-IA3 (substrate L0 SKETCH).

Stage 1 of the HYBRID staged path per PACT-NERV-DESIGN-SYMPOSIUM
(`.omx/research/council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z.md`,
commit `5371d4dd4`). Per the council verdict PROCEED_WITH_REVISIONS:
Stage 1 = Pact-NeRV-IA3 ($0.30 Modal T4 50-LOC γ-only rate-extremal probe;
the CHEAPEST possible empirical experiment per Hotz + Carmack staging
discipline) MUST land BEFORE Stage 2 (Pact-NeRV-A1 600-LOC pose+difficulty+class
triple-conditioning) which MUST land BEFORE Stages 3-5.

Literature anchor: Liu et al. 2022 *"IA3: Infused Adapter by Inhibiting and
Amplifying Inner Activations"*, arXiv:2205.05638 (canonical literature
reference per the FILM-FAMILY-RESEARCH Section 10 Recommendation #5 / Pact-NeRV-IA3
rate-extremal variant). The IA3 paper's central claim: element-wise learnable
γ rescaling (no β shift) is ~6x more parameter-efficient than full FiLM
γ+β while preserving expressiveness for most conditioning tasks. The
empirical question IA3 tests on our contest: does the β term carry
significant per-frame signal on our specific driving video?

Hypothesis (per operator's PACT-NERV symposium Stage 1 verdict): for the
contest rate term, γ-only halves conditioning bytes vs γ+β. The empirical
question is whether the β term carries significant per-frame signal on
our specific video. If IA3 ≈ FiLM (β-noise hypothesis), proceed to
Pact-NeRV-A1 with confidence. If IA3 << FiLM (β-signal hypothesis),
Pact-NeRV-FULL must include β. If IA3 >> FiLM (FiLM-overcapacity
hypothesis), the simpler IA3 IS Pact-NeRV-A2.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^24
       |
       v
    HNeRV-class base decoder (DepthSep + SIREN + PixelShuffle; mirrors ds_nerv
    + ego_nerv canonical sister patterns)
       |
       v
    For each upsample block:
        feature_map = ego-pose-conditioned IA3 γ-only modulation:
          γ(pose) = 1.0 + Linear(pose) [residual form per IA3 paper §3.2]
          feature_map = feature_map * γ.view(B, C, 1, 1)
        [NO β bias projection — this is THE distinguishing primitive vs FiLM]
       |
       v
    rgb_0 / rgb_1: 1x1 Conv on final feature map -> RGB (3 channels)

The IA3 γ-only modulation is parameterized by ego-pose ∈ R^6 (the contest's
canonical pose representation). γ_init=1.0 + Δ residual form ensures the
substrate behaves like the unconditioned base decoder at initialization
(per IA3 paper §3.2 zero-init discipline; sister of adaLN-Zero per
FILM-FAMILY-RESEARCH Section 5).

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

The L0 scaffold ships:
- Substrate architecture + IA3 γ-only modulation (~50 LOC core + scaffold)
- Archive grammar (PIA3 magic + monolithic 0.bin per HNeRV parity L3)
- Inflate runtime (≤150 LOC per HNeRV parity L4)
- Score-aware loss helper routing (Catalog #164 + Catalog #6 MANDATORY
  eval_roundtrip default)
- Test coverage for the canonical archive grammar + IA3 γ-only invariant

Council design memo:
    `.omx/research/pact_nerv_ia3_l0_scaffold_design_20260520T<UTC>.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 22-byte header) |
| L4 inflate ≤ 200 LOC (≤ 150 target), ≤ 2 deps | PASS (target ~130 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; outputs (T, 3, H, W) per HNeRV parity L5) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on ≤ 350 LOC | substrate_engineering exception (~600 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (patches diff yuv6 BEFORE scorer; eval_roundtrip MANDATORY) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED per Forbidden premature KILL) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PIA3)
    parser_section_manifest:   parse_archive() -> 6 sections (header
                               + base_decoder_blob (FP11 weights)
                               + ia3_gamma_proj_blob (IA3 γ projection
                               weights; logical subset)
                               + latents_blob + meta_blob + ego_pose_blob)
    inflate_runtime_loc_budget: ≤ 150 LOC (PR101 GOLD reference)
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli (Quantizr canonical) for weights;
                               int16 latents + raw ego-pose tensor
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~600 LOC (substrate_engineering tag); IA3
                               modulation adds ~50 LOC over sister boost_nerv
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation
                               smoke; γ_proj weight perturbation MUST change
                               rendered frames empirically

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):

- IA3 γ-only modulation = HARD-EARNED-LITERATURE (Liu 2205.05638 demonstrates
  ~6x parameter-efficiency vs full FiLM; the empirical question on OUR
  contest is whether the parameter-efficiency translates to score-axis
  improvement after the rate-distortion tradeoff).
- γ_init = 1.0 (residual form) = HARD-EARNED (IA3 paper §3.2 zero-init
  discipline; sister of adaLN-Zero zero-init per FILM-FAMILY-RESEARCH §5).
- pose_dim = 6 = HARD-EARNED (contest canonical pose representation; matches
  upstream PoseNet's first 6 dims per `upstream/modules.py`).
- Per-block modulation (one γ_proj per upsample block) = CARGO-CULTED at L0
  (sweep at L1: alternatives = final-block-only, every-other-block, scalar
  global rescaling). Per FILM-FAMILY-RESEARCH §8.6: multi-layer is
  HARD-EARNED-EMPIRICALLY-SUPERIOR for video-temporal conditioning per
  TeNeRV + HNeRV ablation.
- HNeRV-class base decoder = HARD-EARNED (PR101 GOLD baseline; the IA3
  modulation is the bolt-on under test, not the base).
- Shared γ_proj across (frame_0, frame_1) of a pair = CARGO-CULTED at L0
  (alternative: per-frame γ_proj doubles head count; cheap variant first
  per Stage 1 discipline).

Operator HYBRID Stage 1 citation (council PROCEED_WITH_REVISIONS, 25 of 27
attendees voted PROCEED_WITH_REVISIONS, 2 voted PROCEED-unconditional;
Contrarian + Assumption-Adversary BOTH vetoed on PROCEED-unconditional
pending Section 13 HYBRID staged path):

    "op-routable #1: Stage 1 PRIORITY 1 — Pact-NeRV-IA3 $0.30 Modal T4
     single-primitive gamma-only rate-extremal smoke (50 LOC; cheapest
     empirical experiment per Hotz + Carmack staging discipline)"

Sister NeRV-family packages (12 total after this lands; sister of
WAVE-3-NERV-LITERATURE-L0-RESCOPED canonical pattern at commit `d9aaf7c13`):

- tc_nerv (TCNeRV)
- block_nerv (BlockNeRV)
- ff_nerv (FFNeRV)
- ds_nerv (DSNeRV; CANONICAL BASE for Pact-NeRV-IA3 IA3 modulation)
- hi_nerv (HiNeRV)
- sane_hnerv (SaneHNeRV)
- boost_nerv (BoostNeRV; sister L0 SCAFFOLD pattern this package mirrors)
- (sister e_nerv / ego_nerv / nervdc / nirvana / coin_plus_plus per CLAUDE.md
  catalog though may not all exist in repo yet)
"""

from .architecture import (
    PactNervIa3Config,
    PactNervIa3Substrate,
    IA3GammaOnlyModulation,
)
from .archive import (
    PactNervIa3Archive,
    PIA3_HEADER_SIZE,
    PIA3_MAGIC,
    PIA3_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import PactNervIa3ScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "IA3GammaOnlyModulation",
    "PIA3_HEADER_SIZE",
    "PIA3_MAGIC",
    "PIA3_SCHEMA_VERSION",
    "PactNervIa3Archive",
    "PactNervIa3Config",
    "PactNervIa3ScoreAwareLoss",
    "PactNervIa3Substrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
