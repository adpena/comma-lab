# SPDX-License-Identifier: MIT
"""tac.substrates.atw_codec_v1 — ATW Codec V1 (Atick-Tishby-Wyner Cooperative-Receiver Codec).

Per the 2026-05-15 grand reunion symposium Phase D Composite #1 (lines
727-770 of `feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md`),
the ATW codec composes THREE foundational information-theoretic frameworks
into ONE training Lagrangian:

* **Atick & Redlich (1990)** — cooperative-receiver theorem: the optimal
  compressor for a known cooperative receiver R minimizes ``H(X | f_R(X))``
  rather than ``H(X)``. For the contest scorer R = SegNet+PoseNet, this
  becomes the Z4 cooperative-receiver loss (β_seg · d_seg + γ_pose · sqrt(d_pose)).
* **Tishby, Pereira & Bialek (1999)** — Information Bottleneck Lagrangian:
  ``L_IB = I(X;T) - β · I(T;Y)`` with closed-form encoder ``p*(t|x) ∝
  p(t) · exp(β · ∫ p(y|x) log p(y|t) dy)``. For the contest β=4.
* **Wyner & Ziv (1976)** — source coding with side information at decoder:
  when decoder has side info ``S`` correlated with source ``X``, the
  rate-distortion function tightens from ``R(D)`` to ``R_WZ(D) = R_X|S(D) ≤
  R(D)``. For ATW, S = published scorer weights (compress-time);
  scorer-class-prior precomputed table (inflate-time).

The composite ATW Lagrangian:

::

    L_ATW = α · B(θ)/N                      (rate from archive bytes)
          + β_seg · d_seg(θ)                (Atick-Redlich SegNet term)
          + γ_pose · sqrt(d_pose(θ))        (Atick-Redlich PoseNet term)
          + κ_IB · I(T; Y_predicted)        (Tishby IB info-preservation)
          + λ_WZ · R_WZ_residual(t | t̂(s))  (Wyner-Ziv side-info residual)
          + λ_pixel · MSE(decoded, GT)      (Z3 baseline residual; default 0)

The defaults (κ_IB=0, λ_WZ=1, λ_pixel=0) recover the CLEAN ATW codec where
the PRIMARY mechanism is Wyner-Ziv side-info compression on per-pair latent
residuals. Knob-zero ablations recover the four corner regimes:

* (0, 0, 0)  → Atick-Redlich pure (= Z4 verbatim)
* (0, 1, 0)  → ATW canonical
* (0.1, 0, 0) → Tishby IB pure
* (0, 0, 1)  → Z3 baseline (pixel-MSE-only)

These four corners are the **probe-disambiguator** regime sweep (Catalog
#125 hook #6) that arbitrates {Atick-only, IB-only, WZ-only, classical}.

**Score movement**: unranked until the `H(latent | scorer_class)` probe and
paired smoke land. The grounded V1 latent-only rate-side bound is only about
``[-0.0027, -0.005]`` if the unmeasured 30-50% latent-byte-saving hypothesis
holds; larger score movement is not claimed by the scaffold.

**Composition with existing primitives**: STACKS on Z4-V2 (β-only branch)
and on A1 substrate (encoder/decoder/latent architecture inherited).
ATW feasibility region is a SUBSET of Z3+A1 polytope per Boyd's convex-
feasibility lens; this does not imply dominance without an empirical lower
score under the same archive/runtime/eval axis.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: ATW1 monolithic single-file ``0.bin`` with WZ side-info head;
  bytewise-distinct from Z3HP1/Z4CR1 grammars (different magic ``b"ATW1"``)
- ``parser_section_manifest``: ATW1 header (4-byte magic + 1-byte version +
  per-tensor config) + encoder_blob + decoder_blob + latent_residual_blob +
  wz_side_info_head_blob + meta_blob
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver per
  HNeRV parity L4 + L7 (encoder/decoder + latent dequant + WZ side-info
  predict + composition)
- ``runtime_dep_closure``: torch + brotli only (HNeRV L4 ≤2 deps)
- ``export_format``: ATW1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``ATWScoreAwareLoss`` routes through canonical
  ``score_pair_components_dispatch`` per Catalog #164; eval-roundtrip
  mandatory; composes Atick-Redlich + Tishby IB + Wyner-Ziv terms
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  ~400 LOC scaffold with three-knob composition (κ_IB / λ_WZ / λ_pixel)
- ``no_op_detector_planned``: WZ side-info head is structurally consumed
  at inflate time (z = z_residual + z_predicted_table[scorer_class_prior]);
  empirical no-op detector verifies non-trivial side-info contribution

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 2 council approval required to lift _full_main
NotImplementedError per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY")
canary_status: ``post_canary_dependent``
canary_dependency: ``lane_z4_cooperative_receiver_loss_step2_20260514``

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS in ATW":

1. **Wyner-Ziv side-info hypothesis** — ``z_residual`` rate savings
   dominate; the Atick-Redlich loss is auxiliary; ATW gain = WZ gain only.
2. **Tishby IB hypothesis** — the IB encoder ``p*(t|x)`` has lower-entropy
   convergence than Atick-Redlich's argmin alone; ATW gain = IB gain.

Probe sweeps (κ_IB, λ_WZ, λ_pixel) ∈ {(0,0,0), (0,1,0), (0.1,0,0), (0,0,1)}
at fixed (β_seg, γ_pose). The canonical four-corner ablation arbitrates
the regime-conditional verdict. Memo at
``tools/probe_atw_kappa_lambda_disambiguator.py`` (planned post Phase 2
dispatch approval; see lane evidence).

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — ATW per-tensor gradient norm (∂L_ATW/∂θ for each θ)
   IS the per-tensor importance signal for bit-allocator; register
   ``sensitivity_map.atw_codec_grad_v1`` (planned post-Stage-1).
2. **Pareto constraint** — adds ``WZ_residual_entropy ≤ ε_WZ`` AND
   ``IB_info_preserved ≥ ε_IB`` to the convex feasibility region intersected
   with A1+Z3 rate/distortion polytope; register ``tac.pareto.atw_codec_v1``.
3. **Bit-allocator hook** — per-tensor importance derived from ATW gradient
   norms; the WZ side-info head receives bit budget proportional to
   prediction quality; register ``bit_allocator.atw_wz_residual_v1``.
4. **Cathedral autopilot dispatch hook** — recipe registered warn-only at
   landing; gated by Catalog #167 smoke-before-full + Catalog #246 anchor-skip;
   ranker v2 (Catalog #219) receives ``literature_anchor=Atick-Redlich1990 +
   Tishby-Bialek1999 + Wyner-Ziv1976`` (-0.01 to -0.03 class-shift reward).
5. **Continual-learning posterior** — every ATW empirical anchor seeds the
   posterior with paired ``(L_atick, L_ib, L_wz, L_pixel)`` measurement;
   substantial MDL-density change from A1+Z3 anchors confirms or refutes
   the ATW hypothesis.
6. **Probe-disambiguator** — four-corner ablation IS the probe; see above.

Cross-references
----------------

- Grand reunion symposium 2026-05-15:
  ``feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md``
  (Composite #1 lines 727-770)
- Grand council evidence review 2026-05-15:
  ``feedback_grand_council_evidence_review_modal_failures_*_20260515.md``
  (Phase B top-3 + Z4 lambda=0 timeout root cause)
- Z4 sister substrate (β-only branch):
  ``src/tac/substrates/z4_cooperative_receiver_loss/``
- D4 sister substrate (Wyner-Ziv on frame_0):
  ``src/tac/substrates/d4_wyner_ziv_frame_0/``
- Canonical Atick-Redlich primitive ATW reuses:
  ``src/tac/codec/cooperative_receiver/atick_redlich.py``
- Wunderkind G1 substitution (scorer-as-decoder-side-info):
  ``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``
- Atick & Redlich (1990) "Towards a theory of early visual processing"
  Neural Computation 2(3):308-320
- Tishby, Pereira & Bialek (1999) "The information bottleneck method"
  Proc. 37th Allerton Conf.
- Wyner & Ziv (1976) "The rate-distortion function for source coding with
  side information at the decoder" IEEE Trans Info Theory 22(1):1-10

Lane: ``lane_atw_codec_design_v1_20260515``
"""

from tac.substrates.atw_codec_v1.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    ATWCodec,
    ATWCodecConfig,
)
from tac.substrates.atw_codec_v1.archive import (
    ATW1_HEADER_FMT,
    ATW1_HEADER_SIZE,
    ATW1_MAGIC,
    ATW1_SCHEMA_VERSION,
    ATW1_SECTION_ROLES,
    ATWCodecArchive,
    ATWCodecArchiveNumpy,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
    parse_atw1_archive_bytes,
)
from tac.substrates.atw_codec_v1.score_aware_loss import (
    ATWLossWeights,
    ATWScoreAwareLoss,
)

__all__ = [
    "ATW1_HEADER_FMT",
    "ATW1_HEADER_SIZE",
    "ATW1_MAGIC",
    "ATW1_SCHEMA_VERSION",
    "ATW1_SECTION_ROLES",
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "ATWCodec",
    "ATWCodecArchive",
    "ATWCodecArchiveNumpy",
    "ATWCodecConfig",
    "ATWLossWeights",
    "ATWScoreAwareLoss",
    "pack_archive",
    "parse_archive",
    "parse_archive_numpy",
    "parse_atw1_archive_bytes",
]
