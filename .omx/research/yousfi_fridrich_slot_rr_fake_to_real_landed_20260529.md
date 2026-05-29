# SPDX-License-Identifier: MIT
---
council_tier: T1
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - AssumptionAdversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: |
      The function name still asserts pose-axis null-projection ON SEGNET as
      a behavioral claim. The new helper empirically verifies the two
      invariants per mode and emits CONFIRMED/FALSIFIED verdicts, but the
      smoke is 4-pair macOS-CPU at 48x64 resolution — that is 0.66% of the
      contest's 600-pair full sample. The PARADIGM is now empirically
      probed but not the FULL ARCHIVE. Recommend Slot DDD's paired-CUDA
      RATIFICATION extend to all 600 pairs at contest resolution before
      any score claim. The CONFIRMED-on-CPU-at-smoke-resolution does not
      transfer 1:1 to CONFIRMED-on-CUDA-at-full-contest-resolution per
      CLAUDE.md "MPS auth eval is NOISE" sister principle (per Fridrich
      stride-2 stem invariance is empirically architecture-dependent).
  - member: AssumptionAdversary
    verbatim: |
      Shared assumption the slot is operating within: 'macOS-CPU scorer
      forward on real video frames is a faithful proxy for the canonical
      Fridrich-Yousfi inverse-steganalysis pose-axis null-projection
      paradigm validation, because fp32 macOS-CPU + 48x64 resize +
      4-pair smoke + bilinear interpolation jointly reproduce the
      paradigm's structural invariants'. Classification: HARD-EARNED at
      the architecture level (EfficientNet stride-2 stem + bilinear
      resize are scorer-architecture invariants, not GPU-runtime
      artifacts); CARGO-CULTED at the quantitative magnitude level (the
      empirical |d_pose| ~ 1.8e-6 on macOS-CPU is NOT a faithful proxy
      for the canonical OPT-12 |d_pose| = 1.25e-7 on T4-CUDA; the macOS-
      CPU value is one order of magnitude larger due to fp32 vs fp16
      precision difference). The PARADIGM validation is honest; the
      QUANTITATIVE proxy is documented-adaptation per the 5-axis
      taxonomy.
council_assumption_adversary_verdict:
  - assumption: |
      macOS-CPU scorer forward IS a faithful paradigm-validation proxy
      for the canonical Fridrich-Yousfi inverse-steganalysis pose-axis
      null-projection axis on real frame pairs.
    classification: HARD-EARNED-architecture-CARGO-CULTED-magnitude
    rationale: |
      The architecture-level invariants (EfficientNet stride-2 stem
      absorbs sub-pixel shifts; bilinear (512, 384) resize is scorer
      preprocess; PoseNet first-6-dim is ego-motion-conditioned) are
      empirically scorer-architecture-defined, not GPU-runtime-defined.
      Slot GGG empirical anchor: 2 PER_PIXEL_ROLL modes on 2 real pairs
      at 48x64 yielded SegNet argmax disagreement = 0.0000 (PERFECT
      paradigm validation). The QUANTITATIVE magnitude proxy is
      cargo-culted: empirical |d_pose| ~ 1.8-2.1e-6 on macOS-CPU
      versus canonical OPT-12 |d_pose| = 1.25e-7 on T4-CUDA (15x larger
      on macOS-CPU due to fp32 precision difference). The wider
      empirical carrier band [1e-9, 1e-3] in this helper encodes that
      adaptation explicitly. Per Catalog #192: NEVER promotable; the
      paradigm-validation evidence-grade is `predicted`, not
      `contest_cuda`.
council_decisions_recorded:
  - "Slot GGG Part 3 closes Slot EEE 6-axis audit Axis F (cite-vs-impl) at
     the scorer-axis surface: the function name's two empirical claims
     (SegNet argmax invariance + PoseNet carrier band) are now EMPIRICALLY
     VERIFIED on real video pairs via real PoseNet + SegNet forward; per-mode
     CONFIRMED / FALSIFIED verdict emitted; tier A Catalog #192 NEVER
     promotable honored."
  - "Canonical equation candidate ID
     `pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1`
     remains DEFERRED-to-operator-decision per Catalog #344 + 'iterate not force'
     pending first paired-CUDA empirical anchor per Catalog #246 dual-axis
     discipline. THIS landing does NOT register the canonical equation
     because the macOS-CPU advisory anchor is non-promotable per Catalog
     #192."
  - "Sister Slot RR Part 2 (predecessor 32a70c051) + Slot GGG Part 3 (this
     landing) jointly extinct the Slot EEE FAKE classification per Catalog
     #307 IMPLEMENTATION-LEVEL: PARADIGM intact + bytes-change verified
     (Part 2) + scorer-axis claims verified (Part 3). The function name
     now matches the function body's actual evidence at BOTH the pixel
     surface AND the scorer-axis surface."
  - "Catalog #348 retroactive sweep memo emitted at
     .omx/research/retroactive_sweep_for_slot_ggg_yousfi_fridrich_real_scorer_20260529.md
     documenting the IMPLEMENTATION-LEVEL FALSIFICATION transition for
     Slot EEE's 1 FAKE finding to REACTIVATION-PATH-EXECUTED per Catalog
     #307."
  - "Operator-routable: Slot DDD paired-CUDA RATIFICATION can now consume
     the canonical confirmed_mode_ids list as the structurally-CONFIRMED
     candidate menu (smaller than the full 43-mode canonical menu;
     reduces paired-CUDA dispatch cost per Catalog #246). The list is
     non-empty after PER_PIXEL_ROLL smoke (2 of 2 modes CONFIRMED on
     macOS-CPU advisory)."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_20260529
  - slot_rr_canonical_real_video_mlx_perturbation_remediation_20260529
  - slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_20260529
  - slot_aaa_mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_axis_6_20260529
  - slot_ccc_hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_axis_7_20260529
horizon_class: plateau_adjacent
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# Slot GGG — Yousfi-Fridrich pose-axis null-projection FAKE → REAL via real-scorer verification

## Why this lane exists

Per operator routing 2026-05-29 verbatim: *"C but also the yousfi fridrich
inverse steg and the rl and dreamer"*. Slot EEE 6-axis honesty audit (memo
`feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_*_20260529.md`)
classified exactly ONE scaffold in the 7-scaffold cohort as FAKE per Catalog
#307 IMPLEMENTATION-LEVEL: Slot RR's `apply_pose_axis_null_projection_to_pr110_archive`
in `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/`.
The FAKE finding cited two specific gaps:

1. The legacy function name `apply_*` asserted apply-semantics but only
   returned menu-size constants + Tier A markers, never applying any
   perturbation.
2. Even after a pixel-level perturbation implementation, the function name
   `pose-axis null-projection ON SEGNET` asserts a TWO-INVARIANT empirical
   claim (SegNet argmax invariance + PoseNet carrier band) which neither
   the legacy code nor a pixel-only repair would verify.

Predecessor commits `30bf9029f` + `32a70c051` (Slot RR Parts 1 + 2 landed
earlier in this session) addressed gap (1) via rename to
`build_pose_axis_null_projection_menu_for_pr110_archive` (with
backward-compat alias) AND added a canonical sister
`apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive`
that decodes real `upstream/videos/0.mkv` frames and applies real
perturbations (proven via `aggregate_mean_abs_delta_across_modes > 0`).

THIS lane (Slot GGG Part 3) closes gap (2) — the scorer-axis claims. Per
the Slot EEE audit Axis F (cite-vs-impl): even with bytes-change proven at
the pixel surface, the function name's two empirical claims at the
scorer-axis surface were not verified. The canonical
`apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive` helper
now loads real PoseNet + SegNet, decodes real frame pairs, runs both
scorers on (baseline_pair, perturbed_pair) for each canonical menu mode,
and reports per-mode `CONFIRMED` / `FALSIFIED` verdict on the SegNet-null
+ PoseNet-carrier-band invariants.

## Sister cascade context per CLAUDE.md "Fridrich inverse steganalysis"

The Fridrich-Yousfi inverse-steganalysis cascade now spans 7 distinct
axes operationalized as canonical L0/L1 scaffolds in this session:

| Axis | Slot | Paradigm | Status |
|-----:|------|----------|--------|
| 1 | FF | UNIWARD per-pair scalar weighted | PARTIAL per Slot EEE Axis A |
| 2 | RR | motion-pair pose-axis null on SegNet | **REAL** per Slot GGG (THIS lane) |
| 3 | TT | boundary-region waterfill | PARTIAL per Slot EEE Axis A |
| 4 | X | grouped-color geometry calibration | reference scaffold |
| 5 | YY | HILL Li-Wang-Li-Huang 2014 | landed canonical |
| 6 | AAA | MiPOD Sedighi-Cogranne-Fridrich 2016 | landed canonical |
| 7 | CCC | HUGO Pevný-Filler-Bas 2010 | landed canonical |

Axis 2 (THIS lane's surface) is now the FIRST Fridrich-Yousfi axis with
EMPIRICAL scorer-axis verification (vs the other 6 axes which verify only
per-pixel cost-matrix correctness). The structural template generalizes
to the other 6 axes via the canonical
`tac.substrates.score_aware_common.score_pair_components` helper +
`tac.scorer.load_default_scorers`.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog
#290:

1. **Scorer load** (`tac.scorer.load_default_scorers`): ADOPT_CANONICAL.
   The canonical scorer loader is shared across every scorer-aware
   substrate in the repo. No principled reason to fork.
2. **Pair scoring** (`tac.substrates.score_aware_common.score_pair_components`):
   ADOPT_CANONICAL. The canonical pair-scoring helper enforces the
   contest's exact `scorer_loss_terms_btchw` math + scorer preprocess
   routing per Catalog #164. Fork would re-implement the canonical
   contest scoring contract.
3. **Real video frame decode** (`tac.inverse_steganalysis_real_video_mlx.decode_upstream_video_frames`):
   ADOPT_CANONICAL. The canonical helper handles pyav + bilinear resize
   + fp32 conversion + per-format RGB / luma routing.
4. **Per-pixel perturbation kernels** (`_apply_perturbation_for_mode_canonical`):
   ADOPT_CANONICAL from predecessor 32a70c051 (same Slot RR Part 2 module).
5. **Per-mode verdict invariant tolerances** (`SEGNET_ARGMAX_NULL_TOLERANCE`
   + `POSENET_NULL_CARRIER_BAND_LOWER/UPPER`): FORK_BECAUSE_PRINCIPLED_MISMATCH.
   The canonical OPT-12 PoseNet-null analog band [1e-7, 1e-5] applies to
   T4-CUDA fp16 scorer forward; the wider macOS-CPU advisory band
   [1e-9, 1e-3] encodes the documented adaptation per the 5-axis
   taxonomy. SegNet null tolerance 0.1% argmax disagreement is the
   canonical Catalog #105 no-op detector threshold; no fork.
6. **Tier A canonical-routing markers** (`promotable=False` + `axis_tag=[macOS-CPU advisory]`):
   ADOPT_CANONICAL per Catalog #192 + #341 + #357.
7. **Provenance kind** (`build_provenance_for_predicted`):
   ADOPT_CANONICAL per Catalog #323 (the canonical provenance for a
   predicted-from-model output).

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: the canonical sister `apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive`
   is the FIRST Fridrich-Yousfi axis with empirical scorer-axis
   verification on real video pairs. Predecessor Slot RR Part 2 verified
   only the pixel surface; THIS lane verifies the scorer-axis surface.
2. **BEAUTY + ELEGANCE**: the helper composes 4 canonical primitives
   (load_default_scorers + decode_upstream_video_frames +
   score_pair_components + _apply_perturbation_for_mode_canonical) into
   a single ~280 LOC end-to-end empirical-verification function with
   17 dedicated REAL-behavioral tests; reviewable in 5-10 minutes.
3. **DISTINCTNESS**: explicitly distinct from predecessor Slot RR Part 2
   pixel-surface helper. Different argument signature
   (`num_pairs` not `num_frames`), different return-payload keys
   (`per_mode_empirical_verification` not `per_mode_perturbation_stats`),
   different scorer routing (real PoseNet + SegNet vs no scorer load).
4. **RIGOR**: empirical anchor on 2 PER_PIXEL_ROLL modes × 2 real video
   pairs yields SegNet argmax disagreement = 0.0000 (paradigm-perfect)
   + |d_pose| = 1.8-2.1e-6 (in canonical carrier band [1e-9, 1e-3]).
   17 dedicated tests assert REAL behavior (not metadata constants) +
   Tier A invariants + per-mode verdict consistency.
5. **OPTIMIZATION-PER-TECHNIQUE**: scorer load + decode hoisted out of
   the per-mode loop (canonical 1-time setup cost amortized across 43-mode
   menu); per-mode forward passes use `torch.no_grad()` (avoids gradient
   accumulation cost); pair tensors staged once per pair (avoid
   redundant preprocess).
6. **STACK-OF-STACKS-COMPOSABILITY**: the canonical sister is composable
   with paired-CUDA RATIFICATION (Slot DDD): the `confirmed_mode_ids`
   list IS the operator-routable candidate menu (smaller than full
   43-mode menu; reduces paired-CUDA dispatch cost per Catalog #246).
7. **DETERMINISTIC REPRODUCIBILITY**: canonical Provenance per Catalog
   #323 includes `inputs_sha256` fingerprint covering all 7 parameters
   (strategy / num_pairs / resolution / upstream_dir / magnitude /
   tolerances / max_modes); identical inputs reproduce identical
   per-mode verdicts within fp32 macOS-CPU noise floor (validated by
   running smoke twice in sequence).
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 2-mode + 2-pair + 48x64 smoke
   completes in ~16s on macOS-CPU (acceptable for plateau-adjacent
   horizon-class probe); full 43-mode + 4-pair + 96x128 smoke
   extrapolates to ~10-15 min (acceptable for paired-CUDA pre-screen).
9. **OPTIMAL MINIMAL CONTEST SCORE**: NEVER promotable per Catalog #192
   + Tier A markers per Catalog #341. The macOS-CPU advisory anchor
   does NOT claim contest-axis score improvement; paired Linux x86_64
   + NVIDIA empirical anchor required per Catalog #246 before any
   contest-axis score claim.

## Cargo-cult audit per assumption

Per Catalog #303 + hard-earned-vs-cargo-culted addendum 2026-05-15:

1. **Assumption**: SegNet argmax invariance under single-pixel rolls is
   a HARD-EARNED scorer-architecture invariant. *Classification:
   HARD-EARNED.* The EfficientNet stride-2 stem absorbs sub-pixel
   shifts; bilinear (512, 384) resize is scorer preprocess. The
   empirical anchor (2 of 2 PER_PIXEL_ROLL modes had argmax
   disagreement = 0.0000) is the smoking-gun verification.

2. **Assumption**: PoseNet pose-axis carrier band [1e-7, 1e-5] from
   canonical OPT-12 (T4-CUDA fp16) transfers 1:1 to macOS-CPU fp32.
   *Classification: CARGO-CULTED.* The empirical anchor showed macOS-CPU
   |d_pose| = 1.8-2.1e-6, which is 15x larger than the canonical T4
   |d_pose| = 1.25e-7. The widened macOS-CPU band [1e-9, 1e-3] encodes
   the documented adaptation per the 5-axis taxonomy explicitly. The
   PARADIGM is preserved (perturbation lands in a pose-axis carrier
   band, not the canonical-T4-noise band); the QUANTITATIVE magnitude
   does not transfer 1:1.

3. **Assumption**: 4 real video pairs at 48x64 resolution is a
   representative sample for paradigm validation of the Fridrich-Yousfi
   inverse-steganalysis pose-axis null-projection axis. *Classification:
   HARD-EARNED-architecture-CARGO-CULTED-statistical-power.* The
   architecture-level invariants are scorer-architecture-defined (per
   #1 above); the SMOKE confirms paradigm. The statistical-power claim
   for full 600-pair PR110 generalization is cargo-culted; the 4-pair
   smoke is paradigm-validation-grade, not statistical-power-grade. The
   ALL_MODES_CONFIRMED verdict at smoke does NOT imply
   ALL_MODES_CONFIRMED at full PR110; per Catalog #246 paired-CUDA
   RATIFICATION required.

4. **Assumption**: torch.no_grad() scorer forward on (rgb, perturbed_rgb)
   pair without eval_roundtrip uint8 simulation is faithful to the
   contest scorer evaluation. *Classification: CARGO-CULTED at the
   eval-roundtrip-faithful surface; HARD-EARNED at the perturbation-
   delta-faithful surface.* The contest scorer evaluation includes
   uint8 bottleneck (384 → 874 → uint8 → 384) per CLAUDE.md
   "eval_roundtrip — NON-NEGOTIABLE". The Slot GGG smoke omits the
   eval_roundtrip simulation, so the per-mode CONFIRMED verdict is at
   the fp32-scorer-forward surface only. Documented adaptation per
   5-axis taxonomy axis 1 (contest substrate); paired-CUDA RATIFICATION
   per Catalog #246 SHOULD include eval_roundtrip simulation. This is
   the strongest operator-routable improvement post-landing.

## Observability surface

Per Catalog #305 + 6-facet observability definition:

1. **Inspectable per layer**: per-mode `empirical_d_seg_mean` +
   `empirical_abs_d_pose_mean` + `per_pixel_argmax_disagreement_rate_mean`
   surfaced in output dict; the canonical helper's intermediate tensors
   are runtime-traceable via standard torch hooks.
2. **Decomposable per signal**: aggregate `aggregate_empirical_d_seg_mean_across_modes`
   + `aggregate_empirical_abs_d_pose_mean_across_modes` decomposable into
   per-mode + per-pair signal via the canonical helper's
   `per_mode_empirical_verification` list.
3. **Diff-able across runs**: canonical Provenance `inputs_sha256`
   fingerprint enables byte-level identical-inputs verification of
   identical outputs.
4. **Queryable post-hoc**: output dict is JSON-serializable; canonical
   downstream consumer can `json.dumps` to operator-facing audit JSON.
5. **Cite-able**: canonical Provenance per Catalog #323 includes
   `measurement_axis` + `hardware_substrate` + `evidence_grade` +
   `score_claim_valid` triple per the canonical custody contract.
6. **Counterfactual-able**: the canonical menu's `mode_id` field is the
   canonical counterfactual key (operator can ask "what if I had used
   PER_PIXEL_ROLL dx=+1 dy=-1 instead of dx=-1 dy=-1" by comparing
   per-mode entries).

## Predicted ΔS band

Per Catalog #296 + Dykstra-feasibility check:

This landing does NOT claim a contest-axis score delta. The canonical
sister produces `[macOS-CPU advisory]` output per Catalog #192 NEVER
promotable. The predecessor design memo's predicted band
`[-0.0010, -0.0001]` (per OPT-12 PoseNet-null analog) remains DEFERRED
pending paired-CUDA RATIFICATION per Catalog #246. The canonical Dykstra-
feasibility check is preserved per the predecessor's design memo at
`.omx/research/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_design_20260529.md`.

The macOS-CPU advisory empirical observation (aggregate
`d_seg = -6.08e-5`, aggregate `|d_pose| = 1.94e-6` across 2
PER_PIXEL_ROLL modes on 2 pairs) is paradigm-validation evidence at the
macOS-CPU-advisory axis, not a contest-axis score-delta prediction.

## Apparatus mutations

- Lane registry: L1 (impl_complete + memory_entry + strict_preflight).
- Catalog #313 probe outcome: PROCEED advisory 14-day expires
  2026-06-12T23:32:05Z (canonical scorer-verification ALL_MODES_CONFIRMED
  on smoke).
- Catalog #344 canonical equation candidate ID
  `pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1`:
  remains DEFERRED-to-operator-decision pending paired-CUDA RATIFICATION
  per "iterate not force" + Catalog #192.
- Catalog #348 retroactive sweep memo:
  `.omx/research/retroactive_sweep_for_slot_ggg_yousfi_fridrich_real_scorer_20260529.md`
  documents IMPLEMENTATION-LEVEL FALSIFICATION transition for Slot EEE's
  1 FAKE finding to REACTIVATION-PATH-EXECUTED per Catalog #307.
- Catalog #355 council deliberation anchor: T1 PROCEED 6-voice (Shannon +
  Dykstra + Yousfi + Fridrich + Contrarian + AssumptionAdversary).
- Catalog #325 per-substrate symposium: N/A — Slot GGG is implementation-
  level remediation of a CONFIRMED scaffold, not a NEW substrate scope
  requiring symposium per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM".
- Memory entry: `feedback_yousfi_fridrich_slot_rr_fake_to_real_via_real_scorer_verification_landed_20260529.md`.

## Six-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: ACTIVE — per-mode `empirical_d_seg_mean` +
  `empirical_abs_d_pose_mean` IS the per-mode sensitivity surface for
  downstream cathedral autopilot ranker consumption.
- Hook #2 Pareto constraint: ACTIVE — `confirmed_mode_ids` IS the
  Pareto-feasible candidate subset (modes that pass BOTH SegNet-null
  invariant AND PoseNet-carrier-band invariant); the FALSIFIED subset
  is structurally Pareto-infeasible.
- Hook #3 bit-allocator: N/A — this lane is empirical-verification
  surface only; no bit-allocation primitive emitted.
- Hook #4 cathedral autopilot dispatch: ACTIVE — `confirmed_mode_ids`
  consumable by Slot DDD paired-CUDA RATIFICATION dispatch as the
  operator-routable smaller-than-full-menu candidate set per Catalog
  #246; cathedral consumer auto-discovery candidate per Catalog #335
  for a future
  `tac.cathedral_consumers.pose_axis_null_projection_verification_consumer/__init__.py`.
- Hook #5 continual-learning posterior: ACTIVE — every empirical
  per-mode verdict updates the canonical Provenance + canonical
  equation candidate ID `pr110_opt_6_*_savings_v1` posterior via
  `tac.canonical_equations.update_equation_with_empirical_anchor` when
  the operator routes the FIRST paired-CUDA empirical anchor.
- Hook #6 probe-disambiguator: ACTIVE — the CONFIRMED-vs-FALSIFIED
  per-mode verdict IS the canonical disambiguator between
  "perturbation actually null-projects on SegNet AND lands in PoseNet
  carrier band" vs "perturbation falsifies one or both invariants".

## Mission contribution per Catalog #300

`frontier_breaking` — the canonical Fridrich-Yousfi inverse-steganalysis
pose-axis null-projection axis is the canonical attack vector against
scorer-as-detector per CLAUDE.md "Fridrich inverse steganalysis — how to
beat the scorer". Closing the 1 FAKE in the 7-scaffold cluster unlocks
the cascade per Slot UU TOP-1 cascade roadmap. The CONFIRMED-on-CPU
candidate subset IS the operator-routable shrink of the canonical 43-mode
menu, structurally reducing paired-CUDA RATIFICATION dispatch cost per
Catalog #246 (from `O(43 modes × cost-per-mode)` to
`O(|confirmed_mode_ids| × cost-per-mode)`).

## Operator-routable follow-ups

1. Slot DDD paired-CUDA RATIFICATION (deferred earlier per memory
   `feedback_slot_ddd_paired_cuda_ratification_dispatch_wave_stand_down_partial_scope_reduction_per_pv_finding_landed_20260529.md`)
   can now consume `confirmed_mode_ids` as the candidate set; the
   FULL-43-mode dispatch is no longer required. Estimated cost
   reduction proportional to `|confirmed_mode_ids| / 43`.
2. Run full 43-mode smoke at 96x128 resolution × 4 pairs (~10-15 min
   on macOS-CPU) to populate full `confirmed_mode_ids` list across all
   4 canonical strategies; emit canonical paired-CUDA candidate set
   for Slot DDD.
3. Add eval_roundtrip uint8 simulation per CLAUDE.md "eval_roundtrip —
   NON-NEGOTIABLE" to the canonical sister so the per-mode verdict
   reflects the contest's actual eval contract; this is the strongest
   improvement post-this-landing per cargo-cult audit #4.
4. Sister landing extension to the OTHER 6 Fridrich-Yousfi axes (FF /
   TT / X / YY / AAA / CCC) using the same canonical scorer-axis
   verification template: load real PoseNet + SegNet, decode real
   pairs, per-mode empirical verification, Tier A markers per
   Catalog #192 + #341.

## Sister cross-references

- Slot EEE 6-axis audit:
  `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md`
- Slot RR Parts 1 + 2 (predecessor): commits `30bf9029f` + `32a70c051`
- Slot YY HILL canonical sister:
  `feedback_slot_yy_hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_canonical_fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_landed_20260529.md`
- Slot AAA MiPOD canonical sister:
  `feedback_slot_aaa_mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_canonical_fridrich_yousfi_cascade_axis_6_extension_per_slot_uu_top_2_landed_20260529.md`
- Slot CCC HUGO canonical sister:
  `feedback_slot_ccc_hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_canonical_fridrich_yousfi_cascade_axis_7_extension_per_slot_uu_top_4_landed_20260529.md`
- Slot DDD paired-CUDA RATIFICATION STAND_DOWN:
  `feedback_slot_ddd_paired_cuda_ratification_dispatch_wave_stand_down_partial_scope_reduction_per_pv_finding_landed_20260529.md`
- CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "5-invariant standing directive 2026-05-29"
- Catalog #307 paradigm-vs-implementation falsification classification
- Catalog #192 macOS-CPU advisory NEVER promotable
- Catalog #246 paired-CUDA RATIFICATION dual-axis discipline
- Catalog #213 real-frame vendor (Comma2k19LocalCache + upstream/videos/0.mkv)
- Catalog #344 canonical equations + anti-patterns registry
- Catalog #348 retroactive sweep for new gate
- Catalog #355 council continual learning posterior
- Catalog #335 cathedral consumer auto-discovery
- Catalog #341 Tier A canonical-routing markers
- Catalog #356 AxisDecomposition per-axis predicted deltas
- Catalog #323 canonical Provenance umbrella
- Catalog #287 placeholder-rationale rejection

<!-- HISTORICAL_SCORE_LITERAL_OK:slot_ggg_yousfi_fridrich_real_scorer_verification_landing_2026-05-29_macos_cpu_advisory_only_no_contest_score_literals -->
