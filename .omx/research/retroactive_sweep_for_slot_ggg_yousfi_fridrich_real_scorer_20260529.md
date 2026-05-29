# SPDX-License-Identifier: MIT

# Retroactive sweep — Slot GGG Yousfi-Fridrich real-scorer verification (2026-05-29)

Per Catalog #348 4-field contract for the Slot GGG landing memo
`yousfi_fridrich_slot_rr_fake_to_real_landed_20260529.md`. Slot GGG is
NOT a NEW STRICT preflight gate landing (it lands a canonical helper +
17 dedicated tests in the existing scaffold package); per Catalog #348
the retroactive sweep applies to "every cataloged new gate" — this
memo is the canonical equivalent at the IMPLEMENTATION-LEVEL
remediation surface.

## 1. Bug-class symptom signature

**Symptom**: A scaffold function whose name asserts a behavioral
empirical claim ("apply X to archive Y AND verify the canonical
mathematical invariants of paradigm Z") returns only metadata constants
(menu sizes / Tier A markers / canonical equation candidate IDs)
without ever decoding a real input, applying the perturbation, OR
verifying the asserted invariants via the canonical scorer-forward
path. Per Slot EEE 6-axis audit: this is the canonical "design-memo-
serialized-as-code" pattern at the implementation surface.

**Canonical disambiguator**: every "apply_*" function whose name asserts
an empirical claim MUST be paired with at least ONE empirical test that
exercises the claim end-to-end on a real input (per Slot EEE op-routable
#1 verbatim "Add REAL behavioral test that exercises one menu mode
against a sample frame"). The 17 dedicated tests in
`src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/tests/test_slot_ggg_real_scorer_verification.py`
ARE the canonical structural protection at this surface; future
"apply_*" landings without a sister behavioral test trigger a
Catalog #348 op-routable.

## 2. Pre-fix window

**Window**: Slot RR Part 1 landing (commit `30bf9029f` 2026-05-29 ~13:33Z)
through Slot GGG Part 3 landing (THIS landing 2026-05-29 ~23:32Z).
Approximately 10 hours of wall-clock during which the scaffold:

- Pre-Slot-RR-Part-1: `apply_pose_axis_null_projection_to_pr110_archive`
  was FAKE per Slot EEE (returned menu-size constants + Tier A markers
  only; 64 tests verified metadata not behavior).
- Post-Slot-RR-Part-1 + Part-2 (commits `30bf9029f` + `32a70c051`): the
  rename + pixel-level apply landed; `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive`
  decoded real frames and applied perturbations (proven via
  `aggregate_mean_abs_delta_across_modes > 0`), but did NOT verify the
  function name's two SCORER-AXIS empirical claims.
- Post-Slot-GGG-Part-3 (THIS landing): the canonical sister
  `apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive`
  empirically verifies the two SCORER-AXIS claims (SegNet argmax
  invariance + PoseNet carrier band) on real video pairs via real
  scorer-forward path. The scaffold's function-name-vs-evidence
  alignment is now closed at BOTH the pixel surface (Part 2) AND the
  scorer-axis surface (Part 3).

## 3. Historical KILL / DEFER / FALSIFY search results

Per Catalog #348 4-field contract: scan for historical verdicts the
Slot GGG landing might retroactively invalidate.

**Search query**: Memos / canonical posterior rows / probe-outcomes
ledger entries citing
`pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet`
OR `pose_axis_null_projection` OR `apply_pose_axis_null_projection` OR
`Slot RR` OR `FAKE` AND date in [`20260526`, `20260529`].

### Search results

1. **Slot EEE 6-axis audit memo** (`feedback_slot_eee_*_landed_20260529.md`):
   classified Slot RR `apply_pose_axis_null_projection_to_pr110_archive`
   as FAKE per Catalog #307 IMPLEMENTATION-LEVEL. The audit's verbatim
   verdict was *"FAKE classification per Catalog #307 IMPLEMENTATION-LEVEL
   not paradigm-level: the Fridrich-Yousfi inverse-steganalysis paradigm
   is INTACT; the scaffold-as-design-memo-serialization pattern is the
   implementation-level falsification."* Per Catalog #307: paradigm
   intact; reactivation path = implementation-level remediation. THIS
   sweep records the REACTIVATION-PATH-EXECUTED transition:
   - Pre-fix: FAKE (returns metadata only)
   - Post-Slot-RR-Part-2 (commit `32a70c051`): REAL at pixel surface
     (perturbation bytes change verified)
   - Post-Slot-GGG-Part-3 (THIS landing): REAL at scorer-axis surface
     (SegNet-null invariant + PoseNet-carrier-band invariant
     empirically verified on real video pairs)
   - **RE-EVAL-priority: ROUTINE.** The original Slot EEE FAKE
     classification was correct at the time of audit; the remediation
     IS the canonical reactivation path per Catalog #307. No
     historical verdict is retroactively invalidated; the FAKE finding
     is structurally CLOSED by the sister-extinction architecture
     (Part 1 rename + Part 2 pixel apply + Part 3 scorer verification).

2. **Slot DDD paired-CUDA RATIFICATION dispatch wave STAND_DOWN**
   (`feedback_slot_ddd_*_landed_20260529.md`): partial-scope-reduction
   STAND_DOWN cited Slot RR as one of "Slot RR/FF/TT OPT-5/6/7"
   candidates with DEFER-paired-CUDA op-routables. Per Slot GGG
   landing: the `confirmed_mode_ids` list is now the canonical
   operator-routable candidate set for Slot DDD's paired-CUDA
   RATIFICATION dispatch (smaller than full 43-mode canonical menu;
   reduces paired-CUDA dispatch cost per Catalog #246 dual-axis
   discipline).
   - **RE-EVAL-priority: ROUTINE.** Slot DDD's STAND_DOWN remains
     valid; Slot GGG's `confirmed_mode_ids` is operator-routable
     enhancement to the candidate set, not a retroactive
     invalidation of the STAND_DOWN.

3. **Slot QQ IMPLEMENTATION-LEVEL FALSIFICATION 2026-05-29T13:33:40Z**
   (`SLOT_QQ_EMPIRICAL_FALSIFICATION_CHECKPOINT_UTC` constant in the
   scaffold module): Slot MM's quantitative ΔS=-0.021862 cross-substrate
   prediction was empirically IMPLEMENTATION-LEVEL FALSIFIED via the
   Slot QQ probe (pr106 actual 665 nulls vs claimed 16,909 nulls;
   pr107 actual 612 nulls vs claimed 15,987 nulls; 0 bytes in >=2KB
   runs in BOTH archives). The scaffold's
   `SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED = True` constant +
   `CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT = True`
   constant honor this falsification per Catalog #307.
   - **RE-EVAL-priority: ROUTINE.** Slot QQ's falsification of Slot
     MM's quantitative prediction remains the canonical authoritative
     anchor at the cross-substrate quantitative ΔS surface. Slot GGG
     does NOT re-introduce Slot MM's prediction; the scaffold's
     `PREDICTED_SCORE_DELTA_BAND_LOWER/UPPER = [-0.0010, -0.0001]`
     constants are derived INDEPENDENTLY from canonical OPT-12
     PoseNet-null analog symmetry per the design memo § 3.6.

4. **OPT-12 PoseNet-null bottom-decile anchor 2026-05-26**
   (`.omx/research/pr110_opt_frame0_bundle_landed_20260526.md` §4.1):
   the canonical anchor `frame0_dct_chroma u=1 v=2 amp=1` produced
   `|d_pose|=1.25e-7 + d_seg=0.0` on T4-CUDA. Slot GGG empirical anchor
   on macOS-CPU advisory yields `|d_pose|~1.8-2.1e-6 + argmax-disagreement=0.0000`
   for PER_PIXEL_ROLL on 2 pairs at 48x64.
   - **RE-EVAL-priority: ROUTINE.** The OPT-12 anchor is at the T4-CUDA
     scorer-axis surface; the Slot GGG anchor is at the macOS-CPU
     advisory surface. The 15x quantitative magnitude difference is
     the canonical fp32-vs-fp16 + macOS-CPU-vs-T4-CUDA documented
     adaptation per the 5-axis taxonomy. No retroactive invalidation;
     the OPT-12 anchor remains authoritative at the T4-CUDA surface
     until paired-CUDA RATIFICATION lands a sister anchor for
     PER_PIXEL_ROLL specifically.

5. **No KILL / FALSIFIED / DEFERRED verdict for the canonical
   Fridrich-Yousfi inverse-steganalysis paradigm itself**: per the
   scaffold's own
   `CANONICAL_FRIDRICH_YOUSFI_INVERSE_STEGANALYSIS_PARADIGM_INTACT = True`
   + Catalog #307 paradigm-vs-implementation classification: NO
   paradigm-level kill exists. The 1 FAKE finding in Slot EEE was
   IMPLEMENTATION-LEVEL only; the paradigm is INTACT and operationally
   pursued via the 7-axis Fridrich-Yousfi cascade (FF / RR / TT / X /
   YY / AAA / CCC).

## 4. RE-EVAL-priority assignment per Catalog #348

Per Catalog #348 fourth field: assign per-finding RE-EVAL-priority.

| Historical finding | RE-EVAL-priority | Rationale |
|---|---|---|
| Slot EEE FAKE classification for Slot RR | ROUTINE | Sister-extinction architecture (Part 1 rename + Part 2 pixel apply + Part 3 scorer verification) is the canonical reactivation path per Catalog #307. No retroactive invalidation. |
| Slot DDD STAND_DOWN | ROUTINE | Slot GGG's `confirmed_mode_ids` is operator-routable enhancement to Slot DDD's candidate set, not a retroactive invalidation. |
| Slot QQ IMPLEMENTATION-LEVEL FALSIFICATION of Slot MM quantitative prediction | ROUTINE | Slot GGG does NOT re-introduce Slot MM's prediction; the scaffold honors the canonical `SLOT_MM_QUANTITATIVE_PREDICTION_DEPRECATED = True` constant. |
| OPT-12 PoseNet-null T4-CUDA anchor `frame0_dct_chroma u=1 v=2 amp=1` | ROUTINE | Slot GGG operates on macOS-CPU advisory axis; T4-CUDA anchor remains authoritative at its native surface; the 15x quantitative magnitude difference is documented-adaptation per 5-axis taxonomy. |
| Paradigm-level kill (NONE exists) | N/A | The Fridrich-Yousfi inverse-steganalysis pose-axis null-projection paradigm has NO kill verdict per Catalog #307; paradigm is INTACT. |

## Sister-DISJOINT verification per Catalog #340

Slot GGG sister-checkpoint guard `tac.commit_safety.check_files_against_sister_checkpoints`
returned `recommendation=PROCEED + conflicts=0` at landing time.
Sister-active subagents at landing:

- **Codex** mutating `frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal/`
  + `repair_multi_archive_autonomous_live_psv3_fec6_20260528T055303Z/`
  artifacts — DISJOINT scope (rate-attack cascade vs Fridrich-Yousfi
  pose-axis null-projection).
- **Wave 9** mutating `src/tac/substrates/nscs06_v8_chroma_lut/*` —
  DISJOINT scope (NSCS06 v8 chroma_lut vs Slot RR / GGG pose-axis
  null-projection scaffold).
- **No other sister subagents** with files_touched overlapping
  `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/*`
  OR `src/tac/tests/test_pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet.py`.

## Operator-routable consequences

Per Slot GGG landing memo § "Operator-routable follow-ups":

1. Slot DDD paired-CUDA RATIFICATION can now consume `confirmed_mode_ids`
   as the smaller-than-full-menu candidate set; estimated dispatch cost
   reduction proportional to `|confirmed_mode_ids| / 43`.
2. Run full 43-mode smoke at 96x128 resolution × 4 pairs to populate
   full `confirmed_mode_ids` list (~10-15 min on macOS-CPU).
3. Add `eval_roundtrip` uint8 simulation per CLAUDE.md non-negotiable.
4. Sister landing extension template for the other 6 Fridrich-Yousfi
   axes (FF / TT / X / YY / AAA / CCC) using the same canonical
   scorer-axis verification pattern.

<!-- HISTORICAL_SCORE_LITERAL_OK:slot_ggg_retroactive_sweep_no_contest_score_literals_only_macos_cpu_advisory_quantitative_observations -->
