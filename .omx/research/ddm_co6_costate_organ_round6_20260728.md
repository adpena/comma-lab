# ddm_co6 — COSTATE ORGAN ROUND 6: consume the 07-28 arc; resolve CO5's silent hold

UTC: 2026-07-28 · axis `[macOS-CPU advisory]` · actuation **NONE** · score_claim **false**
Pointer **0.1910828242 [contest-CPU submittable custody] UNMOVED** (stated first). This unit is
apparatus (organ SENSE/DECIDE + digest honesty); it moves no score. Commit `7c07097b42`.

Operator directive 07-28: *"Need to improve the costate controller and organ as well."* Per the
#247 de-orphan law, every change EXTENDS the canonical costate surfaces in place; nothing is
rebuilt beside them. Organ remains `_dev` / ADVISORY / actuation NONE; heavy/paid stays operator-GO.

## The measured staleness (verified from the digest BEFORE state)

`tools/costate_digest.py` emitted, pre-arc:
- `POINTER 0.18804 [contest-CPU] UNMOVED — everything below is means.` (leads with the banked
  **NON-SUBMISSION** anchor as if it were our frontier)
- `DDM-next: j_paint_dv1_persistent_ground rank=1`
- `DDM-duty: J_paint > R6_rehearsal > DDM_iteration_curves`
- `DDM-CO5: active=0/4 held=4 freshness=[fresh] gate=PREMISE_FALSIFIED_CT1_DELTA_S_PER_HOUR_GATE_NOT_SATISFIED`

Per the staleness-confound law (freshness-at-consumption), an advisory organ ranking from a world
the 07-28 arc has moved past is DEFAULT-HARMFUL × SILENT for every future session that reads it.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, MEMORY.md (07-28 CURRENT-STATE rows; #247/#303/#426/#516/#522 organ
  lineage; the pointer-only + Frontier-scores-are-pointer-only + "off is a tracked queue" laws)
- charter `co6_charter.md`; `docs/operating_manual_craft_handoff.md`
- organ surfaces `src/tac/ddm_costate_organ.py`, `src/tac/ddm_campaign_costate.py`,
  `src/tac/ddm_campaign_evidence_join.py`, `tools/costate_digest.py`
- registered band law `src/tac/canonical_equations/ddm_pp1_correction_stream_position_band_20260728.py`
- committed 07-28 arc: `ddm_fd1_family_d_gn_description_engine_20260728.md` (+DAG, +`ddm_fd1_gn_engine_smoke_20260728.json`),
  `ddm_pp1_direct_partition_pricing_20260728.md` (+DAG, +`ddm_pp1_band_lemma_receipt_20260728.json`),
  `ddm_rp1_rangeA_cell_realized_probe_20260728.md` (+DAG), `ddm_sp1_contour_support_coder_20260728.md`,
  `ddm_sc1_seeded_scene_carrier_20260728.md` (+DAG), `ddm_ch1_recursive_confound_pass_20260728.md`
- CT1 telemetry `ddm_ct1_campaign_telemetry_encode_20260725T111500Z/r6_rehearsal_receipt.json`
  + its codex findings (the ΔS/hour-declined verdict)
- `.omx/state/canonical_frontier_pointer.json` (SoT: cpu 0.18804 bank / effective 0.172 / cuda 0.20533)

## R1 — arc-evidence join (8 content-hashed advisory rows)

`_arc_evidence_rows(repo_root)` reads the committed 07-28 arc artifacts, content-hashes each, and
emits typed rows. All 8 present & git-tracked (reproducible from a fresh checkout).

| finding_id | crux_status | measured headline |
|---|---|---|
| fd1_zero_accept_window | LIVE_CRUX | 6 family-d GN candidates, 0 accepted; slope 0.000%/step; realized d_seg bit-identical 0.0702156745; 99.6% wall-clock = realized-acceptance pricing; capacity-routed |
| fd1_box_solve_s0_hold | SETTLED | S0 box-solve C1/C0 d_seg 1.077×; cells-hold flip 3.757e-4; margin gap 165×/166×; HOLD hardened |
| pp1_direct_partition_price | SETTLED | direct partition 173,616 B lossless; composed explicit route S≈0.189 > 0.172 bar; ≥350KB falsifier NOT reached |
| pp1_band_lemma | LAW | water 1.2731 B/flip; ρ_c 5.015e-4 measured; ρ_u 8.59e-4 derived; band ~[5e-4, 1e-2] |
| rp1_cells_hold | SETTLED | flipped/held pre-round margin 166.5× (0.0337 vs 5.6136); C1 d_seg 3.63e-4 = 2.39× q1; CELLS HOLD |
| sp1_support_race | SETTLED | support 444,394 B > LZMA 421,366 B @ ρ=0.864%; min lossy S=0.27999; FLOOR_DEAD |
| sc1_ep_rank1_pose | SOLVED | e_p SVD frac[0]=0.9986 rank-1; AR-int5 2,039 B (~2KB); pose feasibility-bounded, NOT binding |
| ch1_confound_pass | APPARATUS | 15 rows: 7 CLEAN / 8 SEAM-NAMED / 0 CONFIRMED-CONFOUND (apparatus-validity metadata) |

**NO-FAKE boundaries honored:** (1) fd1's wall-clock figure is **99.6%** (the "99.73%" cited loosely
elsewhere is the RATE share of S=194.43 in the ms2r_r3 memo — a different quantity; NOT folded).
(2) There is **no committed fd2 artifact** and **no "quantum cure"** in any committed 07-28 file
(only uncommitted SSD run logs). The join therefore grounds the realization crux in fd1's committed
**two-rung ladder** (Rung1 grow shared/cross-pair DOF + #383-dual pose-null; Rung2 token-grid +
partition→pixel renderer ≤64KB scorer-in-loop), NOT an uncommitted "fd2" arm.

## R2 — band-position SENSE law (registered equation wired into the organ)

`_band_position(repo_root)` reads the **live base error rate** (d_seg = fraction of flipped sites)
from the committed CT1 exact-n600 verdict (content-hashed) and evaluates the REGISTERED
`ddm_pp1_correction_stream_position_band_v1`. Confirmed: **W_joint d_seg is a DENSITY, not S-units**
(the joint action at that checkpoint is 26.19 in S-units — a different number). The band `density`
argument is defined over the same site-fraction, so the mapping is exact.

- live base d_seg = **0.0705192** → regime **EXPLODE** (above the 1e-2 band upper) → correction-class
  duty multiplier **0.0** ("ABOVE_BAND_SUPPORT_COST_EXPLODES_LOWER_BASE_FIRST"). sp1's 444 KB support
  wall is the measured teeth.
- the fd2 target ≤ ρ_c = 5.0e-4 would be **CONCEDE** (below-band; ship no correction stream) — the
  committed pp1 lower edge; PR130's native 2.97e-4 is the below-band no-correction rail.

The organ now KNOWS where the live base sits and modulates correction-class duties accordingly.

## R3 — refreshed duty queue (DERIVED, not hand-ordered; before/after)

`_refreshed_duties(legacy, arc_rows, band_position)` recomputes the head by an **elimination
argument**, each leg cited to a measured arc row:
1. band-position places the base above the rational band → correction-class duties score 0 (sp1);
2. fd1's zero-accept window seals fixed-capacity GN descent → that duty is exhausted;
3. rp1 cells-hold (166×) + fd1 S0 box-hold settle the cell-space realization (J_paint) premise.
The surviving lever is the capacity/parametrization ladder (fd1 Rung1/Rung2), which becomes the head.

| rank | BEFORE (pre-arc `_duties`, hand-ordered) | AFTER (`_duties_refreshed`, derived) |
|---|---|---|
| 1 | J_paint (cell-space receiver realization) | **FD1_REALIZATION_LADDER_MATERIALIZATION** |
| 2 | R6_rehearsal | R6_rehearsal |
| 3 | DDM_iteration_curves | DDM_iteration_curves |
| demoted | — | J_paint → SETTLED_CELL_SPACE_SUBSUMED_INTO_MATERIALIZATION (rp1+fd1 box-hold) |
| demoted | — | correction/support/explicit-residual → BAND_DEAD_ABOVE_BAND_EXPLODE (band+sp1) |

The pre-arc scheduler line is preserved but relabelled `DDM-next[pre-arc-scheduler]` (its dv1/g3/g4
grounding is untouched — not fabricated away); the refreshed line is the authoritative head.
A regression test (`test_refreshed_duty_not_hand_ordered_reacts_to_band`) proves the ladder is NOT
elevated when the base is in-band + descent not exhausted — i.e. the ranking is a real function of
the inputs, not a fixed order.

## R4 — CO5 re-adjudication (silent hold → tracked named-gate queue)

Re-read CT1 (landed 2026-07-25, **after** the original gate string). Its codex findings state
verbatim it **deliberately declines** to fabricate the campaign ΔS/hour (only steps 0/1/50 have
exact-n600 verdict custody; the batch-local trace is explicitly NOT n600). So the backtest is
**NOT mis-evaluated** — all 4 enhancements are **genuinely unsatisfied**. The fix is to replace the
silent collective `PREMISE_FALSIFIED` with **per-enhancement RE_PREMISE named producer gates**
(disposition DERIVED from backtest + named-gate availability) + a ranked duty-to-measure queue.

| enhancement | disposition | named producer gate | why held |
|---|---|---|---|
| compression_progress_per_effort | RE_PREMISE (queue #1) | CT1V2_SURFACES_TWO_PLUS_EXACT_N600_S_ENDPOINTS_WITH_WALL_CLOCK_AND_BYTE_IDENTITY | CT1 ran 3 verdict steps [0,1,50] but surfaces only step 50 as a full S-row; unblocks regret |
| m34_per_state_dual_consistency | RE_PREMISE (queue #2) | CT1V2_EMITS_PER_STATE_M34_AND_ORGAN_DUAL_WITH_MEASURED_UNCERTAINTY | CT1 emits neither dual |
| pontryagin_bellman_transition_residual | RE_PREMISE (queue #3) | CT1V2_EMITS_ORDERED_ADJACENT_CAMPAIGN_COSTATES_AND_TRANSITION_JACOBIANS | CT1 surfaces 1 endpoint + cadence, no costate/Jacobian |
| regret_bounded_duty_allocation | RE_PREMISE (queue #4) | TYPED_FIRED_DUTY_HISTORY_LEDGER_PLUS_ACTIVE_COMPRESSION_PROGRESS_PER_EFFORT | 12 geometry-cure events are a count, not typed duty history; downstream of #1 |

Verdict: **ACTIVATE 0 · RE_PREMISE 4 · RETIRE 0.** "Off" is now a tracked, ranked, drainable queue
(dependency-topology ordered: compression_progress unblocks regret → ranks #1). No silent holds.

## R5 — digest honesty (pointer + rows)

| line | BEFORE | AFTER |
|---|---|---|
| pointer | `POINTER 0.18804 [contest-CPU] UNMOVED — everything below is means.` | `POINTER submittable 0.1910828 [contest-CPU custody] UNMOVED · effective bar 0.172 · 0.18804 = NON-SUBMISSION bank (borrowed PR128-on-PR110, sha 196acd18, 2026-07-12) — everything below is means.` |
| DDM-next | `j_paint_dv1_persistent_ground rank=1` | `DDM-next[pre-arc-scheduler]: j_paint_dv1_persistent_ground rank=1` |
| DDM-duty | `J_paint > R6_rehearsal > DDM_iteration_curves` | `DDM-duty[07-28-refreshed]: FD1_REALIZATION_LADDER_MATERIALIZATION > R6_rehearsal > DDM_iteration_curves; superseded pre-arc head J_paint (SETTLED_CELL_SPACE...)` |
| DDM-band | (absent) | `DDM-band: base d_seg=0.0705192 regime=EXPLODE (rho_c=0.000502, upper=0.01) -> corrections DEAD: ABOVE_BAND_SUPPORT_COST_EXPLODES_LOWER_BASE_FIRST` |
| DDM-arc | (absent) | `DDM-arc[07-28]: 8/8 rows fd1_zero_accept_window=LIVE_CRUX ... ch1_confound_pass=APPARATUS` |
| DDM-CO5 | `active=0/4 held=4 ... gate=PREMISE_FALSIFIED_...` | `active=0/4 re-premised=4 retired=0 ... gate=RE_PREMISED_TRACKED_QUEUE_4_NAMED_GATES next-gate=CT1V2_SURFACES_TWO_PLUS_EXACT_N600...` |

Pointer sourcing note: the submittable 0.1910828242 is carried as a labelled, waivered
(`# HISTORICAL_SCORE_LITERAL_OK`) fallback in the tool only WHILE the SoT's contest-CPU frontier is
the known borrowed bank (sha `196acd18` match); once a submittable row lands in the pointer JSON its
sha will not match and the digest reads the JSON value directly (auto-retiring the constant). The
effective bar 0.172 is read live from `effective_frontier.score`.

## Join-row counts + verification

- arc-evidence rows folded: **8/8** present, content-hashed to committed artifacts.
- band-position: 1 live base density (0.0705192) → 1 regime (EXPLODE) via the registered equation.
- refreshed duty: 3 ranked + 2 demoted (with cited bases).
- CO5: 4 enhancements → 4 RE_PREMISE named gates + a 4-row ranked duty-to-measure queue.
- tests: `src/tac/tests/test_ddm_co6_costate_organ_round6.py` (12 tests) PASS; sister suites
  `test_ddm_costate_organ.py` + `test_ddm_campaign_costate.py` (2 updated to the new CO5 contract)
  PASS — **45 passed** total. ruff F clean. Pre-existing worktree failures (`test_levelset_*`
  import-time pointer JSON; `test_costate_digest_ncde.py::test_section_omitted_on_short_telemetry`)
  are UNRELATED to this diff (confirmed identical on unmodified main; my diff has zero NCDE refs).

## Honest boundaries

- Organ stays **advisory**: actuation NONE on every new surface; no trainer/launcher/paid path touched.
- Every ranking change traces to a receipt-backed, content-hashed join row — never taste.
- The refreshed head names the fd1 **committed** realization ladder, not an uncommitted fd2 arm.
- The pointer **0.1910828242 [contest-CPU] UNMOVED** — this unit is means (apparatus), not the end.
