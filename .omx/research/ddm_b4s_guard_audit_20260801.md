---
schema: ddm_b4s_guard_audit.v1
date_utc: 2026-08-01
arm: ddm_b4s (task #807 follow-up — OWED-ITEM VERIFICATION + GUARD-THE-GUARD audit)
lane_id: "lane_ddm_b4s_burn4_20260731"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU/MLX advisory — telemetry re-read + statistical audit of a landed guard; NO launch, NO paid dispatch, NO pointer mutation]"
consumes: [ddm_b4s_burn4_charter_20260731.md (§6 OWED list),
  ddm_lg1_lane_guard_20260731.md (#808 — the CONSTRAIN-AND-PROTECT layer),
  ddm_b4s_20260731/{burn4.ALARM, window_0{1,2,3}_decision.json, burn4_endpoint_decision_MAIN.json},
  ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl (64 gate rows, the primary data),
  ddm_gd1_gate_estimator_audit_20260731.json (#817 — the gate-design bias measurement),
  ddm_xp1_20260731/xp1_verdict.json (the 0.12589 budget anchor),
  tools/supervise_ddm_b4s_burn4.py, src/tac/optimization/ddm_lp2_birth_completion.py]
consumers: [MAIN (endpoint adjudication + any burn-5 guard decision), ddm_bs1 (#815)]
tokens: [p0-ledger-ok]
---

# ddm_b4s — the 5 OWED items VERIFIED + the LANE_EROSION guard AUDITED

## §0 POINTER HONESTY FIRST

**The exact frontier did NOT move. `0.1910828242 [contest-CPU]` is UNMOVED.** This unit ran ZERO
training and ZERO scorer jobs. It re-read landed telemetry and audited a landed guard. Every number
below is `[macOS-CPU advisory]`; `score_claim=false`. This is apparatus verification — MEANS, not END.

## §1 STATE CORRECTION (read this first)

The charter §6 OWED list was written pre-fold. **All five items were already PAID at commit
`588a26c819`**, and the burn has since FIRED, run 3 windows over 4.435 h, and TERMINATED on
`burn4.ALARM` / `LANE_EROSION_ROLLBACK_EXCEEDS_CAP` at 2026-07-31T18:43:49Z. No process is live;
`burn4.done` does not exist. Any handoff still describing burn-4 as "HELD, awaiting the fold" is stale.

## §2 THE FIVE OWED ITEMS — verification (all PAID; each re-derived, not confirmed)

| item | verdict | evidence re-derived here |
|---|---|---|
| (a) verify lg1 flags via argparse | **PASS** | all 8 `--lane-guard*` flags exist at `train_tr1_partition_renderer_mlx.py:1335-1351`; names match the lg1 engagement spec exactly. No invented flag. |
| (b) fold + default-off byte-identity | **PASS (structural)** | every lane_guard path — *including the `from tac.optimization import lane_guard` import itself* (L1862) — is inside `if cfg.lane_guard:`. Default-off touches no import, no state, no RNG. See §5 for the honest limit on "byte-identity". |
| (c) `LG1_DUAL_ENGAGED` flipped + rollback wired | **PASS (fully, not half)** | `LG1_DUAL_ENGAGED = True` (L83). The path is complete end-to-end and *executed in production*: `burn4.ALARM` carries `rollback_target_ckpt`, `raise_lambda_init_to: 0.1`, and the λ≥1.0 escalation rung. Not a stub. |
| (d) λ_Lane budget MEASURED, not hand-set | **PASS** | `xp1_verdict.json base_per_class_S_units[1] = 0.12589` → `lane_guard.LANE_BUDGET_S_UNITS = 0.12589` → telemetry `budget_s_units = 0.125890` on all 68 gate rows. Traced end-to-end. |
| (e) re-seal + dry-run + re-smoke | **PASS, and MOOT** | ticket `6206cf56ede3a14d…` (20 levers). **Seal freshness re-tested at current HEAD: recompiles BIT-IDENTICAL** (81 argv tokens, hash match) despite the intervening DSL commit `52d1dbf3e0` — G1 invariance holds. Re-smoke (window_01) PASSED its ΔS gate (+4.24e-7 vs 3.0e-5 noise). READY_TO_FIRE is moot: it fired. |

## §3 THE GUARD AUDIT — the assigned question, and a different answer than expected

### 3.1 The aggregate-masking flaw it was built to catch: **NOT PRESENT** (clean)

`_lane_erosion_verdict` is genuinely Lane-scoped, not aggregate. The chain: it consumes
`key["fit"]` from `_p2_key` → `tools/run_ddm_lp2_birth_completion_key.py --qa91-inventory
qa91_erased_lane.json` → `tac.optimization.ddm_lp2_birth_completion` with `LANE_CLASS_INDEX = 1`,
which **fail-closes on a mismatch against the inventory's `betti0_gt_lane`** rather than trusting the
index. (Index 1 = Lane matches the canonical comma10k order; the module self-validates rather than
hardcoding blind.) The inversion is correct (`slope < -|ε|` vs the birth key's `slope <= ε`), ε is
DERIVED per-window (`t_crit · max(SE_ols, SE_quant)`), no hand-set threshold. And a Lane-eroding
window was genuinely **not extended** — window_03 fired and the supervisor stopped.

**On its assigned axis the guard passes.** The flaw is elsewhere.

### 3.2 The flaw that IS present: the guard draws a terminal verdict from an instrument that MAIN ruled inadmissible for terminal verdicts

`topology_per_class` is computed inside `realized_gate()` from `realized` over `gate_ids`
(`train_tr1…:949`) — i.e. on the **same 36-of-600-pair A1 gate subset** that gd1 (#817) audited, at
the same cadence, with the same `n_points = 5`.

`burn4_endpoint_decision_MAIN.json` §decision_2 says of that instrument, verbatim:

> "This is an L3 verdict-clearance case: the apparatus-validity precondition fails, so no
> load-bearing verdict is admissible from the gate alone."

...and withdrew its own reversal claim on exactly those grounds. Yet the LANE_EROSION guard, reading
the same gate, **terminated a 6-hour burn**. Two MEASURED gd1 defects apply directly, and the second
is worse on Lane than on d_seg:

- **Non-probability sample, unweighted mean.** `n_gate = 36`, block `[447,448,449,450]` carries
  weight 0.1111 in the estimator vs 0.00667 in the population = **16.67× over-weight**.
- **The block is Lane-poor, specifically.** gd1 `proxy_rows.lane_frac`: `block_mean` 0.0049044 vs
  `pop_mean` 0.0058546 = **−16.2% in Lane**; the gate over-states Lane fraction by **+3.34%**
  (`total_design_error_rel_pct` 3.337), of which **29% is removable** by Horvitz-Thompson
  re-weighting. gd1's memo names the residual explicitly: the effect on **"Lane predicates is NOT
  measured and is owed"** — and a Lane *betti0 count* is precisely a Lane predicate.
- **Aliasing.** `n_points=5` at `--gate-every 5` aliases a ~30-gate oscillation into a sign flip
  (gd1). The guard's window is ~1 alias period.

### 3.3 Measured false-positive rate: **25%**, on a series that was rising

I replayed the guard's own rule over its own 64-gate Lane `betti0_realized` series (windows 01-03,
ep644→945), using the guard's exact machinery (`_ols_slope_fit`, `_QUANT_ROUNDING_VAR`,
`DEFAULT_ALPHA_ONE_SIDED`):

| statistic | value |
|---|---|
| whole-burn OLS slope | **+0.9139 comp/gate — POSITIVE (Lane GROWING)** |
| net trend (first-5 mean → last-5 mean) | 522.6 → 575.2 = **+52.6 components, +10.1%** |
| series sd | 24.0 components |
| rolling 5-gate windows firing "eroding" | **15 / 60 = 25.0%** |
| **non-overlapping (independent) windows firing** | **3 / 12 = 25.0%** |
| nominal designed FP rate (`alpha_one_sided`) | 15.866% |
| window_03's firing window | begins at **the global maximum of the entire series** (586 @ ep929, position 59/64) |

The guard fires on a quarter of all windows of a series that grew 10%. Measured 25% exceeds the
nominal 15.9% because the OLS SE assumes independence while the series oscillates — so ε is too
small. And the one firing that mattered began at the global max, which is where regression-to-the-mean
most reliably manufactures a negative slope.

Cross-class corroboration that this is systemic, not a Lane fluke: the sister UNDRIV watch fired
`eroding: true` on window_02 for `net_betti0_realized_delta = −3` — and MAIN adjudicated it RETIRED
(commit `c72d1e7b75`, "UNDRIV_EROSION alarm RETIRED per MAIN adjudication — CONTINUE"). That is the
same rule producing a second false positive that a human had to overturn by hand.

### 3.4 The two protection legs gave OPPOSITE readings on the same window, and no rule reconciles them

The charter calls the topology guard and the lg1 level dual "COMPLEMENTARY... no double-counting."
On window_03 they contradicted each other:

| leg | window_03 reading |
|---|---|
| lg1 **LEVEL** dual (λ_Lane) | Lane is **excellent**: realized Lane S fell 0.122438 → 0.077481 (**−36.7%**), ending **38.5% below** the 0.12589 budget; `g = −0.0484` (deeply satisfied) at every one of 68 gates |
| b4s **TOPOLOGY** guard (betti0 slope) | Lane is **eroding** → ALARM, burn terminated |

Consequence: `burn4.ALARM` records `raise_lambda_init_to: 0.1` — the rollback-and-raise would have
**raised the dual's pressure on a constraint that was satisfied with 38% slack.** Raising λ against a
non-binding constraint injects Lane pressure the KKT conditions do not license and would distort the
primal against the other four classes. The guard can command a λ raise without ever consulting the
dual's own `g`. That is a real design gap, not a tuning nit.

### 3.5 Was the stop wrong?

**No — but it was right for a reason its own signal did not supply.** window_03 genuinely regressed
on the authority metric: n600 `full_confirm` 0.004067128 (w02) → 0.004148441 (w03), i.e. **+0.0081 S
worse**. Stopping was correct. The *overall* n600 delta is what justified it; the Lane-topology
signal did not. The guard was right by accident. A guard with a 25% FP rate that happened to fire on
a genuinely bad window has not been validated — it has been lucky once.

## §4 THE BURN'S ACTUAL RESULT (for MAIN's endpoint decision)

| state | n600 d_seg (`full_confirm`) | seg S = 100·d_seg | vs parent ep641 |
|---|---|---|---|
| parent ep641 (r1c) | 0.004264077 | 0.426408 | — |
| w01 ep665 (re-smoke) | 0.004277157 | 0.427716 | +0.0013 S |
| **w02 ep805 — BEST** | **0.004067128** | **0.406713** | **−0.0197 S** |
| w03 ep945 | 0.004148441 | 0.414844 | −0.0116 S |

**R6 verdict on the charter's pre-registered falsifier** ("endpoint n600 d_seg < 0.00426407708"):
**R6_PAYS** — best endpoint 0.004067128 < 0.004264077, a **−0.0197 S** seg-axis descent.

Two honest caveats on attributing that to R6:
1. **No control was run.** class_weight_lane=1.3 was never A/B'd against a 1.0 continuation at
   matched compute, and dw1/r1c already showed plain CE continuation descends. R6's *marginal*
   effect is not isolated; the falsifier as written ("beat the parent") conflates R6 with "more
   training."
2. **gc15's boundary artifact is unresolved.** MLX Adam `bias_correction=False` injects an LR spike
   worth ~16 epochs of free displacement at every window boundary, 81.7% inside the first 13 epochs.
   Part of this descent may be optimizer artifact. #815 (ddm_bs1) arm B' is the causal A/B and has
   first claim.

Pointer arithmetic, stated plainly: −0.0197 S on the seg axis against a live v4d total of 0.9639878
vs the bar 0.172141. **This does not approach the bar.** It is a MEANS.

## §5 WHAT IS NOT CLEAN (honest negatives)

- **"Byte-identity PROVEN" overstates what exists.** lg1's own memo is honest that the proof is
  structural + noise-floor-bounded, **not** bit-exact: the tr1 vehicle is rerun-nondeterministic with
  identical code+argv on both devices. The one hard receipt is that ep0 realized gate d_seg is
  bit-equal (0.5078303019205729) across all 8 OFF/ON runs. I re-read this; I did not re-run it. The
  charter's phrase "default-off proven byte-identity" should read "structurally gated + ep0-anchored".
- **The λ dual is UNTESTED.** λ_Lane stayed 0.0 across all 68 gates. That is *correct* KKT behavior
  (complementary slackness on a satisfied constraint), not a fake or an inert flag — the telemetry
  rows fire, `g` is computed live, born-mask and margin-floor addends were active independently. But
  the centerpiece of the CONSTRAIN-AND-PROTECT amendment **never engaged**, so its dynamics are
  unexercised. The budget was set at "don't get worse than ep641" and the run got much better
  immediately, so the constraint was a floor never approached.
- **The 25% FP figure is INSTANCE-scoped** to this series/vehicle/gate design. It is a strong
  falsifier of the current ε calibration; it does not falsify topology guarding as a family.
- **I did not fix the guard.** Deliberate: the burn is over so a fix changes nothing retroactively;
  gd1 records that changing the gate *set* is a "campaign-boundary move, pre-register" (HT
  re-weighting is the comparability-preserving alternative); and patching a guard I just audited,
  inside the same unit, is the built-instead-of-paid trap. The debt is named and routed, not built.
- **No endpoint manifest exists.** The charter's fire plan called for `burn4_endpoint_manifest.json`
  (R6_PAYS/R6_CLOSES + P re-estimate + QA80 staleness re-check). Only `burn4.ALARM` was written. The
  R6 verdict in §4 is my re-derivation from window receipts, not the planned artifact.

## §6 NAMED, UNBUILT next actions (for MAIN / operator — not executed here)

1. **Do not let the LANE_EROSION guard hold a terminal verdict in burn-5 as calibrated.** Three
   options, in ascending cost: (i) demote it to ALARM+HOLD-for-adjudication (what UNDRIV was already
   demoted to by hand); (ii) tighten `alpha_one_sided` and/or widen `DEFAULT_WINDOW_GATES` past the
   ~30-gate alias period; (iii) apply gd1's Horvitz-Thompson drop-in
   (`src/tac/optimization/ddm_gd1_gate_estimator.py` — same 36 renders, zero extra scorer cost,
   preserves comparability) so the Lane predicate stops inheriting the 16.67× block over-weight.
2. **Gate the rollback-and-raise on the dual's own `g`.** Raising λ when `g < 0` is not KKT-licensed.
   One-line precondition; must be pre-registered, not slipped in.
3. **Measure gd1's owed residual**: the gate-design bias on *Lane predicates* (betti0), which gd1
   explicitly left unmeasured and which this guard depends on entirely.
4. **Endpoint selection** remains MAIN's per `burn4_endpoint_decision_MAIN.json` §decision_3 —
   n600 on the gate-ranked shortlist (ep809/854/879), with w02 ep805 = 0.004067128 the incumbent.

## §7 verdict_scope ledger

- Guard FP rate 25%: **INSTANCE** (this series, this gate design, this vehicle) — falsifies the
  current ε calibration, not topology guarding as a family.
- Guard is Lane-scoped and correctly inverted: **VERIFIED at source** (`LANE_CLASS_INDEX=1`,
  fail-closed).
- Instrument-admissibility contradiction: **DERIVED** from MAIN's own §decision_2 + gd1's measured
  `lane_frac` rows.
- R6_PAYS: **MEASURED** on the charter's pre-registered falsifier, with the no-control and gc15
  boundary-artifact caveats attached (§4).
- All 5 OWED items: **VERIFIED PAID**, each re-derived from source, not confirmed from the memo.
- No prior negative re-opened. Nothing here is a score claim.

## STORES CONSULTED

CLAUDE.md · AGENTS.md · docs/operating_manual_craft_handoff.md ·
`.omx/research/ddm_b4s_burn4_charter_20260731.md` (§6 OWED) ·
`.omx/research/ddm_lg1_lane_guard_20260731.md` (#808) ·
`.omx/research/ddm_gd1_gate_estimator_audit_20260731.json` + `ddm_gd1_undecided_defaults_audit_20260731.md` (#817) ·
`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/` (ALARM, 3 window decisions, 2 MAIN adjudications,
3 telemetry.jsonl = 64 Lane topology gate rows, 5 tickets) ·
`/Volumes/VertigoDataTier/pact/ddm_xp1_20260731/xp1_verdict.json` ·
`tools/supervise_ddm_b4s_burn4.py` · `src/tac/optimization/{lane_guard,ddm_lp2_birth_completion}.py` ·
`experiments/train_tr1_partition_renderer_mlx.py` (argparse L1335-1351; `realized_gate`/`topology_per_class` L889-950; lane_guard gating L1833-1883, L2271-2279) ·
`src/tac/witness_dsl/spec_tr1_renderer_20260728.py` (3 lane-guard Lever factories) ·
memories: constants_are_poison · verdict_scope ladder · never_launch_weaker_state ·
built_instead_of_paid · audit_mode_is_not_roadmap_mode · negative_existence_claims.

**Pointer 0.1910828242 [contest-CPU] UNMOVED.** [no-triality] [p0-ledger-ok]
