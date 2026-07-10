# T5 CRUCIBLE-3 — P3 RED-TEAM VERDICT vs THE DRAFT (v8 optimal final form) — 2026-07-09

**Phase:** P3 (RED-TEAM + MANDATORY PROVENANCE AUDIT). **Target:** `SYNTHESIS_DRAFT_v8_20260709.md`.
**Surface:** `crucible3_v8`. `[no-triality]` (P3b revision owns leg propagation) · `$0` · no GPU · run
dirs read-only · #205 STOPPED. Pointer contest-CPU **0.19110 UNMOVED** — everything here is
`[macOS advisory · research-signal · NON-PROMOTABLE]` MEANS.

**Fresh-eyes rule honored:** no seat/synthesis disposition is trusted because the draft adopted it. Every
load-bearing number RE-DERIVED from the primary artifact (code / registered equation / reports JSON /
grep), not memo-trusted. `review_status: self-executed, fresh-eyes-UNREVIEWED` (P5/P6 treat any adopted
disposition as a finding-producing round; every "fix" below is unreviewed new design — re-derive it).

**STORES CONSULTED:** the DRAFT (all 503 lines) · CONVENING · DELTA_GROUNDING · ORCHESTRATION_LEDGER · all
six positions S1–S6 · crucible-2 `P3_redteam_verdict` (the pattern) · `docs/operating_manual_craft_handoff.md`
· **PRIMARY CODE re-derived, not memo-trusted:** `src/tac/boundary_math/road_undriv_bulk_field.py` (module
header L1-63 · `bulk_boundary_byte_cost` L385-445 · `_horizon_profile` L448-467 · `horizon_poly_xi_byte_cost`
L470-555) · `src/tac/boundary_math/margin_conditional_residual.py` (L59-62 `WATERLINE_BYTES_PER_FLIP==1.27`
· `waterfill_select` · `conditional_position_bits`) · `src/tac/boundary_math/laguerre_logit_offset.py`
(`menon`/`damped_newton_ot_offsets`/`hard_cell_masses`/`solve_head_offsets` — NO median/quantile path) ·
`src/tac/witness_control/perclass_verdict.py` (L75 `flip_share_by_class` sensor EXISTS) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (L10098 `--pose-finish-engage-on
{muon,sigma_min_plateau}` EXISTS) · `src/tac/canonical_equations/v8_geometric_rate_decomposition_20260709.py`
(rate ledger) · `.omx/research/SPEC_v8_perclass_decomposition_20260708.md` (grep for "41") ·
`reports/delta_R_noise_floor.json` (δ_R). No new measurement taken; reading + arithmetic + grep only.

**verdict_scope discipline:** every finding carries a scope and, where a magnitude call, a measurement cite.
Remaining gap to sub-0.15 = **0.0411 S**; magnitudes quoted ÷0.0411.

---

## HEADLINE (answer-first)

**Findings: 9. WORST (near-break, REVISE): 1 · REVISE: 4 more · CARRY-AS-RISK: 1 · DISSOLVED: 1 · HELD: 2.
Provenance-audit FAILURES: 3 (+1 gap).** The draft is NOT a dead design — its rate provenance is clean and
its structural adjudications (T1 demote-SDF-field, T2 waterfill, phantom-41) survive audit on the numbers.
But its central DE-RISKING claim — that increment-1a is "the cheapest falsifiable decoupling bet, measurable
at Stage-A cost before any paint" — **does not measure what it claims**, and three "principled/wired"
supports (σ_cc′-structural, the flip-weighted-median b_c, the σ_min gate flag) are mis-attached to code that
does not implement them.

**THE SINGLE WORST FINDING (F1, REVISE — the 1a confound):** increment-1a is specified as `composite tropical
argmax → d_seg-through-R (n600, frozen CPU SegNet)` with **NO paint, NO P-C** (draft §A.4 / config
`measure_1a`). But **through-R d_seg is not measurable without RGB** — the frozen SegNet runs on `R(RGB)`,
not on labels. 1a excludes the 1b paint stage, so the only paint available is flat/canonical class-color —
whose d_seg is **floored by the MEASURED flat-paint failure (0.0064; draft §A.4 P-C item, increment-1 §9)**
and reflects **analytic-boundary quality, not the ∂φ_c/∂θ_{c'}=0 decoupling mechanism** the row is supposed
to falsify. Consequences:
1. A 1a "pass" (beats v7.5.2's Road d_seg) could be driven by near-perfect *analytic* boundary placement or
   by the flat-paint floor — NOT by the training-time decoupling that is v8's actual thesis. The measurement
   does not isolate the variable under test.
2. There is **no pre-registered numeric kill criterion.** The draft says "does NOT beat v7.5.2's per-class
   Road floor" but never pins the number, and v7.5.2's counter-force floor is itself UNMEASURED (run-1's
   Road 0.312 @ ep325 is the *pre-actuation birth arm*, DELTA §J). 1a is graded against a moving, unmeasured
   target with a confounded instrument.
3. The whole "1a de-risks the apparatus×5 opportunity cost before paint machinery is spent" (S3/D4/risk-6)
   rests on measuring the decoupling's d_seg *without* paint — which through-R does not permit.
**Fix latent in the draft:** state explicitly WHICH d_seg 1a measures — either (a) MASK-optimal label d_seg
(composite argmax vs GT argmax, NOT through-R; risk-3 warns mask-optimal ≠ score-optimal, so it is a fast
PROXY only), or (b) ORACLE-paint through-R (P-A-style real-frame texture = an UPPER bound, not the ship
number) — and pin a numeric kill threshold vs a MEASURED v7.5.2 Road floor at matched compute. As written,
1a's "falsifiable decoupling bet" is confounded; **missed by all six seats + P2.**

---

## §1 PROVENANCE AUDIT (load-bearing claims → primary artifact)

| claim (draft) | traced to | verdict |
|---|---|---|
| rate ledger 0.061 dominant / 0.140 complete / 0.079 residual + all per-edge rows | `v8_geometric_rate_decomposition_20260709.py` L21/22/30-33/74/75 — every value present; sums reconcile (0.0032+0.0275+0.00344+0.0202+0.007=0.0613; residual 0.04203+0.01892+0.01741=0.07836) | **HELD** (re-derived) |
| horizon dominant **0.0032 / 14.6×** is a REAL byte-close via `horizon_poly_xi_byte_cost` | function EXISTS L470; computes deg-3 `np.polyfit` on the topmost horizon arc, zlib delta-coded coeffs; returns `score_rate_contribution_MEASURED`; carries `residual_sidecar_owed=True` + scope_note "do NOT quote as the complete Road↔Undriv rate" | **HELD** (function real, caveat welded; numeric re-derivation needs execution — not run) |
| `bulk_boundary_byte_cost` = "full Road MASK (~707 B/frame, all Road edges), diagnostic ceiling" | function L385 measures the Road MASK (packbits + row-span RLE, brotli over temporal stack) — it is the Road *region* boundary. **It does NOT compute a "2228-px perimeter"** | **HELD-with-caveat** (S5-V3 correct: the px-scope framing is memo-only; draft A.5 corrected it) |
| "707 B/frame", "426 px = 19% of 2228-px Road perimeter" | NOT emitted by either on-disk byte-cost function; memo-sourced (increment-1 §1 / FEED-bytecost-sharpened) | **FAIL** (F6 — repeated as if from code) |
| "41 edges" in SPEC_v8 §1 is a PHANTOM (grep) | grep SPEC_v8: only "41% of Road's oracle flips" (L17, a percentage); L25 "one field per adjacency-graph EDGE" has NO count | **HELD** (D6 correct; grep-confirmed) |
| flip-weighted b_c = "wire `flip_share_by_class → target_masses`, a ~1-line objective swap in the BUILT damped-Newton solver" | sensor EXISTS (`perclass_verdict.py:75`); but `laguerre_logit_offset` has ONLY `menon` + `damped_newton_ot` (mass-match) — **NO median/quantile path**; mass-matching to flip_share ≠ S1's Hamming median | **FAIL** (F3 — mechanism ≠ label) |
| owed-item #4: "there is no `--pose-finish-engage-on` flag today (only `--pose-finish-start-epoch` + observer)" | flag EXISTS: `--pose-finish-engage-on {muon,sigma_min_plateau}` @L10098 — engages pose-finish on the σ_min conditioning EVENT | **FAIL** (F5 — stale code claim) |
| #226 `waterfill_select` produces the KKT flip-weighted operating point | `WATERLINE_BYTES_PER_FLIP = SEG_VALUE_PER_FLIP·BYTES_PER_SCORE == 1.27 B/flip`; ranks flips by net_value/cost; admits prefix clearing the water level with net_value>0 | **HELD** (F7 — real KKT; data is P-C-owed, not aspirational) |
| δ_R = 0.0196 | `reports/delta_R_noise_floor.json:27` = 0.019590 | **HELD** |
| banked R1 dxi 0.001610 / 0.127 / 7.2 KB | inherited from crucible-2 (audited there, HELD in crucible-2 P3) + DELTA §I P-1 | **HELD** (inherited-MEASURED) |

**Grade-labeling audit:** the draft's evidence-grade tags are, where traceable, honestly applied — the
horizon 0.0032 carries `residual_sidecar_owed`, the 20–50 KB SDF-field cost is tagged CONJECTURED
(matching the module header's own "a GUESS, ~1 order-of-magnitude; NO RD curve fitted"), and the 0.061-vs-
0.140 triple is the right shape. The three FAILs are not grade-laundering of the RATE numbers (those hold);
they are **stale/mismatched claims about the CODE the config compiles against** — exactly the launch-path≠
config-tests class the crucible names.

---

## §2 DECISION ATTACKS (the prompt-directed set + the F-table)

### F1 — increment-1a "d_seg-through-R, NO paint, NO P-C" is a confound — **REVISE (the worst; see HEADLINE)**
Through-R d_seg needs RGB; the only paint 1a admits is flat/canonical, floored at 0.0064 (MEASURED); the
measured d_seg reflects analytic-boundary quality + the flat-paint floor, not the decoupling mechanism. No
numeric kill criterion; graded against v7.5.2's UNMEASURED counter-force floor. `verdict_scope: FORMULATION`
(this "measure the bet before paint" formulation of 1a). Fix: pin which d_seg 1a measures (mask-optimal
PROXY vs oracle-through-R UPPER bound, both caveated) + a numeric threshold vs a MEASURED v7.5.2 Road floor.

### F2 — σ_cc′-structural (risk-4 "principled reason d_seg holds beyond the bet") is mis-attached — **REVISE**
Draft §C risk-4 (from S1 §5): "each carrier field carries its OWN eikonal/length anneal, so per-pair
anisotropic stiffness σ_cc′ is the decomposition itself … a PRINCIPLED reason v8's d_seg mechanism should
hold beyond the decoupling bet." But S1's argument presupposes **TRAINED per-edge SDF fields** with their
own eikonal/length/τ/β anneal (`lever_b_levelset_generator`). **T1 SHIPS analytic/parametric generators**
(horizon poly via `np.polyfit`, openpilot analytic band, sparse bbox sites, static frame0 hood) and
**DEMOTES the trained SDF field** (`bulk_sdf_field: OFF_default`, `lever_b_levelset_generator` = P-C-gated
fallback). An `np.polyfit` horizon and a bbox site have **no eikonal/length anneal → no structural σ_cc′.**
So risk-4's "principled reason beyond the bet" **does not attach to the shipped increment-1a carriers** —
T1's demotion removes the very object S1's σ_cc′ argument requires. `verdict_scope: FORMULATION`. Fix: drop
the σ_cc′-structural claim from increment-1a (rest its d_seg on the decoupling bet alone) OR state that the
carriers ARE eikonal-annealed SDF fields (which contradicts the SDF-field demotion). **The draft carries
BOTH the demotion (T1) and the σ_cc′-via-per-field-anneal (risk-4) without noticing they conflict.**

### F3 — flip-weighted b_c: the "MEDIAN" label ≠ the "OT-mass-to-flip-share" wiring — **REVISE (provenance FAIL)**
Draft A.3 states the form as `b_c − b_{c'} = MEDIAN_{flip-weighted}(m)` (S1's Hamming-optimal threshold) AND
the build as "wire `flip_share_by_class → target_masses` (a ~1-line objective swap in the BUILT damped-Newton
solver)" (S3). These are **different objectives.** `damped_newton_ot_offsets` matches soft-cell-**MASS**
(area) to `target_masses` (Kitagawa–Merigot–Thibert semi-discrete OT); passing `flip_share` still does
**mass-matching** (inflate cell c until its soft area = flip_share_c) — NOT a **median-threshold placement.**
grep confirms `laguerre_logit_offset` has ONLY `menon` + `damped_newton_ot` — **no median/quantile solver
exists.** So: (a) the "~1-line swap" implements OT-match-to-flip-share, an un-analyzed THIRD objective that
may re-inherit N-1's cell-inflation pathology (OT still inflates cells to hit a target); (b) S1's derived
median would need a NEW solver, not a 1-line swap. `verdict_scope: FORMULATION`. Fix: either build the
flip-density-median placement (and stop calling it a 1-line swap) OR analyze whether OT-to-flip-share is
N-1-safe before adopting it — and reconcile A.3's median math with the config's OT wiring.

### F4 — T1 demotion silently drops non-horizon Road↔Undriv boundary into the residual — **REVISE / provenance-gap**
`_horizon_profile` (L448) returns a **single-valued** `y(x)` = topmost Road row with Undriv directly above.
It captures ONLY the top sky/road horizon and structurally **cannot** represent (a) multi-valued horizons
(hills, underpasses, road visible above a near ridge) or (b) **LATERAL** Road↔Undriv boundaries (road
shoulder meeting grass/building/off-road undrivable at the SIDES — Undrivable class-2 INCLUDES non-sky
background, module header L36). The demoted signed-SDF bulk field (distance to the Road MASK) captured these;
the horizon poly does not. The draft's §A.1 catch #4 ACKNOWLEDGES multi-blob correctness "rides on the
RESIDUAL coder" — but this means **T1's demotion INFLATES the residual (T2's named enemy)** with lateral +
multi-valued Road↔Undriv mass. **Provenance-gap:** verify the 0.0189 horizon residual in the rate ledger
actually INCLUDES lateral/multi-valued px — `horizon_poly_xi_byte_cost`'s own `scope_note` defines its
residual as "poly-fit residual + secondary arcs (objects breaking the horizon)", which does NOT obviously
include lateral undriv. If the rollup measured "uncovered = all-Road↔Undriv-boundary − poly-covered", it is
included; if it measured only the horizon-arc neighborhood, the complete 0.0221 UNDER-counts the true
Road↔Undriv completion cost. `verdict_scope: FORMULATION` (the complement construction's coverage claim).

### F5 — owed-item #4 mis-states the code: the engage-on flag EXISTS — **REVISE (provenance FAIL)**
`--pose-finish-engage-on {muon, sigma_min_plateau}` EXISTS (@L10098) and engages pose-finish on the
σ_min(J_ξ) rolling-slope plateau conditioning EVENT (the crucible-2 SEALED A-1 fix). The draft's owed-item
#4 ("there is no `--pose-finish-engage-on` flag today") over-claims the build. **Accurate owed statement:**
the σ_min *conditioning-event engage* mechanism is BUILT; what is owed is (a) the **per-class {Road,Undriv}
d_seg-BASIN conjunct** (the flag's quantity is the aggregate/de-noised σ_min series, not the per-class
d_seg-basin the draft's gate design requires) and (b) the `f_basin=0.9` tuning. Good news (less to build),
but the draft must not cite a whole-flag gap. `verdict_scope: INSTANCE`.

### F6 — "707 B / 426 px / 2228 px" not re-derivable from the byte-cost functions — **CARRY-AS-RISK**
Neither on-disk function emits a perimeter px count; `bulk_boundary_byte_cost` measures the Road MASK
bitmap/RLE. The numbers are memo-sourced. Draft A.5 partially flags the mode= mismatch (S5-V3) but still
repeats 707/426/2228 descriptively. Not on the operating-point path, but a label-provenance drift.
`verdict_scope: INSTANCE`.

### F7 — T2 r\* waterfill well-posedness — **DISSOLVED-ON-INSPECTION**
The prompt asks whether #226's margin-conditional machinery actually produces the flip-weighted operating
point or is aspirational wiring. Re-derived from code: `WATERLINE_BYTES_PER_FLIP == 1.27 B/flip` is the
closed-form KKT break-even (`SEG_VALUE_PER_FLIP 8.48e-7 · BYTES_PER_SCORE`), and `waterfill_select` ranks
flips by descending net_value/cost and admits the prefix clearing the water level with `net_value > 0`
(collateral-aware). **This IS S2's KKT waterfill — real, not aspirational.** The DATA (per-flip net_value =
through-R margin recompute) is correctly flagged HEAVY/P-C-owed. **One carried nuance:** #226 codes FLIPS
(stored argmax corrections against a decoder-known boundary set), while T2's prose frames the residual as
"uncovered boundary px" — S4 reconciles this by reframing the residual AS flip-weighted, so the draft is
internally consistent, but P5 should confirm the "uncovered residual px" and "flips admitted by #226" are
the same set at increment-1. **Coupling to F1:** the r\* operating point is data-gated on the SAME through-R
measurement F1 shows is confounded at 1a — so r\* cannot be measured cleanly until 1b/P-C paint exists.

### F8 — the "Road/Lane 0.042 resists all levers → r\* near 0.140 = rate wash" honest range — **HELD**
S2 §7 and S4 flag it; the draft carries it as `binding_uncertainty` (config) + the D2 attack row, NOT
swallowed. It IS computable (from P-C's `residual_flip_fraction_per_edge`). Honestly carried. (But per F1/F7,
its measurement inherits the 1a confound — the range stays a RANGE until 1b lands.)

### HELD set (survived audit)
- **F9:** rate ledger (registered eq), phantom-41 (grep), δ_R (json), flip_share sensor exists, horizon
  0.0032 function real + caveat welded, banked R1 (crucible-2 audited). All re-derived, all clean.
- **T1 demote-SDF-field is SOUND on the numbers** — S5-V2's "no on-disk function measures the SDF-field cost"
  is CONFIRMED (module header L41-46 self-labels the 20–50 KB a GUESS with no RD curve); the draft correctly
  ships the MEASURED horizon (0.0032) and demotes the un-measured field to a P-C fallback. The T1 *rate*
  adjudication holds; F2/F4 are its σ_cc′-attachment and coverage-completeness gaps, not a rate error.
- **T2 triple {dominant 0.061 cond-floor / complete 0.140 default / r\* MEASURED} + counted-seed THIRD term**
  is the honest presentation S5-V1 demanded — the draft never quotes 0.061 alone. HELD.

---

## §3 BLIND-SPOT SWEEP (what all six seats + P2 missed)

1. **F1 (the 1a through-R-needs-paint confound)** — the deepest miss. Every seat treats "1a = decoupling bet
   measurable before paint" as sound; none noticed that through-R d_seg requires the paint that 1a excludes,
   floored by the 0.0064 flat-paint failure the same seats cite. The de-risking sequencing rests on it.
2. **F2 (σ_cc′ mis-attached to demoted fields)** — S1 asserted σ_cc′-is-structural; no seat checked it
   against T1's demotion of the trained SDF field; P2 carried both.
3. **F3 (median vs OT-mass wiring)** — S1 derived a median, S3 wired an OT-mass swap, P2 fused them as one
   "flip-weighted b_c" without noticing the objectives differ and no median solver exists.
4. **Temporal dimension / Movable slot-churn through-R** — the seats are per-frame; the composite's through-R
   d_seg at n600 requires Movable Hungarian ξ-slot *identity* to be temporally stable across the 600 frames,
   and R's interaction with the composed argmax at class boundaries (paint→R→argmax) is the entire 1b
   surface. Where temporal-ID churn lives in the 1a composite is unspecified. **CARRY-AS-RISK** for P4/P5.
5. **F5 (engage-on flag exists)** — the owed-list over-claims a build gap that is already closed.
6. **Resume/checkpoint story for a 5-carrier trainer** — the draft states "manifest-per-carrier byte-close
   BEFORE composition" but the RESUMABILITY-P0 contract (per-stage checkpoints of a 5-field decoupled trunk)
   is not specified for increment-1a. **CARRY-AS-RISK.** The #384 dry-start gate applies (F5 confirms the
   engage-on gate registers, but the per-class d_seg-basin conjunct is a NEW gate key that must register
   UNCONDITIONALLY at startup or a short dry-start misses it — the crucible-2 W-4 lesson recurs).

---

## §4 THE THREE MOST DANGEROUS (probability × blast × silence)

1. **F1 (1a confound).** Silent: a confounded "pass" ships v8 into the P8 comparison brief as "decoupling
   confirmed" when the win could be analytic-boundary quality or the flat-paint floor. The whole risk-6
   opportunity-cost de-risking is built on it. DECISION-BLOCKING for the comparison brief.
2. **F3 (median≠OT wiring).** The b_c is the "highest-leverage ~0-byte move" (S1: N-1 span = 5.2× the gap);
   shipping OT-match-to-flip-share while believing it is the Hamming median means the ~0-byte lever may
   silently sit on the N-1-adjacent side (mass-matching, just to a different target). MAX silence — the
   guard verifies "closed-form, out-of-loop", not "the objective is the median."
3. **F2 (σ_cc′ mis-attach).** Load-bearing for the "v8 d_seg should hold beyond the bet" story that the
   comparison brief will lean on — but it describes a construction (annealed SDF fields) T1 does not ship.

**Meta (attack my own pass):** F1 rests on "through-R d_seg needs RGB." If the crucible intends 1a to measure
**MASK-optimal** d_seg (composite argmax labels vs GT argmax, NOT through R) as the fast falsifier — a
reading the config's literal "d_seg-through-R (frozen CPU SegNet)" contradicts — then F1 downgrades to "the
config wording is wrong, fix the label," still REVISE. Either way 1a as WRITTEN does not do what it says. I
did NOT execute `horizon_poly_xi_byte_cost` / `bulk_boundary_byte_cost` on `gt_n600.npz` (a $0 read-only P4
probe would confirm the 0.0032 numeric and the proxy-vs-field gap empirically — R3 in the draft's recess
list); my "function real, caveat welded" is SOURCE-INSPECTION verified, not runtime-verified. F3's "OT-to-
flip-share ≠ median" is math re-derived from the solver's docstring + the absence of a quantile path; if a
median routine exists under a name my grep missed, F3 downgrades — flag for P5.

**Disposition to P4/P5:** the draft's RATE provenance is clean and T1/T2/phantom-41 survive on the numbers —
this is NOT a dead design. But **F1 is near-breaking for the increment-1a sequencing** (fix: specify which
d_seg 1a measures + a numeric kill threshold vs a MEASURED v7.5.2 floor), and F2/F3/F5 are three
"principled/wired/built" supports mis-attached to code that does not implement them (fix: drop σ_cc′ from
1a, build-or-analyze the b_c objective, correct owed-item #4). Provenance-audit failures: **3** (F3, F5, F6)
**+ 1 gap** (F4 residual scope). Pointer 0.19110 UNMOVED — this red-team is MEANS.

## STORES CONSULTED (line)
DRAFT SYNTHESIS_DRAFT_v8 · CONVENING · DELTA_GROUNDING · ORCHESTRATION_LEDGER · positions S1–S6 · crucible-2
P3_redteam · PRIMARY CODE (road_undriv_bulk_field.py L385/L448/L470 · margin_conditional_residual.py · 
laguerre_logit_offset.py · perclass_verdict.py:75 · train_levelset_witness_realized_through_R_mlx.py:10098 ·
v8_geometric_rate_decomposition_20260709.py) · SPEC_v8 (grep "41") · reports/delta_R_noise_floor.json ·
docs/operating_manual_craft_handoff.md. $0, no GPU, run dirs read-only, `[no-triality]`.
