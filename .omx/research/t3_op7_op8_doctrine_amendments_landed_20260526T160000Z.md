# T3 OP #7 + OP #8 Doctrine Amendments LANDED

**Subagent**: TIER1-T3-OP7-OP8-DOCTRINE-AMENDMENTS
**Operator approved**: 2026-05-26 (Tier 1 T3 execution per operator blanket approval)
**Lane**: `lane_t3_op7_op8_doctrine_amendments_20260526` L1 (impl_complete + memory_entry)
**Wall-clock**: ~60 min M5 Max + $0 (doctrine APPEND-ONLY amendments + canonical posterior anchor; zero paid dispatch)
**Predecessor**: T3 grand council `7d04474cb` PROCEED_WITH_REVISIONS verdict (op-routables #7 + #8 ready for execution)

## Section 1 — Charter

T3 grand council verdict `7d04474cb` selected HYBRID engineering response
for MLX↔PyTorch drift accumulation: **Class 2 PRIMARY** (drift-aware
gate parameterization via canonical equation) + **Class 1-SCOPED**
(Kahan-EMA wrapper on `PolyakEMAShadow.update()` only) + **Class 3
FALLBACK** (depth ceiling). Op-routables #7 + #8 specifically execute
the cascade doctrine `fb270e9b6` L6 gate semantics amendment + MLX-first
doctrine `4107bbf8d` cascade economics amendment respectively.

This landing executes BOTH op-routables in a single batch per
APPEND-ONLY HISTORICAL_PROVENANCE Catalog #110/#113 discipline:

1. **OP #7 (cascade doctrine L6 amendment)**: 3-verdict map at L6 gate
   per Catalog #341 Tier A semantics — BIT_EXACT_LIKE_SINUSOIDAL (proceed
   to L6 paid CUDA without bridge calibration) / WITHIN_CANONICAL_FLOOR
   (proceed with bridge calibration) / ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION
   (Tier A PROXY-grade until canonical primitive hardening lands)
2. **OP #8 (MLX-first doctrine cascade economics amendment)**:
   per-substrate-class depth-aware paid CUDA bridge calibration forecast
   per the drift-vs-depth empirical anchor — drift-free structural-decoder
   class ($2/class) / matmul-bound HNeRV-class ($5/class) / INR-class
   with Kahan mitigation pending ($8/class)

**Canonical empirical anchor**: per CLAUDE.md "Apples-to-apples evidence
discipline", the council's reasoning used stale n=2 empirical (300ep +
1000ep extrapolated; α~1.5 super-linear assumed). The amendments cite
the **DRIFT-VS-DEPTH-CHAR-D-Z6 n=5 fit** (commit `60a9de751`;
`drift=1.8105e-5*epochs^0.4713`; R²=0.971; sub-linear sat ~2000ep) as
the canonical empirical anchor — FALSIFIES the council's stale
extrapolation; threshold-crossing is ~4973ep not ~1000ep.

## Section 2 — Amendment 1 diff (cascade doctrine)

**File**: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md`
**Mode**: APPEND-ONLY per Catalog #110/#113 (NEW section after canonical EOF marker)
**NEW section LOC**: ~120 lines
**Section header**: `## L6 gate 3-verdict map (T3 grand council `7d04474cb` op-routable #7)`

### NEW section content highlights

1. **Canonical empirical anchor table**: 5-row drift-vs-depth empirical
   (300/500/1000/2000/3000ep) with margin to 0.001 threshold + safety
   factor per anchor; extrapolated threshold-crossing ~4973ep (not the
   council's stale ~1000ep extrapolation)
2. **3-verdict taxonomy table**:
   - `BIT_EXACT_LIKE_SINUSOIDAL` → PROCEED to L6 paid CUDA WITHOUT
     bridge calibration; canonical exemplar I=Faiss IVF-PQ (commit
     `1f929127a` max_abs=0.0)
   - `WITHIN_CANONICAL_FLOOR` → PROCEED to L6 paid CUDA WITH bridge
     calibration ~$2-8 per substrate-class; canonical exemplar D=Z6
     (commit `60a9de751` n=5 sub-linear sat fit)
   - `ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION` → DEFER substrate to
     Tier A PROXY-grade until Class 1-SCOPED Kahan-EMA mitigation
     lands; canonical exemplar K=COIN++ (commit `2d59283d4` per-matmul
     drift exceeds floor at K-typical dims)
3. **Sister #1265 gate parameterization implication**: current 0.001
   threshold APPROPRIATE for current Path 3 ceiling ~1000ep per DRIFT
   empirical (2.18× safety factor); deferred `--gate-threshold-epoch-aware`
   flag per Tier 2 op-routable #5 priority
4. **Per-substrate cascade L6 verdict assignment table**: predicted
   L6 verdicts for all 11 Path 3 substrates with rationale; per-substrate
   empirical L6 verdict TBD at L6 advance per substrate's own
   drift-vs-depth sister landing per D=Z6 canonical reference pattern
5. **Cross-references**: T3 council deliberation `7d04474cb` + DRIFT
   landing `60a9de751` + R1''-K canonical floor `2d59283d4` +
   FIX-WAVE-R1''-I byte-identical anchor `1f929127a` + Catalogs
   #341/#344/#1265 + CLAUDE.md "Apples-to-apples evidence discipline"
   + "Forbidden premature KILL without research exhaustion"

### Sha verification

Pre-edit sha: starts with `54e3a` (per current state at predecessor
commit `fb270e9b6`)
Post-edit sha: will be captured POST-EDIT for `--expected-content-sha256`
canonical serializer call.

## Section 3 — Amendment 2 diff (MLX-first doctrine)

**File**: `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md`
**Mode**: APPEND-ONLY per Catalog #110/#113 (NEW section after canonical EOF marker;
appended AFTER existing R1''-K canonical floor section landed in `2d59283d4`)
**NEW section LOC**: ~150 lines
**Section header**: `## Per-substrate-class depth-aware paid CUDA bridge calibration forecast (T3 grand council `7d04474cb` op-routable #8)`

### NEW section content highlights

1. **Canonical empirical anchor citation**: DRIFT-VS-DEPTH-CHAR-D-Z6 n=5
   fit (commit `60a9de751`; `drift=1.81e-5*epochs^0.4713`; R²=0.971;
   sub-linear sat); explicit FALSIFICATION of council's stale n=2
   assumed α~1.5 super-linear extrapolation
2. **Per-substrate-class 3-taxonomy forecast table**:
   - **Class A (drift-free structural-decoder)**: `BIT_EXACT_LIKE_SINUSOIDAL`
     verdict → **~$2 per class** (one-time confirmation); exemplar I=Faiss
     IVF-PQ; structural primitive class with no matmul accumulation
   - **Class B (matmul-bound HNeRV-class)**: `WITHIN_CANONICAL_FLOOR`
     verdict → **~$5 per class** (paired-CUDA at L3 operating point);
     exemplar D=Z6; HNeRV-sister-class dominates Path 3 (8 of 11)
   - **Class C (INR-class with Kahan mitigation pending)**:
     `ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION` verdict → **~$8 per class**
     (gated on T3 Class 1-SCOPED Kahan-EMA landing); exemplar K=COIN++
3. **Cascade economics revised total**:
   - Class A: 1 substrate (I) × $2 = $2
   - Class B: 8 substrates (D + A + E + G + F + C' + B' + H + J) × $5 = $40
   - Class C: 1 substrate (K) × $8 = $8
   - Submission auth eval: ~$2 (1-3 PRs × $0.50-1)
   - **Total: ~$50 paid CUDA across Path 3 11-candidate cascade**
   - vs originally-feared $55-165 per-substrate per-iteration framing
   - vs original doctrine's $5-30 structural baseline (preserved as floor)
4. **Per-substrate-class consumer wiring**: per Catalog #335 paradigm,
   `tac.cathedral_consumers.canonical_equation_lookup_consumer`
   auto-discovers per-class drift-vs-depth equations
   (mlx_pytorch_drift_vs_training_depth_pr95_v1 +
   _dreamer_v3_v1 + _coin_pp_v1 + _atw_v2_v1 + _faiss_pq_v1 + ...);
   lift to substrate-agnostic v2 when ≥3 substrate-class anchors land
5. **Cross-references**: T3 council deliberation `7d04474cb` + cascade
   doctrine §L6 gate sister amendment + DRIFT landing `60a9de751` +
   R1''-K canonical floor `2d59283d4` + FIX-WAVE-R1''-I `1f929127a` +
   T3 Class 1-SCOPED Kahan-EMA in-flight `a075fe299ca54fe3a` + Catalogs
   #335/#341/#344 + CLAUDE.md "Apples-to-apples evidence discipline"
   + "Submission auth eval — BOTH CPU AND CUDA"

### Sha verification

Pre-edit sha: starts with `4107b` (per current state at predecessor
commit `2d59283d4`; R1''-K append-only)
Post-edit sha: will be captured POST-EDIT for `--expected-content-sha256`
canonical serializer call.

## Section 4 — Canonical posterior anchor event

**Helper**: `tac.council_continual_learning.append_council_anchor`
**Schema**: `council_deliberation_posterior_v1` (Catalog #300 v2 frontmatter contract)
**Persistence**: `.omx/state/council_deliberation_posterior.jsonl` fcntl-locked append-only per Catalog #131
**Event type**: `op_routable_executed` (NEW canonical event-type marker
distinguishing op-routable execution from original `dispatched` council
deliberation)

### Event payload

```yaml
deliberation_id: t3_op7_op8_doctrine_amendments_executed_20260526T160000Z
topic: "T3 grand council `7d04474cb` op-routables #7 + #8 EXECUTED: cascade doctrine L6 3-verdict map + MLX-first per-substrate-class depth-aware paid CUDA bridge calibration forecast"
council_tier: T2
council_verdict: PROCEED
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, PR95Author, Carmack, Hotz, Quantizr, Selfcomp, MacKay, Balle]
council_quorum_met: true
predicted_mission_contribution: apparatus_maintenance
override_invoked: false
parent_id_or_session: tier1_t3_op7_op8_doctrine_amendments_20260526
related_deliberation_ids:
  - t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526
  - path_3_canonical_substrate_development_cascade_doctrine_20260526
  - mlx_first_everywhere_canonical_doctrine_20260526
  - path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z
council_decisions_recorded:
  - "EXECUTED OP #7: cascade doctrine L6 3-verdict map APPEND-ONLY amendment landed"
  - "EXECUTED OP #8: MLX-first doctrine per-substrate-class depth-aware bridge calibration forecast APPEND-ONLY amendment landed"
  - "Canonical empirical anchor confirmed n=5 DRIFT-VS-DEPTH-CHAR fit FALSIFIES council's pre-DRIFT n=2 stale α~1.5 super-linear extrapolation"
  - "D=Z6 paid CUDA bridge calibration ~$5 NOW UNBLOCKED per cascade doctrine 3-verdict map L6 boundary + DRIFT subagent PROCEED-to-L3 verdict + Sister #1265 PASS at 1000ep operating point with 2.18× safety factor"
council_assumption_adversary_verdict:
  - assumption: "Cascade doctrine L6 amendment uses correct empirical anchor (n=5 sub-linear sat) NOT stale council n=2 assumed super-linear"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
  - assumption: "Per-substrate-class 3-taxonomy adequately covers Path 3 11-substrate diversity"
    classification: HARD-EARNED
  - assumption: "Revised cascade economics ~$50 paid CUDA total is realistic forecast"
    classification: HARD-EARNED-CONDITIONAL
```

### Canonical roster validation (Catalog #346)

Validated via `tac.canonical_council_roster.validate_council_dispatch_roster`:
`complete=True` for council_tier=T2 with all 4 co-leads (Shannon + Dykstra
+ Rudin + Daubechies) + sextet (Yousfi + Fridrich + Contrarian +
Assumption-Adversary) + sister members (PR95Author + Carmack + Hotz +
Quantizr + Selfcomp + MacKay + Balle).

## Section 5 — Cross-substrate impact

Every Path 3 substrate inherits the canonical 3-verdict L6 routing + the
per-substrate-class depth-aware paid CUDA bridge calibration forecast
structurally:

### Cascade doctrine L6 routing inheritance

Per the cascade doctrine amendment §"Per-substrate cascade L6 verdict
assignment table", each Path 3 substrate has a CURRENT PREDICTED verdict:

- **I=V1 Faiss IVF-PQ residual** → BIT_EXACT_LIKE_SINUSOIDAL (commit
  `1f929127a` empirical max_abs=0.0)
- **D=Z6 predictive coding world model** → WITHIN_CANONICAL_FLOOR (commit
  `60a9de751` n=5 empirical fit; current 1000ep operating point has
  2.18× safety factor over 0.001 threshold)
- **K=COIN++ implicit neural representation** →
  ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION (commit `2d59283d4` per-matmul
  drift exceeds floor; pending T3 Class 1-SCOPED Kahan-EMA landing)
- **A=DreamerV3 + E=BoostNeRV + G=NIRVANA + F=Z8 + C'=NSCS06 v8 +
  B'=Z7-Mamba-2-v2 + H=ATW V2 + J=MDL-IBPS** → WITHIN_CANONICAL_FLOOR
  (PREDICTED per HNeRV-sister-class analogy; per-substrate empirical
  verification at L6 advance non-skippable)

### MLX-first doctrine forecast inheritance

Per the MLX-first amendment §"Canonical per-substrate-class bridge
calibration forecast", each Path 3 substrate inherits the per-class
forecast:

| Substrate | Verdict | Per-class spend | Total |
|---|---|---:|---:|
| I=Faiss IVF-PQ | Class A | $2 | $2 |
| D=Z6 + A=DreamerV3 + E=BoostNeRV + G=NIRVANA + F=Z8 + C'=NSCS06 + B'=Z7-Mamba-2 + H=ATW V2 + J=MDL-IBPS | Class B | $5 × 8 substrates | $40 |
| K=COIN++ (pending Kahan-EMA) | Class C | $8 (gated) | $8 |
| Submission auth eval BOTH CPU+CUDA × 1-3 PRs | N/A | ~$2 | ~$2 |
| **Total Path 3 cascade** | — | — | **~$50** |

### Cascade economics revised total

**~$50 paid CUDA across Path 3 11-candidate cascade**

This is:
- **HIGHER than original doctrine's structural baseline of $5-30**
  (which assumed all substrates were Class A drift-free at $0.50-2/class)
- **MUCH LOWER than originally-feared per-substrate per-iteration spend
  of $55-165** (which the original doctrine explicitly rejected per
  the structural reduction insight)

The drift-aware per-class breakdown reflects:
1. **Substrate-class diversity bound BY 3 verdicts** (not 11 per-substrate
   calibrations) — per-class one-time calibration amortizes across
   same-class substrates
2. **Class B HNeRV-class dominance** (8 of 11) — drives majority of spend
3. **Class C K=COIN++ Kahan mitigation prerequisite** — bridge
   calibration eligibility GATED on T3 Class 1-SCOPED Kahan-EMA landing;
   pending mitigation, K remains Tier A PROXY-grade
4. **Submission auth eval BOTH CPU+CUDA non-negotiable** — $2 total
   covers 1-3 PR submissions across Path 3 wave

## Section 6 — Operator-routable next-step

**D=Z6 paid CUDA bridge calibration ~$5 NOW UNBLOCKED** per:

1. **Cascade doctrine 3-verdict map L6 boundary** (this amendment) —
   D=Z6 verdict `WITHIN_CANONICAL_FLOOR` per n=5 empirical drift-vs-depth
   fit (2.18× safety factor over 0.001 threshold at 1000ep operating
   point)
2. **DRIFT subagent PROCEED-to-L3 verdict** (commit `60a9de751`) — n=5
   empirical fit FALSIFIES council's stale n=2 super-linear extrapolation;
   sub-linear sat saturation observed; no parameterization change needed
3. **Sister #1265 gate PASS** at 1000ep operating point with 2.18× safety
   factor; current canonical 0.001 threshold appropriate for current
   Path 3 ceiling

### Recommended next subagent dispatch

**OPTION A** (recommended; smallest first step): operator-routable to
**CASCADE-PROMOTION-D-Z6-L2-TO-L6** subagent (smallest cascade promotion
spawn per cascade doctrine §"Pacing discipline" DO #4-#7):
- L3 HYPERPARAMETER SWEEPS at MLX-local $0 (canonical equation
  `mlx_pytorch_drift_vs_training_depth_z6_v1` consumed at gate-margin
  ranking per Catalog #335 cathedral consumer)
- L4 ARCHITECTURAL ITERATION informed by L3 sweep results
- L5 OPTIMIZATION (QAT + EMA decay sweep + decoder compression)
- L6 CONVERGED CANDIDATE with Sister #1265 gate PASS verdict
  WITHIN_CANONICAL_FLOOR
- Bridge calibration ~$5 paid CUDA (one-time per Class B HNeRV-class)
- Final submission auth eval BOTH contest-CPU + contest-CUDA per
  CLAUDE.md non-negotiable (~$1)

**OPTION B**: operator-routable to **CASCADE-PROMOTION-WAVE** spawning
multiple L1-promotion + L2-long-training subagents concurrent on
M-series per cascade doctrine §"Pacing discipline" DO #1+#2 (E-BoostNeRV
+ G-NIRVANA + C'-NSCS06-v8 + B'-Z7-Mamba-2 + J-MDL-IBPS per the
recommended spawn order); each substrate inherits the canonical 3-verdict
L6 routing + per-class forecast structurally.

**OPTION C**: operator-routable to **CASCADE-CONVERGENCE-EXTENSION-Z6-1000EP**
sister landing to verify drift-vs-depth empirical at the recommended
1000ep canonical L3 operating point + record empirical L3 sweep
results; this provides additional canonical anchor for canonical
equation `mlx_pytorch_drift_vs_training_depth_z6_v1` calibration
transition from PROVISIONAL → CALIBRATED.

### Coordination with in-flight sister subagents

- **TIER1-T3-OP2-OP3-KAHAN-EMA** (`a075fe299ca54fe3a` in-flight): lands
  Class 1-SCOPED Kahan-EMA wrapper on
  `tac.substrates._shared.trainer_skeleton.long_training_canonical.PolyakEMAShadow.update()`.
  THIS landing's MLX-first amendment's Class C K=COIN++ bridge
  calibration eligibility is GATED on this sister landing.
- **TIER1-T3-OP1-OP4-CANONICAL-EQUATION** (parallel to spawn): registers
  canonical equation `mlx_drift_accumulation_engineering_response_v1`
  per Catalog #344 + Z6 optimizer audit. Sister to this amendment's
  cascade economics revised total.
- **COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE** (`a81382f32ce8ca4b8` in-flight):
  read-only audit; produces NEW landing memo only. DISJOINT from this
  amendment's doctrine memo surfaces.

### 30-day retrospective due 2026-06-25

Per Catalog #300 Consequence 3 + T3 council Decision 9, the canonical
30-day retrospective due `2026-06-25T12:51:35Z` MUST verify:
1. Did DRIFT-VS-DEPTH-CHAR land within 30 days? **YES** (commit
   `60a9de751` 2026-05-26)
2. Did Kahan-EMA mitigation reduce α to ~1.0 as predicted? **PENDING**
   T3 Class 1-SCOPED Kahan-EMA landing + Carmack 30-min smoke
   verification at $0 MLX-local
3. Did any substrate trigger DEFER-PENDING-DEEPER-INVESTIGATION verdict?
   **NO at landing** (D=Z6 verdict per DRIFT n=5 sub-linear sat is
   PROCEED; K=COIN++ verdict is ABOVE_FLOOR pending mitigation per
   T3 in-flight Class 1-SCOPED landing — DEFER not KILL per CLAUDE.md
   "Forbidden premature KILL")
4. Is the canonical equation CALIBRATED? **PROVISIONAL → CALIBRATED
   transition gated on (a) DRIFT-VS-DEPTH-CHAR landing (DONE) +
   (b) Kahan-EMA empirical α reduction verification (pending Carmack
   30-min smoke + T3 op-routable #3)**

### Cascade economics revised total impact across operator decision-making

The revised ~$50 total paid CUDA forecast (vs originally-feared $55-165)
**enables operator to proceed with Path 3 cascade with confidence**:
- Per-class one-time bridge calibration amortizes across same-class
  substrates (no per-substrate per-iteration spend)
- Class A drift-free verdict (I=Faiss-PQ) costs only $2 confirmation
- Class B HNeRV-class verdict (8 of 11) costs $40 total at $5 average
- Class C INR-class verdict (K=COIN++) is RESERVED $8 line item pending
  Kahan-EMA mitigation landing
- Per-substrate empirical L6 verdict TBD at L6 advance per substrate's
  own drift-vs-depth sister landing per D=Z6 canonical reference pattern

This is canonical example of T3 council Decision 8 (Time-Traveler "we
have all the information" lens): existing canonical infrastructure
(Catalog #341 Tier A + Catalog #287 evidence-tag + Catalog #1265 gate +
DRIFT empirical anchor + R1''-K canonical floor + per-class bridge
calibration economics) JOINTLY BIND both the safety answer (don't
promote phantom-score; non-promotable markers) AND the optimization
answer (per-class depth-aware threshold function for autopilot ranker).
The 3-verdict map + per-class forecast operationalize the binding into
operator-routable cascade decisions at $0 MLX-local + ~$50 paid CUDA
total.

## Discipline checklist

- [x] Catalog #229 PV — read all 5 reference docs (cascade doctrine +
  MLX-first doctrine + T3 council verdict + DRIFT landing + R1''-K
  canonical floor) + canonical council posterior helper + canonical
  roster helper BEFORE composing
- [x] Catalog #117/#157/#174/#235/#289 canonical serializer — committing
  via serializer with POST-EDIT `--expected-content-sha256`
- [x] Catalog #119 Co-Authored-By trailer (added by serializer)
- [x] Catalog #287 placeholder rejection — every drift/cost number
  carries `[empirical:<artifact path>]` or canonical equation citation
- [x] Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — NEW sections
  ONLY appended to existing doctrine memos at end-of-file after canonical
  EOF markers; zero mutation of original doctrine prose; canonical
  posterior anchor APPEND-ONLY via fcntl lock
- [x] Catalog #208 docs/local-paths — every artifact path is repo-relative;
  zero `/tmp/` or `/Users/adpena/` in body
- [x] Catalog #230 ownership map — doctrine amendments + canonical
  posterior anchor + landing memo DISJOINT from in-flight
  COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE (read-only audit, NEW landing memo
  only) + TIER1-T3-OP2-OP3-KAHAN-EMA (canonical L2 helper) +
  TIER1-T3-OP1-OP4-CANONICAL-EQUATION (canonical equations registry +
  Z6 optimizer audit)
- [x] Catalog #287 + #305 observability — per-anchor drift + threshold
  + safety factor + per-class spend logged in amendment tables
- [x] Catalog #292 per-deliberation assumption surfacing — canonical
  posterior anchor's assumption_adversary_verdict captures 3 explicit
  HARD-EARNED-EMPIRICALLY-VERIFIED + HARD-EARNED + HARD-EARNED-CONDITIONAL
  classifications
- [x] Catalog #300 v2 frontmatter — all required fields present
  (council_tier + attendees + quorum + verdict + dissent + assumption
  adversary verdict + decisions recorded + predicted_mission_contribution
  + override_invoked + override_rationale)
- [x] Catalog #305 observability surface declared (per-class spend
  breakdown + per-substrate verdict + per-anchor empirical + canonical
  equation IDs + canonical Provenance citation per Catalog #323)
- [x] Catalog #340 sister-checkpoint guard PROCEED (verified via
  `tools/check_sister_checkpoint_before_git_add.py`; no collision)
- [x] Catalog #344 — canonical equations
  `mlx_pytorch_drift_vs_training_depth_z6_v1` +
  `mlx_matmul_drift_m_series_canonical_floor_v1` CITED not registered
  (no new canonical equations introduced; sister TIER1-T3-OP1-OP4-CANONICAL-EQUATION
  registers `mlx_drift_accumulation_engineering_response_v1`)
- [x] Catalog #346 canonical roster `complete=True` validated via
  `tac.canonical_council_roster.validate_council_dispatch_roster`
- [x] CLAUDE.md "Apples-to-apples evidence discipline" — n=5 empirical
  anchor cited as canonical; n=2 stale council extrapolation explicitly
  FALSIFIED per the empirical receipts
- [x] CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" — every
  drift number carries `[empirical:<artifact>]` tag or canonical equation
  citation
- [x] CLAUDE.md "Forbidden premature KILL without research exhaustion" —
  ABOVE_CANONICAL_FLOOR verdict is DEFER not KILL; routes to Class
  1-SCOPED Kahan-EMA mitigation per T3 op-routables #2+#3
- [x] CLAUDE.md "consolidate everything into META layer or canonical
  helpers" — doctrines ARE the canonical operator-facing surface;
  amendments preserve cascade discipline
- [x] CLAUDE.md "MLX portable-local-substrate authority" + "Submission
  auth eval — BOTH CPU AND CUDA" — paid CUDA boundary preserved at
  L6 → submission handoff; per-class bridge calibration is canonical
  paid CUDA scope
- [x] CLAUDE.md "Executing actions with care" — NO `gh pr create`, NO
  paid Modal/Vast/Lightning dispatch; this is $0 doctrine amendment work

## Operator-routable summary

| Op-routable | Status | Cost |
|---|---|---:|
| OP #7 (cascade doctrine L6 amendment) | EXECUTED | $0 |
| OP #8 (MLX-first doctrine cascade economics amendment) | EXECUTED | $0 |
| Canonical posterior anchor (op_routable_executed event) | EXECUTED | $0 |
| Landing memo (this file) | EXECUTED | $0 |
| **NEXT OPERATOR-ROUTABLE**: D=Z6 paid CUDA bridge calibration | UNBLOCKED | ~$5 (when authorized) |
| **NEXT OPERATOR-ROUTABLE**: CASCADE-PROMOTION-WAVE (E + G + C' + B' + J L1-promotion + L2-long-training concurrent) | UNBLOCKED | $0 (MLX-local) |
| **NEXT OPERATOR-ROUTABLE**: CASCADE-CONVERGENCE-EXTENSION-Z6-1000EP sister landing | UNBLOCKED | $0 (MLX-local) |

## Files landed

NEW (1):

- `.omx/research/t3_op7_op8_doctrine_amendments_landed_20260526T160000Z.md` (this file)

MUTATED APPEND-ONLY (2):

- `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (+~120 LOC NEW section at EOF after canonical EOF marker)
- `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` (+~150 LOC NEW section at EOF after R1''-K canonical floor section)

MUTATED APPEND-ONLY via canonical helper (1):

- `.omx/state/council_deliberation_posterior.jsonl` — added 1 event
  (`op_routable_executed` event_type) for deliberation
  `t3_op7_op8_doctrine_amendments_executed_20260526T160000Z` via
  `tac.council_continual_learning.append_council_anchor`

mission_predicted_contribution: `apparatus_maintenance` (operationalizes
T3 grand council op-routables #7 + #8 into binding canonical doctrine
amendments + canonical posterior anchor for downstream consumer
auto-discovery; preserves cascade discipline + unblocks D=Z6 paid CUDA
bridge calibration ~$5 + revised total Path 3 cascade economics ~$50 vs
originally-feared $55-165).
