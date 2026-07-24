# Codex findings — DDM CO2 campaign costate SENSE/COMPOSE/DECIDE

UTC: 2026-07-24T19:00:25Z  
Lane: `lane_ddm_co2_costate_campaign_sense_20260724`  
Maturity: `_dev`  
Authority: advisory only; `actuation=NONE`; `score_claim=false`; `promotion_eligible=false`  
Landing: **MAIN review required**

## Verdict

The campaign organ and its named consumers now share one hash-lineaged advisory
state. The current state is **instrument-ready but evidence-blocked**: J8F has
not emitted realized event marks, only 8/600 V19 pair joins are exact, and all
162 RD1 typed dimension duals have null train-decision prices. No launch,
run mutation, dispatch, pointer movement, or inferred price occurred.

Current state digest:
`a25a0ba8234ccf56a57b4bbb906c038b6d5ea22c123792e48cbe56ea51a715bb`.
Static-source lineage digest:
`adcc4baed5015704c36c4a1babaddc0c547c919ada069647c14512656d660ff3`.

## A–E consume-or-exclude

### A. SENSE — CONSUMED, with one premise falsified

- CONSUMED the DDM-366 dimension-completeness contract as the telemetry
  authority. Nine standing `ddm_campaign_sense_row.v1` rows now exist:
  joint-null, seg-only, pose-only, joint-visible, projector-rejected,
  temporal-flicker, clip-stationarity, counted-byte delta, and dribble rate.
  All nine honestly read `AWAITING_J8F_MEASUREMENT` today.
- CONSUMED the canonical engine envelope `ddm_event_mark.v1`; the organ does
  not create a parallel verdict schema. A future J8F stream must carry the
  DDM-366 fields inside the event mark's canonical `telemetry` mapping.
- CONSUMED all 162 RD1 typed dimension dual rows plus their MS4D metric-custody
  source. Every null stays null. The three aggregate continuation controls are
  separate, non-additive `DERIVED_FROM_TWO_MEASURED_N600_ENDPOINTS` rows.
- FALSIFIED the snapshot premise that a usable measured per-bucket
  `lambda_bucket = dS/dbyte` table already exists. The live RD1 v5 receipt has
  162 typed rows, 0 actionable rows, and 162 null byte prices. The precise
  blocker is candidate-delta x G4 x dimension counted-byte home plus
  receiver-closed uint8 histograms. This is formulation-scoped; it does not
  close the family.
- CONSUMED MS7 R0/PF3: 25/25 terminal buckets are
  `UNREACHABLE-AND-IGNORED`; the one PF3 class-birth control has measured
  instance radius 1 and remains nonadmissible.

### B. DECIDE — CONSUMED

The exact FEED-603 F1–F7 map is encoded:

| Typed plateau | Fork | Formulation |
|---|---:|---|
| `GRAMMAR_EXPRESSIBLE` | F1 | `EXTEND_THE_LIFT` |
| `PAINT_TEXTURE_BUCKET` | F2 | `TEMPLATE_PAINT_DESCENT` |
| `GOOD_DIRECTION_SLOW_CONVERGENCE` | F3 | `SOLVE_STEP_ALTERNATION` |
| `DESCRIPTION_COST_BOUND` | F4 | `PLANE_DESCENT_REDESCRIBE` |
| `GRAMMAR_UNREACHABLE` | F5 | `DIRECT_COUNTED_FIELD` |
| `ILL_CONDITIONED_SCORER_DEPTH` | F6 | `FEATURE_SPACE_RELAY` |
| `SEG_POSE_COUPLING_PATHOLOGY` | F7 | `LEXICOGRAPHIC_ALTERNATING` |

A realized event mark must also supply its typed residual
`{residual_type, metric_id, value, units}`. The recommendation includes that
trigger evidence and remains `actuation=NONE`. No current J8F event means no
fork is selected.

### C. DYNAMIC-NOT-ARBITRARY — CONSUMED / scoped exclusion

- G2F measured native-pixel knee: 1.0, guarded by the settled memo SHA.
- V16 coupled-solve instance: EXCLUDED as a universal radius because its
  receiver linearization was invalid. Verdict scope is that one formulation.
- V17 boundary-normal matched-prefix radius: DERIVED as 2 lattice quanta from
  the maximal consecutive positive realized-ratio prefix; q=4 ends the prefix.
- MS7 PF3 class-birth radius: MEASURED instance-only at 1 signed quantum.
- Alarm threshold law is `k*sigma`, with `k` derived from the preregistered
  family-wise alpha and measured same-regime noise samples. No alpha/noise
  history means a typed blocker, not a round threshold.
- Candidate K is the number of measured candidate evaluator bands intersecting
  the best candidate's band. No candidate bands means a typed blocker, not a
  literal K.

Every derived value carries a LawRef, provenance rung, source path/hash, and
scope. No Euclidean-naive score ranking was admitted.

### D. CONSUMERS — CONSUMED

`tools/costate_digest.py`, the dashboard observatory, campaign duty queue, and
`lever_registry.campaign_activation_nag` consume views guarded by the same
64-hex state digest. The dashboard caches only while the complete receipt
mtime/size signature is unchanged. Legacy duty keys remain for compatibility;
the campaign queue is:

1. `J8F_MEASURE_CLASS_E_TELEMETRY`
2. `V19_RECEIVER_CLOSED_JOIN`
3. `RD1_DIMENSION_RATE_HOME`

### E. V19 join — EXCLUDED from execution; blocker landed

No join launch was permitted by this advisory task. The organ derives the
current counts from the live G3 x V19 bundle and emits:

`BLOCKED_RECEIVER_CLOSED_V19_EVIDENCE_JOIN_592`

Exact evidence owed: receiver-closed V19 outcome for each missing pair and a
candidate rate allocation or explicit shared-rate home. Current exact coverage
is 8/600; missing coverage is 592/600. This is not a family negative.

## DDM-366 dimension-contract disposition

All class-E rows are standing telemetry. Class-A quantities are consumed
through the MS4D/RD1 scorer metric or guarded by exact receiver acceptance.
Class-B rows remain realized guards. Class-C values are derived from source
curves. Class-D DOFs remain preregistered F1–F7 forks. No silent row was
discarded; dimensions not independently priced are represented by explicit
null-price rows.

## Source custody

| Source | SHA-256 |
|---|---|
| DDM-366 dimension contract | `7ee44ca223433ec8e71563f45835fc51003076f6b4365256f7d4dd4e31256623` |
| MS4D complete metric bundle | `d670eff3dd01d61a24bdebedf045fa8cde2528953660dc6d1e64ba9c2fa94e25` |
| MS7 receiver edges | `5a9108f745316693d466336b16be8778037bf610833e46b3073a25252b610fe4` |
| RD1 lambda frontier v5 | `7449b4fedcbae41baf535018602435f9e36e6f09dd0dcd4d5f8ad4d022cbbce2` |
| J8E compile contract | `50c50ce571675b1f06e29198b282c33280a96d7c43eb6ec8c87c2b714b5a2a89` |
| G2F amplitude curve | `92d860ab35bba158e7fd817edf632d3e3e7fc90b05669402d537c26a6e09a88e` |
| V17 realized curve | `7cad1d17c5697b578234db282e6191ba3f2789b188051473874638392e18f53a` |
| V16 invalid-linearization receipt | `7b2d2c18f7790294bbfbbd875a6c4696e570a49826c8317109ca8a988c27f79d` |
| FEED-603 source | `0267f60b13667d84796f3e7e95a7d73dd16d825c7abca9e2835fcaf6b784bd6e` |

The 2026-07-17 quarantined witness run was not consulted.

## Verification

- Directly affected suites: 47 passed.
- Broader digest/dashboard/atlas regression selection: 57 passed.
- Total selected clean evidence: 104 passed.
- Python compile and Ruff on new/edited TAC modules: clean.
- `git diff --check`: clean.
- Known baseline outside this diff: `test_lever_registry.py` reports three
  pre-existing DSL-emitted/trainer-stale flags
  (`--integer-plane-emitter-basis` and two adjacent policy fields). This lane
  neither introduced nor changed those flags.

## Triality

- DSL/control: typed campaign SENSE rows, canonical event-mark intake, guarded
  consumer views, activation nag.
- DAG: J8F event mark → SENSE → dynamic policy → typed plateau → F1–F7 advisory
  fork; missing evidence → exact duty/blocker rows.
- Equations: scorer-metric marginal score/byte, same-regime family-wise
  noise threshold, evaluator-band K, and scoped validity-radius derivations are
  recorded in the companion equations artifact.

## MAIN landing review

MAIN must independently verify:

1. the canonical `ddm_event_mark.v1` J8F producer extension uses these exact
   DDM-366 telemetry names rather than an adapter schema;
2. RD1's 0/162 actionable state is still current after merge;
3. dashboard payload size/refresh latency remains acceptable with 162 rows;
4. review-tracker passes and the selected 104-test evidence;
5. no authority boundary changes (`_dev`, `actuation=NONE`, no score claim).
