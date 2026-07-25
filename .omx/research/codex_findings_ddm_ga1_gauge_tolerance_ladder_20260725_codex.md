# Codex findings — DDM GA1 fiber-to-gauge tolerance ladder

UTC: 2026-07-25T04:05:00Z  
Lane: `ddm_ga1_gauge_tolerance_ladder`  
Delegation checkpoint:
`codex_delegate:ddm_ga1_gauge_tolerance_ladder:20260725T035215Z`  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
`research_only=true` · `execution_allowed=false` · `score_claim=false` ·
`promotion_eligible=false` · `main_landing_review_required=true`  
Pointer: `0.1910828242 [contest-CPU]` — **UNMOVED**

## Verdict

`CURVE_BLOCKED_SOURCE_CUSTODY;`
`CURRENT_C1_GAUGE_LEVER_DOMINATED_BY_TYPED_MASS_UPPER_BOUND`

The requested measured curve cannot lawfully be emitted from the sealed
producers. The sealed DR2b receipt contains no tolerance rungs and no lossy
rerace rows, and explicitly reports that the SDWL1-to-E2 coordinate transfer
and coordinate crosswalk are blocked. Substituting a newly invented grid would
violate the delegated task.

This blocker is accompanied by a stronger lawful result for the current
typed-home-preserving FIBER→GAUGE lever:

`convertible_fraction <= 151 / 134211 = 0.0011250940682954451`

That is at most **0.112509%** of the currently counted C1 allocation, even if
every currently allocated FIBER byte converts to zero bytes at zero distortion
cost. It is strictly below the pre-registered 5% threshold, so the current
LP1 composition triggers the delegated falsifier by upper bound. This is an
`INSTANCE` disposition only: future C1 retyping and other gauge formulations
remain open.

## Fresh source audit

| surface | SHA-bound current fact | consequence |
|---|---|---|
| DR2b v4 | `priced_sdwl1_rungs=[]`, `lossy_rows=[]`, status `BLOCKED_NO_LAWFUL_SDWL1_TO_E2_TOLERANCE_TRANSFER` | no canonical rung grid exists to reuse |
| LP1 | 134,211 current counted bytes; only `solved_template_outer_home`, 151 bytes, is currently typed `FIBER` | 151 bytes is a hard upper bound for the current typed-home-preserving conversion |
| #580 | 80.6742315% real-linear nullity, but only a one-frame fixture and a 34.1931391% uint8-feasible-basis lower bound; class/margin strata not measured | structural nullity cannot be transferred to current C1 counted bytes |
| #532 | six exact uint8 frames; subset non-promotable | no current-C1 stream/projector/receiver crosswalk |
| RD1 v5 | exact per-class errors sum to 17,927; zero actionable typed dimension duals and zero priced effective quanta | cross-check cannot run; delegated 17,931 endpoint is stale by four errors |
| MS2R R2 | box endpoint is 136,839 allowed errors | delegated box endpoint is confirmed |

The five class names are consumed from RD1 per-class keys. No numeric scorer
channel indices are hardcoded.

## Durable construction

- Strict compiler:
  `src/tac/optimization/ddm_ga1_gauge_tolerance_ladder.py`
- Typed materializer:
  `tools/build_ddm_ga1_gauge_tolerance_ladder.py`
- SHA/byte/schema-pinned config:
  `.omx/research/configs/ddm_ga1_gauge_tolerance_ladder_20260725.json`
- Typed receipt:
  `.omx/research/ddm_ga1_gauge_tolerance_ladder_20260725T035215Z/receipt.json`
- Receipt SHA-256:
  `885119e5316508d72a52f9f04716fb08d3855fcbcb72bae5cf14f3946a398f5c`
- Receipt size: 10,156 bytes
- DAG/feed:
  `.omx/research/ddm_ga1_gauge_tolerance_ladder_DAG_FEED_20260725.md`

The compiler fails on input byte/hash/schema drift. It emits no curve row, no
joint exchange rate, no RD1 disagreement row, and no canonical functional-form
equation when the measured prerequisites are absent. It emits five typed
class×shared-stream SENSE blocker rows; they explicitly remain non-additive
across class strata.

## Exact reactivation edge

Append, rather than replace, all of:

1. a nonempty SHA-bound DR2b rung grid on one explicit coordinate system;
2. an invertible SDWL1 fact ↔ current C1 receiver-coordinate crosswalk;
3. per-current-FIBER-stream #580 range/gauge projection with receiver
   parse-back, #532 uint8 realization, exact R, and argmax custody;
4. joint candidate-delta × self-detected class × dimension-rate homes plus
   per-dimension receiver uint8 absolute-step histograms;
5. one MAIN-selected, SHA-pinned exact endpoint resolving 17,931 versus 17,927.

Only then may the existing compiler's blocked curve surface be superseded by a
per-rung measured table and an RD1 >2× disagreement audit.

## Verification

- deterministic materialization repeated byte-identically;
- focused tests: 4 passed;
- Ruff: clean;
- Python compile: clean;
- `git diff --check`: clean;
- two post-fix clean review passes recorded in
  `.omx/research/reviews/ddm_ga1_gauge_tolerance_ladder_reviews_20260725.json`.

## STORES CONSULTED

Delegated authority; `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; `tac.subagent_contract`; DR2b v4
receipt/config/findings; LP1 receipt/findings; RD1 v5 receipt, dimension
supplement, and findings; #580 source and full-kernel receipt; #532 source and
uint8-lattice receipt; MS2R R2/R3 receipts; canonical lane/subagent/task
surfaces; per-arm and fleet inboxes; current DDM sister memos.

## MAIN landing review

MAIN must independently review the FIBER-only upper-bound premise, the
134,211/151-byte join, the absence of a lawful DR2b grid and current-C1
crosswalk, the 17,931→17,927 endpoint drift, all input hashes, both clean Python
review passes, and the refusal to invent a curve. This artifact is not FIRE,
score, promotion, campaign, or pointer authority.
