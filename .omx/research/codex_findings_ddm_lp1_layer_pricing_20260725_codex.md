# Codex findings — DDM LP1 #669(b+c) layer pricing

`research_only=true` · `execution_allowed=false` · `score_claim=false` ·
`promotion_eligible=false` · `pointer_moved=false` ·
`main_review_required=true`

## Verdict

`CORRECTED_C1_MEASURED_ALLOCATION_IS_SEEDED_CONTROL_ONLY;`
`G4_FREE_CONTEXT_GAIN_IS_FUTURE_STREAM_SCOPED;`
`ALL_25_L3_EXCEPTIONS_HOLD_ZERO_AT_MS7_R0`

The corrected C1 allocation is **134,211 exact measured bytes**, not the
200,000-byte planning equality. The remaining **65,789 bytes are unallocated
headroom**, not an admissible reserve. No G4, DM1, or DM4 byte result was
subtracted from a different semantic object.

Typed receipt:
`.omx/research/ddm_lp1_layer_pricing_20260725T031654Z/ddm_lp1_layer_pricing_receipt.json`

Receipt SHA-256:
`6bd6a5baaa8f5995e93ef594e880beac77e9aa2b2083e661598c84feaba13fd5`

## C1 stream-home audit

All 13 source budget rows are represented. Accounting rows are not
double-charged, and reserves are not relabeled as measurements.

| C1 row class | deepest proven home | type | corrected treatment |
|---|---|---|---|
| predictor + movable worldsheet | L2 chart/grammar | CONTEXT | retain exact measured home bytes |
| receiver profile, manifest, ZIP framing | L1 program/container | PROGRAM | charge video-derived/profile/container bytes; generic receiver code remains free |
| solved scorer template | L4 scorer feature | FIBER | retain 151 measured bytes |
| lane program seed | L2 chart/grammar | CONTEXT | retain 270 measured bytes |
| exception/application reserves | L4 scorer feature | RESIDUAL | allocate zero until a receiver-closed marginal pays |
| coder/container contingency | L1 program/container | PROGRAM | allocate zero; CC2 owns new codec selection |
| subtotal + hard total | accounting only | n/a | recompute, never double-charge |

The C1 fixed exact subtotal is 133,941 bytes. Adding the measured 270-byte
lane seed gives 134,211 bytes. The four reserve rows sum to 65,789 bytes, so
the original 200,000-byte equality remains only a ceiling.

## Residual versus free context

| measured object | explicit bytes | context bytes | counted context params | delta bytes | disposition |
|---|---:|---:|---:|---:|---|
| G4 aggregate pixel-time innovation | 490,794 | 401,633 | 0 | +89,161 | KEEP context for this future stream |
| G4 boundary-distance proxy | 490,794 | 683,211 | 0 | -192,417 | DROP context |
| DM1 25-row semantic records | 4,124 | 1,569 | 0 | +2,555 | KEEP shared semantic container |

The generic context implementation is free interpreter code. Any future
video-derived context parameters must be charged. G4 has not measured a
same-object context encoding for the current C1 or the MS7 flat receiver
object, so applying its 89,161-byte gain there would be fake accounting.

## Stream x stratum costates

The receipt has 25 SENSE rows: 16 boundary and nine cell.

- Every row's deepest information home is L4.
- Historical boundary `SKELETON` is preserved as source provenance and mapped
  to `RESIDUAL` only because this task seals the five-type
  GAUGE/FIBER/RESIDUAL/CONTEXT/PROGRAM vocabulary.
- Cell rows remain `FIBER`.
- The per-row real-coder semantic prices sum to 4,124 bytes.
- The per-row DM4 L3 realization prices sum to 2,871,312 bytes; the shared
  joint L3 object is 3,241,321 bytes.
- DM4's current joint score delta is +1.9515297285056081.
- MS7 reports no mass-paying R0 row, so all 25 allocations are zero and all
  states are `SENSE_HOLD_ZERO_R0`.

## Scoped negatives

- The current DM4 corrected-J/resize-adjoint/ERF/stem-lattice menu is
  nonpaying for this SHA-bound 25-row instance. This is not a representation
  family impossibility result.
- G4 boundary-distance context loses on its measured innovation stream. This
  is not a negative on decoder-derived context generally.
- The 25 MS7 rows fail the current cheapest guaranteed-reach bound. A dynamic
  scorer-recursive reach curve or new same-object context price may reopen a
  row.
- No contest score, promotion, minimum-description result, or frontier
  movement is claimed.

## Durable integration

- Typed compiler:
  `src/tac/optimization/ddm_lp1_layer_pricing.py`
- Materializer:
  `tools/build_ddm_lp1_layer_pricing.py`
- Canonical equation:
  `ddm_lp1_deepest_home_context_waterfill_v1`
- DAG:
  `.omx/research/ddm_lp1_layer_pricing_DAG_FEED_20260725.md`
- CC2 boundary: LP1 selected no new codec and preserves source codec labels.

Two clean post-fix review passes are recorded in
`.omx/research/reviews/ddm_lp1_layer_pricing_reviews_20260725.json`.
The focused surface passed 8 tests; the adjacent DM1/DM2/DM4/MS7 surface
passed 36 tests; and the canonical registry surface passed 29 tests.

## MAIN landing review

MAIN must independently verify:

1. the 13-row C1 inventory and 133,941 + 270 = 134,211 accounting;
2. the same-object firewall on the 89,161-byte G4 gain and 2,555-byte DM1
   semantic gain;
3. the `SKELETON` provenance to five-type `RESIDUAL` mapping;
4. all 25 DM1/DM2/DM4/MS7 foreign-key joins and zero allocations;
5. the canonical registry append and CC2 non-overlap;
6. that pointer, score, promotion, and execution authority remain false.

The lane registry validator also reports 110 pre-existing legacy
missing-evidence paths. None belongs to the new LP1 L0 research-only lane; do
not interpret that repository-wide debt as LP1 validation.
