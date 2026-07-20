# DAG FEED — R1b2 moderate-margin full-kernel MDL + xi[0] compile

`research_only=true` · `$0 LOCAL` · `[macOS-CPU advisory]` · pointer `0.19108 [contest-CPU] UNMOVED`

## State transition

| node | required authority | state at frozen source UTC `2026-07-20T18:12:20Z` | outgoing edge |
|---|---|---|---|
| `C2_control` | exact archive + production receiver + n600 batch16 hard oracle | SETTLED: `94,344 B`, `d_seg=.003515794640406966`, `d_pose=127.36588287353516`, `S=36.10275630841103` | input only; do not remeasure |
| `VJP_n600` | terminal 600 unique immutable per-pair sidecars, no refusal, fresh byte rehash | BLOCKED: `184/600`, `416` missing, pair `11` refused | rank-4/secant materializer |
| `J_rank4_secant` | batch16, rank4, 600 rows; separately typed Frechet tangent + realized uint8 secant per block; base/delta/norm/remainder/cell-crossing/hard-endpoint custody; exact stratum totals | ABSENT | corrected active-set QP |
| `P_bnd` | parse-backed `boundary_coordinate_packet.v1`, n600, localized curvelet/shearlet only | ABSENT | counted archive |
| `K_full_MDL` | offline exact MDL selection over bounded `[0,255]` uint8 preimages before hard oracle + exact replay, two deterministic receiver hashes, zero receiver search | ABSENT | counted archive |
| `xi[0]` | coordinate zero only, counted quantized payload, typed receiver | ABSENT | counted archive |
| `A_r1b2` | deterministic parse-backed archive; carrier delta `<=1,852 B` conditional; total `<=286,680 B` and `<=216,223 B` fixed cap | NOT COMPILED | production receiver |
| `R_decode` | exact bytes, deterministic output, no search, `<=1,800 s` | NOT RUN | hard oracle |
| `E_n600` | same seed1234/batch16 CPU-Torch n600 path as control | NOT RUN | gate/score decomposition |

## Verdict

`DECOMPOSED_PARTIAL_R1B2_PRODUCTION_CUSTODY_BLOCKED` with exact blocker set in `.omx/research/r1b2_mdl_xi0_compile_20260720T181400Z.json`. `verdict_scope=current production custody and exact control instance only; no R1b2 candidate measurement and no boundary/xi/full-kernel family negative`.

## Reactivation predicate

Reactivate only for a terminal no-refusal n600 VJP campaign and all three strict producer manifests: rank-4 first-order+realized secants, receiver-search-free full-kernel compact replay, and coordinate-zero-only xi. The measured frame-0 saturated-kernel/LLL lattice sieve may order full-kernel proposals as a preprocessing candidate, but it is lower-bound-only and must remain behind the exact bounded-intersection numerator, coder, and hard-oracle gates. Then run the compiler, parse back exact bytes, measure decode time, and invoke the inherited n600 hard oracle; do not reactivate for another partial prefix, target-only PDW2, unit/synthetic rows, or the settled control.

## Triality disposition

- DSL/control: `tools/compile_r1b2_mdl_xi0.py` is the sole explicit compile gate; no invented trainer flag.
- DAG: this file records the state transition and exact reactivation predicate.
- Equations: none added; existing measured laws are consumed without promotion.

MAIN review and merge are mandatory; this worktree is not source of truth.
