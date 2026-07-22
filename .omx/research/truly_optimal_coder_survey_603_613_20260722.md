---
title: Truly optimal coder survey for DDM v3 and Task 613
date_utc: 2026-07-22T04:47:13Z
task: 603
feeds_task: 613
lane_id: lane_truly_optimal_coder_survey_603_613_20260722
research_only: true
execution_allowed: false
score_claim: false
verdict: MEASURED_STREAM_SPECIFIC_CODER_ASSIGNMENTS_AND_REPRESENTATION_FLOOR
verdict_scope: Exact local lossless recodes and analytic bit counts over the named real S4, PPCS, and PoseNet-derived source objects; not a final archive, not a PRIMARY semantic ownership proof, and not a contest score
main_landing_review_required: true
---

# Outcome

There is no globally optimal backend for the six v3 streams. On real disk artifacts, three old and
modern families split the wins: Brotli for static masks, manifest state, and globally concatenated
exceptions; raw LZMA1 for the PPCS trajectory and event records; and a 1966 Golomb/Rice code for the
Pose6 temporal-delta proxy. The result is source-shape dependent, exactly as the no-recency doctrine
requires.

Coder swaps alone do **not** close Task #613. A noncomposable diagnostic sum of the six winning
logical streams is `267,791 B` (`446.318 B/pair`), already `3,791 B` above the historical `440 B/pair`
bar before receiver framing and before retaining the rest of PPCS. Adding the measured whole-PPCS
`78,969 B` zlib diagnostic gives `346,760 B`, which is `130,537 B` above the binding `216,223 B`
pointer cap and `192,236 B` above the `154,524 B` stretch cap. These are blocker diagnostics, not an
archive proposal: they mix legacy opaque S4 aliases and a Pose apparatus proxy.

The first cheap v3 action is unambiguous: globally recode the exception records with Brotli-Q11. It
measures `80,478 B`, versus `180,196 B` for current per-record zlib, an isolated saving of `99,718 B`
(`55.34%`). The v3 arm must still prove exact final-ZIP delta, fresh receiver parse-back, and a unique
byte home. Sparse events remain the representation bottleneck: their already-settled LZMA stream is
`181,904 B` by itself, larger than the entire strict cap.

# Ranked per-stream assignments

`MEASURED` means exact local compressed length or exact integer-code bit count. `DERIVED` means an
analytic ceiling without a landed byte stream. All generic-codec rows are stream-local and exclude
changed final-container framing.

| Live stream | Real source shape | Ranked code candidates | Bytes | Theorem / rate reason | Verdict |
|---|---|---|---:|---|---|
| static ground coefficients | Three dense-but-run-regular binary masks, `N=196,608` each; ones `45,706 / 97,206 / 50,345`; only `287 / 136 / 30` one-runs | **1 Brotli-Q11**; 2 Elias-delta run gaps; 3 Rice run gaps; 4 LZMA1; colex loses badly | **610**; 643; 683; 824; 63,990 | Golomb is optimal for geometric run lengths; Elias is universal over unknown integers; Cover colex approaches `log2 C(N,K)` for constant-weight sets, but density makes support enumeration expensive | Keep the settled `610 B` Brotli stream. Explicit old-math run coding is close, not better. |
| xi-curve knots | PPCS trajectory: 10 controls + 254 AR residuals; schema wire `1,063 B` | **1 raw LZMA1**; 2 Brotli-Q11; 3 zlib; 4 colex-time + Rice-values | **204**; 205; 224; 334 | Enumerative support costs `643.285` bits; value Rice terms cost `135 + 1,822` bits plus header. Dictionary modeling captures additional repeated syntax. | Put LZMA/Brotli behind one deterministic selection tag and measure the final v3 payload; a one-byte lead is not universal. |
| Pose6 dxi residuals | Real `gt_poses[600,6]` mapped to existing DDM ordinal uint8 target-code proxy; 3,594 temporal deltas, 3,558 nonzero | **1 temporal delta + Rice k=6**; 2 raw; 3 Brotli | **3,509**; 3,600; 3,604 | Golomb/Rice is the exact geometric-source candidate, with `k` selected by actual bit count. | Use a `min(raw,Rice)` tag on the actual v3 dxi stream. The current `91 B` win is only `2.53%` and the proxy is not a complete dxi/Pose authority stream. |
| sparse events | 10,919 PCE3 records, 812,231 position values, 10,064 nonempty sets | **1 raw LZMA1**; 2 Brotli; 3 zlib; 4 KT1 ceiling; explicit position codes lose | **181,904**; 185,327; 206,145; 207,227; Elias-delta positions 324,606 | LZMA's dictionary model wins empirically. Per-set enumerative/gap costs are swamped by large high-entropy supports and framing. | Keep settled LZMA. This stream alone is `117.72%` of the strict cap; change event representation, not merely entropy backend. |
| entropy state | 2,534-byte canonical S4 manifest, small structured text | **1 Brotli-Q11**; 2 zlib; 3 LZMA1 | **1,086**; 1,287; 1,294 | Small-stream model/header overhead dominates; exact bytes decide. | Brotli current bytes. Schema elision is a new representation and needs receiver proof. |
| exceptions | 1,802 PCOMP3 records, globally framed; 1,094,309 site values | **1 global Brotli-Q11**; 2 global LZMA1; 3 global zlib; 4 KT1 ceiling | **80,478**; 82,620; 100,126; 103,824 | Cross-record dictionary/context reuse dominates. Independent gap/colex position codes are much larger. | Top v3 probe. Isolated stream result only until final archive and receiver closure. |
| full PPCS seed/control | Whole legacy seed object, `884,872 B` raw | existing zlib diagnostic; representation transforms below | **78,969** | This is a description object, not an entropy-optimality proof or complete archive. | Retain as budget input. Do not double-count the 204-byte extracted trajectory when composing a candidate. |

The full machine-readable matrix, source hashes, implementation definitions, and all runner-up bytes
are in `truly_optimal_coder_measurements_603_613_20260722.json`.

# Year-blind candidate disposition

The oldest candidates remain live where their source assumptions match:

- Stern-Brocot/Farey and continued fractions can shorten rational coefficient descriptions; Gauss/
  Lagrange lattice reduction can shorten correlated integer coefficient vectors; Chebyshev,
  Legendre, Gram, and Whittaker can change the xi basis or sampling grammar. None is a lossless
  recoder of the frozen bytes measured here. They require a new receiver semantic object plus the
  same Seg-cell/Pose-tube constraints, so no byte row is fabricated.
- Cover enumerative/colex coding was measured on actual mask/event supports. It is excellent only
  when the support family is small enough; on the dense static masks and large event/exception sets
  it loses by orders of magnitude.
- Canonical-Huffman length-vector rank is leaderboard-proven when the object being sent is a Kraft-
  valid code-length vector. None of the six extracted real objects is such a vector, so re-labeling a
  payload as L26 would be a type error; keep it for a future small-alphabet codebook description.
- Golomb/Rice and Elias were measured, not dismissed as old. Rice wins the Pose temporal proxy;
  Elias-delta nearly reaches Brotli on the static runs.
- Huffman/Tunstall, PPM/CTW, BWT, ANS/rANS/tANS, and learned priors remain model/code candidates, not
  automatic improvements. At these small stream scales, model tables and termination count. The
  existing strict arithmetic container and KT0/KT1 rates do not beat the measured winners. A new CTW,
  PPM, or ANS row must count its model/table/termination/decoder bytes and prove parse-back.
- Slepian-Wolf/Wyner-Ziv side-information bounds become relevant only if the decoder already owns a
  correlated side stream. No such free side-information contract is assumed here.

# #557 and historical-anchor reconciliation

The historical PR101/PR103 lineage does preserve a real arithmetic win: one merged constriction range
stream over the eight largest weight streams plus `latent-hi` saved about `290 B` against the
PR100-style baseline. That is the source of the “adaptive arithmetic beats on latent-like streams”
anchor. It is not reproduced by the current #557 receipt: its current-donor pair-code section is
`20,518 B` under Brotli, versus `35,989 B` for repository IID arithmetic and `37,432 B` for repository
spatial arithmetic. Current base weights are `63,394 B` Brotli versus `66,322 B` IID arithmetic.

Both results are retained with their substrate/model/grouping scope. #557 rejects only its concrete
current-donor context models; it does not close arithmetic coding as a family. The valid cross-task
lesson is that coder choice is stream-shape dependent and every proposed model must be measured on
the exact final stream.

# Action for v3 and Task 613

1. Add deterministic codec tags and exact parse-back for `exceptions={brotli,lzma}` and
   `xi={lzma,brotli}`; select by complete framed final-archive bytes, not isolated payload bytes.
2. Add `pose6={raw,Rice(k)}` with the selected `k` and all headers counted; remeasure on the actual
   signed dxi residual stream, not the ordinal proxy.
3. Leave `static=Brotli`, `events=LZMA`, and `entropy=Brotli` until a changed real payload beats them.
4. Feed the `99,718 B` exception recode as a hypothesis, not an archive delta. The receiving arm must
   emit the final-ZIP marginal and receiver proof.
5. Route subsequent effort to event/PPCS representation reduction. Coder-only waterfill has no path
   to the strict box on the measured objects.

# Custody and stores consulted

- The delegated authority, `CLAUDE.md`, `AGENTS.md`, and
  `docs/operating_manual_craft_handoff.md` (SHA-256
  `40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212`).
- The one-hop coder corpus index and no-recency doctrine; online compression-theory ledger; AQC1
  container audit; #557 build spec, memo, and 2,057,039-byte machine receipt.
- Task #603 owner bundle, target/cap receipts, v1/v2 receiver artifacts, n600 membership receipt, and
  the live v3 arm prompt/inbox.
- Task #574 PPCS trajectory receipt, Task #610 wrong-level sweep, Task #613 budget surfaces, settled
  S4 archive, and read-only `gt_n600.npz` Pose cache.
- Primary publications listed in the companion reference mine. Publication year was never a ranking
  feature.

No source artifact, sacred run directory, scorer, GPU, paid provider, contest evaluator, or pointer
was mutated or invoked. MAIN landing review is required before these assignments enter v3.

0.1910828242 [contest-CPU] — unchanged by construction.
