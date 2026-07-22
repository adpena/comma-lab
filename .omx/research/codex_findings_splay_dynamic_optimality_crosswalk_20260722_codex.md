---
schema: codex_findings.v1
date_utc: 2026-07-22T13:26:15Z
lane_id: lane_splay_dynamic_optimality_crosswalk_20260722
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
verdict_scope: paper-to-live-consumer crosswalk; no coder implementation, payload measurement, launch, archive candidate, or score result
---

# Splay dynamic-optimality crosswalk - one real coder race, no imported rate theorem

## Outcome first

| Rank | Disposition | Named consumer | Evidence and decision | $0 exit / one-line reason |
|---:|---|---|---|---|
| 1 | **ADOPT** - measurement entrant only | `truly_optimal_coder_survey_measurement.v1`, with real stream extraction from `src/tac/optimization/direct_description_entropy_streams.py`; a measured winner may then feed `direct_description_entropy_priced_member.py` | **DERIVED design / UNMEASURED entrant.** Add separately named `jones_splay_prefix_v1` and `mtf_rank_v1` rows on the exact PCE3 event, PCOMP3 component/exception, and nonempty G2CS1 correction-symbol sequences. Compare complete framed bytes with each stream's measured incumbent and the separately labeled KT1 ceiling. | Exact local encode/decode and byte counting only: exit `KEEP_FOR_ARCHIVE_COMPOSITION` iff parse-back is exact and complete framed bytes are strictly below the incumbent on at least one real stream; otherwise exit `N-A_AFTER_MEASUREMENT` per stream. No scorer, GPU, dispatch, or paid service. |
| 2 | **ALREADY-HAVE-BETTER** | v6 `fixed_ar1_hold24` / `xi_pose6_ar1_hold24` in `direct_description_entropy_priced_member.py`, plus `src/tac/optimization/xi_temporal_delta_coder.py` | **MEASURED repository evidence.** Hold24 already measures `107.958333` and xi-hold24 `107.937500 B/added-pair`; the xi coder measures exact n600 terminal bytes. The paper's working-set and dynamic-finger statements bound BST access time, not Kraft-valid code length or archive bytes. | No rate-law adoption: direct receiver-framed byte measurements are sharper. The paper supplies locality intuition only unless the prefix-coder bridge in the equations note is proved and wins the row-1 race. |
| 3 | **N-A** | `project_pixelwise_seg_relaxation()` and `project_rank6_pose_ellipsoid()` in `src/tac/optimization/v10_constructive_solver.py` | **CODE-DERIVED.** The live snapshot uses vectorized dense arrays, a sort of exactly six RGB box crossings followed by at most seven intervals, and a rank-at-most-six Gram solve. `N_CLASSES=5`. There is no long-lived ordered dictionary with a locality-sensitive access sequence. | Current sizes and structure do not match a self-adjusting BST. Reopen only if a profiler identifies a dominant large-`n`, online ordered-set hot loop; none is evidenced here. |
| 4 | **N-A** - method note only | offline-comparator review of `direct_description_entropy_priced_member.py` / Task #613 | **PUBLISHED THEOREM, NON-TRANSFERRED OBJECTIVE.** Charging an online algorithm directly to offline OPT is a useful review pattern, but the paper's OPT minimizes BST accesses plus rotations, not contest description length or evaluator action. | No consumer can ingest the asymptotic factor without a proved objective-preserving reduction from archive decisions to the root-BST model. Do not turn the analogy into a bound. |

**One-line verdict:** race the real Jones/MTF coding family once, but import none of the 2026 BST
time bounds into v6/xi rate claims or V10 data structures.

## What the paper actually proves

The primary artifact is the 68-page local PDF
`.omx/research/papers/arxiv_2607.18498_splay_almost_dynamically_optimal.pdf`, SHA-256
`60b3213d9380f9c2d4da2549133d0222960ded8a0956aa10429c569148e7c5e7`. The model maintains
an ordered `n`-key BST. An access must leave the accessed key at the root; its cost is one plus the
number of rotations.

For an access sequence `X`, Theorem 1 proves

`cost(splay, X) = O((cost(OPT, X) + n) log log n log^2 log log n)`.

Here OPT is an offline dynamic BST that knows the whole sequence and chooses its initial tree. This
is the first sublogarithmic competitive ratio for splay trees, not a constant-factor resolution of
the dynamic-optimality conjecture.

The proof does **not** use an epoch decomposition in arXiv v1. It serializes each access by letting
offline OPT finish its access/rotations before splay acts. It then replaces OPT by a constant-factor
normal-form tree `T*`: ribless, depth `O(log n)`, with rotations only at constant depth. Ranks are
depths in `T*`; heavy paths and lazy intervals absorb rank changes; a pairing-heap abstraction
bounds zig-zigs; bends bound zig-zags. The final proof uses `cost(T*) = O(cost(OPT))`.

The appendix records the already-known working-set and dynamic-finger properties of splay trees.
The conclusion says the new competitiveness result also gives unified- and lazy-finger bounds with
the paper's polylogarithmic multiplicative and initialization factors. Every one of these quantities
is BST operation time. None is stated as compressed length.

## Why row 1 is nevertheless a real entrant

Jones's 1988 work is the separate bridge. The author's
[University of Iowa page](https://homepage.cs.uiowa.edu/~dwjones/compress/) describes adapting
splay balancing to the trie of an adaptive prefix code and links the paper *Application of Splay
Trees to Data Compression*, Communications of the ACM 31(8), 996-1007,
[DOI 10.1145/63030.63036](https://doi.org/10.1145/63030.63036). This establishes a real codec
family; it does not make the 2026 theorem a byte theorem.

The race must keep two interpretations separate:

1. `jones_splay_prefix_v1`: emit the pre-update prefix path, then apply a deterministic splay-prefix
   update at encoder and decoder.
2. `mtf_rank_v1`: deterministically move the referenced token to the front and encode the resulting
   rank with a fully specified integer backend. This is a different grammar, not an alias for Jones.

Both need a sealed initial alphabet/order, explicit EOS or length, strict trailer rejection, exact
parse-back, and all framing/model/selector bytes counted. The race inputs are semantic token
sequences before the incumbent entropy backend, never the already compressed `.lz`/`.br` bytes.
Tokenization choices are separate candidate IDs so a favorable tokenization cannot be misattributed
to the tree update.

### Preregistered comparator rows

| Real stream | Incumbent evidence to beat | Splay/MTF race rule |
|---|---|---|
| sparse PCE3 events | **MEASURED:** raw LZMA1 `181,904 B`; Brotli `185,327 B`; zlib `206,145 B`; **DERIVED:** KT1 ceiling `207,227 B` | Same source-object hash and token grammar; compare exact complete frame. KT1 remains a rate ceiling, not a fake bitstream. |
| global PCOMP3 exceptions/components | **MEASURED:** Brotli-Q11 `80,478 B`; raw LZMA1 `82,620 B`; zlib `100,126 B`; **DERIVED:** KT1 ceiling `103,824 B` | Preserve global cross-record scope and exact record reconstruction. Do not compare a per-record splay stream with the global incumbent. |
| G2CS1 correction symbols | V9 n256 outer archive is **MEASURED advisory** at `72,397 B`, but carries zero G2CS1 symbols | Wait for a nonempty receiver-admitted V10 symbol stream. Empty-input/header-only results are N-A and cannot win. |

The existing #557 current-donor row is a useful prior, not a family closure: Brotli's pair-code
section is `20,518 B`, versus `35,989 B` for repository IID arithmetic and `37,432 B` for repository
spatial arithmetic; base weights are `63,394 B` under Brotli versus `66,322 B` under IID arithmetic.
The historical PR101/PR103 grouping still preserves an approximately `290 B` arithmetic win on a
different stream grouping. These opposite rows are exactly why the splay/MTF decision is per-stream.

The authority also names #604 U5 coder races. This isolated snapshot has task-#604/U-list references
in the operator P0 ledger but no identifiable U5 coder measurement artifact or implementation path.
MAIN must add any subsequently landed exact U5 row to the comparator set rather than letting this
arm invent its bytes. That absence does not block the PCE3/PCOMP3 $0 race.

**INFERRED expectation:** a loss is more likely than a win at these sizes because integral prefix
lengths and adaptation state compete against current dictionary, Rice, and context coders. That is
not a family negative. Locality may still pay on event/component symbols, so exact bytes arbitrate.

## Why the temporal bound does not transfer

The working-set statistic counts distinct keys since a previous access; the dynamic-finger statistic
uses distance in a chosen total order. Neither is a probability assignment. Without a receiver-shared
prefix tree and a proved mapping between emitted path length and the paper's ordered-key BST cost,
there is no Kraft-valid code and no code-length inequality.

The repository already has stronger same-object evidence:

- v6 reduces the n64-to-n256 marginal from `1067.317708` to `107.958333 B/pair` with fixed hold24
  and to `107.937500 B/pair` with xi/Pose6 hold24;
- the xi temporal coder directly measures n600 terminal bytes: settled LBND2 `35,393 B`, identity
  xi-context `42,413 B`, and the planar-3 composed-screw predictor `43,901 B`.

Those are exact framed byte observations for named formulations. A BST runtime theorem cannot
sharpen them. It can motivate the row-1 coder race, and nothing more.

## V10 encoder-side structure audit

The isolated snapshot contains `src/tac/optimization/v10_constructive_solver.py`, not a separate
`ddm_v10` research memo. Its real hot mathematics is array/KKT work:

- scorer geometry is `384 x 512` with five classes;
- the per-pixel box projection sorts six crossing values and scans at most seven intervals;
- Pose is reduced to an at-most `6 x 6` Gram eigensystem;
- pair chunks are processed in a deterministic resumable list order.

Replacing any of these fixed-dimensional/vectorized operations with pointer-based splaying would
change constants and memory access without exposing the access-sequence objective in the theorem.
The negative is scoped to this checked V10 snapshot. A future large online ordered-set loop remains
open, subject to profiling.

## Triality, blocker delta, and pointer honesty

- **DAG:** `splay_dynamic_optimality_crosswalk_DAG_FEED_20260722.md` preregisters the only adoptable
  measurement path and its fail-closed gates.
- **Equations:** `splay_dynamic_optimality_crosswalk_canonical_equations_20260722.md` separates the
  paper theorem, the unproved prefix bridge, and exact contest byte admission.
- **DSL/config:** no live lever is justified before a measured per-stream win. The two coder modes
  remain separately named preregistered interpretations, not invented launcher flags.

Blocker delta: **none to this research-only crosswalk**. The future G2CS1 row is intentionally
ineligible until a nonempty admitted symbol stream exists; the PCE3/PCOMP3 race can proceed at $0.

## STORES CONSULTED

STORES CONSULTED: delegated authority and verified SHA-256; governing contracts; the local paper;
current coder, v6/xi, V9, and ddm_v10 artifacts named below; canonical state stores; and the Jones
1988 primary author page and DOI record.

- Delegated authority file, SHA-256
  `f4eab38395a9f9296a968ec6291692cd9d5bd2ad5515b8915fd407b280641567`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and
  `docs/operating_manual_craft_handoff.md` (verdict first, primary-artifact re-derivation,
  evidence labels, and adversarial conclusion check followed here).
- Local arXiv PDF and rendered pages 1, 3, 13, 48, and 56; model, theorem, proof overview,
  conclusion, normal-form appendix, and working-set/dynamic-finger appendix were read.
- Douglas W. Jones's University of Iowa primary author page and DOI `10.1145/63030.63036`.
- `.omx/research/truly_optimal_coder_survey_603_613_20260722.md`, its machine measurement JSON,
  reference mine, DAG FEED, and canonical-equations note.
- `src/tac/arithmetic_qint_codec.py`, `src/tac/optimization/arith_selfcomp_rate_coders.py`, and
  `.omx/research/arith_selfcomp_rate_coders_20260719_codex.md`.
- v6 memo/receipt/equations, xi temporal coder source and n64/n600 measurement, V9 findings/receipt,
  and `src/tac/optimization/v10_constructive_solver.py` plus its focused tests.
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, both
  delegated inboxes, `.omx/state/operator_p0_ledger.jsonl`, and the current branch/index state.

No source artifact, scorer, evaluator, provider, GPU, run directory, archive, or main worktree was
mutated. MAIN landing review is required before the ADOPT row becomes a build or measurement task.

`0.1910828242 [contest-CPU]` - unchanged by construction.
