---
schema: research_dag_feed.v1
feed_id: FEED-SPLAY-DYNAMIC-OPTIMALITY-20260722
date_utc: 2026-07-22T13:26:15Z
lane_id: lane_splay_dynamic_optimality_crosswalk_20260722
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# FEED-SPLAY-DYNAMIC-OPTIMALITY-20260722

## Outcome

One edge is adoptable: add Jones-style splay-prefix and separately specified MTF-rank candidates to
the existing real-stream coder survey. No theorem-to-rate, theorem-to-V10-data-structure, launch,
or score edge is admitted.

```text
[P0 local PDF: ordered-key root-BST cost theorem]
           |
           +--> [P1 theorem/objective audit]
           |          +--> REFUSE rate-law import
           |          +--> REFUSE V10 hot-loop import
           |          `--> method note only: compare directly with offline OPT
           |
[P2 Jones 1988 splay-prefix family] + [P3 MTF working-set family]
           |
           v
[G0 sealed semantic token stream + source hash]
           |
           v
[G1 deterministic frame + strict exact parse-back]
           |
           v
[G2 complete framed bytes, state, EOS, selector counted]
           |
           v
[G3 same-scope race vs per-stream incumbent + labeled KT1 ceiling]
           |
      +----+------------------+
      |                       |
  no strict win           strict local win
      |                       |
      v                       v
[N-A_AFTER_MEASUREMENT] [G4 final archive marginal + unique byte home]
                              |
                         +----+-----+
                         |          |
                       fail       pass
                         |          |
                         v          v
                    [local only] [eligible for MAIN-reviewed composition]
```

## Nodes and typed evidence

### P0 - Paper theorem

Input: local PDF SHA-256
`60b3213d9380f9c2d4da2549133d0222960ded8a0956aa10429c569148e7c5e7`.

Typed output: `splay_dynamic_bst_competitiveness_v1`, **PUBLISHED THEOREM**. It controls the root-BST
cost `1 + rotations` on an ordered-key access sequence. It has no archive-byte authority.

### P1 - Objective audit

Typed refusals:

- `NO_CODE_LENGTH_REDUCTION`: no proved reduction from Jones prefix-trie length, MTF rank length,
  Shannon/KT log-loss, LZ dictionary bytes, or ZIP bytes to the paper's root-BST objective.
- `NO_V10_ORDERED_SET_CONSUMER`: the checked V10 path is dense vectorized KKT work with six box
  crossings, seven intervals, five classes, and a rank-six Pose solve.
- `METHOD_NOTE_ONLY`: direct charging to a real offline comparator is a review discipline, not a
  repository equation or implementation.

### P2/P3 - Candidate interpretations

They are deliberately distinct:

- `jones_splay_prefix_v1`: a decoder-synchronized splay-prefix tree.
- `mtf_rank_v1`: a decoder-synchronized move-to-front list and a separately named integer rank
  backend.

No payload result may collapse these into a single `splay_mtf` label.

### G0 - Source custody

Each row must bind:

- semantic object schema and source SHA-256;
- tokenization ID and ordered alphabet;
- record aggregation scope (`per_record` or `global_stream`);
- empty/nonempty status;
- incumbent row source hash and scope.

Preregistered live surfaces:

| Surface | Consumer/extractor | Eligibility |
|---|---|---|
| PCE3 sparse events | `src/tac/optimization/direct_description_entropy_streams.py` and the survey's real event extractor | eligible now; keep global/per-record scope identical to comparator |
| PCOMP3 exceptions/components | same survey contract and direct-description entropy stream | eligible now; global stream required to match the incumbent |
| G2CS1 correction symbols | `src/tac/optimization/direct_description_carrier_compose.py` | only after V10 produces a nonempty receiver-admitted stream |

### G1 - Deterministic receiver closure

Required checks:

1. decode(encode(tokens)) is exactly identical, including token dtype/order;
2. repeated frames are byte-identical;
3. malformed prefix paths, invalid ranks, truncation, and trailers refuse;
4. encoder and decoder begin from the same sealed state and make identical updates;
5. no source/scorer-derived table is hidden in receiver code.

### G2 - Complete bytes

`B_local` counts magic/version, dimensions or symbol count, alphabet/state seed, EOS/termination,
model/backend tags, padding, integrity fields, and payload. Any decoder-code delta is separately
reported and must enter `B_archive` at the composition gate.

### G3 - Exact same-scope race

Current comparison anchors:

| Stream | MEASURED incumbent | DERIVED ceiling (not a bitstream) |
|---|---:|---:|
| PCE3 events | raw LZMA1 `181,904 B` | KT1 `207,227 B` |
| PCOMP3 global exceptions | Brotli-Q11 `80,478 B` | KT1 `103,824 B` |

Decision: keep an entrant only when its complete `B_local` is strictly smaller than the same-source,
same-scope incumbent. Beating KT1 but not the measured incumbent is a diagnostic, not adoption.

### G4 - Final archive marginal

A local winner remains a hypothesis until the unchanged semantic candidate is rebuilt with exactly
one payload home, receiver parse-back passes, and

`delta_archive_bytes = candidate_archive_bytes - incumbent_archive_bytes < 0`.

No distortion delta is inferred from losslessness; final composition must verify semantic identity.

## Six-hook wire-in

| Unified hook | Disposition |
|---|---|
| Sensitivity map | N-A before a byte win: the candidate is lossless and does not allocate distortion. Source/stream IDs provide the routing key. |
| Pareto constraint | Bind exact `delta_archive_bytes`; semantic identity requires `delta_d_seg=delta_d_pose=0` only after receiver proof. |
| Bit allocator | A measured row may enter the per-stream coder selector as one more backend. No global winner default. |
| Cathedral/autopilot | Fail closed while `research_only=true`; MAIN may expose only a measured, parse-backed, final-archive-negative row. |
| Continual-learning posterior | Append a typed survey measurement only after the real race. This paper crosswalk itself is not an empirical anchor. |
| Probe disambiguator | Race `jones_splay_prefix_v1` and `mtf_rank_v1` independently on the same token streams; exact bytes arbitrate. |

## $0 exit criteria

1. `N-A_AFTER_MEASUREMENT(stream, mode)` if parse-back fails or complete bytes do not strictly beat
   the measured incumbent.
2. `KEEP_FOR_ARCHIVE_COMPOSITION(stream, mode)` if complete local bytes strictly win; this is not
   promotion.
3. `ARCHIVE_NEGATIVE(stream, mode)` if code/selector/container effects erase the local win.
4. `MAIN_REVIEW_ELIGIBLE(stream, mode)` only when exact semantic parse-back, unique byte home, and
   negative final archive delta all hold.

Expected result is honestly `N-A_AFTER_MEASUREMENT`, but the family remains open until exact real
bytes are observed.

## FORMALIZATION_PENDING

`splay_prefix_bridge_guard_v1` and `lossless_coder_race_admission_v1` are specified in the companion
canonical-equations note but are not registered as executable equations. No DSL lever, planner
edge, or launch configuration is created by this research-only feed.

## STORES CONSULTED

STORES CONSULTED: delegated authority; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; local arXiv PDF and rendered theorem/proof/corollary pages;
Jones University of Iowa primary author page; truly-optimal coder survey, machine receipt, DAG, and
equations; arithmetic qint/self-composition coder surfaces; v6 receipt/equations; xi coder receipt;
V9 findings/receipt; V10 constructive solver/tests; canonical lane, task, frontier-report, progress,
operator-P0-ledger #604 references, and delegated inbox surfaces.

MAIN landing review is required. `0.1910828242 [contest-CPU]` is unchanged.
