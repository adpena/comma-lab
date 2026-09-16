# ddm_psa1 — method and constraint reconciliation

This arm measures byte prices only on move52 archive
`ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e` (179332 B).
No scorer, renderer forward, pose solve, training, Modal, fire, packet, authorization,
or receiver/source-tree mutation is performed. All generated payloads are kept under
`/Volumes/APDataStore/pact/ddm_psa1/`, with hashes. Maximum retention is 1 GiB logical;
accounting excludes `.pending`/`._` and tolerates concurrent renames. The reserve is 8 GiB;
a quota failure blocks rather than deleting bytes. Immutable per-instance checkpoints and
producer snapshots make each detached stage resumable with the same command.

## RECALL EVIDENCE

`RECALL_SEARCHES.json`, `*_recall.txt`, and `EQUATIONS_RECALL.json` record searches of
research memos, canonical equations, indexes/DAGs, design documents and task ledgers.
Queries include `pair.selective`, `per.pair.frame.embedding`, `FiLM.flicker`,
`spatial.*FiLM`, `boundary.jitter`, `SM1S`, and `head.*bias`. Saved grep examples are
bounded to three/four matches per file and 140 lines per surface; they are not an
exhaustive export. The independent reviewer also searched these surfaces before design.

Beyond charter seeds, findings that changed this implementation:

- `ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md` establishes an existing
  600x8 coarse 3-bit pair embedding feeding all four blocks' FiLM. The charter's absence
  claim is false. These are new pair-specific head/last-block DOFs beyond that embedding.
- `collateral_coupling_geometry_and_film_flicker_sidecar_20260718.md` already discusses
  spatial FiLM. Its five-logit scorer-head algebra is not a renderer realization.
- `SPEC_v10_integer_plane_vehicle_20260719.md` and actual move52 code distinguish five
  SegNet classes from the renderer's three RGB outputs. B3 prices the actual bias;
  B5_control preserves the literal five-value byte question but cannot pass the gate.
- `ddm_rbf1_free_post_render_boundary_treatment_20260911.md` closes broad free treatments
  through collateral/pose; it does not close counted pair-and-spatial selection.
- `ddm_sm1_semantic_section_shared_mixer_coder_priced_closed_form_20260909.md` separates
  metadata/scales from adaptive code-stream gains. The shipped SM1S parser rejects a
  suffix; the appended-coder result therefore uses explicit research magic SM1X.
- The shipped RLC1 tail fixes five symbols and 384x512 causal HPAC geometry. Arbitrary
  int4 records have no unchanged tail-context interpretation: NOT_APPLICABLE, not zero B.
- The equations registry distinguishes scorer-head rank-four algebra, renderer foldback
  reach, model-section recode limits and same-object exchange/noise evidence. None makes
  this new section a renderer implementation or a score claim.

## Exact scope

K24 is inherited byte-identically from jrd1 (PCG64 seed 20260916, 12 pool + 12 remainder).
The twelve heaviest residual pairs in pd4's 156-pair pool are distinct from K24; no
replacement was needed. The 36-pair roster is enriched, not a population estimate.

The retained move52 argmax hash is pinned through pd4/pd5 BIND receipts and compared with
the pinned retained DALI target cache, without a scorer or GT decode. Counts reproduce
12147 cells / (600*384*512). Each cell is 100/117964800 S, equivalent to
1.273108215332031 B at 25/37545489 S/B. The median population debt is 22.915947875976556 B
(18 cells), correcting the charter's approximate 15 B. This is inherited macOS advisory
residual debt, not newly measured per-pair contest-CUDA debt. pd4's slightly different
T4-calibrated carry is reported separately when testing sensitivity.

jrx2's 24 actual proposals all target head.weight; it does not establish the cheapest
layer among all layers. A uses that observed head (3x96x3x3), flattened as 3x864. For K24,
its one-hot rank-one vectors reproduce the retained nearest-int4 direction. Heavy12 use
a deterministic existing nonzero code toward zero. This is an explicitly unoptimized
minimum-nonzero price instance, not a learned effective repair.

B3 uses one nonzero RGB bias coordinate and the layer's fp16 scales. C counts two
96-value int4 vectors (scale delta and additive shift), with one nonzero shift and
counted fp16 scales. D counts RGB bias plus up to three run-length records for a clipped
3x3 support at the first actual residual cell. All locations, keys, scales, and values
are counted. Generic shape metadata is deliberately included in the measured grammar.

## Byte legs and gate

1. Standalone: actual Brotli q11 section plus an eight-byte PSAX length/magic footer,
   appended to an otherwise byte-identical archive member. Complete singleton archive
   increments include every header/index/scale/location byte.
2. SM1X: reuse jrx2's exact section walk and the original shipped SM1S `walk`, original
   mixer weights and original model symbols; append counted descriptors and int4 codes.
   Full encode twins and final-ZIP decode are checked. Report both raw rider increment
   and final CK2/Brotli/ZIP archive increment. The latter is the rate price; a raw-stream
   saving is not silently substituted for the final counted archive cost.
3. Shipped tail: N/A by alphabet and spatial schedule. No new tail mechanism is invented.

The zero-record controls price fixed compressed framing. Raw common header is 12 B;
index is 2 B per pair; standalone footer is 8 B. Per-record dimensions, scales and mask
locations remain in the measured price. Joint36 sections are measured separately; they
are not used to amortize away the singleton gate.

A form passes if one actual complete-singleton archive leg costs strictly less than
whole pair debt for at least 4/36 pairs, and its measured empty-section fixed overhead
is under 200 B. B5_control is excluded. Passing justifies MAIN's psa2 charter; it proves
neither positive seg credit, resolved pose, all600 collateral, nor a contest score.
The public receiver does not understand PSA1/SM1X; these are research archives only.
Failure of a singleton gate does not close a form's shared-section variants or the
actuator family. Joint36 prices are aggregate measured costs, not individual marginals.

## Six-hook disposition

`research_only=true`: sensitivity/bit allocator hooks await measured efficacy; Pareto
record is this necessary price bound; autopilot/production dispatch is N/A because no
receiver exists; findings and fire order are retained for MAIN's research/queue consumer;
the standalone versus appended-coder controls disambiguate framing/container effects.
No new canonical score equation or production lever is claimed.
