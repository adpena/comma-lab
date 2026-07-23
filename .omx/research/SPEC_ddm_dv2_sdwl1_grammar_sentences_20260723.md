# DDM DV2 implementation specification — SDWL1

date_utc: 2026-07-23
lane_id: lane_ddm_dv2_grammar_sentences_20260723
delegation_checkpoint_key: codex_delegate:ddm_dv2_grammar_sentences:20260723T142512Z
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true

## Objective

Implement and measure **Scorer-Derived Worldsheet Language v1 (SDWL1)**: an
original typed language whose admitted subjects, predicates, and modifiers are
derived only from the frozen evaluator geometry and this repository's own
mathematical corpus. Compare one complete n600 whole-clip sentence with 600
complete independent descriptions using the real repository #557
left/up-context arithmetic coder and a complete outer-deflate measurement.

The measurement describes an exact declared fact inventory. It must never call
that inventory a lossless pixel partition, a receiver-closed witness, or a
contest score. `described_fraction=1` means exact parse-back of every declared
fact, not exact reconstruction of every source pixel.

## Inputs and custody

- Frozen source cache:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
  with the registered byte count and SHA-256 from the existing G1 compact
  receipt.
- Read only the `lstars`, `margins`, and `gt_poses` members by direct NPY
  memmap; do not copy or mutate the cache.
- Verify the complete source-cache SHA-256 in the real n600 run.
- The label tensor is the source for five partition-cell and five separatrix
  subjects per pair. The margin tensor supplies exact per-stratum margin-band
  counts. The six frozen pair-output scalars are preserved bit-exactly as the
  pair-screw subject.
- No provider, GPU, paid dispatch, scorer execution, or archive candidate.

## Named derivation registry

Every grammar element must reference one or more registry entries. The registry
must include, at minimum:

1. frozen `upstream/modules.py`: SegNet consumes the last frame only, while
   PoseNet consumes the official two-frame YUV6 input;
2. the shared bilinear evaluator map to 512x384;
3. the 2x2 chroma box fact that sub-two-pixel chroma structure is invisible;
4. #559 rank-four Laguerre head flip-distance law and measured Lane normal
   interval;
5. #580 resize decomposition: omit the measured kernel fraction and describe
   only `range(A)`;
6. #391 exact resize adjoint;
7. measured scorer ERF radius bands;
8. measured Fisher/margin co-location;
9. level-set/Morse-Smale cells and separatrices;
10. `se(3)` / Chasles: one pair motion is one screw object;
11. #284/#539 power-diagram cells;
12. #145/#325/#326/#327 road-frame homography, intrinsics/extrinsics, Lane
    polynomial basis, ego kinematics, horizon, and camera height;
13. #52/#73 margin polytopes, #157 waterfill/KKT duality, tropical/max-plus
    piecewise-linear structure, and Whitney bounds.

An element with no named provenance is invalid.

## Typed grammar

Define immutable typed records for:

- subjects: partition cell, separatrix, Lane chart, pair screw, resize-range
  atom;
- predicates: declare, hold, deform, topology delta, transport, project range,
  omit kernel;
- modifiers: stratum, frame role, margin band, ERF band, scale band, chroma
  phase, head normal, and road frame.

The n600 fact inventory has, per pair:

- five partition-cell records: area, exact coordinate sums, bounding box, and
  connected-component count;
- five separatrix records: horizontal/vertical cut counts and four exact
  per-stratum margin-band counts;
- one pair-screw record: the six source float64 bit patterns.

The implementation may add fields only if the real n600 receipt measures them.

## Wire syntax

- Independent magic/version and strict section framing.
- Canonical JSON lexicon/subject schema sections.
- Numeric clause sections encoded only with
  `tac.optimization.arith_selfcomp_rate_coders.encode_spatial_context_arithmetic`.
- Decoder must use the matching repository decoder, reject malformed sizes,
  unknown tags, noncanonical JSON, trailing bytes, hash drift, and
  noncanonical arithmetic streams.
- Measure at least monolithic, typed-section, and stratum-section layouts.
- A whole-clip temporal arm stores pair zero absolutely and later pairs as
  causal deltas. The independent arm resets at every pair.
- The complete compared object is
  `len(zlib9(serialize(all sections/descriptions)))`. Never substitute section
  byte deltas into an existing archive size; outer deflate couples all bytes.
- Parse back and compare the exact semantic tensor SHA-256 for every admitted
  row.

## Required measurements

Emit complete rows for:

1. 600 independent absolute descriptions;
2. one whole-clip absolute sentence;
3. one whole-clip causal-delta sentence;
4. each layout/dimension counterfactual used for selection;
5. MDL counterfactuals for explicit frame indices, repeated provenance,
   redundant event masks, and split topology vocabulary.

Each row must report:

- inner framed bytes;
- complete outer-deflate bytes and SHA-256;
- exact parse-back;
- described fact count and described fraction;
- bytes per described fact;
- temporal-sharing gain where defined.

Dimension admission is empirical: keep only dimensions that reduce the
complete full-coverage outer payload against the corresponding same-semantics
control. Prune zero-use vocabulary and any redundant syntax whose complete
same-semantics payload is not smaller. Preserve the measured counterfactuals in
the receipt.

## Files

- `src/tac/optimization/ddm_dv2_sdwl1.py`
- `tools/measure_ddm_dv2_sdwl1.py`
- `src/tac/optimization/tests/test_ddm_dv2_sdwl1.py`
- `.omx/research/ddm_dv2_sdwl1_n600_20260723/`
- `.omx/research/ddm_dv2_sdwl1_grammar_sentences_20260723_codex.md`
- `.omx/research/ddm_dv2_sdwl1_canonical_equations_20260723.md`
- `.omx/research/ddm_dv2_sdwl1_DAG_FEED_20260723.md`

## Acceptance

The focused command must pass:

```bash
python3 -m pytest -q src/tac/optimization/tests/test_ddm_dv2_sdwl1.py
```

The real n600 command must complete locally and deterministically:

```bash
python3 tools/measure_ddm_dv2_sdwl1.py \
  --source-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --output-dir .omx/research/ddm_dv2_sdwl1_n600_20260723
```

Then rerun with `--resume`; all preserved payload and receipt hashes must
remain unchanged. Run Ruff or equivalent syntax/static checks on the new
Python files. All evidence must remain
`[macOS-CPU frozen-scorer advisory]`, `score_claim=false`,
`promotion_eligible=false`, and pointer-unchanged.

## Do not touch

- no scorer weights, evaluator execution, provider surfaces, live run
  directories, source cache bytes, unrelated hot files, or canonical pointer;
- no inherited decoder, latent, format, payload, or measured-result lineage;
- no candidate archive and no score/rank/promotion claim;
- no commit or merge to MAIN from the implementation worker. MAIN review is
  mandatory after this isolated branch is committed.
