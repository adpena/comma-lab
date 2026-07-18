# MDL / digital-cell-complex measurement specification — 2026-07-18

`research_only=true`

`lane_id=lane_mdl_ms_complex_k_lower_bound_20260718`

`execution_class=cached-local-CPU-read-only-inputs`

`score_authority=[macOS-CPU advisory] NON-PROMOTABLE`

## 1. Question and mandatory correction

Measure concrete description lengths for the frozen n600 SegNet argmax partition and the
existing temporal-ξ sidecar. Do **not** report a concrete MDL length as a lower bound on
universal individual Kolmogorov complexity.

For a fixed decoder `D` and emitted code `c(T)`, the theorem-safe relation is

\[
K_U(T\mid D) \le |c(T)| + O_U(1),
\qquad
K_U(T) \le K_U(D)+|c(T)|+O_U(1).
\]

Keep two objects separate. The fixed evaluator `E` maps an exact witness `Y` to
`T_E=(S,P)`, where `S` is the SegNet argmax partition and `P` is the frozen PoseNet output.
The prompted carrier-description object `T_C=(S,xi_quantized)` contains temporal ξ instead; `E`
does not output ξ. The true, uncomputable quantity `K(T_E|E)` is a lower bound on exact-witness
complexity up to a machine constant. A measured code length is an upper bound on its own decoded
object, so substituting that length for `K(T_E|E)` is an invalid reversal. The numeric universal
lower-bound verdict must therefore be `TRIVIAL_ONLY`; the threshold verdict must be
`INCONCLUSIVE_FOR_UNIVERSAL_K`.

The measured object is a **digital frozen-scorer argmax cell complex**. `lstars` does not contain
a continuous scalar potential, Hessians, transversality evidence, full runner-up logits, or an
exact continuous tie locus, so the tool and memo must not promote it to a classical
Morse–Smale complex.

## 2. Immutable inputs and custody

The tool accepts explicit paths and reads them without mutation:

1. `gt_n600.npz`
   - expected SHA-256:
     `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
   - `lstars`: `(600,384,512)`, integer labels exactly in `[0,4]`
   - `gt_poses`: `(600,6)`, cached frozen-PoseNet target coordinates; diagnostic only, not ξ
   - canonical class order:
     `[Road, Lane, Undrivable, Movable, MyCar]`
2. `necessity_dseg_calibration_20260715/summary.json`
   - consume only the exact-geometry `eps["0.0"]` row for this audit
   - `brotli_q11_bytes=457528` is a measured inner coordinate stream
   - `brotli_q11_bytes_shared_edge_adjusted=228764` is a derived arithmetic estimate, not
     emitted bytes and not a decoder-closed code
3. `reports/r1_dxi_238/n600_shipdxi.json`
   - `xi_bytes=6634` is the measured quantized entropy payload
   - `pose_carrier_section_bytes=7195` is the measured self-contained counted section
   - `xi_q_levels=4096`; the stream is lossy relative to continuous/cached targets
   - measured realized `d_pose=0.0016095471538913576`, hence it cannot support a `(0,0,K)` claim
4. The existing context-partition codec
   `src/tac/boundary_math/context_partition_codec.py` supplies a real self-describing,
   lossless Seg code. Its emitted bytes are a model-family upper bound, not a K lower bound.

The tool must memory-map ZIP_STORED `.npy` members, using the established local-header parser
pattern in `tools/measure_rate_law_ladder_owed.py`, so it does not load the 5 GB cache.

## 3. Tool ownership and interface

Implement only:

`tools/measure_mdl_ms_complex_k_lower_bound.py`

Required typed arguments:

- `--cache-npz`
- `--necessity-summary`
- `--xi-receipt`
- `--out`

Optional verification argument:

- `--expected-cache-sha256`, defaulting to the frozen hash above

The tool performs no training, scorer call, render, provider/GPU dispatch, or live-run access.
It writes one small JSON result atomically. No output may be placed in `/tmp`.

## 4. Required measurements and derivations

The JSON must include explicit `MEASURED`, `DERIVED`, or `INFERRED` labels per field/group.

### 4.1 Frozen target audit

- cache size and SHA-256
- shapes/dtypes and canonical serialization SHA-256 for `lstars` and `gt_poses`
- `lstars` minimum/maximum and exact per-class pixel counts, retaining the canonical order
- fail closed on wrong SHA, shape, count, dtype family, range, or class-order constant

### 4.2 Concrete lossless Seg code

Encode all 600 `lstars` maps with `encode_partition_stack(..., template="temporal")`.
Record real payload bytes, header/model/stream split, bytes/frame, and payload SHA-256.
Perform a bit-exact decode/roundtrip on a deterministic small prefix using a separately encoded
prefix payload. The production codec's existing tests remain the whole-stack decoder contract;
do not claim a full-payload decode was run if it was not.

This row is labeled:

`MEASURED DECLARED-CODE-FAMILY UPPER BOUND; exact partition payload; non-promotable`.

### 4.3 Intended digital-complex model chain

Report, without promotion:

- measured inner contour stream: `457528 B`
- derived post-Brotli `/2` shared-edge estimate: `228764 B`
- inferred/derived 5-class palette/cell-label charge: `15 B` only as the inherited project-model
  assumption, never as a measured emitted section
- measured temporal-ξ payload: `6634 B`
- measured self-contained temporal-ξ section: `7195 B`
- derived optimistic project-model total:
  `228764 + 15 + 7195 = 235974 B`

The result must carry these blockers:

- no complete self-delimiting contour decoder/framing
- division after nonlinear Brotli is not an emitted shared-edge codec
- cached argmax cells are not proven classical Morse–Smale cells
- palette witness has measured nonzero realized Seg distortion
- temporal-ξ section has measured nonzero realized Pose distortion

Therefore `235974 B` is neither lossless `(d_seg,d_pose)=(0,0)` custody, nor an archive, nor a
universal-K lower bound.

### 4.4 Optional exact target-pose diagnostic

For target custody only, losslessly compress canonical little-endian cached `gt_poses` bytes with
a fixed deterministic standard-library LZMA/XZ configuration and verify exact decompression.
Charge an explicit schema header to each independent XZ stream. Label this an independent
declared-family upper bound on cached target coordinates, **not** temporal ξ and not a receiver.
This row must never replace the measured 7195-byte temporal-ξ section in the intended model.

### 4.5 Strict rate ceiling

Derive using exact integer arithmetic where possible:

\[
B < 0.15\,\frac{37{,}545{,}489}{25}=225{,}272.934.
\]

Thus the largest integer byte count strictly below the rate-only `0.15` threshold is `225272`.
Record rate terms at `225272` and `225273`, plus for every reported combined byte count.
For the optimistic `235974 B` model, report the continuous and integer-byte gaps, while stating:
an above-threshold code family does not prove universal K is above threshold.

## 5. JSON verdict contract

The result must expose these fail-closed top-level outcomes:

- `requested_claim_verdict = FALSIFIED_AT_CLAIM_LEVEL`
- `universal_k_numeric_lower_bound = TRIVIAL_ONLY`
- `universal_k_threshold_verdict = INCONCLUSIVE`
- `exact_zero_distortion_receiver_closed = false`
- `score_claim = false`
- `promotion_claim = false`
- `frontier_pointer_delta = NONE`
- `execution_allowed_or_used = false`

It must include the corrected AIT equations as strings and distinguish every source field from a
derived calculation. Do not use the words `measured K`, `K bytes`, or `K lower bound` for any
concrete code length.

## 6. Verification and acceptance

Acceptance requires:

1. `py_compile` and `--help` pass.
2. Existing context-codec unit tests pass.
3. A real cached-n600 invocation produces deterministic JSON and the expected exact hashes/counts.
4. Independent round-1 audit re-derives:
   - inequality orientation;
   - strict integer byte ceiling;
   - canonical class order;
   - the distinction between cached `gt_poses` and temporal ξ;
   - the lack of receiver-closed `(0,0)` custody.
5. Post-edit review-tracker and serializer gates pass with exact content SHA-256 protection.
6. MAIN landing review is mandatory and must adjudicate the premise falsification before any
   equation registration or reuse of the old `MDL <= K` bracket.

## 7. Triality and consumer contract

- **Equation candidate:** correct `K_U(T|D) <= L_C(T)+O(1)` and evaluator data-processing
  relation; keep as candidate/debt pending MAIN review, not a canonical registration in this lane.
- **DAG:** emit a standalone FEED memo connecting frozen n600 target custody to v10/#536 and the
  reverse-waterfill consumer, with receiver-closure blockers explicit.
- **DSL:** not applicable to this read-only measurement; no trainer or launcher configuration is
  created.
- **STORES CONSULTED:** #180/#284/#311/#369 lineages, the necessity solver/calibration artifacts,
  frozen-scorer/intrinsic-coordinate memory, current canonical frontier/lane/task surfaces, and the
  exact input receipts above. Durable citations belong in the final measurement memo.
