# NA4 Receipt - Rate-Axis Prefix Bias

## Answer First

Measured, scorer-free, no launches: QO1/TD1 token payload pricing across
`prefix_n32`, `prefix_n120`, `stratified_n32_seed20260805`,
`stratified_n120_seed20260805`, and full n600. All coder rows round-tripped with
decode equality.

The rate-axis prefix-bias law is **not a single optimistic scalar**. On the pure
matched-coder raw token lattice, prefix/full was small and mostly high priced:
`prefix_n32` was `1.027275x` full under Brotli q11 and `1.013574x` under LZMA1
raw; `prefix_n120` was `1.007642x` and `0.993569x`. On the existing R7 token
frame, `prefix_n32` was optimistic only for Brotli (`0.989000x` full) and was
slightly high for LZMA (`1.007775x`); `prefix_n120` was high for both R7 coders
(`1.025535x`, `1.029800x`).

So the m38 prediction is **falsified as a general law**: prefix rate pricing is
not reliably 3-15% cheaper than population. The observed range on this real token
stream is `0.989000x` to `1.029800x` full-population bytes per pair for prefix
rows, with the larger selector gap showing up between R7 `prefix_n32` and
R7 `stratified_n32` (`0.914862x` Brotli, `0.921397x` LZMA). Effect size stayed
below the 20% falsifier threshold.

OD9 direct rate-axis bias was **not reachable scorer-free** from current custody:
the verified SSD manifest contains only the OD2 `stratified_n32` pair payloads,
not prefix or full-population OD9 payloads. OD9's `1,214,007 B` projection was
therefore not directly rebiased. As an analogy only, applying token-stream n32
selector effects says the stratified OD9 choice was bias-safer than a prefix
choice: a prefix-like n32 selector would have changed the Stage-1 projection by
`-4,642 B` to `-103,358 B` depending on token surface/coder, while the stratified
n32 row itself priced `1.030292x` to `1.093747x` the token-stream full bpp.

Axis: `[macOS-CPU scorer-free coder pricing]`. Paid launches: `0`. Scorer
forwards: `0`. `upstream/evaluate.py`: not run. No exact score claimed.

## Pricing Tables

Denominator for every projection: measured coded bytes divided by measured pairs,
linearly scaled to 600 pairs for byte-routing only. Full-population rows are the
actual 600-pair coded bytes. Decode equality is `true` for every row.

### QO1 Token Raw Lattice

Raw uint8 token lattice, shape `600 x 24 x 32 x 4`, compressed directly. This is
the pure standing-pair coder control, not a receiver frame.

| coder | selection | n/600 | bytes | B/pair | ratio to full B/pair | projected n600 B | decode |
|---|---:|---:|---:|---:|---:|---:|---|
| brotli-q11 | prefix_n32 | 32/600 | 24,749 | 773.406 | 1.027275 | 464,043.8 | true |
| brotli-q11 | prefix_n120 | 120/600 | 91,035 | 758.625 | 1.007642 | 455,175.0 | true |
| brotli-q11 | stratified_n32_seed20260805 | 32/600 | 24,844 | 776.375 | 1.031218 | 465,825.0 | true |
| brotli-q11 | stratified_n120_seed20260805 | 120/600 | 91,013 | 758.442 | 1.007398 | 455,065.0 | true |
| brotli-q11 | full_population | 600/600 | 451,723 | 752.872 | 1.000000 | 451,723.0 | true |
| lzma1-raw | prefix_n32 | 32/600 | 25,039 | 782.469 | 1.013574 | 469,481.2 | true |
| lzma1-raw | prefix_n120 | 120/600 | 92,043 | 767.025 | 0.993569 | 460,215.0 | true |
| lzma1-raw | stratified_n32_seed20260805 | 32/600 | 25,452 | 795.375 | 1.030292 | 477,225.0 | true |
| lzma1-raw | stratified_n120_seed20260805 | 120/600 | 92,476 | 770.633 | 0.998243 | 462,380.0 | true |
| lzma1-raw | full_population | 600/600 | 463,194 | 771.990 | 1.000000 | 463,194.0 | true |

Selector-only effects on the same stream and coder:

| coder | selector comparison | prefix / stratified B/pair | prefix projected minus stratified projected |
|---|---|---:|---:|
| brotli-q11 | n32 | 0.996176 | -1,781.2 B |
| brotli-q11 | n120 | 1.000242 | +110.0 B |
| lzma1-raw | n32 | 0.983773 | -7,743.8 B |
| lzma1-raw | n120 | 0.995318 | -2,165.0 B |

### QO1 R7 Token Frame

Existing self-describing R7 token frame over the same token lattice. This is a
production-token-frame control; its `brotli11` and `lzma1` modes include the R7
base/delta frame structure.

| coder | selection | n/600 | bytes | B/pair | ratio to full B/pair | projected n600 B | decode |
|---|---:|---:|---:|---:|---:|---:|---|
| r7-brotli11 | prefix_n32 | 32/600 | 20,911 | 653.469 | 0.989000 | 392,081.2 | true |
| r7-brotli11 | prefix_n120 | 120/600 | 81,313 | 677.608 | 1.025535 | 406,565.0 | true |
| r7-brotli11 | stratified_n32_seed20260805 | 32/600 | 22,857 | 714.281 | 1.081038 | 428,568.8 | true |
| r7-brotli11 | stratified_n120_seed20260805 | 120/600 | 80,676 | 672.300 | 1.017501 | 403,380.0 | true |
| r7-brotli11 | full_population | 600/600 | 396,442 | 660.737 | 1.000000 | 396,442.0 | true |
| r7-lzma1 | prefix_n32 | 32/600 | 21,393 | 668.531 | 1.007775 | 401,118.8 | true |
| r7-lzma1 | prefix_n120 | 120/600 | 81,977 | 683.142 | 1.029800 | 409,885.0 | true |
| r7-lzma1 | stratified_n32_seed20260805 | 32/600 | 23,218 | 725.562 | 1.093747 | 435,337.5 | true |
| r7-lzma1 | stratified_n120_seed20260805 | 120/600 | 81,313 | 677.608 | 1.021459 | 406,565.0 | true |
| r7-lzma1 | full_population | 600/600 | 398,024 | 663.373 | 1.000000 | 398,024.0 | true |

Selector-only effects on the same stream and coder:

| coder | selector comparison | prefix / stratified B/pair | prefix projected minus stratified projected |
|---|---|---:|---:|
| r7-brotli11 | n32 | 0.914862 | -36,487.5 B |
| r7-brotli11 | n120 | 1.007896 | +3,185.0 B |
| r7-lzma1 | n32 | 0.921397 | -34,218.8 B |
| r7-lzma1 | n120 | 1.008166 | +3,320.0 B |

## OD9 Projection Safety

OD9/PA2 direct custody facts:

| item | measured scope | bytes | projection | status |
|---|---|---:|---:|---|
| OD9 Stage-1 `stage1_only_absolute_u8` | OD2 stratified n32 | 64,747 | 1,214,007 B | recalled |
| OD9 combined Stage-1 plus k4 carriage | OD2 stratified n32 | 66,785 | 1,252,219 B | recalled |
| PA2 shared-context persisted stream, Brotli q11 | OD2 stratified n32 | 66,497 | 1,246,819 B | recalled |
| PA2 shared-context persisted stream, LZMA1 raw | OD2 stratified n32 | 67,946 | 1,273,988 B | recalled |
| OD9 SSD manifest | 42 declared entries | 981,167 payload B checked | n/a | all SHA-256 and bytes matched |

Direct OD9 prefix/full rate bias: **not measured**. Bounded absence: in
`.omx/research/ddm_od9_20260805/od9_ssd_payload_manifest.json` and its declared
SSD payload paths, I found only the OD2 stratified n32 pair payloads:
`[8, 32, 46, 57, 70, 107, 112, 119, 148, 154, 168, 198, 225, 234, 244, 251, 284,
328, 336, 349, 383, 399, 411, 423, 445, 465, 481, 516, 536, 561, 582, 583]`.
No prefix-n32, prefix-n120, stratified-n120, or full-population OD9 payload stream
was present in that scope.

Analog-only safety calculation, applying QO1 token-stream selector ratios to
OD9's Stage-1 `1,214,007 B` projection:

| token surface/coder | stratified n32 / full Bpp | prefix / stratified Bpp | analog corrected full B | analog prefix-like projection | prefix-like delta |
|---|---:|---:|---:|---:|---:|
| raw lattice / brotli-q11 | 1.031218 | 0.996176 | 1,177,255 B | 1,209,365 B | -4,642 B |
| raw lattice / lzma1-raw | 1.030292 | 0.983773 | 1,178,314 B | 1,194,308 B | -19,699 B |
| R7 frame / r7-brotli11 | 1.081038 | 0.914862 | 1,123,002 B | 1,110,649 B | -103,358 B |
| R7 frame / r7-lzma1 | 1.093747 | 0.921397 | 1,109,952 B | 1,118,583 B | -95,424 B |

Interpretation: OD9 used the stratified n32 set, so it did not inherit a prefix
selection by construction. Under the token-stream analogy, the stratified n32
projection is high vs full by `3.0%` to `9.4%`, not optimistically low. The
prefix-like counterfactual is cheaper than stratified in all four token controls,
especially in R7 n32. This is a routing caveat, not an OD9 direct measurement.

## Recall Evidence

Required seed recall:

| source | recalled fact | plan effect |
|---|---|---|
| `.omx/research/ddm_na2_negative_audit_20260803.md` | Prefix subsets are not population-neutral; sign can invert by axis. | Did not use prefix rows as population substitutes. |
| `.omx/research/ddm_na3_20260805/ddm_na3_receipt.md` | Pose prefix ratios are `2.535x` to `4.207x` harder; NA3 retest used `stratified_blocks`, seed `20260805`, n120/600. | Reused NA3 n120 indices and denominator style. |
| `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json` | Exact n120 seed-20260805 stratified ids. | Used as `stratified_n120_seed20260805`. |
| `.omx/research/ddm_od2_20260805/PAIR_SELECTION.json` | Exact n32 seed-20260805 stratified ids; seg ratio `1.009989x`, pose ratio `0.426287x`. | Used as `stratified_n32_seed20260805`; `src/tac/subset_selection.py` regenerated the same ids. |

Original recall beyond charter seeds:

| query/scope | found beyond seeds | changed plan |
|---|---|---|
| `rg -n "shared_context|stage1_only_absolute_u8|64747|1214007" .omx/research/ddm_pa2_20260805 .omx/research/ddm_od9_20260805` | PA2 already verified the OD9 SSD manifest and measured shared-context vs independent on OD2 n32; OD9 Stage-1 projection source was exact. | Verified the manifest again and used PA2 only as recalled n32 stratified context, not population proof. |
| `rg -n "cx1_tokens|token_lattice|R7|encode_token_codes" .omx/research experiments` | TD1 had proved the live QO1 token lattice source; `experiments/ddm_r7_token_coder.py` exposes strict lossless R7 frame coders. | Added the R7 frame control in addition to the pure raw-lattice Brotli/LZMA standing-pair pricing. |
| `rg -n "pair_payload|manifest|stage2_qcoeffs|xi_dxi" .omx/research/ddm_od9_20260805 .omx/research/ddm_pa2_20260805` | OD9 manifest contains only OD2 n32 pair payload NPZ files, and PA2 found no `xi`/`dxi` named sections. | Classified direct OD9 prefix/full bias as bounded absence instead of manufacturing slices. |
| `rg -n "PE3|per-edge|frame_records|selected" .omx/research experiments` | PE3/PK1 context is selected component grammar, not a clean all-pairs per-pair stream for this charter. | Did not include PE3 as a NA4 measured rate-bias stream. |

Memory recall used for execution discipline: prior Pact notes required real coders
only, explicit denominators, selection mode, and projection labels. That changed
the receipt shape: tables first, selection denominators on every row, and OD9
analogies labeled as analogies only.

## Consumer Routing

| consumer | disposition |
|---|---|
| ss1 scope fields | Rate rows need `axis`, `stream`, `coder`, `selection_mode`, `n/population`, `decode_equality`, `ratio_to_full_population_bpp` when full is available, and `projection_scope`. |
| OD9 | Keep as OD2 `stratified_n32` byte-only projection. It is not prefix-biased by construction, but direct OD9 population bias is still unmeasured. |
| PA2 | Shared-context win remains OD2 `stratified_n32` only. The `5.64%` Brotli win supports shared parametrization but does not close n600 population rate. |
| PK1 | Cite OD9/PA2 as stratified-n32 rate evidence only; do not upgrade it to prefix/full-population closure. |
| m88/m96 | Axis triple now has measured rate rows: pose prefix hard, seg prefix easier from the recalled law, token rate prefix small and stream/coder dependent. |

## JSON

Full machine-readable artifact:
`.omx/research/ddm_na4_20260805/na4_rate_axis_bias_measurement.json`

Artifact SHA-256:
`ac595554f8297153dd1f07b513f5fe6704d10391d2709975e2a3b9df4131f5d9`

```json
{
  "schema": "ddm_na4_rate_axis_bias_measurement_v1",
  "axis": "[macOS-CPU scorer-free coder pricing]",
  "paid_launches": 0,
  "scorer_forwards_run": false,
  "evaluate_py_run": false,
  "measured_streams": [
    "qo1_token_raw_lattice",
    "qo1_r7_token_frame"
  ],
  "od9_direct_rate_axis_bias_measured": false,
  "od9_direct_reason": "Only OD2 stratified_n32 pair payloads are present in verified OD9 SSD custody.",
  "final_frontier_line_required": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved."
}
```

## NEXT_IF_RESUMED

1. If OD9-specific rate-axis bias is needed, materialize the same persisted native
   stream for prefix_n32, prefix_n120, stratified_n120, and full n600 under a
   charter that owns any required scorer/solver work; then rerun the same exact
   coders and SSD manifest verification.
2. Apply this NA4 table format to any already-custodied per-pair-decomposable
   n600 stream. Reject streams whose slice changes representation semantics or
   receiver coverage.
3. Do not use prefix_n32 as a cheap proxy for population rate unless the exact
   stream/coder table includes its ratio to full population.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
