# DDM SN1 SegNet telemetry and separatrix asymmetry — implementation spec

Date: 2026-07-23  
Lane: `ddm_sn1_segnet_telemetry_asymmetry`  
Authority SHA-256: `eef197f9221c4ec94e56840aac93d110d6250478b4bdf2233b82c06e4c323ecd`  
Lineage: v10/DDM only  
Execution: local CPU, `$0`, resumable analysis; no remote/GPU launch  
Authority: `[macOS-CPU frozen-SegNet+PoseNet advisory]`,
`score_claim=false`, pointer write forbidden  
Landing: isolated Codex branch; MAIN landing review required

## Outcome

Land one score-neutral instrumentation and analysis path that exposes the
frozen SegNet's stem, every EfficientNet block, pre/post squeeze-excite,
decoder skips/blocks, and final logits; streams the official n600 last frames
without retaining raw activation tensors; exports a typed, SHA-pinned sided
separatrix contract that the DDM e1 renderer and e2 price consumer can parse;
and joins that contract to an exhaustive three-way v19c residual tensor.

The amended scope also measures the frozen SegNet and PoseNet relay products
over layer × channel × pooled-space × time, derives frozen-weight amplitude
and spectral factors, checks the current receiver against the historical #149
survival wall, and attributes realization loss to observable paint/geometry
axes. These products are associations and directional secants, not a hidden
causal-state decomposition or full Jacobian spectrum.

The analysis must distinguish pixel recurrence from record-level constancy.
No static-record claim is permitted merely because the same image coordinate
recurs.

## Files and ownership

- `src/tac/analysis/segnet_internal_telemetry.py`
  - optional read-only Torch hook layer over the immutable upstream SegNet;
  - analysis factory defaults telemetry ON;
  - training factory defaults telemetry OFF and requires an explicit cadence
    reason;
  - online summaries only, with hook cleanup and argmax identity assertion.
- `src/tac/optimization/ddm_sn1_sided_tolerance.py`
  - strict SDWL1 additive sided-tolerance/D2 schema;
  - ordered-pair inner/outer tolerances;
  - asymmetric e2 price computation and e1 export contract.
- `tools/run_ddm_sn1_segnet_telemetry_asymmetry.py`
  - typed config, source/model/cache SHA checks, storage preflight;
  - resumable per-batch checkpoints and atomic final receipts;
  - n600 official-video analysis and bounded three-segment inverse demo.
- `src/tac/analysis/ddm_sn1_error_source_tensor.py`
  - exclusive three-way source assignment, sided-band, cluster-geometry,
    recurrence, paint-mechanism, and solve-menu vocabularies.
- `src/tac/analysis/segnet_amplitude_telemetry.py`
  - bounded online amplitude, target-boundary, receiver-difference, and
    frequency summaries.
- `src/tac/analysis/scorer_native_diff.py`
  - shared SegNet/PoseNet relay selection, frozen-weight analytic inventory,
    deterministic stable-rank estimator, transport, and resumable product
    aggregation.
- `tools/build_ddm_sn1_error_source_tensor.py`
  - exact v19c/DV1/G2/G3/G4/v14/e1 joins, source-budget and solve-menu
    finalization, scorer-native telemetry, and lossless SSD checkpoint
    externalization.
- Focused tests under `src/tac/**/tests/` and `tests/`.
- Dated JSONL/receipt/findings plus DSL/DAG/equations triality artifacts under
  `.omx/research/`.

No file under `upstream/` may change. No training loop is modified.

## Measurement definitions

For ordered classes \(c\ne c'\), with frozen logits \(z\) and flattened
rank-4 head normal \(\Delta w_{cc'}=w_c-w_{c'}\):

\[
  m_{c\to c'}(p)=z_c(p)-z_{c'}(p),\qquad
  D_2(c\to c',p)=\frac{|m_{c\to c'}(p)|}{\|\Delta w_{cc'}\|_2}.
\]

An oriented boundary sample belongs to side \(c\to c'\) only when the center
pixel's winner is \(c\) and at least one 4-neighbor's winner is \(c'\).
The reverse side is measured independently; symmetry is never inferred from
the shared normal norm. Aggregate rows retain pixel count, margin/flip-distance
tails, temporal window, class stratum, source/model identities, and
`verdict_scope`.

The e2 reduced-cost term is sided:

\[
  \rho_{c\to c'} =
  \lambda^{\rm seg}_{c\to c'} E^{\rm out}_{c\to c'}
  +\lambda^{\rm pose}_{c\to c'} E^{\rm pose}_{c\to c'}
  +\lambda_B\Delta B,
\]

so the inner and outer tolerance fields are separately priced. e1 consumes
the same typed record as two renderer-realizable signed normal bounds.

## Telemetry summaries

- final logit margin field and temporal margin evolution;
- per-class logit energy;
- per-layer boundary-band versus cell-interior activation energy;
- stem, every encoder block, every pre/post-SE pair, decoder skip/block, and
  final-logit tap coverage;
- gradient ERF radial response at explicitly recorded boundary probe points;
- ordered-pair suppression matrix, including Road-to-Lane and Lane-to-Road as
  distinct rows;
- full/first-tail/last-tail n600 strata with first-rung next measurements on
  positive rows.

Telemetry OFF and ON must execute the same frozen forward on the same input,
and the tool must fail unless logits' argmax tensors are exactly equal.

## Scorer-native product and frozen-weight factors

For each declared relay, retain exact per-channel moments, an 8×8 native
feature-energy grid, painted-versus-GT contrasts, across-frame and across-pair
trajectories, four spatial-frequency bands, temporal spectrum, directional
secants, and stable-rank summaries. Hooks must execute in stored topological
order. The SegNet product runs from `encoder.model.conv_stem` through
`segmentation_head`; the PoseNet product runs from `vision.stem` through
`hydra.final_layer.pose`.

Frozen-weight analysis records convolution spectral gain by output/input
channel, BatchNorm affine gain, squeeze-excite gates, GELU derivatives, and
PoseNet's actual amplitude inventory: 24 `LayerScale2d`, 8 `BatchNorm1d`,
19 `GELUTanh`, and one `SEModule`. No fictitious per-block PoseNet SE factor is
permitted.

The exact camera-to-scorer resize contract is phase-indexed and retained with
the analytic artifacts. A single global scalar for bicubic-up → uint8 →
bilinear-down is deliberately refused: the chain is polyphase,
border-dependent, and piecewise affine at uint8. Frequency conclusions are
therefore layer/local-operator facts, not a fabricated global transfer curve.

## Exhaustive error-source tensor

Every one of the 2,265,811 measured v19c residual errors is assigned exactly
once:

- `DESCRIBED_BUT_REALIZATION_LOST`: 892,710 (39.3991%);
- `NEVER_DESCRIBED`: 738,090 (32.5751%);
- `STRUCTURALLY_HARD_IRREDUCIBLE`: 635,011 (28.0258%), scoped only to the
  current program plus the tested DV1 extension.

Within the first bucket, observable mechanism attribution is:
587,913 `COARSE_DESCRIPTION`, 208,623 `PAINT_FUNCTION`, and 96,174
`TEXTURE_PRIOR_REGION_ERF`. This is an axis association, not latent-cause
proof. The solve-first output contains 2,649 exhaustive cluster rows.

The current v19c n600 two-sided target-boundary error fraction is
1,613,214 / 4,684,236 = 0.3443921271. The historical mp128 three-frame
reference is 0.1605960279, so the measured contextual ratio is 2.1444622981.
The scopes and axes differ; they are not pooled into a score claim.

## Inverse demonstration

Select at least three low-margin, connected oriented boundary segments from
real official-video frames. Use the exact camera-to-scorer resize path and the
frozen CPU-Torch SegNet to propose a bounded uint8 camera perturbation, then
re-run the frozen scorer. Each receipt records prediction, realized winner/
rival changes, collateral flips, changed camera bytes, and whether the desired
sided transition realized. A failed segment is a scoped negative and remains
in the receipt; it is never silently replaced.

## Acceptance

1. Focused unit tests cover policy defaults, tap completeness, cleanup,
   argmax identity, ordered-pair directionality, strict schema parse-back,
   asymmetric e2 prices, and malformed-custody refusal.
2. The official-video receipt binds the 37,545,489-byte video SHA-256
   `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`,
   frozen SegNet SHA-256
   `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`,
   upstream modules SHA-256
   `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`,
   exact argv, seed, thread count, sample count, and every output SHA.
3. n600 results are `[macOS-CPU frozen-SegNet+PoseNet advisory]`,
   `score_claim=false`, `promotion_eligible=false`; no pointer mutation.
4. Every positive finding includes its first rung. Every negative includes
   `verdict_scope`.
5. `py_compile`, focused pytest, Ruff, `git diff --check`, review-tracker
   passes, serializer commit, and post-edit SHA receipt all pass.
6. The accepted error tensor partitions exactly 2,265,811 errors, the source
   counts close exactly, all serialized floating-point values are finite, and
   a finalize-only replay from the certified SSD checkpoint tree reproduces
   receipt SHA-256
   `ecf9f015fa6999b9bb7602c93027da713bb278389b92d5d1bf0b95f4ced19faa`.

## Explicit non-goals

- no scorer, model, or video bytes in an archive;
- no upstream edit or inflate-time scorer use;
- no score or promotion claim;
- no HNeRV/PR-family vehicle;
- no contest-CPU/CUDA inference from the local advisory axis;
- no claim that aggregate SDWL1 facts identify an RGB receiver.
