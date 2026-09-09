# ddm_ane3 — fixed-output residue remeasurement and ANE registration gate

Tokens: `[no-triality] [p0-ledger-ok]`  
Owner: `ddm_ane3`  
Date: 2026-09-09  
Disposition: `MEASURED-CPU-FALLBACK / ANE-PASS-WITHDRAWN / QUEUED-WITH-A-FIRE-ORDER`  
Authority: `[macOS-CPU/CoreML advisory]`; `score_claim=false`; no frontier movement.

## Result

The conversion defect is fixed in the actual ane2 conversion call: mixed-precision
models now declare `outputs=[ct.TensorType(dtype=np.float32)]` and fail closed unless
the saved Core ML output type is `65568` (`FLOAT32`). The retained smoke package
returns a runtime `float32` output.

The fixed-output numerical result is strong but is **not ANE evidence**. On the
generated decode, all four fixed models are below the `3.3e-05` SegNet flip bar;
`g13stem` measures `1.3131035698784722e-05`. But Core ML could not populate its E5
bundle cache from the managed shell, `MILCompilerForANE` failed, and
`MLComputePlan` measured **0.0% ANE** for every row. The requested
`CPU_AND_NE` execution therefore fell back completely to `MLCPUComputeDevice`.

The charter requires both `g13stem <= 3.3e-05` and at least 75% ANE. The second
predicate is false. I therefore **withdraw the provisional g13stem ANE pass**, do
not register a segmentation backend, and leave `BACKEND_AXIS_VERDICTS` unchanged.
This is an environment-scoped failure to establish the ANE predicate, not evidence
that native g13stem placement is impossible.

PoseNet operations 0:18 were also measured on the CPU-fallback Core ML graph.
No single-op fp32 rescue removes 80% of that surface's dim-0 error; op 3, convolution
`72`, is best at 38.8%. The fixed-output CPU-fallback graph has dim-0 mean absolute
error `0.0497447`, so it does not reproduce the predecessor's pinned ANE interval
`0.150–0.154`. Consequently this is a scoped negative for the CPU-fallback graph;
the named-op question on actual ANE execution remains untested.

## Boundaries

- `upstream/`, the live pointer tree, `submissions/semantic_joint_ctxmix/`, and
  `experiments/ddm_sj1_*` were not edited.
- No Modal work ran. No Metal or MPS result is cited.
- The local scorer lane was not entered. Every comparison consumes the already
  frozen cpu_torch SegNet/PoseNet reference and never regenerates it.
- No contest evaluation ran and no score component changed.
- Converted `.mlpackage` and `.mlmodelc` trees are retained on Vertigo. Per-pair
  flip ledgers, fp16/fp32 activation banks, and all causal pose outputs are retained
  on APDataStore.
- `CPU_AND_NE` is only a request. The attached `MLComputePlan` census, not latency,
  decides placement; here it says CPU fallback.

## Conversion repair and source verification

`experiments/ddm_ane2_engineer_precision_drift.py::_convert_mixed` now does all of
the following:

1. passes an fp32 input tensor type;
2. passes an explicit fp32 output tensor type;
3. verifies the original MIL compute-op sequence before saving;
4. reads the generated spec and refuses any output type other than Core ML
   `FLOAT32` (`65568`);
5. compiles to an explicit retained sibling `.mlmodelc` tree with
   `xcrun coremlcompiler`, avoiding an untracked temporary compilation; and
6. runs predictions from `CompiledMLModel` while preserving the package as the
   provenance source.

`verified-at-source:`

- MIL `65552 = FLOAT16` and `65568 = FLOAT32` are recorded by the saved stage-9
  conversion comparison, `stage9/refconvert_posenet_gt.json`, SHA-256
  `e5fd7b5d9d2d6e9c23631b4331936bcdc16a4c5bf356af1367a4f03f9fd97ac1`.
- The SegNet bar is `SEG_AUTHORITY_FLIP_BAR = 3.3e-5` in
  `src/tac/ane_screening.py`; 75% is the explicit acceptance predicate in the
  ane3 charter, not an empirical constant.
- Pose hd128 is `7.0226e-05` in
  `stage6/targeted_posenet_gt_n120.json`, SHA-256
  `f4c8fd4a62f5b9704eade547a4ca10f279ff75c80f2f65567eab211cc2149274`.
  Against that receipt's exact `d_pose = 7.77e-06`, the multiple is **9.04x**,
  not the charter's rounded 11x. Pose remains refused either way.

The source-verification addendum is also written into the ane2 memo so the corrected
ratio and the four source identities survive independently of this receipt.

## Fixed-output SegNet rows

All rates below use 600 generated-decode frame-1 states and the same retained
cpu_torch fp32 argmax. Denominator: `600 × 384 × 512 = 117,964,800` pixels.
The n120 rate is the exact stratified `0,5,…,595` subset of each retained n600
per-pair ledger, not a second conversion or a prefix.

| backend graph | measured ANE share | n120 flip rate | n600 flip rate | n120 vs n600 | flips / pixels | verdict |
|---|---:|---:|---:|---:|---:|---|
| all fp16, fp32 output | 0.0% | 1.5216404e-05 | 1.4614529e-05 | +4.12% | 1,724 / 117,964,800 | numerical bar passes; ANE gate fails |
| g13, fp32 output | 0.0% | 1.4919705e-05 | 1.4555189e-05 | +2.50% | 1,717 / 117,964,800 | numerical bar passes; ANE gate fails |
| g13stem, fp32 output | 0.0% | 1.3563368e-05 | 1.3131036e-05 | +3.29% | 1,549 / 117,964,800 | numerical bar passes; provisional ANE pass withdrawn |
| tail k=64, fp32 output | 0.0% | 8.0532498e-06 | 7.5615777e-06 | +6.50% | 892 / 117,964,800 | numerical bar passes; ANE gate fails |

Every row's `ops_by_device` contains only `MLCPUComputeDevice`. The fp32 output
repair does explain much of the predecessor's apparent SegNet drift: g13stem moves
from the provisional `2.8856066e-05` to `1.3131036e-05`, a ratio of 0.4551.
That confirms the prediction's upper side but falsifies its proposed lower bound:
`1.3131e-05` is below `0.5 × 2.8856e-05 = 1.4428e-05`.

The per-pair n120/n600 agreement is close enough to support the original sizing
role, but no n120 row is used as a verdict. The n600 rows are the result.

## PoseNet operations 0:18

The instrument exposes and retains both fp32 and fp16 tensors at each boundary op
for 32 stratified pairs, then measures 18 one-op interventions on 120 stratified
pairs. Intervention `i` holds only op `i` fp32 while every other PoseNet compute op
is fp16 and the model output is fp32. The causal fraction is
`1 - rescued_dim0_error / uniform_dim0_error`.

| op | name | type | activation relative L2 | dim-0 error after one-op rescue | pinned error removed |
|---:|---|---|---:|---:|---:|
| 0 | `26` | sub | 1.6342e-04 | 0.0497447 | 0.0% |
| 1 | `_inversed_input.1` | mul | 1.6370e-04 | 0.0505595 | -1.6% |
| 2 | `scale_out.1` | conv | 6.6767e-04 | 0.0516942 | -3.9% |
| 3 | `72` | conv | 2.0762e-03 | **0.0304563** | **38.8%** |
| 4 | `input.7` | add | 1.9321e-03 | 0.0497447 | 0.0% |
| 5 | `input.9` | gelu | 1.9313e-03 | 0.0497447 | 0.0% |
| 6 | `scale_out.3` | conv | 7.6757e-03 | 0.0439219 | 11.7% |
| 7 | `105` | conv | 9.0046e-03 | 0.0566237 | -13.8% |
| 8 | `input.15` | add | 8.1932e-03 | 0.0497447 | 0.0% |
| 9 | `input.17` | gelu | 1.0077e-02 | 0.0497447 | 0.0% |
| 10 | `identity_out.1` | batch_norm | 1.5689e-02 | 0.0544738 | -9.5% |
| 11 | `130` | conv | 1.2486e-02 | 0.0465968 | 6.3% |
| 12 | `input.21` | add | 1.2266e-02 | 0.0497447 | 0.0% |
| 13 | `input.23` | gelu | 2.1598e-02 | 0.0497447 | 0.0% |
| 14 | `identity_out.3` | batch_norm | 2.4483e-02 | 0.0491920 | 1.1% |
| 15 | `scale_out.5` | conv | 2.4273e-02 | 0.0491811 | 1.1% |
| 16 | `out.7` | add | 2.4483e-02 | 0.0497447 | 0.0% |
| 17 | `177` | conv | 1.3380e-02 | 0.0485914 | 2.3% |

Uniform fp16 fixed-output self-MSE is `3.1246136e-04`; uniform fp32 is
`6.0860530e-12`. The largest activation relative-L2 error is at op 16, but holding
that residual add alone fp32 removes 0% of the final dim-0 error. Activation
magnitude alone therefore does not identify a causal source.

No op reaches the 80% bar, and the best op does not reach the charter's 50%
distributed-error falsifier. This closes only the following formulation instance:
single-op fp32 rescues in operations 0:18 on the fixed-output **CPU-fallback**
Core ML graph, stratified n120. It does not localize or close the actual ANE error.

The debug packages are behavior-preserving on the same 32 rows: maximum absolute
final-output delta against the uninstrumented base package is exactly `0.0` for
both fp32 and fp16. The final report is
`/Volumes/APDataStore/pact/ddm_ane3_residue/item2/pose_activation_diff.json`,
SHA-256 `6df378abcd50b4d94d4f9044fd72b4b8fb36525d4b56da72c99e6ae66b922b74`.

## Screening registration

Registration is not performed. `tac.ane_screening` continues to refuse the existing
`ane_fp16_screen` backend for `seg_argmax`, `pose_rank`, and `pose_value`. No
`g13stem` backend name, loader, or positive `BACKEND_AXIS_VERDICTS` row is added.
Doing so with a 0% ANE receipt would be a fake implementation of the charter's
screening axis.

The prior pose closure also remains unchanged: hd128 is still 9.04x exact d_pose
and therefore cannot read pose values. The present CPU-fallback activation sweep
does not reopen it.

## Canonical equation

The fixed-output finding is attached only through
`tac.canonical_equations.registry.update_equation_with_empirical_anchor` to
`scorer_fp16_drift_by_axis_v1` with anchor
`ane3_segnet_fixed_fp32_output_cpu_fallback_n600_20260909`. Its backend and
measurement method explicitly say `coreml_mixed_fp16_cpu_fallback`; its empirical
output records 0.0 ANE and `registration_gate_passed=false`. It is not an ANE
anchor by implication.

Catalog #344 is run against this memo before landing.

## RECALL EVIDENCE

I searched the full local corpus, not only the charter seeds:

- query `FP16ComputePrecision|coreml_fp16_ane|ane_fp16_screen|g13stem|scorer_fp16_drift_by_axis_v1|Neural Engine`
  across `.omx/research/`, `.omx/state/`, `docs/`, `src/`, `experiments/`, and
  `tools/`;
- query `ddm_ane[123]|ane2|g13stem|precision_drift` in the canonical task ledger;
- the canonical equation registry through `tools/list_canonical_equations.py --json`;
- the canonical research index and `sub015_DAG_*` ANE/Core ML FEED blocks; and
- implementation specifications and receipts around T4 per-op precision and T10
  per-example gates.

Beyond the charter's named ane2 memo, the search recovered:

1. `ddm_ane1_ane_screening_lane_20260905.md`: a screening result may only rank or
   prescreen, and every adopted pair must be remeasured by cpu_torch. This keeps
   any future g13stem registration segmentation-only and prevents pose leakage.
2. `ane_ecosystem_survey_20260713.md`: T4 establishes per-op precision selectors,
   T10 requires per-example no-regression evidence, and a compute-unit request is
   not placement proof. This changed the run from a mean-only check to retained
   per-pair ledgers plus `MLComputePlan` gating.
3. `ane_unlock_followup_implementation_spec_20260713.md`: the selected MIL ops and
   exact n600 output must be recorded in one graph. This ruled out substituting a
   hand-split or proxy graph.
4. canonical task rows 732–734 already carry the ane2 residue obligations. They
   prevent treating the environment-scoped shortfall as disappearance of the
   work.

No additional fixed-output g13/g13stem n600 remeasurement was found in that scope.

## Payload custody and reproducibility

| artifact | durable location | custody |
|---|---|---|
| four n600 reports + per-pair ledgers | `/Volumes/APDataStore/pact/ddm_ane3_residue/item1/` | one `.npy` ledger per graph; report records bytes-by-denominator, package tree hash, and ledger SHA-256 |
| fixed SegNet packages/compiled trees | `/Volumes/VertigoDataTier/pact/ddm_ane3_residue/models/item1/` | one stage directory per graph; retained, never scalar-only |
| n32 fp32 activation bank | `/Volumes/APDataStore/pact/ddm_ane3_residue/item2/activation_fp32/` | 32 stratified per-pair compressed NPZ files, each individually hashed in the report |
| n32 fp16 activation bank | `/Volumes/APDataStore/pact/ddm_ane3_residue/item2/activation_fp16/` | 32 matched compressed NPZ files, each individually hashed in the report |
| n120 causal pose outputs | `/Volumes/APDataStore/pact/ddm_ane3_residue/item2/causal/` | uniform fp32/fp16 plus 18 rescue arrays, each retained and hashed |
| PoseNet base/debug/rescue packages | `/Volumes/VertigoDataTier/pact/ddm_ane3_residue/models/item2/` | fixed base, two 0:18 debug graphs, and 18 one-op rescue graphs |
| final activation report | `/Volumes/APDataStore/pact/ddm_ane3_residue/item2/pose_activation_diff.json` | summary plus per-payload manifest and package hashes |

Storage preflight before both detached launches measured at least 17 GiB free on
Vertigo and at least 48 GiB on APDataStore. ITEM 1 ran through
`tools/launch_detached_process.py` and completed with rc 0 in
`.omx/tmp/codex_runs/ddm_ane3_item1.done.done`. ITEM 2's first complete sweep has
rc 0 in `.omx/tmp/codex_runs/ddm_ane3_item2.done.done`; the source-consistent
parity replay has rc 0 in `.omx/tmp/codex_runs/ddm_ane3_item2_parity.done`, and
the final current-source replay has rc 0 in
`.omx/tmp/codex_runs/ddm_ane3_item2_final.done`.

## Verification

- `git diff --check`: pass.
- `bash -n experiments/ddm_ane3_run_item1.sh`: pass.
- Ruff over all three changed Python files: pass.
- Python byte compilation over all three changed Python files: pass.
- `pytest -q src/tac/tests/test_ane_precision.py`: **44 passed**.
- Direct retained-package census: all four SegNet packages and the PoseNet all-fp16
  package declare Core ML output dtype `65568`.
- Retained-artifact audit: all four item-1 ledgers reproduce their report hashes
  and denominators; all 64 activation NPZs and all 20 causal output arrays exist,
  and the hashes carried by the item-2 report match.
- Latency smoke on the retained g13stem compiled tree: pass after correcting the
  metadata path to read I/O names from the source package rather than calling the
  unsupported `CompiledMLModel.get_spec()`.
- `experiments/ddm_ane2_engineer_precision_drift.py`,
  `experiments/ddm_ane3_pose_activation_diff.py`, and
  `tools/ddm_ane3_register_equation_anchor.py` each have two genuine
  `tools/review_tracker.py` review passes after their final functional edit.
- Catalog #344 strict canonical-equation memo check: pass.

## ITEM 4 — native ANE closure for the fixed-output graph

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: `MAIN / native-ANE runtime owner`.
Consumer store: `/Volumes/APDataStore/pact/ddm_ane3_residue/native_ane/` plus
matching model trees under `/Volumes/VertigoDataTier/pact/ddm_ane3_residue/native_ane_models/`.
Fire trigger: a native shell can write the real E5 bundle cache, and a one-pair
preflight for fixed-output g13stem completes without `MILCompilerForANE` or
`IOSurfaceSharedEvent` failure and `MLComputePlan` reports at least 75% ANE. Then
run the same retained n600 g13stem/all-fp16/tail-k64/g13 graph set and the PoseNet
0:18 causal sweep. Register segmentation-only g13stem only if the exact fixed
g13stem row remains at or below `3.3e-05` and at or above 75% measured ANE; pose
stays refused. Fold the task without registration if either predicate fails.

## LIVE-HYPOTHESES

- Native g13stem remains worth one controlled rerun because the fixed output
  removed 54.5% of its provisional flips while its predecessor graph had already
  measured 75.5% ANE. Only the current cache boundary, not the numerical bar,
  prevented registration.
- PoseNet's ANE dim-0 error may be distributed across several stem convolutions
  rather than born at one op: CPU fallback's best single rescue is only 38.8%,
  while the predecessor's fp32 head-of-8 intervention removed about 60%.

## DEAD-ENDS

- Treating `CPU_AND_NE` as placement proof is closed: `MLComputePlan` measured
  every fixed row at 0% ANE after E5/ANEF failure.
- Registering g13stem from the fixed CPU-fallback rate is closed: it would satisfy
  only one of the charter's two predicates.
- Naming the largest activation-difference op as the cause is closed on this
  instance: op 16 has the largest relative-L2 error but its fp32 rescue removes
  0% of dim-0 error.
- Reusing the charter's `11x d_pose` arithmetic is closed: the source receipt's
  exact denominator makes hd128 9.04x.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
