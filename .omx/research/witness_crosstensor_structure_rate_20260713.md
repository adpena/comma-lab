# Witness cross-tensor structure rate lever — n600 lossless byte-close

**Date:** 2026-07-13

**Lane:** `lane_witness_crosstensor_structure_rate_20260713`

**Axis:** `[macOS-CPU/numpy-fp32 advisory] NON-PROMOTABLE`

**Execution:** `$0 LOCAL`; no cloud, paid dispatch, training, protected-run mutation, or exact evaluator actuation

**Verdict:** `ADMIT_JOINT_LOSSLESS_STORAGE_CHART`

**score_claim:** `false`

**promotion_eligible:** `false`

## One-line outcome

The exact n600 V9 witness does **not** admit a new post-hoc shared value codebook, but a derived
lossless joint storage chart reduces the exact archive from **63,659 B to 63,242 B (MEASURED −417 B)**
with an identical full decoded quantized-state hash, giving **DERIVED advisory ΔS =
−0.0002776631834519455** and exactly zero component delta relative to the identity int8 byte-close.

## Authority and custody

- **MEASURED checkpoint:**
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz`,
  379,776 B, SHA-256
  `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c`.
- **MEASURED population:** 600 pairs, 1,200 frame-code rows, real run checkpoint (not a fixture).
- **MEASURED receipt:**
  `experiments/results/witness_crosstensor_structure_rate_20260713/measurement_receipt.json`.
- **MEASURED candidate archive:**
  `experiments/results/witness_crosstensor_structure_rate_20260713/joint_lossless_packet/archive.zip`,
  63,242 B, SHA-256
  `59595ef9c18e514c3cb66a5809f7f2644caab1c3fd7594eacd91196255c58cf9`.
- **MEASURED identity archive:** 63,659 B, SHA-256
  `1056a39427133ee3d160f3612455f191d32496f8039ab41188c52896465c8de1`.
- **ASSUMED constants:** none in the codec selection. The candidate axis chart is the exact minimum
  over the finite checkpoint-derived chart set; the pair transform is selected by exact Brotli byte
  count. Brotli quality 11 is the existing byte-close grammar, not a newly fitted constant.

## STEP 0 — clustering gate before codec construction

The current byte-close quantizer contributes 61,175 base-weight symbols and 22,800 per-frame code
symbols. Thus the base weights are **MEASURED 72.84906222089907%** of the fixed counted int8 symbol
mass. The 17 base tensors give:

| Quantity | Result | Evidence class |
|---|---:|---|
| Pooled entropy `H(Q)` | 6.711633424476405 bits/weight | MEASURED |
| Weighted per-tensor entropy `H(Q|T)` | 5.585418071303699 bits/weight | MEASURED |
| Tensor-identity information `I(Q;T)=H(Q)-H(Q|T)` | 1.1262153531727064 bits/weight | DERIVED from measured entropies |
| One pooled base Brotli stream | 44,006 B | MEASURED |
| Sum of 17 separate tensor streams | 44,032 B | MEASURED |
| Existing pooling benefit | 26 B | DERIVED from measured bytes |

**STEP 0 verdict: `NULL_POSTHOC_EXACT_SHARED_VALUE_CODEBOOK`.** The tensors do not share one
layer-independent value distribution: conditioning on tensor identity removes 1.126215 bits/weight.
More importantly, the shipping representation already stores one implicit shared int8 alphabet
(`−128…127`) and one pooled base stream. An additional exact value codebook has no distinct repeated
value payload to remove. No VQ runtime was built.

`verdict_scope: FORMULATION x INSTANCE — an additional post-hoc exact shared value codebook on this
checkpoint after its symmetric-int8 grids are fixed. This is not a negative verdict on training-time
weight tying, low-rank factorization, VQ-in-the-loop, #110 latent-structure regularization, #242, or
other witness checkpoints.`

## Paying lossless levers

### 1. Derived weight-axis storage permutation

Every nontrivial 2-D tensor supplies two legal storage charts: original C-order or transposed storage.
The tool exhaustively measured all **MEASURED 512** combinations at Brotli quality 11 and selected:

`in_proj.weight`, `film.weight`, `hidden.1.weight`, `hidden.2.weight`, `hidden.3.weight`.

The base stream falls from **MEASURED 44,006 B to 43,826 B (−180 B)**. After the counted inverse-chart
metadata and outer ZIP interaction, the isolated exact archive falls from **MEASURED 63,659 B to
63,510 B (−149 B)**.

### 2. Cross-pair frame-separated modulo-256 deltas

The main FiLM table has **MEASURED 1,200/1,200 unique rows and 600/600 unique pair rows**, so exact-row
dedup is null. The paying formulation separates frame-0 and frame-1 sequences, stores each first row,
then stores pair-to-pair deltas modulo 256. The transform is an exact bijection on int8 symbols.

The code stream falls from **MEASURED 18,469 B to 18,201 B (−268 B)**. After metadata and outer ZIP,
the isolated exact archive falls from **MEASURED 63,659 B to 63,408 B (−251 B)**.

### 3. Existing pose-payload cross-pair coder

The checkpoint contains **MEASURED 14,400 B `xi_stored` plus 14,400 B `dxi`** as float32 tables; the
effective 600×6 twist is 14,400 B before pose quantization. Its quantized rows are **MEASURED 600/600
unique**, so exact-row dedup is also null. The already-built canonical pose coder gives:

| ξ coding chart | Payload | Exact same quantized ξ |
|---|---:|---|
| raw int16 | 7,232 B | MEASURED true |
| `delta_ar` | 3,201 B | MEASURED true |
| `delta_res` | 2,714 B | MEASURED true |

**Verdict:** reuse existing `delta_res`; do not create a duplicate cross-pair pose codec. This payload
is not included in the current pose-blind identity/joint archives, so its 4,518 B raw-to-`delta_res`
saving is a measured sidecar property, not added to the 417 B candidate claim.

`verdict_scope: FORMULATION x INSTANCE — exact-row dedup and the existing lossless integer-residual
coders on this 600-row twist table; not a negative on training-induced pose structure.`

## Composition and component preservation

| Byte-close chart | 0.bin | archive.zip | Δ archive vs identity |
|---|---:|---:|---:|
| Identity | 64,376 B | 63,659 B | 0 B |
| Weight permutation only | 64,230 B | 63,510 B | −149 B |
| Pair delta only | 64,132 B | 63,408 B | −251 B |
| Joint lossless | 63,962 B | 63,242 B | **−417 B** |

The joint manifest grows from **MEASURED 1,879 B to 1,913 B (+34 B)** while the two inner streams
save 448 B, yielding **MEASURED −414 B in `0.bin`** and **MEASURED −417 B in exact `archive.zip`**.
Outer ZIP interactions make the isolated savings non-additive; only the measured composed archive is
the rate verdict.

All four decoded n600 int8 states—every base tensor plus all 1,200 code rows—have the same **MEASURED**
SHA-256:

`c67830f51d58291c7e6f92ef6140fc3599e872b7fb1436874577ea66f32e14fb`.

Therefore, for the deterministic receiver:

- `d_seg(joint) − d_seg(identity) = 0` — **DERIVED exactly from full decoded-state equality**.
- `d_pose(joint) − d_pose(identity) = 0` — **DERIVED exactly from full decoded-state equality**.
- Both shipped baseline and joint decoders are **MEASURED BIT-EXACT** against the NumPy-fp32 oracle on
  the receiver smoke (1 pair / 2 frames, maximum uint8 difference 0).

The selected full-precision checkpoint has a prior **MEASURED n600 CPU** trainer verdict
`d_seg=0.03482035319010417`, `d_pose=21.670739`; these are context, **not** re-labelled as the absolute
int8 byte-closed values. The workspace-only sandbox cannot write the required 3.66 GB full raw to the
SSD tier, so the absolute int8 n600 scorer row was not re-inflated. The lossless candidate’s component
delta is nonetheless closed at n600 by exact equality of the entire decoded state. This distinction is
why `score_claim=false` remains mandatory.

Using the canonical rate law only,

`ΔS = 25·(63,242−63,659)/37,545,489 = −0.0002776631834519455`

is **DERIVED** from **MEASURED exact archive bytes**. It is advisory, not a promoted score.

## Triality

- **DSL:** `GaugeComponent.WITNESS_CROSS_TENSOR_CODER` with
  `WitnessCrossTensorCoderGauge.IDENTITY` (emits nothing) and `AUTO_LOSSLESS` (emits the real
  byte-close argv `--cross-tensor-codec auto_lossless`). The chart owns only joint coding of fixed
  symbols; task #336 retains exclusive ownership of per-tensor sensitivity and bit allocation.
- **Canonical equation:** `witness_lossless_cross_tensor_storage_law_v1` registered through the
  fcntl-locked registry. Its law minimizes exact archive bytes over bijective storage charts after
  `q` is fixed, with `P⁻¹Pq=q` and `Δ⁻¹Δq=q`.
- **DAG:** `FEED-witness-xcodec` appended to
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.

## In-training ceiling and routing

Post-hoc coding found a real but small 417 B gain because current layers retain substantial
tensor-specific distribution information. The higher-ceiling direction is to make the weights
clusterable during training: weight tying, low-rank shared bases, VQ-in-the-loop, or a differentiable
joint-rate/latent-structure regularizer. Route this result as a witness-spec input to ideal-config,
#242, and #110. This lane did **not** edit `spec_v9_cgauge.py` and did not claim that post-hoc VQ is
safe or paying.

Reactivation gate: rerun STEP 0 and exact archive selection on every new checkpoint, quantizer, or
archive grammar. Never transfer the selected transpose mask or the null shared-codebook verdict by
assumption.

## Apparatus delta and verification

- New reusable codec: `src/tac/boundary_math/witness_crosstensor_codec.py`.
- Default-off integration: `tools/levelset_byte_close_and_eval.py`.
- Reproducible probe and receipt: `tools/measure_witness_crosstensor_structure.py`.
- A pre-existing byte-close gate defect was found: `bit_exact_roundtrip_gate` referenced undefined
  local `order`. It now consumes `manifest["base_param_order"]`; the two real packet gates passed.
- Focused tests cover exact base/code inverses, modulo-256 wraparound, default preservation, malformed
  streams, DSL-to-real-flag mapping, canonical equation content, and locked registry roundtrip.

## Pointer-delta honesty and stores consulted

The canonical frontier scan currently reports **MEASURED `[contest-CPU]` 0.1880443979880752** while
`reports/latest.md` still cites 0.1910828242 and is scanner-flagged as drifted. This lane changed
neither surface and makes no frontier claim. `[contest-CUDA]` and `[macOS-CPU advisory]` remain
separate; no transfer was inferred.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`PROGRAM.md`; `docs/vehicle_7_os.md`; v7.5/v8 canonical specs; `reports/latest.md` plus
`tac.frontier_scan.build_frontier_scan_payload`; lane, task, subagent, equation, and probe registries;
latest Codex findings/session memo and Claude council/design memos; the protected V9 checkpoint,
`levelset_best.json`, and `run.log`; #78/#83/#311 weight-codec surfaces; PR95 split-Brotli/permutation
surface; #106/ξ residual surfaces; the canonical byte-close tool and exact generated packet bytes.

No commit was attempted: per mission guardrail, these changes remain uncommitted for main-agent `.py`
review and serializer landing.
