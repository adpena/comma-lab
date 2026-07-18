# Constructive constrained-MDL inverse-solve — local PoC handoff

**Status:** implementation landed; real n600-stratified execution and all numerical result rows are
**PENDING PRIMARY MEASUREMENT**.  This document does not claim a score, promotion, rank/kill verdict,
closed-form global optimum, or pointer movement.

## Scope and truth contract

- `research_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `pointer_moved=false`
- `optimality_claim=none_local_iterative_constrained_solve`
- Authority: `[macOS-CPU advisory]`, non-promotable.
- Negative `verdict_scope`: **INSTANCE** — the selected real v9c2 EMA pairs, scorer-grid-delta
  formulation, frozen CPU SegNet, and declared solver config only. A failure does not kill the
  formulation family or inverse-solve paradigm.

The implementation is `tools/constructive_inverse_solve_harness.py`. It is a $0 local instrument:
no training, paid dispatch, external actuation, sacred-run mutation, or frontier-pointer write occurs.
The base checkpoint, target cache, and scorer are read-only and content-hashed.

## Constructive solve

### Closed-form eliminations (DERIVED)

For one channel, the evaluator resize is the separable linear map

\[
  A(X)=R_h X R_w^\top,
  \qquad (874,1164)\rightarrow(384,512).
\]

The harness reuses `ResizeProjector.build()` for the exact `R_h,R_w`. Every contest row has two
disjoint taps, so each row Gram is diagonal and the minimum-norm right inverse is

\[
  B(Y)=L_h Y L_w^\top,\qquad
  L_h=R_h^\top(R_hR_h^\top)^{-1},\quad
  L_w=R_w^\top(R_wR_w^\top)^{-1}.
\]

`B` is implemented as differentiable sparse scatter, never as a dense camera matrix. The optimized
variable is only `delta ∈ R^(N×3×384×512)`. Its float camera correction is in `range(A^T)`; no
`ker(A)` camera degree of freedom is described or optimized. `range(A)` is the full scorer grid.

Geometry derived from the canonical matrices (not a score measurement):

| quantity | DERIVED value |
|---|---:|
| rank per channel | 196,608 |
| full camera-kernel dimension per channel | 820,728 |
| full kernel fraction | 80.6742315% |
| height-axis blind coordinates | 106 / 874 |
| width-axis blind coordinates | 140 / 1164 |
| axis-blind camera pixels | 230,904 / 1,017,336 = 22.6969261% |
| row-Gram off-diagonal, height/width | 0 / 0 |
| taps per output row, height/width | 2 / 2 |

The 22.6969% figure is the explicit subset of camera pixels unseen because at least one axis is
unused. The 80.6742% figure is the full algebraic kernel dimension. They are different objects and
must not be conflated.

### Iterative convolutional residual (IMPLEMENTED; measurement pending)

The fixed banked v9c2 EMA is rendered by `tools.levelset_byte_close_and_eval`'s canonical NumPy
oracle. The code table is reindexed to the arbitrary selected pairs while original EMA parameters
and original code values remain the render inputs; the temporary canonical blob is used only to
obtain the complete receiver manifest. Persisted self-orient custody must be exactly
`freq_across=32`, `freq_along=8`; the stale generic `freq_along=4` is refused.

The receiver path is:

1. quantize the scorer-grid delta to signed int8;
2. lift through minimum-norm `B`;
3. add to the canonical camera frame, clamp, and hard-round uint8;
4. call the real frozen CPU Torch SegNet `preprocess_input`, whose final-frame bilinear resize is `A`;
5. evaluate all exact target-versus-competitor inequalities.

For target label `L*` and every competitor `k ≠ L*`, the exact winner-cell constraint is

\[
  z_{L^*}(x)-z_k(x)\ge m_{safe}.
\]

Only violated inequalities contribute scorer loss:

\[
  H(\delta)=\sum_{p,k\ne L^*_p}
  [m_{safe}-z_{L^*_p}(F(q(\delta))_p)+z_k(F(q(\delta))_p)]_+.
\]

There is no CE and no incentive to drive an already-satisfied pixel farther into its cell. A small
`log(1+|delta|/qstep)` description proxy and deterministic soft-threshold proximal update bias the
local descent toward shorter payloads. All scorer parameters are frozen and excluded from optimization.
The best iterate is ordered lexicographically by: most margin-feasible pixels, then fewest exact
serialized bytes, then smallest hinge residual. This is a local iterative constrained solve, not a
minimality proof.

The gradient path uses STE for payload quantization and camera rounding. Every reported step is
recomputed through hard int8 payload values and hard camera uint8. Final output reports STE-versus-hard
logit closure and camera row-space leakage after clamp/quantization; it explicitly does **not** claim
`A(Q(BY))=Y`.

## Exact description grammar (IMPLEMENTED)

Every step serializes the same quantized tensor it scores:

`CIS1 | u32 header_len | canonical JSON header | u32 zlib_len | zlib-9(int8 delta LE) |
u32 pose_len | selected gt_poses float32 LE`.

The header fixes schema, pair IDs, tensor shape, dtype, quantization step, pose dtype, and pose shape.
All prefixes, compressed bytes, and six pose scalars per selected pair are counted. The fixed banked
checkpoint is an excluded constant and is reported separately by bytes/hash. This is an exact
**incremental-description** grammar, not an archive byte-close.

## Honest sample policy (IMPLEMENTED; selected rows pending)

Default selection reads cached margins for all 600 pairs without looking at candidate outcomes. It
splits temporal indices into `sample_pairs` equal strata, defines target fragility as
`mean(cached_margin < m_safe)`, and alternates the within-stratum 0.25/0.75 fragility quantile with
pair-index tie breaking. The receipt records temporal stratum, fragility, class histogram, and class
entropy. Explicit pair IDs are allowed only under the label
`explicit_override_not_default_stratified_evidence`.

- Selection policy receipt: **PENDING PRIMARY MEASUREMENT**
- Selected pair IDs: **PENDING PRIMARY MEASUREMENT**
- Per-pair temporal/fragility/class rows: **PENDING PRIMARY MEASUREMENT**

## Resumability and custody (IMPLEMENTED)

`--state` is atomically replaced after every scored step and preserves the already-derived next
stateless iterate, best iterate, completed step, pair IDs, history, input hashes, and complete solver
config. Resume refuses any hash, pair-set, or solver-config drift. `--output` is also atomic. Both
paths refuse `/tmp`-class roots. No decoded frame tree or cache is written, and the final state remains
preserved.

The receipt records the exact rebuild argv, state bytes/hash, checkpoint/cache/modules/scorer hashes,
per-step metrics, final aggregate, and final per-pair rows.

## Triality

### DSL leg

The typed CLI plus content-hashed persistence config is the executable DSL for this research-only PoC:

`checkpoint, gt_cache, upstream_root, output, state, seed, sample_pairs|pair_indices, steps, m_safe,
quantization_step, max_delta, learning_rate, description_proxy_weight, cpu_threads`.

It compiles directly to one deterministic local solve. It contains no hidden optimizer state and no
heavy/paid actuation verb.

### DAG FEED

```text
cached n600 {L*, margins, gt_poses}
  -> deterministic temporal + fragility stratification
  -> read-only banked v9c2 EMA BEST
  -> canonical NumPy base decode (original EMA values; selected code reindex)
  -> quantized scorer-grid RGB delta
  -> sparse minimum-norm right-inverse B
  -> add + clamp + hard uint8 camera frame
  -> real evaluator resize A via SegNet.preprocess_input
  -> frozen CPU Torch SegNet convolution
  -> exact winner-cell margin meter
  -> exact incremental-description meter
  -> atomic per-step state + final advisory receipt
```

### Canonical-equation candidate

Candidate identifier: `constructive_inverse_local_constrained_mdl_v1`.

\[
  \delta^\dagger \in \operatorname*{lexmin}_{\delta\in q\mathbb{Z},\;|\delta|\le d_{max}}
  \left(-|\mathcal F(\delta)|,\;|D(\delta,P^*)|,\;H(\delta)\right),
\]

where `F(delta)` is the set of pixels satisfying every target-vs-competitor margin inequality after
`hard_uint8(base + B(delta))` and the frozen `SegNet∘A`, `D` is the exact grammar above, and `H` is the
satisficing hinge. The harness returns an iterate found by deterministic proximal gradient; it does not
claim the global lexicographic minimizer. Formal registry landing is an explicit MAIN-owned debt because
this worker owns only the harness and memo.

## Primary measurement table

Exact command/config/custody hashes: **PENDING PRIMARY MEASUREMENT**

| step | feasible fraction @ `m_safe` | argmax feasible | violated pixels | newly feasible | regressed | hinge sum | payload B | pose B | total B | nonzero symbols | clip fraction | elapsed s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **PENDING PRIMARY MEASUREMENT** | | | | | | | | | | | |

Final per-pair rows: **PENDING PRIMARY MEASUREMENT**

Hard-authority closure and quantization leakage: **PENDING PRIMARY MEASUREMENT**

Scoped verdict: **PENDING PRIMARY MEASUREMENT**. The evidence command must return nonzero evidence status
if no initially infeasible real pixel enters the declared margin cell, after first writing the honest
receipt.

## Pose and closure debts

- `gt_poses` are stored as six little-endian float32 scalars per selected pair and counted exactly.
  This PoC does not realize those targets through PoseNet. The joint PoseNet realization gate remains open;
  post-hoc storage alone is not claimed to solve pose.
- The incremental grammar is not a complete `archive.zip` and supplies no archive-rate score claim.
- Full n600 byte-close, receiver parse-back in the submission grammar, decode-under-30-min proof, and exact
  contest CPU/CUDA evaluation remain owed.
- The fixed banked curvelet/self-orient receiver is reused. No new basis is fitted or controlled; no
  Fourier basis/controller is introduced.
- Exact pointer delta: **0 by construction in this local-only worker; pointer intentionally untouched.**
- MAIN must perform the merge-boundary review, real PoC run, exact-row insertion, sacred-input metadata/hash
  comparison, frontier-pointer hash comparison, and any serializer commit.

## Worker verification

- Python compile/import: PASS.
- `--help`: PASS.
- `--self-test`: PASS, explicitly `PASS_SYNTHETIC_NOT_EVIDENCE`; covers `AB=I`, `P²=P`,
  `A(I-P)=0`, rank/null dimensions, serialization parse-back, and resume identity.
- Existing resize-null tests: **PENDING PRIMARY VERIFICATION**.
- Real n600-stratified PoC: **PENDING PRIMARY MEASUREMENT**.
