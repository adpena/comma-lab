# Constructive constrained-MDL inverse-solve — local PoC handoff

**Status:** `MEASURED_LOCAL_POC_LANDED_PIXEL` on a deterministic six-pair n600-stratified sample.
The best hard iterate moved 1,183 initially infeasible pixels into their declared winner cells,
regressed 523, and therefore gained 660 net margin-feasible pixels. This document does not claim a
score, promotion, rank/kill verdict, closed-form global optimum, or pointer movement.

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

### STORES CONSULTED

- The named design corpus: `campaign-is-a-constrained-mdl-inverse-solve`,
  `kolmogorov-program-technique-roles-and-gradient-control-surgical-targeting`,
  `frozen-scorer-exact-factorization`, and `null-subspace-rate-measure`.
- SPEC_v10 section 14.11 from `claude/p0_521_spec_v10_capstone_20260717`.
- `reports/latest.md`, the canonical frontier pointer, lane registry, subagent progress ledger,
  latest sister Codex/Claude memos, and the live inbox/broadcast surfaces.
- The canonical NumPy level-set oracle, exact resize projector, banked BEST checkpoint, stored n600
  cache, upstream SegNet source, and frozen SegNet weights named in the custody table below.

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

### Iterative convolutional residual (IMPLEMENTED + MEASURED)

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

## Honest sample policy (MEASURED)

Default selection reads cached margins for all 600 pairs without looking at candidate outcomes. It
splits temporal indices into `sample_pairs` equal strata, defines target fragility as
`mean(cached_margin < m_safe)`, and alternates the within-stratum 0.25/0.75 fragility quantile with
pair-index tie breaking. The receipt records temporal stratum, fragility, class histogram, and class
entropy. Explicit pair IDs are allowed only under the label
`explicit_override_not_default_stratified_evidence`.

Selection policy receipt: `all-600 temporal equal strata; fragility=mean(cached_margin<m_safe);
alternating within-stratum 0.25/0.75 quantile; deterministic pair-index tie break; no candidate
outcome peeking`.

| stratum | pair | fragility fraction | class entropy (bits) | class histogram `[0,1,2,3,4]` |
|---|---:|---:|---:|---|
| `[0,100)` | 90 | 0.0009816487630208333 | 1.6076221906434411 | `[45461,998,97649,2345,50155]` |
| `[100,200)` | 175 | 0.001220703125 | 1.5782229662475902 | `[46286,1195,97839,976,50312]` |
| `[200,300)` | 277 | 0.0009206136067708334 | 1.5611345052958985 | `[46255,865,98266,789,50433]` |
| `[300,400)` | 381 | 0.0011088053385416667 | 1.5756520661935272 | `[44686,1563,99480,874,50005]` |
| `[400,500)` | 424 | 0.0010426839192708333 | 1.6240611274768226 | `[44436,800,98065,3552,49755]` |
| `[500,600)` | 573 | 0.0013376871744791667 | 1.6979937677559303 | `[43033,1204,95974,6737,49660]` |

`MEASURED`: every selected pair covers all five target classes. Selection used only temporal index
and cached target margins; no candidate outcome was consulted.

## Resumability and custody (IMPLEMENTED)

`--state` is atomically replaced after every scored step and preserves the already-derived next
stateless iterate, best iterate, completed step, pair IDs, history, input hashes, and complete solver
config. Resume refuses any hash, pair-set, or solver-config drift. `--output` is also atomic. Both
paths refuse `/tmp`-class roots. No decoded frame tree or cache is written, and the final state remains
preserved.

The receipt records the exact rebuild argv, state bytes/hash, checkpoint/cache/modules/scorer hashes,
per-step metrics, final aggregate, and final per-pair rows.

The preserved state is 26,390,732 bytes with SHA-256
`1771b38725db23818e54ed58479e6ab04b750d30e39dd91789347ff527fbb4ed`.

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
this lane owns only the harness and memo.

## Primary measurement table

### Exact command and config

Working directory:
`/Users/adpena/Projects/pact/.omx/tmp/codex_worktrees/constructive_inverse_solve_20260718_20260718T064317Z`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  VECLIB_MAXIMUM_THREADS=4 PYTHONPATH=src:tools:experiments:. \
  /Users/adpena/Projects/pact/.venv/bin/python tools/constructive_inverse_solve_harness.py \
  --checkpoint /Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z/levelset_witness_ema_BEST.npz \
  --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --upstream-root /Users/adpena/Projects/pact/upstream \
  --output .omx/tmp/constructive_inverse_solve_poc_20260718.json \
  --state .omx/tmp/constructive_inverse_solve_poc_20260718.state.npz
```

Config: seed `20260718`; pairs `[90,175,277,381,424,573]`; steps `10`;
`m_safe=0.039180326461791926`; qstep `0.25`; `max_delta=16`; learning rate `5`;
description-proxy/L1-prox coefficient `1e-5`; CPU threads `4`. Runtime was Darwin arm64,
Python 3.13.12, NumPy 1.26.4, Torch 2.12.1, and zlib 1.2.12.

### Custody

| input/source | bytes | SHA-256 |
|---|---:|---|
| banked `levelset_witness_ema_BEST.npz` | 460,448 | `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef` |
| stored `gt_n600.npz` | 5,078,017,610 | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |
| frozen `segnet.safetensors` | 38,502,892 | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| upstream `modules.py` | — | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| harness at measurement | — | `2c42b79bd222f9b25e42cf45a22d6239b9ec6eff6a679e8c4f8d6f418ad3db61` |
| canonical NumPy oracle module | — | `62be77ee450ea140bbd1b7e1cf31ebb705b4a64d423a2facd625190727b5c6e5` |
| exact resize-projector module | — | `5f49b814ea3d8e4487beb93a5151c40bf7b4351f94d83281a686dc8d635d7824` |

Local receipt SHA-256: `ac40eb556b00deb32764978c0f8054b24926dbfc00ffaad1b1389836cc63535a`.
The fixed checkpoint is an excluded constant, not counted as incremental description bytes.

### Per-step hard results (`MEASURED`, local advisory)

| step | feasible fraction @ `m_safe` | argmax feasible | violated pixels | newly feasible | regressed | hinge sum | payload B | pose B | total B | nonzero symbols | clip fraction | elapsed s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9965684413909912 | 0.9970321655273438 | 4,048 | 0 | 0 | 993.796875 | 3,666 | 148 | 3,814 | 0 | 0 | 6.751135208178312 |
| 1 | 0.9965845942497253 | 0.9970465898513794 | 4,029 | 65 | 46 | 978.2047119140625 | 21,075 | 148 | 21,223 | 16,034 | 1.0921771576022365e-7 | 17.453773790970445 |
| 2 | 0.9966719150543213 | 0.997127115726471 | 3,926 | 361 | 239 | 925.5935668945312 | 64,679 | 148 | 64,827 | 71,502 | 1.638265700876218e-7 | 28.539312583161518 |
| 3 | 0.9967575073242188 | 0.9972169399261475 | 3,825 | 487 | 264 | 865.4568481445312 | 93,412 | 148 | 93,560 | 114,144 | 1.638265700876218e-7 | 39.562512333039194 |
| 4 | 0.9968422651290894 | 0.9973424077033997 | 3,725 | 721 | 398 | 815.7693481445312 | 120,932 | 148 | 121,080 | 158,497 | 2.730442929532728e-7 | 50.573516875039786 |
| 5 | 0.9969185590744019 | 0.9974220991134644 | 3,635 | 800 | 387 | 771.6692504882812 | 138,514 | 148 | 138,662 | 187,781 | 3.822619873972144e-7 | 61.733107540989295 |
| 6 | 0.9969804286956787 | 0.9975009560585022 | 3,562 | 950 | 464 | 732.1480102539062 | 158,150 | 148 | 158,298 | 221,543 | 5.460885859065456e-7 | 73.14381008315831 |
| 7 | 0.9970643520355225 | 0.9975687861442566 | 3,463 | 1,011 | 426 | 697.32177734375 | 172,778 | 148 | 172,926 | 246,879 | 6.553062803504872e-7 | 84.22102312510833 |
| 8 | 0.9970991611480713 | 0.9976306557655334 | 3,422 | 1,131 | 505 | 671.5729370117188 | 188,245 | 148 | 188,393 | 275,427 | 7.645239747944288e-7 | 95.56041779113002 |
| **9 best** | **0.9971279501914978** | **0.9976518154144287** | **3,388** | **1,183** | **523** | **663.0966186523438** | **201,927** | **148** | **202,075** | **299,613** | **8.737417260817892e-7** | **106.71397012495436** |
| 10 | 0.997122049331665 | 0.9976493120193481 | 3,395 | 1,285 | 632 | 663.5784912109375 | 214,753 | 148 | 214,901 | 323,418 | 1.0921771718130913e-6 | 118.4330295000691 |

The 148 pose bytes are a 4-byte length prefix plus 144 bytes for 36 float32 scalars. Step 9 is the
lexicographic best: step 10 has seven more violated pixels and costs 12,826 more bytes.

### Best per-pair hard rows (`MEASURED`)

| pair | feasible fraction | argmax feasible | violated pixels | hinge sum |
|---:|---:|---:|---:|---:|
| 90 | 0.9969685673713684 | 0.9975840449333191 | 596 | 118.1236343383789 |
| 175 | 0.9973042607307434 | 0.997833251953125 | 530 | 102.77851867675781 |
| 277 | 0.9974568486213684 | 0.9978790283203125 | 500 | 92.43904113769531 |
| 381 | 0.9976857304573059 | 0.9980621337890625 | 455 | 88.1300277709961 |
| 424 | 0.9970194697380066 | 0.9975331425666809 | 586 | 118.7891616821289 |
| 573 | 0.9963328242301941 | 0.9970194697380066 | 721 | 142.83627319335938 |

### Hard closure (`MEASURED`) and algebra (`DERIVED`)

- Serialized payload parse-back changed neither the hard camera nor logits: max absolute drift `0`/`0`.
- STE and hard logits closed at max absolute drift `0`.
- Float `AB-I` residual at the best payload: `8.881784197001252e-16`; contest certification
  `AB-I=6.661338147750939e-16`, `P²-P=6.661338147750939e-16`,
  `A(I-P)=5.134781488891349e-16`, relative range/kernel orthogonality
  `5.091550346623569e-18`.
- Hard scorer-grid drift from clamp/uint8 was L2 `133.0297819429746`, max absolute
  `0.4934997558592995`. Post-hard camera-kernel leakage was L2 `91.12644888219783`, max absolute
  `0.7108734422132874`. Therefore `A(Q(BY))=Y` is explicitly false after hard realization.
- Best pre-clamp clip fraction was `8.737417260817892e-7`; max float camera lift was
  `4.693084189487242`.

### Scoped verdict

`MEASURED`: this exact local solve landed real pixels in the target winner polytope. At step 9 it
converted 1,183 initially infeasible pixels, regressed 523, gained 660 net out of 1,179,648, reduced
the hinge by 330.70025634765625 (33.2764436%), and left 3,388 pixels outside the declared margin.

`MEASURED` negative: this configuration did **not** shorten the description relative to the zero-delta
baseline; the best counted description rose from 3,814 to 202,075 bytes (+198,261). It also did not
make the full sampled witness feasible. `VERDICT-SCOPE=INSTANCE x six stratified pairs x 10 steps x
fixed scorer-grid-delta grammar/config`; neither negative kills the inverse-solve family, a different
representation, or a joint Seg/Pose solve.

`INFERRED`: the PoC establishes a constructive fixed-convolution landing mechanism and exposes the
remaining rate tradeoff; it does not establish a minimum-description optimum or full feasible-set
membership.

## Pose and closure debts

- `gt_poses` are stored as six little-endian float32 scalars per selected pair and counted exactly.
  This PoC does not realize those targets through PoseNet. The joint PoseNet realization gate remains open;
  post-hoc storage alone is not claimed to solve pose.
- The incremental grammar is not a complete `archive.zip` and supplies no archive-rate score claim.
- Full n600 byte-close, receiver parse-back in the submission grammar, decode-under-30-min proof, and exact
  contest CPU/CUDA evaluation remain owed.
- The fixed banked pre-existing basis/self-orient receiver is reused. Its manifest omitted an explicit
  `basis_family`, so the canonical oracle retained the checkpoint's inherited default control. The
  measured self-orient fields were `freq_across=32`, `freq_along=8`, `tau=4`, `iters=4`. No new basis
  was fitted or controlled; this harness introduced no Fourier fit or controller.
- Exact pointer delta by this branch: **0**. `reports/latest.md` at the branch base and at the
  post-commit boundary has direct-file SHA-256
  `2c8e987723e68c7b7efcf058776c4052fc4b8990cab733194a7f9923350c291d` and still names the
  `[contest-CPU Linux x86_64]` pointer as `0.1910828242`. The branch diff contains only the harness,
  this memo, and the two lane-registration files; the harness has no pointer-write path.
- **Custody correction (2026-07-18):** an earlier working note reported `700e106c...` as the direct
  `reports/latest.md` SHA. A merge-boundary re-derivation against both the base Git blob and the
  isolated-worktree file could not reproduce that value, so it is withdrawn and must not be used as
  pointer-file custody evidence. This corrects provenance; it is not a pointer move. MAIN must
  independently repeat the score-value and direct-file-hash checks before landing.
- At the final primary check, the banked BEST checkpoint and scorer hashes still matched the pre-run
  values in the custody table; the 5 GB cache retained its pre-run size and mtime. These checks are
  external to the harness receipt, which hashes inputs once before solving. The sacred run as a whole
  remained live and its current EMA/resume/log/observer files advanced externally during this read-only
  PoC; no whole-directory-static claim is made and guarded outputs were written only under this worktree.
- MAIN must perform the merge-boundary code/math/custody review before landing. In particular, review
  the inherited-basis default, `range(A^T)` restriction, hard quantization drift, description grammar,
  sample custody, and the local-only verdict scope.

## Verification

- Primary fresh round-1 review: PASS after repair of resume ordering, sacred-path aliasing, hard
  parse-back/drift metrics, and hinge scaling.
- Ruff and Python compile/import: PASS.
- `--help`: PASS.
- `--self-test`: PASS, explicitly `PASS_SYNTHETIC_NOT_EVIDENCE`; covers `AB=I`, `P²=P`,
  `A(I-P)=0`, canonical rank/null/blind dimensions, canonical payload parse-back, and interrupted-versus-
  uninterrupted next-iterate identity.
- Existing resize-null tests: PASS, `23 passed, 1 skipped`.
- Real n600-stratified PoC: PASS with process exit `0` and status
  `MEASURED_LOCAL_POC_LANDED_PIXEL`.
- Completed-state real replay: PASS with exit `0`, no repeated step, byte-identical receipt SHA-256
  `ac40eb556b00deb32764978c0f8054b24926dbfc00ffaad1b1389836cc63535a`, and unchanged state SHA-256
  `1771b38725db23818e54ed58479e6ab04b750d30e39dd91789347ff527fbb4ed`.
- Lane registry: this advisory/research-only lane is consistently L1 with only `impl_complete=true`.
  Global `lane_maturity validate` still lists 110 older missing evidence paths in this sparse worktree;
  none names this lane, and no pre-existing evidence claim was altered.
- `git diff --check`: PASS.
