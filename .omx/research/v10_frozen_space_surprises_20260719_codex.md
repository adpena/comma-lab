# V10 frozen-space surprises: adversarial source re-derivation

Date: 2026-07-19 UTC  
Task: #564  
Lane: `lane_v10_frozen_space_surprises_20260719`  
Status: `research_only=true`; source review and exact `$0` probes only  
Axis: source-derived plus explicitly tagged `[Darwin-arm64 CPU advisory]` inherited measurements  
Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**  
Authority: no launch, paid dispatch, contest score, promotion, submission, or pointer authority; MAIN landing review is mandatory

## Verdict

The highest-value surprise is not a new carrier. It is that v10 replaced the
Pose term's globally pooled Euclidean sublevel set with 600 invented per-pair
hard constraints. For the same externally declared global budget, the literal
C4/C9 veto is therefore stricter than the frozen score term and can reject an
allocation with a lower recomposed objective; existing conditional rows witness
that gate logic but are not viable archives. The associated
`d_pose=2.5e-4` “binding crossover” is also being asked to carry more authority
than its derivation gives: it is an
equality of scalar-coordinate derivatives, not a feasibility boundary, and its
`1/sqrt(d_pose)` coefficient divergence cancels in the native Pose-error
coordinates.

Four additional findings survive after those two Pose corrections:

1. v10's PDW2 SPEC prose makes both classes win the same exact tie, while the
   settled first-index `argmax` execution makes the cells lexicographically
   half-open; no bad executable consumer was found in this review;
2. historical factorization prose identifies the U/V analysis covectors with
   the primal luma-null plane, but those two planes differ by `30.27914784 deg`,
   so the active Euclidean luma/null split cannot be called U/V sensitivity;
3. under the inspected official default invocation, the video input operator
   yields a distinct final `B=8` batch after 37 `B=16` batches, so uniform-batch
   cache closure is not exact authority; and
4. a shortest finite-field recurrence is a genuinely untried, exact, low-
   confidence coder for the current `(1200,32)` code tensor. Its raw packet has
   a cheap `20,518 B` section prefilter, but only a full parent-archive repack
   can establish a rate or score change.

No item below is a score claim. The measured rows are advisory or conditional
payload evidence, and every proposed gain remains zero until exact receiver-
closed bytes and hard-oracle outputs exist.

## Authority custody and the frozen map re-derived

The primary source bytes reviewed were:

| frozen source | SHA-256 | authority used here |
|---|---|---|
| `upstream/modules.py` | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` | per-pair Pose MSE, last-frame Seg first-max, joint fork |
| `upstream/frame_utils.py` | `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90` | RGB-to-YUV6 map, partial batches, raw pair assembly |
| `upstream/evaluate.py` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` | global distortion pooling, archive bytes, score, default batch geometry |
| `upstream/README.md` | `68ea239d7333696e79716e47a9c4288d2918efbcd8912f78932b0befe0af872b` | 30-minute total and instance/rule-118 boundary |

Let `N=600`, let `e_i in R^6` be the difference in the scored first six Pose
outputs for pair `i`, and let `F` be the total number of last-frame Seg argmax
disagreements over `N*384*512` cells. `modules.py:82-84` and
`evaluate.py:81-92` give exactly

```text
q_i       = ||e_i||_2^2 / 6
D_pose    = (1/N) sum_i q_i = ||e||_2^2 / (6N)
d_seg     = F / (N*384*512)
S         = 100*d_seg + sqrt(10*D_pose) + 25*archive_bytes/37,545,489
S_pose    = ||e||_2 / sqrt(360)                    when N=600.
```

Useful exact prices, all **DERIVED**, are:

| event | score-law price | archive-byte equivalent at fixed other terms |
|---|---:|---:|
| one Seg flip | `8.477105034722222e-7` | `1.27310821533203125 B` |
| one archive byte | `6.658589531221714e-7` | `1 B` |
| the measured 114-flip n600 lattice band | `9.663899739583333e-5` | `145.1343365478515625 B` |
| one unit of global Pose residual norm `||e||_2` | `1/sqrt(360) = 0.05270462766947299` | `79,152.84073653175 B` |

The 114-flip row is cited only as a unit conversion. None of the findings below
claims to cause those flips.

## Ranked surprises

| rank | finding | epistemic status | immediate consumer |
|---:|---|---|---|
| 1 | Replace 600 per-pair Pose caps with the score term's one pooled norm | **DERIVED** source contradiction; existing rows are a **MEASURED advisory logic witness** | amend C4/C9 and #536 |
| 2 | Demote `2.5e-4` from “binding boundary”; optimize the native error norm and test plane proximity only as a candidate trust region | norm reparameterization **DERIVED**; trust-region choice **CONJECTURE**; gain **UNMEASURED** | canonical Pose law, C4 objective/DSL |
| 3 | Correct the SPEC's equality prose and reuse settled first-max execution | **CONFIRMED source semantics + DERIVED SPEC correction**; executable defect **NOT FOUND** | C0; C3/C5 only upon evidence |
| 4 | Separate U/V analysis covectors from primal luma-preserving RGB directions | **DERIVED exact linear algebra**; active metric needs relabel/reinterpretation; score impact **UNMEASURED** | P3/C4/C9 factorization and basis IDs |
| 5 | Try a shortest `GF(257)` recurrence code on each role/channel code stream | theorem **CONFIRMED**, codec **DERIVED**, savings **CONJECTURE** with low confidence | one C6 `$0` A/B only |
| 6 | Preserve the official `37 x B16 + 1 x B8` inference geometry | schedule **DERIVED**, batch-dependent drift **CONJECTURE** | C0 cache closure and C11 custody |

## 1. A Pose-term sublevel set is one global L2 ball, not 600 block balls

### Claim and label

**DERIVED:** C4's constraint `q_i < 2.5e-4 for every pair i` is not present in
the evaluator. For any externally declared global Pose budget `D0`, the score-
term sublevel set is

```text
sum_i ||e_i||_2^2 <= 6*N*D0,
```

one ball in `R^(6N)`. C4 instead intersects 600 six-dimensional balls. The
latter is a strict subset, changes the KKT system from one global dual to 600
pair duals, and prohibits cross-pair rate allocation that the score explicitly
allows. Per-pair telemetry remains valuable, but a per-pair hard cap needs its
own measured robustness justification; it cannot be labeled evaluator-derived.

At `N=600,D0=2.5e-4`, the same Pose-term sublevel set mathematically contains a
point with all error energy in one pair, `q_1=N*D0=0.15`, or any other
distribution with the same sum. This is not a recommendation to concentrate
error; it is a proof that the two feasible sets differ.

### Exact derivation and falsifiable `$0` probe

For `D>0`, the per-pair marginal is

```text
dS_pose/dq_i = sqrt(10)/(2*N*sqrt(D)).
```

At `N=600,D=2.5e-4`, it equals `1/6`. The first-order cost of a change
`Delta q_i=0.001` is therefore about `0.0001666667` score, or `250.30326`
archive-byte equivalents—not the cost obtained by applying `sqrt(10*q_i)` to
that pair in isolation. The exact finite change from this base point is
`0.00016638981097470695` score, or `249.88747270651754` byte-equivalents.

The exact `$0` probe is a pure recomposition of the already-custodied per-pair
vectors and conditional bytes:

1. parse every `q_i`, `d_seg`, and physical/conditional byte row without
   re-running a scorer;
2. compute `D=(1/N)sum q_i` and the literal frozen three-term objective;
3. run the same candidate selection twice: once with the current pair veto and
   once with one pooled Pose dual;
4. preserve per-pair maxima/quantiles as diagnostics in both modes; and
5. refuse any conclusion whose byte field is not exact archive bytes.

Existing n24 precision rows already witness the logical consequence. Drop-2
has global `D=7.5275306e-5` with two pair-cap violations; drop-3 has global
`D=1.4154703e-4` with four. Both global values are below `2.5e-4`. Under the
declared n600-scaled **conditional range-payload** interpretation, drop-1 and
drop-3 recompute to `707.5750040983669` and `457.54699926031174`; drop-3 is
lower by `250.02800483805515` despite being vetoed by C4. These enormous values
prove
that both payload formulations are rate-dead. They are a gate-logic witness,
not candidate archives or contest scores.

### Wall/law interaction, magnitude, and consumer

- **Re-scopes:** `pose_plane_proximity_corollary_v1`, C4, C9, and #536. It does
  not overturn any measured Pose output; it removes an extra feasibility rule.
- **Expected magnitude:** **UNMEASURED**. The first-order local marginal scale
  above is `250.30326 B` per `Delta q_i=0.001` at the cited global point; the
  corresponding exact finite change is `249.88747270651754 B`. Actual recovery
  is whatever exact bytes or Seg improvement the veto currently suppresses.
- **Consumer:** amend C4 to one pooled `D_pose`/score constraint and C9 to one
  global Pose dual. Retain an optional typed tail-risk cap only as
  `ASSUMED_ROBUSTNESS_GUARD`, default OFF until an axis-drift probe earns it.

## 2. The `2.5e-4` “binding crossover” is a coordinate comparison, not a physical wall

### Claim and label

**DERIVED:** the identity

```text
d sqrt(10D)/dD = 5/sqrt(10D) = 100  iff  D=2.5e-4
```

is algebraically correct. What does not follow is that Pose “binds” there, that
`D<2.5e-4` is a feasibility region, or that `100` is an exact derived clamp for
the optimizer. Since `D=||e||^2/(6N)`, the frozen Pose term is the norm

```text
S_pose(e) = sqrt(10/(6N)) * ||e||_2.
```

Away from `e=0`, its gradient has constant norm
`sqrt(10/(6N))=1/sqrt(360)` for `N=600`; at zero it has the corresponding
subgradient ball. The scalar `1/sqrt(D)` factor cancels the `O(||e||)` gradient
of the MSE. A coefficient blow-up can still be a numerical implementation
problem, but a cap that changes this chain rule is a measured training heuristic,
not an exact consequence of the contest score.

This algebra proves only that `rho_p^(2x2)` is not the exact six-output score.
Using source-plane proximity to keep a linearized `J_y` proposal inside a
validated basin is a **CONJECTURE / design recommendation**, not a derived
consequence: it is one candidate typed trust region whose radius would have to
be earned empirically in an A/B against the existing control.

### Exact `$0` probe

Using the existing n24 `J_y`/Pose sidecars only:

1. stack the first-six residuals into `e` and verify
   `sqrt(10*mean_i(||e_i||^2/6)) == ||e||/sqrt(360)` in fp64;
2. for every stored proposal direction, compare the exact norm directional
   derivative with the current `min(100,5/sqrt(10D))*grad(D)` implementation;
3. separately tag proposals outside the measured plane-proximity radius; and
4. rerank the same proposals under `(exact global Pose norm + rate + Seg)` with
   `rho<=r_valid` versus `(rho penalty + pair caps)` before any hard-oracle run.

The probe falsifies the reformulation if rankings are identical and no proposal
is excluded only by the old coefficient/cap; it authorizes neither adopting a
`rho` trust region nor launching training.

### Wall/law interaction, magnitude, and consumer

- **Consistent with measurements, re-scopes prose:** near-source Pose inactivity
  and far-generator Pose destruction remain measured instance facts. The change
  is to the inference “therefore Pose binds at `2.5e-4`” and to C4's objective.
- **Expected magnitude:** **UNMEASURED**. Exact accounting is
  `Delta S_pose=Delta||e||/sqrt(360)`, or `79,152.84073653175` byte-equivalent
  units per unit change in `||e||`. No directional change has been measured.
- **Consumer:** append a reviewed successor to the canonical Pose law and
  implement the global norm directly. Test, rather than assume, a typed `rho`
  stage-boundary trust-region candidate; preserve the old objective as the
  mandatory OFF/control arm.

## 3. First-index argmax makes cells half-open; the v10 SPEC assigns equality twice

### Claim and label

**CONFIRMED source semantics + DERIVED correction:** `SegNet.compute_distortion`
uses `torch.argmax`. On exact co-maxima, PyTorch returns the first index. For a
target class `k`, the exact native-f32 cell is therefore

```text
C_k = intersection_(r<k) {ell_k >  ell_r}
      intersection_(r>k) {ell_k >= ell_r}.
```

For canonical `i<j` and `g_ij=ell_j-ell_i`, v10 SPEC line 167 says class `i`
wins for `g_ij<=0` and class `j` wins for `g_ij>=0`. At `g_ij=0`, only `i`
wins. The symmetric prose is a literal SPEC/certificate defect. Executable
first-max behavior is already settled by
`f32_receiver_arithmetic_exactness_admissibility_v1`, and
`src/tac/boundary_math/power_diagram_witness.py` uses NumPy's first-max
`argmax`. This review found no executable equality consumer that applies the
symmetric inequalities, so the present verdict is **C0 SPEC prose only**. C3 or
C5 becomes implicated only if MAIN locates such a consumer.

The geometric reference is Edelsbrunner and Muecke's
[Simulation of Simplicity](https://doi.org/10.1145/77635.77639): degeneracies
need one consistent symbolic ordering. The implementation authority is the
[PyTorch `argmax` contract](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html),
not the paper.

### Exact `$0` probe

Enumerate all 31 nonempty co-max subsets of five native-f32 logits, every pair's
`g<0`, `g=0`, `g>0`, and both `nextafter(0,+/-inf)` neighbors. For each fixture:

1. compare NumPy and frozen Torch first-max;
2. evaluate the C3/PDW2 adjacency certificate after canonical parse/re-encode;
3. require equality to belong only to the lowest co-max class;
4. test all nine measured PDW2 adjacency edges and ordinary non-tie fixtures;
5. refuse reassociation or backend substitution that changes the winner.

If a future C3 checker routes every decision through literal first-max and never
consumes the symmetric inequalities, it remains unaffected.

### Wall/law interaction, magnitude, and consumer

- **Consistent with:** `f32_receiver_arithmetic_exactness_admissibility_v1` and
  the frame-195 class-0 tie. **Re-scopes:** C0 SPEC prose now; C3/C5 only upon
  executable evidence.
- **Expected magnitude:** zero current executable or payload delta; real-corpus
  incidence **UNMEASURED**. One wrongly certified Seg cell would cost
  `8.477105034722222e-7` score or `1.2731082153320312 B` equivalent. The known
  114 mismatches are not attributed to this defect.
- **Consumer:** correct the C0/SPEC prose and reuse the existing first-max
  authority/helper. Do not create a duplicate comparator ID; audit C3/C5 only
  if an executable consumer of the bad prose is found.

## 4. U/V rows are analysis covectors, not the primal luma-null RGB plane

### Claim and label

**DERIVED exact linear algebra:** away from the tiny upper clamp cells,
`rgb_to_yuv6` has analysis rows

```text
ell = ( 0.299,  0.587,  0.114)
u   = (-0.299, -0.587,  0.886) / 1.772
v   = ( 0.701, -0.587, -0.114) / 1.402.
```

Both `u dot (1,1,1)` and `v dot (1,1,1)` are zero, so
`span{u,v}=(1,1,1)^perp`. But `ell dot u=-0.18790406320541758` and
`ell dot v=-0.10553922967189727`, so that span is not `ker(ell)`. The two
planes' nonzero principal angle is `30.27914784 deg`; the spectral norm of the
difference between their orthogonal projectors is `0.504213367`.

The historical statement in
`.omx/research/frozen_scorer_exact_factorization_20260715.md:44-56` that the
“chroma plane” is both `span{U-row,V-row}` and the orthogonal complement of
`ell` is therefore false. The P3 analysis docstring in
`tools/c2_perclass_stratum_carrier_analysis.py:17-20` repeats it. Its active
energy split at lines 345-374 is internally valid as the Euclidean
`span{ell}`-versus-`ker(ell)` split. It is not a U/V analysis split, however,
so the output must be relabeled/reinterpreted and cannot support earlier U/V or
“chroma sensitivity” conclusions as written.

A primal luma-preserving displacement must satisfy `ell dot delta_rgb=0` and
may use, for example,

```text
(1, -0.299/0.587, 0),  (0, -0.114/0.587, 1).
```

Conversely, mapping an analysis gradient into Y/U/V coordinates requires the
full matrix dual/Gram solve, not treating the analysis rows as an orthonormal
primal displacement basis.

### Exact `$0` probe

Add a source-only 3x3 matrix fixture that checks the dots, ranks, principal
angle, projector norm, and `ell dot delta_rgb=0` for the primal basis. Then scan
every consumer of `span(U,V)`, `chroma_plane`, or `luma_null`; classify each as
analysis-covector, primal-perturbation, or Euclidean diagnostic. For each active
P3 actuator, apply unit synthetic directions through the literal
`rgb_to_yuv6` map and verify the declared invariant before touching real data.

### Wall/law interaction, magnitude, and consumer

- **Does not reopen:** the exact 2x2 Pose visibility law or channel-necessity
  rows measured by literal ablation. It does require reinterpreting any result
  inferred specifically from the active luma/null VJP labels.
- **Expected magnitude:** no score delta is measured. The `0.504213367`
  projector distance is a worst-case unit-sensitivity attribution error, not a
  `50.4%` byte or score prediction.
- **Consumer:** amend the frozen factorization and P3/C4/C9 basis vocabulary;
  relabel the active metric as an `ell`/`ker(ell)` Euclidean diagnostic;
  register separate typed IDs for `yuv_analysis_covectors.v1` and
  `luma_preserving_primal_basis.v1`; require the 3x3 fixture before P3 acts.

## 5. A shortest finite-field recurrence is an exact, falsifiable code-stream probe

### Claim and label

**CONFIRMED theorem / DERIVED codec / CONJECTURE savings (low confidence):** the
exact rate-coder donor measured by #557/#558 has a `(1200,32)` int8 code tensor,
or 64 fixed-role/channel sequences of length 600, and its complete framed
Brotli section is `20,518 B`. The donor SHA-256 is
`6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`;
the quantized code tensor SHA-256 is
`29dca0a7387c3ba3cf7876e1bfbafe24d6944165512939b6b8eb84afc377dda2`
at scale `0.012484688311815262`. The prior context-coder pass tested left/up
sign-magnitude models, not minimum linear recurrence. An unbuilt v10 code
tensor does not inherit this donor's outcome.

Map each signed int8 symbol injectively to `{0,...,255}` in `GF(257)`. For each
fixed `(frame_role,channel)` sequence—never by flattening or alternating roles—
Berlekamp-Massey constructs a shortest LFSR for that finite sequence. Massey's
primary result is [Shift-Register Synthesis and BCH Decoding](https://www.isiweb.ee.ethz.ch/archive/massey_pub/pdf/BI411.pdf),
IEEE Transactions on Information Theory 15(1), 1969, pp. 122-127.

The complete candidate must store, for each of 64 streams, length `L`, `L`
initial symbols, and `L` recurrence coefficients, plus every shape, role,
quantizer scale, frame, termination, and checksum byte. With a 10-bit `L` and
packed 9-bit field elements, the metadata-free raw-packet prefilter is

```text
64 * (10 + 18*mean(L)) / 8 < 20,518
mean(L) < 141.930555...
```

before shared metadata. This is formulation-specific, not a necessary condition
for the BM family: its fields can themselves compress, and `20,518 B` is a
section count rather than a marginal archive delta. A mean complexity near the
unstructured null scale `N/2=300` would make this raw form lose badly. The
same-donor measured temporal delta worsened `20,355 B` to `33,411 B` (`+64%`) in
`src/tac/canonical_equations/t5_crucible_measured_laws_20260707.py`, while the
carrier-agnostic cross-pair codec found no exact dedup and lost with VQ,
second-order/motion delta, and range-code variants in
`src/tac/losses/cross_pair_latent_codec.py`. Those are measured adverse priors;
they are not a claim of zero mutual information and do not kill this distinct
formulation.

### Exact `$0` probe

1. rederive the exact current donor tensor and quantizer metadata by hash;
2. run BM independently on all 64 sequences over `GF(257)`;
3. pack every field and all receiver framing, decode from a clean process, and
   repack the complete parent archive;
4. permit field residue `256` in recurrence coefficients (`-1 mod 257`) but
   reject reconstructed source symbols outside `{0,...,255}`, any tensor/hash
   mismatch, or any omitted fitted coefficient;
5. canonical re-encode; record both physical section bytes versus `20,518 B`
   and exact full-archive bytes versus the parent archive; and
6. treat `>=20,518 B` as a kill only for the standalone raw-section
   formulation, and reject any full-archive non-win or material receiver-runtime
   regression.

No scorer run is needed only if the quantized tensor **and** exact scale, zero
point, dtype, order, shape, and dequantization arithmetic match the donor (or
the dequantized tensor itself is proved identical). Otherwise receiver output
invariance has not been established.

### Wall/law interaction, magnitude, and consumer

- **Outside, but respects:** #557/#558's tested spatial contexts and the dense
  plane-residual rate wall. It targets a small latent tensor with a different
  exact model. It does not reopen the older donor's delta/VQ/dedup negatives.
- **Expected magnitude:** only after exact full-archive repack,
  `Delta S=6.658589531221714e-7*(B_new_archive-B_parent_archive)`. The
  `20,518 B` section corresponds algebraically to approximately
  `0.013662094000160712` score-
  units, but that is not a measured archive-score ceiling or prediction.
  Confidence of any saving is deliberately low.
- **Consumer:** one standalone C6 alternate-coder receipt. Only a strict exact
  parent-archive byte win may add a grammar mode; otherwise append the narrowest
  formulation-scoped negative and do not send it to C9.

## 6. The official frozen map includes a distinct final B8 kernel geometry

### Claim and label

**DERIVED schedule / CONJECTURE impact:** `evaluate.py` defaults to
`batch_size=16`; the CPU/raw paths emit a partial final batch. On CUDA, the
partial `B=8` originates in `fn.experimental.inputs.video` when the video does
not divide uniformly across `max_batch_size`, as documented by
[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.inputs.video.html).
The `DALIGenericIterator` is constructed without a finite `size` or
`reader_name`, so `LastBatchPolicy.PARTIAL` is not the cause here. For the
inspected one-video, 600-pair default invocation, the schedule is exactly 37
batches of 16 and one batch of 8. A cache or parity
receipt measured uniformly at B1, B8, B16, or B32 has not by itself closed the
official mixed geometry. Eval mode removes learned batch-stat coupling, but
backend kernels can still be shape-dependent at native fp32, precisely where
ULP ties matter.

### Exact `$0` probe

Run the final eight custodied GT/candidate pairs once as their official B8 tail
and once as the first eight rows of a B16 call, on each claimed CPU/CUDA
authority axis. Compare preprocessed tensors, all five Seg logits/argmaxes,
first-six Pose outputs, and per-pair distortions. Record every float delta; hard
argmaxes and recomposed distortions must match exactly to claim batch closure.
Otherwise the production cache must preserve the literal mixed schedule.

### Wall/law interaction, magnitude, and consumer

- **Consistent with:** the existing batch-geometry caution; this is its exact
  official-tail instance, not a new batch-stat exploit.
- **Expected magnitude:** **UNMEASURED**. One Seg flip is
  `8.477105034722222e-7` score / `1.2731082153320312 B` equivalent; Pose changes
  use the exact global norm. No drift is assumed.
- **Consumer:** C0/cache identity and C11 evaluator custody. Record the full
  batch-size sequence, not merely a nominal batch size.

## What I checked and found NOT surprising

These paths were inspected and are deliberately not re-opened:

- Exact factor-2 integer-plane feasibility, canonical support fill, and
  preimage-dependent native-f32 noise are already consumed by C5. A generic
  “optimize the fiber” proposal is not new.
- The broad “argmax cell intersect Pose tube” reformulation predates v10. A
  camera-native tube that discards the integer-plane ABI remains receiver-
  construction-blocked; it is not promoted here as a new surprise.
- Stateful whole-video decode and cross-pair prediction are legal, but prior
  frame prediction, cross-pair phase work, lossless temporal deltas, and a
  plane-context probe already exist. C1's independent-plane wording should not
  forbid a future measured C6 context mode, but this is not a novel finding.
- The head's rank-four quotient, affine gauge, ten pair differences from four
  reference rows, and PDW2's `138/133 B` inner packet are settled. No new
  counted sign-state stream exists for a braid/Lehmer code.
- The four luma phases plus two 2x2 chroma averages are exact and already in the
  v10 map. U/V upper-clamp active sets deserve the already-owed P3 measurement,
  but their existence is not a surprise.
- Dense exact plane/numerator residuals remain rate-dead across the measured
  formulation union. The `GF(257)` probe is kept only because it targets the
  small pair-code tensor and has an exact, cheap kill.
- Grid-LSTM/plane-context, categorical spectra, Nielsen information geometry,
  and polynomial compressor ideas have local crosswalks. Their direct mappings
  either lack addressable bytes or are already represented in C6/C9.
- Prefix truncation, missing/trailing-frame behavior, and stale-output paths are
  source-visible but violate the project's full-frame/NO-FAKE contract and were
  already known. They are not legal optimizations.
- Multi-GPU evaluator sharding is by input file; with the one-video list it does
  not distribute pair work. That is a runtime planning constraint, not a score
  reformulation. Integer-plane CUDA parallelism inside inflate remains open.
- `AllNorm` uses frozen eval state; no cross-sample batch-stat channel was found.
- The `230,904` doubly blind camera pixels/frame and approximately `52%`
  resize-null response are already measured priors, not automatic byte savings.

## C-chain amendment map

| consumer | amendment proposed for MAIN review | what remains unchanged |
|---|---|---|
| C0 | reconcile Pose aggregation/crossover wording, SPEC equality prose, basis IDs, and official mixed-batch custody | append-only law history and source hashes |
| C3 | no change unless an executable consumer of the symmetric SPEC inequalities is found | PDW2 scorer-free pullback, settled first-max behavior, and full-neighborhood collateral gate |
| C4 | one global Pose norm/dual; A/B a typed `rho` trust-region candidate; optional tail guard is typed and OFF | exact 2x2 YUV6 map and native hard oracle |
| C5 | no new comparator; reuse native-f32 first-max and audit only upon executable evidence | preimage policy A/B and no scorer in decode |
| C6 | permit the BM mode only after an exact full parent-archive byte win; use `<20,518 B` only as the raw-section prefilter | complete framing, fitted-state charging, parse/re-encode |
| C9 | global Pose marginal/KKT, per-pair telemetry diagnostic rather than veto | one shared rate lambda and non-additive pool accounting |
| C11 | preserve/verify the official `37xB16+B8` sequence per authority axis | exact archive bytes, runtime, CPU/CUDA separation, operator GO |

The first code-changing unit should be a MAIN-reviewed C0 reconciliation and
source-only tests. This memo does not authorize a training or evaluator launch.

## Triality, scope, and review discipline

- **Equation leg:** the frozen score is re-expressed as global Hamming count,
  global Pose Euclidean norm, and exact archive bytes; the color-plane and
  lexicographic-cell corrections are explicit finite-dimensional algebra.
- **DSL leg:** proposed typed surfaces are `global_pose_norm.v1`, an A/B-only
  `pose_proximity_trust_region_candidate.v1`, distinct analysis/primal color-
  basis IDs, and an optional `gf257_lfsr_pair_code.v1` that is absent unless it
  wins. Equality execution reuses the settled first-max surface rather than
  inventing another comparator ID.
- **DAG leg:** C0's reconciliation precedes C4; C3/C5 are revisited only upon an
  executable tie-consumer finding; the one coder probe branches only into C6;
  C9 consumes the corrected global Pose curve; C11 retains final authority.
- **Verdict scope:** source-derived corrections plus inherited local advisory
  measurements. No contest CPU/CUDA score, receiver archive, or pointer move.
- **Pointer delta:** exactly none; `0.1910828242 [contest-CPU Linux x86_64]`
  remains the canonical pointer read from `reports/latest.md`.
- **Sacred-run delta:** none. No file under
  `experiments/results/levelset_n600_witness_20260717T113932Z/` was written.
- **Self-review:** each finding underwent premise, collision, magnitude, and
  authority attacks. Items were stopped before five rounds when the verdict
  stabilized; collisions are recorded in the NOT-surprising section.

MAIN must independently review at least:

1. the source algebra for global Pose pooling and the native-error gradient;
2. whether any per-pair cap has a separate operator/axis-robustness authority;
3. every equality consumer in C3/PDW2, not just the SPEC prose, before widening
   the present C0-only verdict;
4. active uses of “chroma plane” and any prior U/V inference before changing a
   measured code path;
5. the complete fitted/framing byte charge and full parent-archive repack in the
   speculative recurrence probe; and
6. the exact evaluator invocation behind the mixed-batch schedule on each
   claimed authority axis.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5 §8 and the v8 SPEC; delegated Task
#564 authority; `reports/latest.md`; lane/task/subagent ledgers; live arm inbox
and broadcast ledger; the four frozen upstream sources and hashes above;
`.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md`;
`.omx/research/SPEC_v10_capstone_RECONCILED_20260719.md`;
`.omx/research/spec_v10_reconciliation_and_kkt_verify_20260719_fable.md`;
the inverse-solve, secant, lattice, y-hat, PDW2, flattened-KKT, factorization,
and paper-crosswalk corpus; canonical Pose/f32/Seg-rate/head laws; current donor
rate-coder receipts; and `docs/operating_manual_craft_handoff.md` (SHA-256
`40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212`).

This review followed the craft handoff: re-derive from frozen source, preserve
sacred bytes, label every number, scope every negative, keep advisory axes
separate, checkpoint durable progress, and require MAIN review before landing.
