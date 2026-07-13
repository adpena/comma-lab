# P0 backward-mechanism pivot: checkpointing, adjoint, and reversibility

**Date:** 2026-07-13
**Checkpoint:** `p0_checkpoint_backward`
**Authority:** research-only wall-clock analysis and local training-gradient diagnostics
**Pointer delta:** **UNMOVED**. This memo is MEANS, not a score result.
**Mutation scope:** new memo only; no trainer, costate, live-run, scorer, DAG, DSL, equation-registry,
or submission mutation; no training; no paid/provider/GPU actuation.

> **Outcome:** `{82%-backward COMPUTE-/kernel-bound (DERIVED; the 82% share itself is diagnostic-only,
> not in-loop-confirmed) · full checkpointing makes the teacher approximately 18%-46% slower, never
> faster · adjoint-O(1) feasible N for the exact frozen scorer · reversible-approx feasible N with a
> certified bound · scoped verdict WORTH-BUILD-PHASE-SPLIT-THREAD-CONTROL; NO-GO-compute-bound for
> checkpointing/adjoint/reversible wall-clock use}`

`verdict_scope`: exact frozen `upstream/modules.py::SegNet` =
`smp.Unet('tu-efficientnet_b2', classes=5)`; input-costate VJP with frozen parameters; one real
receiver pair at `384x512`; local macOS-arm64 CPU / Torch 2.12.1 fp32 training-gradient diagnostic;
architecture-derived FLOPs and logical storage; no transfer to the unresolved in-loop component
shares, n600 trajectory fidelity, another host/build, contest CPU/CUDA, evaluator, archive, or score.

## 1. Gate first: compute-bound or activation-memory-traffic-bound?

### 1.1 Exact graph, not the architecture nickname

The scorer is not only an EfficientNet-B2 encoder. It is an EfficientNet-B2 encoder **plus a
high-resolution U-Net decoder**. A real instantiated-graph shape pass gives:

| exact convolutional graph component | DERIVED MAC/pair | share |
|---|---:|---:|
| EfficientNet-B2 encoder | 2,477,551,552 | 25.0022% |
| U-Net decoder | 7,290,224,640 | **73.5693%** |
| 5-class segmentation head | 141,557,760 | 1.4285% |
| **total** | **9,909,333,952** | 100% |

The 125 exact `Conv2d` nodes split by mechanism:

| convolution kind | nodes | DERIVED MAC/pair | share |
|---|---:|---:|---:|
| depthwise | 23 | 216,769,536 | **2.1875%** |
| pointwise `1x1` | 90 | 2,218,314,688 | 22.3861% |
| spatial dense/grouped | 12 | 7,474,249,728 | **75.4264%** |

Therefore an argument that “EfficientNet is depthwise, therefore its backward is memory-bound” is
false for this complete scorer. Depthwise MBConv work is only 2.19% of convolution MACs; the dense
high-resolution U-Net decoder is the dominant arithmetic surface.

The B2 encoder contains 23 MBConv/depthwise-separable blocks. Sixteen have same-shape residual skips;
seven do not. Four non-skip blocks stride by two, and other non-skip blocks change the representation.
Every encoder block contains squeeze-excitation. The U-Net decoder has five upsample + skip-concatenate
stages and ten dense `3x3` convolutions.

### 1.2 Frozen-input VJP FLOPs

All scorer parameters are frozen. The required derivative is only the input costate

```math
\lambda_x = J_{\mathrm{SegNet}}(x)^\top\lambda_z,
```

not parameter gradients. For a convolution, the input VJP is a transposed contraction with the same
leading MAC count as the forward convolution. Hence, counting a multiply and add as two FLOPs,

```math
F_{\rm fwd,conv}=2(9.909333952\times10^9)=19.818667904\ {\rm GFLOP},
```

```math
F_{\rm VJP,conv}\simeq F_{\rm fwd,conv}=19.818667904\ {\rm GFLOP}.
```

Nonlinearities, batch normalization, upsampling, concatenation, and SE add VJP work, but the frozen
graph does **not** pay a convolution weight-gradient contraction. The architecture therefore predicts
a backward/forward arithmetic ratio of order one, not the diagnostic `2472/537 = 4.603x` ratio.

### 1.3 Activation storage and arithmetic intensity

The exact real-pair autograd graph saved:

- **MEASURED diagnostic:** 658,879,572 B = **628.357 MiB** of unique non-parameter storage;
- **MEASURED diagnostic:** 37,980,192 B = 36.221 MiB of unique frozen-parameter storage;
- **MEASURED diagnostic:** 869 saved-tensor events, 737,813,364 logical bytes before storage
  deduplication.

The exact convolution nodes have a DERIVED minimum logical input+output+weight traffic of
488,054,864 B per pass. Even a deliberately pessimistic traffic model that charges that minimum for
both passes **and** charges all unique saved non-parameter storage once on write and once on backward
read gives

```math
I_{\rm pessimistic}
=\frac{2(F_{\rm fwd,conv})}
 {2(488{,}054{,}864)+2(658{,}879{,}572)}
=17.280\ {\rm FLOP/B}.
```

This pessimistically double-charges writes that overlap convolution outputs, so it is intentionally
biased toward a memory-bound diagnosis. It still leaves 97.81% of MACs in pointwise/dense convolutions.

A one-thread Torch profiler on the exact weights and real pair attributes 32.50% self CPU to
`aten::_slow_conv2d_forward` and 30.31% to `aten::_slow_conv2d_backward`; these execute 14,886 inner
slow-convolution calls apiece beneath the 125 graph nodes. The whole profiled pass allocated 2.14 GiB
through `aten::empty`, but allocation bytes are not a bandwidth counter. The load-bearing observation
is that wall clock sits in slow convolution kernel execution and slicing, not in a capacity stall that
discarding activations would remove.

### 1.4 Gating verdict

**DERIVED verdict: COMPUTE-/KERNEL-EXECUTION-bound, not activation-memory-traffic-bound.** More
precisely, this local CPU path is dominated by dense convolution arithmetic plus Torch's slow
convolution implementation/dispatch. It is not blocked on retaining 628 MiB of activations for a
single pair. The machine has ample capacity for the present batch-one path; the already-settled scorer
batch-dependence blocks laundering memory savings into an exact larger-batch speedup.

This verdict is strong enough for the checkpointing sign even though hardware counters were not
collected: checkpointing re-executes the exact slow convolution kernels that dominate the profile.
It removes storage but does not remove any of those contractions.

## 2. Reconcile the accounting's `2472 ms/pair` backward

The accounting derives `2472 ms = 3009 ms forward+backward - 537 ms forward`, hence an 82.15%
diagnostic backward share. It also records that `3009 ms x 600 = 1805 s/epoch`, while the real in-loop
epoch is 169.7 s: the diagnostic absolute is about 10.6x (rounded there as ~12x) too heavy.

The clean reconciliation is:

1. **MEASURED:** the accounting's `2472 ms` is not an in-loop duration.
2. **DERIVED:** the exact frozen-input VJP has approximately the same convolution FLOPs as its forward,
   because only `grad_input`, not `grad_weight`, is required.
3. **MEASURED diagnostic:** a warmed exact real-pair local profile has forward and backward of the same
   order. In the matched phase-control experiment below, static one-thread medians were 329.669 ms
   forward and 379.039 ms backward: backward share 53.49%, not 82%.
4. **UNKNOWN:** the precise source of the old 4.603x backward/forward ratio is not isolated. Cold graph,
   measurement-boundary, state, and instrumentation differences are candidates, not findings.
5. **Therefore:** neither the 2472 ms absolute nor the 82% ratio transfers to the training loop. The
   existing D-A in-loop timer remains the authority needed to determine the true `_teacher_fwd` and
   `_teacher_bwd` shares.

This does not rescue checkpointing. If the old 82/18 ratio transferred, checkpointing adds at least the
18% forward again. If the fresh 53/47-ish ratio transferred, it adds about 46%. The sign is adverse in
both cases.

## 3. Technique 1 — gradient checkpointing

Let

- `T_f` = ordinary teacher forward time;
- `T_b` = ordinary input-VJP time with stored activations;
- `c in [0,1]` = fraction of forward work inside checkpointed segments that must be recomputed;
- `Delta_mem` = wall time removed specifically by avoiding activation-memory stalls.

Then the wall-clock admission law is

```math
T_{\rm ckpt}=T_f+T_b+cT_f-\Delta_{\rm mem},
```

```math
T_{\rm ckpt}<T_f+T_b
\iff \Delta_{\rm mem}>cT_f.
```

For this compute-/kernel-bound path, `Delta_mem approximately 0`, so

```math
\frac{T_{\rm ckpt}}{T_f+T_b}
\ge 1+c\frac{T_f}{T_f+T_b}>1\quad(c>0).
```

**DERIVED wall-clock effect:**

- using the accounting's diagnostic `p_f=0.18`, full checkpointing is at least **+18% teacher time**;
- using the fresh exact one-thread diagnostic, full recomputation is **+44.23% teacher time**
  (`416.981/942.682`), before checkpoint framework overhead;
- if the teacher were 95% of the in-loop step, those conditional whole-step penalties would be
  approximately **+17.1% to +42.0%**.

The memory benefit is real: at most approximately 628 MiB/pair of non-parameter saved storage is
available to eliminate, less segment boundaries, U-Net skip tensors, and mandatory live values. That
is a **capacity** benefit, not a wall-clock benefit. The present task asks for wall clock.

**Scoped verdict: `NO-GO-compute-bound` for checkpointing as a wall-clock lever.** Reactivation:
only a future measured path that is capacity-stalled/OOM, or hardware-counter evidence that
`Delta_mem > cT_f`, can reopen it. It remains a valid memory-capacity mechanism in that different
scope.

## 4. Technique 2 — continuous-depth / Neural-ODE adjoint

For a genuine continuous-depth model

```math
\dot h(t)=f(h(t),t;\theta),\qquad a(t)=\frac{\partial L}{\partial h(t)},
```

the continuous adjoint obeys

```math
\dot a(t)=-\left(\frac{\partial f}{\partial h}\right)^\top a(t),
```

and can reconstruct/integrate the state and adjoint backward with asymptotically constant activation
memory. The original Neural ODE paper presents this continuous-depth constant-memory mechanism; ANODE
then shows why reverse reconstruction can be numerically unstable or give inconsistent gradients and
recovers accuracy with checkpoint-like state retention/recomputation.

That model does not match this scorer:

- only 16/23 encoder MBConv blocks are same-shape residual blocks;
- stride-2 blocks and channel-changing blocks are discrete jump maps, not a fixed-state flow;
- SE contains global spatial reduction and state-dependent channel gating;
- the five U-Net decoder stages upsample and concatenate separately retained encoder skips;
- 73.57% of convolution MACs live in that non-continuous U-Net decoder, not the residual encoder.

One can write a **hybrid** continuous system with jump conditions for stride/channel changes and U-Net
skip injections. Its exact discrete adjoint at each jump is just the transpose-Jacobian VJP PyTorch
already computes. Achieving O(1) memory then requires reconstructing or recomputing the discrete states,
which reduces to checkpointing/reversibility and adds compute.

Approximating B2+U-Net by a new ODE therefore changes the frozen teacher. It is a surrogate-family
proposal requiring costate/trajectory fidelity gates, not a cheaper exact backward mechanism.

**DERIVED wall-clock effect:** exact discrete semantics gain no contraction removal and add forward
reconstruction/ODE evaluations; approximate continuous semantics have unknown fidelity and normally add
solver function evaluations. On a compute-bound graph, the O(1)-memory trade is not a speed trade.

**Scoped verdict: adjoint-O(1) feasible `N` for the exact EfficientNet-B2 U-Net scorer.** A continuous
surrogate remains a distinct, unvalidated formulation; this is not a Neural-ODE-family kill.

Primary sources: [Calin, *Deep Learning Methods of Mathematical Physics*, vol. I,
DOI 10.1142/14702](https://doi.org/10.1142/14702); [Chen et al., *Neural Ordinary Differential
Equations*](https://arxiv.org/abs/1806.07366); [Gholami et al., *ANODE: Unconditionally Accurate
Memory-Efficient Gradients for Neural ODEs*](https://arxiv.org/abs/1902.10298).

## 5. Technique 3 — reversible approximation

A plain residual block is

```math
y=x+F(x).
```

It is not automatically reversible. If `Lip(F)=L<1`, fixed-point inversion

```math
x_{k+1}=y-F(x_k)
```

has the certified error contraction

```math
\|x_k-x^*\|\le L^k\|x_0-x^*\|.
```

No such `L<1` certificate exists for the frozen B2 residual branches. Their expansion/pointwise/depthwise
convolutions, SiLU, batch norm, and global SE gate were not spectrally constrained for invertibility.
Even a local successful iteration would be an empirical inverse, not a global bounded-error mechanism.

RevNet-style additive coupling can reconstruct activations exactly, but it requires a different
architecture. Retrofitting channel coupling changes the frozen scorer. It also cannot cover the seven
non-skip/shape-changing encoder blocks, and it does not remove the U-Net's separately needed skip states.
The 16 superficially eligible residual blocks live in an encoder that is only 25.00% of convolution
MACs.

**DERIVED wall-clock effect:** reconstruction evaluates `F` again (or repeatedly for fixed-point
inversion). This saves storage while adding compute to the smaller encoder fraction. It does not reduce
the dominant dense U-Net VJP. Approximate inversion additionally perturbs the costate with no certified
bound.

**Scoped verdict: reversible-approx feasible `N` for this frozen scorer with bounded error and wall-clock
gain.** A newly trained reversible surrogate is a different family/formulation, not killed here.

Primary sources: [Gomez et al., *The Reversible Residual Network*](https://arxiv.org/abs/1707.04585);
[Behrmann et al., *Invertible Residual Networks*](https://proceedings.mlr.press/v97/behrmann19a.html).

## 6. Technique 4 — does the one-thread standard help the backward?

**No. The backward thread law is different from the forward thread law.** The operator-approved
one-thread standard remains binding for training forwards, and auth eval remains untouched. The question
here is whether its forward result transfers to the VJP. It does not.

### 6.1 Direct matched diagnostic

Protocol: exact frozen weights; real decoded receiver pair 0; exact CE input costate; Torch 2.12.1 fp32;
interop=1; two warmups per arm; in-process ABBA-style arm groups; 14 samples/arm; timing includes
`torch.set_num_threads` calls; no MPS/CUDA; no file output. The mixed arm uses one intra-op thread for
the forward, switches to six immediately before `torch.autograd.grad`, then restores one before the next
forward.

| arm | forward median | backward median | total median | costate vs static-1 |
|---|---:|---:|---:|---|
| static 1 thread | 329.669 ms | 379.039 ms | 718.055 ms | reference |
| **1-thread forward / 6-thread backward** | 326.258 ms | **309.188 ms** | **641.696 ms** | byte-identical on pair 0 |
| static 6 threads | 565.924 ms | 379.988 ms | 951.120 ms | byte-identical on pair 0 |

Derived from the measured medians:

- mixed backward speedup: **1.2259x**; backward time reduction **18.4284%**;
- mixed teacher forward+backward speedup: **1.1190x**; time reduction **10.6342%**;
- if the accounting's `p_backward=0.82` transferred, the conditional teacher speedup ceiling would be
  **1.1780x** (`15.1113%` time reduction);
- the in-loop whole-step gain remains **UNKNOWN** until `_teacher_fwd/_teacher_bwd` timers run.

The final pair-0 costate SHA-256 for all three arms was
`74ed23a57e8a9514d7b4904a1eaf3b0fde5a2549a5ab1c9b9f686877c59254cc`; maximum absolute and relative-L2
differences from static one-thread were both zero. This is a one-pair training-gradient diagnostic, not
n600 or cross-build exactness authority.

**Scoped verdict: `WORTH-BUILD-PHASE-SPLIT-THREAD-CONTROL`.** This is the only technique in the arm with
a measured favorable backward sign. Build means a bounded standalone/in-loop profile and typed law—not
activation here. Admission must require:

1. existing D-A in-loop component timers on at least the faithful n24 linear-accumulation surface;
2. thread-switch overhead included at every pair;
3. n600 matched training-gradient/trajectory gate under the operator-accepted training-only authority;
4. exact forward remains at the canonical one-thread law; auth eval remains a separate untouched process;
5. fail-closed fallback to the one-thread training standard on any drift or non-positive in-loop gain.

## 7. Scoped decision table

| mechanism | storage effect | compute effect | DERIVED wall-clock sign | verdict |
|---|---|---|---|---|
| segment/full checkpointing | saves subset/up to ~628 MiB per pair | recomputes `c*T_f` | slower by +18%-46% of teacher for full coverage | `NO-GO-compute-bound` |
| continuous adjoint O(1) | O(1) only for a genuine ODE | backward solve/reconstruction | slower or different teacher | exact feasible `N` |
| reversible residual approximation | saves eligible residual activations | one or more `F` re-evaluations | slower; decoder untouched | bounded-error feasible `N` |
| forward-1/backward-6 phase split | unchanged | parallelizes VJP only | pair-0 diagnostic +10.63% teacher time | `WORTH-BUILD` bounded |

## 8. Canonical-equation harvest (clean law; registry mutation deferred)

**Candidate equation id:** `checkpoint_wallclock_admission_law_v1`
**Status:** `VERIFIED_VIA_SOURCE_INSPECTION` for the identity; local empirical anchor for the sign;
`FORMALIZATION_PENDING` because the canonical registry is a dirty shared surface and this arm is
new-files-only.

```math
T_{\rm ckpt}=T_f+T_b+cT_f-\Delta_{\rm mem},\qquad
G_{\rm ckpt}:=\frac{T_f+T_b}{T_{\rm ckpt}},
```

```math
G_{\rm ckpt}>1\iff\Delta_{\rm mem}>cT_f.
```

For a compute-/kernel-bound frozen input-VJP, `Delta_mem approximately 0`, hence `G_ckpt <= 1`, strictly
less than one for every nonzero checkpointed fraction. This is the general admission law: **memory
savings become a wall-clock lever only when the measured time removed by relieved memory pressure
exceeds the recomputed forward time.**

The sibling phase-thread law is

```math
T_{1\to k}=T_f(1)+T_{\rm switch}(1\to k)+T_b(k)+T_{\rm switch}(k\to1),
```

and is admitted only if `T_{1->k}<T_f(1)+T_b(1)` under matched in-loop measurement and the training-only
costate fidelity gate.

## 9. DAG FEED for main harvest

### FEED-p0-checkpoint-backward-20260713 — compute-bound NO-GO; phase-thread split survives

Exact instantiated `tu-efficientnet_b2` U-Net derivation gives 9.909 GMAC/pair forward, with 73.57% in
the dense U-Net decoder and only 2.19% in depthwise convolutions. Frozen input-only VJP has approximately
the same convolution FLOPs as forward and saves ~628 MiB/pair of nonparameter activations at most.
Therefore checkpointing/adjoint/reversible formulations trade extra convolution evaluations for storage
on a compute-/slow-kernel-bound path: checkpointing is `NO-GO-compute-bound` for wall clock (+18%-46%
teacher time for full recompute); exact adjoint-O(1) and bounded reversible approximation are infeasible
for the discrete stride/SE/U-Net graph. The accounting's 2472 ms / 82% backward is diagnostic-only and
does not reconcile with the exact graph or warmed matched pair (static-one backward share 53.49%); the
in-loop timer remains owed. A one-thread-forward/six-thread-backward phase split is the sole survivor:
pair-0 exact costate bytes matched and teacher fwd+bwd improved 718.055->641.696 ms = 1.1190x in a local
ABBA diagnostic. `verdict_scope=pair0 local Torch 2.12.1 training-gradient diagnostic; no n600/in-loop/
contest/score transfer`. Main next action: bounded D-A in-loop profile, then n600 training-only fidelity;
auth eval untouched. Pointer unmoved.

The shared DAG was not edited because this task explicitly permits new files only; this block is ready
for main to harvest after shared ownership clears.

## 10. Provenance and stores consulted

Local custody:

- git HEAD: `3187c44f3e6cbc7b96e99de75cba4b7a1e2e1bb5` (dirty shared worktree preserved);
- runtime: `macOS-26.4-arm64`, Torch `2.12.1`, fp32, CPU only;
- frozen SegNet weights SHA-256:
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
- `upstream/modules.py` SHA-256:
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
- `tools/profile_segnet_blocks.py` SHA-256:
  `133a83fd3f536071c5dcba1c0f0d0e465ba6474c3ba28456a9e6acb33506a7bb`;
- accounting memo SHA-256:
  `967d68a46827d5674cc3f4a723457930b860a46ecdae27cc85b561cb4195def2`;
- seed `0`; `PYTHONHASHSEED=0`; real receiver pair 0 decoded through canonical
  `frame_utils.yuv420_to_rgb`; exact frozen scorer; no synthetic timing fixture.

**STORES CONSULTED:** full `CLAUDE.md` authority map and task-relevant bodies; full supplied/local
`AGENTS.md` authority map and task-relevant bodies; full `docs/operating_manual_craft_handoff.md`;
`PROGRAM.md`; v7.5 SPEC §8; top Claude memory entries; relevant Codex throughput/costate memory index;
current main branch/worktree, lane/subagent ownership surfaces, and checkpoint tool;
`.omx/research/per_epoch_detailed_accounting_20260713.md`; task #455 and #456 memos/receipts;
operator one-thread training standard; `upstream/modules.py`; exact frozen weights; instantiated
SMP/timm module graph; `tools/profile_segnet_blocks.py`; exact real-pair local Torch profiler;
Calin/Neural-ODE/ANODE/RevNet/i-ResNet primary bibliographic sources. Paid providers, live runs,
MPS, CUDA, Metal, `upstream/evaluate.py`, and submission artifacts were not actuated.

## 11. Adversarial limits

- No hardware performance counters were collected. “Compute-bound” here includes slow convolution
  implementation/dispatch; it means **not activation-storage-traffic-bound**, which is the decision
  boundary checkpointing needs.
- The fresh measurements are one-pair local diagnostics. They decide mechanism sign, not n600
  trajectory fidelity or full-epoch savings.
- The 82% backward share is not promoted. The D-A in-loop decomposition remains owed and can change the
  magnitude of the phase-thread gain, not the checkpointing sign under the derived gate.
- Phase-split costate identity was measured on pair 0 only. It is a build candidate with fail-closed
  gates, not a trainer setting and not an auth-eval setting.
- No canonical equation or shared DAG registry was mutated. The exact harvest payloads above preserve
  the clean law without colliding with live shared state.
