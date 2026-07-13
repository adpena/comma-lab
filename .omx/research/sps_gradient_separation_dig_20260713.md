---
title: "SPS gradient-role separation dig for the V9 CGauge witness"
date: 2026-07-13
lane_id: lane_sps_gradient_separation_probe_20260713
authority: "[macOS-CPU local probe; NON-PROMOTABLE]"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: FORMULATION-INSTANCE-PROBE
---

# SPS gradient-role separation dig — the gradient math transfers; the architecture does not

## ANSWER FIRST

**CONFLICT VERDICT — NO-GO at this checkpoint/probe scope.** The actual epoch-275 program has no
temporal gradient to conflict with: temporal screw starts at epoch 450 and phase advection at epoch
726, so the live temporal norm is **MEASURED 0** and its cosine is undefined. In the explicitly
counterfactual fully-armed probe at the same real checkpoint, the aggregate trunk gradients are
**aligned**, not anti-aligned:

- `100*d_seg` gradient versus `0.4*phase + 0.1*screw`: cosine **+0.107189**; **0.000%** of trunk
  parameter weight lies in negative-cosine tensors.
- `d_pose` gradient versus temporal: cosine **−0.00000693**, numerically near-orthogonal; **41.58%**
  of tensor weight lies in negative-cosine tensors, but the global direction has no material
  anti-alignment.
- fully armed `100*d_seg + d_pose` versus temporal: cosine **+0.099696**; **0.000%** of trunk
  parameter weight lies in negative-cosine tensors.

One of four pairs is materially anti-aligned (`pair=225`, seg-versus-temporal cosine **−0.278842**,
negative-cosine tensor-weight fraction **1.0**); the other three are aligned
`[+0.104326,+0.209701,+0.119975]`. This is **heterogeneous per-pair interference that cancels in the
aggregate**, not the global two-role conflict needed to justify duplicating streams. A full n600
probe at an engaged checkpoint is owed before any family verdict.

**SEPARATION DESIGN VERDICT — NO-GO to build.** V9 has no token sequence, no KV state, and no
activation that must simultaneously predict now and carry information forward. Its persistent
object is the shared parameter vector. Once SPS is translated into this weight space, the proposal
is ordinary multi-task gradient routing/gradient surgery plus optional parameter adapters. The
measured aggregate does not currently need that treatment. The candidate is recorded as
`reformulation-queue`, not `needs-build` or `measured`.

**STEPS ECONOMICS — conditionally favorable, not measured here.** If and only if separation keeps
the frozen-teacher call/VJP count unchanged, the teacher fraction really is `f_T=0.95`, and it
doubles only the other 5%, then its per-step factor is
`0.95 + 2*0.05 = 1.05`; break-even requires a strict step-count speedup **`r > 1.05x`**, equivalent
to more than **4.7619%** fewer steps. The current component-profile receipt explicitly says the
95/5 decomposition is **BLOCKED_NOT_MEASURED**, so `f_T=0.95` is an **ASSUMED prompt anchor**, not a
MEASURED Pact fact. PCGrad-like separate VJPs do not satisfy the unchanged-teacher premise and can
move the break-even toward `1.95–2.00x`.

**Pointer delta:** ZERO. No launch, no trainer edit, no run-directory mutation, no score claim.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` and
  `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `.omx/research/cgauge_master_action_and_parametrization_20260711.md`;
  `.omx/research/covariance_totality_texture_trunk_verdict_20260710.md`;
  `.omx/research/dpose_covariance_mirror_audit_20260711.md`;
  `.omx/research/t1_phase_advection_seal_439_20260711.md`.
- `.omx/research/steps_dimension_95kill_20260713{,_SPEC}.md`;
  `.omx/research/frozen_replay_convex_head_95kill_20260713.md`;
  `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json`.
- Read-only run surfaces under
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/`.
- SPS primary sources: [paper](https://arxiv.org/abs/2607.01218),
  [project](https://lil-lab.github.io/sps/), and [official code](https://github.com/lil-lab/sps).
- Weight-space comparators: [PCGrad](https://arxiv.org/abs/2001.06782),
  [MGDA](https://arxiv.org/abs/1810.04650), [CAGrad](https://arxiv.org/abs/2110.14048),
  [Nash-MTL](https://arxiv.org/abs/2202.01017), and
  [In Defense of the Unitary Scalarization](https://arxiv.org/abs/2201.04122).

## 1. What SPS actually establishes

### 1.1 The transferable equation

For an autoregressive sequence with position-specific hidden state `h_i`, SPS decomposes the
parameter gradient by the uses of the state at position `i`:

```text
                 1       T-1                T-1       T
grad_theta L = ----- [ sum grad_{theta_i} l_i  +  sum sum grad_{theta_i} l_j ]
                T-1      i=1                 i=1     j=i+1
                         prediction role        future-state role
```

The first term asks the current state to predict the next token. The second asks that same state to
remain useful to all future predictions. SPS changes the causal computation graph: persistent input
slots accumulate future-state gradients, while ephemeral predict slots have a local window (`w=64`)
and receive loss only at predict positions.

This is not merely a loss reweighting in the paper. It changes which **activation** survives and
which downstream losses can reach it. The shared Transformer parameters still generate both kinds
of slot, but the hidden-state roles are structurally separated.

### 1.2 Source-truth corrections and caveats

- The 1.678B table reports standard NLL `2.458` and SPS NLL `2.390`: the change is **−0.068 NLL**
  (an improvement), not `+0.068` if the sign is read as a raw delta.
- The headline token efficiency is **2.6x** at the cited scaling comparison, but it is evidence for
  the SPS language-model architecture. It is not a prior on V9 optimizer steps.
- SPS's gradient analysis measures the **magnitude balance** of present versus future components. It
  does not report the negative cosine test requested here. A large future gradient is not, by
  itself, evidence of conflict.
- The paper's principal large runs are mostly single-seed. Its result is strong evidence for its own
  causal architecture, not a universal law that any two gradient roles should be split.

### 1.3 Why the architecture does not transfer

The V9 witness is a coordinate INR:

```text
coordinate basis -> shared HOSC trunk -> FiLM(pair code) -> SDF/texture heads -> R -> scorers
```

It has no autoregressive token axis, no KV cache, and no activation whose continued presence carries
scene state to later optimization steps. SGD/Adam state persists, but optimizer state is not the
same mathematical object as a causal hidden state. Trunk weights are reused across pairs and epochs;
they are shared parameters in a multi-objective optimization problem.

Therefore the only honest transfer is:

```text
Does one shared parameter block receive materially anti-aligned gradients from two named losses?
```

If yes, the relevant literature is multi-task optimization. If no, SPS has no witness-native bite.

## 2. Question 1 — does the conflict exist here?

### 2.1 Probe contract

Tool: `tools/probe_sps_gradient_role_conflict.py`  
Receipt: `experiments/results/sps_gradient_separation_probe_20260713/receipt.json`

The probe:

1. memory-maps the real n600 `ZIP_STORED` GT cache without inflating/copying its 5 GB payload;
2. reconstructs the exact active d_seg-aware feature taper (MEASURED min/max/mean
   `0.953616/1.046156/1.0`, matching the run log `0.9536/1.0462/1.0`);
3. loads the newest deploy EMA, epoch 275, SHA-256
   `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0`, read-only;
4. runs the repository Torch CPU twin with real frozen SegNet and PoseNet;
5. validates the loaded witness against the portable NumPy deploy forward: argmax equal,
   `cos(phi)=1.000000138`, max `|delta phi|=4.58e-5`, max `|delta RGB|=6.87e-5`;
6. validates the gradient-preserving YUV6 PoseNet preprocessing twin against the official
   evaluation-only (`@torch.no_grad`) helper: max pose-output delta `3.81e-6`;
7. computes gradients only for the 45,024 scalar core-trunk parameters
   `in_proj.* + hidden.{0..3}.*`, excluding code, FiLM, heads, palette, and pose carrier.

The direct MLX attempt failed closed because the headless sandbox exposes no Metal device. This is
therefore a deterministic **Torch/NumPy parity probe**, not an MLX-training-authority result and not
a contest score.

Pairs `[75,225,375,525]` are deterministic interior representatives of four strata. The conflict
rule is explicitly practical rather than metaphysical: aggregate cosine `<= -0.05` and negative
elementwise product on at least 10% of trunk scalars. Both thresholds are **ASSUMED probe policy**;
the continuous statistics remain in the receipt so no conclusion depends on hiding the cutoff.

### 2.2 Live program versus fully armed mechanism

At checkpoint epoch 275:

```text
w_screw_live = 0 because epoch 275 < 450
w_phase_live = 0 because epoch 275 < 726
w_pose_live  = 0 because epoch 275 < 726
```

Hence the actual temporal gradient is exactly zero. Calling it aligned or conflicting would be
fake; the live cosine is **undefined**.

To test mechanism geometry rather than schedule state, the counterfactual probe evaluates the
already-typed eventual raw objective at the frozen checkpoint:

```text
L_pair = 100 L_seg + 1 L_pose
L_temp = 0.4 L_phase-advection + 0.1 L_temporal-screw
```

No optimizer step is taken and no checkpoint is written.

### 2.3 Aggregate measurement

| trunk gradient pair | global cosine | negative-cosine tensor weight | elementwise negative-product weight | reading |
|---|---:|---:|---:|---|
| `100 L_seg` vs `L_temp` | **+0.107189** | **0.0000** | 0.454158 | aligned |
| `L_pose` vs `L_temp` | **−0.00000693** | 0.415778 | 0.514192 | near-orthogonal, layer-heterogeneous |
| `100 L_seg + L_pose` vs `L_temp` | **+0.099696** | **0.0000** | 0.457667 | aligned |

The elementwise sign statistic is not the PCGrad conflict criterion: coordinate bases can rotate it,
and almost half of scalar products may be negative while the vector dot product remains positive.
It is reported only because the task requested a material weight fraction. The stronger
parameter-block fact is that **every trunk tensor** has positive cosine for both the seg-only and
fully-armed aggregate comparisons, so the negative-cosine tensor-weight fraction is zero.

Gradient norms expose a second fact:

```text
||g_seg||  = 464.2782
||g_pose|| = 185.3867
||g_temp|| =   2.1153
```

The temporal direction is not only non-conflicting in aggregate; it is much smaller at this early
checkpoint. That is an instance measurement, not an argument for deleting the term at its later
engagement stage.

### 2.4 Per-pair heterogeneity

`100 L_seg` versus temporal cosines are:

```text
pair  75  +0.104326
pair 225  -0.278842   material conflict; all trunk tensors negative
pair 375  +0.209701
pair 525  +0.119975
```

This one conflicting stratum is a legitimate reformulation signal. It does **not** justify an
architecture-wide split because the optimizer consumes an aggregate batch/epoch objective and that
aggregate is aligned. It does justify preserving the following queue:

1. remeasure n600 at/after temporal engagement;
2. report the cosine distribution, not only its mean;
3. if anti-aligned strata persist, compare stratified batching/scalarization with PCGrad before any
   extra parameter stream.

### 2.5 Verdict and scope

**NO-GO_FOR_SPS_SEPARATION_ON_THIS_PROBE** at **FORMULATION-INSTANCE-PROBE** scope.

This is not a family kill. Reactivation requires either:

- n600 at a real phase/screw-engaged checkpoint with aggregate cosine `<= -0.05`; or
- a repeatable negative-cosine subpopulation whose isolation improves the canonical scalar action
  under a matched, equal-step local control.

No canonical conflict law closes: one checkpoint, four pairs, temporal terms counterfactually armed,
and direct MLX unavailable. A canonical-equation file would overstate authority, so none is minted.

## 3. Question 2 — the separation design, and why it stays on paper

### 3.1 The least-wrong witness-native split if future evidence reverses

If an engaged n600 measurement establishes real conflict, the minimal decomposition is:

```text
state block theta_s:
  in_proj + hidden trunk; covariant geometry; xi/phase transport and persistent gauge stores

prediction block theta_p:
  pair code z_p + its FiLM adapter; small pair-local output adapter/head if proven necessary

L_pair = 100 L_seg + L_pose
L_temp = L_phase + L_screw + L_xi-transport
```

Loss routing should then be defined as a typed dependency mask:

```text
grad_{theta_p} L = grad_{theta_p} L_pair
grad_{theta_s} L = grad_{theta_s} L_temp + A(g_pair^s, g_temp^s)
```

where `A` is first the ordinary scalar sum control. Only if the measured shared-block dot product is
negative should `A` be tested as a surgery operator. The simplest PCGrad arm is

```text
if <g_pair, g_temp> < 0:
    g_pair <- g_pair - <g_pair,g_temp> / ||g_temp||^2 * g_temp
```

with the symmetric/random-order variant preregistered, because asymmetric projection silently
declares one task sovereign. CAGrad/MGDA/Nash-MTL are comparators, not free improvements. A plain
unitary scalarization remains the control because the contest already supplies one canonical scalar
action.

The existing FiLM **code** is only analogous to a predict slot in one weak sense: it is pair-local.
It is not ephemeral. It persists in the archive, affects both frames, and is optimized over the
whole run. Calling it an SPS slot would import false causal semantics. A new pair-local head would
also add counted bytes and partially reverse the sealed single-trunk architecture, so it needs a
measured score-unit-per-byte case before admission.

### 3.2 L87 `d_seg = d_cov + d_gauge` is not a routing mask

The attractive proposal is:

```text
covariant terms -> state stream
gauge/pair terms -> prediction stream
```

That proposal is **REFUTED-AS-A-PRINCIPLE** by the meaning of the existing law:

- `d_cov + d_gauge` decomposes realized distortion debt, not necessarily the differentiable training
  loss or its parameter support.
- phase is a **gauge zero-mode**, yet phase advection is explicitly temporal and must persist across
  pairs. Gauge therefore cannot mean ephemeral prediction.
- pair-local SegNet CE sees both covariant geometry and gauge survival errors. It cannot be routed
  wholly away from the state trunk without starving the trunk of its primary supervision.
- movable reaction events are pair-local but state-relevant; `xi` transport is temporal but may be
  implemented by a pair-indexed carrier.

So L87 supplies a taxonomy for *what debt exists*, not a stop-gradient theorem for *where each loss
must backpropagate*. Any future routing mask must be derived from the actual Jacobian support and
measured gradient geometry.

### 3.3 Why weight-space SPS reduces to known gradient surgery

There is no distinct “future-state activation” to isolate. Both losses differentiate shared weights.
The available interventions are exactly the known multi-task ones:

- reweight the scalar losses;
- project or combine shared-parameter gradients;
- assign losses to disjoint parameter subsets/adapters;
- change sampling so conflicting strata are not destructively averaged.

This is PCGrad/MGDA/CAGrad/Nash-MTL territory. It should be named
`gradient_role_routing`, not advertised with SPS's language-model token-efficiency number. The one
SPS contribution that survives is the diagnostic question: *did we force one object to serve two
gradient roles?* The answer here is “not materially in the aggregate.”

### 3.4 DSL/candidate design — held, not built

Canonical pool row:

```text
candidate: sps_weight_space_gradient_role_separation
status: reformulation-queue
form_class: optimizer-stage
dsl_lever: N/A — design-only; trainer edit explicitly forbidden in this task
gate: n600 engaged-checkpoint conflict, then scalarization/PCGrad control before adapter split
```

If reactivated, the future typed DSL design must contain:

- mode: `off | telemetry | pcgrad | split_adapter`;
- named parameter groups and named loss groups, with unknown names refused;
- cosine threshold and material-fraction policy as provenance-carrying values;
- stage-boundary-only engagement; no per-step loss-weight mutation;
- telemetry for raw cosines, projected norms, task loss, score components, and extra scorer VJPs;
- optimizer/controller state in atomic resume checkpoints and every stage checkpoint;
- OFF-path byte identity and a deterministic NumPy/Torch reference for the projection algebra;
- a fail-closed runtime-cost guard that distinguishes same-teacher-call routing from extra VJPs.

No trainer source is edited here.

## 4. Question 3 — STEPS-dimension economics

Let:

```text
C_0 = C_T + C_W
f_T = C_T / C_0
k_T = separated teacher-cost multiplier
k_W = separated witness-cost multiplier
r   = N_steps,baseline / N_steps,separated
```

Then the exact compositional wall law is:

```text
C_sep / C_0 = k_T f_T + k_W (1-f_T)
Wall_sep / Wall_0 = [k_T f_T + k_W (1-f_T)] / r
break-even iff r > k_T f_T + k_W (1-f_T)
```

### 4.1 The requested optimistic composition

Under all three assumptions:

```text
f_T = 0.95       ASSUMED prompt anchor
k_T = 1          same scorer forward and same scorer VJP
k_W = 2          witness-side forward/backward doubled
```

the factor is:

```text
1*0.95 + 2*0.05 = 1.05
```

Thus any **strict** step efficiency greater than `1.05x` wins wall time; equivalently the treatment
must use fewer than `1/1.05 = 95.2381%` as many steps, a reduction greater than **4.7619%**.

This is a useful conditional claim. It is not yet a Pact measurement. The canonical current-wall
receipt says the full component split is `INCOMPLETE_BLOCKED_NOT_COMPOSABLE`, with MLX scorer
forward, backward, R, render, and loss-term costs unmeasured. It explicitly refuses to substitute
stale 78/22 or stripped microbench rows.

### 4.2 Why PCGrad may forfeit the 1.05 threshold

PCGrad needs the two gradients separately. If implementation requires a second backward/VJP through
the frozen scorer, the overhead lands on the supposed 95%, not merely the 5% witness:

```text
k_T=2, k_W=1  -> factor 1.95
k_T=2, k_W=2  -> factor 2.00
```

Retaining a shared forward graph can avoid a second forward but not automatically a second scorer
VJP. A genuinely disjoint architecture can route a single summed loss in one backward, but then its
extra parameter stream adds bytes and is not “gradient surgery.” These cases must not share one
economics label.

### 4.3 SPS's 2.6x does not compose as a prior

If one **ASSUMED** the language-model `r=2.6` step/token efficiency and the optimistic Pact factor
`1.05`, the algebra would yield `1.05/2.6 = 0.404` baseline wall. That number is **SPECULATIVE and
inadmissible**: tokens are not optimizer steps, the causal architecture is absent, and this probe
measures no convergence acceleration. It is intentionally not entered in the candidate pool as an
estimated `delta_S`.

## 5. Triality and apparatus disposition

### Equations

- Candidate conditional wall law is written in §4.
- **Canonical equation: N/A-with-rationale.** The conflict law does not close (n4, one checkpoint,
  counterfactual temporal engagement, Torch parity rather than MLX). The `f_T=0.95` component premise
  is also unmeasured in the current canonical receipt. Registering either would violate NO-FAKE.

### DSL

- No trainer edit.
- Canonical curriculum-pool row recorded via
  `tac.witness_dsl.curriculum_candidate_pool.record_candidate` with validated status
  `reformulation-queue`.
- Future factory contract is specified in §3.4 only; it is not claimed built.

### DAG

- `FEED-SPS-GRADIENT-ROLE-SEPARATION-20260713` is in
  `.omx/research/sub015_DAG_sps_gradient_separation_20260713.md`.

### Durable evidence

- Probe tool SHA-256: `0c41324ca965ba3b16a603ea1cf09b8434c9d556f95765cbf7e2253f8ecb1736`.
- Receipt SHA-256: `f493088357c9ef0147e407088e7f614699f9c2fa1c28207577750c690ca5c91e`.
- Source checkpoint SHA-256:
  `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0`.
- Run directory mutated: **false**.

## 6. Verdict scope, reformulation queue, and own-round review

**verdict_scope:** `FORMULATION-INSTANCE-PROBE` — SPS-like architectural stream duplication for the
epoch-275 V9 trunk under the measured four-pair gradient geometry.

**Not killed:** temporal consistency, phase advection, gradient-conflict methods as a family, an
engaged-checkpoint measurement, or per-pair stratification.

**Reactivation:** n600 at an engaged checkpoint; aggregate material anti-alignment or stable
subpopulation anti-alignment; compare scalarization/stratified batching/PCGrad before architecture.

**Own-round adversarial checks:**

1. **Did the probe invent a conflict by turning future losses on early?** No. The receipt presents the
   live gradient first (zero/undefined cosine) and names the future-weight result counterfactual.
2. **Did the positive aggregate hide pair conflict?** The full per-pair list is reported; pair 225 is
   explicitly anti-aligned and routed to reformulation.
3. **Does “45% negative scalar products” contradict positive cosine?** No. The dot product is the
   signed magnitude-weighted sum. Coordinate sign fraction is not rotation-invariant and is not the
   PCGrad test; zero negative-cosine tensor weight is the more useful material-fraction result.
4. **Did Torch replace the requested MLX authority silently?** No. MLX failed closed for absent Metal;
   forward and pose-preprocess parity are measured; authority remains local/non-promotable.
5. **Is the 95/5 split being laundered as measured?** No. It is labeled ASSUMED, and the current
   component receipt's blocker is quoted. The exact general equation is supplied so later measured
   fractions can replace it.
6. **Does L87 prove a loss mask?** No. Its gauge channel includes persistent temporal phase, which is
   a direct counterexample to “gauge means ephemeral prediction.”
7. **Would a PCGrad build be cheap?** Not necessarily. Separate scorer VJPs may make its break-even
   `1.95–2.00x`, which is why the pool row remains reformulation-only.

**Final status:** durable NO-GO instance finding; pointer unchanged; no launch; uncommitted for main
review.
