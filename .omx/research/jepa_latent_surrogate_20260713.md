# JEPA lineage for a frozen-SegNet latent surrogate and costate VJP

**Date:** 2026-07-13  
**Role:** SOL xhigh deep-math design/adversarial read  
**Status:** `DESIGN / MEANS / research_only=true / UNCOMMITTED / NO TRAINING OR HEAVY LAUNCH`  
**Owner:** `jepa_latent_surrogate`  
**Pointer authority:** unchanged. The surrogate is a throughput controller, not the witness and not the frontier pointer.

## Lead verdict

> **{latent-target-better-than-logit/label: NO versus smooth logits as a blanket claim; YES versus hard labels or a directly regressed raw costate—the best default is the four-dimensional centered-logit decision quotient, while full penultimate features are an overcomplete comparator · costate-VJP-via-surrogate: YES conditionally—one small-student backward can replace the teacher forward+backward between exact anchors, but only joint value-and-Jacobian/VJP fidelity makes that statement true; derived diagnostic-slice cost is `C_S,VJP + (C_T+U)/K`, with `C_T=3009.070 ms` and an inclusive 95%-kill budget of `150.453 ms/step` · composes-with-#484: YES as a differentiable prefix/target arm, NO for the current detached NumPy capture · n=1-prior: WORTH only as initialization or a weak matched-data regularizer; NO-GO as a frozen generic invariant encoder or default LeJEPA SIGReg · overall: WORTH-AN-ARM-feeds-455/484}**

This is a real, but narrow, draw from the JEPA lineage. “Predict a decision-sufficient smooth representation rather than pixels or hard decisions” is useful. JEPA does **not** supply the missing theorem that matching a latent value matches its derivative with respect to RGB. That derivative is the prize here, so a JEPA-value loss without a Sobolev/VJP gate is philosophical alignment, not a SegNet-backward replacement.

## 0. Authority, accounting, and scope

### Evidence labels

- **MEASURED (local diagnostic harness, not in-loop authority):** exact SegNet forward `F_T = 537.045463 ms/pair`; exact SegNet forward+input-backward `C_T = 3009.069611 ms/pair`; sample counts `n=9` and `n=6`, respectively. Source: `.omx/research/onpolicy_surrogate_95kill_20260713.md` and `.omx/research/per_epoch_detailed_accounting_20260713.md`.
- **DERIVED from those same-run measurements:** `B_T=C_T-F_T=2472.024148 ms`, or `82.1524%` of the diagnostic teacher forward+backward. An ideal free forward gives only `C_T/B_T=1.21725x`; an ideal free backward with the teacher forward retained gives `C_T/F_T=5.60301x` on that slice.
- **CAVEAT / UNVERIFIED in-loop:** the diagnostic absolutes are about 12x heavier than the `169.7 s/epoch, n600` in-loop vehicle. Even the 82.15% ratio is a candidate decomposition until an in-loop component timer measures it.
- **MEASURED (prior n600 research signals, non-promotable):** direct/cached input-costate formulations had costate cosine about `0.0014–0.0017`; the deeper post-SE nonlinear support arm reached `29.46%` mass versus its preregistered `47%` bar. Those negatives scope the tested formulations, not all differentiable students.
- **DERIVED from source inspection:** the frozen segmentation scorer is `smp.Unet('tu-efficientnet_b2', classes=5, activation=None)`; its exact terminal head is a `3x3 Conv2d(16,5)` with 725 parameters. Segmentation uses the last frame, resizes to `384x512`, and evaluator authority is the final class argmax.
- **ASSUMED pending measurement:** a small RGB-to-decision-latent student can meet the VJP and wall-time gates. No such result is claimed here.

### Verdict scope

`TARGET-FORMULATION x DESIGN`: frozen five-class SegNet, last-frame per-pixel CE through the repository's actual R surface, a small differentiable student, and periodic exact-teacher anchors. Negatives below do **not** kill JEPA, representation distillation, full penultimate distillation as a comparator, temporal priors for the organ, or other scorer architectures. They reject only (i) the claim that a generic penultimate JEPA target is automatically better than centered logits and (ii) the claim that latent-value matching alone licenses an input-costate replacement.

No empirical score claim is made. Any future comparison must be real on-policy `n600`; NumPy-fp32 is the deterministic reference. Only a **MEASURED in-loop** component/wall-clock win moves throughput, and only a byte-closed exact evaluator row on the exact archive bytes can move score or the frontier pointer.

## 1. What the JEPA lineage actually transfers

### Source-grounded reading

1. **I-JEPA** predicts target-block representations from context representations and explicitly avoids reconstructing unpredictable pixel detail. It learns both context and target encoders at ImageNet scale; the target encoder is an EMA teacher. This supports the *target-selection heuristic*, not input-Jacobian fidelity for a fixed contest scorer. Primary: [I-JEPA paper](https://arxiv.org/abs/2301.08243), [official implementation](https://github.com/facebookresearch/ijepa).
2. **V-JEPA** carries feature prediction into video, again learning representations from large video corpora rather than reconstructing pixels. It is evidence that latent prediction can discard nuisance detail while retaining predictable structure, but its learned invariances are not known to preserve SegNet's color/boundary sensitivities. Primary: [official V-JEPA repository](https://github.com/facebookresearch/jepa).
3. **LeJEPA (2025)** replaces teacher/student asymmetry with a prediction loss plus SIGReg toward an isotropic Gaussian embedding distribution. Its risk result concerns learning broadly useful embeddings when the downstream task is unknown. Here the downstream map is known and frozen. Collapse prevention is not the central problem, and forcing isotropy can suppress task-real anisotropy. Primary: [LeJEPA paper](https://arxiv.org/html/2511.08544), [official implementation](https://github.com/galilai-group/lejepa).
4. **NextLat (2025)** is the closest structural analogy: a small dynamics model predicts final-layer pre-logit hidden states and a frozen output head contributes a KL term. Yet the representation and dynamics are jointly shaped for next-token belief states; its theorem is not a theorem about a frozen vision scorer's derivative with respect to RGB. Primary: [NextLat paper](https://arxiv.org/html/2511.05963).
5. **`lucidrains/x-jepa`** is a useful experimental index/implementation surface, not independent evidence that this transfer works for SegNet VJPs. Primary: [repository](https://github.com/lucidrains/x-jepa).

**Adversarial conclusion:** JEPA gives a real target-design prior—predict only a smooth sufficient statistic—but no admission law. The fixed frozen encoder, one-video support, and derivative objective remove the conditions under which the main JEPA results were established.

## 2. Question 1 — latent target versus logit/label/costate target

### 2.1 The clean target law: quotient out the softmax gauge

At one output pixel, let teacher logits be `u_T(x) in R^K`, with `K=5`, and

```text
P = I - (1/K) 11^T,              q_T(x) = P u_T(x).
```

`P` removes the common-class shift. For any scalar `a(x)`:

```text
argmax_c [u_c + a] = argmax_c u_c
softmax(u + a1)    = softmax(u)
CE(u + a1, y)      = CE(u, y).
```

Therefore, **DERIVED**:

```text
argmax u_T = argmax q_T,
softmax(u_T) = softmax(q_T),
CE(u_T,y) = CE(q_T,y).
```

The centered logits have only `K-1=4` independent class dimensions per pixel. They are a smooth, decision-sufficient latent for both the training CE and evaluator argmax. This is the exact analog of JEPA's “do not predict irrelevant detail,” but it is derived from the frozen scorer rather than imported as a generic representation claim.

If `z_T in R^(16 x H x W)` is the penultimate feature field and `H_T` is the frozen terminal `3x3 Conv2d(16,5)`, then

```text
q_T = P H_T z_T.
```

Only `P H_T z_T` can affect CE/argmax. The remaining directions of `z_T` lie in a large decision-null space for this head. Literal penultimate matching asks the student to emit 16 channels rather than four independent class coordinates and penalizes feature differences that cannot affect the target loss. The 725-parameter frozen head is structurally small, but its runtime is **UNMEASURED** here.

### 2.2 Adversarial target comparison

| Target | Smooth? | Decision sufficient? | Input-VJP guarantee from value fit? | Derived disposition |
|---|---:|---:|---:|---|
| Hard class labels / argmax | No | Yes for current decision only; margins lost | No | **NO-GO as sole VJP target.** Cheap but manufactures arbitrary off-boundary geometry. |
| Full logits `u_T` | Yes | Yes | No | Viable, but wastes one exact common-shift gauge dimension. |
| **Centered logits `q_T=P u_T` / four margins** | Yes | Yes for CE and argmax | No; must add VJP/Sobolev gate | **Recommended default arm.** Minimal smooth decision quotient. |
| Full penultimate `z_T` + fixed terminal head | Yes | Yes after the head | No | **Comparator arm, not default.** It may be easier for a convolutional student to factor spatial structure, but that is ASSUMED and must beat quotient logits at matched parameters/teacher calls. |
| Direct raw input costate `g_T` | A vector field, not a scalar latent | It is the desired local derivative | Direct matching measures it, but a free vector regressor need not be integrable | Tested simple formulations are scoped negative; retain only as a distillation term on a scalar-composed student. |

This gives the requested answer: **penultimate-latent is not DERIVED to be better than logits**. It is plausibly smoother than hard labels, but logits are already smooth and closer to the exact decision/loss. The stronger arm is JEPA-inspired *quotient latent distillation*, not “copy all penultimate features.”

### 2.3 Where the analogy breaks

- The SegNet teacher is frozen; we cannot jointly shape its representation to become predictable, isotropic, or sufficient.
- The support is one contest video/vehicle and a narrow on-policy witness tube, not an ImageNet/VideoMix-style corpus. “n=1” here means one video/trajectory family, not literally one training pixel.
- SegNet's evaluator-critical nuisance is not human-unpredictable pixel detail. Tiny chroma/boundary changes can be precisely the authority-bearing signal. JEPA-style invariance can erase it.
- The required object is `dL/dx`, not representation quality under a broad downstream-task distribution. None of the cited JEPA results proves preservation of this Jacobian.

**Question-1 verdict:** `WORTH` only after narrowing the target to centered logits or a matched penultimate comparator with the fixed head. `NO-GO-metaphor` for a generic “latent is smoother, therefore its RGB costate is faithful” inference.

## 3. Question 2 — the VJP prize and the #484 composition

### 3.1 One small backward is a real route, with a missing condition

Let a small differentiable student produce `q_S(x;psi)` and define

```text
L_S(x,y) = CE(q_S(x;psi), y),
r_S       = softmax(q_S) - e_y,
g_S       = dL_S/dx = J_S(x)^T r_S.
```

The exact teacher is `g_T=J_T^T r_T`. Add and subtract `J_S^T r_T`:

```text
g_S - g_T = (J_S-J_T)^T r_T + J_S^T (r_S-r_T),
```

so, **DERIVED**:

```text
||g_S-g_T||
 <= ||J_S-J_T||_op ||r_T|| + ||J_S||_op ||r_S-r_T||.
```

This is the crux. Matching centered logits drives the second term, but does not control the first. A small network can agree at sampled values while having the wrong RGB Jacobian. The arm therefore needs a scalar-composed student plus a teacher-anchor derivative term:

```text
L_fit = alpha * ||q_S-q_T||_Huber
      + beta  * E_v[(<g_S-g_T,v>)^2]
      + gamma * (1-cos(g_S,g_T)),
```

where `v` are fixed seeded Rademacher renderer/tangent probes. Since `E[vv^T]=I`,

```text
E_v[(<g_S-g_T,v>)^2] = ||g_S-g_T||_2^2.
```

The finite probe estimate is **INFERRED/APPROXIMATE** until measured; the expectation identity is **DERIVED**. Exact teacher costates are still paid at anchors. Unlike an unconstrained direct costate regressor, `g_S` is the gradient of the explicit scalar `L_S`; it is conservative/integrable by construction (away from ordinary network nondifferentiabilities).

### 3.2 Cost law and the 95%-kill feasibility boundary

Define per training step, on the deliberately narrow SegNet teacher slice:

```text
C_T       = exact frozen-SegNet forward + input backward,
C_S,VJP   = student forward + fixed-head/loss + student input backward,
U         = student update cost paid at one exact anchor,
K         = exact-anchor cadence,
C_bar(K)  = C_S,VJP + (C_T + U)/K,
S_T(K)    = C_T / C_bar(K).
```

Using the same-run diagnostic `C_T=3009.069611 ms`, an inclusive 95% teacher-slice kill requires

```text
C_bar(K) <= 0.05 C_T = 150.453481 ms.
```

At `K=20`, `C_T/K=150.453481 ms` before any student or update work. Therefore:

```text
K=20 + positive C_S,VJP or U  => inclusive 95%-kill is impossible.
```

For any chosen student cost, the **DERIVED** necessary cadence is

```text
K >= (C_T + U) / (0.05 C_T - C_S,VJP),
```

with the denominator required to be positive. This is diagnostic-slice algebra, not an in-loop speed claim. It also exposes why keeping the full teacher forward every step cannot attack the whole 82% path: even a free backward would leave the measured `537.045463 ms` forward floor.

### 3.3 Composition with #484 pre-SE locus

The composition is structurally valid only as:

```text
x --R--> differentiable prefix p_phi(x)
       --> small predictor a_psi(p_phi(x))
       --> q_S  OR  z_S --frozen 3x3 head--> q_S
       --> CE --> one backward through head + predictor + prefix --> g_S.
```

Its charged cost is

```text
C_S,484 = C_prefix,fwd + C_prefix,VJP
        + C_predictor,fwd + C_predictor,VJP
        + C_fixed_head/loss.
```

Source inspection of the live sibling's current implementation found two candidate captures at the inputs to EfficientNet-B2 SE blocks (`144x96x128` and `288x48x64`). They are pre-SE for their own blocks, but four/seven earlier SE reductions already precede them, so global end-to-end tileability is false. More importantly, the current path uses `detach().clone()` and a NumPy head. **DERIVED:** it cannot propagate `dL/dx` as written.

Thus `#484` composes **YES** as a future differentiable-prefix/student architecture or teacher target source, but **NO** as a drop-in VJP replacement. Its present cost fractions are operation proxies, not wall time; no speedup follows until both prefix forward and prefix VJP are timed in-loop. This memo does not edit or relabel the running sibling arm.

**Question-2 verdict:** a scalar latent student is a real path to the 82% backward because it replaces the entire teacher computation between anchors, not just teacher forward. Admission is conditional on Jacobian/VJP fidelity and total charged in-loop cost. The current #484 capture is not that path yet.

## 4. Question 3 — can a JEPA prior cure one-video starvation?

### What is worth an arm

A self-supervised driving-video encoder can provide initialization for road layout, egomotion, object persistence, and multiscale spatial features. That may reduce the number of exact teacher anchors needed. This is **INFERRED from representation-transfer practice**, not measured for this scorer. A permissible arm is:

1. initialize only the student trunk from an open driving-video JEPA prior;
2. replace/adapt the output with the five-class decision quotient;
3. fine-tune all layers on real on-policy states from `0.mkv`;
4. match scratch and prior arms on parameter count, exact-teacher calls, update steps, and `n600` held-out trajectory windows;
5. let exact costate/trajectory gates override the prior.

### What is overfit-hostile or irrelevant

- Freezing the generic prior is `NO-GO`: generic invariances may delete SegNet-specific chroma, resize, and boundary sensitivity.
- Default LeJEPA SIGReg is `NO-GO` for this arm: the task is known, teacher geometry is anisotropic, and representation collapse is not the problem. A weak teacher-covariance/whitening penalty can be a separately preregistered comparator, never an assumed default.
- Next-latent temporal prediction is not required for direct SegNet VJP: the scorer consumes the last frame and is memoryless. It can feed the costate organ's trajectory model (#428/#431), where dynamics matter, but that is a different consumer and verdict scope.
- Corpus pretraining does not cure missing on-policy support by itself. It can lower sample complexity, but only refreshed real-teacher anchors control extrapolation off the current witness tube.

The organ evidence already says one-trajectory data are plateau-heavy and simple learned temporal models lose to persistence. Therefore the prior is `FEED-organ-428` only as an initialization/feature arm. Synthetic trajectory diversity under #434 remains separate from synthetic video pretraining, and real walk-forward adoption remains binding.

**Question-3 verdict:** `WORTH` as a weak initialization prior; `NO-GO` as an authority, frozen representation, or substitute for real on-policy `n600` gates.

## 5. Preregistered arm for main review (no launch here)

### Arm Q — recommended

- Target: `q_T = P u_T` at the actual scorer input/R surface.
- Student: smallest differentiable RGB-to-four-margin convolutional model that preserves exact spatial output geometry.
- Losses at exact anchors: centered-logit Huber + full input-costate cosine/relative-L2 + fixed seeded directional-VJP probes.
- Runtime: compute `CE(q_S,y)` and one input backward through the student between anchors.
- Exact fallback: fail closed to teacher refresh on trust-region or trajectory-gate violation.

### Arm Z — JEPA-literal comparator

- Target: teacher penultimate `z_T`, composed with the exact frozen `3x3 Conv2d(16,5)` head and then centered.
- Match Arm Q in student parameters, teacher calls, states, and wall-time accounting.
- Win condition: lower charged `C_S,VJP` **and** no worse exact VJP/trajectory fidelity. Feature MSE alone is not a win.

### Arm P — prior comparator

- Arm Q initialized from a driving-video JEPA trunk versus scratch, all layers fine-tuned.
- No default SIGReg; any regularizer gets a separately typed DSL mode.

### Mandatory measurement gates

1. **In-loop decomposition first:** measure teacher forward, teacher input backward, renderer VJP, student forward, student VJP, anchor update, and whole matched window on the same live-equivalent path. The diagnostic 82.15% ratio cannot authorize architecture selection alone.
2. **Real `n600`, early/boundary/late:** no small-subset empirical verdict.
3. **Value:** centered-logit error, per-class margin error, and exact argmax disagreement through R.
4. **Derivative:** input-costate cosine and relative-L2, fixed directional-VJP residuals, renderer-gradient cosine/dot product, and signed exact-CE descent.
5. **Trajectory:** exact CE, `d_seg`, `d_pose`, and common-controller endpoint parity; preserve holistic class-anchor/island/pose/rate facets.
6. **Economics:** use `C_bar(K)` with every charged term; `K=20` cannot be called a 95% inclusive kill.
7. **Authority:** NumPy-fp32 reference and framework parity first; MPS/MLX remain advisory. Only exact byte-closed contest evaluation can move the pointer.
8. **Launch contract if later authorized:** typed DSL only, deterministic seed, storage waterfall, resumable disk state, atomic periodic and per-stage checkpoints, all stage checkpoints preserved, and success-only certified cleanup.

## 6. Triality and system wire-in

### Equation leg — landed as a new, isolated proposal

`src/tac/canonical_equations/segnet_decision_quotient_surrogate_20260713.py` provides:

- deterministic NumPy-fp32 `centered_logits_numpy`;
- the input-costate error bound;
- the amortized teacher-slice cost law;
- `CanonicalEquation` definition `segnet_decision_quotient_surrogate_v1`.

The shared `.omx/state/canonical_equations_registry.jsonl` was already dirty before this lane. Its append is therefore `DEFERRED_MAIN_COLLISION`, not silently performed. Main can invoke the module's explicit `populate_*` surface after ownership review.

### DSL leg — proposed, default-off, not edited

Extend the existing scorer-surrogate typed DSL with an enum-like target mode, never an invented launch flag:

```text
target_mode in {
  input_costate_direct_legacy,
  centered_logit_sobolev,
  penultimate_fixed_head_sobolev
}
```

`centered_logit_sobolev` is the proposed default for a new research arm, with `research_only=true`, no live argv, exact-anchor fallback, and separate `prior_mode in {scratch, driving_jepa_init}`. No DSL file was edited in this pass.

### DAG leg — FEED text, append deferred to main

```text
FEED-455-484-JEPA-QUOTIENT (DESIGN; research_only=true)
  SOURCE:
    per_epoch_detailed_accounting_20260713
    onpolicy_surrogate_95kill_20260713
    pre_se_locus_20260713 [running sibling; source-inspected only]
    I-JEPA / V-JEPA / LeJEPA / NextLat primary sources
  DERIVATION:
    frozen SegNet logits -> quotient P=I-11^T/5 -> q_T=P u_T
    q_S + CE -> g_S=J_qS^T(softmax(q_S)-e_y)
    admission requires value term + Jacobian/VJP term
  CONSUMERS:
    #455 on-policy surrogate target
    #484 differentiable-prefix comparator
    #428 organ initialization prior [separate temporal verdict scope]
  BLOCKERS:
    in-loop forward/backward decomposition UNMEASURED
    n600 quotient-student VJP fidelity UNMEASURED
    #484 current detached NumPy path is not input-differentiable
    K=20 inclusive 95%-kill infeasible with positive student/update cost
  ADMISSION:
    n600 early/boundary/late exact trajectory parity
    positive fully charged in-loop wall-time gain
    byte-closed exact row for any score/pointer claim
  VERDICT_SCOPE:
    WORTH-AN-ARM for target formulation; no family promotion; no score authority
```

### Six-hook wire-in disposition

- Sensitivity map: projected VJP residual by class/pixel/renderer tangent is the contribution.
- Pareto constraint: admit only if Seg/Pose trajectory debt is non-worsening at lower charged wall time; bytes unchanged by the training controller.
- Bit allocator: no direct runtime payload; consumer remains unchanged until a trained witness produces an exact byte-closed row.
- Cathedral/autopilot: default-off arm gated by exact-anchor fallback and component timer.
- Continual learning: any n600 arm result must append value/VJP/wall-time posterior evidence, including negative scope.
- Probe-disambiguator: Arm Q versus Arm Z is the required arbitration for centered-logit versus penultimate target.

## 7. Final scoped disposition

`WORTH-AN-ARM-feeds-455/484`.

The actual draw is not “JEPA will learn SegNet.” It is: **distill the smallest smooth statistic that the frozen decision/loss reads, then make its derivative an explicit target.** For this scorer that statistic is the centered-logit quotient, not automatically the full penultimate representation. A small scalar-composed student can, in principle, replace the expensive teacher forward+backward between anchors, so this is a real route to the candidate 82% backward. It remains `ASSUMED` until in-loop timing and real `n600` VJP/trajectory measurements clear the gates.

Pointer delta: **none**. This pass changed no live run, trainer, witness controller, #455/#484 artifact, score, archive, or frontier pointer.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`; `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/master_gradient_anchors.jsonl`; `.omx/state/modal_call_id_ledger.jsonl`; `.omx/state/cost_band_posterior.jsonl`; `.omx/state/continual_learning_posterior.jsonl`; `.omx/state/canonical_task_status.jsonl`.
- Latest local Codex findings/session, council/design/directive surfaces available at session start; relevant DAG sections for #428/#431/#434/#455/#484.
- `.omx/research/per_epoch_detailed_accounting_20260713.md`; `.omx/research/onpolicy_surrogate_95kill_20260713.md`; frozen-replay round 2/3/4/5 memos; `.omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md`; `.omx/research/costate_organ_capabilities_limits_envelope_20260711.md`; `.omx/research/synthetic_data_nvidia_sota_organ_434_20260711.md`.
- `upstream/modules.py` and the exact local SegNet/loss call sites; read-only current #484 source/policy/probe surfaces.
- Primary literature/repositories linked in Section 1. No claim relies on `x-jepa` as scientific authority.
