# Per-class convergence A/B implementation specification

**Status:** FROZEN BUILD SPEC; local build/preflight only; no Metal training launch.

**Pointer:** submittable `0.1910828242` `[contest-CPU]`; defensive non-submission bank
`0.1880443980`. Both are **UNMOVED**. A typed config, dry-start, or advisory trajectory is
MEANS, never a score or promotion result.

**Lane:** `perclass_convergence_ab` (L0 at specification time).

## 1. Question and verdict scope

Measure whether a non-arithmetic loss, step-native basis, or additive-multiplicative optimizer
equalizes the slow rare-class convergence (Lane and Movable) against common classes on the exact
real-n600 cohort. The primary comparison is Arm A vs Arm B. Arms C and D are sibling mechanism
probes and must emit the same trajectory core.

The existing analyzer `tools/probe_ordinal_perclass_convergence.py` is immutable in this landing.
Its `ordinal_perclass_convergence_trajectory.v1` core is the contract. A negative is
**INSTANCE-scoped** to the exact arm, 150-epoch horizon, vehicle, and optimizer/basis values. It is
not a FORMULATION/FAMILY/PARADIGM kill.

The causal limit is explicit: a label-smooth witness has the DERIVED real-n600 temporal-spike floor
`d_floor = 0.005318`; thin-Lane MCF erasure is a geometry mechanism that these arms may not cure.
The harness reports actual spike/non-spike error support and `max(d_seg - d_floor, 0)`, but never
renames Lane-hard error as a proved MCF fraction.

## 2. Matched arms

All arms use real `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, `num_pairs=600`, the same
seed, fresh initialization, pair-order generator, EMA, representation substrate, single CE-stage
schedule, 150 epochs, evaluation/checkpoint cadence, and non-treatment objective values.

* **A / CE control:** `seg_loss=ce`. The emitted argv and receipt treatment MUST NOT contain
  `margin_target_end`.
* **B / zero-margin hinge:** existing trainer form `seg_loss=margin_hinge` with the exact
  `margin_target_end=0.0`. Add a narrowly named typed DSL lever
  `ZeroMarginWinnerRivalHinge`; do not alter the settled additive `MarginBandSatisficing` controller.
* **C / step-native:** Arm A plus the existing `StepNativeActivation` with
  `basis=step_basis`, `beta 1.0 -> 8.0`, linear anneal, `omega=1.0`, and FINER bias enabled.
  This is the existing trainer-safe hosc-to-partition-indicator route, not an invented
  `--activation step_basis` token.
* **D / M+Adam:** Arm A plus a typed `MPlusAdam` optimizer lever. Implement paper Algorithm 1
  exactly in MLX and a deterministic NumPy-fp32 reference. The treatment is exploratory transfer
  from low-precision LLM training to fp32 single-video INR; no convergence claim is implied.

Arm A and B custody MUST match every analyzer-required field, including identical initial EMA
bytes and optimizer fingerprint. C may differ in model/init custody and D in optimizer custody;
they still emit the same trajectory schema for direct `derive_rates` use, but the unchanged
CE-vs-margin CLI must not be tricked into accepting them as Arm B.

## 3. Bounded horizon

`EPOCHS=150`, `eval_every=25`, `ckpt_every=25`, preserved stage/final checkpoints ON.

This is DERIVED from the measured #205 early CE slopes: ep100->125 relative slope
`-1.14e-3`, ep125->150 `-8.22e-4`, with d_seg still descending at ep150. Thus 150 epochs captures
the early-rate slowdown/divergence and includes the complete 100-epoch step-native anneal without
spending a 3000/50k pointer-run budget. This is a rate probe, not a convergence claim.

## 4. M+Adam law and optimizer contract

Per element and accepted optimizer update `t`:

```
u_t = beta1*u_{t-1} + (1-beta1)*g_t
v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
u_hat = u_t/(1-beta1^t)
v_hat = v_t/(1-beta2^t)
u_add = -eta_a*u_hat/(sqrt(v_hat)+eps)

g_exp = ln(2)*w_t*g_t
v_exp = beta2*v_exp_prev + (1-beta2)*g_exp^2
v_exp_hat = v_exp/(1-beta2^t)
u_tilde_mul = -eta_m*g_exp/(sqrt(v_exp_hat)+eps)
rho = sign_nonzero(w_t)*max(abs(w_t), tau)
u_mul = u_tilde_mul/rho
w_{t+1} = (1-eta_a*weight_decay)*w_t + w_t*u_mul + u_add
```

`sign_nonzero(0)=+1` only makes the paper's threshold denominator defined; the multiplicative
contribution is still exactly zero at `w=0`, so the additive branch owns zero/sign crossing.

Requirements:

* class state: additive first/second moments plus multiplicative exponent second moment;
* bias correction on both second moments and the additive first moment;
* learning-rate schedule remains the trainer's `--lr` for `eta_a`;
* `eta_m` and `tau` are typed flags, positive, persisted in resume sidecars, and checked by the
  resume-divergence guard;
* no Muon switch is reachable in this single-stage ticket;
* NumPy-vs-MLX parity is tested when MLX is available; NumPy tests remain runnable headless;
* optimizer state must round-trip through the existing flattened resume state.

The ticket values are preregistered exploratory values, not claimed optima:
`eta_m = eta_a` and `tau = 1e-6`. Their value-provenance is
`operator_requested_matched_first_probe` plus the paper equation; changing either requires a new
ticket, not an ad-hoc launch flag.

## 5. Telemetry and custody

Add independent default-on `--perclass-convergence-telemetry` observability. It reuses the same
realized-through-R argmax maps already requested by verdict telemetry and adds no scorer forward.
It writes, atomically in the run directory:

* append-only `perclass_convergence_trajectory.jsonl` diagnostic rows; and
* analyzer-ready `perclass_convergence_trajectory.json`, rematerialized from the in-memory/durable
  trajectory after each observation.

The JSON root schema is `ordinal_perclass_convergence_trajectory.v1`. Required custody:

* authority exactly includes `cohort=real-n600`, `pair_count=600`;
* seed and pair-order SHA;
* model fingerprint and exact initial-EMA SHA;
* optimizer and curriculum fingerprints;
* non-treatment config SHA (remove only the arm's declared treatment fields);
* GT-cache/data SHA; and
* preregistration SHA of this frozen specification.

Arm A treatment is exactly `{seg_loss: ce}`. Arm B treatment is exactly
`{seg_loss: margin_hinge, margin_target_end: 0.0}`.

Each trajectory row has strictly increasing accepted `update`, restart-monotonic
`wall_time_seconds`, and exactly the canonical five classes Road/Lane/Undrivable/Movable/MyCar;
there is no Sky. Each class contains exactly `all`, `hard`, and `easy` d_seg rates.

Hard/easy is the existing GT-logit margin contract:

```
hard := GT top1-top2 margin < 0.5
easy := GT top1-top2 margin >= 0.5
```

This threshold reuses trainer `--hardness-band=0.5`; it is unrelated to the two-pixel spatial
annulus. Positive support is required for every class/stratum on real n600; zero support refuses
receipt materialization rather than encoding a fake zero rate.

Extra row/root fields, validated by the recorder's own tests, carry:

* per-class/per-stratum flip and pixel counts;
* liveness (`accepted_updates`, `weights_stepped`, accepted fraction);
* actual GT-temporal single-frame-spike, non-spike, and endpoint-unclassified error contributions,
  whose sum equals total d_seg;
* `oracle_label_floor_dseg=0.005318` and `excess_above_oracle_floor`.

Persist global accepted-update count and accumulated active wall seconds in every resume sidecar.
Increment the count only after a successful `opt.update`. On resume, restore both before the next
verdict and refuse duplicate/nonmonotonic receipt rows. A fresh baseline is update 0; a resumed run
must not append a duplicate baseline.

## 6. Typed ticket and containment

Add four explicit named configs to `tools/launch_witness_run.py`:

* `perclass_convergence_ce_20260714`
* `perclass_convergence_margin_20260714`
* `perclass_convergence_stepnative_20260714`
* `perclass_convergence_mplus_adam_20260714`

Their compiler starts from sealed v7.5.2 at its feasible parent budget, then derives a true
single-stage 150-epoch child. Strip inherited stage caps/events and disable every later curriculum,
Muon, pose, and auxiliary-loss treatment identically. Preserve the representation/performance
substrate. Reject any non-n600 cache, non-600 pair count, non-150 horizon, non-fresh resume, or
confounded treatment field. Compilation is pure and must stamp operator-GO-required,
research-only, score-claim false, pointer-moved false.

The launcher must remain the only actuation surface. This lane may run:

1. pure compilation/dry-run;
2. `tools/witness_memory_preflight.py` against each emitted `launch.sh`; and
3. governed `--dry-start 2` boot verification.

It MUST NOT start the 150-epoch Metal job. Main receives the exact four governed commands.

## 7. Acceptance tests

1. DSL parser/registry tests: all new flags are real and typed; CE omits margin target; B emits exact
   zero target; C uses existing hosc/FINER route; D emits only optimizer treatment.
2. Config matching: A/B argv differ only in declared loss treatment; A/C only basis treatment; A/D
   only optimizer treatment. A/B non-treatment hash, init EMA, seed/order/curriculum/optimizer
   custody are identical.
3. Telemetry pure tests: named order, threshold boundary, supports, rate arithmetic, temporal
   partition sum identity, JSON safety, atomic materialization, duplicate/restart refusal.
4. Resume tests: accepted update and elapsed seconds survive save/load; M+Adam state survives
   flatten/restore; optimizer/eta_m/tau drift fails closed.
5. Optimizer tests: exact one/two-step NumPy equations, zero/sign-crossing, weight decay ordering,
   fp32 determinism, optional MLX parity.
6. Existing analyzer accepts synthetic A/B receipts produced by the recorder and derives finite
   update/wall rates. Existing focused tests remain green.
7. All four emitted `launch.sh` files pass memory preflight at <=0.70 RAM and `--dry-start 2`
   reports `boot_ok`; otherwise hand main an exact REFUSE/blocker, never a fake green.

## 8. Triality and handoff

* DSL leg: the two new lever factories plus the four typed tickets; reuse StepNativeActivation.
* DAG leg: standalone `perclass_convergence_ab_DAG_FEED_20260714.md` to avoid shared-DAG collision.
* Equation leg after measurement:

```
alpha_c,s^(arm) = -TheilSen(d_seg[c,s](u), u)
G^(arm) = mean_common(alpha_all) - mean_rare(alpha_all)
closure(arm) = 1 - G^(arm)/G^(CE)
curable_excess(u) = max(d_seg(u) - 0.005318, 0)
```

Dominant/contributory/inert/tradeoff thresholds remain those in the existing analyzer. Even a
positive closure is bounded if it does not reduce the curable excess or cross the label-smooth
floor by a demonstrated appearance-phase mechanism.

Main should launch **sequentially**: A then B first; C and D only after the decisive pair, unless
the four actual memory receipts prove concurrent safety without colliding with live #205/#432/#445.

