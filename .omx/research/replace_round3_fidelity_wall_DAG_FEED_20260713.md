# Standalone DAG FEED — REPLACE round-3 fidelity wall

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round3_fidelity_wall_20260713`  
**Node:** `FEED-95kill-fleet/replace-round3-fidelity-wall`  
**Status:** `NO_GO_REGISTERED_ROUND3_RUNGS`; `research_only=true`; shared-DAG append `DEFERRED_MAIN`  
**Pointer delta:** `NONE`

## Recall boundary

Round 2 is settled and is not re-derived here. Its fixed-replay machinery has a verified fp32 convex contraction certificate and content-addressed n600 replay custody, but its 31-feature chart measured heldout input-costate cosine `0.0014157933865487525` and relative L2 `1.0000018705777456`. The negative verdict covers that formulation and instance only; frozen scorer features, fixed nonlinear lifts, target localization, FORE successors, other charts, and other seeds remain open.

FORE is consumed but not activated. The current 600 isolated replay states are not transition-complete Markov tuples and carry no current-policy support receipt. A future identified successor must use `occupancy_current / occupancy_replay`; current weights remain disabled.

## Preregistered edge

```text
sealed n600 V9 replay, checkpoints {ep150, ep251, ep275}, seed455
  -> fixed 480 train / 120 heldout split
  -> exact CPU SegNet label call once per unique state
  -> capture frozen local prefix before first squeeze-excite/global pooling
  -> chart X = {bias, 32 prefix channels, source class, source margin, stage}
  -> convex ridge predicts exact prefix adjoint
  -> exact frozen-prefix VJP maps prediction to input RGB costate
  -> RUNG 1: linear chart
       -- if cosine >= 0.07078966932743762 and positive-dot states >= 0.60: STOP/GO
  -> RUNG 2: one seed455 fixed 16-frequency RFF lift, no width sweep
       -- same direction gates; if pass: STOP/GO
  -> RUNG 3a: source-margin 4.7%-area localizer
       -- if retained input-costate L2-square fraction >= 0.47: STOP/GO-LOCALIZER
  -> RUNG 3b: fixed-RFF convex costate-mass localizer
  -> scoped verdict + reformulation queue
```

The direction bar is `50 *` the round-2 measured cosine noise, not the dead inherited `-0.16` comparator. Its squared cosine corresponds to at least `0.501118%` normalized projected directional energy. Both aggregate cosine and a `60%` cross-state positive-dot guard are required.

## Closed laws

For frozen local prefix `phi` and convex head `W`, the input-costate prediction is

`lambda_hat_x = J_phi(x)^T lambda_hat_phi`, with `lambda_hat_phi = X_phi W`.

A fixed RFF map changes `X_phi`, not the optimization geometry in `W`; the round-2 spectral ridge contraction certificate therefore remains applicable to the lifted head.

For a binary support projector `M` and exact costate `lambda`, define retained L2-square fraction `rho = ||M lambda||_2^2 / ||lambda||_2^2`. Then

`cos(lambda, M lambda) = sqrt(rho)`.

This is a conditional exact-teacher identity, not a learned direction claim. Selecting support does not cheapen a dense exact teacher call.

The call law separates label acquisition from heldout validation:

`C_teacher = A_label + V_validation + c_label D`,

with `A_label=480`, `V_validation=120`, `D=7200`, and `c_label=0` for cached same-state reuse. Therefore label-only amortization is `D/A_label=15x`; inclusive amortization is `D/C_teacher=12x`. Validation calls do not amortize labels.

## Honest compute boundary

The frozen chart cuts before the first squeeze-excite/global pooling operation and is spatially local/tileable. This composes with the routed margin × class-pair waterfill as a future surrogate compute policy; it does not make the exact SegNet tileable. Exact SegNet support remains full-frame because of the measured 685-pixel halo and 23 global squeeze-excite reductions.

The derived cost fraction counts convolution FLOPs from observed real tensor shapes and reports omitted batch norm, activation, interpolation, loss/head work, and autograd bookkeeping separately. Contended wall timing is diagnostic only.

## Wire-in

- Sensitivity map: heldout exact/predicted input costates, renderer gradients, and costate-mass localization reductions.
- Pareto constraint: direction fidelity, exact calls, and prefix compute; no archive byte or score claim.
- Bit allocator: no direct actuator; the localizer may later consume the same margin × class-pair sensitivity object.
- Cathedral/autopilot: `REFUSE`; no live/paid launch or trainer argv.
- Continual learning: final receipt, dated memo, this FEED, canonical equation, and one advisory probe-outcome row.
- Probe disambiguator: this fixed rung ladder; first passing rung stops later measurement.

## Measurement append

Receipt: `experiments/results/replace_round3_fidelity_wall_20260713/measurement_receipt.json`, `172738` bytes, SHA-256 `83704e64d1e5a70c00cf96c19330ff8453459e1024f957bceb48f99972157d75`.

| Rung | MEASURED result | Gate | Verdict |
|---|---:|---:|---|
| pre-SE linear | costate cosine `0.0016650255538056325`; rel-L2 `1.0000004372846978`; positive-dot `0.916667` | cosine `>=0.07078966932743762`; positive-dot `>=0.60` | FAIL cosine |
| fixed RFF | costate cosine `0.0016791964165317613`; rel-L2 `1.0000003871015077`; positive-dot `0.916667` | same | FAIL cosine; best direction rung |
| source margin | retained L2-square mass `0.1634677541848741`; cosine identity `0.40431145690528497` | mass `>=0.47` | FAIL |
| RFF log mass | retained mass `0.024426459564827255`; cosine identity `0.1562896655727027` | mass `>=0.47` | FAIL |
| exact 4.7%-area oracle | retained mass `0.5278150212253758`; cosine identity `0.72650878950318` | diagnostic | target family remains open |

The direction winner is RFF only in the argmax-over-failed-rungs sense: it reaches `1.186046x` round-2 noise, far below the preregistered `50x` requirement. The NO-GO is `FORMULATION x INSTANCE`, not a family verdict.

Derived convolution cost is `226492416 / 39637335808 = 0.005714118050141177` of full forward-plus-input-backward convolution FLOPs (`175.005x` conv-only ideal ratio). Batch norm, activations, interpolation, loss/head math, and autograd bookkeeping are omitted and named; host-contended wall ratios are not promoted.

Clean-run exact calls are `480` training labels + `120` validation = `600`, giving `15x` label-only and `12x` inclusive amortization over `7200` cached uses. Campaign-conservative custody charges every invalid-attempt start: `626` training and `746` all calls, giving `11.501597444089457x` and `9.651474530831099x`. Campaign receipt SHA-256: `51fcd984bfd93662e74d64e3ad577ed0e302097ed1b55b2c92075dbaead0b664`.

FORE remains `NO_GO_CURRENT_INSTANCE__CONDITIONAL_FORMULATION_OPEN`; no weights or teacher calls are attributed to it. The exact heldout scratch (`283307760` bytes) was certified and deleted after 120-state direction and support reductions sealed; blockers `[]`.

Canonical equation: `tac.canonical_equations.replace_round3_fidelity_wall_20260713`. Shared equation/DAG registry writes remain deferred to main review, per the operator's uncommitted handoff.

Canonical probe outcome: advisory scoped-KILL row `replace_round3_fidelity_wall_v9_n600_seed455_20260713`. Lane maturity is L1 on `impl_complete` only; research-only and no promotion gates are claimed.
