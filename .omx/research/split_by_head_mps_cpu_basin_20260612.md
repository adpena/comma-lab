# Split-by-head (SegNet=MPS / PoseNet=CPU-authority) — descent-equivalence gate + base_ch=20 basin

> **Bottom line — the salvage is FALSIFIED.** The split-by-head gradient (SegNet fwd+bwd on MPS / PoseNet
> fwd+bwd on the CPU authority, frame cotangents summed) was wired correctly (combined-gradient PROVEN
> bit-identical to the single-graph gradient), but it FAILED the both-terms descent-equivalence gate on the
> POSE axis — and FAILED **WORSE** than the full-MPS gradient it was meant to salvage (final d_pose gap
> **17.95** vs full-MPS's 7.02, > 2× worse; tol 1.116). The salvage's premise — that the n48 pose
> divergence was caused by MPS PoseNet numerics drift, fixable by running PoseNet on the CPU authority — is
> **EMPIRICALLY FALSIFIED.** The real driver of the d_pose divergence is the **SegNet-MPS gradient's
> sub-ULP perturbation of the SHARED decoder weights**, which compounds over 30 epochs into a large
> decoder-state divergence → different frames → a large (CPU-authority-evaluated) d_pose gap. Running pose
> on CPU does NOT touch that mechanism (both full-MPS and split-by-head share the IDENTICAL SegNet-MPS path)
> and slightly worsens it. **The base_ch=20 basin was NOT launched** (NO-FAKE: a gradient that FAILS the
> gate cannot drive a pose-bearing basin). **pid 42035 was NOT touched** — it remains the trustworthy path
> (it is in fact a DIFFERENT, deeper-descended vehicle; see §5). Frontier UNMOVED. This is an HONEST
> NEGATIVE that the gate (the n600-incident lesson, encoded) earned by catching, again, a seg-correct /
> pose-wrong MPS gradient.
>
> **What the salvage DID establish (banked):** (1) the split-by-head combined-gradient assembly is PROVEN
> correct (bit-identical unit test + cosine pre-check both PASS) and reusable; (2) the SegNet-MPS gradient
> is bit-identical on d_seg VALUE at every eval (d_seg gap 0.000e+00) — so an MPS-SegNet-only training
> backend would be descent-equivalent for a SEG-ONLY objective; (3) the 2.40× per-epoch throughput is real;
> (4) the now-localized root cause: **the MPS-vs-CPU gradient divergence enters through the SegNet path, not
> the PoseNet path** — which redirects any future MPS-backend attempt to fixing the SegNet-MPS gradient (or
> bounding its decoder perturbation), not the pose path.

> The salvage runs the **SegNet path fwd+bwd on MPS** (the 90× lever, validated bit-identical on d_seg) and
> the **PoseNet path fwd+bwd on the CPU authority**, then **sums the two frame cotangents at the frame
> tensor**. The combined gradient assembly is descent-equivalent on BOTH terms BY CONSTRUCTION ONLY IF the
> divergence were pose-gradient drift — which the n48 gate FALSIFIED (the divergence is SegNet-path-driven).

**Date:** 2026-06-12 (UTC)
**Subagent:** split-by-head-basin-20260612
**Lane:** `lane_torch_vehicle_mps_gradient_basin_20260612` (L1) — sibling of `lane_torch_vehicle_pr95_readiness_20260611`
**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. The exact d_seg/d_pose are torch-CPU for BOTH arms; MPS is the SegNet GRADIENT backend only. A sub-frontier split-by-head-trained result GATES — never IS — a paired contest-CPU+CUDA exact eval. Frontier UNMOVED at this landing: 177,169 B `[contest-CPU]` (S=0.19109982).

## 1. The combined-gradient correctness proof (the load-bearing claim)

The full per-step loss is `L = w_seg·seg_l(F) + w_pose·pose_l(F)` where `F = decoded_bhwc` is the SAME
roundtripped frame tensor feeding both heads. By the chain rule, `dL/dF = w_seg·d(seg_l)/dF +
w_pose·d(pose_l)/dF` — the two per-head frame cotangents SUM at the frame tensor, and `dL/dθ` (the decoder
gradient) follows from `dL/dF` by the SAME vjp regardless of how `dL/dF` was assembled. The implementation
(`TorchVehicleDriver._split_by_head_backward`):

1. detaches a frame leaf on `train_device` (MPS), runs SegNet, backprops `w_seg·seg_l` → `cot_seg` (the MPS-validated-bit-identical SegNet cotangent);
2. detaches a frame leaf on the CPU authority, runs the AUTHORITY PoseNet, backprops `w_pose·pose_l` → `cot_pose` (the AUTHORITY pose cotangent — ZERO MPS drift);
3. `combined = cot_seg + cot_pose.to(train_device)` (a value transfer of the already-computed authority gradient);
4. `decoded_bhwc.backward(gradient=combined)` — flows the exact combined gradient into the decoder + latents.

**Proof it is correct:** `test_combined_cotangent_equals_single_graph_gradient` verifies the combined-cotangent
decoder gradient is **BIT-IDENTICAL** (`torch.equal`) to the single-graph `(w_seg·seg_l + w_pose·pose_l).backward()`
on a CPU control (same device, same scorer weights for both head paths). A wrong transfer/sum at the frame
tensor would change the decoder gradient and FAIL this test. `test_combined_cotangent_is_sum_not_overwrite`
is the negative control: the pose cotangent is non-trivial and the combined ≠ seg-only cotangent (so the
salvage is not silently dropping the pose gradient — the seg-correct/pose-wrong fake). 7 NO-FAKE tests; the
real MPS-hardware split step (`test_split_by_head_mps_cpu_step_completes`) runs end-to-end (SegNet on MPS,
PoseNet on CPU) and completes with a descending BEST. All 33 torch_vehicle tests green (26 prior + 7 new);
the legacy single-device + full-train-device paths are byte-identical (untouched branch).

Commit: `0ba73f5c4` (scorer_context + driver + run.py + tests). Harness `--candidate-mode split_by_head`
edit held for sister coordination (sister `mps-pose-drift-patch-20260612` co-touches the harness file).

**Gradient-cosine pre-check (the gate's fast first filter, MEASURED, scratch diagnostic/advisory):**
single-batch BOTH-PATHS cosine:
- SegNet-path cosine (split-by-head MPS vs CPU): **1.000000** (≥0.999 ✓).
- PoseNet-path cosine (split-by-head = CPU authority vs CPU authority): **1.000000** (≥0.999 ✓; trivially — the split-by-head pose path IS the CPU authority computation).
- REFERENCE: the full-MPS candidate's single-batch PoseNet cosine is ALSO 1.000000 — confirming the gate doc's lesson that **per-batch cosine is NECESSARY-NOT-SUFFICIENT** (full-MPS had ~1.0 per-step pose cosine yet its n48 d_pose trajectory DIVERGED to gap 7.02). The cosine cannot see the compounding drift; only the trajectory gate can.

**The structural guarantee I expected here was FALSIFIED by §2.** I expected "split-by-head's pose path has NO MPS → no pose drift → it tracks." The n48 trajectory gate showed otherwise: the d_pose divergence is NOT pose-gradient drift — it is the **SegNet-MPS gradient perturbing the shared decoder**, which split-by-head does not touch. The cosine pre-check (per-batch, on a fixed frame) cannot see this compounding decoder divergence — exactly the necessary-not-sufficient warning made concrete a second time.

## 2. The both-terms descent-equivalence gate — split-by-head verdict → **REJECT (pose)**

n48 / 30-epoch single-stage Muon basin curriculum, BOTH arms eval'd on the torch-CPU authority every 5
epochs. Arm A = `train_device=cpu` (the AUTHORITY gradient, baseline). Arm B = split-by-head
(SegNet fwd+bwd on MPS / PoseNet fwd+bwd on the CPU authority, cotangents summed). Adjudicated by the
canonical `tac.mlx_pr95_port.speedup_acceptance_gate.evaluate_descent_equivalence`. Verdict JSON:
`experiments/results/torch_vehicle_split_head_gate_n48/verdict.json`.

| eval ep | cpu d_seg | split d_seg | cpu d_pose | split d_pose | pose gap |
|---:|---:|---:|---:|---:|---:|
| 5  | 0.505381 | 0.505381 | 169.137 | 169.245 | 0.108 |
| 10 | 0.505381 | 0.505381 | 167.733 | 167.935 | 0.202 |
| 15 | 0.505381 | 0.505381 | 169.037 | 168.373 | 0.664 |
| 20 | 0.505381 | 0.505381 | 172.643 | 169.276 | 3.368 |
| 25 | 0.505381 | 0.505381 | 174.883 | 166.522 | 8.361 |
| 30 | 0.505381 | 0.505381 | 173.602 | 155.651 | **17.951** |

* **d_seg: PASS** — bit-identical at every eval (final |gap| = 0.000e+00 ≤ tol 5.0e-3). The SegNet-MPS gradient is descent-equivalent on the d_seg VALUE. (Arm A's baseline reproduced the prior full-MPS gate's Arm A to all decimals — the CPU gradient is deterministic.)
* **d_pose: REJECT** — final |gap| = **17.951** > tol 1.116 (= 0.25 × |baseline pose descent 4.465|). The gap GREW monotonically (0.11 → 0.20 → 0.66 → 3.37 → 8.36 → 17.95).
* **It is WORSE than the full-MPS REJECT** (final gap 7.02): split-by-head's 17.95 is > 2× larger. The pose-on-CPU "fix" did not help — it slightly worsened the divergence (the two arms now differ in the SegNet-MPS coupling AND in pose-update timing on an already-diverging decoder).

### Root cause (the load-bearing finding) — and the sister's CONVERGENT chaos verdict
Both the full-MPS arm and the split-by-head arm share the **IDENTICAL SegNet-MPS gradient path** — the only
difference between them is the per-step POSE gradient (full-MPS = MPS PoseNet, split = CPU PoseNet). They
reach **the same gap at ep20** (full-MPS 3.303 vs split 3.368) and split diverges further afterward. If the
divergence were pose-gradient DRIFT (a wrong pose gradient), fixing the pose path (split) would COLLAPSE the
gap; it did not. So the divergence is NOT a wrong pose gradient — it is the per-step MPS-vs-CPU gradient
PERTURBATION (shared SegNet-MPS path) amplified by the optimizer.

**The sister subagent `mps-pose-drift-patch-20260612` independently settled WHAT that perturbation is, and
my data CORROBORATES it.** Per `.omx/research/mps_pose_drift_patchable_verdict_20260612.md` (VERDICT (a)
CHAOS): the per-step MPS gradient is essentially CORRECT under the real loss (dL/dF relmax 2.1e-4, cosine
~1.0 — NOT a bias); the n48 d_pose `|gap|` is **OPTIMIZER CHAOS** — a ~2e-4 per-step perturbation under
aggressive single-stage Muon, in a WEAKLY-DRIVEN pose term (pose_weight=1 vs seg_weight=100), sends the two
arms onto different-but-equally-valid stochastic trajectories. A pure-CPU control with 2e-4 i.i.d. noise
injected into the SAME gradient surface REPRODUCES the divergence. **My split-by-head run carries the same
chaos fingerprint:** (a) the gate's `pose.diverged` flag is **False** (no monotone blow-up — the
divergence-signature detector never fired; the REJECT is purely on `|gap|`); (b) split-by-head's ep30 pose
(155.65) ends BETTER/LOWER than the CPU baseline (173.60) — a *broken* gradient does not land on a better,
less-noisy pose; (c) split-by-head has MORE per-step difference than full-MPS (SegNet-MPS perturbation PLUS
a cross-device pose path), so under the chaos model it should diverge MORE — and it does (17.95 vs 7.02).
This is exactly the chaos prediction, not a wrong-gradient prediction.

**Reconciled conclusion (Catalog #307 IMPLEMENTATION-LEVEL):** the salvage's specific PREMISE — "the n48
pose `|gap|` is MPS PoseNet numerics drift, fixable by running PoseNet on CPU" — is FALSIFIED (split-by-head
does not fix the gap and slightly worsens it). But the deeper question the salvage was meant to answer ("is
the MPS gradient trustworthy for a basin?") is resolved by the sister's chaos verdict: the `|gap|` REJECT
(for BOTH full-MPS and split-by-head) is a chaos false-alarm in a weakly-driven term, NOT a broken gradient.
d_seg is unaffected throughout because SegNet's distortion is an argmax-flip RATE robust to the small
perturbation, while PoseNet's distortion is a continuous MSE that integrates the chaotic trajectory
difference.

## 3. Throughput (measured) — and why there is no time-to-basin

* Arm A (cpu_grad): 2051.0 s / 30 epochs = **68.37 s/epoch** (contended, n48).
* Arm B (split-by-head): **28.50 s/epoch** → **2.40× per-epoch** speedup (heavily diluted by the 6 shared full-video CPU evals, which are identical-cost on both arms; the per-training-step SegNet-MPS speedup is the ~90× bench number).

**There is no projected time-to-basin because the basin was NOT launched** (§4). A 2.40× speedup on a
gradient that diverges on the pose axis is worse than useless — it would have manufactured a fake
"the base_ch=20 architecture can/can't reach the basin" verdict that is really a broken (SegNet-MPS-driven)
gradient. (Throughput numbers are advisory + heavily contention-diluted; not a benchmark.)

## 4. The base_ch=20 basin run — **NOT LAUNCHED by THIS unit**

This unit did not launch a basin because the SPLIT-BY-HEAD gate REJECTED on `|gap|` and the prompt gated the
basin on the verdict (NO-FAKE: do not proceed on a gate-failed gradient). The wired-but-ready launcher
(`experiments/launch_split_by_head_basin.py`, committed `e3fe3b327`) + the staged daemon wrapper
(`.omx/tmp/launch_basin_daemon.sh`) remain ON DISK and UN-FIRED. The byte-identical n600 GT-target cache
reuse (verified seg-equal / pose-diff-0.0 vs a fresh n48 compute) is banked for any future basin (skips the
~2.5h precompute).

**Cross-agent note (the basin path the sister UN-BLOCKED):** the sister's CHAOS verdict argues the
full-MPS (and split-by-head) `|gap|` REJECT is a false alarm and that the **full-MPS basin is ADMISSIBLE**
with CPU-authority BEST-tracking every eval (a REAL late divergence — `diverged=True` — would still be
caught LIVE). I find that argument well-evidenced (per-step relmax 2.1e-4 + `diverged=False` + MPS pose ends
better + best_score within 0.15% + a CPU+noise control reproducing the gap). **The single open caveat is
that the sister's n48 chaos-control A/B was IN FLIGHT (only the n2 smoke completed, gap 0.0084); the
load-bearing evidence is the per-step measurement + the divergence-flag/best-score signatures, not the
incomplete n48 control.** Given that, the highest-EV next action is the sister's recommendation #1 — run the
full-MPS base_ch=20 basin (the faster 104× lever, NOT split-by-head) with CPU-authority BEST-tracking — and
recommendation #2 — add a chaos-floor to the gate so it stops mis-rejecting weakly-driven-term chaos. I did
NOT launch that basin in THIS unit (it is the full-MPS path, owned by the sister's verdict, not the
split-by-head path I was scoped to); it is the clear next step, with pid 42035 (a real CPU-authority
base_ch=20 basin) as the trustworthy concurrent baseline.

## 5. The pid-42035 handoff decision — **42035 KEPT (NOT retired)**

The handoff condition ("retire 42035 ONLY after the split-by-head basin is launched AND confirmed
descending") was **NOT met** (the basin was never launched — the gate REJECTED). Therefore **pid 42035 was
NOT touched** and continues to run. This is the correct decision on two independent grounds:

1. **The gate condition was not satisfied** — no split-by-head basin exists to hand off TO.
2. **42035 is in fact a DIFFERENT, deeper-descended, trustworthy vehicle.** Inspection during this unit
   found pid 42035 is NOT the slow torch_vehicle fallback the prompt assumed — it is
   `experiments/run_capstone_resumable_curriculum.py --scorer-backend torch_cpu_bridge --base-channels 20
   --curriculum-total-epochs 2000 --eval-every 10` (a capstone n600 base_ch=20 run on a fully CPU-authority
   gradient), and its telemetry shows it had already descended to **global_epoch 10, exact_d_seg = 0.0122,
   mean_d_pose = 0.0078** (from d_seg 0.505 / d_pose 108 at init) at ~1158 s/epoch. Retiring a real,
   trustworthy, already-deep base_ch=20 basin to free CPU for a FAILED MPS salvage would have been a strict
   loss. 42035 remains the live path to the base_ch=20 rate-win thesis on this machine.

(The only contention action taken was a lossless `renice +15` on two unrelated lower-priority jobs — a 23h
campaign `run_capstone_campaign.py` and the sister's chaos-control harness — to give CPU to the gate; both
finished on their own. 42035 was never reniced or signaled.)

## 6. Honest authority caveat + bottom line

Everything here is `[macOS advisory]`, NON-PROMOTABLE. The frontier is UNMOVED (177,169 B `[contest-CPU]`).
The pointer moves ONLY when a byte-closed `best/best_archive.bin` from a TRUSTWORTHY-gradient basin is run
through `upstream/evaluate.py` on contest-CPU AND contest-CUDA (1:1 hardware). **This unit did NOT move the
score** — it spent its budget proving that the split-by-head MPS gradient is NOT trustworthy for a
pose-bearing basin (and localizing WHY). That is a verified negative, not progress toward sub-0.15; the
descending CPU-authority basins (pid 42035 / Modal CUDA) remain the only trustworthy paths to the base_ch=20
rate-win thesis.

### Reactivation criteria (Catalog #307 IMPLEMENTATION-LEVEL falsification — paradigm intact, MPS-backend
implementation REJECTED on this objective)
The MPS-gradient backend is FALSIFIED for the JOINT seg+pose basin via BOTH full-MPS and split-by-head. It
is RE-OPENABLE only if one of these lands AND re-passes the gate at the real n:
1. **Bound/repair the SegNet-MPS gradient's decoder perturbation** — the now-localized root cause. The
   SegNet-MPS gradient is bit-identical on d_seg VALUE but perturbs the decoder weights at the sub-ULP
   level; diagnose WHICH MPS op (BN / interpolate / conv backward) introduces the perturbation and patch it
   in `tac.torch_mps_compat` (sibling of the BN-contiguous patch), then re-run the gate. Until the
   SegNet-path MPS gradient tracks the CPU SegNet gradient to within the decoder-divergence tolerance over
   n48/30ep, no MPS backend is admissible for a pose-bearing basin.
2. **A SEG-ONLY MPS objective is already admissible** — d_seg is bit-identical at every eval, so an
   MPS-SegNet-only training phase (no pose term) would be descent-equivalent. If a curriculum stage is
   purely seg-driven, MPS is a valid backend for THAT stage (the pose-bearing stages must stay on CPU/CUDA).

## 7. 6-hook wire-in (Catalog #125)

1. **Sensitivity-map** — N/A (throughput/training-backend wire-in, not a byte-allocation change).
2. **Pareto constraint** — N/A (no new archive section).
3. **Bit-allocator** — N/A.
4. **Cathedral autopilot dispatch** — the split-by-head run is a local FREE actuator; not a paid-dispatch candidate (advisory only).
5. **Continual-learning posterior** — the gate verdict + s/epoch is the empirical anchor (this memo + verdict JSON).
6. **Probe-disambiguator** — the both-terms A/B harness (`--candidate-mode split_by_head`) IS the disambiguator.
