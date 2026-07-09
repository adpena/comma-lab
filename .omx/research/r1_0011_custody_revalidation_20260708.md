# R1's d_pose 0.0011 — CUSTODY RE-VALIDATION (#238) — 2026-07-08

**Task:** re-validate the load-bearing anchor for the ONLY remaining pose door — **joint pose-descent where
the render co-adapts** — namely R1's claimed **d_pose 0.0011** (symposium `council_pose_carrier_optimal_form_symposium_20260703.md`
§1 L0 "SOLID shippable floor 0.105"). Read-only, $0, memory-aware (pid 63069 live). Every number below is
`[macOS-CPU advisory] NON-PROMOTABLE`. **Pointer 0.19110 UNMOVED.**

**Custody verdict in one line:** the 0.0011 is a **VALID measurement** (frozen CPU-torch PoseNet, the EXACT
contest d_pose definition, n600, through-R, EMA-shadow, valid liveness) — **the charter's prime suspect
(ξ-MSE vs PoseNet-output-MSE conflation) is DEFINITIVELY REFUTED** — **BUT it is NOT byte-closed**: the
0.0011 lives entirely in a trained per-pair `dxi` table that the current serializer does **not** ship. So
the number is a real TRAINING-SIDE d_pose, and joint pose-descent is a real, justified door — but the
symposium's "SOLID **shippable** floor 0.105" is **OVER-STATED**; shippability is **PENDING #238** (serialize
the trained ξ_eff + re-measure through inflate). **Verdict row fired: (a)-with-caveat** — joint pose-descent
is REAL and a dedicated run/byte-close is justified; the specific 0.105 contribution is provisional.

---

## 1. Provenance chain of the 0.0011 (traced to the primary artifact)

| layer | artifact | what it says |
|---|---|---|
| symposium claim | `council_pose_carrier_optimal_form_symposium_20260703.md:31,62` | "R1 store-nothing ξ: d_pose 0.0011 → contribution 0.105 … the current shippable pose floor" (tagged SOLID) |
| DAG verdict record | commit `148636537` / FEED-poseladder (DAG `:7506`) | "R1 (#245) MEASURED the trained store-nothing pose descending through-R: d_pose 62.44→**0.0011** (plateau ep1074/1093 ~0.00108), d_seg HELD ~0.0046" |
| **the RUN (primary artifact)** | `experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z/` | `launch.sh` + `relaunch.log` verdict rows + `levelset_witness_ema_mlx.npz` (has the descended dxi) |

**R1 = task #245**, a JOINT pose-descent run: warm-started from a **converged d_seg witness**
(`--resume-from levelset_n600_v2_attrclean_20260630T194549Z`, mod-dim 26) at **ep1001 inside the Muon
finisher**, `--w-pose 1.0`, `--pose-carrier --pose-carrier-source generated --pose-carrier-residual-mode
table`, `--verdict-pairs 0` (ALL 600) `--verdict-batch 32` (the OOM fix), `--eval-every 1`, `--async-verdict`.
A per-pair `dxi` (600×6) residual is child-attached → EMA/AdamW/Muon-tracked.

### The measured descent (relaunch.log `stage":"verdict"` rows — frozen CPU-torch PoseNet, n600, through-R)
| epoch | d_seg | d_pose | ep_loss (liveness) |
|---|---|---|---|
| 1000 | 0.004526 | **97.196** | — |
| 1001 | 0.004502 | **62.443** | 713.5 |
| 1021 | 0.005309 | **0.003343** | 166.7 |
| 1040 | 0.004984 | **0.001842** | 155.6 |
| 1054 | 0.004951 | **0.001397** | 152.0 |
| 1074 | 0.004746 | **0.00108** | 146.8 |
| 1093 | 0.004619 | **0.001085** | 144.0 |
| 1108 | 0.004586 | **0.001012** | 144.4 |

Every row is stamped `axis:"[macOS-CPU advisory] NON-PROMOTABLE"`. d_seg **held** ~0.0046 across the descent
(seg⊥pose held — frame0 is 100% in the SegNet-null since SegNet reads `x[:,-1]`=frame1 only). This is a
genuine, smooth, plateauing descent — NOT a single suspicious point.

---

## 2. Custody classification (apparatus-validity precondition — each leg CHECKED)

The charter named the confound classes to check. Verdict on each:

1. **ξ-MSE vs PoseNet-OUTPUT-MSE conflation (THE PRIME SUSPECT) — REFUTED.** The verdict routes through
   `cpu_verdict_d_pose_batch` (`experiments/train_witness_realized_through_R_mlx.py:790-809`), which returns
   `MSE(PoseNet(generated_pair)[:6], gt_pose_list[i])`. The target `gt_pose_list` = the cache `gt_poses`,
   which is built by `_cpu_pose_raw(posenet_cpu, real_f0, real_f1)` (`:644`) = **PoseNet(real_pair)[:6]**.
   Verified the cache: `gt_n600.npz['gt_poses']` shape (600,6), values are PoseNet OUTPUTS (e.g. pair-0 =
   `[34.24, -6.7e-5, 1.7e-3, -8.6e-4, -1.3e-2, -2.1e-4]`), **not** raw ego ξ. So d_pose =
   MSE(PoseNet(gen)[:6], PoseNet(real)[:6]) = the **exact contest definition**
   (`d_pose = MSE(PoseNet(generated)[:6], PoseNet(original)[:6])`). The 0.0011 is NOT a pose-parameter-space
   number. **Prime suspect dead.**
2. **Frozen-PoseNet authority, not a proxy — CONFIRMED.** `cpu_verdict_d_pose_batch` runs the frozen
   CPU-torch PoseNet under `torch.inference_mode()`, NEVER MPS/MLX. (The earlier "Track B classmean proxy
   4.97" that appears in the 07-02 wiring memo is a DIFFERENT, PRE-residual proxy — not the source of 0.0011.)
3. **through-R — CONFIRMED.** Generated pair = frame1 witness render, frame0 = warp(the witness's OWN
   camera-native render, ξ_eff) — `_pc_verdict_f0_uint8` warps `_fwd_numpy(f0)`→`_torch_R_to_camera_uint8`,
   the same R path as byte-close (wiring memo `store_nothing_pose_carrier_wiring_20260702.md` §5; frame0
   decode bit-exact on the n6 smoke, `max_abs=0`).
4. **n — CONFIRMED n600.** `--verdict-pairs 0` = all 600 pairs (chunked at 32 for the OOM spike). Not a subset.
5. **EMA-shadow vs live — CONSERVATIVE-DIRECTION.** The verdict is on the EMA shadow. During a *descent*,
   the EMA shadow LAGS the (better) live weights → the reported d_pose is if anything an **over-estimate**,
   never a favorable-direction artifact. The EMA-lag confound (MEMORY L81/CURRENT-STATE) inflates d_pose
   during descent; it cannot fake a LOW value.
6. **Liveness / lever-active — CONFIRMED.** ep_loss 143-713 (nonzero → training happening, not frozen);
   `__cfg_w_pose = 1.0` and the checkpoint carries a **non-zero trained `pose_carrier.dxi`** (below) → the
   lever under test was genuinely ACTIVE. No `frozen_epoch` / `spike_deadlock` signature.

**Result: the measuring state was VALID for the interpreted window.** The 0.0011 passes the L3
verdict-clearance precondition as a *training-side advisory measurement*.

---

## 3. The decisive shippability finding — the 0.0011 lives in an UN-SHIPPED dxi table

Read-only npz inspection of R1's checkpoint (`levelset_witness_ema_mlx.npz`, 55 keys):
```
pose_carrier.xi_stored  (600,6) f32  absmean 0.2296   # deterministic calibration ξ
pose_carrier.dxi        (600,6) f32  absmean 0.00382  # the TRAINED per-pair residual (EMA shadow)
__cfg_w_pose            ()      f64  = 1.0            # JOINT training (NOT w_pose=0)
```
(resume_state.npz additionally carries liveP/emaP copies + the AdamW `dxi.m/.v` moments — fully resumable.)

- **The descent is entirely the dxi's doing.** At ξ_stored alone (calibration, no dxi) the read-back d_pose
  is 2.562 (`stage":"pose_carrier","s_t_fit":{"0.044":2.562}` in run.log) — the "cap ~2.5". The trained
  `dxi` table (600 free 6-vectors) is what carried 62.44→0.001. A per-pair 6-DOF table steering a per-pair
  6-dim PoseNet target is generically solvable **once the render co-adapts** (which it did — `w_pose=1.0`,
  Muon kept training the render for ~100 epochs). This is exactly the door the 07-08 A2/A2+ ladder left
  open ("NOT killed = joint pose-descent training; render co-adapts").
- **BUT the current byte-close serializer does NOT ship dxi.** `build_pose_carrier_section`
  (`tools/levelset_byte_close_and_eval.py`, per `finding1_store_nothing_pose_rate_resolution_20260703.md`
  §2) writes `xi = xi_from_pose_calibration(gt_poses[p], s_t, s_r, pitch)` — a **deterministic recompute**,
  NOT `xi_stored + dxi`. The 07-08 arms-measured memo confirms it: *"byte-close REBUILDS a deterministic
  GT-calibration and does NOT load [the trained dxi]"* (`pose_carrier_arms_measured_20260708.md:80`).
  ⇒ a byte-close of the store-nothing carrier reproduces the ~1.99 **deterministic no-dxi floor**, NOT
  R1's 0.001. **R1's 0.0011 is therefore not yet expressible in a shipped archive.**

**#238 = the real, un-done test:** serialize the trained ξ_eff = ξ_stored + dxi (600×6 fp16 ≈ 7,200 B,
rate ≈ 0.0005 — cheap and rule-118-legal, it IS the store-nothing payload), then re-measure d_pose through
inflate at n600. Only that closes the SOLID/shippable gap.

---

## 4. Reconciling 0.0011 (R1) vs ~1.79 (run-1) — NOT a contradiction (different checkpoints)

The apparent 1600× clash dissolves: the 07-08 arms/L2 re-measurements used **run-1** (the LIVE #205,
mod-dim-32, **params=117527**, ep~200), NOT R1's **mod-dim-26** ep1108 descended checkpoint.
- run-1 @ ep200: dxi barely trained → 1.99 deterministic → 1.79 with its early dxi (~11% refinement,
  `pose_carrier_arms_measured_20260708.md:45`). It simply has not done the dedicated joint pose-descent.
- R1: converged d_seg witness + ~100 dedicated pose epochs → dxi fully descended → 0.001.
- The A2/A2+ ladder (floors ~1.2) solved 6-DOF on a **FIXED** render — a *different formulation*; its own
  verdict says "R1's 0.0011 NOT reproducible post-hoc on a fixed render — IF real it needed JOINT
  pose-descent co-adapting the render" (FEED-poseladdermeasured). No measurement CONTRADICTS R1; none
  independently CONFIRMS it either (R1's actual mod-dim-26 checkpoint was never re-measured).

Also corrected: FEED-posesolve's "(1) R1 (#245): SOLO descent vs **frozen w_pose=0** render" is
**factually wrong** — R1 was `--w-pose 1.0` (checkpoint `__cfg_w_pose=1.0`), a JOINT descent. The verdict
stands regardless (the render co-adapting is exactly what made it work), but the DAG line should be
corrected on the next triality pass.

---

## 5. Re-measurement status (charter step 3)

**Artifacts PRESENT and loadable** (checkpoint + non-zero dxi + optimizer moments + resume state), so R1 is
re-measurable **in principle**. A faithful $0 re-measure was **NOT attempted** because it is not actually
$0/read-only-cheap under the current apparatus + CONTAINMENT (pid 63069 live, ≥10 GiB floor):
- the **byte-close tool CANNOT load the trained dxi** (`arms_measured:80`) → it would report the ~1.99
  deterministic floor, i.e. it does **not test R1's claim** without a serializer change (that IS #238);
- reproducing the training verdict render path requires the **MLX-GPU camera-native warp render** of the
  mod-dim-26 EMA on n24 — a heavy op, inappropriate to fire read-only while the live run holds memory.

The primary measurement (the relaunch.log n600 verdict rows) is itself the frozen-PoseNet authority result,
and I VERIFIED the measurement PATH end-to-end (§2), so the custody verdict does not require re-execution to
refute the prime suspect. **Faking a lighter proxy re-measure would be the exact surrogate-≠-authority
anti-pattern; declined.**

---

## 6. VERDICT (graded, verdict_scope)

- **Row (a) FIRES, with caveat — joint pose-descent is REAL.** R1's 0.0011 is a genuine frozen-CPU-torch
  PoseNet, contest-definition, n600, through-R, EMA-conservative measurement in a valid measuring state.
  The charter's prime suspect (ξ-MSE conflation) is **REFUTED**. Therefore **a dedicated joint pose-descent
  run is justified** — the render co-adapting to make the cheap warp pose-legible is the ONLY measured path
  to low pose (consistent with `pose_l2_truedepth_probe_measured_20260708.md:131-133`), and this witness
  architecture is **NOT** honestly pose-blocked at 1.79 (that framing describes an *un-descended* checkpoint).
- **Caveat (the honest limit — NOT row (b)):** the 0.0011 is **NOT byte-closed**. It lives entirely in the
  trained per-pair `dxi` table, which the current serializer does not ship (it recomputes the deterministic
  calibration ξ → ~1.99). So the symposium's *"SOLID **shippable** floor 0.105"* is **over-stated** →
  downgrade to **"SOLID training-side advisory; shippability PENDING #238."** verdict_scope: this caveat is
  at the **byte-close/serialization** level, NOT a measurement-validity failure.
- **Row (b) does NOT fire** — 0.0011 is not a measurement artifact; the confound the charter feared is absent.
- **#238 (the decisive next step, operator-GO):** (i) serialize ξ_eff = ξ_stored + dxi (fp16, ~7.2 KB,
  ~0.0005 rate) in store-nothing mode; (ii) re-measure d_pose through the real inflate/decode at n600 on
  R1's checkpoint (or on a fresh dedicated joint pose-descent arm carried to convergence). GO = the joint
  door is both real AND shippable → pose ~free legally; NO-GO (byte-close d_pose ≫ 0.0011) = the dxi doesn't
  survive serialization/inflate → pose stays a budget item and sub-0.15 rides d_seg+rate.

**Pointer 0.19110 UNMOVED.** All numbers `[macOS-CPU advisory] NON-PROMOTABLE`.
