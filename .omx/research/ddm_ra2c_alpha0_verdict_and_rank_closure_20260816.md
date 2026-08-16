# ra2c — the α=0 anchor closes the EUCLIDEAN carrier-rank ladder, and names the door left open

`date_utc: 2026-08-16` · `owner: MAIN` · `axis: [macOS-CPU advisory, upstream/evaluate.py n600]`
`score_claim: false` · `promotable: false` · `verdict: REFUSED_ON_ADVISORY_BAR (α=0)`
receipt: `/Volumes/APDataStore/pact/ddm_ra2c_alpha_ladder_a0_20260816/retained/RA2C_ALPHA_LADDER.json`
spliced raw sha256 `dbd30b21ed44ef1ac79d8540cf8b02717f26642c222a88f1200b48d9bdeeaff5` (3,662,409,600 B, retained)

## 1. What α=0 measured

`archive.zip` held byte-identical (182,759 B); only frame_0 rendering varied. So the delta is pure
distortion and the rate credit is applied analytically through the registered affordance law.

| quantity | value |
|---|---|
| d_pose base → α=0 | 1.4747e-4 → **51.67767334** (ratio **350,428×**) |
| d_seg base → α=0 | 0.00042714 → **0.00042714 (exactly unchanged)** |
| advisory bar at full 22,161 B credit | 1.9162× — **over by 182,880×** |
| T4 bar at full credit (ratio-transfer) | 7.7229× — over by 45,375× |
| α=1 control | **600/600 pairs byte-identical** |
| build / evaluate | 192.6 s / 393.8 s |

Two products beyond the refusal:

1. **d_seg is provably untouched by any frame_0 carrier edit.** SegNet reads the last frame only.
   Every carrier question is a pure (pose, rate) trade — a confound class removed for the rest of
   the ladder.
2. **`ddm_ra2b_carrier_chain_control` is validated at n600**, not just ra2b's 3-pair proof. The
   mirror of the shipped f26 → selector-split → CPR1 → compensation → render → `apply_pixel_mode`
   chain reproduces the base render byte-for-byte on every pair.

## 2. The damage law, and the tolerance it implies

Two anchors — ratio = 1 at α=1 (proven by the control) and 350,428 at α=0 — fix
`ratio − 1 = K·err^p` with **K = 350,427**. `p = 2` is the principled value: d_pose is a quadratic
form in the pose residual. The tolerance any byte-returning approximation must meet:

| rung | bytes back | credit S | advisory bar | max coeff error (p=2) |
|---|---:|---:|---:|---:|
| α=0 (delete all) | 22,161 | 0.014756 | 1.9162× | 0.162% |
| rank-4 | 14,662 | 0.009763 | 1.5731× | 0.128% |
| rank-11 | 1,514 | 0.001008 | 1.0532× | 0.039% |

## 3. MEASURED: the carrier has no low-rank structure — and Eckart–Young makes that a BOUND

Exact SVDs of the three factorizable objects in the shipped archive:

| object | shape | cond | rank-4 rel err | rank-11 rel err |
|---|---|---:|---:|---:|
| basis | 12 × 2304 | 3.61 | 52.08% | 14.71% |
| coeff | 600 × 12 | 5.40 | 49.20% | 11.22% |
| rendered field `C·B` | 600 × 2304 | 17.32 | **25.15%** | **4.23%** |

Every spectrum is flat; the carrier is 12-dimensional and effectively full rank. **By
Eckart–Young the truncated SVD is the OPTIMAL rank-r approximation in Frobenius norm**, so these
are not estimates of one method — they are lower bounds on *any* rank-r approximation in that norm.

Against the tolerances in §2 the best rung misses by 33× (rank-11: 4.23% vs 0.039%) and rank-4 by
196× (25.15% vs 0.128%). Rescue would require a damage exponent **p > 5** (rank-11) or **p > 7**
(rank-4) — implausible for a smooth quadratic scorer term.

**VERDICT: the Euclidean carrier-rank ladder is CLOSED.** `verdict_scope: FORMULATION` — rank-r
truncation of the shipped 12-dim carrier in the Frobenius/Euclidean metric, priced against the
registered rate-credit affordance law on the hv1 ep0634 base.

## 4. CORRECTION — the 6.00% figure I was carrying does not reproduce

I have been quoting "ra1b measured rank-4 exhaustive at 6.00% coefficient error" as the ladder's
input. Measured directly on the shipped objects, rank-4 error is **49.20%** (coeff), **52.08%**
(basis), **25.15%** (rendered field). None is 6.00%. Either that figure measured a different
object or metric, or my carry of it was wrong. The closure above rests only on figures measured
in this run; the 6.00% number is withdrawn from the argument and should not be re-cited without
its own receipt.

## 5. THE DOOR THIS LEAVES OPEN — and it is exactly the level error we named today

Eckart–Young is optimal in the **Frobenius** norm. d_pose does not read Frobenius; it reads the
PoseNet-Jacobian-weighted (Fisher) norm. A rank-r approximation *optimal in the pose metric* can
carry far less pose error at the same rank than the Euclidean truncation — the low-energy
directions the SVD discards are not the pose-relevant ones unless the two metrics align.

This is the same defect named this morning in
`rate_credit_ladders_run_largest_first_and_euclid_ladders_are_a_level_error_20260816`: *a ladder
ranked by Euclidean MSE is a LEVEL error when the score reads a scorer.* §3 closes the Euclidean
ladder honestly; it does **not** close the pose-metric ladder, which has never been built.

**Named successor (owner MAIN, $0 + one advisory slot):** whiten the coefficient space by the
measured PoseNet Jacobian (the `ms4`/`ms3` pose-quadratic custody bundle already holds the ≤6-dim
per-pair form), take the rank-r truncation *there*, and re-price against the same bars. Falsifier:
if the pose-metric rank-4 error also exceeds 0.128%, the carrier family is closed in both metrics
and the remaining pose route is exclusively the js1 joint line.

## 6. Honest limits

- Advisory axis, single instrument. The T4 column assumes the d_pose RATIO transfers; that is most
  defensible at α=0, where the perturbation dwarfs instrument noise, and is **not** a score claim.
- `K` is a two-point fit of a one-parameter model. §3's closure does not depend on the fit being
  exact — it depends on the *order of magnitude* of K and on Eckart–Young, both of which hold under
  any `p ≤ 5`.
- The α ladder returns bytes only at its α=0 endpoint; interior α values change fidelity without
  changing the coded payload, which is why no interior α was measured.
