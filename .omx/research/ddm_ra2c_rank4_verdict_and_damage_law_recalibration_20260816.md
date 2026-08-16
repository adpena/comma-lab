# ra2c rank-4 — REFUSED, and the damage law over-predicts by 9.23× (a correction to today's own memo)

`date_utc: 2026-08-16` · `owner: MAIN` · `axis: [macOS-CPU advisory, upstream/evaluate.py n600]`
`score_claim: false` · `promotable: false` · `verdict: REFUSED_ON_ADVISORY_BAR (rank-4)`
receipt: `/Volumes/APDataStore/pact/ddm_ra2c_rank4_20260816/retained/RA2C_ALPHA_LADDER.json`
spliced raw sha256 `36a81b9daae3af95094af01af3160f12557b95a24d4959a54c0f3e4cbd1de498`
(3,662,409,600 B, retained) · build 192.2 s / evaluate 391.0 s / rc=0 in 587 s

## 1. The measurement

`archive.zip` held byte-identical at 182,759 B; only frame_0 rendering varied.

| quantity | value |
|---|---|
| rank / true carrier rank | 4 / 12 |
| Frobenius rel err (rendered field `C·B`) | **25.1449%** |
| energy kept | 93.677% |
| rowspace round-trip max abs | 4.16e-15 (exact; refuses above 1e-6) |
| d_pose base → rank-4 | 1.4747e-4 → **0.35402399** (ratio **2,400.65×**) |
| d_seg base → rank-4 | 0.00042714 → **0.00042714 (exactly unchanged)** |
| advisory bar at 14,662 B returned | 1.5731× — **over by 1,526×** |
| T4 bar (ratio-transfer) | 4.7394× — over by 507× |
| α=1 control | **600/600 pairs byte-identical** |

**VERDICT: REFUSED.** rank-4 is not affordable on either axis.

## 2. THE FINDING — the damage law over-predicts by 9.23×

The two-anchor law (`ratio − 1 = K·err²`, `K = 350,427`, fit at α=1 and α=0) predicts
**22,158×** at 25.1449% error. Measured: **2,400.65×**.

```
K_eff(at rank-4's error) = (2400.65 − 1) / 0.25144894²  =  37,953
K_endpoint (α=0 two-point fit)                          = 350,427
over-prediction factor                                  =    9.23×
```

**K is not constant along the ladder.** The endpoint-fitted K over-estimates damage at
intermediate error by an order of magnitude. This is a statement about the LAW's calibration —
it is **not** evidence about subspace choice, and must not be read as one.

### Corrected closure margins (this supersedes today's α=0 numbers)

The α=0 memo computed tolerances with the over-estimating `K = 350,427`. With the measured
`K_eff = 37,953` the tolerances loosen by √9.23 ≈ 3.04×:

| rung | bytes back | bar | tolerance @ K=350,427 (published) | tolerance @ K_eff=37,953 (corrected) | achieved | miss |
|---|---:|---:|---:|---:|---:|---:|
| rank-4, advisory | 14,662 | 1.5731× | 0.128% | **0.389%** | 25.14% | **64.7×** |
| rank-4, T4 | 14,662 | 4.7394× | 0.438% | **0.993%** | 25.14% | **25.3×** |
| rank-11, advisory | 1,514 | 1.0532× | 0.039% | **0.118%** | 4.23% | **35.8×** |

**The Euclidean ladder stays CLOSED — but the margin is 3× thinner than I published this
morning.** rank-4 missed by 196× in the α=0 memo's arithmetic; measured, it misses by 64.7×
(advisory) / 25.3× (T4). The closure survives on the loosest axis, which is the stronger
statement, so the verdict does not depend on which bar you price against.

## 3. Against the pre-registration — a near-miss on my own falsifier

Pre-registered (`FEED-ra2c-rank4`, written before the run):

- *"ratio ≈ 22,157× (within ~2×) ⇒ the pose metric tracks Frobenius here"* — **NOT MET** (9.23× off).
- *"ratio ≪ prediction (≥10× below) ⇒ Frobenius is the WRONG norm"* — **NOT MET, by one tick**
  (9.23× < 10×).

Neither branch fired. The result landed in the gap between them — which is precisely the
**third outcome** the operator's mid-run correction forced me to name in the erratum, before the
result was known: *Euclid tracks in ORDER but not in OPTIMUM.* Recorded as a near-miss rather
than rounded into the nearer branch.

**What the corpus already said, and what this adds.** The standing directive is that cosine /
Euclidean constructs are *"hardly ever optimal for our thing"* while remaining real signal, and
[[m65]] records a MEASURED Euclid-vs-Fisher cosine **sign flip**. This row is a third data point
in the same family, on a new surface: the Euclidean proxy got the DIRECTION right (rank-4 is
catastrophically over any bar — refused on both axes) and the MAGNITUDE wrong by 9.23×.

## 4. What this does and does not license about the pose-metric ladder

**Does:** it proves the two metrics differ materially on this object — enough to move a published
closure margin by 3×. Frobenius is a usable screen and a mis-calibrated ranker, exactly as the
corpus says.

**Does NOT:** it does not measure a Fisher-optimal rank-4 subspace, and the 9.23× must not be
transferred as though it were headroom from choosing a better subspace. Those are different
quantities; conflating them would be the metric-transfer error one surface over.

**The bar the pose-metric truncation must clear, stated honestly.** To become affordable at
rank-4 on the LOOSEST (T4) axis, a pose-metric-optimal rank-4 subspace must carry
`(2399.65 / 3.739) = 642×` less damage than the Frobenius-optimal one — i.e. **25.3× less
pose-relevant error at the same rank** under the quadratic form. On the advisory axis, 64.7×.

That is a large ask for a 12-dimensional object whose spectrum is flat (cond 17.32 on the
rendered field). I record it as **UNLIKELY on the evidence**, not as closed: Eckart–Young bounds
the Frobenius case only, so no theorem shuts the pose-metric door, and the honest cost of finding
out is now low (§5).

## 5. Cost correction — the successor is cheaper than the α=0 memo implied

Fisher curvature ↔ (−margin) Pearson **0.978**, so the margin field IS the Fisher surrogate; and
the `ms3`/`ms4` metric-custody bundle already carries the per-pair ≤6-dim pose quadratic plus
matched Fisher-vs-Euclid cosine and rel-norm rows. The pose-metric rung is a whitening of an
existing 600×12 coefficient matrix against producers that already exist — not a from-scratch
build.

**Named successor (owner MAIN, $0 + one advisory slot):** whiten the coefficient space by the
measured PoseNet quadratic, truncate to rank-r *there*, re-price against these same bars, and
report the dual-metric pair (never one alone, per [[m65]]).
**Falsifier:** if the pose-metric rank-4 damage also exceeds the 4.7394× T4 bar, the carrier
family is closed in BOTH metrics at INSTANCE scope and the remaining pose route is exclusively
the js1 joint line.

## 6. Two products beyond the refusal

1. **d_seg is now confirmed exactly invariant to frame_0 carrier edits on a THIRD independent
   treatment** (α=0, α=1, rank-4 — identical to 8 significant figures each time). Every carrier
   question is a pure (pose, rate) trade; this is no longer an argument from SegNet's
   `x[:, -1]` slice, it is measured three ways.
2. **The rank-truncation harness is defect-checked in both directions** — the α=1 control renders
   with the UNTRUNCATED coefficients (600/600 byte-identical) while the emitted frames carry the
   truncation, so a broken treatment cannot hide behind a passing control.

## 7. Honest limits

- Advisory axis, single instrument. The T4 column assumes the d_pose RATIO transfers; that is a
  weaker assumption here than at α=0, since the perturbation is 146× smaller. Not a score claim.
- `K_eff` is a one-point recalibration at a single error magnitude. It establishes that K varies
  along the ladder; it does not establish the functional form of that variation. Two points fix a
  line, not a curve — same caution the α=0 memo owed and now owes twice.
- `p = 2` is retained as the principled exponent (d_pose is a quadratic form). The 9.23× could
  alternatively be read as `p ≠ 2`; distinguishing requires a third rung.
