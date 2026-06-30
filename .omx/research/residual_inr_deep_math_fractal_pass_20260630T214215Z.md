# Residual-INR hybrid — FINAL recursive fractal deep-math pass (pre-GPU)

**UTC** 20260630T214215Z · `[macOS-CPU/numpy advisory · deep-math review artifact · NON-PROMOTABLE]` ·
**pointer 0.19110 UNMOVED.** CPU-only, NO GPU, NO launch; the live n600 daemon was untouched (read-only).
Author = independent deep-math reviewer. Every verdict is a MEANS; the only END is a byte-closed residual-INR
exact row < 0.19110 (CPU/CUDA, never MPS).

**Scope.** The design under review: a residual INR trains on `GT_partition − bulk_through_R` (the Lane+Movable
residual annulus), composed at decode as `where(isin(bulk_warped_label,{Lane,Movable}), INR_rgb, bulk_rgb)`;
config `--hidden-dim 48 --mod-dim 16 --epochs 1500 --curriculum`; bulk = per-class stratified pose-warp
(Road=ground-homography, hood=identity, sky=rotation-only) + lane geometry; byte-close int8+brotli.

---

## HEADLINE (the one finding that matters)

**The residual config is INHERITED, not derived. `--mod-dim 16 --hidden-dim 48` were chosen as "half of the
full-partition 96/32" (a smaller-AREA → smaller-INR heuristic). The deep-math says the residual is NOT
lower-dimensional than the full partition — it is the SAME-or-HIGHER dimensional HARD part.** The composition
removes the EASY, low-rank, low-frequency mode (the rank-8 ego-homography → Road bulk) and hands the INR the
HIGH-rank, HIGH-frequency tail (the ~8–13-dim nonlinear lane orbit + movables). Sizing that tail SMALLER than
the full partition is backwards. Two compounding errors:

1. **mod-16 is at/below the under-embedding red line for the residual.** Whitney floor for the lane orbit alone
   (nonlinear ID ~8–13, MEASURED: AE-knee 8 / MLE 13) is 2m+1 ≈ **17–27**. mod-16 is below 17 *before* movables
   are added. The certification doc itself calls mod-16 "the under-embedding red line" — for the *full*
   partition (ID ~9). The residual is not easier.
2. **The residual INR paints in IMAGE space**, so its per-pair code must re-encode the ego-motion the bulk
   warp ALREADY paid for (the lane image-position rides the same ground-homography as Road). MEASURED (L3,
   FEED-jm): the image-space lane is rate-EXPENSIVE (~65 KB/600, adjacent-frame IoU only 0.284). The derived
   fix — **canonicalize the residual to the ground/IPM frame and share the stored pose** — collapses the code
   to its TRUE survival dimension (~2–4) AND removes the image-space rate. Until that build lands, the
   image-space residual must be sized at the lane-orbit Whitney floor (mod ≈ 19–21), not 16.

**Top-3 refinements the binding run should adopt** (derivations below):
- **R1 [highest EV, launch-blocking-ish]: `--mod-dim 16 → 19–21`** + run a $0 residual-ID measurement first
  (TwoNN/MLE on the residual descriptor) to confirm. mod-16 risks under-embedding the residual manifold.
- **R2 [high EV, structural]: SPLIT the residual** — STORE movables as a multi-body codec (~0.9–2.7 KB, reaches
  d_seg ~0.0008; the movables memo's explicit verdict), let the INR carry ONLY lane-survival. Folding movables
  into the image-space INR is RD-suboptimal AND mis-masked (the ego-warp-derived movable mask lags true f1
  position).
- **R3 [high EV, curriculum]: re-derive the epochs/curriculum for the residual** — the residual has NO smooth
  bulk to descend, so the CE warmup is wasted; weight toward tau/l7/Muon and EARLY-STOP at the residual knee
  (1500 is inherited from a full-partition guess and likely over-trains past the intrinsic-jitter floor).

---

## Q1 — Residual manifold dimension: is mod-16/hidden-48 DERIVED for the residual? → **REFINE-TO mod-19–21 + NEEDS-MEASUREMENT**

**Derivation (set / info-theory / geometry lenses).**

The full-partition nonlinear intrinsic dim is MEASURED ~9 (band-local motion-invariant descriptor 200×95:
PR=5.9, PCA-knee=10, TwoNN=13.6, MLE=9.6; GT-partition manifold TwoNN≈19.65 on the raw 21-frame cover). The
certification correctly applies Whitney (a smooth m-manifold embeds in 2m+1 linear coords): m≈9 → mod-floor 19,
m≈13 → 27. It explicitly flags **mod-16 as "BELOW Whitney floor → under-embedding risk"** for the full
partition.

Now decompose the partition by the MEASURED rank/staticness structure (FEED-it/G8):
- rank-8 = 95.6% of cross-pair variation = the **ground-plane homography orbit** (8 = dim of the planar
  homography group): Road(0), sky/Undriv(2), lane-POSITION all ride it.
- The 4.4% tail (rank-8→16 = 3%, plus a flat ~1.4% beyond) = **movables (independent motion) + lane-survival**.

The composition assigns **BULK = {Road, Undriv, MyCar}** (the deterministic warp captures the rank-8 homography
+ static classes), **LEARN = {Lane, Movable}** (the INR). So the INR's target = the partition MINUS its
lowest-dimensional, most-compressible mode. What is LEFT:

- **Lane (entirely in LEARN, image space):** MEASURED nonlinear lane-orbit ID = **AE-knee 8 / MLE 13**; the
  linear "store-the-flips" sidecar is rank **53/60 ≈ full-rank** (NO-GO ×3) → the lane's compressibility is
  *nonlinear* and ~8–13-dim. Critically, the lane "~8-dim orbit" IS the homography orbit (FEED-it: "same
  object"). Because the bulk does NOT place the lane (Lane is LEARN-tier, not BULK), **the INR's per-pair code
  must itself carry the full 8-dim lane image-position orbit** — i.e. the SAME ego-motion the bulk warp
  computed for Road, re-encoded.
- **Movables (in LEARN):** globally high-rank (the rank-8→16 gap), per-object low-rank (deg-2 trajectory, a
  handful of coeffs); ~3 objects/frame × ~6-DOF instantaneous ≈ up to ~18 per-pair DOF.

**Therefore the residual's intrinsic dim ≈ the full partition's (≈ 9–13), NOT lower** — it contains the full
partition's single hardest mode (the lane orbit) plus movables, having shed only the EASY Road homography.
Whitney floor 2m+1 ≈ **19–27**.

**Verdict: mod-16 is INHERITED ("half of 32"), not derived, and is at/below the residual's under-embedding red
line. REFINE-TO mod-19–21 for the image-space residual; the deeper fix (Q6) is canonicalization, which would
then JUSTIFY a small mod (~4–8) honestly.** $0 NEEDS-MEASUREMENT before launch: run TwoNN/MLE on the residual
descriptor (`GT_partition − bulk_through_R` band-local on the Lane+Movable annulus, the existing
`residual_target.npz` is the input) — this is the missing number; it has never been measured (the cert measured
the full partition, not the residual).

**hidden-48 (separate, possibly worse-derived).** hidden is the trunk's spectral-unfolding capacity (params ≈
hidden²·n_hidden, so 96→48 is a **4× param cut**, far more aggressive than the cert's own 96→88 down-arm). The
heuristic is "residual is sparse (small area) → less trunk." But the residual is **all-boundary, all
high-frequency**: MEASURED (L2, FEED-fs) the lane residual after the centerline is a "HIGH-FREQUENCY per-row
width residual… the ragged ±1px detail = the trained generator's job." A smooth INR's spectral bias is already
the binding difficulty at the annulus; halving the trunk while the content is PURE high-frequency boundary
(no smooth interiors to coast on) cuts capacity exactly where it is scarcest. **Verdict: hidden-48 is the
RISKIER under-derived knob.** Sparse-area ≠ low-complexity. Recommend hidden ≥ 64 for the image-space residual,
or measure the residual's spatial spectral content; do not bank the 4× cut un-measured. (Note the rate
asymmetry that makes this affordable: hidden is base-weight rate, mod is per-frame-code rate ×600 — so bytes
favor cutting mod over hidden, the OPPOSITE of this config's instinct.)

---

## Q2 — Descent dynamics of the residual sub-problem → **REFINE the curriculum + epochs (don't inherit)**

**Derivation (calculus / physics / optimization-geometry lenses).**

The full d_seg descent is critical-slowing power-law near a rate-distortion topological transition
(Agmon–Tishby ISIT 2021; our curriculum is deterministic annealing, Rose 1998 F=D−T·H). The residual
sub-problem differs in three structural ways:

1. **No smooth-bulk regime.** The full problem's fast initial drop comes from nailing the easy bulk interiors
   (the rank-8 homography classes). The residual STARTS at the slow tail — it IS the hard annulus from epoch 0.
   The curriculum (CE→tau→l7→Muon) is a homotopy of relaxations designed to descend from a smooth start; CE
   (smooth convex surrogate) has little to do on a target that is all sharp boundary. **CE warmup is largely
   wasted for the residual; weight epochs toward tau/l7/Muon (the sharpening the annulus needs).** The proven
   full-partition schedule (CE saturates ~ep275, tau ~ep450) does NOT transfer.

2. **Rougher, multi-modal landscape.** The residual's "transitions" are not one clean RD transition — they are
   the ~2700 dashed-lane birth-death events / 600 frames + per-movable boundary locks (MEASURED). This is a SUM
   of many small critical-slowings → a rough landscape favoring per-stage RE-TREATMENT (the "different stages
   need different treatment" discipline — margin-engage spike-skip, moment reset at boundaries) and the
   root-tracking anneal, more than a single long Muon tail.

3. **The descent plateaus at an IRREDUCIBLE floor, not at zero.** MEASURED (W6, FEED-kb): static-hood
   (warp-free) = 32% of bulk flips → a large fraction of the residual is INTRINSIC texture-dependent per-frame
   SegNet jitter (pixels where SegNet's OWN margin ≈ 0 — a coin-flip). No smooth low-dim INR fits coin-flips at
   any mod/hidden. So the residual descent is power-law-toward-an-intrinsic-floor, and **1500 epochs likely
   over-trains past the residual knee** (wasted wall-clock; optimality-triad violation). The n200-DOE
   discipline applies verbatim: set epochs by the MEASURED saturation knee + early-stop-on-plateau, do not fix
   1500.

**Verdict: REFINE.** Same critical-slowing CLASS, but (a) shorter/skipped CE, (b) per-stage re-treatment for
the rough landscape, (c) epochs set by the residual knee not inherited 1500. NEEDS-MEASUREMENT: the residual's
own saturation curve (a short residual pilot gives the per-stage knees, exactly the n200-DOE role).

---

## Q3 — Rate-distortion of the residual → **REACHABLE ONLY IF the residual is SPLIT (movables→store) and canonicalized; as-configured it is RD-suboptimal**

**Derivation (RD / entropy lenses).**

S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489. With pose on the stored sidecar (√(10·3.4e-5)=0.0184) and
store-rate ~0.0060 (≈9 KB), the budget at residual-bytes B is:
- sub-0.19: 100·d_seg + 25·B/37.5M < 0.1656 → at B=30 KB (0.0200), **d_seg < 1.46e-3**; at B→0, d_seg < 1.65e-3.
- sub-0.15: 100·d_seg + 25·B/37.5M < 0.1256 → at B=30 KB, **d_seg < 1.06e-3**; at B=50 KB (0.0333), d_seg < 0.92e-3.

From the bulk floor d_seg ~0.0185 (W11, MEASURED, k=0 reproduces 0.01851 exactly), the residual INR must deliver
a **13–18× reduction** at ≤30–50 KB. The residual's RD function SPLITS by term:

- **Movables:** the multibody memo MEASURED a STORE path reaching d_seg ~0.0008 at **0.9–2.7 KB** (K=50→0.9 KB,
  K=150→2.7 KB), vs a 40 KB break-even — and a WARP/predict path hard-floored at 0.00082 (per-object placement
  error, irreducible). **Explicit verdict there: STORE, do NOT predict.** This design folds movables into the
  learned INR (predict), which is the dominated branch AND costs INR capacity/bytes for what a 1–2.7 KB template
  store does better. **RD-suboptimal as configured.**
- **Lane-survival:** the binding term (GAP2). Its RD is hurt twice: (i) IMAGE-space → ~65 KB/600 iid (L3,
  rate 0.043) because ego-motion moves the lane every frame — the INR amortizes better than iid but its code
  still pays the ego-motion the bulk already bought (Q6 redundancy); (ii) the intrinsic-jitter floor (W6) —
  a chunk is irreducible (naive margin-keyed dither = 177,926 B/600 = rate 0.118 = PR95-scale to store; "free
  bulk + tiny trained lane" thesis DEAD for warp+naive-store). The OPEN question — does the lane-survival
  manifold have low-rate exploitable structure the trained INR can capture (dash periodicity, openpilot deg-3
  centerline prior, the ragged ±1px detail of L2) — is GENUINELY only settled by the GPU run.

**Verdict: the residual RD-optimum is reachable, but the as-configured single-image-space-INR is NOT on it.**
The reachable path is: movables → cheap store codec (≤2.7 KB, d_seg→~0.0008); lane → canonicalized INR (ground
frame, share pose) attacking ONLY the survival residual. The design's own honest "OPEN QUANTITY" is correct
that the lane-survival efficacy is the wall; the deep-math adds that the design's chances materially improve if
the residual is split and canonicalized BEFORE the burn, rather than discovering the suboptimality from a null
result.

---

## Q4 — Warp geometry optimality → **OPTIMUM-CONFIRMED (well-grounded) with two minor depth-stratification notes**

**Derivation (geometry / physics lenses).** The per-class stratification follows the projection law: ROTATION
(roll/pitch/yaw) → depth-INDEPENDENT global warp (hits all depths equally); TRANSLATION → depth-DEPENDENT
parallax ~|t|/depth (near big, far ~0). So:
- **MyCar/hood = identity** — rigid to camera, IoU 0.994 → correct (0-byte clamp #139).
- **sky/Undrivable = rotation-only KRK⁻¹** — depth→∞, parallax→0 → correct for the sky majority.
- **Road = ground-plane homography(pose)** — on-plane, 8-DOF from ego (R,t)+EON intrinsics (fx=fy=910, cx=582,
  cy=437, h=1.22 m); MEASURED +15–17% d_seg, calibration CLOSES → correct.
- **Lane/Movables = learned residual** — lane rides the homography but survival is annulus; movables off-orbit.

**Two notes (minor, both land in the residual = what the INR is for):**
1. **Class-2 (Undrivable) conflates sky (∞, rotation-only correct) with NEAR-field undrivable** (guardrails,
   barriers, off-road structures at finite depth, which DO have parallax under forward motion). Rotation-only
   slightly under-warps the near part. The correct stratification is per-DEPTH; per-CLASS is a proxy. IoU 0.995
   says the error is small (most class-2 is far/sky), so acceptable, but it is an approximation, not exact.
2. **Road assumes a PLANAR ground.** Real roads have grade/crest/banking; on a 60 s drive the planar homography
   leaves a slope residual. Second-order; calibration "closes" the dominant term.

**Plus the W8 caveat (important framing, not a flaw):** MEASURED, the d_seg-optimal warp ≈ near-identity
(s_t≈−0.0014) — the SegNet argmax is flow-ROBUST and does NOT carry pose; d_pose-optimal warp (s_t≈+0.16)
WRECKS d_seg 7×. So the warp's job is NOT to lower d_seg directly — it is to (a) get the deterministic bulk's
GEOMETRY right (a RATE move: a correct bulk → a smaller residual) and (b) feed d_pose via the stored sidecar.
The warp should be calibrated to MINIMIZE the geometric residual (≈ geometric scale), NOT to minimize d_seg.
The design uses the warp to render the bulk — consistent. **No class is mis-assigned; verdict OPTIMUM-CONFIRMED
with the two depth-stratification notes logged as residual contributors.**

---

## Q5 — Byte allocation (int8+brotli) → **REFINE: not entropy-optimal; the CODE wants temporal-AR / low-rank coding**

**Derivation (entropy / signal lenses).** int8+brotli per-tensor is the SAFE base (≈ PR100 level) but leaves
two measured structural wins on the table:

1. **The CODE is the dominant counted payload** (per-pair modulation, ~rate-linear in mod×n_pairs; reviewer
   measured int8+brotli code = 33,553 B @ mod-32). It is TEMPORALLY COHERENT — movables move smoothly, and if
   canonicalized (Q6) the lane code is near-static along the ego-path. **brotli on a flat int8 stream is
   STRUCTURALLY BLIND to (a) temporal autocorrelation across frames and (b) matrix low-rank** (FEED-fl: the
   exact lens brotli misses). The principled coders: temporal-delta + raw LZMA (PR95 L24/L25), OR low-rank
   factorization (store U,V + AR residuals along the ego-coherent code path — the nuclear-norm ‖C‖_* penalty
   #110/A6, exploiting the MEASURED eff-rank contraction 25.8→21.9). This is the bigger of the two wins.
2. **The WEIGHTS** want the PR101/PR103 ladder over raw brotli: per-tensor byte-maps (zig/negzig/twos/off),
   storage perms, split brotli streams, and range/arithmetic coding (constriction.Categorical, PR103 silver) on
   the high-entropy tensors. For a small hidden-48 trunk these are ~hundreds of bytes each — proportionally
   smaller than the code win but free given offline compute.

**Reverse-waterfill** (allocate bits per-coefficient by d_seg sensitivity, the closed-form KKT allocator) is
the principled per-coefficient allocator and dominates uniform int8 quantization — but the structural prize is
modeling the temporal/low-rank correlation that brotli cannot see, not the per-coefficient bit-depth.

**Verdict: REFINE.** int8+brotli is fine for the FIRST byte-close (de-risks the pipeline) but is NOT the
entropy-optimal coder; the RD-binding refinement is temporal-AR / nuclear-norm low-rank coding of the per-frame
code (the dominant payload), then the PR101 weight ladder. Each is a MEASURED byte delta at byte-close time.

---

## Q6 — Composition: does the hybrid compose optimally? → **NO — three cross-terms the action misses**

**Derivation (set / algebra / VCM lenses).** The composition is
`composed = where(mask, INR, bulk)`, mask = `isin(bulk_warped_label, {Lane, Movable})` (+optional dilate),
scored as `argmax(SegNet(composed))`. Three interactions the action S_τ does not optimize over:

1. **The mask is bulk-derived and FIXED — the INR cannot move the seam.** The INR optimizes its RGB to win
   d_seg GIVEN the seam, but if the optimal partition wants the boundary ELSEWHERE (a pixel the bulk labeled
   Road that should be lane-controlled), the hard mask forbids it. Dilation is a crude widening, not a fix. A
   more optimal composition is a SOFT / learnable composition weight (or letting the INR also adjust the
   override region). **Mis-placed seams are an uncaptured cross-term.**

2. **Image-space residual ⇒ REDUNDANT ego-motion encoding (the deepest one).** The bulk warp computes ego-motion
   for Road via the ground-homography. The lane (LEARN-tier, image space) rides the SAME ground plane, so the
   INR's per-pair code must RE-encode that same ego-motion to track the moving image-space lane (L3: image-space
   lane IoU 0.284 adjacent frames, ~65 KB iid). The action treats bulk and residual as independent terms, but
   they SHARE the pose. **Derived fix: canonicalize the residual INR to the ground/IPM frame and warp it by the
   ALREADY-STORED pose (FEED-iu/iv canonicalization).** Then the lane is ~static in canonical coords → the
   per-pair code drops from ~8 (image orbit) to ~2–4 (survival/dash phase) → mod can be honestly SMALL AND the
   65 KB image-space rate vanishes. This is the single structural change that would make the design's
   "size-down" instinct CORRECT instead of under-derived. (It is a v2 BUILD — a coordinate-warp on the INR
   input — not a launch knob; for the launch-now run, image-space mod must be ≥19–21 per Q1.)

3. **d_seg is NOT additive across the seam (SegNet receptive field couples bulk⊕INR).** The byte-accounting
   narrative ("bulk floor 0.0185 − INR gain → 6e-4") is additive fiction: SegNet's conv stem mixes bulk-RGB and
   INR-RGB within its receptive field at the seam, so a clean bulk interior pixel near the seam can FLIP when
   the INR paints adjacent. This is CAPTURED in the loss IF training scores the composed render (it does — good,
   confirmed) and the inflate parity is bit-exact (confirmed). So the action is faithful; only the human
   accounting is heuristic — the realized composed d_seg is the authority (the design correctly defers to the
   GPU run). **No bug; flag the additive narrative.**

4. **Movables mask MIS-PLACEMENT (a real correctness risk, reinforces R2).** The mask = `isin(bulk_warped_label,
   {…,Movable})`, but the bulk ego-warp places f0's movables at the WRONG f1 location (movables are OFF the
   ego-orbit — that is precisely why they are residual). A fast lead car's mask lags its true position → the
   INR cannot fix the uncovered pixels (and paints stale ones). **Movables need their OWN placement (the
   multibody store, R2), not the ego-warp-derived mask.**

**Verdict: the composition does NOT compose optimally.** The action is faithful where it counts (trained +
scored + inflated on the composed render, bit-exact parity), but it MISSES the seam-placement freedom (#1), the
ego-motion redundancy (#2, the big one — canonicalize), and the movables mis-mask (#4). #3 is fine (faithful
loss, heuristic narrative).

---

## SUMMARY TABLE

| Q | Topic | Verdict | Derived refinement |
|---|---|---|---|
| 1 | residual manifold dim | **REFINE / NEEDS-MEASUREMENT** | mod 16→**19–21** (lane-orbit ID 8–13 → Whitney 17–27; mod-16 under-embeds). hidden 48→**≥64** (residual is all-high-freq boundary; 4× cut un-measured). Measure residual ID ($0). |
| 2 | descent dynamics | **REFINE** | shorter/skip CE; per-stage re-treat (rough multi-modal landscape); epochs by residual KNEE not inherited 1500; expect plateau at intrinsic-jitter floor (W6). |
| 3 | residual RD | **REFINE (suboptimal as-configured)** | SPLIT: movables→store codec (≤2.7 KB, d_seg→0.0008, the memo's verdict); INR carries lane-survival only. |
| 4 | warp geometry | **OPTIMUM-CONFIRMED** | minor: class-2 conflates sky(∞)/near-undrivable(parallax); Road planar-ground; both small, land in residual. Calibrate warp to geometric-residual (W8), not d_seg. |
| 5 | byte allocation | **REFINE** | int8+brotli OK for first close; add temporal-AR / nuclear-norm low-rank CODE coding (brotli blind to temporal+low-rank, FEED-fl) + PR101 weight ladder. |
| 6 | composition | **NOT OPTIMAL (3 cross-terms)** | canonicalize residual to ground frame + share pose (kills ego-motion redundancy, the big win); soft/learnable seam; movables own placement; additive-d_seg narrative is fiction (loss is faithful). |

## means ≠ ends
All numbers ADVISORY (`[macOS-CPU/numpy advisory · NON-PROMOTABLE]`); pointer **0.19110 UNMOVED**. This pass
certifies the residual-INR config is **inherited, not derived**, and gives the derived refinements. The exact
row is moved only by a byte-closed residual-INR eval (CPU/CUDA, never MPS). The single highest-EV $0 action
BEFORE the burn: measure the residual sub-manifold ID (TwoNN/MLE on `residual_target.npz`) to confirm mod-19–21
and de-risk the under-embedding. Anchors: `n600_final_config_cert_arch_basis_optimizer_20260630T191137Z`
(full-partition cert — the residual was never separately certified), `residual_only_trainer_mode_landed_*`,
`movables_multibody_residual_*` (store-not-predict), DAG FEED-it/iu/iv (rank-8 homography + canonicalization),
FEED-fs/jm/kb/lj/lk/ll (L2/L3/W6/W8/W11), CLAUDE.md witness-capstone + "different stages need different
treatment" + rule-118.
