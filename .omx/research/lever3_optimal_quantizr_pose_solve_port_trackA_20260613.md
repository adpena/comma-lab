# The optimal port of Quantizr's pose-solve into Track A (Lever-3 v2 design)

**Operator question (2026-06-13):** *"what is the optimal port of quantizr pose solve for track A?"* + *"draft that design memo."*
**Authority:** design memo; all numbers `[contest-CPU advisory]` until byte-closed exact eval. Grounded in
Quantizr PR#55 ACTUAL code (`reverse_engineering/quantizr_pr55/inflate.py`) + our HNeRV
(`src/tac/torch_vehicle/pose_film.py`).

## Quantizr's pose solve (what we're porting)
1. **STORE** the 6 GT pose scalars per pair (range-coded ~1 KB) — Wyner-Ziv side info; pose is stored, not
   reconstructed from pixels.
2. `pose_mlp = Linear(6→cond) → SiLU → Linear(cond→cond)` builds a conditioning embedding.
3. A **dual-head** generator on a shared trunk: **`frame1_head` is pose-FiLM-conditioned**;
   **`frame2_head` is static** (no pose). d_seg is carried by a SEPARATE stored mask; d_pose by the
   pose-FiLM on frame1. The pair's relative motion = how the pose-conditioned frame1 differs from the
   static frame2.
4. The FiLM is **residual-contained**: `x = act(residual + (conv(x)·(1+γ) + β))` — the identity path
   anchors the output, so even unbounded γ/β are a *bounded correction*, not a free modulation.

Quantizr's 0.33 is RATE-dominated (0.200 = ~300 KB grayscale masks.mkv). Distortion (seg 0.00061,
pose 0.00051) is excellent **and stable** because the two axes live in different heads.

## The key realization: our HNeRV ALREADY has the dual-head
`pose_film._forward_with_film` (= the vendored HNeRV forward):
```
x = stem(z) → [v1 FiLM here, WRONG] → sin → cascade(blocks/skips ×6) → refine
f0 = sigmoid(rgb_0(x)) * 255     # frame_0
f1 = sigmoid(rgb_1(x)) * 255     # frame_1
return stack([f0, f1], dim=1)    # (B, 2, 3, 384, 512)
```
`rgb_0` and `rgb_1` are **two separate final heads** on the shared post-cascade feature `x` — structurally
identical to Quantizr's `frame1_head` / `frame2_head`. And the contest **SegNet reads only the LAST frame
`x[:, -1] = f1`** (verified: `driver._segnet_logit_margin_map` / SegNet reads the decoded last frame);
**`f0` is seg-invisible.** PoseNet reads BOTH frames (the pair → relative pose).

## THE OPTIMAL PORT (Lever-3 v2)
Put the pose-FiLM **on the `rgb_0` head ONLY (the seg-invisible frame), as a residual — leave `rgb_1`
(the SegNet-read frame) FiLM-clean:**
```
x = stem(z) → sin → cascade → refine          # SHARED feature, NO FiLM on the stem (v1 bug removed)
cond = pose_mlp(stored_pose[idx])             # Linear(6→C)→SiLU→Linear(C→C)
x0   = x + film_resid(x, cond)                # RESIDUAL FiLM, zero-init → identity at init
f1   = sigmoid(rgb_1(x)) * 255                # CLEAN seg frame — carries d_seg, FiLM-free
f0   = sigmoid(rgb_0(x0)) * 255               # pose-conditioned frame — carries the f0→f1 motion
```
Mapping to Quantizr: **`f1` = "static frame2"** (seg-clean reference, carries d_seg) · **`f0` =
"pose-FiLM frame1"** (carries d_pose). `film_resid` mirrors Quantizr's residual block:
`film_resid(x,cond) = (proj/conv(x))·(1+γ) + β` with the branch **zero-init** so `film_resid≡0` at init →
`f0` renders bit-equal to the vendored `rgb_0(x)` → the **byte-identity / basin-resume contract holds**.

## Why this is optimal (3 properties, all the others fail ≥1)
1. **Full d_seg/d_pose decoupling at ZERO extra rate.** `f1` (SegNet's frame) has no FiLM → pose-FiLM
   variance *physically cannot* perturb d_seg. We get Quantizr's decoupling **without** storing masks
   (we render `f1` clean; the only pose payload is the 6 scalars ~1 KB). → beats Quantizr on rate
   (89 KB vs 300 KB) AND matches its stability.
2. **Residual containment** → bounded pose perturbation (kills the v1 stem-FiLM variance: best 0.00043 but
   spikes 0.0046).
3. **High-res, head-local injection** (post-cascade, 384×512, on one head) → minimal leverage vs v1's
   6×8 shared-stem max-leverage.

| Design | d_seg/d_pose decoupled? | stable (residual)? | extra rate | verdict |
|---|---|---|---|---|
| **v1 (stem FiLM on shared x)** | NO (couples) | NO (γ·x+β) | 0 | unstable, couples |
| later-shared-cascade-stage FiLM | NO (still shared) | YES | 0 | half-fix (stability only) |
| Quantizr (stored mask + dual head) | YES | YES | **+~210 KB masks** | rate regression for us |
| **Lever-3 v2 (rgb_0-head residual FiLM)** | **YES** | **YES** | **0** | **OPTIMAL for Track A** |

## How it composes with the other Track-A levers (the stack)
- **Lever-2/5 (d_seg)** optimize `rgb_1` (the clean seg frame) via the soft-cosine surrogate @ T=0.3 (gate:
  1.58× CE). Now that `f1` is FiLM-free, the seg levers have a clean target — no FiLM noise to fight.
- **Lever-3 v2 (d_pose)** optimizes `rgb_0` via the residual pose-FiLM. Decoupled from `f1`.
- **Score-domain Lagrangian** (100·seg_l + 1·√(10·pose_l)) unchanged (exact contest weights); the
  decoupling means the two terms' gradients land on different heads → no cross-axis interference.
- **ITEM-B variable-codec** (rate −0.005) + finishing-kit residual (−0.003) compose on top (weights/bytes).

## Wire-in + validation (the build contract)
- New wrapper variant in `pose_film.py` (or a v2 module): FiLM moved from stem → `rgb_0` residual; `rgb_1`
  path untouched. Default-OFF byte-identical (FiLM branch zero-init AND the driver flag off → vendored).
- Tests: (a) identity-at-init bit-equality (both off AND on-at-init); (b) `rgb_1` output is INVARIANT to
  the stored pose (the decoupling proof — changing pose6 must NOT change f1); (c) `f0` DOES change with
  pose6 (the FiLM is live); (d) residual containment bounds the f0 perturbation.
- $0 real-scorer A/B: v2 (rgb_0-head) vs v1 (stem) — measure **d_pose variance** (std across steps/slices)
  AND **d_seg invariance to pose** (v2's d_seg must be pose-independent; v1's is not). v2 should show lower
  d_pose variance AND d_seg fully decoupled.

## v3 / Track-B (noted, not now)
A dual-head with a CHEAP stored seg structure (contour/boundary-MDL codec, NOT 300 KB ffmpeg) + pose +
thin RGB render — full Quantizr decoupling with an even-cheaper-than-89 KB seg carrier. Higher effort;
the rgb_0-head port above gets ~all the decoupling benefit at zero new architecture, so v3 is a later bet.

## Bottom line
The optimal port is **not** "FiLM at a later cascade stage" — it's **the residual pose-FiLM on the
`rgb_0` (seg-invisible) head, leaving `rgb_1` FiLM-clean.** Our HNeRV's two-head split makes this a
near-free adoption of Quantizr's decoupled-dual-head pose-solve, at our 89 KB rate instead of his 300 KB.

## Landing record — Lever-3 v2 BUILT + MEASURED (2026-06-13, partner leverD3-v2-rgb0)

**Module:** `src/tac/torch_vehicle/pose_film_v2.py` — `PoseFiLMHNeRVWrapperV2` (residual
pose-FiLM on the `rgb_0` head ONLY; `rgb_1` FiLM-clean) + `_PoseCondMLP`
(`Linear(6→C)→SiLU→Linear(C→C)`) + `_RGB0ResidualFiLM` (`proj(x)·(1+γ)+β`, zero-init
proj+beta → `film_resid≡0` at init) + `inflate_film_decoder_v2` (numpy-portable). Reuses the
v1 additive pose-section grammar + `_FiLMEvalDecoder` cursor adapter verbatim.

**Tests:** `src/tac/torch_vehicle/tests/test_pose_film_v2_rgb0.py` — 19 NO-FAKE tests, all green;
v1 regression (13) green; ruff clean. Covers: identity-at-init bit-equality (off + on-at-init +
idx=None + film_resid≡0); f1 BIT-invariant to pose6 (init AND trained — the decoupling); f0 DOES
change with pose6 (NO-FAKE); f1==vendored rgb_1 exactly; f1-only loss → ZERO grad on FiLM (gradient
decoupling); residual containment bound; pose_mlp/film_resid shapes; eval-cursor misroute control;
byte-closed inflate v2 round-trip.

**MEASUREMENT 1 — d_seg DECOUPLING (real frozen SegNet, base_ch20 forkpoint, contest-CPU advisory):**
contest d_seg = per-pixel argmax-flip rate of f1's SegNet mask (verified `upstream/modules.py:108`
`x = x[:, -1, ...]` → f1 IS the SegNet frame; `:112` = argmax-flip). Vary stored pose (GT vs +3σ scramble):
- **v1 (stem FiLM): d_seg 0.198 → 0.299, delta = 0.100** (pose COUPLES into d_seg — the bug).
- **v2 (rgb_0 head): d_seg 0.003459930419921875 → 0.003459930419921875, delta = 0.0 EXACTLY** (decoupled).
Verified again directly on the real base_ch20 EMA decoder: f1 bit-invariant to pose with a TRAINED FiLM;
f0 DOES change (max|Δf0|=230.5). The decoupling is STRUCTURAL, not an init artifact.

**MEASUREMENT 2 — d_pose VARIANCE (v2 vs v1, identity-init, 2 slices, contest-CPU advisory):**
At **lr=1e-3** (the well-conditioned operating point), v2 wins on mean, std, final, AND convergence speed:
| slice | arm | mean | std | final | min |
|---|---|---|---|---|---|
| 0 (off 0) | v1 stem | 8.09e-4 | 5.54e-4 | 1.69e-4 | 1.69e-4 |
| 0 (off 0) | **v2 rgb0** | **5.92e-4** | **4.80e-4** | **1.60e-4** | **1.60e-4** |
| 1 (off 6) | v1 stem | 6.93e-4 | 4.53e-4 | 1.35e-4 | 1.35e-4 |
| 1 (off 6) | **v2 rgb0** | **3.66e-4** | **4.45e-4** | **1.00e-4** | **1.00e-4** |
v2 = 27%/47% lower mean, lower std, lower final on both slices; converges by step ~3 vs v1 still
descending at step 9.

**HONEST caveat (LR sensitivity — NOT in the original prediction):** at the aggressive **lr=1e-2**,
v2's full-resolution (384×512) `proj` 1×1 conv OVERSHOOTS transiently (step1 d_pose spikes to 2.4e-2
from zero-init's large first Adam step) before converging — WORSE early variance than v1's smaller
6×8 stem FiLM at that LR. So the stability win is real but LR-conditioned: v2 needs a moderate LR
(≤1e-3) to realize the residual-containment benefit; at high LR its head-local high-res leverage
hurts the transient. Recommendation for #116: use lr≤1e-3 for the v2 FiLM params (or a warmup/lower
LR group), where v2 dominates v1 on every metric.

**3-clean adversarial review (SEALED):** R1/R2/R3 all clean across the 3 lenses —
Lens1 identity-at-init bit-equal verified on REAL base_ch20 (not just test base_ch8);
Lens2 decoupling REAL + NO-FAKE (f1 bit-invariant + f0 varies + upstream source confirms f1 IS the
SegNet frame + GT targets verified real, 100% nonzero pose, all 5 seg classes);
Lens3 variance reduction real on 2 slices at lr=1e-3 (+ honest lr=1e-2 caveat).

**#116 handoff (driver wire-in for the from-0 run):** the driver currently imports v1
`PoseFiLMHNeRVWrapper` + hard-codes the `pose_film.` key prefix in the EMA split (`driver.py:1068-1082`).
To run v2: (1) import `PoseFiLMHNeRVWrapperV2`; (2) split on the `pose_mlp.`/`film_resid.` prefixes;
(3) use `inflate_film_decoder_v2`; (4) set the FiLM param LR group ≤1e-3. Default-OFF byte-identity is
preserved (v2 is identity-at-init + the driver flag gates the wrapper).

Authority: torch-CPU TRUSTED, NO MPS. `[contest-CPU advisory]` NON-PROMOTABLE until byte-closed exact eval.
Artifacts: `.omx/research/lever3_v2_variance_slice0_lr1e3_20260613.json` +
`.omx/research/lever3_v2_variance_slice1_lr1e3_20260613.json` +
`.omx/research/lever3_v2_variance_slice0_20260613.json` (lr=1e-2 spike evidence).
