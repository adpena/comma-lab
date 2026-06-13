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
