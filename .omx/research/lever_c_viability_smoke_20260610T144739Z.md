# Lever-C viability smoke — VERDICT (task #62)

**Subagent:** `task62_lever_c_viability_smoke`. **Authority of every number below:** `[local CPU-torch
advisory]` — exact upstream PoseNet/SegNet (`DistortionNet`) on CPU, GT decoded via
`upstream/frame_utils.yuv420_to_rgb` ONLY, S/terms recomputed from components (the rounded field lies).
`[macOS-MLX research-signal]` for the conv-decoder forward (numpy↔torch RGB parity 1.0 within 1 LSB).
**NOT** the contest 600-sample harness → non-promotable per the authority ladder. `$0` spend, no GPU,
**no paid dispatch fired**, **NO MPS**. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes. Secondary gate:
sub-0.15. **Pre-registration:** `.omx/research/lever_c_viability_smoke_DESIGN_20260610T135727Z.md`.

---

## 0. PRE-REGISTRATION (written BEFORE measurement; see DESIGN memo §3)

**PREDICTION:** the fresh-init small **conv** per-pair-latent frame1 decoder (PR95-L18 Conv+PixelShuffle
+bilinear-skip+sin), being structurally able to represent sharp argmax boundaries (unlike the
coordinate-MLP), CAN hold **d_seg < 0.01 AND d_pose < 0.01 JOINTLY at < 120 KB**. The conv inductive
bias breaks the #61 antagonism the coordinate-INR could not.

**KILL/DEFER CRITERION:** if the conv decoder CANNOT hold both terms below 0.01 at < 120 KB even in
smoke (one term blocks the other, or holding both needs near-frontier size), then the score-native
carrier converges to full-frontier-size (the rate advantage is gone). **FIRE → DEFER the score-native
crux + pivot** (lever D contour-coding of the residual / R1+R2+R3 lossless bank / AFSR-1). Record WHICH
term blocked, at what byte cost.

**RESULT vs pre-registration: PREDICTION REFUTED. KILL/DEFER CRITERION FIRES → FAIL → pivot.** The conv
decoder did NOT break the antagonism. It moved POSE (d_pose 11.99 constant → 0.105, a 114× win) but
moved **d_seg by ZERO** — the trained decoder's exact d_seg is **identical to a flat constant frame**
(0.50732 vs constant-control 0.50692). The seg term is blocked by 50× (gate needs < 0.01), and both
configs hit/exceed the byte budget. The blocking term is **d_seg**, at a byte cost of 104 KB (config A).

---

## 1. The RD point (the $0 viability gate — authority)

8 pairs, 180 epochs, M5 Max CPU-torch, exact frozen PoseNet+SegNet, GT via `yuv420_to_rgb`,
EMA-shadow inference checkpoint, numpy↔torch parity 1.0. Two capacity configs (the conv decoder makes
frame1; frame0=GT0 for the pure frame1 RD point — frame0 is SegNet-invisible so frame1 carries BOTH
the d_seg and the pose debt PoseNet reads):

| config | params | bytes (int8+brotli) | **exact d_seg** | **exact d_pose** | seg_term 100· | pose_term √(10·) | **joint-hold <120KB** |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **A** (seed32, ch 32-24-16-12-8) | 118,087 | **104,426** | **0.50732** | **0.10540** | 50.73 | 1.027 | **NO** |
| **B** (seed48, ch 48-32-24-16-12) | 250,587 | 214,290 | **0.69528** | 0.08546 | 69.53 | 0.924 | **NO** |
| constant-frame control (flat mean color) | — | — | **0.50692** | 11.989 | 50.69 | 10.95 | — |

**joint-hold = NO for both.** Gate (d_seg < 0.01 AND d_pose < 0.01 at < 120 KB) FAILS on d_seg by 50×.

## 2. The decisive finding: the conv decoder moves POSE but moves d_seg by ZERO

The single sharpest number in this smoke: **config A's trained-decoder exact d_seg (0.50732) is
statistically identical to a flat constant mid-gray frame (0.50692)** — a delta of +0.0004, i.e. the
conv decoder achieves ZERO d_seg improvement over painting the whole frame one color. Meanwhile it
crushed pose (constant d_pose 11.99 → 0.105, 114×). The decoder learned a frame whose **luma gives
PoseNet a good 6-dim pose** but whose **SegNet argmax partition is no better than a blank frame**.

This is the #57/#61 antagonism, now measured directly in the conv (HNeRV-class) family and EVEN SHARPER
than the coordinate-INR result:
- #61 coordinate-INR: pose-trained frame1 → d_seg 0.733 (catastrophic seg). Lever-C conv: d_seg pinned
  at the constant-frame floor 0.507 (no seg signal at all).
- The conv inductive bias did NOT help the seg term. **More capacity made d_seg WORSE** (config B 250k →
  d_seg 0.695, above the constant floor — the bigger decoder destabilised; the training trajectory
  shows B's recon diverging 3028 → 10596 while seg_ce climbed 2.80 → 3.76). This reproduces the #57
  non-monotone-capacity instability inside the conv family.

**The training trajectory (both configs) is the mechanism:** seg_ce (boundary-weighted CE against GT
SegNet argmax through the frozen SegNet) NEVER descended — it CLIMBED from ~2.9 to ~3.1-3.8 as the
warm-schedule shifted weight onto the seg+pose objective, while recon plateaued/diverged and pose
oscillated down. The decoder spent its capacity on whichever term the schedule emphasised and never
held both — the literal definition of the antagonistic constraint.

## 3. WHY the conv decoder cannot move d_seg (the mechanism, vs lever_b which CAN)

The contest d_seg is the per-pixel argmax-flip rate between GT-frame1's SegNet logits and the
candidate-frame1's SegNet logits (`upstream/modules.py:112`). There are two ways to drive it down:
- **lever_b (works):** generate the 5-class SegNet **logit map DIRECTLY** at 384×512 and CE it against
  the GT argmax. lever_b reaches d_seg 0.0116 in 30 epochs (CE → 0.08). The generator outputs the
  label space the metric reads — a short, well-conditioned path.
- **lever-C (does NOT work):** generate an **RGB frame**, run it through the **frozen EfficientNet-B2
  SegNet**, and CE the resulting logits. The RGB→SegNet map is a deep, ill-conditioned, highly
  non-convex composition; the gradient to the conv decoder is weak and the decoder cannot find an RGB
  image whose SegNet argmax matches GT below the constant-frame floor while ALSO carrying pose luma.

The honest conclusion: **a frame that LOOKS like the scene (recon) is NOT a frame whose frozen-SegNet
argmax matches GT.** The frontier's 177 KB HNeRV decoder solves this only by amortising BOTH frames as
genuinely high-fidelity RGB at full resolution — exactly the full-renderer the score-native carrier was
trying to avoid. The score-native rate advantage (a cheap label-map frame1) is fundamentally
incompatible with the pose constraint (PoseNet needs real luma in frame1). They are the two halves of
the #61 wall, and the conv family does not bridge them.

## 4. VERDICT: FAIL → DEFER the score-native frame1 crux + PIVOT (NOT a kill of the primitive)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-LEVEL: the conv per-pair decoder
PRIMITIVE is real + working in isolation (it crushes pose 114× over constant; numpy-portable; parity
1.0; scorer-free inflate path). The PARADIGM (score-native cheap-frame1 carrier) is what is falsified
on the full S — across BOTH the coordinate-INR (#57/#61) AND now the conv (HNeRV-class) family. The
pre-registered conclusion is confirmed: **pose-relevant luma is NOT cheap enough; the score-native
frame1 representation converges to full-frontier-size** (HNeRV-parity lesson 5 — the full RGB renderer
dominates a single-component slot). The cheap-frame1 score-native carrier is DEFERRED.

### The pivot (pre-registered FAIL branch — the next launchable unit)

The score-native cheap-frame1 axis is exhausted across two architecture families; do NOT iterate a third
frame1 carrier. Re-route to the rate-side levers that do NOT depend on a cheap frame1:

1. **Lever D — contour-code the seg RESIDUAL on top of the 177 KB frontier (highest-priority, the
   honest rate lever):** keep the frontier's full-RGB frame1 (which already lands d_seg 5.6e-4 + d_pose
   2.9e-5) and attack the ARCHIVE BYTES, not the frame representation. The `contour_codec` /
   `boundary_solver` (#52/#55) encode only the small-margin boundary band; combined with the seg
   free-budget (#52 margin polytope) this is a rate-reduction on the EXISTING frontier, not a new
   carrier. This is the lever that survives the #62 finding: the frontier already holds both terms;
   the only open question is bytes.
2. **R1+R2+R3 lossless entropy bank on the frontier payload** (the `lane_pr110_payload_entropy_recode`
   line that produced the current 0.19110 frontier): per-tensor byte-maps (PR95 L21), split brotli
   streams (L23), range/arithmetic coding (L30) on the frontier decoder weights — pure rate, no frame
   change, no scorer risk.
3. **AFSR-1 rate campaign** (if a fresh frontier-class substrate is wanted): a full-RGB per-pair conv
   decoder trained for BOTH frames jointly (NOT a cheap frame1) — i.e. accept that frame1 must be
   full-fidelity and compete with the frontier on TOTAL bytes, which is the HNeRV-class campaign the
   #57/#61/#62 chain has now triply-confirmed is the only score-native path that can hold both terms.

**Eval gate ("advisory S beats frontier 0.19110 OR sub-0.15"):** NOT met (best d_seg 0.507 ≫ frontier
5.6e-4; full S ≫ frontier). **NO paired exact eval launched** (correct fail-closed: do not spend $ to
confirm a non-improvement). `$0` spent. No campaign pre-registered (the PASS branch did not fire); the
pivot above IS the next launchable unit.

## 5. Wire-in (Catalog #125)

1. **sensitivity-map — ACTIVE:** the new sensitivity input is the constant-frame-floor anchor: the conv
   decoder's d_seg (0.507) == constant-frame d_seg (0.507), i.e. the RGB→frozen-SegNet path carries ZERO
   d_seg-reduction gradient at score-native capacity. The waterfiller must treat frame1's d_seg as
   un-amortisable through an RGB carrier — d_seg bytes belong in the LOGIT-space generator (lever_b) or
   in the full-fidelity frontier RGB, never in a cheap RGB carrier.
2. **Pareto — ACTIVE:** the RD point {d_seg 0.507, d_pose 0.105, 104 KB} (config A) and {0.695, 0.085,
   214 KB} (config B) extend the #61 U-shaped frame1 surface into the conv family: MORE conv capacity
   moves d_seg the WRONG way (0.507 → 0.695) and the byte cost crosses the frontier budget. The
   Pareto-feasible frame1 move is NOT a bigger conv carrier; it is keep-the-frontier-frame1 + attack
   bytes (lever D).
3. **bit-allocator — ACTIVE:** allocating 104-214 KB to a conv frame1 carrier buys d_pose (0.105) but
   ZERO d_seg — a mis-allocation for the dominant seg term. The allocator should NOT route frame1 bytes
   to an RGB conv carrier; route seg bytes to the logit-space generator (lever_b) or to the frontier.
4. **cathedral-autopilot — gate NOT met:** advisory d_seg 0.507 ≫ frontier; no paired-eval dispatch.
5. **continual-learning — ACTIVE:** reseeds the planner: (a) the conv per-pair decoder crushes pose
   114× over constant (d_pose 0.105 at 104 KB) — the POSE sub-problem is solvable by the conv carrier
   too (sister of the #57 coordinate-INR pose carrier); (b) the conv decoder moves d_seg by ZERO
   (pinned at the constant-frame floor 0.507) — the RGB→frozen-SegNet path is NOT a viable d_seg
   carrier at score-native capacity; (c) MORE conv capacity makes d_seg WORSE (0.507→0.695) and
   destabilises training (recon diverges) — the #57 non-monotone-capacity instability is intrinsic to
   the score-aware-against-frozen-scorer objective, not the architecture family; (d) the score-native
   cheap-frame1 carrier is now FALSIFIED across BOTH coordinate-INR and conv families → the score-native
   frame1 converges to full-frontier-size; (e) the next levers are rate-side on the EXISTING frontier
   (lever D contour residual / R1+R2+R3 entropy bank), not a third frame1 carrier.
6. **probe-disambiguator — RESOLVED:** "can a per-pair-latent CONV decoder break the #61 antagonism and
   hold d_seg AND d_pose jointly below 120 KB?" → **NO** (d_seg pinned at constant-frame floor; pose
   solved but seg unmovable through RGB→frozen-SegNet). "does more conv capacity help d_seg?" → NO
   (worse: 0.507→0.695 + training divergence). "is the score-native cheap-frame1 axis viable?" → NO
   (falsified across two families). The next probe: lever D contour-residual rate reduction on the
   frontier (do NOT probe a third frame1 carrier).

## 6. Deliverables + cross-references

- **Module (NO-FAKE, tested):** `src/tac/boundary_math/conv_pair_decoder.py` (the lever-C conv per-pair
  decoder: PR95-L18 Conv+PixelShuffle+bilinear-skip+sin block math, numpy-portable forward = inflate
  reference, int8+brotli byte accounting, checkpoint I/O) + 20 behavior tests
  (`tests/test_conv_pair_decoder.py`: numpy↔torch parity for conv3x3/pixelshuffle/bilinear + full
  forward; per-pair/per-pixel variation; **zero-latents collapse all pairs to ONE frame** (proves the
  latent is load-bearing); **a constant decoder fails**; byte cost tracks capacity + num_pairs; quant
  round-trip; the null-space margin-free-budget weight redistributes boundary-heavy mean-1; the
  Jacobian saliency weight changes the gradient vs uniform). 112 boundary_math tests green (20 new +
  92 existing, 0 regressions); ruff clean.
- **Trainer (NO-FAKE, internal-consistency guard):** `tools/lever_c_train_conv_pair_decoder.py` (the
  joint seg+pose trainer: torch conv decoder mirror; the THREE original terms — null-space-primary
  recon via margin polytope #52 + Jacobian-aimed pose via posenet_jacobian_saliency #61 +
  argmax-polytope-constrained seg via boundary-weighted CE; differentiable rgb_to_yuv6 + eval_roundtrip
  STE in the inner loop; EMA-0.997 shadow as the inference checkpoint; exact d_seg/d_pose re-measured on
  the numpy-decoded frame; refuses a stub via elapsed ≥ epochs·MIN_SEC).
- **Artifacts (SSD tier):** `/Volumes/VertigoDataTier/pact/lever_c_task62_20260610/` (smokeA_118k/ +
  smokeB_250k/ decoder.npz + train_result.json per config; logs/).
- **Cross-refs:** `frame1_dual_fidelity_ptnc_20260610T133713Z.md` (#61 verdict — the wall + the
  pre-registered lever-C spec this executed) · `score_native_pose_carrier_20260610T125000Z.md` (#57 —
  the coordinate-INR pose ceiling) · `lever_c_viability_smoke_DESIGN_20260610T135727Z.md`
  (pre-registration) · `src/tac/boundary_math/{margin_polytope,posenet_jacobian_saliency,lever_b_generator,contour_codec,boundary_solver}.py`
  (the reused surfaces + the lever-D pivot targets) · CLAUDE.md HNeRV-parity lesson 5 (full renderer
  not single-component slot — the canonical articulation of this triply-confirmed verdict).
