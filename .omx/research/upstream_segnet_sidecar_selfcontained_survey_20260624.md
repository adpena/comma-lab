---
title: "Upstream SegNet-sidecar / self-contained-seg survey — the seg analog of our L13 pose carrier (ALL upstream + intake searched)"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU-only research+memo; NEVER MPS; no GPU; no dispatch; no PR; no exact-eval"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-24
subagent: upstream-segnet-sidecar-selfcontained-survey-20260624
verdict: NO_CHEAP_FLAT_SEG_SIDECAR; THREE_ADOPTABLE_SEG_MECHANISMS (mask-conditioned renderer / seg-action boundary-flip sidecar / SegNet-guided bit-routing prior)
cross_refs:
  - .omx/research/witness_L13_optimal_pose_carrier_result_20260621.md          # our pose carrier (the thing to find a seg analog of)
  - .omx/research/CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md
  - .omx/research/island_representation_level_intrinsic_dim_20260624T041154Z.md  # GO-GENERATOR: seg islands full-rank-in-pixels, 8-dim nonlinear manifold
  - .omx/research/curve_core_gate_RED_survival_wall_and_the_pincer_20260618.md   # texture-dependence survival wall
  - .omx/research/long_thin_tail_lane_marking_codec_math_20260623.md            # the 3-structure decomposition (islands = lane markings)
upstream_sources_read:
  - upstream/submissions/{damir_bearclaw_003,damir_bearclaw_002,v4_qp_aq2_roi,neural_inflate}/   # ROI/QP-map/neural
  - experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/repo/  # FULL submissions tree incl. fp4_mask_gen, qzs3_range_mask, quantizr, qpose14_*, delta_codec, mask2mask(leaderboard-only), ph4ntom_drv, tomasdousek, jas0xf
  - experiments/results/public_pr{95,100,101,103,105,106,108,112}_intake*/       # HNeRV-family (pure-RGB-render, no seg sidecar)
---

# Upstream SegNet-sidecar / self-contained-seg survey

**Operator question:** *"Did any other PRs have useful SegNet sidecars, or self-contained seg approaches
similar to our POSE solution, that we can adopt or learn from?"*

**TL;DR (HONEST).** YES — three distinct seg mechanisms exist upstream, and one of them (the **qpose14
"seg-actions"** boundary-flip sidecar, **236 bytes**, on the #1-leaderboard 0.32 submission) is the
*structural twin* of our L13 pose carrier: a tiny self-contained correction that exploits the same
sparsity our pose carrier exploits (PoseNet target = 6-dim; SegNet decision = argmax → only boundary
pixels matter). BUT the **direct analog of our pose carrier — a SMALL flat sidecar that stores the
scored quantity and re-derives it — is DOMINATED for seg** and the existence-proof cross-check confirms
why: the d_seg island stratum is **full-rank in every linear basis** (`island_representation_level_intrinsic_dim`
→ GO-GENERATOR), so the seg "target" is NOT 6-dim-flat the way pose is. The realizable seg analogs are
(b) the **236-byte seg-action boundary-flip sidecar** and (c) **SegNet-guided bit-routing**, both of
which we can adopt today; the "store-the-mask-and-render" family (a) is a full generator, not a sidecar.
Pointer UNMOVED 0.19110. `[contest-CPU advisory]` NON-PROMOTABLE. NO score claim.

---

## 1. The asymmetry — VERIFIED, not assumed

Both SegNet **and** PoseNet run on the submission's **OUTPUT RGB frames**. You cannot inject a stored
mask or pose tensor into the scorer (confirmed by reading every inflate.py: they all emit `.raw` RGB
frame bytes; the scorer re-derives masks/poses from those pixels). So a "seg sidecar" can only be:

- **(a) mask-CONDITIONED renderer** — store the 5-class mask, render RGB from it so the rendered RGB
  re-derives back to (approximately) that mask. (qpose14 / quantizr / fp4_mask_gen / qzs3 / mask2mask.)
- **(b) correction sidecar** — a tiny per-tile edit applied on top of a base render that flips SegNet
  argmax in the right places. (qpose14 **seg-actions**; PR101 L27 per-pair single-dim correction is the
  *pose-flavored* sibling.)
- **(c) bit-routing / ROI prior** — spend MORE bits where SegNet decisions live, FEWER where they don't.
  (v4_qp_aq2_roi SegNet-guided per-block QP map; damir_bearclaw ROI/middle-band; delta outside-deadzone.)

**The pose↔seg asymmetry (the load-bearing finding).** Our L13 pose carrier works because the pose
target is a **6-dim** vector per pair — a ~22.5 KB amortized INR can paint a frame0 (SegNet-invisible)
whose frozen-PoseNet output hits the 6-dim target. The seg target has **no such low flat dimension**:
`island_representation_level_intrinsic_dim` (2026-06-24) measures the d_seg-binding class-1 island
stratum at **linear rank k95 = 412 px / 61 DCT / 29 contour / 94 motion** (all ≫ a flat witness budget),
collapsing only to an **8-dim NONLINEAR** AE manifold (phase-shuffle control proves it real). So a flat
"store-the-seg-quantity" sidecar is structurally dominated; the seg quantity is a *generator-or-mask*
object, not a 6-number object. **This is why every upstream seg approach is either a full mask-conditioned
generator (a) or a sparse boundary/bit-routing edit (b/c) — never a flat carrier.**

---

## 2. The survey table

Score axis: leaderboard `[contest-CPU]` (from PR81 README leaderboard, the official mirror). "Stores"
= what the archive actually carries. "Self-contained like our pose carrier?" = does it store the scored
proxy + re-derive it (YES), or just hint an encoder (PARTIAL), or neither (NO)?

| submission (PR) | score | stores | mechanism | self-contained? | adopt / learn verdict |
|---|---|---|---|---|---|
| **qpose14** (#63) | **0.32** | 5-class mask (range-coded ~219KB) + FP4 gen ~56KB + pose (varint) + **seg-actions 236 B** | (a) mask-cond renderer **+** (b) **seg-action boundary-flip sidecar** | **YES** (mask) **+ YES** (action sidecar is the twin of our pose carrier) | **ADOPT (b): the 236-B seg-action sidecar is the #1 takeaway.** Also the integration template: mask-render base, then sparse boundary correction. |
| quantizr (#55) | 0.33 | 5-class mask (AV1) + FP4 `JointFrameGenerator` weights + pose | (a) mask-conditioned renderer | YES (mask) | LEARN: canonical mask-conditioned arch (EMA 0.997 / KL-T2 / eval_roundtrip; already in our CLAUDE.md L14-L32). Dominated by qpose14 (no action sidecar). |
| **fp4_mask_gen** (#62) | **0.37** | `model.pt.br` FP4 gen + **`mask.obu.br`** (AV1-CRF63 5-class mask, one/pair) + pose.bin | (a) mask-conditioned renderer; FiLM-on-pose, dual-head (f1 pose-cond, f2 static) | **YES** — *exact* structural twin of our carrier at the GENERATOR scale (stores the scored proxy=mask, renders RGB from it) | **LEARN (the cleanest reference impl).** `JointFrameGenerator.forward(mask2,pose6)`: 5-class embed → coord-grid → shared trunk → FiLM(pose) frame1 + static frame2. This IS the seg-carrier; adopt the arch as our generator's mask-conditioning path. |
| qzs3_range_mask (#≈63 family) | 0.33-ish | **range-coded mask 159 KB** (9-ctx adaptive arithmetic coder, `range_mask_codec.cpp`) + 37 KB FP4 gen + pose | (a) mask-cond renderer + **custom mask entropy coder** | YES (mask) | **ADOPT (mask byte win):** the 9-context arithmetic mask coder is 159 KB vs fp4's AV1-OBU 219 KB for the SAME 600 masks — a ~27% mask-rate cut. Reuse for any mask we store. |
| selfcomp (#56) | 0.38 | grayscale-LUT analog mask + block-FP weight self-compression | (a) mask-as-grayscale-LUT renderer | YES (mask) | LEARN: grayscale-LUT mask is an alt mask carrier; dominated by range-coding. (Council member; already canonical.) |
| mask2mask (#53) | 0.60 | — (leaderboard-only; **no source in any intake**) | name implies (a) mask→mask transform | unknown | LEARN-ONLY (no code). The 0.60 score says raw mask→mask without a strong RGB renderer is dominated by mask-conditioned-RGB. |
| **v4_qp_aq2_roi** (#44) | **2.02** | AV1 video + **SegNet-derived per-64×64 QP-offset map** (sky +5 / road-boundary −5) — *map is an encoder hint, NOT stored* | (c) **SegNet-guided bit-routing** | PARTIAL (seg spent at compress-time as QP hints; not in archive) | **ADOPT (c) the routing PRIOR:** runs the REAL frozen SegNet at compress-time, finds class-boundary blocks, routes bits there. The routing *signal* is exactly our generator's capacity-routing prior. |
| damir_bearclaw_002 (#30) | 1.98 | AV1 + hand-authored ROI polygon (central driving corridor), outside-region denoise/chroma-collapse | (c) geometric ROI prior | NO (no seg stored) | LEARN: the corridor-polygon is a cheap hand-prior for "where SegNet looks." Dominated by v4's *measured* SegNet map. |
| damir_bearclaw_003 (#39) | 5.09 | middle-50% band only (FFV1), synth top/bottom filler | (c) crop-to-corridor | NO | **LEARN (the canonical insight, not the score):** README states *"SegNet derives most of its useful semantic structure from the central driving corridor; upper/lower bands are position-biased."* This is the verbal root of our island/corridor routing. |
| roi_gop300_c34 / roi_v2 / av1_roi_lanczos_unsharp (#43/#48/#31) | 2.01/1.94/1.95 | AV1 + ROI preprocess | (c) ROI prior | NO | LEARN: ROI family, all ~1.9-2.0; dominated by neural. |
| delta_codec (#61) | 3.83 | low-res base + signed residual per pair; **outside-corridor deadzone/coarser step** | (b)-flavored residual + (c) corridor-aware quantization | NO (no seg) | LEARN: `--outside-deadzone`/`--outside-delta-step` = corridor-aware rate routing on a residual codec. A routing knob, not a seg carrier. |
| neural_inflate (#49) | 1.89 | tiny REN (PixelUnshuffle→3×conv→PixelShuffle residual) int8/bz2 + AV1 base | post-filter renderer (no seg, no pose) | NO | LEARN: minimal residual-refiner arch; not seg. |
| HNeRV family — PR95/100/101/103/105/106/108/112 | 0.19-0.23 | RGB-render INR weights + per-PAIR pose latents (+ L27 per-pair single-dim **pose** correction sidecar) | **(none for seg)** — pure-RGB-render; the L27 sidecar targets pose/quality, NOT seg | NO seg sidecar | **LEARN (the negative):** the entire frontier HNeRV family carries NO seg sidecar — they render RGB and let SegNet re-derive. Their seg quality is *architectural* (full renderer), confirming §1: there is no flat seg carrier worth storing at the frontier. The L27 correction sidecar is the *pose/quality* sibling of qpose14's seg-actions. |

(Also present, source-light: `ph4ntom_drv`, `tomasdousek/ditcher`, `jas0xf_adversarial_neural_representation`
— all RGB/representation codecs, no seg-specific sidecar; not re-listed.)

---

## 3. The three mechanisms, mapped to the pose carrier

**(a) Mask-conditioned renderer = our pose carrier at GENERATOR scale.** fp4_mask_gen's
`JointFrameGenerator.forward(mask2, pose6)` is *literally* the seg-carrier: store the scored proxy (the
5-class mask), embed it, render RGB so the frozen SegNet re-derives ≈ that mask. It is "self-contained"
in the same sense our pose carrier is — but it is a **full generator + a stored mask stream** (~56-220 KB),
not a 22 KB flat carrier, *because* the seg quantity is full-rank (§1). This is the dominant seg family
(0.32-0.38) and it is exactly what our live generator d_seg campaign is.

**(b) Seg-action boundary-flip sidecar = the EXACT twin of our pose carrier (the headline).** qpose14
(#1, 0.32) renders the base from the mask-conditioned generator, then runs a greedy search over a tiny
codebook of **per-tile color-push actions** (`action_specs`: road/sky + RGB directions × {2,4,6,8,12,16}
amplitudes) and stores a varint `(frame, tile, action)` record list — **236 bytes** for the whole video
— that, applied at inflate-time, nudges pixels in specific tiles so the frozen SegNet **argmax flips**
toward the GT class. This is the seg analog of our saliency-confined pose carrier: it spends bytes ONLY
on the sparse boundary tiles where the scored decision actually changes, exactly as our PoseNet-Jacobian
saliency confines carrier capacity to the 6-dim pose tube. **Both exploit: the scorer's output is
low-effective-rank (pose 6-dim; seg argmax = sparse boundary flips) so a tiny targeted edit moves it.**

**(c) SegNet-guided bit-routing prior = our generator's capacity-routing prior.** v4_qp_aq2_roi runs the
REAL frozen SegNet at compress-time and emits a per-block QP map (more bits at road/class boundaries,
fewer in uniform sky). The bits are spent in the encoder (not stored), but the *routing signal* — "paint
the class-boundary support, free the uniform interior" — is identical to the island/lane-marking routing
in `long_thin_tail_lane_marking_codec_math` and the PoseNet-saliency routing in our carrier.

---

## 4. Synthesis — is a cheap self-contained SEG sidecar structurally possible?

**A flat "store-the-seg-and-re-derive" sidecar (the literal pose-carrier analog): NO — DOMINATED.**
The existence-proof cross-check is decisive: the d_seg-binding island stratum is **full-rank in every
reconstruction-faithful linear basis** (k95 = 412 px / 61 DCT / 29 contour / 94 motion ≫ any witness
latent budget; `island_representation_level_intrinsic_dim` → GO-GENERATOR). It collapses only to an
**8-dim NONLINEAR** manifold. Pose is genuinely ~6-dim-flat; seg is 8-dim-nonlinear-curved-through-
high-linear-rank. So you cannot store the seg quantity in a small flat code and re-derive it the way the
22 KB INR re-derives the 6 pose numbers. This is consistent with the **survival wall** (`curve_core_gate_RED`):
static-store seg debt does not survive the round-trip because the seg signal is texture/boundary-dependent,
not a smooth low-rank field. **Confirmed, not refuted.**

**A SPARSE boundary-targeted seg sidecar (qpose14 seg-actions): YES — and it is the real twin.** It does
not store the seg field; it stores a **236-byte list of which boundary tiles to nudge**, exploiting the
argmax-sparsity instead of low flat rank. This is the structurally-honest seg analog of our pose carrier:
*both* spend bytes only on the scorer's effective support (pose tube / seg boundary tiles), *both* are
self-contained, *both* are tiny. The difference is mechanism (carrier-render vs. apply-edit), forced by
the rank asymmetry. This is the adoptable diamond.

---

## 5. RANKED top-3 adoptable ideas (concrete wire-in to our generator + custom witness format)

**#1 — qpose14 seg-action boundary-flip sidecar (236 B) → add to the custom witness format as a
post-render correction op.** After the trained generator emits each pair's RGB, run a greedy frozen-CPU-
SegNet search over a small per-tile color-push action codebook (port `action_specs` + the tile/varint
`(frame,tile,action)` record format from `qpose14_r55_segactions_minp/probe_more_seg_actions_minp.py`)
and store the action list as a new witness-format section. **Wire-in:** the inflate interpreter applies
the action atlas to the rendered tiles before writing `.raw`; the encoder side is the greedy search
already written upstream. This directly attacks d_seg (the sole remaining witness wall per CAPSTONE §9)
at **sub-KB** cost, exactly the way our pose carrier closed the pose wall. **Highest EV: tiny bytes,
targets the live d_seg debt, source code exists to port, and it is the proven #1-leaderboard lever.**

**#2 — qzs3 9-context adaptive-arithmetic 5-class mask coder (`range_mask_codec.cpp`, 159 KB vs
AV1-OBU 219 KB) → adopt as the mask-stream codec IF/when we store a conditioning mask.** For any
mask-conditioned generator path, range-coding the 5-class argmax stream beats AV1-OBU by ~27% on the
same 600 masks. **Wire-in:** replace the mask carrier with the 9-ctx arithmetic coder; the .cpp is a
drop-in entropy primitive. Conditional on us choosing a mask-conditioned generator (the 0.32-0.38
family) over a pure-RGB INR.

**#3 — v4_qp_aq2_roi SegNet-guided routing signal → fold into the generator's capacity-routing prior
(NOT bits, but training emphasis).** Run the real frozen SegNet at compress-time to produce the
per-block class-boundary map and use it to **weight the generator's d_seg loss / capacity allocation**
toward boundary blocks (sky = free, road-boundary = paid) — the exact island/lane-marking routing from
`long_thin_tail_lane_marking_codec_math`. **Wire-in:** add a SegNet-boundary saliency weight map to the
generator's seg loss term (the seg sibling of our PoseNet-Jacobian saliency map already in
`posenet_jacobian_saliency.py`). $0 to build; sharpens the live generator campaign on the d_seg islands.

---

## 6. NO-FAKE ledger
- READ (this turn, all source-faithful, $0): upstream/submissions/{damir_bearclaw_002/003, v4_qp_aq2_roi,
  neural_inflate} inflate/preprocess/qpmap; PR81 full submissions tree (fp4_mask_gen inflate.py,
  quantizr inflate.py, qzs3_range_mask inflate.py + range_mask_codec.cpp, qpose14_*_segactions
  probe_more_seg_actions_minp.py, delta_codec README); PR81 README leaderboard (official scores);
  PR101/106 trees (HNeRV-family, no seg sidecar). Verified the asymmetry by reading the inflate output
  path (all emit RGB .raw → scorer re-derives).
- DERIVED: seg has no flat low-dim carrier (full-rank linear basis, 8-dim nonlinear manifold,
  GO-GENERATOR) → the literal pose-carrier analog is dominated; the realizable seg analogs are the
  236-B seg-action boundary sidecar (b) and the SegNet-routing prior (c). Mask-conditioned renderers
  (a) are full generators, not sidecars.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; nothing dispatched/built; `[contest-CPU
  advisory]` NON-PROMOTABLE; the 236-B sidecar's d_seg gain on OUR generator is UNMEASURED (an adopt
  CANDIDATE, the next $0/byte-closed probe, not a result).

## 7. 6-hook wire-in
- #1 sensitivity-map: ACTIVE — proposes a SegNet-boundary saliency weight map (seg sibling of
  posenet_jacobian_saliency), top-3 idea #3.
- #2 Pareto: ACTIVE — the 236-B seg-action sidecar is a sub-KB Pareto point on (bytes, d_seg).
- #3 bit-allocator: ACTIVE — SegNet-guided routing prior (idea #3) is a per-block d_seg allocation prior.
- #4 cathedral autopilot dispatch: N/A — survey/advisory; no archive-deployable row this turn.
- #5 continual-learning posterior: N/A — `[contest-CPU advisory]`, no exact-eval anchor.
- #6 probe-disambiguator: ACTIVE — resolves "is a self-contained seg sidecar possible like the pose
  carrier?" → flat-carrier NO (rank), sparse-boundary-action sidecar YES (argmax sparsity).
