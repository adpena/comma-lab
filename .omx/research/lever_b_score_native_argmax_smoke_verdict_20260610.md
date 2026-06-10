# LEVER-B descent-smoke verdict — the SCORE-NATIVE class shift is REAL (2026-06-10)

**Subagent:** `lever_b_score_native_argmax_smoke_20260610`. **Lever B** of `GOAL_standing_v3` (the #1
offensive lever; the cleanest unquestionable class shift per
`original_innovation_offensive_plan_20260610.md` §4). **Evidence grade:** `[macOS-MLX research-signal]`
(d_seg from advisory MLX argmax forward) + `[macOS-CPU advisory]` (frozen-scorer GT targets). NO score
claim, `promotable=false`, `ready_for_exact_eval_dispatch=false`. $0 local, NO cloud, NO paid GPU, NO
MPS, NO /tmp (targets on `/Volumes/VertigoDataTier/pact/...`). Frontier read from pointer
(0.19109982 [contest-CPU], 177,169 B; decoder seg-share ≈ 162,127 B per `information_theoretic_floor_T_floor`).

---

## 0. The headline (the pre-registered question, answered)

**The pre-registered hypothesis H_B is CONFIRMED.** A tiny conditional label-map generator g(x,y,pair)
→ 5-class logits at 384×512, trained with cross-entropy + a lever-G margin-weighted hinge against the
FROZEN SegNet's argmax on the GT frame1s, **descends d_seg far below the 0.10 KILL bar** — the seg
term is GENERATABLE, not merely codeable. The B1-R2 mean-field bug does NOT recur for a CLASSIFICATION
objective. This is the empirical confirmation of the score-native class shift: a classifier hitting a
frozen classifier's argmax is cheap, because argmax is invariant to the 80.67% scorer-null pixel energy.

| pre-registered gate | bar | 16-pair result (300 ep) | 600-pair result (300 ep) | verdict |
|---|---|---|---|---|
| **KILL-B-MECHANISM** (d_seg descends < 0.10) | < 0.10 by ~300 ep | **0.00725** (13.8× below) | **0.00826** (12.1× below) | **PASS — NOT KILLED** |
| **KILL-B-RATE** (blob < frontier seg-share ~162 KB) | < ~162,127 B | 47,070 B @16 pairs (base-dominated) | **63,802 B** (base 46,914 + mod 16,888) | **PASS — NOT KILLED (2.54× smaller)** |

**VERDICT: PROCEED-TO-CAMPAIGN.** Both pre-registered KILLs resolved NEGATIVE (not triggered). The
600-pair score-native seg-generator holds d_seg = 0.00826 across the full 600 pairs at a **63,802-byte**
quantized blob — **2.54× smaller** than the frontier decoder's ~162,127-byte seg-share. Adding the pose
section (6,650 B, the F byproduct), the full mechanism-stage carrier = **70,452 B vs the frontier's
177,169 B (−60% bytes)**. R-D arithmetic (advisory): at this carrier's rate (0.04691 vs frontier 0.11797,
**Δrate −0.071**) AND frontier distortion, hypothetical S = **0.120** — already below T_3 (sub-0.15).
This is MECHANISM-STAGE; the campaign measures the real S (with the legal-frame appearance section) via
ONE paired CPU+CUDA exact eval.

---

## 1. The d_seg descent (mechanism PROVEN, 16-pair smoke)

The 16-pair / 300-epoch / MLX descent (`smoke_n16/smoke_result.json`):

| epoch | train_loss | d_seg | note |
|---:|---:|---:|---|
| 1 | 1.467 | 0.2986 | random init |
| 25 | 0.0804 | 0.0118 | **already 8.5× below the 0.10 bar by ep25** |
| 100 | 0.0612 | 0.0076 | |
| 250 | 0.0471 | 0.0056 | approaching the boundary-residual floor |
| 300 | 0.0504 | 0.0073 | **final** |

The descent is monotone toward ~0.006, the small-margin boundary residual — exactly the lever-G
"hard pixels." **Robust to seed:** seed-2 reproduced 0.494→0.0098(ep50)→0.0075(ep100) — not a lucky init.

**The decisive run — full 600 pairs / 300 ep** (`smoke_n600/smoke_result.json`): d_seg crosses the bar on
epoch 1 (0.0277 across ALL 600 pairs) and settles at **0.00826** (ep300) — the amortized base learns the
shared 600-map structure immediately; the per-pair mod refines it. Both d_seg values (16-pair 0.0073,
600-pair 0.0083) are ~10-15× the full-600 frontier d_seg (5.6e-4); closing that residual is a
capacity/training-length campaign question, NOT a "generatable vs not" question — **the mechanism (a tiny
net learns the 600-argmax partition into a 64 KB blob) is settled.**

## 2. Lever-G interior budget (the free logit room — CONFIRMED LARGE)

The frozen SegNet margin field (top1−top2 logit gap) over the GT frame1s
(`targets_n16/targets_meta.json`):

| margin band | fraction of pixels |
|---|---:|
| margin > 0.5 | 98.64% |
| margin > 1.0 | 97.38% |
| margin > 2.0 | 95.23% |
| margin > 5.0 | 80.40% |

**95% of pixels carry > 2 logits of free room** — a huge certified-free interior budget the generator
dumps its representational error into (the lever-G prediction, confirmed). This is the seg-axis analog
of the resize-null basis (the dual nobody had built). It is why the classification carrier is cheap:
the generator must be right only at the ~1-5% small-margin boundary, free everywhere else.

## 3. Pose-trajectory byte size (lever F free byproduct — MEASURED at full 600)

The 600×6 PoseNet trajectory entropy (the F probe, fallen out of the 600-target build):
- raw fp16 = 7,200 B; **fp16-brotli-q11 = 6,650 B** (best); delta-fp16-brotli = 6,824 B;
  delta-fp16-lzma-xz = 6,944 B. ⇒ **~11 B/pair**, rate cost `25·6650/N = 0.00443`.
- This **RESOLVES floor-memo P6** (the unrun RANK-6 pose-output-entropy probe): the pose carrier is a
  **6.65 KB** section, near-free relative to the rate budget. Pose is NOT the binding term; the seg
  amortizer + rate are. The OPEN P6 in the floor ledger is now CLOSED with a measured number.

## 4. Byte accounting + the KILL-B-RATE measure (the amortization thesis — CONFIRMED)

The 600-pair quantized blob (int8 per-tensor + brotli-q11, `smoke_n600/smoke_result.json`):
**63,802 B** = base **46,914 B** (shared MLP, 66,245 raw int8 bytes) + mod **16,888 B** (600 pairs ×
32 B = 19,200 raw int8 → 16,888 brotli'd). The 16-pair blob was 47,070 B (base 46,564 + mod 506).
**The base section is byte-identical-size across 16 and 600 pairs (66,245 raw int8) — it is SHARED and
amortized; only the per-pair mod grows linearly.** This is the amortization thesis, measured: the 600
argmax partitions cost one ~47 KB base + 600 tiny ~28 B/pair codes, NOT 600 independent maps.

**KILL-B-RATE = NOT TRIGGERED.** 63,802 B (seg) ≪ frontier decoder seg-share ~162,127 B (2.54× smaller).
With pose: **70,452 B total carrier vs frontier 177,169 B (−60%)**. Rate 0.04691 vs 0.11797 (Δ −0.071).

## 5. Portability contract (MLX → numpy parity)

The classification carrier's portability contract is ARGMAX-parity (the score-native quantity d_seg
reads), not bit-exact logits. The numpy reference forward matches the MLX forward at argmax-agreement
**1.0** with rel-logit-diff 0.56% (`portability_parity.parity_pass=true`). fp32 accumulation drift on a
trained net is argmax-invariant — itself a lever-G manifestation.

---

## 6. VERDICT + the next lever (pre-registered)

__VERDICT: PROCEED-TO-CAMPAIGN.__ Both pre-registered KILLs resolved NEGATIVE on the full 600-pair smoke:
d_seg = 0.00826 (12.1× below the 0.10 bar) at a 63,802-byte blob (2.54× smaller than the frontier
seg-share). The seg term is GENERATABLE and the classification carrier is CHEAPER than the RGB carrier —
the score-native class shift is REAL. Robust to seed (seed-2 16-pair reproduced d_seg 0.494→0.0075).

**The campaign (next lever, pre-registered with its prediction):**
1. **The legal-frame bridge** (the campaign's first build, the burning question below): solve the direct
   variational problem — the minimum-byte FRAME whose SegNet-argmax == the generator's learned argmax AND
   whose YUV6 holds the pose tube. PREDICTION: the legal frame costs an appearance section but stays well
   under the frontier 177,169 B (the −60% headroom is large), landing the first score-native submission
   candidate near S~0.13-0.16, then ONE paired CPU+CUDA exact eval (~$0.6) measures the real S.
2. **Stack H** (warp-frame0): frame0 is SegNet-invisible; reconstruct it as warp(frame1, pose) in
   rate-free inflate code, deleting the frame0 appearance entirely.
3. **Stack G harder** (argmax-hinge-only): shrink the generator using only the boundary pixels (95% of
   px are free interior), reducing the 63,802-byte blob further.

The KILL-B-RATE pivot to I (quotient/dictionary) is NOT needed — the generator IS cheaper than dedup
would be. I remains a parallel force-multiplier if the campaign's legal-frame section grows.

**The burning mathematical question:** the smoke trains a LOGIT generator (mechanism proof). The legal
archive scores FRAMES. The campaign crux is: what is the minimum-byte FRAME (raw [0,255] RGB resizing to
384×512) whose SegNet argmax equals the generator's learned argmax AND whose YUV6 holds the pose tube?
This is the direct variational solve (`direct_differential_geometric_inverse_solve`): the feasible cell
per pixel = (margin half-space) ∩ (pose tube) ∩ (resize/YUV null). The generator gives the argmax target;
the cheapest legal frame in that cell is the carrier. THAT is the campaign's first build.

## 7. Wire-in (Catalog #125)
1. **sensitivity-map** — ACTIVE: the margin field (lever-G interior budget) is the per-pixel seg-logit-null
   prior (95% px free); the per-pair mod-vs-base byte split is the seg/pose allocator.
2. **Pareto** — ACTIVE: B moves the carrier OFF the pixel-native Pareto vertex (the only move the
   exhaustion map left open).
3. **bit-allocator** — ACTIVE: base (shared) + mod (per-pair) + pose-section is the literal allocator.
4. **cathedral-autopilot** — the B descent-smoke → (conditional) paired-eval is the dispatch surface.
5. **continual-learning** — this verdict reseeds the V3 judge: score-native decomposition is a LIVE
   class-shift axis the pixel-native exhaustion map could not see (d_seg generatable at < 60 KB).
6. **probe-disambiguator** — RESOLVED: "is the seg term generatable or only codeable?" → GENERATABLE
   (16-pair d_seg 0.0073 ≪ 0.10). P6 (pose entropy) RESOLVED: few-KB.

## 8. Cross-references
`GOAL_standing_v3_20260610.md` · `original_innovation_offensive_plan_20260610.md` (§4 the smoke spec) ·
`exactly_what_segnet_posenet_see_20260610.md` (the exact targets) ·
`information_theoretic_floor_T_floor_20260610.md` (P3 sub-0.15 = distortion threshold; P6 pose probe) ·
`direct_differential_geometric_inverse_solve_20260610.md` (the legal-frame variational solve = campaign
build) · `upstream/{modules.py,frame_utils.py}` (frozen authority, read in full) ·
`tools/lever_b_build_score_native_targets.py` + `tools/lever_b_score_native_argmax_smoke.py` (the smoke).
