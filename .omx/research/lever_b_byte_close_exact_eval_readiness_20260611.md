# Lever-B byte-close → exact-eval READINESS — verdict (2026-06-11)

**Subagent:** `lever_b_byte_close_exact_eval_readiness_20260611`. **Task:** determine the exact
byte-close state of the lever-B score-native carrier, build/recompute the byte-closed advisory S,
verify the EMA-shadow-vs-live export, and produce the paired exact-eval dispatch packet — OR a precise
blocker. **Authority of every number:** `[local CPU-torch advisory]` (exact upstream SegNet/PoseNet on
CPU per the landed candidate manifests; GT via `frame_utils.yuv420_to_rgb`) + `[macOS-MLX
research-signal]` (generator forward). NOT the contest 600-sample harness on 1:1 hardware →
non-promotable per the GOAL authority ladder. `$0` spend, no GPU, no paid dispatch, **NO MPS**.
`promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`.

Frontier read from pointer: **S = 0.19109982 [contest-CPU], 177,169 B, sha `b46897267…`** (rate term
0.11797). T_1 = sub-0.19.

---

## 0. HEADLINE — lever B is NOT an exact-row pointer-mover NOW (decide-don't-defer: NO dispatch)

**The byte-close path EXISTS, is CLEAN, and is already built — but the byte-closed full-S does not
beat the frontier, by a wide margin, for a STRUCTURAL reason that no amount of byte-tuning closes.**

The prompt's premise ("advisory d_seg=0.00826 … IF byte-closed and d_seg/d_pose hold under exact
eval, S drops well below T_1") rests on a category error that the landed evidence (tasks #56/#57/#73,
three independent subagents) already resolved on 2026-06-10:

> The **0.00826 d_seg is the generator's LOGIT-argmax in isolation** — a classifier hitting a frozen
> classifier's argmax. The contest scores **FRAMES**, not logits. When the generator's argmax is
> rasterized into a contest-legal RGB frame (the only thing an archive can ship), TWO things break:
> (1) the rasterized frame's *own* SegNet argmax d_seg rises to 0.064 (palette can't reproduce the
> boundary bands), and (2) a piecewise-constant palette frame is **pose-blind** — d_pose collapses to
> 2.67–12.66 vs the frontier's 2.4e-5. The "d_seg/d_pose hold" assumption is FALSE under frame scoring.

**Two byte-closed candidates already exist** (lossless parity proven, scorer-free numpy inflate):

| candidate | archive sha256 | bytes | rate_term | d_seg | d_pose | seg_term | pose_term | **advisory S** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `score_native_candidate_20260610` (#56) | `1e851e69…` | 72,217 | 0.0481 | 0.0228 | 12.66 | 2.28 | 11.25 | **13.58** |
| `score_native_candidate_pose_20260610T124641Z` (#57) | `b9eb0f6e…` | 85,590 | 0.0570 | 0.0642 | 2.675 | 6.42 | 5.17 | **11.65** |

vs frontier **0.19110**. **Neither beats the frontier; neither hits sub-0.19.** The eval gate
("advisory S beats frontier OR sub-0.15") is NOT met → **correct fail-closed: NO paired exact eval
launched** (do not spend the <$5 budget to confirm a 60×-worse non-improvement). $0 spent.

**The rate class-shift IS real and byte-closed** (−59% bytes, 72KB vs 177KB, rate Δ −0.071). It is
just not, by itself, a score win — the −0.071 rate saving is dwarfed by the +11 to +13 pose/seg
distortion the score-native FRAME representation incurs.

---

## 1. The R-D arithmetic — why even the BEST hypothetical busts T_1 (the decisive math)

The prompt cites "S = 0.120" for lever B. That figure is **mathematically inconsistent with the
generator's own measured d_seg.** Recomputed (`D = 37,545,489`):

- **Generator-d_seg seg term alone:** 100 · 0.00826 = **0.826** — already **4.3× over T_1 (0.19)**,
  before adding ANY pose or rate term. A seg term of 0.826 cannot be inside a sub-0.19 score.
- The "0.120" / "S~0.13-0.16" figures in the smoke verdict + GOAL_v3 scoreboard substitute the
  **frontier's d_seg (5.6e-4)** into the seg term while taking the carrier's rate — i.e. they assume
  the generator trains 15× further down to the frontier's seg level. That is an aspirational R-D
  extrapolation, NOT a measured byte-closed point. With the generator's ACTUAL d_seg (0.00826) the
  best-conceivable carrier (generator seg + a *hypothetical* frontier-level pose 2.4e-5, which the
  palette frame physically cannot deliver) computes to **S = 0.888**, not 0.120.

| scenario | d_seg | d_pose | bytes | **S** | status |
|---|---:|---:|---:|---:|---|
| frontier (the bar) | 5.6e-4 | 2.4e-5 | 177,169 | **0.19110** | the bar |
| smoke-verdict hypothetical (0.120) | **5.6e-4 (frontier, NOT carrier)** | ~0 | 70,452 | 0.120 | aspirational, not measured |
| best-case carrier (gen d_seg + impossible frontier pose) | 0.00826 | 2.4e-5 | 70,452 | **0.888** | physically unreachable (palette pose-blind) |
| **#57 byte-closed (real)** | 0.0642 | 2.675 | 85,590 | **11.65** | MEASURED, fail-closed |

**Lever B has TWO binding blockers, not one:**
1. **POSE (dominant, structural):** the palette/argmax frame1 carries no luma motion → d_pose
   2.67–12.66; pose_term 5–11. (Tasks #56/#57/#73.)
2. **SEG (secondary, also binding):** the generator's d_seg (0.00826) is ~15× the frontier's
   5.6e-4 → seg_term 0.83 alone exceeds T_1, even if pose were free. (This math.)

Both must be solved for an exact-row win. Closing pose alone leaves S ≈ 0.9 (seg-bound).

---

## 2. EMA-shadow-vs-live export verification (the prompt's critical-context check) — CLEAN

The prompt flagged the just-fixed EMA poisoned-tree bug (commit `bf8c43867`): a constant-decay weight
EMA shadow can freeze short-run export at ~init, producing a catastrophic real d_seg.

**VERIFIED CLEAN — there is NO EMA in the lever-B path at all.** `grep -niE "ema|shadow|use_ema|warmup"`
over `tools/lever_b_train_generator_checkpoint.py`, `tools/lever_b_score_native_argmax_smoke.py`, and
`src/tac/boundary_math/lever_b_generator.py` returns **zero matches**. The MLX trainer optimizes LIVE
weights with no shadow; the generator checkpoint and the int8-quantized blob are derived directly from
the live trained weights. The portability parity that the export relies on (`argmax_agreement = 1.0`,
`parity_pass = true`) was measured on those same live weights. **There is no stale shadow that could
be shipped** — the lever-B export is structurally immune to the `bf8c43867` bug class. (The smoke
config carries no `use_ema_for_eval`; the carrier candidates int8-quantize the live MLP. The candidate
manifests' lossless-parity proofs over 4–8 pairs confirm the exported decoder reproduces exactly the
frames that were scored.)

---

## 3. Byte-close state (the prompt's question 1) — COMPLETE and contest-shaped

A builder emitting a contest-compliant `archive.zip` for the score-native carrier ALREADY EXISTS and
has run:
- **Builders:** `tools/score_native_build_byte_closed_candidate.py` (#56, 5-section `SCNP1` monolith:
  seg-cfg+gen, palette, pose-traj) + `tools/score_native_assemble_pose_carrier_candidate.py` (#57,
  adds luma-carrier frame0). Reusable modules: `src/tac/boundary_math/{lever_b_generator,
  legal_frame_bridge,amortized_luma_carrier}.py` (+ 43 behavior tests, NO-FAKE, green).
- **Archive grammar:** single ZIP member `x`, length-prefixed sections, `MAGIC=SCNP1\x00`. Monolithic
  per HNeRV-parity L20.
- **Inflate runtime:** `inflate.py` is **numpy-portable + scorer-free** — verified: `import numpy as np`
  only, NO torch/MLX, NO SegNet/PoseNet loaded ("Both INRs are pure-numpy coordinate nets (portable;
  NO MLX/torch)"). Satisfies the GOAL substrate law + the strict no-scorer-at-inflate rule.
- **Lossless parity:** `all_match=true` over 8 pairs (#56) / 4 pairs (#57) — archive parse-back frame
  sha == direct forward sha. The carrier byte-closes correctly.

So the byte-close is NOT the blocker. The carrier compiles to a clean, portable, scorer-free,
lossless contest archive. The blocker is purely that the carrier's **distortion** (seg+pose) is far
above the frontier.

---

## 4. The paired exact-eval dispatch packet (the prompt's question 3) — TEMPLATE, GATED (not fired)

Per the autonomy contract + NO-FAKE class 8 (no exact dispatch on a candidate that does not beat the
advisory bar), the packet is recorded as a CONDITIONAL template, ARMED for the moment a future
pose+seg fix produces a candidate with advisory S < frontier. Flags verified against real argparse
(`tools/claim_lane_dispatch.py`, `tools/dispatch_modal_paired_auth_eval.py`). **DO NOT run these on
the current candidates — they would burn the budget confirming a 60× non-improvement.**

```bash
# (1) FREE local advisory contest-CPU eval (macOS-CPU = [advisory] not [contest-CPU] per 1:1-hw rule;
#     run it as the advisory predictor on the EXACT bytes that would ship). Needs a submission_dir
#     with inflate.sh + inflate.py + archive.zip + the contest video names file.
.venv/bin/python upstream/evaluate.py --device cpu \
    --submission-dir experiments/results/<candidate_dir>/submission_dir \
    --uncompressed-dir upstream/videos \
    --video-names-file upstream/public_test_video_names.txt \
    --report experiments/results/<candidate_dir>/report_advisory_cpu.txt
# (recompute S from components; the rounded report field lies. GT decode ONLY via yuv420_to_rgb.)

# (2) Claim the lane BEFORE any cloud dispatch (fcntl-locked; refuses same-lane conflict in 24h TTL):
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id lever_b_score_native_paired_exact_eval_<YYYYMMDD> \
    --platform modal --instance-job-id <fc-id-after-spawn> \
    --agent lever_b_byte_close --status eval \
    --notes "score-native carrier paired CPU+CUDA exact eval; sha <archive_sha>"

# (3) Paired CPU+CUDA exact eval on contest hardware (Modal T4), within <$5 budget. --execute fires;
#     omit it for a dry-run plan. Follow Modal .spawn HARVEST-OR-LOSE (harvest within 24h via
#     tools/harvest_modal_calls.py).
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --archive experiments/results/<candidate_dir>/archive.zip \
    --expected-archive-sha256 <archive_sha256> \
    --submission-dir experiments/results/<candidate_dir>/submission_dir \
    --label lever_b_score_native_<YYYYMMDD> \
    --lane-id-base lever_b_score_native_paired_exact_eval \
    --gpu T4 \
    --json-out experiments/results/<candidate_dir>/paired_exact_eval.json \
    --execute
```

**Missing prerequisites for the packet** (the small builds a future fix must add): neither candidate
dir has a `submission_dir/` with `inflate.sh` (only `inflate.py`); the contest CPU eval + the Modal
dispatch both consume a submission tree (`inflate.sh archive_dir output_dir file_list`). That is a
~30-LOC wrapper, deferred — there is no point building it for a candidate that fails the gate.

---

## 5. The precise BLOCKER + smallest next steps (the reactivation path)

**BLOCKER (exact):** the score-native carrier's contest-legal FRAME representation cannot
simultaneously hold (a) the SegNet argmax cell AND (b) the PoseNet tube at low byte. The palette/argmax
frame1 is pose-blind (`src/tac/boundary_math/legal_frame_bridge.py` palette-paint → d_pose 12.66); the
amortized-INR pose carrier hits a 0.0036 ceiling 124× above the tube and is NON-monotone in capacity
(`src/tac/boundary_math/amortized_luma_carrier.py`; task #57 RD sweep). The Dykstra feasibility solve
(task #73) PROVED the cheap-feasible set cell∩tube∩cheap is EMPTY below ~400 KB/pair for a generic
low-rank/sparse basis — the only basis that holds pose under compression at frontier byte is the
**learned HNeRV nonlinear basis the frontier already occupies** (`dykstra_legal_frame.py` returns
delta=0 optimal from the frontier pair). This is HNeRV-parity lesson 5 (full renderer, not
single-component slot), derived not assumed.

**Smallest next steps (priority-ordered; this is lever C, not more lever B):**
1. **Replace the palette frame1 with a per-pair RGB carrier trained JOINTLY against BOTH SegNet
   (d_seg) AND PoseNet (d_pose)** — i.e. a unified frame decoder. Open question: can a per-pair-latent
   *convolutional* (HNeRV-class, not coordinate-MLP) carrier reach frontier d_seg=5.6e-4 +
   d_pose=2.4e-5 at < 177 KB? The coordinate-INR family demonstrably cannot (pose ceiling 0.0036,
   seg floor 0.0068). File: a new conv per-pair decoder; reuse the score-aware loss in
   `tools/score_native_train_luma_carrier.py` (differentiable rgb_to_yuv6 + eval_roundtrip).
2. **Dykstra solve with C = the HNeRV decoder's per-pair latent manifold** (not SVD+sparse): re-run
   `tools/legal_frame_feasibility_smoke.py`'s byte-floor probe projecting δ onto the learned decoder's
   reachable set — that measures the TRUE legal-frame byte floor (task #73 reactivation criterion).
3. **Drive the generator's d_seg from 0.00826 → ~5e-4** (lever C capacity/training-length campaign) —
   necessary regardless, since the seg term alone (0.83) busts T_1 at the current generator.

The honest conclusion (#57 §5, restated): **the score-native seg+palette carrier is DOMINATED on pose
by a full-RGB per-pair decoder, which is the frontier's HNeRV.** Lever B's contribution is the proven
−59% rate headroom + the measured pose/seg geometry; the actual frontier move requires lever C (a
unified score-aware frame decoder smaller than 177 KB), not a better lever-B bridge.

---

## 6. Scoreboard + wire-in

**UPPER:** frontier 0.19110 [contest-CPU], UNMOVED. Lever-B byte-closed advisory S = 11.65 (best),
60× above frontier — NOT a pointer-mover. **LOWER:** S_floor = 0.11797 [advisory] unchanged.

**Wire-in (Catalog #125):** (1) sensitivity-map ACTIVE — the two-blocker decomposition (pose 5–11 +
seg 0.83) is the new seg/pose marginal input. (2) Pareto ACTIVE — the carrier sits OFF the rate vertex
(−59%) but ON a pose cliff; the feasible move is lever C, not a bigger bridge. (3) bit-allocator ACTIVE
— byte breakdown is the literal allocator; next term = a unified appearance section. (4) cathedral
autopilot — gate NOT met, no dispatch. (5) continual-learning ACTIVE — reseeds the V3 judge: lever B's
"S~0.12" is an aspirational extrapolation; the MEASURED byte-closed S is 11.65, pose+seg dominated;
the binding lever is C (unified decoder). (6) probe-disambiguator RESOLVED — "is the byte-closed
lever-B carrier an exact-row pointer-mover NOW?" → NO (S 11.65 ≫ 0.19, two structural blockers); "is
the export EMA-poisoned?" → NO (no EMA in the path); "is the inflate numpy-portable + scorer-free?" →
YES.

## 7. Cross-references
`lever_b_score_native_argmax_smoke_verdict_20260610.md` (the mechanism proof — logit, not frame) ·
`score_native_first_candidate_20260610T112433Z.md` (#56, byte-closed S=13.58, pose collapse) ·
`score_native_pose_carrier_20260610T125000Z.md` (#57, byte-closed S=11.65, RD ceiling + frame1-dual
diagnosis) · `legal_frame_feasibility_dykstra_20260610T175421Z.md` (#73, cheap-feasible set empty →
HNeRV basis required) · `GOAL_standing_v3_20260610.md` (lever menu; C is the reactivation) ·
`src/tac/boundary_math/{lever_b_generator,legal_frame_bridge,amortized_luma_carrier,dykstra_legal_frame}.py`
· `experiments/results/score_native_candidate_{20260610,pose_20260610T124641Z}/` (the byte-closed
archives + scorer-free inflate + parity proofs).
