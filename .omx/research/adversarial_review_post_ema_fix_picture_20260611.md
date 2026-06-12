# Adversarial review of the post-EMA-fix picture — second set of eyes (2026-06-11)

**Role:** Contrarian + Assumption-Adversary. **Authority of every number cited:** as-tagged from the
source artifact (`[contest-CPU]` exact only for the pointer; everything capstone is `[macOS advisory]` /
`[macOS-MLX research-signal]`, `score_claim=false`). **Spend:** `$0`, read-only, no GPU, no code change,
no MPS. The exact pointer **0.19109982 [contest-CPU], 177,169 B, sha `b46897267…`** is UNMOVED; nothing
here moves it. This review CHALLENGES the current honest picture before a long frontier-class GPU train.

**One-line frame:** the EMA fix is real and the picture is *more honest* than a week ago — but the
session has swapped one over-claim ("seg-walled") for a softer one ("pose solved, d_seg the only wall"),
and the single most dangerous gap (advisory↔exact eval decouple) is still OPEN in the very daemon whose
trajectory is being used to justify the GPU train.

---

## (a) CLAIM-BY-CLAIM VERDICT TABLE

| # | Claim | Verdict | Why / cheapest decisive check |
|---|---|---|---|
| 1a | "EMA warmup fix is correct" | **SOUND** | `warmup_ema_decay = min(d,(1+t)/(10+t))` is the canonical timm/diffusion warmup; correct for a weight-INIT EMA (Adam `/(1−d^t)` bias-correction would be WRONG — it assumes zero init). Diagnostic table (GAP 0.466→0.023) is a clean before/after. Two NO-FAKE guard tests assert ramp + tracking. |
| 1b | "consolidating 7 EMAs introduced no regression / every weight-EMA routes through warmup" | **OVERSTATED** | `tac.training.EMA.update()` (`src/tac/training.py:528`) STILL uses constant `self.decay` — it does NOT call `warmup_ema_decay`. The memo's "every weight-EMA in the repo routes through warmup_ema_decay" is false for the canonical TORCH EMA (used by `self_compress`, `segmap_renderer`, `psd_lumaskip_renderer`, `joint_scorer_aware_training`, `nerv_mask_codec`, …). Check: `grep -n "warmup_ema_decay" src/tac/training.py` → only a docstring ref, no call. |
| 1c | "the held-back torch EMA is dormant (no active poisoned short torch run)" | **SOUND-for-now, FRAGILE** | No torch trainer daemon is running (only the MLX capstone daemon + one Modal *eval* probe `pr110pp`, which is an eval not a train). BUT the torch EMA is one short `train_*.py` launch away from re-poisoning. There IS a partial mitigation (`EMA.decay_from_total_steps`, window-vs-total matching) but it is OPT-IN and does NOT fix the short-run lag the way warmup does. Check: before ANY torch train, `grep "EMA(" <trainer>` and confirm it either warms up or sets decay from total steps. |
| 1d | "no OTHER measurement poisons lurk (int8 export/reload, bicubic-vs-bilinear, d_pose roundtrip, GT decode)" | **PARTLY OPEN — the real risk** | d_pose-roundtrip poison FIXED (now routes `bridge.exact_d_pose` w/ same bicubic-up/bilinear-down/uint8 roundtrip, `capstone_trainer.py:408-412`). BUT **A2 (score the RELOADED int8 archive) and A3 (bicubic-vs-bilinear inflate parity) are NOT closed in the RUNNING daemon** — `exact_d_seg`/`mean_d_pose` measure the LIVE float MLX render (`capstone_trainer.py:389 self._render(...)`), not the int8 archive that ships. The `ReloadedInt8Advisory` module (`advisory.py`) EXISTS but is a separate post-hoc step the daemon does not call. So the daemon's d_seg=0.0198 / d_pose=4.4e-4 are float-weight numbers, NOT archive numbers. **This is Conclusion-3 [E1] — explicitly flagged "no capstone advisory number is a trustworthy inflate.sh→evaluate.py predictor today" — and it is STILL TRUE for the daemon feeding the GPU decision.** Cheapest check: run `score_reloaded_int8_archive` on the daemon's current export and diff vs live (one `$0` call). |
| 2a | "Pose is SOLVED (d_pose ~1e-3 via stored_latent carrier)" | **OVERSTATED + MISATTRIBUTED** | Pose is NOT carried by the stored 28-d latent. It is carried by a **separate stored 6-dim GT pose-store** (`export.py:13` `(u32 pose_len, pose_blob)`, 600×6 fp16 ≈ 7.2 KB raw → brotli ~kilobytes) **FiLM-injected** on per-frame features (`vq_nerv_bundle.py:16,228`). This is Quantizr's store-the-answer trick, not a learned/synthesized pose. So "the latent solves pose" is wrong; "we STORE the GT pose and FiLM it in" is right — and that carries a per-pair BYTE COST the carrier-design memo's "decoupled pose-store fallback" acknowledges (+1,557 B at the floor estimate, more in practice). |
| 2b | "d_pose will hold at 600 pairs / under int8 / under real PoseNet" | **UNVERIFIED** | 48 pairs, float render, proto/CPU scorer. Two independent risks: (i) FiLM-injected GT pose must survive PoseNet reading the *rendered low-fidelity luma* — Conclusion-2 warns "a FiLM over content cannot synthesize the ego-motion flow FastViT reads"; here it's GT-scalar-FiLM, less risky, but UNPROVEN on a small decoder's render. (ii) int8 quant of the decoder perturbs the render PoseNet reads. Check: `score_reloaded_int8_archive` d_pose at 600 pairs vs live. |
| 2c | "d_pose 4.4e-4 is a real per-pair mean, not a few-pair artifact" | **UNVERIFIED** | The daemon logs only the MEAN. d_pose is a GLOBAL-pool MSE with a concave √ — a handful of bad pairs can dominate sqrt(10·mean). Check: log/inspect the per-pair d_pose distribution (max, p95) for one eval, not just the mean. ~30 min, $0. |
| 3a | "the small 85K basis plateaus ~0.02 (d_seg-walled)" | **FALSE / already breaking** | The LIVE post-fix daemon: stage1 CE → d_seg **0.0198**; stage2 softplus **starts at 0.0165 and is descending** (`trajectory.jsonl` last row). The "plateau ~0.02" is a stage-1-CE artifact; the curriculum's loss-form schedule is re-accelerating exactly as the synthesis memo's OPEN question predicted it might. Stages 2–8 (softplus→smooth→L7→…) are UNTESTED but the trend is down, not flat. Do NOT call 0.02 a wall. |
| 3b | "frontier needs 162–229K params for d_seg=5.6e-4; 85K fights the physics; need frontier-class params" | **OVERSTATED (rests on a PROJECTION, not a curve)** | The "256K → 2× shrink to 2.8e-4" number is a **2026-05-09 grand-council VOTE/projection** (`grand_council_fields_medal_theoretical_floor_20260509.md:220`), NOT a measured param↔d_seg curve. The frontier's 5.6e-4@~177KB IS an exact-CPU row, but "you need ≥162K to reach it" is an inference from the leaderboard CLUSTER, not a controlled capacity sweep we ran. The poisoned-tree audit itself REOPENED this (claim #5 "UNCERTAIN→REOPENED"). The honest statement: "CE-only plateaus high; whether base_ch=20 reaches 5e-4 under the full curriculum is the open measurement the daemon is running." |
| 3c | "argmax-CE is the right objective" | **QUESTIONABLE — under-explored** | d_seg is an argmax-FLIP RATE concentrated at razor-thin SegNet boundaries (lever-G diag: disagree px have top1−top2 median 0.156). CE on the whole frame spends gradient on the 95%+ already-correct interior. A boundary-weighted / margin hinge (lever-G margin field, only at inflate-forbidden boundaries — but available at TRAIN time) is the architecture×objective dual the GOAL names (J_scorer·J_renderer in rewarded directions). The curriculum's softplus/smooth/L7 stages are a partial move here; a margin objective is untested at 85K. This is the highest-upside UNVERIFIED on the seg axis. |
| 4 | "Lever B is a dead negative" | **SOUND** | Robustly measured by 3 subagents (#56/#57/#73). Two STRUCTURAL blockers: (i) seg term 100·0.00826 = **0.826 alone busts T_1** (the "S=0.12" figure substituted the FRONTIER's d_seg into the carrier's rate — an aspirational extrapolation, not a measured point); (ii) palette frame1 is pose-blind (d_pose 2.67–12.66). Dykstra solve (#73) proved cell∩tube∩cheap empty below ~400 KB/pair for a generic basis. The −59% rate headroom is real but dominated by +11 pose. A hybrid (B seg-blob ⊕ C pose decoder) is NOT a fix — it still ships the pose-blind palette frame1; the pose decoder would have to render frame1 anyway, at which point it IS lever C. Correctly fail-closed (no dispatch). |
| 5 | "a long frontier-class HiNeRV GPU train is the highest-EV next exact-row action" | **OVERSTATED / PREMATURE** | See §(c). The frontier is only **0.0011 above T_1** — the cheapest sub-0.19 row is NOT a fresh long train; it's closing the advisory↔exact gap on the carrier already descending, OR an entropy/rate recode on the CURRENT frontier. A long GPU train before A2/A3 are closed risks chasing a float-render number that the int8 archive does not honor. |

---

## (b) TOP 3 HIDDEN RISKS THAT COULD WASTE A LONG GPU TRAIN

1. **The advisory↔exact decouple is STILL OPEN in the daemon (the #1 risk).** Every d_seg/d_pose the
   GPU-train decision rests on is the **LIVE float MLX render**, not the **int8 archive** that ships,
   and the inflate camera-upscale (bilinear) differs from train/eval (bicubic). Conclusion-3 named this
   [E1] and said it was a launch-blocker; it has NOT been closed for the running daemon (the
   `ReloadedInt8Advisory` exists but is not in the loop). A long train that optimizes the float number
   can land an int8 archive 2–3× worse on d_seg at the razor-thin boundaries where the whole residual
   lives. **Close it FIRST with a `$0` reloaded-int8 + bicubic-inflate smoke on the current export.**

2. **Pose is stored, not learned — and its byte cost + int8 survival are unmeasured at 600 pairs.** The
   "pose solved" headline hides that pose = a stored GT pose-store (~kilobytes/600 pairs) + FiLM, and
   that the FiLM'd pose must survive (a) int8 decoder quant and (b) PoseNet reading a *smaller* decoder's
   render. If FiLM-pose degrades on the shrunken decoder's render (the exact Conclusion-2 failure mode,
   in a milder form), the train produces a carrier that holds seg but not pose — the LEVER-B failure mode
   wearing a different hat. The 48-pair float number does not derisk this.

3. **The capacity thesis flips on a projection, not a curve — risking the WRONG param budget.** If the
   GPU train is sized "frontier-class (≈100–160K) because 85K fights the physics," that sizing rests on a
   2026-05-09 council vote, not a measured sweep, AND it is contradicted by the live daemon (85K base_ch=20
   is at d_seg 0.0165 and still falling under the curriculum). Committing GPU $ to a bigger net for the
   wrong reason wastes both the bytes (rate term) and the train. The daemon's full-curriculum LIVE d_seg
   is the measurement that should SET the param budget — let it finish (or run the controlled capacity A/B)
   before sizing the train.

---

## (c) RANKED NEXT EXACT-ROW LEVERS BY EV (most-likely-to-land-sub-0.19-soonest)

The frontier is **0.0011 above T_1**. The fastest sub-0.19 is the smallest honest move, not the biggest.

| rank | lever | predicted ΔS | cost | next command (verify flags first) |
|---|---|---|---|---|
| **1** | **Close the advisory↔exact gap on the daemon's current export** (A2 reloaded-int8 + A3 bicubic inflate + `inflate.sh→evaluate.py` smoke on a tiny real archive). NOT itself a pointer-mover, but it is the GATE that makes every downstream number real and is the cheapest way to find out if the descending carrier is already a sub-0.19 candidate. | gates ±0.01–0.05 of phantom | `$0`, ~1h | `.venv/bin/python -c "from tac.capstone_vq_nerv.advisory import score_reloaded_int8_archive; ..."` on `capstone_c1prime_honest_b20_n48` export; then `upstream/evaluate.py --device cpu --submission-dir …` advisory on the byte-closed archive (recompute S from components). |
| **2** | **Rate/entropy recode on the CURRENT 177,169 B frontier** (the cleanest sub-0.19 that does not need a train). The GOAL scoreboard says frozen-byte rate is "EXHAUSTED at lossless" — VERIFY that claim is current, because even a small lossless rate cut on 177 KB at constant distortion lowers S directly (25/D = 6.66e-7 ΔS/byte; ~1,650 B ≈ −0.001 = crosses T_1). | −0.001…−0.003 if any headroom | `$0`–$0.6 | re-run the latent/decoder entropy probe on the frontier member-x; if any coder beats the current section, byte-close + ONE paired exact eval. |
| **3** | **Let the daemon finish stages 2–8, then byte-close + ONE paired CPU+CUDA exact eval IF advisory S (on the int8 archive) beats frontier.** This is the real candidate the session is building — but it is rank 3, not the default, because it is gated on rank-1 (the int8/bicubic gap) and its advisory must beat the bar BEFORE any spend (NO-FAKE class 8). | carrier predicted ~0.11–0.15 IF distortion closes (advisory) | `$0` train (running) + $0.6 eval | harvest `DONE_MARKER`; `score_reloaded_int8_archive`; if S<frontier → `tools/dispatch_modal_paired_auth_eval.py --execute`. |
| **4** | **Margin/boundary-weighted seg objective at the current 85K** (the architecture×objective dual). Highest seg-axis upside, cheap, MLX-local — the residual lives at SegNet boundaries and CE under-spends there. Run as a controlled A/B vs the CE curriculum at identical config. | could break d_seg below 0.003 (advisory) | `$0`, MLX | new loss arm in the curriculum; A/B at matched pairs/base_ch/epochs/seed, report last-10-eval mean LIVE + int8 d_seg. |
| **5** | **The long frontier-class fresh-init train (lever C / C1′)** — the class-shift bet. Genuinely the innovation-gate winner and the sub-0.118 path, BUT it is the most expensive and most uncertain, and should be sized by the daemon's measured capacity curve (rank 1+3), not launched on the projection. Do it AFTER 1–4 have either landed sub-0.19 or proven they can't. | −0.04…−0.08 IF the smaller amortizer holds the cell (advisory band, unproven) | $1–5 GPU | only after the int8-honest capacity knee is measured; pre-register the §3 band as the falsifiable prediction. |

---

## SUMMARY (the second set of eyes, 8 lines)

The EMA warmup fix is genuinely correct and the picture is more honest than last week — but the session
traded the retracted "seg-walled" over-claim for a NEW soft one. **The single biggest over-claim:
"pose is solved."** Pose is not solved — it is *stored* (a 6-dim GT pose-store + FiLM, Quantizr's
store-the-answer trick), measured on 48 pairs of LIVE FLOAT render with a proto scorer, never on the
int8 archive at 600 pairs; "the latent solves pose" is a misattribution and the byte cost scales with
pairs. **The single biggest hidden risk: the advisory↔exact decouple [E1] is STILL OPEN in the very
daemon feeding the GPU decision** — d_seg=0.0198 / d_pose=4.4e-4 are float-MLX-render numbers, not the
int8 + bicubic-inflate archive that ships, so a long train can optimize a phantom and land 2–3× worse at
the boundary pixels where the entire residual lives. **The #1 ranked next lever is therefore NOT the long
GPU train — it is the `$0` reloaded-int8 + bicubic-inflate smoke that turns every capstone number real**;
the frontier is only 0.0011 above T_1, so the cheapest sub-0.19 is closing that gap on the already-
descending carrier (and re-checking the "rate exhausted" claim), not a fresh frontier-class train. Two
smaller corrections: "85K plateaus at 0.02" is already FALSE (stage 2 is at 0.0165 and descending), and
the "need frontier-class params" sizing rests on a 2026-05-09 council projection, not a measured curve —
let the daemon's full-curriculum LIVE-and-int8 d_seg SET the param budget before spending GPU on it.
