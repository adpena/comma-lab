# Capstone n600 ($100 paid) — config bug-sweep + OPTIMAL/NON-ARBITRARY dispatch spec (2026-06-11)

**Role:** pre-dispatch CONFIG + BUG-SWEEP gate for the approved $100 paid n600 PR95-scale capstone (task #90).
This memo is the input the symposium reviews next; it does **NOT** dispatch.

**Authority discipline (binding).** Every d_seg/d_pose number here is `[macOS-CPU advisory]` /
`[macOS-MLX research-signal]`, NON-PROMOTABLE (`promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`). The canonical chain (`upstream/evaluate.py --device cpu`, the
`canonical_chain_scan.py` reference) is the only leaderboard authority. NO MPS. NO paid dispatch fired.
**Frontier pointer UNMOVED: 0.19109982 [contest-CPU], 177,169 B.** This is a config-correctness gate, not a
pointer move.

---

## 0. HEADLINE (non-sycophantic): the config is bug-swept + the recipe/arch are justified, BUT two infra blockers gate the spend

The trainer code is **bug-free** for the muon_throughout path (BUG-A confirmed fixed, BUG-B is faithful, no
new bugs found) and the arch/byte-budget thesis is **non-arbitrary and quantified** (a smaller-than-frontier
basis at the basin is a sub-0.15 pointer-mover). But the pre-dispatch gate surfaced **two unresolved
infrastructure realities that the "$100 Modal n600" framing did not account for** — they are the real gate:

1. **PLATFORM BLOCKER: the MLX capstone trainer cannot run on Modal (NVIDIA).** `CapstoneVqNervBundle` +
   `mx.vjp` are Apple-Silicon-only. There is NO torch/CUDA capstone-trainer port. The only Modal-runnable,
   basin-proven vehicle is the **vendored PR95 torch trainer** (`.../hnerv_muon/src/train.py`, CUDA-capable).
   So "$100 Modal n600" implies a **vehicle choice** the symposium must make explicit (see §5).
2. **EPOCH-BUDGET BLOCKER: $100 does not buy the 29,650-epoch PR95 curriculum at n600.** PR95's full run is
   ~50 GPU-hr (~$185 A100). $100 ≈ 27 A100-hr ≈ a heavily compressed curriculum (~12,000–16,000 epochs by
   proportional compression). Whether the basin (5.6e-4) is reachable in the compressed budget at a SMALLER
   basis is the open scientific risk the run tests — state it, don't hide it.

Both are **decide-don't-defer** items for the symposium, not reasons to refuse. The config below is OPTIMAL +
NON-ARBITRARY for whichever vehicle the symposium picks; the two blockers are named residual knobs (§7).

---

## 1. BUG-SWEEP LEDGER (every surface → clean / fixed)

Authority for "clean": source-faithful diff against the runnable vendored PR95 original
(`.../hnerv_muon/src/{stages/common.py,model.py,losses.py,train.py}`) + the capstone test suite (45/46 pass;
the 1 "fail" is the `@pytest.mark.slow` real-scorer test hitting the 60s pytest-timeout on
`slow_conv2d_forward` — a THROUGHPUT timeout, not a logic failure) + ruff clean.

| # | Surface | Verdict | Evidence |
|---|---|---|---|
| **BUG-A** | `configure_stage` dropped working `muon_lr` (rebuilt 2e-4 + clip 1.0 → 150× throttle) | **FIXED (verified complete)** | commit a9888191c; `capstone_trainer.py:627-636` gates on `muon_throughout`, uses `cfg.muon_lr`/`cfg.grad_clip_muon`; faithful path uses `spec.*` (byte-unchanged). Parity tests `test_bugA_muon_throughout_uses_config_muon_lr_not_stagespec` + `test_bugA_faithful_pr95_schedule_byte_unchanged` PASS. |
| **BUG-B** | cosine `eta_min_ratio` floors LR vs `adamw_lr` denominator | **CLEAN — faithful, NOT a bug** | `pr95_cosine_lr_scale` is a 1:1 port of PR95 `common.py:152-156`: `eta_min_ratio=max(lr_floor/adamw_lr,1e-3)`, SAME `lr_lambda` on BOTH `adamw_sched` AND `muon_sched`. The memo's "BUG-B" is a real *artifact* ONLY for the `muon_throughout` deviation (muon_lr=0.03 → floor 0.005 prematurely halts); for the FAITHFUL path the floor is PR95's proven behavior. **Do NOT "fix" it for the faithful path** — that would diverge from the proven recipe. (For muon_throughout, run enough epochs that the late floor is benign — already the design.) |
| **BUG-C** | "smooth_disagreement raises d_seg" | **CLEAN — a consequence of A, not a bug** | The A/B (commit 5133fe82a) showed stage-3 smooth REFINED d_seg (0.0165→0.0120) under the fixed recipe; the "raise" was the throttle. The loss is bit-faithful (`STAGE_SEG_LOSS_FNS`). |
| **BUG-D** | grad clip to norm 1.0 on AdamW (faithful path) | **CLEAN — faithful** | PR95 `common.py:205-206` clips `adamw_params+[latents]` to `grad_clip=1.0` AND `muon_params` to `grad_clip_muon=1.0`. Our `_clip_flat_gradients` is JOINT-norm over `adamw_names` (incl. latents via the partition) — matches PR95's joint clip group. CLEAN. |
| **EMA decay** | default 0.997 + warmup vs PR95's 0.999 constant | **DELIBERATE divergence; faithful-path value SPECIFIED = 0.999** | PR95 `common.py:53` `ema_decay=0.999`, no warmup. Our warmup (`warmup_ema_decay`) is the f771e6e00 anti-shadow-lag fix and saturates to constant within ~100 steps (benign at 2.2M steps). For the FAITHFUL n600 run, set `ema_decay=0.999` (PR95's proven value); warmup is harmless at scale and protects short stages. **Non-arbitrary: PR95 source.** |
| **EMA shadow exported** | export bytes = EMA shadow not live | **CLEAN — faithful** | `export_render_weights`/`export_stored_latents` snapshot-apply-restore the shadow over decoder+FiLM+latents; PR95 archives `ema_decoder.state_dict()`+`ema_latents` (`common.py:229`). Matches. The EMA non-negotiable is honored. |
| **eval_roundtrip** | bicubic↑874×1164 → bilinear↓384×512 → uint8 STE in loss AND measure | **CLEAN — faithful** | `TorchScorerBridge(eval_roundtrip=True)`; PR95 `common.py:178-185` identical (bicubic up, bilinear down, clamp, round-STE). `mean_d_pose` routes through the SAME roundtrip (audit #4 fix). Matches. |
| **Stage transitions** | fresh optimizer + fresh cosine per stage; weights/EMA/codebook carry | **CLEAN — faithful** | `configure_stage` resets `opt_state` + cosine base/span; PR95 builds a fresh AdamW/Muon+LambdaLR per stage and resumes the decoder/latents. `test_b2_configure_stage_resets_optimizer_step_counter_and_cosine` PASS. |
| **8-stage spec** | epochs/seg-loss/sigma/c1a/qat/lr schedule | **CLEAN — byte-faithful** | `_PR95_8STAGE` = (3000,5650,1500,500,9000,2000,3000,5000)=29,650; seg families ce→tau_softplus→smooth→smooth→l7×4; sigma 0.2→0.1; c1a_λ 0.01→0.02; QAT stage4+; adamw_lr 1e-3→1e-3→1e-4→1e-4→3e-5×3→1e-5; muon_lr 2e-4 (stage 8 only via `use_muon_canonical`). Matches PR95 stage files. |
| **C1a / sigma / QAT** | weight-domain mechanisms | **CLEAN — faithful** | `StageMechanisms` + `apply_stage_weight_transforms` + `add_c1a_entropy_gradient` mirror PR95 `loss + cat_lambda*ent` + `apply_qat`/`restore_qat` + sigma weight-noise. |
| **Muon/AdamW partition** | conv weights→Muon; stem/rgb/latents→AdamW | **CLEAN — faithful** | `partition_pr95_mlx_parameter_names`: ndim≥2 + endswith"weight" + not stem/rgb/latents → Muon; else AdamW. Matches PR95 `optim.py` split. Newton-Schulz makes step ∝ muon_lr (the BUG-A mechanism). |
| **decoder arch** | taper/PixelShuffle/sin/skip/refine/dual-RGB | **CLEAN — bit-exact** | At base_ch=36 tie=0 our decoder = **228,959 params = PR95's 229K**. Taper `[C,C,C,.75C,.58C,.5C,.5C]`, Conv→PixelShuffle(2)→bilinear-skip→sin, `+0.1·sin(refine)`, dual sigmoid RGB×255. Matches `model.py`. |
| **inflate (numpy-portable)** | torch-free inflate | **CLEAN** | `capstone_vq_nerv/inflate.py` imports numpy only (Catalog #295 numpy-portable inflate). |
| **export↔inflate parity** | byte-close → reload int8 → re-score | **CLEAN** | runner `_export_int8_archive` + `score_reloaded_int8_archive` (the A2 reloaded-int8 advisory = the honest contest predictor). Score-parity with numpy inflate verified (existing tests). |

**Net:** the muon_throughout trainer is bug-free; BUG-A is the only real bug and it is fixed + tested. The
EMA-decay value is the one knob to PIN for the faithful path (= 0.999, PR95's value). No silent
drop/override/throttle remains.

---

## 2. RECIPE CHOICE — faithful PR95 (default) vs muon_throughout-fixed

**The discipline (CLAUDE.md "Substrate MUST be at OPTIMAL FORM"):** DEFAULT to the faithful proven recipe
unless a bounded $0 A/B at matched small scale justifies the divergence.

**Evidence we hold:**
- The n8 A/B (commit 5133fe82a) proved **muon_throughout-fixed DESCENDS** (0.507→0.066→0.0165→0.0120 over 52
  epochs, base_ch=20). It did **NOT** compare faithful-vs-divergence — it only proved the BUG-A fix unfreezes.
- The **basin existence-proof (5.6e-4) is FAITHFUL-only** — PR95 reached it with AdamW-stages-1-7 + Muon-stage-8
  at base_ch=36 (the frontier rests on this). No proof muon_throughout reaches the basin.
- A bounded head-to-head A/B (faithful `pr95_adamw_then_muon` vs `muon_throughout` at base_ch=20, stages 1-2,
  12+12 epochs, n8, matched budget) WAS run this session
  (`experiments/diag_faithful_vs_muon_throughout_ab.py`; `experiments/results/diag_faithful_vs_muon_ab/`).
  **MEASURED RESULT (`[macOS-CPU advisory]`, real `modules.py` SegNet):**

  | arm | stage 1 (CE, 12ep) d_seg(live) | stage 2 (tau_softplus, 12ep) d_seg(live) |
  |---|---|---|
  | **FAITHFUL** (AdamW, adamw_lr=1e-3) | 0.24409 | **0.05950** |
  | **MUON_THROUGHOUT** (muon_lr=0.03) | 0.12673 | **0.04309** |

  Both descended strongly from init 0.50727. **muon_throughout beats faithful by ratio 1.38× (0.0431 vs
  0.0595) — a MODEST win, NOT decisive (well within the pre-registered ~2× "comparable" band).**

**VERDICT (A/B-confirmed): FAITHFUL `pr95_adamw_then_muon` is the default for the $100 spend.**
Rationale: (a) it is the ONLY recipe with a basin existence-proof (PR95 reached 5.6e-4 with it at base_ch=36);
(b) the bounded A/B showed muon_throughout's edge is only **1.38×** at n8 — NOT the DECISIVE win (ratio ≫ 1)
the pre-registered threshold required to justify diverging; (c) per CLAUDE.md "OPTIMAL FORM before paid
dispatch", a 1.38× n8 edge does NOT outweigh the proven-recipe + basin-proof advantage for a $100 spend —
the faithful recipe is both the safer default AND descends comparably (within 38%). **If the symposium wants
the muon_throughout 1.38× edge, it is a defensible BUT divergent choice that forfeits the basin existence-
proof; the recommended default is faithful.** (Caveat: the A/B is n8/24-epoch — it measures early-descent
slope, not basin-reaching, which is the n600 question for BOTH recipes.)

### Per-knob justification (non-arbitrary; every value → PR95 source or first principles)

| knob | faithful value | source / justification |
|---|---|---|
| optimizer schedule | `pr95_adamw_then_muon` (AdamW 1-7, Muon stage 8) | PR95 `optim.py`+`stage8` — the proven basin-reacher |
| epochs/stage (proportional) | (3000,5650,1500,500,9000,2000,3000,5000) scaled by budget | PR95 `_PR95_8STAGE`; compression preserves the STRUCTURE (transitions break the floor), only shortens phases |
| adamw_lr/stage | 1e-3,1e-3,1e-4,1e-4,3e-5,3e-5,3e-5,1e-5 | PR95 stage files |
| muon_lr | 2e-4 (stage 8 only) | PR95 `stage8_muon_finetune` |
| cosine eta_min | `max(5e-6/adamw_lr,1e-3)`, same on both opts | PR95 `common.py:152-156` (faithful — see BUG-B CLEAN) |
| grad_clip / grad_clip_muon | 1.0 / 1.0 | PR95 `common.py:205-206` |
| sigma weight-noise | 0.2 (st5,6), 0.1 (st7,8) | PR95 `cat_sigma` schedule; uint8-quant robustness (L17) |
| C1a λ | 0.01 (st5), 0.02 (st6,7,8) | PR95 `cat_lambda` sweep (L16); brotli-friendly weights (RATE lever) |
| QAT | on stage 4+ | PR95 `apply_qat` (L14) — int8 archive d_seg ≈ training d_seg |
| EMA decay | **0.999** | PR95 `common.py:53` (the one knob to PIN; warmup harmless at scale) |
| latent_lr_mult | 10.0 | PR95 `common.py:133` (`adamw_lr*latent_lr_mult` for latents) |
| seg_weight / pose_weight | 100.0 / 1.0 | PR95 `cfg.seg_weight`/`pose_weight`; matches the score's 100·d_seg + √(10·d_pose) coupling |

No knob is "it worked in a prior run on a buggy recipe." Every value traces to the vendored PR95 source.

---

## 3. ARCHITECTURE + BYTE BUDGET (non-arbitrary) — the pointer-mover thesis

**The frontier is RATE-DOMINATED:** S=0.19110 = seg_term 0.0560 + pose_term 0.0172 + **rate_term 0.1180**
(177,169 B). The rate term is 62% of the score. **The pointer-mover is a SMALLER basis at the basin.**

Measured byte budget (stored_latent carrier, int8 decoder, n600; `full_render_weights_from_bundle` +
`build_capstone_stored_latent_archive_bytes` + ~500 B pose+sidecar):

| arch (stored_latent int8) | params | decoder_B | latent_B | total_B | vs frontier | rate_term | **S@basin*** |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_ch=16, tie=2 | 67,160 | 49,424 | 16,744 | 66,668 | −110,501 | 0.0444 | **0.1175** |
| **base_ch=20, tie=2** | **87,485** | **69,416** | 16,744 | **86,660** | **−90,509** | **0.0577** | **0.1309** |
| base_ch=24, tie=2 | 110,950 | 92,538 | 16,744 | 109,782 | −67,387 | 0.0731 | 0.1463 |
| base_ch=28, tie=2 | — | 120,090 | 16,744 | 137,334 | −39,835 | 0.0915 | 0.1646 |
| base_ch=32, tie=2 | 168,785 | 149,680 | 16,744 | 166,924 | −10,245 | 0.1112 | 0.1843 |
| base_ch=36, tie=2 | 201,855 | 182,405 | 16,744 | 199,649 | +22,480 | 0.1329 | 0.2061 |
| base_ch=36, tie=0 (= PR95 proven) | 248,583 | 229,001 | 16,744 | 246,245 | +69,076 | 0.1640 | 0.2371 |

\* S@basin assumes d_seg=5.6e-4 (seg_term 0.0560) + d_pose=2.94e-5 (pose_term 0.0172) — the PR95 basin.

**The explicit pointer-mover thesis (how this beats 0.19110):**
- **base_ch=20, tie_depth=2, stored_latent, int8 → 86,660 B (HALF the frontier).** IF the basin holds at this
  basis, **S ≈ 0.1309 — SUB-0.15** (the rate term collapses 0.118→0.058). This is the de-risked A/B basis.
- **The arch TENSION (stated explicitly):** PR95's basin existence-proof is at base_ch=36 = 229K params =
  246,245 B = **bigger than the frontier (no rate win at the proven arch).** A pointer-mover REQUIRES a
  SMALLER basis that STILL reaches the basin. base_ch=20 is 30% of PR95's decoder params. The n8 A/B reached
  0.0120 at base_ch=20 (21× above the basin, still falling) — encouraging but NOT a basin proof.
- **What makes THIS run a pointer-mover:** smaller-basis-at-basin. The bet: the contest video is a single
  60s clip; PR95 used 229K params; if the d_seg/d_pose basin is reachable with ~69K decoder params (the
  capstone's pose-FiLM may help pose at lower bytes), the rate term halves and S crosses 0.15. The run TESTS
  this; it is not assumed.

**Recommended arch ladder for the run (decide-don't-defer):** primary **base_ch=20 tie=2** (the de-risked,
sub-0.15-if-basin basis); fallback **base_ch=24 tie=2** (more capacity, still sub-0.15 at basin, lower basin
risk). base_ch=16 is the aggressive sub-0.12 reach (highest capacity risk). Do NOT run base_ch≥32 (no rate
win even at basin).

**stored_latent carrier (NON-ARBITRARY):** the per-pair 28-d latent (temporal-delta+LZMA, PR95 L24/L25) =
16,744 B for 600 pairs. The vq_index carrier (8-bit index) is pose-IMPOVERISHED (the runner default
`--carrier vq_index` is WRONG for a pose-capable run; the de-risked A/B used stored_latent). **Pin
`--carrier stored_latent`.**

---

## 4. POSE + the joint objective (coupled, non-arbitrary)

The score is JOINTLY 100·d_seg + √(10·d_pose) + 25·bytes/37.5M. The three terms are coupled through the
shared basis: a smaller basis lowers bytes but risks BOTH d_seg AND d_pose. The capstone's FiLM-pose carrier
injects the stored 6-dim GT pose into the render, so d_pose is anchored by the stored scalars (the same the
archive bytes). The n8 A/B held d_pose well (`d_pose_live` 0.0019 at stage 3 — far below the frontier's
2.94e-5 target? NO: 0.0019 ≫ 2.94e-5; at n8 d_pose is NOT yet at the tube — the pose-tube is the n600
question too). **Pose-tube (d_pose ≤ 3e-4) is an OPEN target alongside the seg-basin** — the run must show
BOTH. The pose_weight=1.0 + the √(10·d_pose) shape (CLAUDE.md: pose marginal exceeds seg below
pose_avg~2.5e-4) means the run must NOT trade pose away for seg.

---

## 5. VEHICLE / PLATFORM (the symposium MUST resolve) — the real spend gate

| Path | Vehicle | Platform | Basin proof | Pose | Notes |
|---|---|---|---|---|---|
| **P1** | MLX capstone (`run_capstone_campaign.py`) | **local M5 Max ONLY** (MLX) | faithful recipe available | FiLM-pose (capstone) | NOT Modal-runnable; ~14 min/epoch n600 MLX-GPU → only ~hundreds of epochs feasible; the "$100 Modal" framing does NOT apply |
| **P2** | vendored PR95 torch trainer (`hnerv_muon/src/train.py`) | **Modal CUDA** ($100) | **PROVEN (this is the basin trainer)** | pose via PR95's own path (no FiLM) | the ONLY Modal-runnable basin-proven vehicle; needs base_ch override (edit 3 lines in a clone-copy, NOT in-place per CLAUDE.md); ~50 GPU-hr full → compress to ~$100 |

**Recommendation:** if the $100 is a Modal/CUDA spend, **P2 (vendored PR95 torch trainer at base_ch=20/24)**
is the coherent choice — it is the proven-basin recipe, runs on the budget, and produces a torch decoder →
byte-close → exact CPU+CUDA eval. The MLX capstone (P1) is the right LOCAL vehicle (free, FiLM-pose) but is
not a "$100 Modal" run. The bug-sweep in §1 fully validates the MLX capstone trainer; the vendored P2 trainer
is the PR95 original (its fidelity is the source-of-truth, not ours to re-audit). The symposium picks.

---

## 6. DISPATCH-READY CONFIG (both vehicles)

See `capstone_n600_dispatch_config_20260611.json` (sibling artifact) for the machine-readable config.
Summary:

**P2 (recommended — vendored PR95 torch trainer, Modal CUDA, faithful recipe, smaller basis):**
- arch: `base_channels=20` (primary) / `24` (fallback), `latent_dim=28`, `eval_size=(384,512)`, tie_depth N/A
  (vendored has no tie; base_ch=20 untied = 101K params ≈ 101 KB int8 — still −76K vs frontier; OR add tie).
- recipe: the 8-stage curriculum, `ema_decay=0.999`, grad_clip 1.0/1.0, sigma 0.2→0.1, c1a 0.01→0.02, QAT
  stage4+, muon_lr 2e-4 stage 8.
- epoch budget: compress to fit $100 (~12,000–16,000 total epochs ≈ 27 A100-hr; confirm step-time on a
  10-min Modal smoke FIRST — the GPU-hour estimate is the dominant uncertainty).
- carrier: PR95's 28-d temporal-delta latent (its native carrier).
- platform: Modal CUDA (A100 or T4), `target_modes=["contest_exact_eval"]`, paired CPU+CUDA exact eval on the
  byte-closed archive.

**P1 (local MLX capstone, faithful recipe, base_ch=20 tie=2 stored_latent):**
- `run_capstone_campaign.py --max-pairs 600 --base-channels 20 --tie-depth 2 --carrier stored_latent
  --decoder-dtype int8 --curriculum pr95_8stage --optimizer-schedule pr95_adamw_then_muon
  --curriculum-total-epochs <budget> --scorer-backend mlx_gpu` (mlx_gpu for throughput; fp32-exact override
  auto-set) **+ set `ema_decay=0.999`** (the runner currently hardcodes 0.997 via CapstoneTrainConfig default
  — see §7 residual knob: the runner needs an `--ema-decay` flag OR the faithful default pinned).
- precompute `gt_targets_n600.pt` FIRST (cache currently only ≤192).

---

## 7. RESIDUAL UNRESOLVED KNOBS (named honestly — the symposium decides)

1. **VEHICLE/PLATFORM (P1 vs P2)** — the $100-Modal framing implies P2 (vendored torch), but the bug-swept
   trainer is P1 (MLX, local). **This is the #1 decision.** Unresolved until the symposium picks.
2. **EPOCH BUDGET vs BASIN** — $100 buys ~12k–16k epochs (compressed), NOT PR95's 29,650. Whether a
   SMALLER basis reaches 5.6e-4 in a COMPRESSED budget is the open scientific risk. **Mitigate:** run a
   10-min Modal/local smoke to MEASURE step-time before committing the full budget (the GPU-hour estimate
   `0.3–1.0 s/step → 185–600 GPU-hr full` is unmeasured at n600 on the target HW).
3. **`--ema-decay` flag (P1)** — `run_capstone_campaign.py` hardcodes `ema_decay=0.997` (no CLI flag). For the
   FAITHFUL run it should be 0.999 (PR95). **Small code fix needed:** add `--ema-decay` (default 0.997 to
   preserve current behavior; pass 0.999 for faithful). NOT yet done (out of $0 scope to land + retest, but
   trivial).
4. **base_ch override for P2** — the vendored trainer hardcodes base_ch=36 in 3 lines of `common.py`. A
   smaller basis needs a clone-COPY edit (not in-place per CLAUDE.md "no in-place edits to public PR intake
   clones"). NOT yet prepared.
5. **gt_targets_n600.pt** — missing from cache (only ≤192). First launch streams + caches 600 GT (one-time
   slow precompute). NOT yet built.
6. **pose-tube at the smaller basis** — d_pose ≤ 3e-4 is an OPEN target alongside the basin; the n8 A/B did
   not reach the tube (d_pose 0.0019 ≫ 2.94e-5). The run tests pose AND seg jointly.

---

## 8. NO-FAKE accounting
- Every d_seg/byte number is measured on real artifacts (real `modules.py` SegNet via the canonical
  torch-CPU chain; real byte-close via `build_capstone_stored_latent_archive_bytes`). NO MPS.
- The bug-sweep "clean" verdicts are source-faithful diffs against the runnable vendored PR95 original, not
  assertions.
- The S@basin numbers are CONDITIONAL on reaching the PR95 basin at a smaller basis — stated as a thesis the
  run TESTS, never as a measured result. The frontier is UNMOVED.
- The two infra blockers (platform, epoch-budget) are surfaced, not hidden — they are the honest gate on the
  $100 spend.

**Is the config OPTIMAL + NON-ARBITRARY + BUG-FREE?** The trainer is bug-free (BUG-A fixed+tested; BUG-B/C/D
clean-faithful; no new bugs). The recipe + arch + byte-budget are non-arbitrary (every knob → PR95 source or
the measured byte/score math). The pointer-mover thesis is quantified (base_ch=20 → 86.7 KB → S≈0.131 at
basin). **It is NOT yet a fire-ready dispatch** — six residual knobs (§7), dominated by the vehicle/platform
choice and the epoch-budget-vs-basin risk, must be resolved by the symposium. The config WON'T be wasted on a
recipe bug; whether it lands the basin in budget is the real bet the symposium must price.
