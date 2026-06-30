# θ* per-lever A/B campaign — READY-TO-FIRE arm specs (task #183)

**UTC** 2026-06-30T19:19:03Z · **git** `b8c990941` · **tag** `[macOS-MLX advisory · design/ready-to-fire · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
**NO GPU was used to produce this.** This is the ready-to-fire DESIGN; every arm awaits **operator GO** (one GPU; the Muon arm or a steer owns it). means≠ends: the A/B **ranks levers** (a MEANS); only a **byte-closed n600 exact row < 0.19110** moves the pointer. Fills the `<θ*>` values of `post_muon_application_plan_optimal_form_20260630T1710Z.md` §3 with flag-validated arms.

---

## 0. Warm-start substrate — VERIFIED + 3 LOAD-BEARING CORRECTIONS

**Verified paths (all exist):**
- Muon BEST (EMA shadow, d_seg **0.0036976** @ ep1000): `experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz` (343 KB) ✓ + `levelset_best.json` ✓
- Muon ep1000 **resume sidecar** (live+EMA+opt+epoch): `experiments/results/levelset_thetastar_muon_arm/levelset_resume_state.npz` (1.1 MB) ✓ — its EMA shadow == the BEST (ep1000 was the final = best epoch).
- n200 gt cache: `experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz` (1.6 GB) ✓

**The BEST Muon arm's exact config** (from `run_muon.log`; the front-end+curriculum that EVERY warm-start arm must replicate so the loaded decoder shape matches): `--render-h 384 --render-w 512 --hidden-dim 96 --mod-dim 32 --activation hosc --siren-init --curriculum --palette-anchor --self-orient --reorient-every 50 --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64 --chroma --w-seg 100 --w-pose 0.0 --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 --tau-softplus-tau 0.3 --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-softplus-start-epoch 300 --l7-start-epoch 600 --eikonal-weight 0.01 --length-weight 0.001`; `in_feat=88` (curvelet_cols 40 + dir_w 8).

### CORRECTION 1 — resume from the **resume_state.npz** sidecar, NOT the EMA `BEST.npz` (else the schedule re-softens the partition)
The EMA-only `BEST.npz` carries **no epoch** (`_load_resume_state` L296 → `epoch=0` → `start_epoch=1`). Resuming from it with the default temp/curriculum would restart at **temp 1.0** (anneal from start) **and the CE stage** (epoch 1 < tau-start 300) — RE-SOFTENING the converged sharp argmax partition we are trying to surgically repair. **Fix: resume from `levelset_resume_state.npz`** (epoch=1000 + opt restored) → `start_epoch=1001`, curriculum continues in `l7_softplus` (1001 > l7-start 600), and `--anneal-epochs 1000` clamps temp/LR at their END values (temp 0.05, LR=lr-end) for the whole window. The EMA shadow inside it == the BEST weights, so the verdict/byte-close is identical to the BEST.
*(If you must use `BEST.npz` per a literal reading, it is equivalent ONLY with the pins `--no-curriculum --seg-loss l7_softplus --softmax-temp-start 0.05 --softmax-temp-end 0.05`. Recommended path is `resume_state.npz`.)*

### CORRECTION 2 — DROP `--structured-init` / `--lane-prior-phi1` from every warm-start arm (clobbered no-op)
Code order: structured-init pretrain (L936–1011, modifies `out_sdf`/φ) runs **before** resume; resume `model.update(...)` (L1639) **overwrites ALL live params including `out_sdf`** → the structured-init/lane-prior φ is **completely discarded**. In the BEST chain it only mattered at the original from-scratch CE start (baked into the resumed weights). In a warm-start it is a **silent no-op + wasted 600-step pretrain**. Dropping it changes nothing in the loaded weights (bit-faithful to the resumed state). **Consequence for the campaign: `--lane-prior-phi1 mode/bias` is NOT a valid warm-start A/B lever** (toggling it from a resume is a guaranteed no-op = a FALSE "lane-prior doesn't help" verdict, the NO-FAKE silent-no-op class). → **routed to the v2 from-scratch campaign** (§5).

### CORRECTION 3 — `--max-bank-freq` (and all front-end flags) are NOT valid warm-start A/B levers (shape break)
`in_feat` (L850/864) derives from the curvelet bank size + dir feats. Changing `--max-bank-freq` / `--self-orient` / `--n-dir-freqs` / `--freq-across` / `--freq-along` / `--hidden-dim` / `--mod-dim` / `--chroma` changes the decoder `in_proj` shape → resume `model.update` shape-mismatch → broken/corrupt resume. The "bandwidth anneal 16→32→64" lever therefore **cannot be a warm-start arm** (and `--max-bank-freq` is a single static cap, not an anneal). → **routed to v2 from-scratch** (§5). Every warm-start arm holds the front-end block above FIXED.

### The warm-start regime (all arms identical except the one lever)
- **Resume:** `--resume-from experiments/results/levelset_thetastar_muon_arm/levelset_resume_state.npz` (ep1000 → start 1001; opt best-effort → fresh AdamW moments since we drop `--muon-start-epoch` = a clean new re-treatment AdamW stage; Muon-finish the WINNER in compose).
- **Window:** `--epochs 1150` (150-epoch cap) with **harvest-driven early-stop at the knee** (DoE reframe: arms are saturation MEASUREMENTS, not grinds — stop when Δd_seg/25ep falls into the noise; there is no `--early-stop` flag, so the harvest monitor on `levelset_best.json` stops the daemon).
- **Schedule:** `--anneal-epochs 1000` (clamps temp→0.05 sharp + LR→lr-end for the window — keeps the converged partition), `--lr 1e-3 --lr-end 1e-4` (= BEST tail; the surgical hinge concentrates its gradient on the ~1.3%-of-pixels annulus, so it acts even at the low global LR while the bulk is undisturbed — minimal control drift). *Operator knob: if a lever shows null signal at lr-end 1e-4, re-treat (do NOT kill) at `--lr-end 3e-4`.*
- **Verdict:** `--verdict-pairs 96 --async-verdict --eval-every 25` (realized through-R, numpy-fp32 EMA-shadow authority, background thread so the GPU never idles).
- **Resumable + per-stage ckpts (mandatory):** `--stage-checkpoints --ckpt-every 25` (per-stage preserved + 25-epoch rolling crash window; EMA-shadow + atomic + best-preservation already hardened in `3da9a6b10`). Plus the whole-run archiver `tools/archive_witness_checkpoints.py`.
- **Lever engage:** each lever's `--*-start-epoch 1000` (≤ start_epoch 1001 → engages immediately at resume; the engage re-treats the spike-guard).
- **Shared control:** ONE warm-start no-lever run (the common block + zero levers) → all lever arms are scored as `Δd_seg_i = d_seg(arm_i) − d_seg(control)` against it (cheaper than N paired controls). **EXCEPTION:** the hardness lever (A4) needs its OWN matched control (same `--hardness-oversample`, uniform vs weighted — same total steps).

**Common block (identical in every command below):**
```
--resume-from experiments/results/levelset_thetastar_muon_arm/levelset_resume_state.npz \
--gt-cache experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz --num-pairs 200 \
--epochs 1150 --anneal-epochs 1000 --lr 1e-3 --lr-end 1e-4 --lr-schedule \
--render-h 384 --render-w 512 --hidden-dim 96 --mod-dim 32 --activation hosc --siren-init \
--curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600 --tau-softplus-tau 0.3 \
--softmax-temp-start 1.0 --softmax-temp-end 0.05 --palette-anchor --chroma \
--self-orient --reorient-every 50 --freq-across 32 --n-dir-freqs 2 --freq-along 4 --max-bank-freq 64 \
--w-seg 100 --w-pose 0.0 --ema-decay 0.997 --accum-pairs 8 --grad-clip 1.0 \
--eikonal-weight 0.01 --length-weight 0.001 \
--verdict-pairs 96 --async-verdict --eval-every 25 --mlx-device gpu --seed 0 \
--stage-checkpoints --ckpt-every 25
```
**Per-epoch wall-clock (MEASURED from the Muon log, n200/render-384/async-verdict/custom-backward): ~36 s/epoch** (range 31–46; the spread is the async-verdict thread). → **150-ep arm ≈ 1.5 GPU-h; early-stop @ ~knee (≈60–100 ep) ≈ 0.6–1.0 GPU-h.**

---

## 1. The valid warm-start levers (7 arms) — flag-validated commands, grids, predicted EV

Predicted EV is grounded in the **measured per-stage attribution** (`witness_per_stage_attribution_20260630T165037Z.md`) + **birth-death** (`birth_death_persistence_dseg_20260630T172510Z.md`): **Lane = PRIMED** (margins moving to GT, 47.2%→24.3% mislabeled — the witness's job); **Road = STUCK-causal** (pose-governed, the v2 screw-warp's job, margin-saliency on Road may not convert); **the annulus razor-thin** (flip-rate 0.764 for GT-margin<0.10, ~0 above) → levers that widen the sub-0.10-margin Lane band have the highest EV.

> Append each lever's flags to the **Common block**. `--out-dir experiments/results/levelset_retreat_<arm>_<utc>`.

### A1 — margin-saliency (LEVER-4, all-class annulus) · OFF in BEST · **HIGH EV**
All-class GT-margin-saliency-weighted realized hinge `sal=exp(-margin/τ)` — defends 100% of the flip-prone band (Road 47/Lane 19/Undriv 14/Mov 9/MyCar 11 %). Class-agnostic. Lane portion is PRIMED → converts; Road portion is STUCK → may not (still net-positive on the non-Road boundaries).
- **Grid (weight):** `{10, 30}` — hinge-term is a normalized mean ∈O(0.3–1.0); base seg-loss ≈42 (log `ep_loss`); weight 10 ≈ surgical 7–24% of loss, 30 ≈ assertive 30–70%. (Follow-on `100` only if 10→30 is monotone-down.) **τ follow-on:** at the winning weight, one arm `--margin-saliency-tau 0.3` (tighter focus toward GT-margin p1~0.38).
```
# A1a (weight 10):
--margin-saliency-weight 10 --margin-saliency-tau 0.5 --margin-saliency-target 0.5 --margin-saliency-start-epoch 1000
# A1b (weight 30): --margin-saliency-weight 30 (else identical)
```
- **Predicted Δd_seg:** **−0.0003 … −0.0008** (the dominant lever; PRIMED Lane + all other inter-class edges). **~1.5 GPU-h/arm (×2).**

### A2 — lane-thin dropped-dash prior (LEVER-B, the dominant residual) · OFF in BEST · **HIGH EV**
Realized through-R hinge weighted by a precomputed thin-lane density map (nonzero only on thin GT-lane dashes). Birth-death: lane dashes <3px flip **92%** vs 19% for the largest; <5px dashes **93% missed**. Directly targets the PRIMED Lane long-tail that is the witness's binding residual.
- **Grid (weight):** `{10, 30}`. **radius follow-on:** at the winning weight, one arm `--lane-thin-radius 2` (the ~2px lane tube) vs default 4.
```
# A2a (weight 10):
--lane-thin-weight 10 --lane-thin-radius 4 --lane-thin-target 0.5 --lane-thin-class 1 --lane-thin-start-epoch 1000
# A2b (weight 30): --lane-thin-weight 30 (else identical)
```
- **Predicted Δd_seg:** **−0.0004 … −0.0010** (most-targeted at the measured dominant residual). **~1.5 GPU-h/arm (×2).**

### A3 — hosc-β step-native sharpen (the named UNSWEPT L∞ lever) · β=4 const in BEST · **MED-HIGH EV**
`hosc = tanh(β·sin(ω·u))`; β→∞ ⇒ STEP-native (topology-matched chart for the piecewise-constant argmax, NO Gibbs). Birth-death found part of the residual is **ringing/Gibbs overshoot** at boundaries → sharpening the activation suppresses it. Activation-param only (no shape change → valid warm-start).
- **Grid (β-end):** `{8, 16}` annealed from 4 over the window.
```
# A3a (β 4->8):
--hosc-beta 4 --hosc-beta-end 8 --hosc-beta-anneal cosine --hosc-omega 1.0
# A3b (β 4->16): --hosc-beta-end 16 (else identical)
```
- **Predicted Δd_seg:** **−0.0001 … −0.0005** (suppresses ringing flips at ALL boundaries; some risk the β-4-trained weights destabilize at β16 → that's the A/B). **~1.5 GPU-h/arm (×2).**

### A4 — hardness-weighted code-fit (LEVER-5) · OFF in BEST · **MED EV** · *own matched control*
Waterfill extra per-epoch pair-iteration steps toward HARD pairs (`realized` source = per-pair baseline realized d_seg, the sharper signal). The FAIR A/B is at FIXED `--hardness-oversample`: **weighted** (extras ~ hardness) vs **uniform** (same total steps) — so A4 is its OWN control pair, not vs the shared control.
```
# A4-lever (oversample 0.5, weighted, realized):
--hardness-oversample 0.5 --hardness-weighted --hardness-source realized --hardness-power 1.0
# A4-control (oversample 0.5, UNIFORM extras): --hardness-oversample 0.5  (omit --hardness-weighted)
```
- **Predicted Δd_seg:** **−0.0001 … −0.0004** (caveat: GT-margin per-pair spread only 1.31×; the `realized` source is sharper but the win is modest). **~1.5 GPU-h × 2 (lever + its uniform control).**

### A5 — DM1 conditioning (byte-free FiLM rank cure) · OFF in BEST · **MED-LOW direct-d_seg EV** · firewall-gated
`--film-stiefel` (per-step Stiefel orthonormalization of `film.weight`) + `--code-spectral-entropy-weight β` (spread the code spectrum). Together raise PR(M) 1.19→4.57 at **0 added bytes**. Measured M2 caveat: capacity≠d_seg directly; the real EV is conditioning/held-out amortization (less binding at n200 fixed-pairs). Loss-term + per-step projection, no shape change → valid warm-start. `--dm1-telemetry` logs PR(M).
- **Grid (β):** `{1e-3, 1e-2}`.
```
# A5a (β 1e-3):
--film-stiefel --code-spectral-entropy-weight 1e-3 --dm1-telemetry
# A5b (β 1e-2): --code-spectral-entropy-weight 1e-2 (else identical)
```
- **Predicted Δd_seg:** **−0.0000 … −0.0003** (conditioning; primary value is the n600 amortization, measured here as a side-signal). **~1.5 GPU-h/arm (×2).**

### A6 — lane-edge (LEVER-3, class-1-only) · weight 0 in BEST · **LOW EV (ablation only)**
The class-1-only predecessor of A1. Single ablation arm to CONFIRM the prediction *all-class A1 > class-1-only A6* (i.e. that the all-class generalization earns its keep). Dominated by A1.
```
# A6 (weight 30, class 1):
--lane-edge-weight 30 --lane-edge-class 1 --lane-margin-target 0.5 --lane-edge-start-epoch 1000
```
- **Predicted Δd_seg:** **−0.0001 … −0.0004** (only the 19% Lane share of the flip band; A1 should dominate). **~1.5 GPU-h (×1).**

### A7 — UNIWARD on the A1 winner (Fridrich inverse-steg) · **SECOND-ORDER (wave-2, post-A1)**
Requires `--margin-saliency-weight>0` (it modifies A1's saliency to DOWN-weight textured/SegNet-undetectable regions → concentrate on the smooth survivable boundary; Fridrich square-root-law). Runs on the **A1 winning weight**, so it is sequenced after A1 resolves.
- **Grid (β):** `{4, 8}`.
```
# A7a (uniward β 4, on A1 winner W*):
--margin-saliency-weight <A1_W*> --margin-saliency-tau 0.5 --margin-saliency-target 0.5 --margin-saliency-start-epoch 1000 \
--margin-saliency-uniward --margin-saliency-uniward-beta 4
# A7b (uniward β 8): --margin-saliency-uniward-beta 8 (else identical)
```
- **Predicted Δd_seg (marginal over A1):** **−0.0001 … −0.0003** (refines allocation toward the smooth flip-prone boundary). **~1.5 GPU-h/arm (×2, wave-2).**

---

## 2. RANKED priority (highest predicted Δd_seg per GPU-hour first)

| Rank | Arm | Lever | Predicted Δd_seg | EV basis | GPU-h (early-stop … full) |
|---|---|---|---|---|---|
| **1** | A2 | lane-thin dropped-dash | −0.0004 … −0.0010 | the MEASURED dominant residual (thin Lane dashes, 93% <5px missed; PRIMED; most-targeted) | 0.6–1.0 … 1.5 ×2 |
| **2** | A1 | margin-saliency all-class | −0.0003 … −0.0008 | defends 100% of the flip band; PRIMED Lane converts; broad | 0.6–1.0 … 1.5 ×2 |
| **3** | A3 | hosc-β step-native | −0.0001 … −0.0005 | suppresses Gibbs/ringing flips at all boundaries (birth-death) | 0.6–1.0 … 1.5 ×2 |
| **4** | A4 | hardness realized | −0.0001 … −0.0004 | waterfill hard pairs (sharper realized source; spread modest) | 0.6–1.0 … 1.5 ×2 (own ctrl) |
| **5** | A7 | UNIWARD (on A1) | −0.0001 … −0.0003 marginal | refine A1 toward smooth survivable boundary (wave-2) | 0.6–1.0 … 1.5 ×2 |
| **6** | A5 | DM1 conditioning | −0.0000 … −0.0003 | byte-free PR(M) cure; primary value = n600 amortization | 0.6–1.0 … 1.5 ×2 |
| **7** | A6 | lane-edge (ablation) | −0.0001 … −0.0004 | confirms all-class A1 > class-1-only | 0.6–1.0 … 1.5 ×1 |

**Total campaign GPU-time (one GPU, sequential):**
- 1 shared control + wave-1 lever arms (A1×2, A2×2, A3×2, A4×2[lever+uniform], A5×2, A6×1) = **12 arms**; + τ/radius follow-ons ≈ 2; + A7 wave-2 ×2; + 1 composite + (Muon-finish composite).
- **≈ 16 arms × ~0.8 GPU-h (early-stop median) ≈ 13 GPU-h**, range **~10 GPU-h (aggressive early-stop) … ~26 GPU-h (full 150-ep, no early-stop)**. Sequential on the single GPU; arms are cheap because the per-stage-preserved BEST ckpt is the shared warm-start (no from-scratch cost).

---

## 3. HARVEST + COMPOSE plan (means → the n600 recipe → the exact pointer-mover)

**Per-arm harvest** (fold into a `harvest_arm` step; CPU/$0, off the GPU):
1. Scalar: `{knee-epoch, final d_seg, Δd_seg vs control, slope-at-stop}` from `levelset_best.json` / `levelset_train_result.json`.
2. **Per-stage per-pixel/annulus attribution** via `tools/witness_per_stage_annulus_attribution.py` (EXISTS, 29 KB; numpy-fp32, read-only on the arm's preserved ckpt vs the control ckpt) → the {corrected / primed / regressed / untouched} annulus map + per-class residual table + the recommended per-residual repair mechanism (learn/store/deterministic/UNIWARD/combo), per the surgical-repair toolbox.
3. Tag every row `[macOS-MLX research-signal] score_claim=false promotable=false`.

**Keep/re-treat (NO premature KILL):** keep a lever iff `Δd_seg < −noise` (advisory). A null/positive lever is **re-treated** (higher lr-end, or routed to v2 from-scratch — e.g. lane-prior-phi1/bandwidth), never killed (Forbidden-premature-KILL; janky-warm-start ≠ paradigm falsification).

**COMPOSE:** stack the winning levers at their winning values → ONE warm-start composite (from the SAME BEST resume_state) → measured composite Δd_seg. **Watch interactions:** A1 (all-class) and A2 (lane-thin) BOTH target the Lane boundary → expect **sub-additive**; A3 (β) is orthogonal (activation); A7 modifies A1. Then **Muon-finish** the composite (`--muon-start-epoch <window-mid> --muon-lr 0.002` — the BEST's proven Muon config) — the expensive Muon tail is worth paying once, last, on the best substrate.

**→ n600 recipe table (the DoE deliverable per the campaign reframe):** n200 DETERMINES {which levers win, their values, the lever ORDER, the per-stage saturation epochs/slopes, the warm-start/recursion structure}; these TRANSFER to n600. n600 TESTS the achievable d_seg FLOOR (lower-variance grad). The composite winning config + saturation curves → the operator-approved n600 launch recipe (mod-dim re-measured ~26 at n600 per the generalization probe, NOT 21; directional basis from ep0; root-tracking anneal).

**→ exact row (the ONLY pointer-mover):** byte-close the composite EMA via `tools/witness_byte_close_and_eval.py` → exact n600 CPU/CUDA eval (`upstream/evaluate.py`, NEVER MPS) → a real row vs 0.19110. **Awaits operator GO.**

---

## 4. FLAG-VALIDATION block (NO-FAKE; dogfood vs the real argparse)

Validator `scratchpad/validate_flags.py` parses all `add_argument("--…")` from `experiments/train_levelset_witness_realized_through_R_mlx.py` (**116** flags) and checks every flag this campaign emits (**78**):

```
AUTHORITATIVE flags parsed from argparse: 116
EMITTED flags in campaign: 78
RESULT: ALL PASS   (78/78 emitted flags exist in the trainer argparse; 0 FAIL)
PATH EXISTS: experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz
PATH EXISTS: experiments/results/levelset_thetastar_muon_arm/levelset_resume_state.npz
PATH EXISTS: experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz
```
Every emitted flag (frozen-common block + A1–A7 + the excluded-lever refs lane-prior-phi1/structured-init/max-bank-freq) is a real trainer flag. The 3 levers that PASS flag-validation but are **EXCLUDED from warm-start** for the structural reasons in §0 (lane-prior-phi1 = resume-clobbered no-op; max-bank-freq/self-orient front-end = shape break) are real flags routed to §5.

---

## 5. EXCLUDED from warm-start → v2 from-scratch campaign (NOT killed; wrong vehicle for a warm-start A/B)
- **lane-prior-phi1 mode/bias** (`--lane-prior-phi1 --lane-prior-phi1-mode {replace,bias} --lane-prior-phi1-bias-scale` + required `--structured-init`): φ-init is clobbered by `--resume-from` → A/B-testable ONLY from a from-CE-scratch start. Belongs to the v2 from-scratch redesign (it already rides the from-scratch BEST chain via the original CE start).
- **bandwidth / max-bank-freq + directional-basis geometry** (`--max-bank-freq`, `--self-orient`, `--n-dir-freqs`, `--freq-across`, `--freq-along`): change `in_feat` → break warm-start resume. A/B-testable only from-scratch (the front-end is fixed once the decoder is trained). The directional basis (−48%) is ALREADY ON in BEST; its A/B happened at the original from-scratch design.

These are the from-scratch knobs for the v2 S²WL redesign (`project_gr_unified_action_full_witness_architecture_20260629`), tested in that campaign, not this warm-start one.

---

## 6. Launch wrapper (for the operator GO; NOT executed here)
Each arm launches via the proven durable-daemon + memory-guard + perf-env pattern (from the Muon arm's `run_muon.log`), one GPU sequential:
- `tools/safe_run.py --rss-mb 90000 --timeout <s> --label levelset_retreat_<arm> -- .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py <common block> <lever flags>`
- **Perf env (launch-gate, b46c79ac6 default-ON):** `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` — VERIFY the `custom_grouped_backward active=true` log line at launch (a correct run is ~17× slow if unset). `tools/witness_autoconfig.py` emits this.
- Memory-guard ≥10 GB floor (scale-safeguarded; never kills the control-plane). Resumable + per-stage ckpts are in the common block.

**Anchors:** `post_muon_application_plan_optimal_form_20260630T1710Z.md` (§3 the `<θ*>` slots this fills), `theta_star_witness_lever_stack_and_variational_levelset_frame_20260627` (the campaign), `feedback_per_stage_per_pixel_annulus_attribution_surgical_repair_toolbox_20260630` (the harvest attribution), `witness_per_stage_attribution_20260630T165037Z` + `birth_death_persistence_dseg_20260630T172510Z` (the EV evidence). **pointer 0.19110 UNMOVED.**
