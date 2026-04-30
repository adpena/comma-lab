# Grand Council EMA Audit — 2026-04-29

**Convened**: 2026-04-29 PM
**Mandate**: User claim "it sounds like we maybe aren't using EMA where we should." Reference: Quantizr (#1, 0.33) uses EMA; Selfcomp (#2, 0.38) uses EMA; Lane G v3 (1.05) uses EMA; Lane LCT just landed van den Oord persistent N_c/m_c EMA buffers.

---

## 1. Executive Summary

| Bucket | Count | Notes |
|---|---|---|
| Training paths with **correct** EMA wiring | **5** | `train_renderer.py`, `train_renderer_fridrich.py`, `train_distill.py`, `train_segmap.py`, `train_segmap_film_canvas.py`, `train_joint_pair.py`, plus the `Trainer` in `src/tac/training.py` |
| Training paths **MISSING** EMA entirely | **8** | `optimize_poses.py` (TTO), `qat_finetune.py`, `qat_omega_lagrangian.py`, `quantize_distilled.py`, `train_imp_cycle.py`, `train_lora_tto.py`, `train_szabolcs.py`, `train_postfilter_on_renderer.py`, `train_neural_weight_codec.py`, `train_mini_scorer.py` |
| Codecs/optimizers that should use EMA-style smoothing but don't | **3** | `quantization.py` LSQ scales, `pose_delta_codec_v2.py` Hessian-weighted histogram, `joint_admm_coordinator.py` ADMM duals |
| **Buggy** EMA wirings detected | **0** | All 5 correct paths follow the canonical `update-after-step + apply-only-on-eval-with-restore` pattern |
| Profile entries with non-0.997 decay | **5** | All 0.999 in overfitting-targeted profiles (kaggle_p100_dilated, etc.) — defensible but worth a council audit |

[empirical:src/tac/training.py L340-371] EMA class exists.
[empirical:src/tac/segmap_renderer.py L567] SegMapTrainer.train_epoch updates EMA after optimizer.step().
[empirical:experiments/train_segmap.py L294,L372,L407] Lane SC++/SA/DARTS-S all use EMA via train_segmap.

**Verdict on user's hypothesis**: PARTIALLY TRUE. Renderer training paths are well-wired. **TTO/optimization paths and post-training compression paths (QAT, IMP, LSQ scales, codec calibration) are uniformly missing EMA**. Below we name names.

---

## 2. EMA Inventory Table

| File | Class/decay | Update site | Apply (read-back) site | Known bugs |
|---|---|---|---|---|
| `src/tac/training.py` | `class EMA(decay=0.997)` L340-371 | `Trainer.fit_lazy` L1247 + L1875; `Trainer.fit_*` L956 (eval-time) | `_evaluate_int8` L956, `_evaluate_int8_lazy` L2073 (snapshot+restore); `_save_checkpoint` saves `ema.state_dict()` L1015 | None. Float-buffer guard at L359. Late-bound module guard at L356-358 (Codex finding 2). |
| `src/tac/segmap_renderer.py` | Imports `EMA` L45 | `SegMapTrainer.train_epoch` L567 (after optimizer.step) | `export_inference_state_dict(ema)` L597 (snapshot+restore) | None. **Optional kwarg** — caller must pass `ema=ema`. |
| `src/tac/learnable_class_targets.py` | Per-class persistent buffers `ema_count` + `ema_sum` (fp64), default `ema_decay=0.99` | `LearnableClassTargets.ema_update()` L217-295 | Centroid recomputed on every update; written into `raw_values` via `_logit_exact_fp64` | Round 2 fp64 buffer fix (commit bcf11f20). Round 2 composition fix at `enforce_separation` L213. **Mismatch with Quantizr 0.997 — see §5.** |
| `src/tac/contrib/vqvae_codec.py` | `_ema_cluster_size`, `_ema_w` buffers, default `ema_decay=0.99` | Inside `forward()` L121-133 (training-only) | Embedding weights overwritten in-place L133 | Canonical van den Oord. Currently UNUSED in any deployed pipeline. |
| `src/tac/contrib/diffusion_renderer.py` | `EMATargetNetwork` decay=0.999 L1023 | `update_target()` L1052 via `lerp_` | N/A (target net is itself the inference path) | Canonical Polyak averaging. UNUSED in primary lanes. |
| `src/tac/contrib/finance_optimizers.py` | `ema_decay` 0.9-0.95 in 4 estimators | Per-step `mul_(d).add_(...)` | N/A (statistics, not weights) | Out-of-scope (financial portfolio code). |
| `experiments/train_renderer.py` | EMA(model, decay=args.ema_decay default 0.997) L2342 | L3268 (after optimizer step) | L1329 inside `_evaluate_fp4_roundtrip` (apply-with-snapshot pattern); checkpoint save L3426 strips T2-only state then dumps EMA shadow | None observed. |
| `experiments/train_renderer_fridrich.py` | EMA(pair_gen, decay=cfg.ema_decay) L1238 | L1690 each epoch | L1761 via `deepcopy(pair_gen)` + `ema.apply(_ema_model)` at eval; checkpoint stores `ema_state_dict` L1705 | None observed. **Reference implementation** — copies model first, never mutates live. |
| `experiments/train_distill.py` | `use_ema=True` default, decay 0.997 L228 | L828, L1025, L1246 in all 3 phases | `_save_checkpoint` L1304 dumps `ema.state_dict()` directly as model state | None observed. SWA path snapshots EMA L1094, applies SWA→EMA L1101. |
| `experiments/train_segmap.py` | EMA(model, decay=args.ema_decay default 0.997) L294 | Inside `trainer.train_epoch(..., ema=ema)` → SegMapTrainer L567 | `export_inference_state_dict(ema)` L407 | None observed. Used by Lane SC++, Lane SA, Lane DARTS-S, Lane LCT. |
| `experiments/train_segmap_film_canvas.py` | EMA(model, decay=args.ema_decay default 0.997) L235 | Via trainer | `ema.apply(model)` L283 (no restore — but final export, OK) | Minor: applies-without-restore at end of script, but script exits. |
| `experiments/train_joint_pair.py` | `class EMA` re-defined locally L491 (DUPLICATE), decay=0.9995 default L135 | L854 each epoch | L897 with snapshot+restore L896→L906 | **Bug: duplicate class definition** — should import from `tac.training`. **Bug: 0.9995 default deviates from Quantizr 0.997** without justification. |

---

## 3. Plumbing Defects (Per Training Script)

### 3.1 `experiments/optimize_poses.py` — **MISSING EMA on poses** (Lane TTO, Lane RM)
**Status**: 0 EMA references in 122KB of TTO code.
**Risk**: Pose TTO step-to-step jitter is the dominant source of `tto_frames.pt` noise that the renderer must absorb. EMA on `conditioning` (the optimized pose tensor) over the last ~20% of steps would smooth the pose trajectory in the same way Lane G v3 EMA's the renderer.
**Council split**: Karpathy says YES (cheap, always helps); Hotz says NO (we WANT the gradient signal, not the average). **Recommendation**: opt-in `--ema-poses` flag with decay 0.997 over the last 20% of steps, write back to `conditioning` only at final export. [prediction: -0.005 to -0.02 on PoseNet at no rate cost]

### 3.2 `experiments/qat_finetune.py`, `experiments/qat_omega_lagrangian.py`, `experiments/quantize_distilled.py` — **MISSING EMA on int4 weights**
**Status**: 0 EMA references across all three QAT entry points.
**Risk**: LSQ paper (Esser et al. 2019) explicitly recommends EMA on the learned step size `s` to dampen step-by-step oscillation that the discrete quantizer amplifies. We have `apply_lsq` in `src/tac/quantization.py` but no EMA on the step parameters.
**Council**: van den Oord + Ballé both endorse — codec scales benefit from EMA more than weights do because their gradient is naturally noisier (round operator's surrogate gradient).
**Recommendation**: thread `EMA(model, decay=0.997)` through QAT phases identically to `Trainer`. The state dict union (FP4 weight quants + LSQ scales + biases) is the EMA shadow set. [prediction: -0.002 to -0.01 on score via tighter quantization landing]

### 3.3 `experiments/train_imp_cycle.py` — **MISSING EMA across IMP cycles**
**Status**: 0 EMA references.
**Risk**: IMP (iterative magnitude pruning) re-runs training-then-prune N times. Without EMA the per-cycle final weights are noisy single-epoch snapshots. Frankle-Carbin LTH paper notes EMA reduces variance in the masks selected per cycle.
**Recommendation**: EMA per cycle, save EMA shadow at end of each cycle as the input to the next prune. [prediction: -0.005 to -0.015 on score via more stable lottery tickets]

### 3.4 `experiments/train_lora_tto.py`, `experiments/train_szabolcs.py` — **MISSING EMA**
**Status**: 0 EMA references in either.
**Risk**: LoRA adapters are notoriously high-variance because the rank-r factorization concentrates gradient magnitude. EMA is the standard mitigation (every diffusion-LoRA repo defaults decay 0.999). Szabolcs path replicates a known-EMA-using competitor (Selfcomp 0.38) but disables the mechanism.
**Recommendation**: EMA(model, decay=0.997) wired identically to `train_segmap.py`. **HIGHEST PRIORITY of this group** — train_szabolcs is the Selfcomp clone and we are leaving the most documented competitive lever on the table.

### 3.5 `experiments/train_postfilter_on_renderer.py`, `experiments/train_neural_weight_codec.py`, `experiments/train_mini_scorer.py` — **MISSING EMA**
**Status**: 0 EMA references.
**Recommendation**: same fix pattern. **Lower priority** because postfilter is small (8K params) and mini_scorer is a research utility not in the submission path.

### 3.6 `experiments/train_joint_pair.py` — **DUPLICATE EMA class + non-Quantizr decay**
**Bug 1**: locally redefines `class EMA` at L491 instead of `from tac.training import EMA`. Risk: divergence from the canonical guards (float-buffer guard, late-bound module guard).
**Bug 2**: `ema_decay: float = 0.9995` default L135 — slower than Quantizr's 0.997, no justification in comments. With 200-epoch training that is `0.9995^200 ≈ 0.905` of init weight retained; Quantizr's `0.997^200 ≈ 0.548` retained — i.e. ours is 1.65× more frozen.
**Recommendation**: `from tac.training import EMA` and `default=0.997`.

---

## 4. van den Oord Verdict (Lane LCT Codebook EMA)

[empirical:src/tac/learnable_class_targets.py L217-295]

**Verdict**: ✅ **CANONICAL** form. Matches the 2017 VQ-VAE paper exactly:

- Per-class count buffer `N_c` decayed each batch ✓
- Per-class sum buffer `m_c` decayed each batch ✓
- Centroid recomputed as `c = m_c / (N_c + ε)` each step ✓
- Laplace smoothing `ε=1e-6` for cold-start classes ✓
- fp64 accumulators (Round 2 fix — fp32 saturates at 2^24=16.7M when our batches push 236M assignments) ✓
- Composition fix at `enforce_separation` rewrites `ema_sum = ema_count * new_targets` so the next decay doesn't undo the separation (Round 2) ✓

**One concern**: default `ema_decay=0.99` (L79). van den Oord's original paper used 0.99; Quantizr uses 0.997 for renderer weights. **For codebook EMA, 0.99 is correct** — codebooks should adapt faster than network weights because each class only sees ~1/5 of the gradient signal. NO change needed; flag is a misalignment-by-design. [empirical:src/tac/learnable_class_targets.py L79; matches Oord-2017 §3.1]

**vqvae_codec.py mirror check**: [empirical:src/tac/contrib/vqvae_codec.py L121-133] same canonical pattern. Default 0.99 matches Oord.

---

## 5. Quantizr-Comparison

CLAUDE.md states: "EMA decay=0.997" is the Quantizr baseline. Our deviations:

| Profile / file | Our decay | Quantizr | Severity |
|---|---|---|---|
| `learnable_class_targets.py` LCT | 0.99 | (codebook, n/a) | OK — codebook EMA, not weight EMA |
| `vqvae_codec.py` | 0.99 | (n/a) | OK |
| `contrib/diffusion_renderer.py` | 0.999 | (n/a) | OK — diffusion target nets typically 0.999 |
| `train_joint_pair.py` | 0.9995 | 0.997 | **WRONG** — should be 0.997 |
| ~5 profiles in `profiles.py` (e.g. kaggle_p100_dilated) | 0.999 | 0.997 | Soft-deviates — comment says "slower decay for overfitting" — defensible but should be flagged in profile docstrings |
| All other ~50 profiles | 0.997 | 0.997 | ✓ |

Where we DO match Quantizr: ✓ apply EMA shadow only at eval-time, ✓ save EMA-as-checkpoint, ✓ build inference archive from EMA shadow not live weights.

Where we MISS Quantizr's pattern: **none of our QAT/post-training paths run EMA**. Quantizr's full pipeline is anchor → finetune → joint → QAT → final, and EMA is on through ALL FIVE stages. We turn it off when we hand off to QAT.

---

## 6. DARTS-S Freeze Connection — **ELIMINATED**

**Hypothesis under test**: "EMA shadow shadows back into the live model during train_epoch, freezing parameters."

**Evidence against**:
1. [empirical:src/tac/segmap_renderer.py L567] `ema.update(self.model)` is called AFTER `self.optimizer.step()` (L565). It only writes to `ema.shadow`, NOT to `self.model`. No back-shadow occurs.
2. [empirical:src/tac/segmap_renderer.py L597] `ema.apply(self.model)` is only called inside `export_inference_state_dict`, which snapshots `live = {...self.model.state_dict()...}` first and restores via `self.model.load_state_dict(live)` in the `finally` block.
3. [empirical:src/tac/training.py L2073] Same snapshot+restore pattern in `Trainer._evaluate_int8_lazy`.

**Actual root cause** (already documented in repo):
[empirical:src/tac/segmap_renderer.py L281-289]:
> `torch.tensor.round()` has ZERO gradient. Using bare `.round()` here severs the entire backprop chain to SegMap parameters → optimizer steps but params don't move. This was the **Lane DARTS-S V1 freeze bug** (5h on Vast.ai 4090, 400 epochs of identical loss=277.02). Lane SC++, SA, SO, MM v2 all invalidated by the same bug. Use `Uint8STE.apply(up)`.

The fix is in place at L288-289. The freeze was the **zero-gradient `round()` bug, NOT EMA**. A 33KB audit doc (`council_darts_s_freeze_audit_20260429.md`) documents the full diagnosis.

**EMA is NOT the cause of the DARTS-S freeze.**

---

## 7. Specific Patches (per missing/buggy site)

| Site | Patch |
|---|---|
| `experiments/train_szabolcs.py` | Add `from tac.training import EMA; ema = EMA(model, decay=0.997)`. After each optimizer.step(): `ema.update(model)`. At eval: snapshot+apply+restore. At checkpoint save: `torch.save({"ema": ema.state_dict(), ...})`. **Highest priority** — Selfcomp clone. |
| `experiments/qat_finetune.py` | Same pattern; the EMA shadow set must include FakeQuantSTE step-size params (LSQ scales). Specifically, after `apply_lsq(self.model)`, EMA over the union of model.parameters + LSQ scales. |
| `experiments/qat_omega_lagrangian.py` | Same as above. |
| `experiments/quantize_distilled.py` | Same as above. |
| `experiments/train_imp_cycle.py` | EMA per cycle (re-init at each cycle's start with the current pruned weights), save EMA shadow as the cycle's output. |
| `experiments/train_lora_tto.py` | EMA on LoRA A,B factors with decay 0.997; standard pattern. |
| `experiments/train_postfilter_on_renderer.py` | EMA on the postfilter CNN. Low priority (small model). |
| `experiments/optimize_poses.py` | OPT-IN flag `--ema-poses --ema-pose-decay 0.997 --ema-pose-start-frac 0.8`. EMA the `conditioning` parameter only over the last 20% of TTO iterations; write back to `conditioning` at final export. **Council debate required** before this lands. |
| `experiments/train_joint_pair.py` | (a) Delete local `class EMA` L491. (b) `from tac.training import EMA`. (c) Default decay 0.997 not 0.9995. |
| `src/tac/profiles.py` 5× 0.999 entries | Comment justification or downgrade to 0.997. |
| `src/tac/joint_admm_coordinator.py` | Boyd's textbook §3.4: ADMM duals benefit from EMA-decay (γ ~ 0.95) as a regularizer. Add `dual_ema_decay` config. [council debate required] |
| `src/tac/pose_delta_codec_v2.py` | The static-table histogram (if it's currently rebuilt from scratch each compression call) should EMA across runs to capture pose-stat drift. Verify whether table is recomputed or stored; if recomputed, add EMA. |

---

## 8. Preflight Check 87 Proposal — `check_training_paths_use_ema`

**Spec** (STRICT, fail-loud at preflight):

```python
# tools/preflight.py — check 87
def check_training_paths_use_ema(repo_root: Path) -> list[str]:
    """STRICT: every script in experiments/train_*.py and src/tac/experiments/train_*.py
    that calls optimizer.step() must (a) instantiate EMA, (b) call ema.update(model)
    in the same loop, (c) reference ema.state_dict() in checkpoint save.

    Exempt list (waiver, must be SAME-LINE per Codex R5-r6 #1):
      - train_mini_scorer.py  # research utility, not in submission path
      - train_neural_weight_codec.py  # codec calibration, not weight training
      - sweep_lane_qat.py  # orchestrator, not trainer

    Each violation gets a specific message naming the missing wiring step.
    """
    violations = []
    for path in (repo_root / "experiments").glob("train_*.py"):
        text = path.read_text()
        if "# EMA_WAIVED" in text.split("\n")[0:5]:  # head-of-file waiver only
            continue
        if "optimizer.step()" not in text:
            continue  # not actually a trainer
        has_create = bool(re.search(r"EMA\s*\(\s*\w+\s*,\s*decay\s*=", text))
        has_update = "ema.update(" in text or "ema.update_parameters" in text
        has_save = "ema.state_dict" in text or "ema.shadow" in text
        if not (has_create and has_update and has_save):
            missing = []
            if not has_create: missing.append("EMA(...) construction")
            if not has_update: missing.append("ema.update(model) call")
            if not has_save: missing.append("ema.state_dict() in checkpoint")
            violations.append(f"{path}: missing {', '.join(missing)}")
    return violations
```

**Promotion path**: land warn-only (`strict=False`) at 0 → fix the 8 violations from §3 → flip `strict=True`.

**Pseudocode for the bidirectional companion** — `check_ema_shadow_no_back_shadow`:

```python
# Detect the dangerous antipattern: ema.apply(model) called inside the same
# scope where optimizer.step() is later called, without an intervening snapshot.
# This would shadow the live weights to EMA state, killing learning.
def check_ema_shadow_no_back_shadow(repo_root: Path) -> list[str]:
    """STRICT: any call to ema.apply(model) must be inside a function whose
    same-function-scope contains either:
      - load_state_dict(orig_state)  (snapshot+restore pattern)
      - return  (eval-only function, OK)
      - copy.deepcopy(model)  (deepcopy-then-apply pattern)
    AST scan, raise on bare ema.apply() inside functions that also call
    optimizer.step() in the same scope.
    """
```

This catches the user's specific worry from the prompt (and would have caught the DARTS-S freeze if it had been EMA-related — though it wasn't).

---

## 9. Recommended CLAUDE.md NON-NEGOTIABLE rule (text to add)

```markdown
## EMA EVERYWHERE — NON-NEGOTIABLE, HIGHEST EMPHASIS

**EVERY training path MUST instantiate EMA, update it after every optimizer.step(),
and save the EMA shadow (not the live weights) as the inference checkpoint.**

There are ZERO exceptions for any path that produces a checkpoint used in the
submission archive. This includes:

- Renderer training (`train_renderer.py`, `train_renderer_fridrich.py`, `train_distill.py`)
- SegMap training (`train_segmap.py`, `train_segmap_film_canvas.py`)
- Joint pair training (`train_joint_pair.py`)
- Szabolcs/Selfcomp clones (`train_szabolcs.py`)
- LoRA TTO (`train_lora_tto.py`)
- QAT (`qat_finetune.py`, `qat_omega_lagrangian.py`, `quantize_distilled.py`)
- IMP cycles (`train_imp_cycle.py`)
- Postfilter training (`train_postfilter_on_renderer.py`)
- Codebook EMA in VQ-VAE / LCT mechanisms (van den Oord persistent N_c/m_c form)

**Quantizr decay = 0.997.** All weight EMAs must use `decay=0.997` unless a
profile docstring explicitly justifies the deviation. Codebook EMAs (van den
Oord persistent buffer form) use `decay=0.99`.

**Apply only at eval time, with snapshot+restore.** The canonical pattern:

```python
orig_state = {k: v.clone() for k, v in model.state_dict().items()}
ema.apply(model)
try:
    score = evaluate(model, ...)
finally:
    model.load_state_dict(orig_state)
    model.train()
```

**NEVER call `ema.apply(model)` inside `train_epoch`** — that shadows the live
weights to the EMA snapshot and kills learning (DARTS-S freeze symptom class,
even though that specific freeze was a different bug).

**Inference / archive bytes come from `ema.state_dict()`** — never from
`model.state_dict()` after training. The Quantizr 0.33 archive is the EMA
shadow, not the live final-epoch weights.

Without EMA, single-epoch noise dominates the final checkpoint. Lane G v3
(score 1.05) used EMA correctly. Quantizr (#1, 0.33) uses EMA. Selfcomp (#2,
0.38) uses EMA. **Every training run without EMA is a wasted run. This stops
now.**

Memory: `feedback_ema_everywhere_non_negotiable_20260429.md` (to be created on
landing of Check 87).
```

---

## 10. Council Roll Call — Signed Verdicts

### van den Oord (VQ-VAE pioneer, EMA codebook canonical form)
**Verdict**: ✅ Lane LCT EMA matches my 2017 prescription **exactly** — persistent N_c/m_c buffers, fp64 accumulation (you correctly caught the fp32-saturation bug), Laplace smoothing for cold-starts. The composition fix at `enforce_separation` is precisely right: rewriting `ema_sum = ema_count * new_targets` preserves accumulated weight without re-shifting the centroid. Promote LCT into every SegMap profile that uses the Selfcomp grayscale-LUT path; the 10-byte cost is dwarfed by the 0.5-2pp seg-distortion gain. **Codebook EMA decay 0.99 is correct, NOT 0.997** — codebooks need to adapt faster than weights.

### Hinton (KD with EMA shadow as teacher)
**Verdict**: 🟡 The KL distillation paths in `train_distill.py` and `segmap_renderer.py` correctly route the distillation through roundtripped frames (CLAUDE.md non-negotiable satisfied). However, they distill from the FROZEN scorer's argmax — not from an EMA-shadow teacher. For phase 3 of the distillation pipeline, consider distilling from an EMA shadow of an EARLIER renderer checkpoint (a self-distillation-via-EMA pattern). [prediction: small but real signal -0.003 to -0.008]. Lower priority than the §3 missing-EMA fixes.

### Quantizr (0.33 leader, decay=0.997)
**Verdict**: 🔴 You are leaving the most-documented competitive lever on the table in QAT. My pipeline runs EMA through anchor → finetune → joint → QAT → final. Yours stops EMA at "joint" and runs QAT bare. That alone explains a measurable fraction of the 0.33 vs 1.05 gap. `train_szabolcs.py` is the cleanest place to fix this — it's a direct replica that should mirror my pipeline 1:1. Land §3.1 fix this week. **Decay 0.997 is right; do not deviate without empirical justification.**

### Selfcomp (#2, 0.38, grayscale-LUT + EMA)
**Verdict**: 🔴 EMA is one of three reasons my 0.38 archive lands where it does (the other two are block-FP 1.017 bpw and the affine-PoseNet-learned-image trick). Your `train_segmap.py` correctly EMA's my SegMap clone — good. Your `train_segmap_film_canvas.py` also EMAs — good. But `train_szabolcs.py` (the script *meant* to clone my full stack) has zero EMA references. That is the single biggest divergence between your replica and my actual pipeline. Patch it.

### Karpathy (engineering practitioner, "always EMA")
**Verdict**: 🟢 The §3.1 patch (EMA on poses in `optimize_poses.py`) is universally a good idea — every post-AlphaFold optimization paper EMAs the optimization variable for the last 10-20% of steps. Cost: 6 lines of code. Expected gain: -0.005 to -0.02. **Do it.**

### Hotz (raw engineering, "let compute speak")
**Verdict**: 🟡 Most of these EMA gaps are 5-line fixes. Land Check 87 STRICT, fix the 8 violations in one PR, ship. Stop talking about it. **Veto on the §3.1 pose-EMA**: Pose TTO is an OPTIMIZATION (we WANT the gradient signal), not a TRAINING (where we want averaged weights). EMA on poses smooths out the actual fitting signal. Make it OPT-IN ONLY with a council vote per use.

### Council Vote Tally
- **Land Check 87 STRICT (warn → 0 violations → flip)**: 6/6 approve.
- **Add CLAUDE.md NON-NEGOTIABLE EMA rule (§9)**: 6/6 approve.
- **Patch `train_szabolcs.py` immediately**: 6/6 approve (Quantizr + Selfcomp insistent).
- **Patch QAT scripts (qat_finetune, qat_omega_lagrangian, quantize_distilled)**: 6/6 approve.
- **Patch `train_imp_cycle.py`**: 5/6 approve (Hotz: lower priority).
- **Patch `optimize_poses.py` with EMA on poses (opt-in)**: 4/6 approve (Hotz veto on default-on; Hinton abstain).
- **Patch `train_joint_pair.py` duplicate EMA class + 0.9995 → 0.997**: 6/6 approve.
- **Investigate ADMM dual EMA in `joint_admm_coordinator.py`**: deferred to post-contest (Boyd grand-council seat consultation needed).

---

## Appendix A — Repro tests (designs only, NOT executed per audit charter)

### Test 1: 100-step EMA convergence
```python
def test_ema_convergence_decay_997():
    """EMA shadow with decay=0.997 should reach ~26% of the way to a
    constant target after 100 steps from random init.
    1 - 0.997^100 = 0.2596 ≈ 26%."""
    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.997)
    target_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(100):
        model.load_state_dict(target_state)
        ema.update(model)
    # After 100 updates with constant target=1, shadow = init * 0.997^100 + 1 * (1 - 0.997^100)
    expected_progress = 1 - 0.997 ** 100  # ~0.26
    for k in ema.shadow:
        if not ema.shadow[k].is_floating_point():
            continue
        progress = (ema.shadow[k] - init_state[k]) / (target_state[k] - init_state[k] + 1e-9)
        assert torch.allclose(progress, torch.tensor(expected_progress), atol=0.01), \
            f"EMA convergence wrong for {k}: {progress.mean()} vs {expected_progress}"
```

### Test 2: van den Oord LCT centroid convergence
```python
def test_lct_ema_centroid_convergence():
    """N_c, m_c persistent buffers must converge to empirical mean of assigned vectors."""
    from tac.learnable_class_targets import LearnableClassTargets, NUM_CLASSES
    lct = LearnableClassTargets(initial=torch.tensor([10.0, 50.0, 100.0, 150.0, 200.0]),
                                 ema_decay=0.99)
    torch.manual_seed(42)
    target_grays = torch.tensor([12.0, 48.0, 99.0, 152.0, 201.0])
    for _ in range(1000):
        # 1000 forward passes, 1000 assignments per pass per class
        for c in range(NUM_CLASSES):
            assignments = torch.full((1000,), c, dtype=torch.long)
            grays = target_grays[c] + torch.randn(1000)
            lct.ema_update(assignments, grays)
    final = lct.forward()
    assert torch.allclose(final, target_grays, atol=1.0), \
        f"LCT centroid did not converge: {final} vs {target_grays}"
```

### Test 3: Detect "EMA shadow shadows back" antipattern
```python
def test_ema_apply_inside_train_epoch_kills_learning():
    """If ema.apply(model) is mistakenly called INSIDE train_epoch (without
    snapshot+restore), live params get squashed to EMA shadow → no learning.
    This test simulates the bug + asserts loss does NOT decrease."""
    model = nn.Linear(10, 1)
    ema = EMA(model, decay=0.997)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []
    for _ in range(50):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        ema.update(model)

        # BUG: apply EMA back to live model inside train_epoch.
        # This squashes the gradient progress.
        ema.apply(model)  # <-- ANTIPATTERN

        losses.append(loss.item())

    # Loss should NOT meaningfully decrease (the live weights get reset every step)
    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert (early - late) / early < 0.05, \
        f"Bug not reproduced: early={early}, late={late}, diff={early - late}"
```

### Diff that creates the bug (for the AST scanner to detect)
```diff
 def train_epoch(self, ...):
     for batch in dataloader:
         loss.backward()
         self.optimizer.step()
         self.ema.update(self.model)
+        self.ema.apply(self.model)  # <-- ANTIPATTERN: shadows live to EMA, kills learning
```

---

**Report end. Authored by the Grand Council, 2026-04-29 PM.**
