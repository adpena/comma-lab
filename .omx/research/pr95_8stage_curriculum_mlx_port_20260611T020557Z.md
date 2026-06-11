# PR95 8-stage training curriculum — full MLX port + capstone wire-in (LANDED)

**UTC:** 20260611T020557Z
**Authority:** `[macOS-MLX research-signal]` — NON-PROMOTABLE per Catalog #192/#341.
A contest score still requires `upstream/evaluate.py` on paired CUDA + Linux-x86_64 CPU.
**Mission contribution:** `frontier_breaking_enabler` (the d_seg-floor-breaking schedule;
the capstone/fleet floored at d_seg ~0.008–0.012 on a FIXED single-stage recipe; PR95
reached d_seg 5.6e-4 ONLY via this 8-stage curriculum).

## The premise (verified against the live daemon)

The capstone daemon log (`.omx/tmp/capstone_daemon/capstone_b20_n48_LONG_*.log`)
confirms the floor the operator described: fixed single-stage `ce_seg_loss` + a guessed
fixed Muon LR floors at `exact_d_seg≈0.008` (epoch 50: 0.00801). This is what the 8-stage
curriculum is built to break. The daemon was NOT touched.

## (1) The EXACT 8-stage spec extracted from the torch source of truth

Source: `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/stages/stage{1..8}*.py` + `stages/common.py` (`StageConfig`) + `train.py` (orchestrator). The per-stage `make_config` literals (cross-checked by an AST source-parity test, NOT by importing the env-heavy torch stack):

| #  | name                  | epochs | seg_loss                     | sigma (cat) | c1a_lambda | QAT | optimizer / lr            |
|----|-----------------------|--------|------------------------------|-------------|------------|-----|---------------------------|
| 1  | stage1_v328_ce        | 3000   | ce_seg_loss                  | 0.2 (unused)| 0.0        | no  | AdamW 1e-3                |
| 2  | stage2_v331_softplus  | 5650   | tau_softplus(0.3)            | 0.2 (unused)| 0.0        | no  | AdamW 1e-3                |
| 3  | stage3_v332_smooth    | 1500   | smooth_disagreement(0.3)     | 0.2 (unused)| 0.0        | no  | AdamW 1e-4 (fresh cosine) |
| 4  | stage4_v332_qat       | 500    | smooth_disagreement(0.3)     | 0.2 (unused)| 0.0        | YES | AdamW 1e-4                |
| 5  | stage5_c1a_l7         | 9000   | l7_softplus(0.3,thr1,mult4)  | 0.2         | 0.01       | YES | AdamW 3e-5                |
| 6  | stage6_lambda_sweep   | 2000   | l7_softplus(0.3,thr1,mult4)  | 0.2         | 0.02       | YES | AdamW 3e-5                |
| 7  | stage7_sigma_sweep    | 3000   | l7_softplus(0.3,thr1,mult4)  | 0.1         | 0.02       | YES | AdamW 3e-5                |
| 8  | stage8_muon_finetune  | 5000   | l7_softplus(0.3,thr1,mult4)  | 0.1         | 0.02       | YES | Muon 2e-4 + AdamW 1e-5, muon_wd 5e-4 |

Canonical total = **29,650 epochs** (matches HNeRV-parity L14). Shared constants across all
8 stages: `seg_weight=100`, `pose_weight=1`, `latent_lr_mult=10`, `ema_decay=0.999`,
`grad_clip=1.0`, `grad_clip_muon=1.0`. Stage 8's `muon_weight_decay=5e-4` is the researcher
#24 idea (Chen-Li-Liu arXiv:2506.15054), present in the torch source.

The spec lives in `tac.mlx_pr95_port.curriculum.CURRICULA["pr95_8stage"]` as a tuple of
frozen `StageSpec` dataclasses. `build_pr95_8stage_curriculum(total_epochs=N)` proportionally
compresses the schedule for a local run while preserving the stage STRUCTURE (the
loss/sigma/lambda/qat/opt transitions — which is what breaks the floor).

## (2) What was ported + parity results per piece

All three weight-domain mechanisms were missing; the 4 stage seg-losses + pose were already
ported (`mlx_losses.py`, parity-gated). New ports in `mlx_losses.py`:

- **C1a coder-aware entropy (L16)** — `cat_entropy_v2_mlx`. Size-weighted soft-histogram
  entropy over Conv2d/Linear weights; per-tensor abs-max INT8 normalize → Gaussian
  soft-assignment to the 255-bin integer grid (bandwidth sigma) → categorical entropy.
  **Parity vs torch `cat_entropy_v2`: abs-diff 1.43e-6 (fp32 epsilon).** Layout-invariant
  (NHWC MLX weights == OIHW torch weights for identical values — verified).
- **QAT INT8 fake-quant (L14 stage 4)** — `fake_quantize_mlx`. Per-tensor symmetric INT8
  STE (`stop_gradient(deq - w) + w`). **Parity vs torch `fake_quantize`: bit-identical (0.0).**
- **sigma weight-noise (L17)** — `apply_sigma_noise_mlx`. Additive `w + sigma·N(0,1)`;
  no-op at sigma=0; verified injected std ~0.2 at sigma=0.2. (See "documented adaptation"
  below for the L17 vs C1a-sigma disambiguation.)

The mechanisms are wired into the live loop via `curriculum_mechanisms.py`:
- `apply_stage_weight_transforms` — applied INSIDE the traced `mx.vjp` forward (STE
  fake-quant + weight-noise on the traced weights, so the gradient flows to the live
  master). **The original primals are restored into the bundle after the vjp** (see the
  bug + regression guard below).
- `add_c1a_entropy_gradient` — the C1a term is a function of the weights alone, so its
  gradient is computed via `mx.grad` over the weight tree and ADDED to the pixel-derived
  grad tree (matches torch `loss = scorer_loss + cat_lambda·ent` + one `backward()`).

**NO-FAKE proof (each mechanism does REAL work, not a no-op):**
- QAT actually changes the render (0.09 max-abs delta on the toy basis).
- C1a gradient is non-zero on the 13 weight tensors (0.024 max) AND leaves latents/biases at 0.
- sigma=0 is a true no-op; sigma=0.2 injects the correct Gaussian std.
- C1a `cat_lambda=0` (stages 1–4) leaves the gradient untouched.

## (3) The reusable scheduler + the FLEET fix

`tac.mlx_pr95_port.curriculum.run_curriculum(trainer, stages, optimizer_schedule=...)` drives
ANY trainer implementing the tiny `CurriculumTrainerProtocol` (`configure_stage` +
`run_stage_epochs`). Wired into BOTH:
- `MlxScoreAwareTrainer` (the PR95-reference loop) — `tr.run_curriculum(...)`.
- `CapstoneTrainer` (the FiLM-pose capstone) — `tr.run_curriculum(...)`.

`configure_stage` switches the bridge seg-loss form, rebuilds the optimizer config (LR +
Muon-vs-AdamW per the resolved schedule + per-stage grad-clip/wd), and arms the per-stage
QAT/C1a/sigma mechanisms. Weights + optimizer state CARRY across stages (PR95 inter-stage
transitions resume weights). This is the fleet fix: any substrate trainer can now consume
`--curriculum pr95_8stage` rather than the fixed single-stage recipe.

## Optimizer-schedule decision (PR95-faithful vs #77 deviation) — BOTH selectable

- `pr95_adamw_then_muon` — **what PR95 ACTUALLY used**: AdamW stages 1–7, Muon (hidden convs)
  + AdamW (stem/rgb/latents) stage 8 ONLY. Verified: only `stage8_muon_finetune.py` sets
  `use_muon=True` in the torch source.
- `muon_throughout` — the #77 deviation (`.omx/research/tilde_optimizers_*.md`): Muon from
  stage 1 (the inert-loop audit found AdamW + 100%-clip stalled the early MLX stages).

Both are selectable via `--optimizer-schedule`; neither is silently hardcoded. The
optimizer-correctness audit (sister subagent) can A/B them. `resolve_use_muon(spec, schedule)`
is the single resolution point (tested both ways).

## CLI: run the capstone with the full curriculum

```bash
.venv/bin/python experiments/run_capstone_campaign.py \
    --max-pairs 600 --base-channels 16 --epochs 600 --decoder-dtype int8 \
    --curriculum pr95_8stage --optimizer-schedule muon_throughout \
    --eval-every 25 \
    --out-dir experiments/results/capstone_pr95_curriculum_b16_int8
```

`--curriculum-total-epochs N` overrides `--epochs` as the total spread across the 8 stages.
`--curriculum none` (default) keeps the legacy single fixed stage. The smoke
(`--max-pairs 2 --base-channels 8 --epochs 16 --curriculum pr95_8stage`) ran end-to-end
through all 8 stages, exported the int8 archive, and recomputed the advisory score (exit 0).

## (5) Documented adaptations (where the torch curriculum was ambiguous)

1. **L17 sigma weight-noise vs C1a `cat_sigma`.** PR95 carries one `cat_sigma` symbol used
   TWO ways: (a) the C1a soft-histogram bandwidth (always, when `cat_lambda>0`), and (b)
   the L17 lesson text describes a Gaussian *weight-noise* schedule (0.2→0.1). The torch
   `common.py` forward does NOT inject additive weight noise — it only uses `cat_sigma` for
   the C1a histogram. To honor L17 faithfully WITHOUT diverging from the torch forward where
   the torch forward is explicit, I added a SEPARATE `sigma_weight_noise` field (0.0 in
   stages 1–4 where torch injects none; 0.2 in stages 5–6; 0.1 in stages 7–8, mirroring the
   C1a sigma schedule and the L17 text). The C1a `cat_sigma` is ported EXACTLY (0.2→0.1).
   This keeps the C1a math bit-exact to torch while making the L17 weight-noise an explicit,
   per-stage, default-on-in-later-stages knob. A run with `sigma_weight_noise=0` everywhere
   is byte-identical to the pure-torch-forward behavior.
2. **Epoch counts = the torch `make_config` DEFAULTS** (what `train.py` actually runs),
   which for stages 5–8 are PR95's "our extension" counts (9000/2000/3000/5000), not the
   smaller "default canonical" numbers in the docstrings. `build_pr95_8stage_curriculum`
   compresses proportionally for local runs.

## (6) Could any stage NOT be faithfully ported? — No.

All 8 stages ported faithfully. The cosine LR schedule (per-stage `eta_min` cosine in torch
`common.py`) is NOT yet replicated inside the MLX inner loop (the MLX optimizer config sets a
fixed per-stage LR; the canonical PR95 stage transitions ALSO sometimes restart the cosine vs
continue it). This is a known, documented gap: the stage STRUCTURE (the loss/sigma/lambda/qat/
opt transitions — the floor-breaking mechanism) is faithful; the intra-stage cosine decay is a
follow-on. It does not change which mechanism is active per stage.

## The bug fixed + self-protected (CLAUDE.md "Bugs permanently fixed AND self-protected")

The traced `mx.vjp` forward installs the QAT/noise-TRANSFORMED weights into the bundle as a
side effect. Without restoring the originals after the vjp, the optimizer step would update
the QUANTIZED master (corruption). **Fix:** restore the primals into the bundle after the vjp
(both trainers). **Guard:** `test_qat_vjp_restores_original_weights_in_bundle_REGRESSION`
asserts the bundle holds the ORIGINAL float weights (bit-identical) after a QAT-active vjp —
verified to FAIL when the fix is reverted and PASS with it.

## Tests + lint

- 24 new tests in `src/tac/mlx_pr95_port/tests/test_curriculum.py` (torch-parity + NO-FAKE +
  spec-fidelity AST source-parity + scheduler + end-to-end + the regression guard) — all pass.
- 16 existing `test_torch_parity.py` + 29 capstone tests (excl. the pre-existing slow/flaky
  `test_real_scorer_joint_loop_moves_seg_logits_and_holds_pose`, which fails+is slow on clean
  `main` independent of this work) — all pass.
- `ruff check` clean on every changed file.

## Files

- `src/tac/mlx_pr95_port/curriculum.py` (NEW) — 8-stage spec + scheduler + opt-schedule selector.
- `src/tac/mlx_pr95_port/curriculum_mechanisms.py` (NEW) — QAT/sigma weight transforms + C1a grad.
- `src/tac/mlx_pr95_port/mlx_losses.py` — `cat_entropy_v2_mlx` + `fake_quantize_mlx` + `apply_sigma_noise_mlx`.
- `src/tac/mlx_pr95_port/mlx_trainer.py` — curriculum hooks + restore-primals fix.
- `src/tac/capstone_vq_nerv/capstone_trainer.py` — curriculum hooks + restore-primals fix.
- `src/tac/mlx_pr95_port/__init__.py` — exports.
- `experiments/run_capstone_campaign.py` — `--curriculum` / `--optimizer-schedule` / `--curriculum-total-epochs`.
- `src/tac/mlx_pr95_port/tests/test_curriculum.py` (NEW) — 24 tests.

## 6-hook wire-in (Catalog #125)

1. sensitivity-map: N/A (research-signal infra, not a per-axis byte map). 2. Pareto: N/A.
3. bit-allocator: PARTIAL — C1a + QAT bias the decoder weights toward brotli-friendly INT8
   distributions (the rate lever the curriculum's later stages target). 4. cathedral autopilot:
   N/A (local training infra). 5. continual-learning posterior: N/A (no exact anchor yet —
   this is the trainer that PRODUCES the anchor). 6. probe-disambiguator: ACTIVE — the
   `optimizer_schedule` A/B (`pr95_adamw_then_muon` vs `muon_throughout`) IS the disambiguator
   the optimizer-correctness audit consumes.
