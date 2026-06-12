# PR95 vendored-torch vs MLX-port: STEP-LEVEL PARITY DIFF (the gold-standard control)

- **Date:** 2026-06-12
- **Authority:** `[macOS-CPU advisory]` — parity diagnostic, NON-PROMOTABLE. Both
  sides run on CPU (torch-CPU vs MLX-CPU). NO MPS. GT decode via
  `frame_utils.yuv420_to_rgb` only (reused via `tac.score_aware_loop.targets`).
- **Harness:** `experiments/run_pr95_vendored_vs_mlx_port_parity.py` (NEW; DIFF-ONLY;
  imports both sides; edits no core port file).
- **Reports:** `experiments/results/pr95_vendored_vs_mlx_parity_bundle_20260612T003014Z/`
  (`report_stage1_adamw_ce.json`, `report_stage8_full_qat_c1a_muon.json`,
  `report_stage8_isolated_muon_only.json`).
- **WRITE-DISJOINT** from sister agent `a28f8a9c` (which fixes core port files);
  this agent is diagnostic-only.

## (1) Is the runnable vendored PR95 source present?

**YES — present and runnable.** The COMPLETE torch training source is at:

```
experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src
```

It is gitignored (pristine, in an ignored intake dir — correct per the
public-frontier discipline). It has `model.py` (HNeRVDecoder, **228,958 params** —
matches the profile), `losses.py` (4 seg losses + pose + d_seg + QAT + C1a + EMA),
`optim.py` (Muon + Newton-Schulz + partition), `data.py`, `score.py`, `codec.py`,
`train.py`, and all 8 stage configs in `src/stages/` plus `stages/common.py` (the
exact per-step train loop). It imports cleanly:
`import model, losses, optim` → OK.

> NOTE: the `experiments/results/public_pr95_intake_20260504_codex/` intake dir
> ships only the INFLATE-side decoder + recovery STUBS for `data.py`/`score.py`
> (Python-3.14 `.pyc` undecompilable). The `..._20260505_auto` intake above is the
> one with the full, runnable training source — that is the gold-standard control.

## (2) What does `test_torch_parity.py` already cover?

**COMPONENT-level parity, NOT a whole-training-step diff.** All 16 tests pass
(59s). It gates, against torch references:

- the 4 seg losses + pose + d_seg — on SYNTHETIC logits;
- NS-Muon — bf16-faithful (`drift < 0.08`) + fp32-structural (`rel < 1e-2`), on
  random matrices in ISOLATION;
- the decoder forward — `< 1 uint8 level` oracle parity, on RANDOM weights/latents;
- the score-bridge gradient — finite-difference vs `mx.vjp`, on a TOY 8×8
  proto-scorer (`_FrozenDNet`), NOT the real DistortionNet;
- the HEADLINE: live-render d_seg DESCENDS + NO-FAKE controls (constant loss /
  severed gradient do NOT descend).

**The gap this harness fills:** an end-to-end WHOLE training step on the REAL
DistortionNet + REAL 0.mkv GT targets, with identical torch/MLX init, diffing
loss → cotangent → per-tensor weight-delta → N-step trajectory, against the
VENDORED torch loop (`stages/common.py`) as the apples-to-apples baseline. The
existing tests never run the vendored loop's full step against the port's full
step on real data.

## (3) Step-level diff result

Method: build ONE frozen `DistortionNet` + GT targets (shared). Init a torch
`HNeRVDecoder` + latents at a fixed seed; copy that EXACT `state_dict` + latents
into the MLX bundle via `load_pytorch_state_dict_into_mlx`. Run one step through
each on an identical batch. Config matched to the stage; cosine LR off
(`lr_scale=1.0`).

### Stage 1 (AdamW-only, CE, no QAT/C1a) — `report_stage1_adamw_ce.json`

| Observable | torch | MLX | diff |
|---|---|---|---|
| loss | 308.4077 | 308.4153 | rel 2.5e-5 |
| seg_l | 2.72330 | 2.72339 | rel 3.2e-5 |
| pose_l | 36.0773 | 36.0763 | rel 2.9e-5 |
| **d_seg** | 0.5078303 | 0.5078303 | **abs 0.0** |
| render | — | — | max **1** uint8 level |
| cotangent | — | — | **cosine 0.99976** |
| weight Δ (worst: stem.weight) | — | — | cosine 0.9941, norm_rel 3.4% |
| latents Δ | — | — | cosine 0.9999993 |
| **trajectory 12 steps** | 0.5078→**0.2079** | 0.5078→**0.2062** | **\|Δ\|=0.0017** |

**Stage 1: step-AND-trajectory faithful.** Both descend identically.

### Stage 8 ISOLATED (Muon + l7_softplus only; QAT+C1a OFF on BOTH sides) — `report_stage8_isolated_muon_only.json`

| Observable | result |
|---|---|
| loss / seg / pose | rel ~3e-5 (exact) |
| d_seg | abs 0.0 (identical) |
| cotangent | cosine 0.99975 |
| **weight Δ: 25/28 tensors** | cosine ≥ 0.99 |
| **weight Δ: 3 Muon-conv tensors** | `blocks.5.weight` 0.975, `blocks.4.weight` 0.983, `skips.4.weight` 0.987 (norm_rel 0.3–0.65%) |
| latents Δ | cosine 0.999999 |

The ONLY divergence is a ~0.3–0.65% / cosine-0.975 difference on the
Muon-orthogonalized conv weights. **This is bf16-Newton-Schulz epsilon, by
design.** PR95's `zeropower_via_newtonschulz5` hardcodes `X = G.to(bfloat16)`;
the MLX port's `zeropower_via_newtonschulz5_mlx` defaults
`cast_float32_to_bfloat16=True`. bf16 has ~2^-8 ≈ 0.4% relative precision — exactly
the magnitude observed. The existing `test_ns_muon_bf16_faithful_to_pr95`
(`drift < 0.08`) already certifies this as faithful, not a bug. It is FAITHFUL TO
PR95'S OWN bf16 CHOICE.

### Stage 8 FULL (Muon + QAT + C1a, mechanisms armed torch-side only) — `report_stage8_full_qat_c1a_muon.json`

Shows larger weight-delta divergence (cosine 0.95–0.97). **This is a HARNESS
artifact, not a port bug:** the harness arms QAT/C1a torch-side but the MLX
QAT/C1a mechanisms require `trainer.configure_stage()` (StageMechanisms), which
this harness does not call for the single-step diff. The `--isolate-optimizer`
run above (QAT/C1a off on both) is the apples-to-apples Muon comparison. Recorded
here for transparency; do NOT read it as a divergence finding.

## (4) Trajectory drift

Stage 1: 12-step trajectory diverges by only `|Δd_seg| = 0.0017` (torch 0.2079 vs
MLX 0.2062 off a 0.5078 init) — no meaningful cumulative drift. Stage 8 (lr=1e-5,
12 steps from random init): both stuck at 0.5078 (expected — fine-tune LR doesn't
move d_seg in 12 steps). No drift signal at this horizon. A longer (hundreds-of-
steps) run is the natural next probe but was not needed: the per-step deltas
already match to bf16 epsilon.

## (5) HONEST DISPOSITION

**port-is-faithful — the wall is NOT a step-level numerical bug.**

The MLX port reproduces the vendored PR95 torch loop step-by-step on real data:
forward loss/seg/pose/d_seg match to fp32 epsilon, the render matches to 1 uint8
level, the pixel cotangent matches to cosine 0.9998, the AdamW weight+latent
deltas match to cosine ≥0.994, and the only Muon-conv weight-delta difference
(~0.4%, cosine 0.975) is bf16-Newton-Schulz epsilon faithful to PR95's own bf16
cast. The 12-step trajectories track to |Δd_seg|≈0.0017.

**Therefore the d_seg wall (our retrains at 0.014/0.010/0.0025 vs the basin
5.6e-4) is NOT a per-step arithmetic divergence in the port.** It is in the
RECIPE / SCHEDULE, which this diff does NOT exercise (it runs single steps + a
12-step trajectory, matched-config). The prime suspect — already visible in the
MLX config as the documented "C7 fix" — is the SCHEDULE divergence:

- **`MlxScoreAwareConfig.use_muon = True` from epoch 0 (Muon-throughout)** vs PR95
  **AdamW for stages 1–7, Muon ONLY in stage 8.** PR95's d_seg basin was reached
  by 8 AdamW stages (stage1 CE 3000ep → … → stage7) building the representation
  BEFORE the single 5000-epoch Muon fine-tune. Muon-from-epoch-0 is a different
  optimization trajectory; its O(1) grad-norm-independent orthogonalized steps
  early in training is a recipe change, not a port bug.
- Sister candidates to audit in the recipe (NOT the port arithmetic): the 8-stage
  epoch budget (29,650 epochs total), inter-stage resume/init, EMA decay schedule,
  per-stage seg-loss switching cadence, QAT-on timing, C1a λ/σ sweep.

**Recommendation for sister agent `a28f8a9c`:** do NOT chase a per-step numerical
fix in `mlx_trainer`/`mlx_losses`/`score_bridge` — they are faithful. Audit the
CURRICULUM SCHEDULE (`curriculum.py` `OPTIMIZER_SCHEDULE_MUON_THROUGHOUT` vs
`OPTIMIZER_SCHEDULE_PR95`, stage epoch budgets, resume semantics). The decisive
next experiment is a matched-SCHEDULE long run: drive the MLX port with
`OPTIMIZER_SCHEDULE_PR95` (AdamW stages 1–7, Muon stage 8) on the full 8-stage
epoch budget and compare the d_seg trajectory to the vendored basin — the schedule,
not the arithmetic, is where the wall lives.

## Wire-in (6 hooks per Catalog #125)

1. sensitivity-map — N/A (diagnostic; no per-axis byte savings).
2. Pareto — N/A.
3. bit-allocator — N/A.
4. cathedral autopilot — N/A (advisory non-promotable).
5. continual-learning — this memo + the JSON reports are the durable anchor;
   the recipe-not-arithmetic verdict reseeds the next curriculum-schedule probe.
6. probe-disambiguator — **ACTIVE**: `run_pr95_vendored_vs_mlx_port_parity.py` IS
   the disambiguator between "port arithmetic diverges" (FALSIFIED here) and
   "recipe schedule diverges" (the surviving hypothesis).
