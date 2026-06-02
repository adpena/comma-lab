# Codex Findings - scorer-only objective authority

date_utc: 2026-06-02T12:40:09Z
agent: codex
cwd: /Users/adpena/Projects/pact
authority: false_local_and_control_plane_only
score_claim: false
promotion_eligible: false
exact_cpu_cuda_eval_executed: false

## Directive preserved

Human visual fidelity is not an optimization authority for this contest. PSNR, SSIM, LPIPS, perceptual quality, and human-looking output are forbidden as model-size or archive-selection terms unless a separate proof shows they causally improve the contest auth eval scorer.

Allowed selection terms are:
- SegNet last-frame distortion.
- PoseNet pair distortion.
- Archive ZIP byte rate term.

## Code wiring

- `tac.analysis.nerv_modelsize_ladder` now emits `objective_authority`.
- `tac.analysis.hinerv_archive_size_ladder` now emits the same `objective_authority`.
- `tac.analysis.nerv_control_inventory` now emits the same `objective_authority`.

The measured HiNeRV archive ladder still carries the required allocator bindings:
- adaptive quantization by decoder-weight group;
- ablation/zeroing of nonpositive-value groups;
- waterfilling against the fixed contest byte price;
- inverse-steg saliency binding;
- packed-zero and entropy-coded low-value groups.

## Durable artifacts

- Latest measured ladder with scorer-only authority: `.omx/research/hinerv_archive_size_ladder_20260602T124000Z.json`
- Latest inventory with scorer-only authority and measured ladder attached: `.omx/research/nerv_control_inventory_20260602T124009Z.json`

## Verification

- Focused pytest: `74 passed`.
- Ruff: clean.
- Py compile: clean.

## Remaining blockers

- No non-rate scorer replay is attached to the measured ladder.
- No exact CPU/CUDA replay was executed.
- The scorer-only objective must be enforced inside the next score-aware trainer and allocator, not only in reports.
