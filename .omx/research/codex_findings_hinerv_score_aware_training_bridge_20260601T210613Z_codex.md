# Codex Findings: HiNeRV/SNeRV Score-Aware Training Bridge

UTC: 2026-06-01T21:06:13Z
Agent: Codex
Axis: compact learned carriers / MLX score-aware training route
Status: LANDED as planner contract, no score claim

## Verdict

The corrected carrier premise is now encoded as a reusable planner surface:
`tac.substrates._shared.mlx_score_aware.carrier_training_plan`.

The contract refuses the false-authority pattern where a projected-small
HiNeRV/SNeRV archive is treated as score movement while fit is unusable. A
carrier with tiny projected rate but high `d_seg` / advisory score routes to
`run_score_aware_decoder_weight_training_full_main`, not promotion or exact
auth.

## Corrected Evidence Preserved

- G3 adjoint exactness is useful, but only closes the gradient-transport risk.
  It is not a score claim.
- Near-zero latent JVP leverage demotes post-hoc latent allocation. The
  allocation target becomes decoder weights.
- `--modelsize` is a real structural rate knob, but cheap-by-construction is
  meaningless until the fit half lands.
- Optimizer/QAT are rate-axis tools: C1a coder-aware regularization, sigma
  noise, Quant-Noise, Muon/AdamW staging, and NVRC learned quantization must be
  part of the decoder-weight training plan rather than a finishing pass.

## Code Landing

- Added `build_score_aware_carrier_training_plan(...)`.
- Exported it from the canonical MLX score-aware package.
- Added no-MLX contract tests proving:
  - cheap-but-unfit HiNeRV evidence routes to decoder-weight training;
  - low latent leverage demotes latent posthoc allocation;
  - missing real SegNet/PoseNet teachers blocks readiness;
  - plausible local fit still routes to byte-closed local replay, not auth;
  - truthy score/promotion/exact-auth fields are rejected.

## Automation Consequence

Queue/trainer code should consume the planner row before launching HiNeRV/SNeRV
campaigns. The next executable step is to bind this row into the compact-carrier
queue runner so evidence from 2/32/128/600-pair smokes automatically selects:

1. decoder-weight score-aware training when fit is bad;
2. local byte-closed replay when fit is plausible and archive proof exists;
3. exact CPU then CUDA only after a true local win.

No large artifact was produced by this landing.

