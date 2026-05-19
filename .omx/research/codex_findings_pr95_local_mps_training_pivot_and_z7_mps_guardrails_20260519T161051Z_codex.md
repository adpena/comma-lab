# Codex Findings: PR95 Local MPS Training Pivot And Z7 MPS Guardrails

generated_utc: 2026-05-19T16:10:51Z
author: codex
lane_id: lane_pr95_local_mps_source_faithful_training_probe_20260519
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Findings

1. The correct local PR95 control root is the full public intake:
   `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon`.

2. The public PR95 source tree is locally portable enough for a real MPS training
   smoke. A 1-epoch Stage 1 run over the full 600-pair video completed on MPS
   with `PYTORCH_ENABLE_MPS_FALLBACK=0` through the PR95 archive-build/eval loop.
   Artifact:
   `experiments/results/pr95_local_mps_source_faithful_smoke_20260519T1612Z/manifest.json`.

3. The smoke is advisory only. It proves local Apple Silicon can run the lifted
   PR95 training stack as a velocity and transfer probe. It does not provide
   leaderboard, rank, kill, or promotion authority. Authority still requires
   byte-closed archive/runtime replay through exact contest CPU and contest CUDA.

4. Two execution hazards were confirmed independently by subagents:
   - PR95 `train.py` selects only CUDA-or-CPU; MPS needs an explicit harness or
     source patch.
   - PR95 `compress.sh` looks for checkpoints under `src/ckpts`, while
     `train.py` writes under sibling `ckpts`; a direct 50-hour shell burn can
     train successfully and then fail the final zip step.

5. I landed a Pact-side runner instead of mutating the historical public intake:
   `tools/run_pr95_local_training_probe.py`. It sets `COMMA_CHALLENGE_ROOT`,
   selects `auto|cuda|mps|cpu`, refuses MPS fallback unless explicitly allowed,
   records seed/source-tree hash/public-archive hash/device/Torch/platform, and
   provides per-stage epoch overrides for smoke runs.

6. Z7 Mamba-2 MPS/MLX guardrails are tightened in the same pass:
   - MPS local training and MLX metadata are false-authority by default.
   - `mamba_ssm` auto-selection is gated to Linux+CUDA.
   - Z7 handoff rejects MPS inflate verification and requires CPU/CUDA
     `inflate_verify` evidence before dispatch commands are surfaced.
   - The 10-epoch paired exact eval correction is: lower score wins; recurrent
     beat same-byte static by 0.3376169220 on contest-CUDA and 0.2381903294 on
     contest-CPU, but the absolute packet remains non-frontier.

## PR95 Local Training Result

Canonical public PR95 MPS smoke:

```text
manifest: experiments/results/pr95_local_mps_source_faithful_smoke_20260519T1612Z/manifest.json
source_tree_sha256: 11a18e07427f57e3e9ac963902c2a21083eca62fcd52a949aa66406dff3ae2db
public_archive_sha256: e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a
device: mps
torch: 2.11.0
fallback: PYTORCH_ENABLE_MPS_FALLBACK=0
stage: stage1_v328_ce
epochs: 1
eval_every: 1
wall_seconds: 54.2971
advisory_score: 83.26466405793659
archive_bytes: 247247
```

This falsifies the strong conservative assumption that PR95 training is not
MPS-portable at all. It does not prove gradient convergence equivalence, final
frontier quality, Muon-stage MPS stability, or exact CPU/CUDA transfer.

## Required Parity Ladder

Before local-trained PR95 weights influence score decisions:

1. Source control: PR head SHA, source-tree hash, public archive SHA, dependency
   lock, seed, PRNG state, device, Torch version, and `COMMA_CHALLENGE_ROOT`.
2. Source replay control: existing PR95 archive replayed through source runtime
   with raw-output manifest and component recomputation.
3. Training smoke: differentiable YUV6 patch, eval-roundtrip-in-loss, archive
   parser/build, stage output paths, and seconds/epoch.
4. Local burn: stage checkpoint manifests, final decoder/latent hashes, and
   Muon stability evidence.
5. Archive closure: final `archive.zip` and 3-arg `inflate.sh` path, no sidecars,
   no scorer loads at inflate.
6. Authority replay: exact contest CPU and exact contest CUDA with archive SHA,
   runtime-tree SHA, raw-output aggregate SHA, component deltas, and score.

## Next Action

Run a longer local Stage 1 window via:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  .venv/bin/python tools/run_pr95_local_training_probe.py \
  --device mps \
  --stage-epochs 1=25 \
  --eval-every 5 \
  --output-dir experiments/results/pr95_local_mps_source_faithful_stage1_25ep_<stamp>
```

Use the resulting loss/component trend to decide whether local MPS is a serious
long-burn training substrate or only a smoke/profiling substrate. MLX/Metal port
work should wait until this PyTorch-MPS parity ladder shows stable transfer
signals, because MLX would no longer be source-faithful and must earn its own
archive/parity contract.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  tools/run_pr95_local_training_probe.py \
  src/tac/tests/test_run_pr95_local_training_probe.py \
  src/tac/optimization/mamba2_predictor.py \
  experiments/train_substrate_time_traveler_l5_z7_mamba2.py \
  src/tac/tests/test_z7_mamba2_scaffold.py \
  tools/verify_z7_exact_eval_handoff.py \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py \
  src/tac/quantization_wave/mlx_inference_path.py \
  src/tac/quantization_wave/tests/test_quantization_wave_helpers.py \
  tools/mlx_bitnet_158_pilot.py

bash -n scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools:upstream \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/tests/test_run_pr95_local_training_probe.py \
  src/tac/tests/test_z7_mamba2_scaffold.py \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py \
  src/tac/quantization_wave/tests/test_quantization_wave_helpers.py

Result: ruff passed, shell syntax passed, 102 pytest tests passed.
```
