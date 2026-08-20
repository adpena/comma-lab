# 11 — Evidence and citations

## Local codebase evidence

- `src/tac/contest_eval_contract.py`:
  - score formula snippets,
  - archive byte authority,
  - SegNet last-frame,
  - PoseNet both-frame YUV6,
  - derivative `5/sqrt(10*d_pose)`.
- `src/tac/archive_byte_profile.py`:
  - `CONTEST_ORIGINAL_BYTES = 37_545_489`,
  - byte price `25 / CONTEST_ORIGINAL_BYTES`.
- `src/tac/training/long_training_canonical.py`:
  - current live/EMA archive selection exports and hashes candidates,
  - current selection proxy is still local false authority,
  - no parse-back replay hook in selection.
- `src/tac/substrates/_shared/mlx_score_aware/loss.py`:
  - target-region min-ratio floor emits score-weighted unsolved argmax mass,
  - direct-live PoseNet emits score term and marginal at current ref.
- `.omx/research/pr95_8stage_curriculum_forensic_20260513.md`:
  - PR95 eight stages,
  - eval-roundtrip,
  - QAT timing,
  - parse-back selection,
  - differentiable YUV6 fix.
- `.omx/research/snerv_official_mfu_hfr_tub_forward_parity_20260605T125926Z.json`:
  - MFU/HFR primitive proof,
  - TUB/full-stack blockers,
  - receiver proof not source-forward bound.

## External sources

- HNeRV: official paper/repo establishes content-adaptive embeddings and high-resolution capacity controls.
- HiNeRV: official repo documents `bitstream-q`, `torchac`, patch-size/batch-size behavior, and compressed bitstream evaluation.
- SNeRV: paper/repo establishes DWT LF/HF, MFU, HFR, and TUB temporal extension.
