# Codex Findings: PR95 MLX Stage 8 Latent Package Blocker

UTC: 2026-05-27T00:24:56Z

## Summary

The next PR95/HNeRV MLX closure step was attempted against the latest Stage 8
long-training smoke checkpoint with `--latents-from-pt`. The package command
failed before writing a contest archive because the checkpoint only contains 4
learned latents, while the public PR95 archive contract requires 600 pairs.

This is a correct fail-closed blocker: substituting source archive latents would
exercise packaging plumbing, but it would not prove that MLX-trained latents can
produce a PR95-compatible runtime-consumed archive.

## Evidence

Command:

```bash
.venv/bin/python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt experiments/results/pr95_mlx_long_training_execute_smoke_20260525T1845Z/stage08_converge_low_lr_epoch000008_20260525T184202Z.pt \
  --source-archive-zip experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip \
  --source-submission-root experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon \
  --output-submission-dir experiments/results/pr95_mlx_long_training_stage08_latents_from_pt_package \
  --latents-from-pt \
  --report-out experiments/results/pr95_mlx_long_training_stage08_latents_from_pt_package_report.json
```

Failure:

```text
PR95 meta {'n_pairs': 600, 'latent_dim': 28, 'base_channels': 36, 'eval_size': [384, 512]} does not match latent shape (4, 28)
```

## Interpretation

The current Stage 8 MLX artifact is a local smoke/timing/export artifact, not a
full PR95 reproduction checkpoint. It is still useful for optimizer/runtime
plumbing, but it must not be promoted as trained-latent archive closure.

## Next Required Work

- Add a PR95 MLX queue mode that trains or initializes all 600 latent codes with
  the public PR95 metadata contract.
- Once a 600-pair checkpoint exists, rerun `--latents-from-pt`, runtime
  consumption proof, full declared-file-list inflate parity, and scorer drift
  characterization.

## Guard Landed

`tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py` now validates
latent rank, latent count, and latent dimension against the source PR95 archive
metadata before touching the output submission directory. The CLI returns a
concise fatal error with `checkpoint_latent_count_mismatch` instead of emitting a
lower-level traceback.

Regression:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pr95_mlx_pytorch_archive_package.py -q
```

No score claim, promotion eligibility, rank/kill eligibility, or exact dispatch
authority is produced by this failed package attempt.
