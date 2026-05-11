# Kaggle PR106 latent score-table harvest (2026-05-11)

Purpose: classify the Kaggle PR106 latent-sidecar run without losing the useful
compress-time CUDA signal and without promoting a proxy artifact as contest
evidence.

## Verdict

- kernel: `adpena/comma-lab-pr106-latent-score-table`
- final provider status: `KernelWorkerStatus.ERROR`
- terminal classification: `failed_proxy_harvested_table_complete_no_score_claim`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`

The kernel failed after producing the full latent score table. The failure was a
portable bundle/import bug in `experiments/build_pr106_latent_sidecar.py`: the
builder hardwired the forensic public-PR intake source path and did not prefer
the bundled contest runtime source under `submissions/pr106_latent_sidecar/src`.
That bug class is fixed and covered by a regression test.

## Harvested Signal

- evidence_dir:
  `reports/raw/kaggle_ingested/kaggle_pr106_latent_score_table_20260511T143012Z`
- table:
  `pr106_latent_score_table/latent_run/score_table/score_table.npy`
- table_sha256:
  `66460c6380dcc4dc89997fbe234005d109718c9ad61d653c11fd97fb94d94497`
- table_shape: `[600, 57]`
- candidate_count: `57`
- strict_improvement_pair_count: `600`
- best_improvement_mean: `0.0025113055016845465`
- best_improvement_max: `0.005965337157249451`
- ready_for_builder: `true`
- source_zero_bin_sha256:
  `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- source_archive_sha256:
  `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`

The `source_archive_sha256` differs from the local PR106 public-intake ZIP SHA
because Kaggle exposed/reframed the same `0.bin` payload as a deterministic
stored ZIP. This is an archive-framing difference, not a PR106 payload or model
difference; the `source_zero_bin_sha256` matches the local PR106 payload.

## Local Hardening Proof

Commands run after the fix:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_kaggle_output_ingest.py \
  src/tac/tests/test_pr106_latent_sidecar.py \
  src/tac/tests/test_pr106_latent_score_table.py \
  src/tac/tests/test_kaggle_pr106_latent_score_table.py -q

.venv/bin/python -m ruff check \
  src/tac/deploy/kaggle/kaggle_output_ingest.py \
  src/tac/tests/test_kaggle_output_ingest.py \
  experiments/build_pr106_latent_sidecar.py \
  src/tac/tests/test_pr106_latent_sidecar.py --select F
```

Both passed. A local score-table builder smoke also passed after reconstructing
the same deterministic stored-ZIP framing used by Kaggle:

- search_mode: `score_table`
- device: `cpu --smoke` for builder/wire validation only
- score_table_manifest_validated: `true`
- n_corrections: `600`
- sidecar_bytes: `561`
- archive_zip_bytes: `186808`
- delta_bytes_vs_pr106: `+569`
- score_claim: `false`

## Promotion Boundary

This result is useful score-lowering input, not a score. The next valid
promotion path is:

1. materialize the sidecar archive from this table through the fixed builder;
2. run no-op proof and archive/runtime closure checks;
3. record old/new archive SHA-256s, sidecar bytes, and consumed payload proof;
4. dispatch exact contest-CUDA auth eval under a fresh lane claim;
5. only then compare to `[contest-CUDA]` frontier rows or use the result to rank,
   kill, or submit.

Do not treat the Kaggle table as `[contest-CUDA]`, `[contest-CPU]`, or macOS CPU
auth evidence. It is a compress-time CUDA search artifact.
