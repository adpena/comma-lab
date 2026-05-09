# A1 per-pair latent sidecar pre-dispatch custody blocker (2026-05-09)

## Classification

`experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z`
is **refused-dispatch / custody-blocked** as inspected on 2026-05-09.

This is not a method negative and not a score claim. It is an archive/runtime
custody failure before exact eval.

## Evidence

Command:

```zsh
test -f experiments/results/a1_per_pair_latent_sidecar_resampled_20260509T103155Z/submission_dir/archive.zip
```

Result: exit code `1`.

Manifest fields:

```text
lane_id = lane_a1_inflate_time_bias_correction_sweep
ready_for_exact_eval_dispatch = true
new_archive_sha256 = 238f3adeeeee8995118ccfe78f685e464c167cb15dc4ff0f9174b2cd7428a702
new_archive_bytes = 178316
candidate_archive_path = null
submission_dir = null
```

Local files present:

```text
sidecar_manifest.json
sidecar_search.log
submission_dir/
```

No archive file was present under `submission_dir/`.

## Failure class

- Missing scored archive bytes.
- Missing archive path in manifest.
- Stale/mismatched lane id inherited from the A1 bias-correction sweep.
- `ready_for_exact_eval_dispatch=true` is invalid for this inspected artifact.

## Reactivation criteria

Regenerate the sidecar candidate with:

- present `submission_dir/archive.zip`;
- archive bytes and SHA-256 matching manifest;
- sidecar-specific lane id;
- runtime-tree SHA or custody manifest;
- member/section SHA-256s;
- no-op proof that the sidecar bytes are consumed by inflate;
- local runtime smoke;
- exact CPU GHA only after custody passes;
- CUDA dispatch only after CPU-positive evidence and a fresh lane claim.
