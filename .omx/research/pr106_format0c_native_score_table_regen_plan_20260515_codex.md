# PR106 format0C native score-table regeneration plan - 2026-05-15

score_claim: `false`
promotion_eligible: `false`
ready_for_exact_eval_dispatch: `false`
dispatch_attempted: `false`

## Scope

Prepare the next score-bearing PR106 format0C native score-table path without
remote/GPU dispatch. The key guard is that old score-table manifests generated
against PacketIR format `0x07` / legacy single-member `0.bin` custody must not
be reused for the native format0C single-member `x` archive.

## Canonical CLIs

- Score-table producer:
  `experiments/build_pr106_latent_score_table.py`
- Local materializer:
  `tools/materialize_pr106_latent_score_table_candidate.py`
- Legacy 0x01 builder used only by the materializer for raw/0x01 sources:
  `experiments/build_pr106_latent_sidecar.py --search-mode score_table`

Current code status:

- The producer accepts `--runtime-dir` and `--archive-member`, auto-detects
  `x`, and has `--dry-run-plan`.
- The materializer uses `format0c_packet_ir_native` when the source archive is
  format `0x0C`, emits a single-member `x` archive, and refuses to route
  non-0x01 PacketIR through the legacy builder.
- The score-table manifest validator requires explicit
  `source_archive_member_sha256` for format0C and does not allow the legacy
  `source_zero_bin_sha256` fallback on format0C `x`.

## Existing stale manifest blocker

Existing Kaggle latent score-table evidence:

`reports/raw/kaggle_ingested/kaggle_pr106_latent_score_table_20260513_codex_clean/pr106_latent_score_table/latent_run/score_table/score_table_manifest.json`

It targets the old source custody:

- `source_archive_sha256=cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`
- `source_archive_member_name=null`
- `source_archive_member_sha256=null`
- `source_zero_bin_sha256=7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- `runtime_dir=null`

Native format0C source archive:

`experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`

- archive SHA-256:
  `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- archive bytes: `186327`
- single member: `x`
- member payload SHA-256:
  `852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749`
- sidecar format id: `0x0C`

Attempting to materialize the stale Kaggle manifest against the format0C `x`
archive fails closed with:

`ValueError: score table manifest source_archive_member_sha256 mismatch for format0C source archive`

This is the desired blocker. Do not bypass it by copying the old
`source_zero_bin_sha256` value into a format0C plan.

## Generated local dry-run regeneration plan

Command run locally, with no CUDA and no remote dispatch:

```bash
.venv/bin/python experiments/build_pr106_latent_score_table.py \
  --pr106-archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --out-dir experiments/results/pr106_format0c_scoretable_regen_plan_20260515_codex \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --archive-member x \
  --delta-radius 2 \
  --latent-dim 28 \
  --n-pairs 600 \
  --dry-run-plan
```

Artifacts:

- `experiments/results/pr106_format0c_scoretable_regen_plan_20260515_codex/score_table_manifest.json`
  - SHA-256:
    `30a48f7a219de600f9d8ef166165a51d334b3224d07764647862a0ebbebfc5b8`
  - schema: `pr106_latent_score_table_plan_v1`
  - source member: `x`
  - source member SHA-256:
    `852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749`
  - sidecar format id: `0x0C`
  - runtime dir: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
  - expected shape: `[600, 113]`
  - dispatch blockers include `dry_run_plan_only` and
    `requires_real_cuda_score_table`
- `experiments/results/pr106_format0c_scoretable_regen_plan_20260515_codex/candidate_grid.npy`
  - SHA-256:
    `35903b191807c4fb11c1484c0984cbb4c2c09fc3fa647e4d22e569b54528fb43`

## Next score-bearing step

After an explicit operator-approved compute step and lane claim, regenerate the
real CUDA score table against exactly the same archive/member/runtime contract:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_format0c_latent_score_table \
  --platform local_or_provider_cuda \
  --instance-job-id <format0c-score-table-job-id> \
  --status active_dispatching \
  --notes "PR106 format0C latent score-table generation; archive_sha=56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7; member=x; member_sha=852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749"

.venv/bin/python experiments/build_pr106_latent_score_table.py \
  --pr106-archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --out-dir <score-table-output-dir> \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --archive-member x \
  --delta-radius 2 \
  --latent-dim 28 \
  --n-pairs 600 \
  --device cuda \
  --batch-pairs 2 \
  --candidate-batch-size 8 \
  --resume-checkpoint \
  --lane-id lane_pr106_format0c_latent_score_table \
  --instance-job-id <format0c-score-table-job-id>

.venv/bin/python tools/materialize_pr106_latent_score_table_candidate.py \
  --source-archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --score-table-npy <score-table-output-dir>/score_table.npy \
  --score-table-manifest <score-table-output-dir>/score_table_manifest.json \
  --output-dir <materialized-format0c-score-table-candidate-dir> \
  --delta-radius 2 \
  --top-k 600
```

The materialized archive remains non-promotional until runtime decode/apply
proof, paired `[contest-CUDA]` and `[contest-CPU]` auth eval, and result review
land with terminal lane-claim evidence.

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_pr106_latent_score_table.py \
  src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py
```

Observed: `20 passed`.

```bash
.venv/bin/ruff check \
  experiments/build_pr106_latent_score_table.py \
  tools/materialize_pr106_latent_score_table_candidate.py \
  src/tac/packet_compiler/pr106_latent_sidecar_selection.py \
  src/tac/tests/test_pr106_latent_score_table.py \
  src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py
```

Observed: `All checks passed!`
