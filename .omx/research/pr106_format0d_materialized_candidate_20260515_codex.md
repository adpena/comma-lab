# PR106 Format0D Materialized Candidate - 2026-05-15

## Status

This is a byte-closed candidate build, not a score claim.

- Candidate archive: `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`
- Candidate bytes: `186876`
- Candidate SHA-256: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- Source archive bytes: `186327`
- Source archive SHA-256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- Rate delta vs source: `+549` bytes, `+0.00036555656526407205` score units
- Materialization manifest SHA-256: `dd527859bf416da83302d07ff1f077f4ff3d3cdcb2606617e48c2f22956ef9e8`
- Build metadata SHA-256: `bb64673057e3c92adbaa8185466a68e7b982bd954a50b99f439a63e921f02c6a`

## What Changed

The source PR106 format0C PacketIR grammar could not represent all strict
score-table choices. The materializer now emits format0D when the unfiltered
strict best table contains incompatible format0C rows. Runtime apply order is:

1. base format0C corrections
2. extra PR101-ranked no-op corrections

Format0D materialization diagnostics:

- `output_format_id`: `0x0D`
- `extra_second_dim_pair_count`: `552`
- `extra_nonzero_pair_count`: `570`
- `output_packet_payload_sha256`: `15a5dccba352838df7a8dd190a8782e51afa53b475bcf07921f05ce88c10785e`

## Authority Boundary

The manifest keeps:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `packet_ir_score_affecting_payload_changed=true`
- `packet_ir_charged_archive_bytes_changed=true`

The candidate requires paired exact eval before any promotion:

- exact `[contest-CUDA]`
- exact `[contest-CPU]`
- runtime decode/apply proof for the semantic PacketIR format0D candidate
- adjudicated component recomputation

Do not compare this to PR101/FEC6 CPU or PR106 format0C CUDA until both axes
are harvested against the same archive/runtime custody.

## Reproducer

```bash
.venv/bin/python tools/materialize_pr106_latent_score_table_candidate.py \
  --source-archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --score-table-npy reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z/pr106_latent_score_table/latent_run/score_table/score_table.npy \
  --score-table-manifest reports/raw/kaggle_ingested/kaggle_pr106_format0c_latent_score_table_repair2_20260515T204229Z/pr106_latent_score_table/latent_run/score_table/score_table_manifest.json \
  --output-dir experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex \
  --top-k 600
```
