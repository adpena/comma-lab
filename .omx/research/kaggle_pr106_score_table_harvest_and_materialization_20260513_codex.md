# Kaggle PR106 score-table harvest and local materialization (2026-05-13)

## Scope

Recovered the completed Kaggle PR106 latent and y-shift score-table kernels
through the canonical paginated SDK path, not the raw `kaggle kernels output`
CLI. Kaggle remains a diagnostic/proxy substrate here:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

No exact-eval dispatch was launched from this work.

## Code changes

- Added `kaggle>=1.6,<2` to the `cloud` optional dependency and refreshed
  `uv.lock`.
- Extended `src/tac/deploy/kaggle/kaggle_output_ingest.py` so score-table
  ingest recognizes both:
  - `pr106_latent_score_table_manifest_v1`
  - `pr106_yshift_score_table_manifest_v1`
- Hardened ingest against stale `latest/` directory reuse: when
  `kaggle_output_download_manifest.json` is present, only files named by that
  manifest are copied into the evidence directory.
- Added `tools/harvest_kaggle_pr106_yshift_score_table.py`, symmetric with the
  existing latent harvester.
- Updated both PR106 score-table harvesters to include the matching small source
  archive (`pact_pr106_*_workspace/inputs/pr106_archive.zip`) needed for local
  materialization.
- Surfaced the y-shift Kaggle bundle/harvest tools in `tools/operator_briefing.py`.

## Clean harvested evidence

### Latent score table

- Kernel: `adpena/comma-lab-pr106-latent-score-table`
- Evidence dir:
  `reports/raw/kaggle_ingested/kaggle_pr106_latent_score_table_20260513_codex_clean`
- Source archive SHA-256:
  `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`
- Source `0.bin` SHA-256:
  `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- Score table shape: `[600, 113]`
- Score table SHA-256:
  `326038a5e2dbb9daf8498e7e8c07fdfc15d832b443b7ad1758cab4f0517001fe`
- Strict improving pairs: `600`
- Mean pair objective improvement before rate: `0.003279004944488406`
- Manifest says `ready_for_builder=true`, `ready_for_exact_eval_dispatch=false`.

### Y-shift score table

- Kernel: `adpena/comma-lab-pr106-yshift-score-table`
- Evidence dir:
  `reports/raw/kaggle_ingested/kaggle_pr106_yshift_score_table_20260513_codex_clean`
- Source archive SHA-256:
  `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`
- Source `0.bin` SHA-256:
  `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- Score table shape: `[1200, 343]`
- Score table SHA-256:
  `c51440d83064be6f3a08b30dce1e4ff4d59d8bb6dcf7a97bd369d67bf6289fe0`
- Strict improving frames: `420`
- Mean frame objective improvement before rate: `0.0009999644244089723`
- Manifest says `ready_for_builder=true`, `ready_for_exact_eval_dispatch=false`.

## Local byte-closed materialization

Both local materializations used the clean evidence dirs and the matching
`cb9976bd...6182` source archive. Builder manifest validation passed.

### Latent materialization

- Output:
  `experiments/results/pr106_latent_score_table_materialized_20260513_codex_clean/sidecar_archive.zip`
- Archive bytes: `186822`
- Archive SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- Materialization manifest:
  `experiments/results/pr106_latent_score_table_materialized_20260513_codex_clean/materialization_manifest.json`
- Classification: byte-closed candidate, not promotion eligible.

### Y-shift materialization

- Output:
  `experiments/results/pr106_yshift_score_table_materialized_20260513_codex_clean/pr106_yshift_sidechannel_archive.zip`
- Archive bytes: `186664`
- Archive SHA-256:
  `00ca1a4200a8f4b3c884582457664eb128420872b55f6c474d2456d6b8f51635`
- Materialization manifest:
  `experiments/results/pr106_yshift_score_table_materialized_20260513_codex_clean/materialization_manifest.json`
- Classification: byte-closed candidate, not promotion eligible.

## Diagnostic Kaggle/P100 auth results

These are copied from the Kaggle diagnostic logs and are not exact contest
authority:

- Latent archive `7f926...3a3f`: `0.2066238854574151 diagnostic_cuda`,
  GPU `Tesla P100-PCIE-16GB`, `score_claim=false`.
- Y-shift archive `00ca...1635`: `0.2096169552710872 diagnostic_cuda`,
  GPU `Tesla P100-PCIE-16GB`, `score_claim=false`.

The current strongest PR106 exact CUDA path remains the PR101-grammar R2 packet
(`c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`,
about `0.2066181 [contest-CUDA]`). These Kaggle diagnostics do not supersede it.

## Next score-lowering implications

1. Do not dispatch the y-shift standalone packet as a score-lowering exact eval;
   the diagnostic signal is worse than the current PR106 R2 grammar floor.
2. Treat latent score-table materialization as a custody bridge and known archive
   reproduction, not a new frontier.
3. The useful next branch is compositional: apply PacketIR/runtime-consumed
   grammar and sidecar transforms to the current `c48631...0383` substrate, with
   non-no-op proof and exact CUDA only after claim.
4. Keep Kaggle for search/table production; exact promotion still requires the
   canonical exact-eval provider path and terminal claim evidence.
