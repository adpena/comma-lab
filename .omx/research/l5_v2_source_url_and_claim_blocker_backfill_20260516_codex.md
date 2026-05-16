# L5-v2 Source URL And Claim-Blocker Backfill - 2026-05-16

## Context

The L5-v2 / Time-Traveler ledgers cited paper families and public PR lineage by
name, but several paper-facing surfaces lacked direct source URLs/DOIs and
explicit claim blockers. That made the docs weaker as OSS/paper provenance and
increased the risk that planning priors could be quoted as empirical results.

## Fix

Backfilled primary source URLs and claim-scope constraints into:

- `.omx/research/l5_v2_latest_neural_video_codec_source_basis_20260516_codex.md`
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
- `.omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md`

The backfill covers:

- official comma challenge repository;
- public frontier PRs #95, #100, #101, #103, and #106;
- HNeRV CVF/arXiv;
- DCVC-RT arXiv and Microsoft DCVC repository;
- TeCoNeRV arXiv and project page;
- Atick-Redlich, Rao-Ballard, Friston, Slepian-Wolf, and Wyner-Ziv source IDs.

## Claim Discipline

The edited docs now explicitly block:

- Time-Traveler score bands from public/paper empirical wording without paired
  CPU/CUDA exact artifacts;
- byte budgets from being treated as measured archive bytes without an exact
  section manifest, archive SHA, and runtime tree/content SHA;
- DCVC-RT/TeCoNeRV/HNeRV sources from authorizing contest score claims without
  byte-closed archive/runtime integration and scorer-aware training;
- provider cost rows from being treated as live cost evidence without a fresh
  provider-rate snapshot at dispatch time.

## Verification

This was a documentation/provenance-only patch. Source URLs were refreshed
online on 2026-05-16 before editing.
