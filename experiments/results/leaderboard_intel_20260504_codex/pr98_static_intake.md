# PR98 Public Frontier Static Intake - 2026-05-04

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/98
- title: `hnerv_muon_finetuned_from_pr95 (0.1963)`
- evidence: `external_public_archive_static_preflight_only`; no score claim from this intake
- external report recomputed score: `0.19625777542725248`
- external display/exact hint: `0.20` / `0.1963`
- archive: `experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip`
- bytes/SHA-256: `178392` / `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`
- release download matches archive: `True`
- strict ZIP valid: `True`

## Members

| name | bytes | compressed | method | sha256 |
|---|---:|---:|---|---|
| 0.bin | 178284 | 178284 | stored | fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f |

## Payload Anatomy

- `meta_brotli`: `80` bytes, raw `80`, sha `49385dd99c02228dd0ed7649586fe5026940573a5675911ee7cb24ab2e8bdca7`
- `decoder_brotli`: `162343` bytes, raw `229022`, sha `825485793ee85c7c419130fdb1b6c12a2d44d95f7acc67a97778a07f921229c7`
- `latents_brotli`: `15849` bytes, raw `33720`, sha `a1841a326b5a0cb886df27de318002cf42badea0446589c8c9ef6e45d16e4392`

## Runtime Risks
- External PR body claim only until exact CUDA replay of these archive bytes lands in local custody.
- Requires GPU for inflation per PR body; runtime chooses cuda when available, so T4 exact replay is required.
- Archive is a GitHub release asset, not a committed PR file; release API custody and SHA match the downloaded local artifact.
- Raw PR inflate.sh assumes submissions/<name> package layout; pr98_runtime/inflate.sh is an adapted local wrapper with payload code unchanged.
- Runtime depends on brotli, numpy, and torch; dependency closure must be staged for Lightning/T4.
- Postprocess subtracts one RGB count from selected channels after upsample; cross-device rounding drift remains possible until exact replay.

## Next Commands

Active claim already exists; do not create a duplicate claim unless the current claim is terminal or stale.

```bash
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip --inflate-sh experiments/results/leaderboard_intel_20260504_codex/pr98_runtime/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/leaderboard_intel_20260504_codex/pr98_exact_eval_work
```

Lightning submit shape after source-manifest staging and while the existing claim is active:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z --archive /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr98_archive.zip --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --inflate-sh /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr98_runtime/inflate.sh --machine T4 --studio lossy-compression-challenge --teamspace comma-lab --user adpena --python-bin .venv/bin/python --local-artifact-dir experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z --source-manifest .omx/state/exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z_manifest.json --dispatch-lane-id public_pr98_hnerv_muon_finetuned_t4_replay --expected-archive-sha256 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb --expected-archive-size-bytes 178392 --adjudicate --baseline-score 0.23089404465634825 --baseline-archive-bytes 178277 --predicted-band 0.18 0.24 --max-posenet-dist 0.01 --max-segnet-dist 0.01 --max-sane-score 1.0 --component-reference-label pr95_stemperm_a++ --delta-key score_delta_vs_pr95_stemperm
```
