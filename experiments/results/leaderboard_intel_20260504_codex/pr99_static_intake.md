# PR99 Public Frontier Static Intake - 2026-05-04

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/99
- title: `hnerv_muon_lc submission (0.20)`
- evidence: `external_public_archive_static_preflight_only`; no score claim from this intake
- external report recomputed score: `0.19668072586615531`
- external display/exact hint: `0.20` / `0.19667`
- archive: `experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip`
- bytes/SHA-256: `178546` / `278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb`
- release download matches archive: `True`
- strict ZIP valid: `True`

## Members

| name | bytes | compressed | method | sha256 |
|---|---:|---:|---|---|
| 0.bin | 178438 | 178438 | stored | 79b9f89e709c2892560822284a4465f17d5df8c6148b923e889683ab04bdb1d5 |

## Payload Anatomy

- `decoder_brotli`: `161883` bytes, raw `228958`, sha `52e8ebbc900955b1e9981be1b85d02c244b3197c8da710971426802e45a0bd0a`
- `scales_fp16`: `56` bytes, sha `6a07396d717befc30bd971556129d5aa383d0f0005fda7bbe949a2d1723ee019`
- `latents_brotli`: `15868` bytes, raw `33720`, sha `863634aba9a956d13eab8708133239ee861bb797cbad4ff2f39a46939987c709`
- `corrections_brotli`: `615` bytes, raw `1202`, sha `acf78b2a8c3f0a601ffd8927d5201d763c20f6eda69bd447df0d3df914d738f0`

## Runtime Risks
- External PR body/CI claim only until exact CUDA replay of these archive bytes lands in local custody.
- PR body says no GPU inflation and CI ran CPU, but runtime selects cuda when available; CUDA replay can drift from CPU CI.
- Archive is both committed in the PR and available as a GitHub release asset; committed archive, release download, and local artifact SHA match.
- Original inflate.sh can pip install brotli if missing; exact replay should stage dependency closure so inflate does not depend on network.
- Latent-correction sidecar is charged inside 0.bin, not external, but it is inference-time score-affecting side information.
- Author-linked CI run head SHA differs from the PR head SHA in API metadata, so CI success is provenance context, not exact archive custody.

## Next Commands

Active claim already exists; do not create a duplicate claim unless the current claim is terminal or stale.

```bash
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip --inflate-sh experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/leaderboard_intel_20260504_codex/pr99_exact_eval_work
```

Lightning submit shape after source-manifest staging and while the existing claim is active:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval --job-name exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z --archive /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr99_archive.zip --repo-dir /teamspace/studios/this_studio/pact --upstream-dir /teamspace/studios/this_studio/pact/upstream --inflate-sh /teamspace/studios/this_studio/pact/experiments/results/leaderboard_intel_20260504_codex/pr99_runtime/inflate.sh --machine T4 --studio lossy-compression-challenge --teamspace comma-lab --user adpena --python-bin .venv/bin/python --local-artifact-dir experiments/results/lightning_batch/exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z --source-manifest .omx/state/exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z_manifest.json --dispatch-lane-id public_pr99_hnerv_muon_lc_t4_replay --expected-archive-sha256 278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb --expected-archive-size-bytes 178546 --adjudicate --baseline-score 0.23089404465634825 --baseline-archive-bytes 178277 --predicted-band 0.18 0.24 --max-posenet-dist 0.01 --max-segnet-dist 0.01 --max-sane-score 1.0 --component-reference-label pr95_stemperm_a++ --delta-key score_delta_vs_pr95_stemperm
```
