# Alpha-Geo Diagnostics And Lane 12 Readiness - 2026-04-30

Scope: Alpha-Geo CPU geometry diagnostics, Lane 12 `jsonfix40` artifacts, and
bounded reproducibility hardening. No score promotion claim.

## Evidence Boundary

Alpha-Geo diagnostics remain empirical CPU tensor checks. They can reject an
Alpha candidate before CUDA spend, but they cannot promote, rank, kill a method
family, or supersede exact CUDA auth eval.

Score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Reviewed Inputs

- `experiments/diagnose_nerv_geometry.py`
- `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_geo_1_visual_primitives_design_20260430_agent.md`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/nerv_provenance.json`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3.json`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_a_base.json`

## Lane 12 `jsonfix40` Evidence

Exact CUDA result is an A-grade local CUDA negative for the measured
implementation/config, not an Alpha/NeRV family kill:

```text
archive = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
archive_bytes = 296478
device = cuda via provenance
gpu = NVIDIA GeForce RTX 4090
n_samples = 600
score_recomputed_from_components = 26.03719330455429
avg_segnet_dist = 0.03528685
avg_posenet_dist = 49.7784996
```

Training provenance confirms the current config targeted fresh SegNet argmax
masks:

```text
gt_masks_source = segnet
nrv_payload_bytes = 23594
final_eval_disagreement_rate = 0.012479493882921007
roundtrip_disagreement_rate = 0.012479333207011223
```

That matches the Alpha redesign hypothesis: rate improved, but geometry did
not preserve the decoded baseline mask distribution consumed by the existing
renderer and poses.

## New Diagnostic Artifact

I reran the upgraded diagnostic against Lane G v3 without overwriting older
JSON:

```text
experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_2px_hashes.json
sha256 = 72dbb6e98c5ac9fa598b84d44381e72d9c4ae8e2c7e504907281f038f99c3b45
```

Command:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --output-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_2px_hashes.json \
  --threshold-preset exploratory
```

Expected exit code: `2`, because exploratory gates fail.

Key results:

```text
global_disagreement = 0.012303928799099393
boundary_1px_disagreement = 0.2086177911086304
boundary_2px_disagreement = 0.14883144511692872
boundary_3px_disagreement = 0.11633853036183021
boundary_5px_disagreement = 0.08223161952370056
lane_marking_recall = 0.2115568938212039
vehicle_undrivable_recall = 0.9950934805331972
pair_transition_disagreement = 0.009507171571470149
missing_component_rate = 0.4611606740560512
max_matched_centroid_jump_px = 289.6654980546722
overall_pass = false
```

New custody fields record:

```text
baseline archive sha256 = 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b
baseline masks.mkv sha256 = d3eeb82ce28b988476a920265751cca3d9fa2ca1364de4f33a1c7e970b7895e9
baseline decoded mask sha256 = cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
candidate archive sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
candidate masks.nrv sha256 = b3258dad7550f5e7a496f4834ed9990a3e9900a192c9045dca951e751412a4de
candidate decoded mask sha256 = 5d8504ac2bb018a123fa238dbcb55615ca50278942c03bf49425df46023389b4
```

## Code/Test Hardening Landed

- Added default 2px boundary-band diagnostics and gates to match the Alpha spec
  and Alpha-Geo-1 design packet.
- Added CLI output custody metadata for source files, ZIP members, resolved
  member names, member SHA-256, decoded-mask SHA-256, diagnostic config, and
  command argv.
- Added ZIP duplicate-member rejection before diagnostic extraction.
- Added focused tests for the 2px gate, CLI custody hashes, nested ZIP member
  metadata, and duplicate ZIP rejection.

## Verification

```bash
.venv/bin/python -m py_compile experiments/diagnose_nerv_geometry.py src/tac/tests/test_lane12_nerv_geometry_diagnostics.py
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_geometry_diagnostics.py -q
```

Result:

```text
10 passed in 1.12s
```

## Readiness And Blockers

Alpha diagnostics are more reproducible for CUDA/L2 gate decisions now, but
Lane 12 Alpha-Geo-1 is still blocked from promotion-grade dispatch by missing
implementation paths:

1. `experiments/train_nerv_mask.py` still lacks a decoded-baseline `masks.mkv`
   target mode. Current production target mode is `gt_masks_source=segnet`.
2. No pose regeneration path is wired to accept a decoded candidate mask stream
   and record candidate mask SHA plus renderer SHA.
3. Renderer embedding L2 drift diagnostic is still missing.
4. `jsonfix40` evidence is exact local CUDA on RTX 4090, not T4-equivalent
   A++ evidence. This is enough to retire this measured implementation/config,
   but not to make broader family claims.

Next candidate gate command template:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline <baseline_archive.zip> \
  --baseline-member masks.mkv \
  --candidate <alpha_geo_1_candidate_archive.zip> \
  --candidate-member masks.nrv \
  --output-json <evidence_dir>/alpha_geo_1_vs_baseline_geometry.json \
  --threshold-preset promotion
```

Only if that passes or has a reviewed exception should CUDA auth eval spend be
queued:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <alpha_geo_1_candidate_archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence_dir>/auth_eval_work
```

Changed files from this turn:

- `experiments/diagnose_nerv_geometry.py`
- `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_codex_2px_hashes.json`
- `.omx/research/alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`
