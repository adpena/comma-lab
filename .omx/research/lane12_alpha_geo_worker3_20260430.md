# Lane 12 Alpha-Geo Worker 3 - 2026-04-30

Scope: Lane 12 NeRV / Alpha-Geo-0 local diagnostics and remote fail-closed
guards. No score claim, no CUDA auth eval, no retraining dispatch.

## Evidence Boundary

Alpha-Geo-0 is CPU tensor geometry evidence only. It can reject a candidate
before score spend, but cannot promote, rank, kill, or retire a method family.
Exact score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Local Alpha-Geo-0 Run

Artifacts existed and were used:

```text
baseline = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
baseline_member = masks.mkv
baseline_sha256 = 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b

candidate = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
candidate_member = masks.nrv
candidate_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
```

Command:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --baseline-member masks.mkv \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --output-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json \
  --threshold-preset exploratory
```

Result:

```text
exit_code = 2
output_json = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json
output_json_sha256 = 95549ace15e86512bb425ca5ec3983e4023fb384a7683abf6901fcd177cdff12
overall_pass = false
global_disagreement = 0.012303928799099393
boundary_1px_disagreement = 0.2086177911086304
boundary_2px_disagreement = 0.14883144511692872
boundary_3px_disagreement = 0.11633853036183021
boundary_5px_disagreement = 0.08223161952370056
pair_transition_disagreement = 0.009507171571470149
pair_transition_f1 = 0.095099661402374
stable_false_flip_rate = 0.0013034438031416468
missing_component_rate = 0.4611606740560512
max_matched_centroid_jump_px = 289.6654980546722
```

Custody hashes recorded by the diagnostic:

```text
baseline masks.mkv sha256 = d3eeb82ce28b988476a920265751cca3d9fa2ca1364de4f33a1c7e970b7895e9
candidate masks.nrv sha256 = b3258dad7550f5e7a496f4834ed9990a3e9900a192c9045dca951e751412a4de
baseline decoded mask sha256 = cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
candidate decoded mask sha256 = 5d8504ac2bb018a123fa238dbcb55615ca50278942c03bf49425df46023389b4
```

Verdict: jsonfix40 fails Alpha-Geo-0 exploratory geometry gates against Lane G
v3/base masks. This remains empirical diagnostic evidence only.

## Fail-Closed Remote Guard

`scripts/remote_lane_nerv.sh` now fails closed before retraining unless
`.omx/state/lane12_nerv_l2_clearance.json` is a valid JSON object with:

```text
lane_id in {lane_12_nerv_mask_codec, lane_12_nerv}
cleared_for_retraining_unblock = true
lane12_l2 = true
geometry_gate_passed = true
grand_council_clean_passes >= 3
evidence = non-empty string or non-empty string list
```

The script still defaults to:

```text
BASE_ARCHIVE = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
GT_MASKS_SOURCE = decoded-baseline
DECODED_BASELINE_MEMBER = masks.mkv
RUN_AUTH_EVAL = 0
```

For `RUN_AUTH_EVAL=1`, the script now requires both:

```text
POSE_REGEN_PROVENANCE = existing candidate pose-regeneration provenance file
ALPHA_GEO_PROVENANCE = existing Alpha-Geo JSON with pass_fail.overall_pass=true
```

Before exact eval, it also checks that the Alpha-Geo JSON candidate archive
SHA matches the archive just built by the run, the baseline SHA matches
`BASE_ARCHIVE`, and the resolved members are `masks.nrv` and `masks.mkv`.

Local fail-closed smoke:

```text
WORKSPACE=/Users/adpena/Projects/pact
LOG_DIR=<temp>
command = bash scripts/remote_lane_nerv.sh
exit_code = 1
first_fatal = missing Lane 12 L2 clearance packet
training_started = false
nvdec_probe_started = false
editable_install_started = false
```

## Blockers

- `.omx/state/lane12_nerv_l2_clearance.json` is absent locally, so new NeRV
  retraining should not be dispatched.
- jsonfix40 fails Alpha-Geo-0 geometry gates and must not be exact-eval
  promoted.
- No candidate pose-regeneration provenance exists for a mask-changing Lane12
  archive.
- No passing Alpha-Geo geometry packet exists for a new decoded-baseline
  candidate archive.

## Exact Next Remote Command

Do not run this until the L2 clearance packet exists and passes the guard. With
that packet in place, the next remote command is build-only by default and will
not run exact eval:

```bash
GT_MASKS_SOURCE=decoded-baseline \
DECODED_BASELINE_PATH=/workspace/pact/experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
DECODED_BASELINE_MEMBER=masks.mkv \
RUN_AUTH_EVAL=0 \
L2_CLEARANCE_PATH=/workspace/pact/.omx/state/lane12_nerv_l2_clearance.json \
LOG_DIR=/workspace/pact/lane_12_nerv_alpha_geo1_decoded_baseline_results \
bash /workspace/pact/scripts/remote_lane_nerv.sh
```

After that build-only run, run Alpha-Geo against the produced archive and only
consider exact eval if geometry passes and candidate pose-regeneration
provenance exists.
