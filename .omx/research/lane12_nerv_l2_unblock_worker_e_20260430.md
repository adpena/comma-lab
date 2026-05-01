# Lane 12 NeRV L2 / Alpha-Geo Unblock - Worker E - 2026-04-30

Scope: Lane 12 NeRV / Alpha-Geo clearance review only. No MCP tools, no paid
dispatch, no CUDA eval, no score promotion claim.

## Evidence Boundary

This note records dispatch readiness and missing evidence. It does not promote,
rank, kill, or retire a method family. Exact score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Reviewed

- `scripts/remote_lane_nerv.sh`
- `scripts/launch_lane_with_retry.py`
- `experiments/diagnose_nerv_geometry.py`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`
- `src/tac/tests/test_lane12_nerv_geometry_diagnostics.py`
- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/`
- `.omx/research/alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `.omx/research/alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`
- `.omx/research/lane12_alpha_geo_dispatch_readiness_20260430_codex.md`
- `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`

## Current State

The local L2 clearance packet is absent:

```text
.omx/state/lane12_nerv_l2_clearance.json
```

The current `jsonfix40` Lane 12 archive has exact local CUDA negative evidence
for the measured implementation/config only:

```text
archive = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
archive_bytes = 296478
n_samples = 600
score_recomputed_from_components = 26.03719330455429
avg_segnet_dist = 0.03528685
avg_posenet_dist = 49.7784996
gpu = NVIDIA GeForce RTX 4090
gpu_t4_match = false
```

Training provenance for `jsonfix40` confirms the retired target path:

```text
gt_masks_source = segnet
nrv_payload_bytes = 23594
final_eval_disagreement_rate = 0.012479493882921007
roundtrip_disagreement_rate = 0.012479333207011223
```

The latest local Alpha-Geo artifacts for `jsonfix40` fail exploratory geometry
gates:

```text
artifact = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json
diagnostic = alpha_geo_0_nerv_geometry
score_evidence_grade = empirical
device = cpu
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

Decoded mask custody in the same artifact:

```text
baseline_archive_sha256 = 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b
baseline_member = masks.mkv
baseline_member_sha256 = d3eeb82ce28b988476a920265751cca3d9fa2ca1364de4f33a1c7e970b7895e9
baseline_decoded_mask_sha256 = cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
candidate_archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
candidate_member = masks.nrv
candidate_member_sha256 = b3258dad7550f5e7a496f4834ed9990a3e9900a192c9045dca951e751412a4de
candidate_decoded_mask_sha256 = 5d8504ac2bb018a123fa238dbcb55615ca50278942c03bf49425df46023389b4
```

## Existing Gates

`scripts/remote_lane_nerv.sh` is already fail-closed before training unless
`L2_CLEARANCE_PATH` points to a valid packet with:

```text
lane_id in {lane_12_nerv_mask_codec, lane_12_nerv}
cleared_for_retraining_unblock = true
lane12_l2 = true
geometry_gate_passed = true
grand_council_clean_passes >= 3
evidence = non-empty string or non-empty string list
```

It also defaults to:

```text
BASE_ARCHIVE = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
GT_MASKS_SOURCE = decoded-baseline
DECODED_BASELINE_MEMBER = masks.mkv
RUN_AUTH_EVAL = 0
```

For `RUN_AUTH_EVAL=1`, it additionally requires:

```text
POSE_REGEN_PROVENANCE = existing candidate pose-regeneration provenance
ALPHA_GEO_PROVENANCE = existing Alpha-Geo JSON with pass_fail.overall_pass=true
```

The exact-eval path also checks that Alpha-Geo candidate and baseline archive
SHA-256 values match the archive built by the run and `BASE_ARCHIVE`.

`scripts/launch_lane_with_retry.py` blocks unrelated retraining scripts until
Lane 12 L2 clearance exists, but it intentionally allows Lane 12 labels and
relies on `scripts/remote_lane_nerv.sh` to fail closed. That means a Lane 12
launcher invocation can still allocate a remote instance before the remote
script exits. A pre-instance Lane 12 block would need a launcher change, which
is outside this Worker E write scope.

## Minimal L2 Clearance

Do not create `.omx/state/lane12_nerv_l2_clearance.json` from the current
`jsonfix40` artifacts. Minimal contest-grade L2 unblock should require all of
the following:

1. A real clearance packet at `.omx/state/lane12_nerv_l2_clearance.json` with
   the required boolean fields above and `grand_council_clean_passes >= 3`.
2. Evidence paths that exist locally and cite the candidate archive, baseline
   archive, Alpha-Geo JSON, deterministic archive manifest/provenance, and
   Grand Council review notes.
3. A passing Alpha-Geo diagnostic packet for the exact candidate archive:
   `diagnostic=alpha_geo_0_nerv_geometry`,
   `score_evidence_grade=empirical`, `scorer_proxy=false`, `device=cpu`,
   `pass_fail.overall_pass=true`, full `1200x384x512` mask shape, and
   candidate/baseline archive SHA-256 values matching the cited archives.
4. The Alpha-Geo packet must compare candidate `masks.nrv` to baseline
   `masks.mkv` from the decoded Lane G v3 anchor or a reviewed successor
   baseline. Current `jsonfix40` does not pass this gate.
5. Three new clean Grand Council passes after the passing Alpha-Geo packet.
   Earlier design-only clean passes do not clear the current failed geometry
   evidence.
6. For exact eval, not merely build-only candidate generation, add candidate
   pose-regeneration provenance with the candidate mask SHA and renderer SHA,
   then run the canonical CUDA auth eval on the exact archive bytes.

## Verdict

No-go for Lane 12 retraining dispatch and exact eval dispatch.

Allowed next work:

- read-only analysis,
- build-only local/unit tests,
- implement pose-regeneration provenance tooling,
- generate non-promotable Alpha-Geo diagnostics,
- harvest existing artifacts.

Do not run `scripts/remote_lane_nerv.sh` through a remote launcher for Lane 12
until the L2 packet exists. If a local build-only smoke is needed, run it only
after an explicit L2 packet exists or with a documented one-off override that
does not claim score evidence.

## Commands

Clearance absence:

```bash
test -f .omx/state/lane12_nerv_l2_clearance.json
```

Remote script syntax:

```bash
bash -n scripts/remote_lane_nerv.sh
```

Existing Alpha-Geo failure inspection:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json")
d = json.loads(p.read_text())
print(d["pass_fail"]["overall_pass"])
print(d["global"]["global_disagreement"])
print(d["boundary_bands"]["2"]["disagreement_rate"])
print(d["temporal"]["pair_transition"]["disagreement_rate"])
print(d["components"]["centroid"]["missing_component_rate"])
PY
```

Next candidate diagnostic template after a new build-only candidate exists:

```bash
.venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --baseline-member masks.mkv \
  --candidate <candidate_archive.zip> \
  --candidate-member masks.nrv \
  --output-json <evidence_dir>/alpha_geo_vs_lane_g_v3.json \
  --threshold-preset promotion
```

Exact eval template only after Alpha-Geo pass plus pose-regeneration provenance:

## Runtime Unblock Addendum - 2026-05-01

Implemented a diagnostic-only runtime unblock in
`experiments/diagnose_nerv_geometry.py`:

- bounded CPU coordinate streaming for `.nrv` mask decode, avoiding the
  multi-GB full coordinate-grid materialization;
- deterministic predecoded mask cache keyed by source/member hashes and
  requested dimensions;
- bounded global reservoir for advisory boundary-distance samples;
- explicit no-claim switches for residual-region scan skip and temporal
  component-track proxy skip while preserving full scalar mask metrics.

Focused real artifact run:

```bash
/usr/bin/time -p .venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --num-frames 1200 \
  --height 384 \
  --width 512 \
  --threshold-preset none \
  --mask-cache-dir experiments/results/lane_12_nerv_20260430_codex_jsonfix40/predecoded_mask_cache \
  --residual-region-count 0 \
  --visual-component-classes 1,2 \
  --visual-disable-temporal-tracks \
  --visual-boundary-distance-sample-cap 128 \
  --visual-boundary-distance-global-sample-cap 8192 \
  --output-json experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_pfp16_visual_primitives_bounded_20260501.json
```

Runtime: `real 161.31s`, cache hits for both baseline `masks.mkv` and
candidate `masks.nrv`. Output remains empirical CPU diagnostics only:
`promotion_eligible=false`, `score_claim_eligible=false`,
`exact_eval_claim=false`.

Observed full-scalar metrics in the bounded packet:

```text
frames = 1200
visual_frames = 1200
global_disagreement = 0.012303928799099393
pair_transition_disagreement = 0.009507171571470149
next_action = repair_or_retrain_before_exact_eval_spend
blockers = global_disagreement, boundary_2px_disagreement,
           pair_transition_disagreement, critical_missing_rate,
           critical_missing_area_rate
```

This does not create or justify an L2 clearance packet. It only makes repeated
Lane 12 vs PFP16 visual-primitives triage locally feasible without decode
memory cutoff.

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate_archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence_dir>/auth_eval_work
```
