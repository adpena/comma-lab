# Lane 12 / Alpha-Geo Dispatch Readiness Guardrail - 2026-04-30

Scope: Lane 12 NeRV dispatch/readiness and Alpha-Geo-0/1 mask comparison only.
No score claim and no paid compute launched.

## Evidence Boundary

This is code/test hardening and dispatch-readiness analysis. It cannot promote,
rank, kill, or retire a method family. Exact score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

## Inspection Result

Current canonical local artifacts exist:

```text
Lane G v3 base archive:
  path = experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
  bytes = 694074
  sha256 = 9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b
  members = renderer.bin, masks.mkv, optimized_poses.pt

Lane 12 jsonfix40 negative archive:
  path = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
  bytes = 296478
  sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
  exact CUDA score = 26.03719330455429
```

Dispatch was not exact-ready from canonical artifacts. The remote script still
allowed a default rerun of the retired fresh-SegNet target path and defaulted
the base archive to `submissions/robust_current/archive.zip`, not the Lane G v3
archive used by the Alpha-Geo comparisons.

## Patch

Changed `scripts/remote_lane_nerv.sh` to fail closed around the known unsafe
paths:

- Defaults `BASE_ARCHIVE` to
  `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`.
- Defaults `GT_MASKS_SOURCE=decoded-baseline`.
- Defaults `RUN_AUTH_EVAL=0`, so an accidental launch stops after deterministic
  archive build and cannot create score evidence.
- Blocks `GT_MASKS_SOURCE=segnet` unless `ALLOW_RETIRED_SEGNET_TARGET=1` is set
  for a documented forensic rerun.
- Requires `POSE_REGEN_PROVENANCE` and `ALPHA_GEO_PROVENANCE` for
  `RUN_AUTH_EVAL=1`; the stale-pose bypass has been removed.
- Requires a valid `.omx/state/lane12_nerv_l2_clearance.json` before any new
  NeRV retraining starts.

Changed `src/tac/tests/test_lane12_nerv_dependency_closure.py` to assert the
new defaults and that the exact CUDA eval path remains present but gated.

## Readiness Verdict

Training/build-only Alpha-Geo-1 candidate generation is scriptable from the
canonical Lane G v3 artifact, but exact dispatch is still blocked by missing
pose-regeneration provenance and candidate-vs-baseline geometry gate results.

Do not queue paid exact auth eval for Lane 12 until a candidate archive records:

1. decoded-baseline target custody,
2. Alpha-Geo geometry diagnostics in band or a reviewed exception,
3. regenerated pose provenance against the candidate mask stream, and
4. deterministic archive manifest and payload closure.

## Next Commands

Local focused verification:

```bash
.venv/bin/python -m py_compile src/tac/tests/test_lane12_nerv_dependency_closure.py
bash -n scripts/remote_lane_nerv.sh
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py -q
git diff --check
```

Exact dispatch command: none is ready yet.

Training/build-only remote command template after explicit approval for paid
candidate generation:

```bash
.venv/bin/python scripts/launch_lane_with_retry.py \
  --lane-script scripts/remote_lane_nerv.sh \
  --label lane_12_nerv_alpha_geo1_decoded_baseline_20260430 \
  --max-dph 0.30 \
  --predicted-band 0.95 1.30 \
  --estimated-cost 1.00 \
  --max-retries 3
```

Exact eval remains a separate reviewed step and must set `RUN_AUTH_EVAL=1` only
after pose-regeneration provenance exists.

## Worker D Delta

Additional review on 2026-04-30 found one remaining build-only hardening gap:
the Lane 12 archive rebuild copied every non-mask member from `BASE_ARCHIVE`.
That could propagate duplicate members, hidden/system sidecars, traversal
paths, or stale debug payloads before exact-eval validation. The rebuild block
now fails closed on unsafe/unexpected base members, preserves only
`renderer.bin` plus exactly one optimized-pose artifact, writes deterministic
permissions/timestamps, and validates the rebuilt `masks.nrv` archive with
`tac.submission_archive.validate_archive`.

Focused verification passed:

```bash
bash -n scripts/remote_lane_nerv.sh
.venv/bin/python -m py_compile src/tac/tests/test_lane12_nerv_dependency_closure.py
.venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py -q
git diff --check -- scripts/remote_lane_nerv.sh src/tac/tests/test_lane12_nerv_dependency_closure.py
```

Result: `11 passed in 1.36s`. A local fail-closed smoke exited at the missing
L2 clearance packet before heartbeat, NVDEC probe, editable install, training,
or exact eval.

Fastest admissible next step remains design/custody, not dispatch: produce a
real `.omx/state/lane12_nerv_l2_clearance.json` only after a passing
Alpha-Geo packet and three clean Grand Council passes exist. Until then, do not
run `scripts/remote_lane_nerv.sh` or `scripts/launch_lane_with_retry.py` for
Lane 12 retraining or exact eval.

## Worker C Delta

Review date: 2026-04-30. Scope was read-only review plus this readiness ledger.
No MCP tools, CUDA eval, paid dispatch, or score promotion were used.

Reviewed surfaces:

- `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/`
- `experiments/diagnose_nerv_geometry.py`
- `scripts/remote_lane_nerv.sh`
- `src/tac/tests/test_lane12_nerv_dependency_closure.py`
- Lane 12 / Alpha progress docs, including
  `alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`,
  `lane12_alpha_geo_worker3_20260430.md`,
  `alpha_pose_preserving_redesign_spec_20260430_codex.md`, and current
  Shannon-floor readiness progress.

Current jsonfix40 facts remain unchanged:

```text
archive = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
archive_bytes = 296478
members = optimized_poses.bin, renderer.bin, masks.nrv
gt_masks_source = segnet
nrv_payload_bytes = 23594
score_recomputed_from_components = 26.03719330455429
avg_segnet_dist = 0.03528685
avg_posenet_dist = 49.7784996
n_samples = 600
```

The reviewed code now supports a safer Alpha-Geo-1 direction in principle:
`train_nerv_mask.py` has decoded-baseline target custody covered by
`test_lane12_nerv_dependency_closure.py`, and `diagnose_nerv_geometry.py`
records source archive/member SHA-256 plus decoded tensor SHA-256. However, the
current dispatch launcher intentionally blocks new NeRV retraining before a
valid L2 clearance packet exists.

Local non-CUDA commands run:

```bash
bash -n scripts/remote_lane_nerv.sh
```

Result: passed.

```bash
test -f .omx/state/lane12_nerv_l2_clearance.json
```

Result: exit code `1`; the local L2 clearance packet is absent.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/diagnose_nerv_geometry.py \
  --baseline experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
  --baseline-member masks.mkv \
  --candidate experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip \
  --candidate-member masks.nrv \
  --output-json /tmp/lane12_alpha_geo_0_vs_lane_g_v3_worker_c_20260430.json \
  --threshold-preset exploratory
```

Result: exit code `2`, expected for a failed Alpha-Geo exploratory gate. This
was CPU tensor evidence only and wrote outside the repo. Output SHA-256:

```text
ec57211cab49de1cb44dd6282916426b0aaf088b653790995f3ef6217f665ffa
```

Key local diagnostic results:

```text
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
overall_pass = false
baseline decoded mask sha256 = cce3a986341c40df9b9ebca24ff96e16c4b41b40b388dc2af86161ba76e2b4e9
candidate decoded mask sha256 = 5d8504ac2bb018a123fa238dbcb55615ca50278942c03bf49425df46023389b4
```

Readiness determination:

- A local Alpha-Geo-0 diagnostic against the existing jsonfix40 archive is safe
  and was rerun. It rejects the existing jsonfix40 candidate again.
- No new local or remote build-only Lane 12 candidate should be run now. The
  launcher requires `.omx/state/lane12_nerv_l2_clearance.json` before NeRV
  retraining, and that packet is absent locally.
- Exact eval is still blocked by missing candidate pose-regeneration
  provenance and lack of a passing Alpha-Geo packet for a decoded-baseline
  candidate.
- jsonfix40 remains scoped negative evidence for that measured
  implementation/config only. It does not kill Alpha, NeRV, INR, or mask
  compression.

Files changed by Worker C:

- `.omx/research/lane12_alpha_geo_dispatch_readiness_20260430_codex.md`

## Worker D L2 Packet Review

Review date: 2026-04-30. Scope was Lane 12/Alpha L2 unblock packet review
only. No MCP tools, CUDA eval, paid dispatch, archive rebuild, or score
promotion were used.

Reviewed surfaces:

- Lane 12 / Alpha research docs:
  `council_lane_12_nerv_design_20260430.md`,
  `council_lane_12_nerv_round{1,2,3}_20260430.md`,
  `alpha_geo_diagnostics_lane12_readiness_20260430_codex.md`,
  `lane12_alpha_geo_worker3_20260430.md`, and
  `alpha_pose_preserving_redesign_spec_20260430_codex.md`
- `experiments/diagnose_nerv_geometry.py`
- `experiments/train_nerv_mask.py`
- `scripts/remote_lane_nerv.sh`
- latest local jsonfix40 geometry diagnostic:
  `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json`

Exact launcher-enforced L2 clearance packet criteria:

```text
path = .omx/state/lane12_nerv_l2_clearance.json
JSON type = object
lane_id in {"lane_12_nerv_mask_codec", "lane_12_nerv"}
cleared_for_retraining_unblock = true
lane12_l2 = true
geometry_gate_passed = true
grand_council_clean_passes = integer >= 3
evidence = non-empty string or non-empty string list
```

The packet should not be written from syntax alone. The cited evidence must
include a passing Alpha-Geo geometry artifact for the candidate to be retrained
or advanced, plus the new clean Grand Council passes after the Round 3 reset.
For a geometry artifact to justify `geometry_gate_passed=true`, it should be
from `experiments/diagnose_nerv_geometry.py`, compare the candidate
`masks.nrv` stream against the canonical Lane G v3 decoded `masks.mkv` stream,
record source archive SHA-256 and decoded-mask SHA-256 for both sides, use full
1200 x 384 x 512 x 5 geometry, and have `pass_fail.overall_pass=true`.
Alpha-Geo evidence remains empirical CPU tensor evidence; it can unblock the
next build-only candidate generation, but it is not score evidence.

Current jsonfix40 diagnostic cannot support L2 clearance:

```text
diagnostic_json = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json
diagnostic_sha256 = 95549ace15e86512bb425ca5ec3983e4023fb384a7683abf6901fcd177cdff12
overall_pass = false
global_disagreement = 0.012303928799099393        # gate 0.003 exploratory
boundary_2px_disagreement = 0.14883144511692872   # gate 0.005 exploratory
pair_transition_disagreement = 0.009507171571470149 # gate 0.004 exploratory
missing_component_rate = 0.4611606740560512       # gate 0.0 exploratory
max_component_centroid_jump_px = 289.6654980546722 # gate 1.0 exploratory
```

The exact CUDA `jsonfix40` archive remains scoped negative evidence for the
measured implementation/config only:

```text
archive = experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip
archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
archive_bytes = 296478
gt_masks_source = segnet
n_samples = 600
device = cuda
gpu = NVIDIA GeForce RTX 4090
score_recomputed_from_components = 26.03719330455429
avg_segnet_dist = 0.03528685
avg_posenet_dist = 49.7784996
```

Minimal next build-only experiment:

- Not runnable now: local `.omx/state/lane12_nerv_l2_clearance.json` is absent,
  and `scripts/remote_lane_nerv.sh` checks it before heartbeat, NVDEC probe,
  editable install, training, archive build, or exact eval.
- The AGENTS.md allowance for build-only lanes does not override the Lane 12
  retraining guard. The current Lane 12 "build-only" launcher still invokes
  `experiments/train_nerv_mask.py --device cuda`, so it is a new retraining
  lane and must remain blocked until the L2 packet exists and passes.
- Once a valid L2 packet exists, the minimal admissible Lane 12 command is the
  decoded-baseline, no-auth-eval build-only run below. It generates a candidate
  archive only; it does not create score evidence.

```bash
WORKSPACE=/workspace/pact \
BASE_ARCHIVE=/workspace/pact/experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
GT_MASKS_SOURCE=decoded-baseline \
DECODED_BASELINE_PATH=/workspace/pact/experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip \
DECODED_BASELINE_MEMBER=masks.mkv \
RUN_AUTH_EVAL=0 \
L2_CLEARANCE_PATH=/workspace/pact/.omx/state/lane12_nerv_l2_clearance.json \
LOG_DIR=/workspace/pact/lane_12_nerv_alpha_geo1_decoded_baseline_results \
bash /workspace/pact/scripts/remote_lane_nerv.sh
```

Immediate allowed non-paid commands are diagnostics and verification only:

```bash
bash -n scripts/remote_lane_nerv.sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/diagnose_nerv_geometry.py
test -f .omx/state/lane12_nerv_l2_clearance.json
jq -r '.pass_fail.overall_pass' \
  experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo_0_vs_lane_g_v3_worker3_20260430.json
```

Blockers:

- `.omx/state/lane12_nerv_l2_clearance.json` is absent locally.
- Latest jsonfix40 Alpha-Geo diagnostic fails exploratory geometry gates and
  cannot be cited as `geometry_gate_passed=true`.
- Round 3 reset means old Lane 12 council passes are insufficient; the packet
  needs three clean passes after the reset.
- No passing decoded-baseline candidate Alpha-Geo packet exists.
- No candidate pose-regeneration provenance exists; this blocks exact eval even
  after any future build-only candidate is produced.
