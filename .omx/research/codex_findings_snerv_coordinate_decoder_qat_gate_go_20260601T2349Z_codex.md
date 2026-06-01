# Codex Findings: SNeRV Coordinate Decoder/QAT Gate GO

UTC: 2026-06-01T23:49:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Authority: false-authority; local continuation only; no score claim; no promotion;
no exact-eval readiness

## What Landed

Extended the SNeRV scorer-loop decoder/QAT smoke with a bounded
`top_weight_coordinate` search mode. Instead of probing one global random sign
direction, the smoke now evaluates signed perturbations on the largest-magnitude
decoder atoms and records component deltas through the same receiver-replayed
SNAR1 packet path.

Files changed:

- `src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py`
- `tools/run_snerv_scorer_loop_decoder_qat_smoke.py`
- `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py`

The acceptance rule is unchanged and strict: receiver replay must pass, PoseNet
must be no worse, SegNet must improve, and advisory score must improve.

## Coordinate Smoke Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --n-pairs 1 --levels 2 --max-trials 4 --search-mode top_weight_coordinate \
  --qat-bits 8 --step-map-bins 16 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_coordinate_smoke_20260601T2348Z.json
```

Artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_smoke_20260601T2348Z.json`
- SHA-256: `264b52d2ab5bc330832d03f48bf2e45374d6ac490f2a5329a3bf48b3172394e3`
- Search mode: `top_weight_coordinate`
- Scorer-loop evaluations: `9`
- Receiver contract: satisfied
- Accepted improvement: `true`
- Ready for pose-guard gate: `true`
- Ready for exact eval: `false`

Baseline:

- `d_seg=0.002471923828125`
- `d_pose=0.002065342850983143`
- `score=0.6145014925781653`
- `archive_bytes=335801`

Best local candidate:

- Label: `coord_040_minus`
- `d_seg=0.0024363200645893812`
- `d_pose=0.001498965546488762`
- `score=0.5896630206799555`
- `score_delta=-0.02483847189820987`
- `d_pose_delta=-0.000566377304494381`
- `d_seg_delta=-0.00003560376353561878`
- `archive_bytes=335805`

## Pose-Gate Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_coordinate_smoke_20260601T2348Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_coordinate_pose_gate_20260601T2349Z.json
```

Artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_pose_gate_20260601T2349Z.json`
- SHA-256: `73634132479ba50eb8d7912e43bcf0beed8f55b580e83e34af318270a68fd258`
- Verdict: `GO_LOCAL_CONTINUATION_ONLY`
- Accepted rows: `7`
- Blockers: `[]`
- Still false-authority:
  - `score_claim=false`
  - `promotion_eligible=false`
  - `ready_for_exact_eval_dispatch=false`

Top accepted rows:

- `coord_001_minus`: `score_delta=-0.024959690447777794`,
  `pose_delta=-0.0006186984246596694`,
  `seg_delta=-0.0000152587890625`.
- `coord_040_minus`: `score_delta=-0.02483847189820987`,
  `pose_delta=-0.000566377304494381`,
  `seg_delta=-0.00003560376353561878`.
- `coord_039_minus`: same measured component deltas as `coord_040_minus`.
- `coord_001_plus`: `score_delta=-0.024457652108826444`,
  `pose_delta=-0.0005319478223100305`,
  `seg_delta=-0.0000457763671875`.

## Verification

```text
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  tools/run_snerv_scorer_loop_decoder_qat_smoke.py
All checks passed!

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  src/tac/tests/test_snerv_pose_guarded_decoder_gate.py -q
10 passed in 0.67s
```

## Engineering Consequence

The prior random direction smoke was NO-GO because the component tradeoff split:
one direction helped PoseNet/score but hurt SegNet, while the opposite helped
SegNet but hurt PoseNet. The coordinate response mode found multiple local
directions that improve SegNet, PoseNet, and advisory score simultaneously on a
real-frame 1-pair receiver-replayed SNAR1 packet.

Next step is **bounded non-exact local continuation**, not promotion: run the
same coordinate search on a slightly broader local set, preserve per-pair
component deltas, and then convert the winning coordinate patch into a byte-closed
decoder-delta grammar or mixed-precision decoder payload before any full-600 or
contest-axis evaluation.
