# Renderer Group Allocator Worker - 2026-05-03

Scope: local-only renderer tensor/group allocator work for the C-089 frontier.
No Lightning, Modal, Vast.ai, T4, L40S, or other remote GPU dispatch was
performed, and no active dispatch claims were edited.

## Objective

Current A++ frontier anchor:

- archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- score: `0.3154707273953505`
- bytes: `276342`
- SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`

At unchanged components, sub-`0.314` requires archive bytes at or below
`274133`, i.e. at least `2209` bytes saved.

## Code Landed

- Added `experiments/build_renderer_group_allocator_candidates.py`.
  - Extracts the PR75 single-member payload into logical runtime members.
  - Preserves every non-renderer member byte-for-byte.
  - Decodes the QZS3 JointFrameGenerator state.
  - Generates deterministic low-risk frame2-only MQZ1 block-allocation
    policies and optional CLI policies.
  - Records per-tensor raw byte deltas, output archive bytes, SHA-256,
    runtime unpack summaries, source-evidence guard status, and local
    pose-safety preflight status.
  - Fails closed on known A-negative parent archives, non-frontier source
    evidence, no-op policies, high-risk prefixes unless explicitly allowed,
    and candidates that do not pass runtime unpacking.
- Added `src/tac/tests/test_build_renderer_group_allocator_candidates.py`.
- Hardened `submissions/robust_current/unpack_renderer_payload.py` so
  self-describing PR75/P3-P6 payload validation accepts renderer formats that
  the inflate runtime already supports: `QZS3`, `MQZ1`, `QBF1`, `QFAI`,
  pickle, and zipped renderer payloads. The prior guard accepted only `QZS3`,
  which made charged MQZ1 renderer allocation candidates fail before
  pose-safety.

## Local Search

Command:

```bash
.venv/bin/python experiments/build_renderer_group_allocator_candidates.py \
  --output-dir experiments/results/renderer_group_allocator_worker_20260503 \
  --force \
  --max-generated-policies 24 \
  --max-build-candidates 24 \
  --max-preflight-candidates 24 \
  --preflight-max-pairs 5
```

Artifacts:

- `experiments/results/renderer_group_allocator_worker_20260503/summary.json`
- `experiments/results/renderer_group_allocator_worker_20260503/dispatch_recommendation.json`
- `experiments/results/renderer_group_allocator_worker_20260503/tensor_group_profile.json`
- Per-candidate `archive.zip`, `build_manifest.json`, and
  `pose_safety_preflight.json`.

Preflight sampled pairs: `[0, 150, 300, 449, 599]`.
Thresholds: mean abs `<= 3.0`, RMS `<= 8.0`, max abs `<= 80.0`.

## Results

The allocator built and locally preflighted `21` byte-closed MQZ1 candidates.
No candidate is exact-eval-ready.

Best byte-saving candidate:

- `group_frame2_head_b128`
- bytes: `276011`
- SHA-256:
  `19abbe2ba353dee0888c379150392f189d958ccd32faf922e3f793eaed5fa569`
- delta vs C-089: `-331` bytes
- byte-only score projection: `0.31525032808186704`
- local pose-safety: failed, max abs delta `170.275390625`

Safe local candidates:

| candidate | bytes | delta vs C-089 | local pose-safety | note |
| --- | ---: | ---: | --- | --- |
| `group_frame2_head.pre_b64` | `276575` | `+233` | pass | byte-regressive |
| `top1_lowrisk_tensor_raw_gain` | `276630` | `+288` | pass | byte-regressive |

Top byte-saving candidates all failed local pose-safety on max absolute output
delta despite acceptable mean/RMS deltas. This says frame2-only block
coarsening has a sharp local-error tail: the average perturbation is small,
but isolated pixel/region excursions are large enough to fail the dispatch
gate and are likely PoseNet-risky.

Top rows:

| candidate | bytes | delta | safe | max abs | mean abs |
| --- | ---: | ---: | --- | ---: | ---: |
| `group_frame2_head_b128` | `276011` | `-331` | false | `170.275390625` | `2.0189833641052246` |
| `top12_lowrisk_tensor_raw_gain` | `276102` | `-240` | false | `183.58157348632812` | `2.0477283000946045` |
| `top8_lowrisk_tensor_raw_gain` | `276170` | `-172` | false | `161.53759765625` | `1.8335872888565063` |
| `group_frame2_head_b96` | `276186` | `-156` | false | `188.18797302246094` | `2.0373613834381104` |
| `group_frame2_head_b64` | `276210` | `-132` | false | `137.8223419189453` | `1.6323579549789429` |

## Decision

Do not dispatch any renderer group allocator candidate from this pass.

Reason: the byte-saving candidates are two orders of magnitude short of the
`2209`-byte unchanged-component target and fail local pose-safety; the only
local-safe candidates are byte-regressive. Exact CUDA eval would be low-EV
unless used strictly as a runtime-custody diagnostic, not as a sub-`0.314`
attempt.

## Next Engineering Implication

The useful result is structural: MQZ1 per-group block allocation is now
available and guarded, but the current low-risk frame2-only search cannot
bridge the score gap. The renderer path still needs a semantic/self-compressed
renderer or learned export from the active fixed-renderer burns, not a
hand-built block-size allocator alone.

High-EV follow-up if continuing this lane:

1. Apply the allocator to harvested trained renderer exports after raw
   transplant and pose-safety preflight.
2. Add a tail-aware allocator objective that penalizes max-abs/local cliff
   risk, not just mean/RMS or raw bytes.
3. Use exact component-response or full-frame local diffs to learn a
   per-tensor trust region before trying larger shared/frame1 groups.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile \
  experiments/build_renderer_group_allocator_candidates.py \
  src/tac/tests/test_build_renderer_group_allocator_candidates.py \
  submissions/robust_current/unpack_renderer_payload.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_renderer_group_allocator_candidates.py \
  src/tac/tests/test_unpack_renderer_payload_fixedslice.py \
  -q
```

Result: `12 passed in 2.86s`.

