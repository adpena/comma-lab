# Q-FAITHFUL Snapshot Exact-Screen Contract Guard - 2026-05-02

## Scope

This note records the hardening for the Q-FAITHFUL H100 snapshot-wave bug
class where half-frame mask archives reached exact-screen preparation without
charged zoom/warp geometry and without explicit export/runtime contract
metadata. The observed failure mode is catastrophic PoseNet collapse when an
inflate path expands half-frame masks through a degraded duplicate/identity path
instead of the intended packed geometry.

Evidence grade: engineering guardrail. No retraining, cloud launch, or score
claim was performed.

## Permanent Guard

`scripts/q_faithful_snapshot_loop.py` now emits a
`qfaithful_snapshot_runtime_contract_v2` screen contract in each snapshot
manifest and mirrors the key fields in QFAI export metadata. The contract
records:

- mask frame contract: operator-declared `full` / `half`, or `auto` via
  `ffprobe` frame-count evidence.
- required export metadata keys for snapshot runtime custody.
- zoom/warp geometry presence, bytes, SHA-256 when supplied.
- whether the current repack runtime contract preserves that geometry.
- `promotable_exact_screen=false` plus deterministic non-promotable reasons
  when the contract is incomplete.

`--eval-mode run` now fails closed if the contract is non-promotable. Command
or dry-run modes may still write a manifest, but the manifest labels the screen
non-promotable and withholds `command_ready`.

Full-frame archives preserve legacy behavior: an explicit or detected
1200-frame mask contract is promotable without zoom/warp geometry.

## Current Half-Frame Status

The current QZS repack path preserves only `renderer.bin`, `masks.mkv`, and
`optimized_poses.bin`. Therefore a half-frame Q-FAITHFUL snapshot remains
non-promotable even if a local `zoom_scalars.bin` is supplied, because the
geometry is not yet charged and preserved in the repacked exact-screen archive.

Reactivation criterion: update the archive builder/repacker/runtime contract so
zoom/warp geometry is an allowed charged member, survives single-blob and
multi-member repacks, and is validated by inflate-side metadata before exact
CUDA screening.

## Local Verification

Commands run:

```bash
.venv/bin/python -m py_compile scripts/q_faithful_snapshot_loop.py src/tac/tests/test_q_faithful_snapshot_loop.py
.venv/bin/python -m pytest src/tac/tests/test_q_faithful_snapshot_loop.py -q
```

Result: `9 passed in 0.49s`.

## Geometry-Closed Candidate Byte Screen - 2026-05-02

Follow-up candidate conversion was performed under
`experiments/results/qfaithful_geometry_closed_candidates_20260502/` without
touching the public trace or active H100 component-trace artifacts.

Source snapshot: `postprocess_fixed_snapshot_20260501T2146Z_fix1`, using the
prior exact-screened no-zoom `qzs3_pr64_qp1` payload as the controlled base.
The prior L40S diagnostic archive was
`a34f493b77e3a2ccba7e059134127e9b3cb6e774a41862143d369fa3f5fc81af`
at 273,103 bytes; its `p` member was preserved byte-for-byte in the primary
geometry-closed candidate.

Primary candidate:

- Archive:
  `experiments/results/qfaithful_geometry_closed_candidates_20260502/2146_qzs3_pr64_qp1_direct_zoom_v2/archive.zip`
- SHA-256:
  `f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61`
- Bytes: 274,257
- Member order: `p`, `zoom_scalars.bin`
- Charged geometry: `zoom_scalars_v2.bin`, 1,200 bytes,
  SHA-256 `5eaa0d0ecd53568134e4fe0d20874a515c7cf96e82c2c7e0e71bdbf19fd950bb`
- Byte delta versus prior no-zoom archive: +1,154 bytes
- Evidence grade: empirical byte screen only; no score claim.

Secondary controlled candidate:

- Archive:
  `experiments/results/qfaithful_geometry_closed_candidates_20260502/2146_qzs3_pr64_qp1_direct_zoom_v1/archive.zip`
- SHA-256:
  `8a3d4e8348f9e356dbf614d69f5fe122fdcd6dfc50160671b734b2d38a01eb75`
- Bytes: 274,325
- Charged geometry: `zoom_scalars_v1.bin`, 1,200 bytes,
  SHA-256 `accf5b88c811d6a79707ab19b1314646bb26aa4977f77eaaf4cbfad9ea1dba76`
- Evidence grade: empirical byte screen only; no score claim.

Additional repack-built variants were produced for comparison:

- `2146_qzs3_pr64_qp1_zoom_v2/archive.zip`,
  SHA-256 `89a7ba8b9e05a09d2a28bfee1b07e52f1af2235677c240dbde3b7269403c175f`,
  274,145 bytes.
- `2146_qzs3_pr64_qp1_zoom_v1/archive.zip`,
  SHA-256 `2d45289d0bde42f046bceaa0542fdbf5cd78f8f3f54aff101b663eacbca96ec5`,
  274,213 bytes.

The repack variants are deterministic and preserve charged geometry, but their
`p` payload bytes differ from the prior no-zoom exact snapshot. For the next
CUDA screen, prefer the direct `zoom_v2` candidate because it isolates the
transform to the charged geometry member.

Local byte-screen artifacts:

- `candidate_byte_screen_summary.json`
- `2146_qzs3_pr64_qp1_direct_zoom_v2/byte_screen.json`
- `2146_qzs3_pr64_qp1_direct_zoom_v2/build_provenance.json`
- `2146_qzs3_pr64_qp1_direct_zoom_v2/unpack_smoke_summary.json`

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_q_faithful_snapshot_loop.py -q
```

Result: `11 passed in 0.42s`.

Next exact CUDA diagnostic command, after claiming the lane and staging the
archive on an H100/L40S CUDA host:

```bash
WORKSPACE=/workspace/pact \
ARCHIVE_PATH=/workspace/pact/experiments/results/qfaithful_geometry_closed_candidates_20260502/2146_qzs3_pr64_qp1_direct_zoom_v2/archive.zip \
ARCHIVE_LABEL=qfaithful_2146_pr64_qp1_direct_zoom_v2_h100_diag \
LOG_DIR=/workspace/pact/experiments/results/qfaithful_geometry_closed_candidates_20260502/h100_diag_2146_pr64_qp1_direct_zoom_v2 \
PREDICTED_LOW=0.30 \
PREDICTED_HIGH=25.0 \
CONTROLLED_BASELINE="qfaithful_2146_pr64_qp1_nozoom_l40s_score_22.07" \
bash scripts/remote_archive_only_eval.sh
```

This remains non-promotable until exact CUDA auth eval lands on the exact
archive SHA above and the result is adjudicated. A T4 promotion run should wait
until the fast-chip diagnostic shows the geometry-closed archive is in band.

## H100 Exact Diagnostic Result - 2026-05-02T07:15Z

The H100 exact diagnostic landed as an implementation-negative contract result,
not a Q-FAITHFUL family kill:

```text
harvest_dir=experiments/results/vast_harvest/qfaithful_geometry_closed_h100_20260502T0700Z
archive_sha256=f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61
archive_bytes=274257
hardware=NVIDIA H100 80GB HBM3
contest_auth_score=22.147631187370024
component_trace_score=22.147632116385466
component_trace_cross_check=true
posenet=46.54520035
segnet=0.00390678
classification=A-negative diagnostic contract failure
```

Inflate/runtime logs show the actual failure:

```text
archive members after unpack: renderer.bin, masks.mkv, optimized_poses.bin, zoom_scalars.bin
renderer: QZS3 JointFrameGenerator (qpose14-style packer)
mask contract: 600 half-frame masks
runtime warning: half-frame masks but no zoom_warp -- degraded duplicate path
```

The direct geometry-closed candidate preserved and charged `zoom_scalars.bin`,
but the exported renderer did not expose `use_zoom_flow` and therefore never
consumed those bytes. The failed predicate was renderer-geometry consumption,
not archive byte custody.

Permanent guard:

- `scripts/q_faithful_snapshot_loop.py` now records
  `renderer_zoom_contract` in `qfaithful_snapshot_runtime_contract_v2`.
- Half-frame snapshots with charged zoom geometry are non-promotable unless
  the renderer contract proves zoom consumption.
- The failure reason is
  `zoom_warp_geometry_not_consumed_by_renderer`.
- Focused verification:
  `src/tac/tests/test_q_faithful_snapshot_loop.py -> 11 passed`.

Next Q-FAITHFUL work must either export a real `use_zoom_flow=True` renderer
whose header/runtime consumes `zoom_scalars.bin`, or switch to a full-frame
snapshot. Merely adding charged geometry members to a PR67/QZS3
JointFrameGenerator archive is retired for this implementation.

## Runtime Mask-Expansion Supersession - 2026-05-02T09:25Z

The 07:15Z guard was too narrow: it treated zoom consumption as necessarily a
renderer `ego_flow` input. The more general contest-faithful contract is that
charged `zoom_scalars.bin` may also be consumed by the inflate runtime for
half-frame mask expansion before renderer invocation. This is still fully
charged and deterministic because the geometry member is inside the archive and
the runtime does not load a scorer.

Permanent fix:

- `submissions/robust_current/inflate_renderer.py` now loads charged
  `zoom_scalars.bin` whenever half-frame masks are present, even if
  `renderer.use_zoom_flow` is false.
- The loader validates fp16 byte length and pair count against the mask
  contract and fails closed on mismatch.
- `scripts/q_faithful_snapshot_loop.py` now distinguishes
  `renderer_consumes_ego_flow=false` from
  `runtime_consumes_zoom_warp_for_mask_expansion=true`.
- `AGENTS.md` records the durable protocol: zoom geometry must be consumed by
  either renderer ego-flow or pre-render half-frame mask expansion.
- Focused verification:
  `src/tac/tests/test_q_faithful_snapshot_loop.py` plus
  `src/tac/tests/test_inflate_renderer_zoom_geometry.py` passed
  (`13 passed`).

The exact same archive bytes were then re-screened on H100 SXM with the
patched runtime:

```text
harvest_dir=experiments/results/vast_harvest/qfaithful_zoom_runtime_fix_h100sxm_20260502T0920Z
archive_sha256=f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61
archive_bytes=274257
hardware=NVIDIA H100 80GB HBM3
contest_auth_score=22.147631187370024
posenet=46.54520035
segnet=0.00390678
runtime_proof=Loaded zoom_scalars.bin for half-frame mask expansion; warped 600 masks to 1200 frames
classification=A-negative diagnostic implementation failure
```

Conclusion: the zoom-consumption bug class is fixed, but it was not the main
confound for this Q-FAITHFUL snapshot. The candidate still collapses after
correct mask expansion, so future Q-FAITHFUL work must improve the exported
renderer/checkpoint/pose basin or use a full-frame/geometry-trained successor.
Do not spend T4 on this archive.
