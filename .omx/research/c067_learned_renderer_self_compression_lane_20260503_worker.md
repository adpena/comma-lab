# C067 Learned Renderer Self-Compression Lane - Worker Memo

Date: 2026-05-03
Author: Codex worker
Scope: local planning artifact only. No GPU dispatch performed.

## Executive Boundary

The C067 A++ frontier archive is the fixed source of truth for this lane:

- archive: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- bytes: `276214`
- SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- score: `0.31561703078448233`
- unchanged-distortion sub-0.30 archive-byte ceiling: `252760`
- required unchanged-distortion savings: `23454` bytes

The logical source members are:

| member | bytes | SHA-256 |
| --- | ---: | --- |
| `renderer.bin` | 59288 | `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb` |
| `masks.mkv` | 223385 | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `optimized_poses.bin` | 7200 | `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f` |

This lane must preserve `masks.mkv` and `optimized_poses.bin` byte-for-byte. The only acceptable changed member is `renderer.bin`, plus deterministic archive container changes required to repack the public PR64/fixed-slice payload.

## Current Evidence

1. `experiments/results/c067_renderer_self_compression_v2_20260503_review/plan.json` already exhausts the local byte-only recoding direction enough for dispatch triage. The best non-fail-closed local candidate saves only 87 renderer-stream bytes, far below the 1024-byte dispatch gate and far below the 23454-byte unchanged-distortion gap.
2. The best byte-looking global QZS3 reblock family is fail-closed from exact CUDA negative evidence because it is in the measured PoseNet-collapse family. Do not resurrect that family as a burn seed without a new geometry escape contract.
3. `experiments/results/trained_renderer_export_unlock_20260503_review/plan.json` reports `blocked_no_h100_dispatch`: no accepted non-surrogate trained-renderer archive passed the current readiness scan.
4. The explicit ready-looking QBF1 preflight artifact at `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_renderer_blockfp_preflight.json` is not a promotion seed. Its exact CUDA eval at `experiments/results/lightning_batch/exact_eval_trained_qbf1_b0512_g7e_20260503T0142Z/contest_auth_eval.json` scored `17.72267562501643` recomputed (`17.72` final), with `avg_posenet_dist=29.82484055`, `avg_segnet_dist=0.0026408`, `n_samples=600`, and `archive_size_bytes=283432`. The fixed masks and poses match C067, so this is renderer-induced PoseNet collapse plus byte regression.

## Tooling Audit

- `src/tac/self_compress.py` implements the older Lane-S style `SelfCompressingConv2d` path for eligible `Conv2d` layers, with protected scorer-sensitive modules and rate penalties. It is useful design precedent, but it is not the current C067 JointFrameGenerator path by itself.
- `src/tac/renderer_export.py` has deterministic `SCv1` export/load support for self-compressed renderers, plus the broader renderer export machinery. For current C067, the already-reviewed runtime wire formats are `QZS3`, `MQZ1`, and `QBF1` JointFrameGenerator-compatible renderer payloads.
- `submissions/robust_current/inflate_renderer.py` has fail-closed loader branches for `SCv1`, `QZS3`, `MQZ1`, and `QBF1`. These runtime paths do not require scorer loads at inflate time.
- `scripts/remote_lane_s_self_compress.sh` and `scripts/remote_lane_w_v2_learnable_hardness.sh` are Lane-A-era remote scripts. They use older direct member ZIP assumptions and `optimized_poses.pt` style inputs. Do not run them verbatim for C067. Reuse only their staged discipline: NVDEC probe, provenance, dead-flag scan, deterministic training, export, preflight, then exact auth eval.
- `experiments/preflight_trained_renderer_transplant.py` is the current local fail-closed gate for C067 trained-renderer transplant candidates. It extracts the C067 source archive, requires the logical members, rejects source-surrogate renderers unless explicitly allowed, accepts `QZS3`/`MQZ1`/`QBF1` renderer exports, validates JointFrameGenerator state locally, packs QBF1 variants, and emits readiness JSON without dispatch.
- `experiments/plan_trained_renderer_export_unlock.py` is the current local scanner and byte-target calculator. Its constants match the C067 frontier and the sub-0.30 unchanged-component byte ceiling.

No new code is required for this worker slice. The missing piece is not a deterministic local planner or preflight; it is a fresh trained renderer candidate that is not in the stale QBF1 exact-negative lineage.

## H100 Burn Contract

Goal: a 12-24 hour H100 train/export/preflight cycle for a new Q-FAITHFUL JointFrameGenerator successor that keeps C067 masks and poses fixed, exports renderer-only candidates, and blocks exact eval unless local archive custody and byte gates pass.

Non-negotiable gates:

1. The source archive SHA-256 must equal `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
2. Every candidate must preserve:
   - `masks.mkv` SHA-256 `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
   - `optimized_poses.bin` SHA-256 `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f`
3. The candidate renderer must not equal the source renderer SHA-256 `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`.
4. For an unchanged-distortion sub-0.30 claim, local archive bytes must be `<=252760` before exact CUDA eval is worth spending. If bytes are higher, dispatch only if the training run has a concrete component-improvement hypothesis and the exact eval claim is framed as diagnostic.
5. Exact eval, if later run, must use `archive.zip -> inflate.sh -> upstream/evaluate.py`, preferably through `experiments/contest_auth_eval.py --device cuda`, after a dispatch claim is created.

## Concrete 12-24 Hour H100 Command Skeleton

Run these on the H100 workspace after syncing the current checkout and artifacts. The commands are concrete in path and flag shape; replace only `RUN_ID`, `ETA_UTC`, and Lightning/job identifiers.

### 0. Prepare Fixed C067 Inputs

```bash
set -euo pipefail
cd /teamspace/studios/this_studio/pact

export RUN_ID=c067_qfaithful_h100_selfcompress_20260503_seed20260503
export RUN=experiments/results/${RUN_ID}
export C067_ARCHIVE=experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip
mkdir -p "${RUN}/inputs" "${RUN}/logs"

.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json
from experiments.build_blockfp_c067_archive import extract_runtime_members

run = Path("experiments/results/c067_qfaithful_h100_selfcompress_20260503_seed20260503")
archive = Path("experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip")
expected_archive_sha = "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
data = archive.read_bytes()
actual_archive_sha = hashlib.sha256(data).hexdigest()
if actual_archive_sha != expected_archive_sha:
    raise SystemExit(f"wrong source archive SHA: {actual_archive_sha}")
members, _contract = extract_runtime_members(archive)
manifest = {
    "source_archive": str(archive),
    "source_archive_bytes": archive.stat().st_size,
    "source_archive_sha256": actual_archive_sha,
    "members": {},
}
for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
    payload = members[name]
    out = run / "inputs" / name
    out.write_bytes(payload)
    manifest["members"][name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
expected = {
    "renderer.bin": "5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb",
    "masks.mkv": "a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb",
    "optimized_poses.bin": "5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f",
}
for name, sha in expected.items():
    if manifest["members"][name]["sha256"] != sha:
        raise SystemExit(f"{name} SHA mismatch: {manifest['members'][name]['sha256']}")
(run / "inputs" / "fixed_c067_member_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
```

### 1. Train A Fresh Fixed-Mask/Fixed-Pose Q-FAITHFUL Renderer

This command uses the checked-in argparse surface: `q_faithful_dilated_88k`, `eval_roundtrip=True` from the profile, C067 masks forced through `--mask-noise-mkv` and `--mask-noise-prob 1.0`, and C067 poses through `--qfaithful-training-poses`.

```bash
export PYTHONHASHSEED=1234
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -u src/tac/experiments/train_renderer.py \
  --profile q_faithful_dilated_88k \
  --video upstream/videos/0.mkv \
  --device cuda \
  --seed 20260503 \
  --deterministic \
  --tag "${RUN_ID}" \
  --output-dir "${RUN}/train" \
  --qfaithful-training-poses "${RUN}/inputs/optimized_poses.bin" \
  --mask-noise-mkv "${RUN}/inputs/masks.mkv" \
  --mask-noise-prob 1.0 \
  --auth-eval-masks "${RUN}/inputs/masks.mkv" \
  --auth-eval-poses "${RUN}/inputs/optimized_poses.bin" \
  --no-auth-eval-on-best \
  --wall-clock-timeout 82800 \
  2>&1 | tee "${RUN}/logs/train_renderer.log"
```

Abort criteria during training:

- any fallback to CPU/MPS;
- all-zero or wrong-shape `qfaithful_training_poses`;
- mask count/geometry mismatch from `masks.mkv`;
- any run that silently drops `eval_roundtrip=True`;
- any checkpoint/export that reuses the source renderer bytes.

### 2. Export/Repack Snapshots Without Dispatch

Use the snapshot loop to export Q-FAITHFUL checkpoints to C067-layout archives. Keep `--eval-mode none` or `command`; do not use `run` in this local burn step.

```bash
.venv/bin/python -u scripts/q_faithful_snapshot_loop.py \
  --workspace "$PWD" \
  --python-bin .venv/bin/python \
  --checkpoint-dir "${RUN}/train" \
  --checkpoint-glob "training_state_*.pt" \
  --masks-mkv "${RUN}/inputs/masks.mkv" \
  --mask-frame-contract auto \
  --poses-pt "${RUN}/inputs/optimized_poses.bin" \
  --output-root "${RUN}/snapshots" \
  --profile q_faithful_dilated_88k \
  --state-source ema_shadow \
  --renderer-codec qzs3 \
  --qzs3-block-size 32 \
  --submission-layout pr64_mask_first_single_blob \
  --pose-codec raw \
  --brotli-quality 11 \
  --eval-mode none \
  --dispatch-claim-mode none \
  2>&1 | tee "${RUN}/logs/q_faithful_snapshot_loop.log"
```

If the training checkpoint does not include `ema_shadow`, rerun the export with `--state-source auto` only after recording that downgrade in the run manifest. Do not exact-eval a non-EMA export unless its manifest explicitly records the state source and the byte gate is compelling.

### 3. Extract Candidate Renderer And Run Local Transplant Preflight

For each snapshot archive that beats the source renderer bytes or shows a plausible component-improvement reason, extract the logical renderer and run the transplant preflight. This example assumes the snapshot loop produced `${CANDIDATE_ARCHIVE}`.

```bash
export CANDIDATE_ARCHIVE="${RUN}/snapshots/<candidate>/archive.zip"
export CANDIDATE_DIR="${RUN}/preflight/$(basename "$(dirname "${CANDIDATE_ARCHIVE}")")"
mkdir -p "${CANDIDATE_DIR}/unpacked"

.venv/bin/python - <<'PY'
from pathlib import Path
from experiments.build_blockfp_c067_archive import extract_runtime_members
import os

archive = Path(os.environ["CANDIDATE_ARCHIVE"])
out = Path(os.environ["CANDIDATE_DIR"]) / "unpacked"
members, _contract = extract_runtime_members(archive)
for name in ("renderer.bin", "masks.mkv", "optimized_poses.bin"):
    (out / name).write_bytes(members[name])
PY

.venv/bin/python -u experiments/preflight_trained_renderer_transplant.py \
  --source-archive "${C067_ARCHIVE}" \
  --renderer-export "${CANDIDATE_DIR}/unpacked/renderer.bin" \
  --output-dir "${CANDIDATE_DIR}" \
  2>&1 | tee "${CANDIDATE_DIR}/preflight.log"
```

Codex integration addendum: after the 2026-05-03 pose-safety hardening, this
first transplant preflight is intentionally not dispatch-ready. It builds
byte-closed QBF1 candidates and manifests, then a matching pose-safety report
must be generated for the exact selected source/candidate archive SHA pair:

```bash
export SELECTED_CANDIDATE_ARCHIVE="${CANDIDATE_DIR}/trained_qbf1_b0512/archive.zip"

.venv/bin/python -u experiments/preflight_renderer_transplant_pose_safety.py \
  --source-archive "${C067_ARCHIVE}" \
  --candidate-archive "${SELECTED_CANDIDATE_ARCHIVE}" \
  --output-json "${CANDIDATE_DIR}/pose_safety_preflight.json" \
  --max-pairs 5 \
  2>&1 | tee "${CANDIDATE_DIR}/pose_safety_preflight.log"
```

If and only if the pose-safety JSON reports
`safe_for_exact_eval_dispatch=true`, rerun the transplant preflight with the
gate supplied:

```bash
.venv/bin/python -u experiments/preflight_trained_renderer_transplant.py \
  --source-archive "${C067_ARCHIVE}" \
  --renderer-export "${CANDIDATE_DIR}/unpacked/renderer.bin" \
  --output-dir "${CANDIDATE_DIR}_posegate" \
  --pose-safety-json "${CANDIDATE_DIR}/pose_safety_preflight.json" \
  2>&1 | tee "${CANDIDATE_DIR}_posegate/preflight.log"
```

Generated exact-eval command shapes must use concrete Lightning machine
classes such as `g7e.4xlarge`; symbolic accelerator names are not valid
copy-paste submit commands on the current Studio backend.

Then rescan readiness:

```bash
.venv/bin/python -u experiments/plan_trained_renderer_export_unlock.py \
  --scan-dir "${RUN}/snapshots" \
  --scan-dir "${RUN}/preflight" \
  --output "${RUN}/trained_renderer_export_unlock_plan.json"

jq '.readiness, .byte_targets, .candidates[0:5]' "${RUN}/trained_renderer_export_unlock_plan.json"
```

### 4. Exact-Eval Submission Gate For Later

Do not dispatch from this worker memo. If a candidate passes local gates, claim the lane before any non-dry-run Lightning exact eval:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id c067_trained_renderer_self_compression_blockfp \
  --platform lightning \
  --instance-job-id "${RUN_ID}_exact_eval" \
  --agent codex \
  --status eval_pending \
  --predicted-eta-utc "${ETA_UTC}" \
  --notes "C067 renderer-only trained burn; fixed masks/poses; source_sha=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
```

Only after that claim exists, use the `h100_lightning_commands` emitted by `experiments/preflight_trained_renderer_transplant.py` or `experiments/plan_trained_renderer_export_unlock.py`. The exact-eval command must include:

- `--dispatch-lane-id c067_trained_renderer_self_compression_blockfp`
- source archive SHA metadata
- fixed masks SHA metadata
- fixed poses SHA metadata
- `--adjudicate`
- a dry-run pass before submission

## Promotion Criteria

A candidate can be promoted for exact CUDA eval only if the local artifacts show all of:

- source archive SHA matches C067;
- `masks.mkv` and `optimized_poses.bin` SHA-256s match the C067 logical members above;
- `renderer.bin` differs from the C067 source renderer;
- archive construction is deterministic and zip-slip safe;
- `inflate_renderer.py` recognizes the renderer magic through a reviewed loader;
- local preflight reports a non-surrogate JointFrameGenerator renderer;
- unchanged-distortion archive bytes are `<=252760`, or the exact eval is explicitly diagnostic for a measured component-improvement hypothesis;
- no stale QBF1 `trained_qbf1_b0512` artifact is used as a seed or promotion candidate.

## Recommendation

The fastest credible sub-0.30 attempt is not another local QZS3 reblock. It is a fresh H100 Q-FAITHFUL renderer training run with C067 masks and poses forced in training, followed by snapshot export to C067 public PR64 layout, transplant preflight, and exact CUDA auth eval only after the byte gate and dispatch claim pass.
