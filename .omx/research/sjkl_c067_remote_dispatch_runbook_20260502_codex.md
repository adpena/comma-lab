# SJ-KL C067 Remote Dispatch Runbook - 2026-05-02

Scope: bounded no-spend runbook slice for `sjkl_c067`. This document and
`scripts/remote_lane_sjkl_c067.sh` define the remote path; they do not dispatch
paid compute by themselves.

## Contract

- Lane id: `sjkl_c067`
- Candidate: C067 fixedslice source archive plus charged `sjkl.bin`
- Remote driver: `scripts/remote_lane_sjkl_c067.sh`
- Source archive default:
  `experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip`
- Renderer tensor source default:
  `experiments/results/renderer_output_for_postfilter.pt`
- Score policy: `score_claim=false` until `contest_auth_eval.json` exists from
  `archive.zip -> inflate.sh -> upstream/evaluate.py` with `--device cuda`
- CUDA policy: fail closed when `nvidia-smi`, `torch.cuda.is_available()`, or the
  fast-chip regex check fails. `build_sjkl_residual.py` is invoked only with
  `--device cuda`, and the wrapper rejects CPU/MPS/advisory fallback text.

## Pre-Dispatch Checks

Run these locally before any spend:

```bash
git status --short
bash -n scripts/remote_lane_sjkl_c067.sh
.venv/bin/python -m pytest src/tac/tests/test_remote_lane_sjkl_c067_script.py -q
.venv/bin/python -m pytest src/tac/tests/test_prepare_sjkl_pair_tensors.py src/tac/tests/test_sjkl_basis.py src/tac/tests/test_inflate_renderer_sjkl_runtime.py -q
```

## Claim First

Use the exact job name in the claim row before launching any remote process:

```bash
JOB_NAME="sjkl-c067-l40s-$(date -u +%Y%m%dT%H%M%SZ)"
ETA_UTC="$(date -u -v+3H +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u -d '+3 hours' +%Y-%m-%dT%H:%MZ)"

tools/claim_lane_dispatch.py claim \
  --lane-id sjkl_c067 \
  --platform lightning \
  --instance-job-id "$JOB_NAME" \
  --agent codex:sjkl_c067 \
  --predicted-eta-utc "$ETA_UTC" \
  --status build_eval_ready \
  --notes "bounded SJ-KL C067 tensor-prep CUDA-build pack exact-eval; score_claim=false until contest_auth_eval.json"
```

If the helper reports an active conflict, do not dispatch. Coordinate through
`.omx/state/active_lane_dispatch_claims.md` or choose a different lane id.

## Stage Source And Artifacts

This stages source plus the large renderer tensor and C067 source archive. It
does not launch GPU work:

```bash
RUN_ID="sjkl_c067_stage_$(date -u +%Y%m%dT%H%M%SZ)"

.venv/bin/python scripts/lightning_repro_workspace.py \
  --remote "${LIGHTNING_SSH_TARGET:?set LIGHTNING_SSH_TARGET}" \
  --remote-pact "${LIGHTNING_REMOTE_PACT:-/teamspace/studios/this_studio/pact}" \
  --run-id "$RUN_ID" \
  --manifest-out ".omx/state/${RUN_ID}_lightning_source_manifest.json" \
  --requirements-mode uv-sync \
  --python-bin .venv/bin/python \
  --require-cuda \
  --artifact experiments/results/renderer_output_for_postfilter.pt \
  --artifact experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip
```

## Exact Remote Command

Do not run the remote command until the main agent/operator approves spend.
This command runs tensor prep, CUDA SJ-KL build, deterministic archive pack, and
exact CUDA auth eval:

```bash
ssh -o BatchMode=yes -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
  -o ConnectTimeout=15 "${LIGHTNING_SSH_TARGET:?set LIGHTNING_SSH_TARGET}" \
  "cd ${LIGHTNING_REMOTE_PACT:-/teamspace/studios/this_studio/pact} && \
   WORKSPACE=${LIGHTNING_REMOTE_PACT:-/teamspace/studios/this_studio/pact} \
   PYBIN=.venv/bin/python \
   SJKL_RUN_ID=$JOB_NAME \
   SJKL_FAST_CHIP_REGEX='H100|L40S|A100|RTX 4090|RTX 6000|RTX PRO|A10G' \
   SJKL_SOURCE_ARCHIVE=experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip \
   SJKL_RENDERER_FRAMES=experiments/results/renderer_output_for_postfilter.pt \
   SJKL_OUTPUT_DIR=experiments/results/$JOB_NAME \
   SJKL_PREDICTED_LOW=0.20 \
   SJKL_PREDICTED_HIGH=0.60 \
   bash scripts/remote_lane_sjkl_c067.sh"
```

The remote output directory must contain, at minimum:

- `source_artifact_manifest.json`
- `tensors/sjkl_pair_tensor_prep_manifest.json`
- `build/sjkl_manifest.json`
- `pack/archive_pack_manifest.json`
- `exact_eval/contest_auth_eval.json`
- `contest_auth_eval.json`
- `heartbeat.log`
- `remote_lane_sjkl_c067.log`

## Terminal Claim Row

When the remote process finishes, append a terminal row with the same
`lane_id` and `instance/job_id`:

```bash
tools/claim_lane_dispatch.py claim \
  --lane-id sjkl_c067 \
  --platform lightning \
  --instance-job-id "$JOB_NAME" \
  --agent codex:sjkl_c067 \
  --predicted-eta-utc "$ETA_UTC" \
  --status completed_score_pending_json_review \
  --force \
  --notes "remote finished; inspect exact_eval/contest_auth_eval.json before any score claim"
```

Use `--status failed_<reason>` instead if any stage fails. Do not leave the
active claim open.

## 2026-05-02 Codex Dispatch

Pre-dispatch correction:

- The remote driver must not add `sjkl.bin` as a sibling ZIP member beside
  C067's packed `p` payload. That form is charged by archive bytes but is not
  the intended logical renderer-payload contract and may be skipped by the
  runtime. The corrected driver uses
  `experiments/build_sjkl_c067_archive.py` and fails if `sjkl.bin` is not in
  the packed logical member list.

Structural local smoke:

- `experiments/results/sjkl_c067_packer_contract_smoke_20260502T151210Z/`
  packed a synthetic valid `sjkl.bin` into the actual C067 archive and
  `submissions/robust_current/unpack_renderer_payload.py` recovered
  `renderer.bin`, `masks.mkv`, `optimized_poses.bin`, and `sjkl.bin`.

Actual queued diagnostic:

- Claim row: `sjkl_c067_l40sdiag_20260502T151434Z`.
- State path:
  `.omx/state/sjkl_c067_l40sdiag_20260502T151434Z_batch_jobs.json`.
- Machine: Lightning `g6e.4xlarge` / observed SDK machine `L40S`.
- Initial status: `Pending`, cost `$0.00`.
- Evidence policy: L40S is diagnostic only. T4/equivalent promotion must use
  exact archive bytes emitted by the completed job, not a rebuilt approximation.
