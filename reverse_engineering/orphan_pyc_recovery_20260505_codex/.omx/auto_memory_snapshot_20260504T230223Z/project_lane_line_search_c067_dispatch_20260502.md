---
name: Lane LS-C067 PR67-style line-search dispatch BLOCKED on Vast.ai zero-credit — code+artifacts landed at HEAD 43fc168b ready for relaunch
description: 2026-05-02 ~13:30Z. Subagent built complete dispatch chain for PR67-style R(D)-joint pose-0 coordinate descent on C-067 anchor (276,214B 0.31561703 [contest-CUDA T4]). Lane script remote_lane_line_search_c067.sh + C-067 source archive custody at experiments/results/lane_line_search_c067_20260502/ committed at 43fc168b. Vast.ai dispatch FAILED at offer-search → instance-create boundary because account balance is -$0.17 (negative). NO GPU spend, no archive emitted. Recovery is one operator credit-top-up away.
type: project
originSessionId: subagent-line-search-c067-2026-05-02
---

## TL;DR

| Item | Value |
|---|---|
| **Mission** | PR67-style R(D)-joint pose-0 coordinate descent on C-067 anchor |
| **Status** | BLOCKED at instance-create on Vast.ai |
| **Reason** | Vast.ai balance -$0.17 (negative); `vastai create instance` returns 400 "Your account lacks credit" |
| **Code state** | LANDED at HEAD 43fc168b (lane script + custody artifacts) |
| **Dispatch claim** | active_lane_dispatch_claims.md updated to `failed_no_credit_vast_ai` |
| **Custody** | source archive sha 226475de42... 276214B verified locally |
| **Cost spent** | $0 (no instance was created) |
| **Recovery** | Operator funds Vast.ai $5+ → re-run identical dispatch one-liner |

## What got built (LANDED at HEAD 43fc168b)

### `scripts/remote_lane_line_search_c067.sh` (242 LOC)

Three-stage canonical lane script:

- **Stage 0a**: `bootstrap_runtime_deps()` (sourced from `scripts/remote_archive_only_eval.sh`)
- **Stage 0b**: NVDEC probe via `scripts/probe_nvdec.sh --ensure-dali`
- **Stage 1**: Verify C-067 anchor archive SHA matches `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- **Stage 2**: `experiments/line_search_pose_refinement.py --radii "1,2,3,5,8" --passes 2 --device cuda:0` (the canonical PR #67 port)
- **Stage 3**: `experiments/contest_auth_eval.py` on the refined archive

CLI flags VERIFIED via `grep "add_argument"` against both target scripts (CLAUDE.md NEVER-INVENT-CLI-FLAGS rule).

### `experiments/results/lane_line_search_c067_20260502/`

- `source_archive.zip` (276,214 bytes, sha `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`) — gitignored
- `source_metadata.json` (committed) — slice contract for `experiments/line_search_pose_refinement.py` (mask_br=219472, model_br=55965, pose_br=677)

Source archive copied byte-identical from
`experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip`,
which is byte-identical to the C-067 anchor at
`experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`.

## Failure point

```
$ vastai create instance 31817923 --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
    --disk 60 --label lane_line_search_c067 --ssh \
    --env "NVIDIA_DRIVER_CAPABILITIES=all" --raw
{"error": true, "status_code": 400, "msg": "Your account lacks credit; see the billing page."}
```

`vastai show user`:
- Balance: **-0.17** USD (negative)
- Bal. Thld: 15.0 (Enabled: True)
- Got Signup Credit: 1
- Last4: empty (no payment method on file)

The account has been overdrawn. Cheapest RTX 4090 offer at the moment was id=30101198 at $0.2538/hr (24.6GB VRAM, 217GB disk, reliability 0.9964) — exactly what the canonical search filter found. No issue with the launch script or filter; the issue is account balance only.

## Recovery — exact operator one-liner

**Step 1**: fund Vast.ai with at least $5 (1.50 cap × 3.3x safety margin):
- https://cloud.vast.ai/billing/

**Step 2**: re-claim and dispatch:

```bash
# Append fresh dispatch claim
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_line_search_c067 \
  --platform vast.ai \
  --instance-job-id "PENDING_CREATE" \
  --agent operator \
  --predicted-eta-utc <eta> \
  --status provisioning \
  --notes "Re-dispatch after Vast.ai credit top-up; resumes BLOCKED dispatch from project_lane_line_search_c067_dispatch_20260502.md"

# Run the canonical full launch
.venv/bin/python scripts/launch_lane_on_vastai.py full \
  --lane-script scripts/remote_lane_line_search_c067.sh \
  --label lane_line_search_c067 \
  --predicted-band 0.310 0.318 \
  --estimated-cost 1.50 \
  --council-priority 2 \
  --max-dph 0.30 \
  --min-disk-gb 60 \
  --anchor-dirs experiments/results/lane_line_search_c067_20260502
```

The lane script self-bootstraps uv + ffmpeg + ._ purge. NVDEC probe fires at Stage 0b. Total wall-clock ~30-60min on RTX 4090.

## Predicted gain (per CLAUDE.md memory references)

`reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`:
- Distortion improvement: +0.001 to +0.003 score
- Rate improvement (smaller QP1 brotli output): +0.0004 to +0.001 score
- **Combined: ~0.001-0.005 score improvement** on the C-067 anchor (276,214B 0.31561703)
- Predicted band: [0.310, 0.318]

If achieved, would be the first contest-CUDA improvement against C-067 baseline. Distance to Quantizr leader 0.33 reduces from -0.015 to -0.018 (negligible) but stacks deterministically with future moves.

## What changed between mission spec and reality

The mission spec said "Cost: $0.50 (RTX 4090 ~30-60 min)". Search showed cheapest RTX 4090 at $0.2538/hr → $0.25 worst-case for 1h. Mission spec also said "DO NOT silently exit (per prior NeRV refusal-misread lesson) — explicit failure report mandatory" — this memo IS that explicit failure report.

## Lessons / preflight gap

1. **Lane launcher should call `vastai show user` and fail loud at balance < estimated_cost BEFORE searching offers**. Currently it fails ~100ms further along, after searching offers, with a less-clear JSON-decode error in `create_instance()`. Filing as a candidate STRICT preflight check (preliminary slot 91): `check_vastai_balance_sufficient` in `scripts/launch_lane_on_vastai.py:cmd_phase1`.

2. **Cross-agent claim ledger worked**: codex's prior `lane_line_search_pose_refinement` claims (2026-05-02T00:41Z & T00:57Z) were both terminal (`completed_no_frontier`, `completed_score=...`), so this fresh attempt did not conflict. The serializer's `--force` allowed the claim status to be updated from `provisioning` → `failed_no_credit_vast_ai` cleanly.

3. **Lane registry**: `lane_line_search_c067` is NOT yet in `.omx/state/lane_registry.json` — should be added at Level 1 (impl_complete + memory_entry + deploy_runbook) when the dispatch lands a real measurement.

## Cross-refs

- `reports/raw/leaderboard_intel_20260501/pr67_line_search.py` (194 LOC reference implementation)
- `experiments/line_search_pose_refinement.py` (1474+ LOC local port; takes `--metadata-path` instead of pr67's brittle 7-bucket lookup)
- `src/tac/qp1_pose_codec.py` (152 LOC, byte-identical to PR #67's encode_qp1)
- `src/tac/quantizr_qzs3_codec.py:674` (`load_qzs3` for the C-067 renderer)
- `src/tac/quantizr_faithful_renderer.py` (JointFrameGenerator)
- `experiments/results/lane_line_search_c067_20260502/` (custody artifacts)
- `scripts/remote_lane_line_search_c067.sh` (242 LOC dispatch script)
- `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md` (full reverse-engineering writeup)
- `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md` (C-067 byte layout)
- `.omx/state/active_lane_dispatch_claims.md` (claim row at 2026-05-02T13:26:25Z)
- HEAD `43fc168b` — Lane LS-C067 code + custody landed
