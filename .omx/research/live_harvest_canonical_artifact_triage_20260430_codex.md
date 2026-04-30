# Live Harvest Canonical Artifact Triage - 2026-04-30

Author: Codex
Checked at: 2026-04-30T19:44:24Z through 2026-04-30T19:46:19Z

Scope: live Vast inventory, `.omx/state/vastai_active_instances.json`,
`.omx/state/active_dispatches.md`, recovered `experiments/results/recovered_*`
metadata and snapshots, `scripts/reconcile_vast_dispatch_state.py`, and the
current live harvest status/audit docs.

This is a custody and harvest-control note, not a score ledger. It intentionally
omits score claims for incomplete lanes. Training logs, proxy metrics, CPU
Modal evals, stale sidecars, and recovered anchor archives are non-promotable.

## Commands Run

- `date -u +%Y-%m-%dT%H:%M:%SZ`
- `.venv/bin/python scripts/reconcile_vast_dispatch_state.py --max-items 80`
- `.venv/bin/vastai show instances --raw`
- Read-only SSH probes of live Vast instances using `ps`, `find`, and `date`.
- Local scans of recovered metadata, archives, `contest_auth_eval.json`,
  provenance, archive sizes, and archive SHA-256 values.

No remote instances were stopped, destroyed, or modified. No remote artifacts
were copied during this pass.

## Live State

Reconciliation at 2026-04-30T19:44Z reported:

- Live Vast instances: 4.
- Local tracker rows: 204.
- `active_dispatches.md` active rows: 3.
- Tracker rows missing from live Vast: 200.
- Live instances missing from tracker: 0.
- Active-dispatch rows missing from live Vast: `35899435`, `35899552`,
  `35899275`.
- Live instances missing from `active_dispatches.md`: `35885106`, `35906669`,
  `35907873`.

Lane 19 is live as `35899850`, but the active-dispatch table still points at
stale attempt `35899435`.

| Instance | Label | Probe status | Canonical archive/eval/provenance availability | Harvest status |
|---|---|---|---|---|
| `35885106` | `lane_hm_s_2026-04-30_b_a2` | Vast says running, GPU 0%; `ps` shows no lane Python/bash process. | Remote lane dir has `segmap_weights.tar.xz`, train weights, train/run/heartbeat logs, and `provenance.json`. No lane-local `archive*.zip`, `eval_work/contest_auth_eval.json`, or `auth_eval.log` found. | Diagnostic-only harvest. Do not score or rank. |
| `35899850` | `lane_19_logit_margin_2026-04-30_b_a4` | Active `train_renderer.py`, GPU active. | Remote lane dir has checkpoints, telemetry, train/run/heartbeat logs, and `provenance.json`. No lane-local archive or exact eval JSON yet. | Monitor only until Stage 4 emits exact archive/eval. |
| `35906669` | `lane_sa_segmap_clone_2026-04-30_codex_a2` | Active `train_segmap.py`, GPU active. | Remote lane dir has logs and `provenance.json`. No lane-local archive or exact eval JSON yet. | Monitor only until Stage 5 emits exact archive/eval/adjudication. |
| `35907873` | `lane_h_v3_joint_halfframe_2026-04-30_codex_a4` | Active `train_renderer.py`, GPU active. | Remote lane dir has checkpoints, telemetry, logs, and `provenance.json`. No lane-local archive or exact eval JSON yet. | Monitor only until Stage 4 emits exact archive/eval/adjudication. |

## Canonical Artifact Availability

| Artifact | Archive | `contest_auth_eval.json` | Provenance | Triage |
|---|---|---|---|---|
| `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/` | Present: `archive/archive.zip`, 686635 bytes, SHA-256 `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`. | Present: `eval/contest_auth_eval.json`, CUDA, 600 samples, matching archive SHA. | Present: build, eval, custody, source, upstream, run-command, and review artifacts. | Complete deploy packet. |
| `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/` | Present: `archive_lane_12_nerv.zip`, 296478 bytes, SHA-256 `864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97`. | Present: `contest_auth_eval.json`, CUDA, 600 samples, matching archive SHA. | Present: adjudication, dispatch, NeRV, logs. | Complete negative implementation evidence only; not a family kill. |
| `experiments/results/recovered_35793092_lane_sz_phase2_c/.../lane_sz_phase2_results/` | Present: `archive_lane_sz.zip`, 3388 bytes, SHA-256 `6c99778a407522a3f591842cdd7ea56a96d3bbc313f69cdedd5800aef2ac32a0`. | Missing. `eval_work/provenance.json` and extracted renderer exist, but no `contest_auth_eval.json`. | Present: `provenance.json`, `eval_work/provenance.json`, train/run/heartbeat logs. | Needs canonical CUDA rerun before any score or failure claim. |
| `experiments/results/lane_uniward_v8_modal/lane_uniward_results/` | Present: `archive_lane_uniward.zip`, 694045 bytes, SHA-256 `74bc09803a7cbca6a6220f3d302c5e6b8cb5ca2af37812e77bfef5d0a21d4080`. | Present but `provenance.device=cpu`; non-promotable. | Present. | Exact CUDA rerun only if still scientifically useful. |
| `experiments/results/lane_lane_mm_v2_modal/lane_mm_grayscale_lut_results/` | Present: `archive_lane_mm.zip`, 1133750 bytes, SHA-256 `4cb2d97e7ce2ffdfbcfce0718c039bcf729e5f153d03af974268db27584f6f2c`. | Present but `provenance.device=cpu`; non-promotable. | Present. | Exact CUDA rerun required before ranking. |
| `experiments/results/lane_lane_gp_v3_modal/lane_gp_results/` | Present: `archive_lane_gp.zip`, 692568 bytes, SHA-256 `f731285c50cfd02d16de152010b83daebd747f7ec343e362b5f2647d55afa968`. | Present but `provenance.device=cpu`; non-promotable. | Present. | Forensics only unless exact CUDA rerun is requested. |
| `experiments/results/lane_g_v3_omega_w_v2_stack_landed/` | Missing exact archive bytes locally. Local ZIP SHA scan found no file matching `eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625`. | Present: CUDA, 600 samples, records archive SHA `eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625`. | Present. | Custody-blocked until the exact archive is recovered or a new archive is built and re-evaluated. |
| `experiments/results/lane_20_balle_2026-04-30_a1_recovered/` | No lane archive. | Missing by design. | Present. | Empirical/no-op fallback only. |
| `experiments/results/lane_j_imp_crashed_cycle0/` | No valid lane archive. | Missing. | Present. | Run abort/engineering failure only. |

## Recovered Snapshot Findings

There are 25 `recovery_metadata.json` files under `experiments/results/recovered_*`.
Most `archive_zip` fields point at broad workspace anchors such as
`lane_a_landed/archive_lane_a.zip` or `lane_g_v3_landed/archive_lane_g_v3.zip`.
Those files are not lane-local results for the recovered instance.

Lane-local recovered evidence found:

- `recovered_35793092_lane_sz_phase2_c`: has `lane_sz_phase2_results/archive_lane_sz.zip`,
  lane provenance, train logs, `eval_work/archive.zip`, and `eval_work/provenance.json`;
  lacks `contest_auth_eval.json`.
- `recovered_35886131_scripts_remote_lane_omega_w_v2_stack.sh`: has
  `lane_g_v3_omega_w_v2_stack_results` logs/provenance only; no exact archive
  was recovered there.
- `recovered_35898712_scripts_remote_lane_20_balle.sh`: has lane logs/provenance
  only and is superseded by the local recovered Lane 20 diagnostic directory.
- `recovered_35885106_scripts_remote_lane_hm_s_segmap_homography.sh` and
  `recovered_35902406_scripts_remote_lane_pfp16_stack.sh`: no
  `recovery_metadata.json`; files present are anchor/baseline salvage only.

Do not attribute `submissions/robust_current/*`, `baseline_dilated_h64_0_90/*`,
Lane A, or Lane G anchor files inside recovered snapshots to the recovered lane.

## Exact Next Harvest Commands

Refresh non-destructive live truth before any copy:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/vastai show instances --raw | tee ".omx/state/vastai_show_instances_${STAMP}.json"
.venv/bin/python scripts/reconcile_vast_dispatch_state.py --json > ".omx/state/vastai_reconcile_${STAMP}.json"
```

Probe live lane completion without modifying remotes:

```bash
ssh -p 15106 root@ssh8.vast.ai 'cd /workspace/pact && date -u && ps -eo pid,stat,etime,pcpu,pmem,args | grep -E "remote_lane|train_|contest_auth_eval|archive|zip|tar|python" | grep -v grep || true && find lane_hm_s_segmap_homography_results -maxdepth 3 -type f \( -name "archive*.zip" -o -name "contest_auth_eval.json" -o -name "RESULT_JSON" -o -name "provenance.json" -o -name "auth_eval.log" -o -name "*.log" -o -name "segmap_weights.tar.xz" \) -ls'
ssh -p 19850 root@ssh2.vast.ai 'cd /workspace/pact && date -u && ps -eo pid,stat,etime,pcpu,pmem,args | grep -E "remote_lane|train_|contest_auth_eval|archive|zip|tar|python" | grep -v grep || true && find lane_19_logit_margin_results -maxdepth 3 -type f \( -name "archive*.zip" -o -name "contest_auth_eval.json" -o -name "RESULT_JSON" -o -name "provenance.json" -o -name "auth_eval.log" -o -name "*.log" \) -ls'
ssh -p 26668 root@ssh2.vast.ai 'cd /workspace/pact && date -u && ps -eo pid,stat,etime,pcpu,pmem,args | grep -E "remote_lane|train_|contest_auth_eval|archive|zip|tar|python" | grep -v grep || true && find lane_sa_segmap_clone_results -maxdepth 3 -type f \( -name "archive*.zip" -o -name "contest_auth_eval.json" -o -name "RESULT_JSON" -o -name "provenance.json" -o -name "auth_eval.log" -o -name "*.log" -o -name "segmap_weights.tar.xz" \) -ls'
ssh -p 27872 root@ssh5.vast.ai 'cd /workspace/pact && date -u && ps -eo pid,stat,etime,pcpu,pmem,args | grep -E "remote_lane|train_|contest_auth_eval|archive|zip|tar|python" | grep -v grep || true && find lane_h_v3_results -maxdepth 3 -type f \( -name "archive*.zip" -o -name "contest_auth_eval.json" -o -name "RESULT_JSON" -o -name "provenance.json" -o -name "auth_eval.log" -o -name "*.log" \) -ls'
```

Harvest HM-S diagnostics only. This is not a score harvest unless an archive and
exact eval appear later:

```bash
mkdir -p experiments/results/live_harvest_35885106_lane_hm_s_segmap_homography_20260430
rsync -avz --partial -e 'ssh -p 15106' \
  root@ssh8.vast.ai:/workspace/pact/lane_hm_s_segmap_homography_results/ \
  experiments/results/live_harvest_35885106_lane_hm_s_segmap_homography_20260430/
find experiments/results/live_harvest_35885106_lane_hm_s_segmap_homography_20260430 -type f -print0 \
  | xargs -0 shasum -a 256 \
  > experiments/results/live_harvest_35885106_lane_hm_s_segmap_homography_20260430/SHA256SUMS
```

Harvest live lanes only after the listed lane-local archive, exact eval JSON,
and provenance exist:

```bash
mkdir -p experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430
rsync -avz --partial -e 'ssh -p 19850' \
  root@ssh2.vast.ai:/workspace/pact/lane_19_logit_margin_results/ \
  experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430/
test -f experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430/archive_lane_19.zip
test -f experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430/eval_work/contest_auth_eval.json
test -f experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430/provenance.json
find experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430 -type f -print0 | xargs -0 shasum -a 256 > experiments/results/live_harvest_35899850_lane_19_logit_margin_20260430/SHA256SUMS

mkdir -p experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430
rsync -avz --partial -e 'ssh -p 26668' \
  root@ssh2.vast.ai:/workspace/pact/lane_sa_segmap_clone_results/ \
  experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430/
test -f experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430/archive_lane_sa_segmap_clone.zip
test -f experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430/eval_work/contest_auth_eval.json
test -f experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430/provenance.json
find experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430 -type f -print0 | xargs -0 shasum -a 256 > experiments/results/live_harvest_35906669_lane_sa_segmap_clone_20260430/SHA256SUMS

mkdir -p experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430
rsync -avz --partial -e 'ssh -p 27872' \
  root@ssh5.vast.ai:/workspace/pact/lane_h_v3_results/ \
  experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430/
test -f experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430/archive_lane_h_v3.zip
test -f experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430/eval_work/contest_auth_eval.json
test -f experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430/provenance.json
find experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430 -type f -print0 | xargs -0 shasum -a 256 > experiments/results/live_harvest_35907873_lane_h_v3_joint_halfframe_20260430/SHA256SUMS
```

Rerun exact CUDA eval for the recovered SZ archive on a CUDA host before any
claim:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/recovered_35793092_lane_sz_phase2_c/workspace/pact/lane_sz_phase2_results/archive_lane_sz.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir "experiments/results/recovered_35793092_lane_sz_phase2_c/workspace/pact/lane_sz_phase2_results/cuda_rerun_${STAMP}"
```

Rerun selected Modal CPU artifacts on a CUDA host only if they remain worth
adjudicating:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/lane_uniward_v8_modal/lane_uniward_results/archive_lane_uniward.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir "experiments/results/lane_uniward_v8_modal/lane_uniward_results/cuda_rerun_${STAMP}"
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/lane_lane_mm_v2_modal/lane_mm_grayscale_lut_results/archive_lane_mm.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir "experiments/results/lane_lane_mm_v2_modal/lane_mm_grayscale_lut_results/cuda_rerun_${STAMP}"
.venv/bin/python experiments/contest_auth_eval.py --archive experiments/results/lane_lane_gp_v3_modal/lane_gp_results/archive_lane_gp.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir "experiments/results/lane_lane_gp_v3_modal/lane_gp_results/cuda_rerun_${STAMP}"
```

Recover Omega-W-V2 exact archive bytes before using the existing JSON:

```bash
find experiments/results -type f -name '*.zip' -print0 \
  | xargs -0 shasum -a 256 \
  | rg '^eba8e4360e6366ca10905b58e9ec1d12b3480c78b041548daa21da0c46c31625'
```

If that search stays empty, any rebuilt Omega-W-V2 archive is a new artifact and
must receive a fresh canonical CUDA eval.

## Immediate Rules For Next Operator

- Do not destroy or stop the four live Vast instances from this report.
- Do not rank Lane 19, SA clone, H-V3, or HM-S from training/proxy logs.
- Do not score HM-S without a lane-local archive and CUDA
  `eval_work/contest_auth_eval.json`.
- Do not treat recovered `archive_zip` fields pointing at Lane A/Lane G anchors
  as recovered-lane archives.
- Do not promote Modal CPU `contest_auth_eval.json` files.
- Preserve exact copied bytes, SHA-256 manifests, logs, provenance, command
  lines, and remote hardware identity before any cleanup.
