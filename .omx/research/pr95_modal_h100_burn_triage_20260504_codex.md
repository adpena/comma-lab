# PR95 Modal H100 full-burn triage - 2026-05-04

Scope: local recovery/status only for the owned Modal H100 PR95 HNeRV/Muon
full burns. No new GPU jobs were started and no exact eval was run.

Champion comparison target:

- PR95 stemperm A++ score: `0.23089404465634825`
- Champion archive bytes: `178277`
- Champion archive SHA-256:
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`

## Status commands

- `.venv/bin/python experiments/modal_recover_lane.py --label pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z`
- `.venv/bin/python experiments/modal_recover_lane.py --label pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z`
- `.venv/bin/modal volume ls comma-train-lane-results pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z/`
- `.venv/bin/modal volume ls comma-train-lane-results pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z/`
- `.venv/bin/modal app list`
- `.venv/bin/modal volume get --force comma-train-lane-results pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z/ .omx/state/modal_pr95_h100_burn_triage_20260504`
- `.venv/bin/modal volume get --force comma-train-lane-results pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z/ .omx/state/modal_pr95_h100_burn_triage_20260504`

Both Modal function calls were still running at poll time:

- fix1 call: `fc-01KQS22WSZ7YR3ZJYXVPPYE4VB`
- fix2 call: `fc-01KQS25G854XJWFKWCCMYZTDTT`

Modal app status showed the two matching detached `comma-train` apps with one
active task each.

## Latest recovered snapshots

fix1:

- Local archive path:
  `.omx/state/modal_pr95_h100_burn_triage_20260504/pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z/results/owned_pr95_hnerv_muon_full_burn_20260504T083644Z/archive.latest.zip`
- Snapshot manifest:
  `.omx/state/modal_pr95_h100_burn_triage_20260504/pr95_hnerv_muon_full_burn_modal_h100_fix1_20260504T0836Z/results/owned_pr95_hnerv_muon_full_burn_20260504T083644Z/snapshot_latest_manifest.json`
- Recorded at: `2026-05-04T09:33:45Z`
- Archive bytes: `213888`
- Archive SHA-256:
  `749c3e4ac12bd3893b697d2a4790165014e5bc02db2c74d88ffb55973fdf7a1e`
- Latest member bytes/SHA:
  `213780`,
  `e4600b4fb405998e4d79ef6e1a77dad2731964b7547162575e6ce1595f08607c`
- Source checkpoint payload:
  `/tmp/pact/submissions/hnerv_muon/ckpts/run_20260504_083653/stage1/best_archive.bin`
- Latest train line: epoch `550/3000`, proxy score `0.4366`, seg `0.00204`,
  pose `0.000819`, archive `213780`

fix2:

- Local archive path:
  `.omx/state/modal_pr95_h100_burn_triage_20260504/pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z/results/owned_pr95_hnerv_muon_full_burn_20260504T083752Z/archive.latest.zip`
- Snapshot manifest:
  `.omx/state/modal_pr95_h100_burn_triage_20260504/pr95_hnerv_muon_full_burn_modal_h100_fix2_20260504T0838Z/results/owned_pr95_hnerv_muon_full_burn_20260504T083752Z/snapshot_latest_manifest.json`
- Recorded at: `2026-05-04T09:34:54Z`
- Archive bytes: `213866`
- Archive SHA-256:
  `412350c396541f3c898a63aac338f1132871815783a9a9f80e6511ce57b9678e`
- Latest member bytes/SHA:
  `213758`,
  `da80522242dc76e37d7d9b7a60a0b7693f7b1e901eda986a52b7b22ebaba35d9`
- Source checkpoint payload:
  `/tmp/pact/submissions/hnerv_muon/ckpts/run_20260504_083804/stage1/best_archive.bin`
- Latest train line: epoch `525/3000`, proxy score `0.4435`, seg `0.00216`,
  pose `0.000727`, archive `213758`

## Triage decision

No candidate is plausibly better than current PR95 stemperm from this recovery
pass. The latest recovered fix1/fix2 snapshots are still training snapshots,
not exact-eval evidence, and both are roughly `35.6 KB` larger than the
`178277`-byte champion while proxy scores remain far above the champion exact
score (`0.4366` and `0.4435` vs `0.23089404465634825`).

Next required step if a later snapshot becomes competitive: recover the exact
archive path, record bytes and SHA-256, add/verify a dispatch claim, then run
the canonical exact CUDA auth eval. Do not claim score from these snapshots.
