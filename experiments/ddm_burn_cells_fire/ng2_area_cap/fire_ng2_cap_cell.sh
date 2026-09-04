#!/bin/bash
# MAIN fire of ng2's area-cap cell AS SEALED (the LOSE branch of the ng1 decision rule). Preconditions CHECKED.
set -euo pipefail
cd /Users/adpena/Projects/pact
S=/Volumes/APDataStore/pact/ddm_ng2_area_cap
SRC=/Volumes/VertigoDataTier/pact/ddm_ng2_area_cap/sealed_source_54161c2800
D=$(date -u +%Y%m%d)
# 1. Metal free: no cell/trainer alive; ng1's DONE receipt present.
pgrep -f 'ddm_qbt1_qbflow_trainer|run-config' >/dev/null && { echo "REFUSE: a trainer/cell is alive"; exit 2; }
NG1_RCPT=$(python3 -c "import json; print(json.load(open('/Volumes/APDataStore/pact/ddm_ng1_warm_transition/launch/seed_20260902_warm_transition/launch_manifest.json'))['done_receipt_path'])")
[ -s "$NG1_RCPT" ] || { echo "REFUSE: ng1 warm cell done receipt absent ($NG1_RCPT)"; exit 2; }
[ -s /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng1_warm_transition/runs/seed_20260902_warm_transition/RESULT.json ] || { echo "REFUSE: ng1 RESULT.json absent"; exit 2; }
# 2. Disk reserve (memo: 8 GiB reserve; cell ≈1.4 GB).
FREE_GIB=$(df -g /Volumes/APDataStore | awk 'NR==2{print $4}'); [ "$FREE_GIB" -ge 10 ] || { echo "REFUSE: APDataStore free ${FREE_GIB} GiB < 10"; exit 2; }
# 3. Fresh claims.
SC=ddm_ng2_scorer_$D; MC=ddm_ng2_metal_$D
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $SC --platform local_macos_cpu --instance-job-id ng2_area_cap_$D --agent claude-main --status active_eval --ttl-hours 8 --notes "ng2 one-sided area-cap cell (lambda_Lane 2799.8, lambda_Movable 7587.4 from the trainer's bincount) vs measured cold control seed_20260902; sealed 83d43153d; falsifiers: S_hat(5k)<0.425149 and @2k<0.485677; Lane/Movable area @2k within 1.03"
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $MC --platform local_mlx_metal --instance-job-id ng2_area_cap_$D --agent claude-main --status active_eval --ttl-hours 8 --notes "ONE Metal fire: ng2 area-cap cell (~2.95 h)"
# 4. Authorize through the chain driver's own functions.
.venv/bin/python $S/fire/authorize_cap_cell.py --scorer-claim-id $SC --metal-claim-id $MC --write
# 5. Launch EXACTLY the sealed argv from the ng2 memo.
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng2_area_cap/launch/seed_20260902_area_cap_control_native100 \
  --cwd $SRC \
  --purpose "NG2 area-cap cell seed_20260902_area_cap_control_native100" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 2.3959503173828125 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt ng2_area_cap_DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config $S/authorized_configs/seed_20260902_area_cap_control_native100.json
echo "FIRED $(date -u +%H:%M:%SZ) — claims $SC / $MC"
