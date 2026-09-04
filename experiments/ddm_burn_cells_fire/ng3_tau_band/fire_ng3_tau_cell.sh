#!/bin/bash
# MAIN fire of ng3's τ-band cell from its RE-ROOTED sealed config (validated PASS inside the sealed tree). Preconditions CHECKED.
set -euo pipefail
cd /Users/adpena/Projects/pact
S=/Volumes/APDataStore/pact/ddm_ng3_tau_band
SRC=/Volumes/VertigoDataTier/pact/ddm_ng3_tau_band/sealed_source_eed6f963c4
D=$(date -u +%Y%m%d)
# operator 2026-09-04: saturate CPU+GPU+ANE — concurrent Metal cells are ADMITTED by the memory guard, not refused by serial order.
NG3_PEAK_GIB=42; FREE_NOW=$(( $(vm_stat | awk '/Pages free/{gsub("\\.","",$3); print $3}') * 16384 / 1073741824 )); INACT=$(( $(vm_stat | awk '/Pages inactive/{gsub("\\.","",$3); print $3}') * 16384 / 1073741824 )); RECLAIM=$((FREE_NOW+INACT))
[ "$RECLAIM" -ge $((NG3_PEAK_GIB+16)) ] || { echo "REFUSE: memory guard — reclaimable ${RECLAIM} GiB < $((NG3_PEAK_GIB+16)) GiB (ng3 peak ${NG3_PEAK_GIB} + 16 margin)"; exit 2; }
echo "memory guard ADMITS: reclaimable ${RECLAIM} GiB (free ${FREE_NOW} + inactive ${INACT}); concurrent Metal cells: $(pgrep -f run-config | wc -l | tr -d " ")"
FREE_GIB=$(df -g /Volumes/APDataStore | awk 'NR==2{print $4}'); [ "$FREE_GIB" -ge 10 ] || { echo "REFUSE: APDataStore free ${FREE_GIB} GiB < 10"; exit 2; }
FREE_MEM_GIB=$(( $(vm_stat | awk '/Pages free/{gsub("\\.","",$3); print $3}') * 16384 / 1073741824 )); echo "free memory ≈ ${FREE_MEM_GIB} GiB (ng3 smoke peaked at 41.5 GiB; the launcher derives its own budget)"
SC=ddm_ng3_scorer_$D; MC=ddm_ng3_metal_$D
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $SC --platform local_macos_cpu --instance-job-id ng3_tau_band_$D --agent claude-main --status active_eval --ttl-hours 8 --notes "ng3 tau-band cell tau_for_step(2*delta_R -> delta_R) vs measured cold control seed_20260902; sealed c046decb3, pins re-rooted; falsifiers: S_hat(5k)<0.425149 and @2k<0.485677; fixed-tau surrogate peaks with d_seg_hat; Lane share 1.28-1.90x lower"
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $MC --platform local_mlx_metal --instance-job-id ng3_tau_band_$D --agent claude-main --status active_eval --ttl-hours 8 --notes "ONE Metal fire: ng3 tau-band cell (~3 h; peak RSS 41.5 GiB in smoke)"
.venv/bin/python $S/fire/authorize_tau_cell.py --scorer-claim-id $SC --metal-claim-id $MC --write
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng3_tau_band/launch/seed_20260902_tau_band_control_native100 \
  --cwd $SRC --purpose "NG3 tau-band cell seed_20260902_tau_band_control_native100 (pins re-rooted to the sealed tree)" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib 41.5 \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt ng3_tau_band_DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config $S/authorized_configs/seed_20260902_tau_band_control_native100.json
echo "FIRED $(date -u +%H:%M:%SZ) — claims $SC / $MC"
