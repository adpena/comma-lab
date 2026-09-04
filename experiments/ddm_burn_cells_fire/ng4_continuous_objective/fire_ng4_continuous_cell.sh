#!/bin/bash
# MAIN fire of ng4's continuous-objective cell (τ HELD 0.05, duals CARRIED) from its RE-ROOTED sealed config (validated PASS
# inside the sealed tree, sha 93f82b12…). Admission through tools/cell_admission.py (gv1) — never inline vm_stat arithmetic.
set -euo pipefail
cd /Users/adpena/Projects/pact
S=/Volumes/APDataStore/pact/ddm_ng4_continuous_objective
SRC=/Volumes/VertigoDataTier/pact/ddm_ng4_continuous_objective/sealed_source_50e2cd2808
D=$(date -u +%Y%m%d)
CFG=$S/sealed_configs/seed_20260902_continuous_objective_control_native100.rerooted.json
[ "$(shasum -a 256 $CFG | cut -d' ' -f1)" = "93f82b126dfddebc49e32ffd49734329c8d19142bbaa61450eff8f21e19b9267" ] || { echo "REFUSE: rerooted config sha drift"; exit 3; }
# ng4 measured: a concurrent Metal cell costs ~45 GiB of system availability (RSS 2.4 GiB is a fiction) — declare the real number.
NG4_PEAK_GIB=45
.venv/bin/python tools/cell_admission.py admit --candidate-peak-gib $NG4_PEAK_GIB || { echo "REFUSE: cell_admission (rc=$?)"; exit 2; }
FREE_GIB=$(df -g /Volumes/APDataStore | awk 'NR==2{print $4}'); [ "$FREE_GIB" -ge 6 ] || { echo "REFUSE: APDataStore free ${FREE_GIB} GiB < 6"; exit 2; }
SC=ddm_ng4_scorer_$D; MC=ddm_ng4_metal_$D
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $SC --platform local_macos_cpu --instance-job-id ng4_continuous_$D --agent main --status active_eval --notes "ng4 continuous-objective cell: frozen CPU scorer milestones (τ held 0.05, duals carried)"
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id $MC --platform local_mlx_metal --instance-job-id ng4_continuous_$D --agent main --status active_eval --notes "ng4 continuous-objective cell: MLX Metal 5,000 steps, declared ~45 GiB system availability"
.venv/bin/python $S/fire/authorize_continuous_cell.py --scorer-claim-id $SC --metal-claim-id $MC --write
$SRC/.venv/bin/python $SRC/tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ng4_continuous_objective/launch/seed_20260902_continuous_objective_control_native100 \
  --cwd $SRC --purpose "NG4 continuous-objective cell seed_20260902_continuous_objective_control_native100 (tau held, duals carried; pins re-rooted)" \
  --authority MAIN --derive-resource-budgets --measured-peak-rss-gib $NG4_PEAK_GIB \
  --measured-thread-need 4 --walltime-cap-s 18000 --done-receipt ng4_continuous_DONE.json \
  -- $SRC/.venv/bin/python $SRC/experiments/ddm_qbr1_born_fairform_burn_prep.py \
     run-config $S/authorized_configs/seed_20260902_continuous_objective_control_native100.json
echo "FIRED $(date -u +%H:%M:%SZ) — claims $SC / $MC"
