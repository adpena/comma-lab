# DRAFT — MAIN fills <HARVEST_JSON> from run1/ after the poller writes MODAL_REMOTE_RESULT.json; verify S from components first.
.venv/bin/python tools/pointer_move_packet.py \
  --harvest <HARVEST_JSON> \
  --seal .omx/research/ddm_pc3_20260911/SEAL_ddm_pc3_cap1_predictor_refit_move44_contest_cuda.json \
  --archive /Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime/archive.zip \
  --lane-id ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911 \
  --lane-name "ddm_pc3 CAP1 pose-carrier predictor refit on move 44 (bit-identical codes, -160 B)" \
  --call-id fc-01M28K31SPRPMJ1A9JYY9W42YB \
  --move-number 45 \
  --headline "pointer move 45: S <EXACT_S> @ 180,246 B [contest-CUDA T4 n600] — pc3 pose-carrier predictor refit at bit-identical codes (-160 B; decode byte-identical to move 44), normal seal with inherited t4_direct through pr18's behavior digest" \
  --mechanism-file .omx/research/ddm_pc3_packet_inputs_20260911/mechanism.md \
  --not-claimed-file .omx/research/ddm_pc3_packet_inputs_20260911/not_claimed.md \
  --next-file .omx/research/ddm_pc3_packet_inputs_20260911/next.md \
  --equations-leg "S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 44 (cold n600 raw byte-identical, 3,662,409,600 bytes) and B 180,406 → 180,246: ΔS = 25·(−160)/37,545,489 = −1.0653743e-4 exactly; projected S 0.13713836673884064" \
  --projected-score 0.13713836673884064 \
  --projection-note "conditional arithmetic from the -160 B predictor refit on move 44's raw-identical decode; realized − projected must be within float rounding of 0" \
  --apply --commit
