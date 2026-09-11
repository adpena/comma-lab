# DRAFT — MAIN fills <HARVEST_JSON> from run1/ after the poller writes MODAL_REMOTE_RESULT.json; verify S from components first.
.venv/bin/python tools/pointer_move_packet.py \
  --harvest <HARVEST_JSON> \
  --seal .omx/research/ddm_hpr1_20260911/SEAL_ddm_hpr1_retrain_control_contest_cuda.json \
  --archive /Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime/archive.zip \
  --lane-id ddm_hpr1_retrain_control_move46_contest_cuda_20260911 \
  --lane-name "ddm_hpr1 retrained HPAC prior (shipped geometry, 60 ep under cl2's law) on move 46 (-642 B)" \
  --call-id fc-01M29166GTAZC3TA5V8W8Q54E1 \
  --move-number 47 \
  --headline "pointer move 47: S <EXACT_S> @ 179,359 B [contest-CUDA T4 n600] — hpr1 retrained HPAC prior in its shipped geometry (-642 B vs move 46; decoded field and raw byte-identical), normal seal inheriting move 46's measured t4_direct leg" \
  --mechanism-file .omx/research/ddm_hpr1_packet_inputs_20260911/mechanism.md --not-claimed-file .omx/research/ddm_hpr1_packet_inputs_20260911/not_claimed.md --next-file .omx/research/ddm_hpr1_packet_inputs_20260911/next.md \
  --equations-leg "S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 46 (decoded field and cold n600 raw byte-identical) and B 180,001 → 179,359: ΔS = 25·(−642)/37,545,489 = −4.2748145e-4 exactly; projected S 0.13654774984742127" \
  --projected-score 0.13654774984742127 \
  --projection-note "conditional arithmetic from the -642 B retrained prior on move 46's field; realized minus projected must be within float rounding" \
  --apply --commit
