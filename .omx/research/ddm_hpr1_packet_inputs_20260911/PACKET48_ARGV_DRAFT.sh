# DRAFT — MAIN fills <HARVEST_JSON> (run1/MODAL_REMOTE_RESULT.json) after harvest; complete the seal first; verify S from components.
# completion: PYTHONPATH=<repo> tools/make_candidate_seal.py --complete-first-fire-intent .omx/research/ddm_hpr1_20260911/v2/PREFIRE_INTENT_ddm_hpr1_comp_even_on_refit.json \
#   --first-measurement-authorization .omx/research/ddm_hpr1_20260911/v2/FIRST_MEASUREMENT_AUTHORIZATION.json \
#   --candidate-t4-receipt /Volumes/VertigoDataTier/pact/ddm_hpr1_first_measurement/run1/MODAL_REMOTE_RESULT.json \
#   --out .omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json
# mirror: tools/write_completed_seal_anchor_mirror.py --seal <SEAL> --lane-id ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911 --label ddm_hpr1_first_measurement_t4_run1_20260911
.venv/bin/python tools/pointer_move_packet.py \
  --harvest <HARVEST_JSON> \
  --seal .omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json \
  --archive /Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime/archive.zip \
  --lane-id ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911 \
  --lane-name "ddm_hpr1 even rounding on the refit HPAC prior (composition of moves 46 and 47's levers) on move 47 (-248 B)" \
  --call-id fc-01M29A1CWQBBDT4XXTQ1K7TG88 \
  --move-number 48 \
  --headline "pointer move 48: S <EXACT_S> @ 179,111 B [contest-CUDA T4 n600] — hpr1 even rounding on the refit HPAC prior (-248 B vs move 47; decoded field and raw byte-identical), first-measurement chain with its own measured t4_direct leg" \
  --mechanism-file .omx/research/ddm_hpr1_packet_inputs_20260911/mechanism_move48.md \
  --not-claimed-file .omx/research/ddm_hpr1_packet_inputs_20260911/not_claimed_move48.md \
  --next-file .omx/research/ddm_hpr1_packet_inputs_20260911/next_move48.md \
  --equations-leg "S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 47 (decoded field and cold n600 raw byte-identical) and B 179,359 → 179,111: ΔS = 25·(−248)/37,545,489 = −1.6513302e-4 exactly; projected S 0.13638261682704697" \
  --projected-score 0.13638261682704697 \
  --projection-note "conditional arithmetic from the -248 B rounding of the refit prior on move 47's field; realized minus projected must be within float rounding" \
  --apply --commit
