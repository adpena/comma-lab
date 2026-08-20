# SNeRV LF Payload Archive Recode

- schema: `snerv_lf_payload_archive_recode.v1`
- axis: `[receiver-proof:false-authority]`
- mode: `int64_lzma`
- source packet: `2979` bytes `a4666b4973553a6276a6502c524278cb01f271ed5b3eb31a99dc77323b902210`
- candidate packet: `3065` bytes `8c8f203ec7daa155888e18c32e75b923dbbba319a9652ba664b25677b4872182`
- packet byte delta: `86`
- LF byte delta: `86`
- LF planes exact: `True`
- unchanged sections exact: `{'metadata_payload': True, 'decoder_payload': True, 'step_map_packet': True}`
- receiver frame proof: `skipped_by_output_byte_guard`
- receiver contract satisfied: `True`

## Blockers
- `receiver_frame_streaming_proof_skipped_by_output_byte_guard`
- `not_packaged_as_contest_archive_zip`
- `full_video_scorer_replay_missing`
- `paired_contest_cpu_cuda_auth_eval_missing`
