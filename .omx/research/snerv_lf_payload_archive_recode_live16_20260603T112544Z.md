# SNeRV LF Payload Archive Recode

- schema: `snerv_lf_payload_archive_recode.v1`
- axis: `[receiver-proof:false-authority]`
- mode: `int64_lzma`
- source packet: `738516` bytes `375c8a36e51cc20ea636c1fd69860b2145843171743d9f1911fd2dd3c5205154`
- candidate packet: `738537` bytes `c7e05a4854fe13b38470159055581ad1e89aa78764660eded7587496f17773eb`
- packet byte delta: `21`
- LF byte delta: `21`
- LF planes exact: `True`
- unchanged sections exact: `{'metadata_payload': True, 'decoder_payload': True, 'step_map_packet': True}`
- receiver frame proof: `skipped_by_output_byte_guard`
- receiver contract satisfied: `True`

## Blockers
- `receiver_frame_streaming_proof_skipped_by_output_byte_guard`
- `not_packaged_as_contest_archive_zip`
- `full_video_scorer_replay_missing`
- `paired_contest_cpu_cuda_auth_eval_missing`
