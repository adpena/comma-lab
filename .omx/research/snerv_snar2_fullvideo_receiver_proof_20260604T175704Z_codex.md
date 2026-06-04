# SNeRV SNAR2 Full-Video Receiver Proof - 2026-06-04T17:57:04Z

Axis: `[receiver-safe false-authority]`

This records the two SSD-backed full-video receiver package proofs run after
the SNAR2 binary-header implementation landed locally. Both artifacts decode
through the generated receiver runtime and preserve the full-video receiver
output, but neither is a scorer or exact CPU/CUDA authority claim.

## Result

- Wire format: `snar2`
- Operation: `snar2_fixed_binary_header_receiver_metadata_prune`
- Source packet bytes: `1485285`
- Candidate packet bytes: `139123`
- Packet byte delta: `-1346162`
- Source header bytes: `1346233`
- Candidate header bytes: `71`
- Candidate runtime archive bytes: `155530`
- Full-video receiver contract: `true`
- Runtime consumption proof: `true`
- Score claim: `false`
- Ready for exact eval dispatch: `false`

## Artifacts

| Candidate | Report SHA-256 | Packet SHA-256 | Archive bytes | Archive SHA-256 |
| --- | --- | --- | ---: | --- |
| `native_rate_aware_training_snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc11e0_p1_mfu1-2-4_hfr0_t0_tmhaar1_adspectra_oms0p05_int8_symm` | `1041d994ab501c31cec3a6fc04fa57f00c4518b43c468b5ad504bece7168b0ae` | `c64d2cccd793c976eb8bd6e432754677b80721a2a930823afdfc55d806d16c30` | `155530` | `da6200c72506092677e8dd955d3bf8f5287ded10f6c87da03678a8b6de75ef48` |
| `native_rate_aware_training_snerv_np600_haar_lv5_lfb1p5_stepb0p5_fc9e0_p1_mfu1-2-4_hfr0_t1_adbase_int2_symmetric_ceil178000_los` | `10e9935cd0fec43b20201423ab58e8d144f1cc3ba03b5f0bc42ab18e435fa585` | `c64d2cccd793c976eb8bd6e432754677b80721a2a930823afdfc55d806d16c30` | `155530` | `da6200c72506092677e8dd955d3bf8f5287ded10f6c87da03678a8b6de75ef48` |

Full reports remain on SSD under each candidate's
`snar2_header_minimized_fullvideo/snerv_snar_header_minimization.json` path.
The compact machine-readable mirror is
`.omx/research/snerv_snar2_fullvideo_receiver_proof_20260604T175704Z_codex.json`.

## Remaining Blockers

- `snerv_snar_header_minimization_false_authority`
- `full_video_scorer_replay_missing`
- `paired_contest_cpu_cuda_auth_eval_missing`
