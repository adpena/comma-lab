# PR100-PR107 reproduction and deconstruction ledger

Created: `2026-05-08T10:55:00Z`

Evidence grade: `[empirical:local-custody-inventory]`; no score claim.

| PR | name | leaderboard | local replay | bytes | members | same-archive scored/total evals | missing proof count | first blockers |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 100 | hnerv_lc_v2 | 0.195 | cuda:0.228269 | 178981 | 1 | 1/1 | 5 | compress_entrypoint_missing, compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required |
| 101 | hnerv_ft_microcodec | 0.193 | cuda:0.226353 | 178258 | 1 | 1/1 | 5 | compress_entrypoint_missing, compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required |
| 102 | hnerv_lc_v2_scale095_rplus1 | 0.195 | cuda:0.228394 | 178981 | 1 | 1/2 | 2 | compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required |
| 103 | hnerv_lc_ac | 0.195 | cuda:0.227765 | 178223 | 1 | 1/2 | 4 | compress_entrypoint_missing, compress_to_archive_1to1_reproduction_required, same_archive_structured_exact_eval_json_missing |
| 104 | qhnerv_ft_best | 0.231 | no_same_archive_score_to_compare | 178637 | 1 | 0/0 | 6 | compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required, exact_cuda_replay_missing |
| 105 | kitchen_sink | 0.198 | cuda:0.230437 | 177857 | 1 | 1/2 | 3 | compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required, research_note_missing |
| 106 | belt_and_suspenders | 0.209 | cuda:0.209457 | 186239 | 1 | 1/26 | 2 | compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required |
| 107 | apogee | 0.229 | cuda:0.229331 | 178392 | 1 | 1/1 | 5 | compress_entrypoint_missing, compress_to_archive_1to1_reproduction_required, decode_reencode_parity_proof_required |

## Required next proof

Every row must advance from member-prefix inventory to byte-level grammar, decode/re-encode parity, compress-to-archive reproduction, and exact CUDA structured JSON before it is used as a promoted stack atom.

Public leaderboard drift is tracked per replay mode. For PR100+ public submissions, GitHub comments show CUDA and CPU evals can differ materially; do not compare a local CUDA replay to a CPU leaderboard row without labeling the device and runtime tree.
