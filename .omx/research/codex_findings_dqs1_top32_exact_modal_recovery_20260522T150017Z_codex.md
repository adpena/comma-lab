# Codex Findings - DQS1 Top32 Exact Recovery And Compact Packet

Generated: 2026-05-22T15:02:28Z
Agent: codex
Scope: decoder-q selective runtime, exact Modal auth eval recovery, compact pair encoding

## Authority

- Raw-u16 DQS1 top32 exact CPU is score authority on `[contest-CPU]`.
- Raw-u16 DQS1 top32 exact CUDA is score authority on `[contest-CUDA T4]`.
- Compact gap-ULEB DQS1 top32 exact CPU is score authority on `[contest-CPU]`.
- Compact gap-ULEB DQS1 top32 exact CUDA is score authority on `[contest-CUDA T4]`.
- MLX, macOS CPU advisory, locality controls, and feedback labels remain `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Exact Recovered Results

### Raw-u16 DQS1 top32

- Archive: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/archive.zip`
- Archive SHA-256: `3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`
- Archive bytes: `178592`
- DQS1 payload bytes: `75`
- Tail SHA-256: `52716aa2ff7ddd80c2fdf3c09fbf0c564391fad64201cd4063415576e83fe5df`
- `[contest-CPU]`: `0.1920502563025898`
  - SegNet distortion: `0.00055978`
  - PoseNet distortion: `0.00002943`
  - Rate component: `0.11891708215599484`
  - Result: `experiments/results/modal_auth_eval_cpu/dqs1_top32_selective_decoderq_paired_20260522T143930Z_cpu/modal_cpu_auth_eval_result.json`
- `[contest-CUDA T4]`: `0.2262117428884571`
  - SegNet distortion: `0.00066252`
  - PoseNet distortion: `0.00016845`
  - Rate component: `0.11891708215599484`
  - Result: `experiments/results/modal_auth_eval/dqs1_top32_selective_decoderq_paired_20260522T143930Z_cuda/modal_cuda_auth_eval_result.json`

Verdict: raw-u16 DQS1 top32 improved the prior FEC6 CPU frontier by
`1.0605785158157577e-06`, then was superseded by compact gap-ULEB. It is a
CUDA regression versus the PR106 CUDA anchor.

## Compact gap-ULEB DQS1 top32

- Commit: `fb14164d6` (`Harden DQS1 compact runtime packet`)
- Pair encoding: `sorted_gap_uleb`
- Archive: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/submission_dir/archive.zip`
- Archive SHA-256: `e12f5cfe93f9dbf624549466cda62d00a01e10bee8d1e0ea8a635af69247908a`
- Archive bytes: `178560`
- DQS1 payload bytes: `43`
- Tail SHA-256: `3095b29107f281e2cacb88b5a043069567a7f46c72e3e359010265ef55e9a58b`
- Locality control: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/locality_controls_top32_gap_uleb.json`
  - `locality_controls_passed=true`
  - Selected-frame mismatch count: `0`
  - Unselected-frame mismatch count: `0`
  - Raw SHA: `dee3ee3cf6c308f8dc2f11b3e611cc27ef75b3d452163bb1274e94603a268a00`
- `[contest-CUDA T4]`: `0.22619043540195719`
  - SegNet distortion: `0.00066252`
  - PoseNet distortion: `0.00016845`
  - Rate component: `0.11889577466949491`
  - Result: `experiments/results/modal_auth_eval/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cuda/modal_cuda_auth_eval_result.json`
- `[contest-CPU]`: `0.19202894881608987`
  - SegNet distortion: `0.00055978`
  - PoseNet distortion: `0.00002943`
  - Rate component: `0.11889577466949491`
  - Result: `experiments/results/modal_auth_eval_cpu/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cpu/modal_cpu_auth_eval_result.json`

Verdict: compact gap-ULEB removes `32` charged bytes without changing decoded
raw output in locality control. Both exact axes confirm identical scorer
components versus raw-u16 plus the lower rate component. This is the current
scanner-derived CPU frontier and a CUDA regression versus PR106.

## Bug Classes Extincted

- Planner/runtime split-brain: packet planner now emits a mode byte with low-nibble frame policy and high-nibble pair encoding; materializer and generated `inflate.py` both decode that same mode byte.
- Raw-u16 compatibility: `raw_u16` remains default and old payloads still parse.
- Compact canonicalization: `sorted_gap_uleb` enforces sorted unique pairs, positive nonzero gaps after the first pair, canonical ULEB encodings, no trailing bytes, and u16 bounds.
- Feedback false-authority gaps: local feedback now rejects exact auth payloads,
  requires local advisory axis/evidence, requires raw-output SHA custody, checks
  materialized pairs/frames against locality pairs/frames, and explicitly cannot
  rank, kill, promote, or suppress exact replay.
- Artifact tracking: compact plans, archives, locality outputs, and Modal result JSONs remain under ignored `experiments/results/*`; durable signal is summarized here and in refreshed Markdown surfaces.

## Verification

- `ruff check src/tac/optimization/decoder_q_selective_runtime_feedback.py src/tac/tests/test_decoder_q_selective_runtime_feedback.py tools/build_decoder_q_selective_runtime_feedback.py src/tac/optimization/decoder_q_selective_runtime_packet.py src/tac/optimization/decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py tools/plan_decoder_q_selective_runtime_packet.py`
- `.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_runtime_feedback.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py -q`
- `tools/run_decoder_q_selective_runtime_locality_controls.py` on compact gap-ULEB top32
- `tools/build_decoder_q_selective_runtime_feedback.py` on real raw-u16 top32 local advisory/locality/materialization artifacts
- `tools/recover_modal_auth_eval.py` on raw-u16 CPU/CUDA and compact CPU/CUDA

## Next Actions

1. Build the next compact selector family with gap-ULEB as default: top22,
   top32, top64, and marginal-add/drop Pareto sweep.
2. Feed raw-u16 and compact exact/advisory/locality rows into sign calibration
   with false-authority gates intact.
3. Exact-replay only rows whose locality and strict calibration gates pass, and
   keep CPU/CUDA axis deltas separate.
