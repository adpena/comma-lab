# PR75 Action Dictionary V2 Worker - 2026-05-03

## Scope

Worker A local-only PR75 tile-action/action-dictionary v2 pass. No remote GPU,
Lightning, Modal, Vast.ai, or lane-claim edits were performed. Outputs are
deterministic archive candidates and byte/trace planning artifacts only until
exact CUDA auth eval measures the exact bytes.

## Inputs

- Current A++ frontier: `c067_pr75_qp1_top40_p6`, score
  `0.3154707273953505`, bytes `276342`, SHA-256
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`.
- C067 source archive/trace:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z`.
- PR75 source archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip`.
- Action trace:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z/component_trace.json`.
- Frontier interaction trace:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/component_trace.json`.

## Builder Additions

- Added explicit one-based top-N drop/add selection policies:
  `top<N>_drop<R[_R...]>[_add<R[_R...]>]`.
- Added P5 charged custom-dictionary wildcard transforms:
  `_wilddiramp4`, `_wilddiramp6`, `_wilddiramp8`, and `_wilddirmean`.
- Existing duplicate/no-op guards remain active; P6 remains fixed-dictionary
  pair-delta-varint, while wildcard forms require the already-supported P5
  custom dictionary parser.

## Artifacts

- Matrix:
  `experiments/results/pr75_action_dict_v2_worker_20260503/candidate_matrix.json`.
- Decoded stream closure:
  `experiments/results/pr75_action_dict_v2_worker_20260503/decoded_stream_closure.json`.
- Recommendation/template summary:
  `experiments/results/pr75_action_dict_v2_worker_20260503/candidate_recommendations.json`.

All final matrix candidates unpack through
`submissions/robust_current/unpack_renderer_payload.py`. Closure checks report
`mask_renderer_match_c067=true`, `pose_qp1_matches_current_frontier=true`, and
`seg_tile_actions_matches_manifest_runtime_raw=true` for each candidate.

## Top Local Candidates

| priority | candidate | bytes | SHA-256 | decoded closure | recommendation |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `c067_pr75_actions_top40_drop40_add41_p6/archive.zip` | `276341` | `209fe4b08be802a2d960ae10aa801af32598fc9ce495016626d2f2a0f22d5266` | closed | local CUDA optional; no remote dispatch |
| 2 | `c067_pr75_actions_top40_drop39_add41_p6/archive.zip` | `276342` | `2fcc1982247e5e6ab570820b5ee7e553cd53a2e26dd0dc366773d5c51d9bbd79` | closed | local CUDA optional; no remote dispatch |
| 3 | `c067_pr75_actions_top40_drop39_40_add41_42_p6/archive.zip` | `276342` | `cfbcdb9e413657616b0f915c742740e4f879b0b2dcdf82b53d7819f344f3a6fc` | closed | local CUDA optional; no remote dispatch |
| 4 | `c067_pr75_actions_top40_drop40_p6/archive.zip` | `276340` | `3ddae1da96ae3f2d50335b88c80abf564f64729760f40dc7fb7ee2a05266ad46` | closed | local CUDA optional; no remote dispatch |
| 5 | `c067_pr75_actions_top40_drop39_p6/archive.zip` | `276340` | `4adc1102af251bfdc046bdbc264e4400ed58bb369e490e3fe5e440dbf4d679ba` | closed | local CUDA optional; no remote dispatch |

Exact-eval template for any row:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/pr75_action_dict_v2_worker_20260503/exact_eval_work_<candidate>
```

## Read

The closest P6 drop/swap candidates are micro-moves around the current
frontier. The best trace-ranked candidate is 1 byte smaller than the current
frontier and has a planning-only estimated score `0.31546827267490696`, but
the expected movement is far below `1e-4` and prior PR75 action-only traces
have reordered under exact T4. These should not consume remote dispatch
capacity unless an operator explicitly chooses a cheap exact slot.

The P5 wildcard dictionary screens are parser-closed but rate-heavy:
`top40_wilddiramp6_p5` is `276444` bytes, `top40_wilddiramp8_p5` is `276445`
bytes, `top40_wilddirmean_p5` is `276481` bytes, and
`top25_wilddirmean_p5` is `276421` bytes. Without component evidence that the
custom deltas buy a large nonlinear SegNet/PoseNet improvement, they are
`do_not_dispatch`.

## Verification

- `.venv/bin/python -m py_compile experiments/build_pr75_tile_action_subset_candidates.py src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`
