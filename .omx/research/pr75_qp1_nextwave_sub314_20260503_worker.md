# PR75/QP1 Nextwave Sub314 Local Candidate Wave - 2026-05-03 Worker

Scope: local-only archive generation and custody after the PR75/QP1 P6 wave.
No remote GPU, Lightning, Modal, Vast.ai, exact CUDA eval, or dispatch claim was
created in this pass.

Write scope used:

- `experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/`
- `.omx/research/pr75_qp1_nextwave_sub314_20260503_worker.md`

Frontier anchor: C088 PR75 top40 P3 exact T4 score
`0.3155226919767294`, bytes `276386`, SHA-256
`9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`.

## Artifacts

- Ranked matrix:
  `experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/candidate_matrix.json`
- Validation summary:
  `experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/validation_summary.json`
- Action-builder matrix:
  `experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/action_compiler_candidates/candidate_matrix.json`
- CMG3/YF rebuild manifest:
  `experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/cmg3_yf_top1base_top0064/build_manifest.json`

## Local Builds

Built PR75 action-policy candidates with the existing deterministic
`experiments/build_pr75_tile_action_subset_candidates.py` surface only. Inputs
were the C063 fixed-slice T4 frontier archive, public PR75 archive, T4
component traces for C063/actions-only/top40/top25/top25-ampminus1, and the
existing L40S top40-ampminus2 calibration trace.

Built one representative CMG3/Yousfi-Fridrich large-byte-screen archive with
`experiments/build_cmg3_adaptive_runs_candidate.py`, using the existing
`c067_top1_yf_sparse_pair_frame_class_top0064` policy. The archive was rebuilt
twice under the output directory and stayed byte-identical.

## Ranked Dispatch Read

These are exact byte-custody local candidates only. They are not score claims.
Any future exact eval must first use the lane-claim protocol.

| rank | candidate | bytes | SHA-256 | planning band / reason |
| ---: | --- | ---: | --- | --- |
| 1 | `c067_pr75_actions_positive_poseharm_ampminus1_p6` | `276389` | `244c366a7d07ff185091dbbcf7ecb1e0308d11d9e7467bc2ae2eb2f8b6bd0a6a` | Center `0.31547501271162115`; best T4-trace component move. Keeps positive actions, shrinks pose-harm records, changes `31` action ids. |
| 2 | `c067_pr75_actions_pose_safe_positive_ampminus1_p6` | `276317` | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | Center `0.3155074207043554`; `-69` bytes vs C088, pose-safe subset, `28` changed action ids. |
| 3 | `c067_pr75_actions_lag_eval_top40_signedposemix1_p6` | `276416` | `0486e9f2adb8f883160599ed3bccd371b2c1ffdf23585edbcd4308435451e9f4` | Center `0.3154943130669148`; nonlinear signed residual escape, but has duplicate pair/tile residuals and higher interaction risk. |
| 4 | `c067_pr75_actions_top67_ampfit_p6` | `276421` | `ef790959626ceb2736ece906938e671708fd5357fcd4db8944018e301789a97e` | Center `0.3155047719372756`; full-action amplitude-fit probe. |
| 5 | `c067_pr75_actions_all_ampminus1_p6` | `276424` | `09a4896554447b94e430a7b07814cb017a999e7d5267c9021455f95127be51a7` | Center `0.315506769514135`; full-body uniform amplitude shrink. |
| 6 | `c067_pr75_actions_poseharm_ampminus1_p6` | `276425` | `002ef3335a8d66b10240024b670d0a143e278d38ec939c20b689d33848f2fe9c` | Center `0.3155074353730881`; shrinks only negative-pose-contribution action records. |
| 7 | `c067_pr75_actions_positive_poseharm_ampminus1_p3` | `276447` | `2ea3bc61262a3e0e225458ec165483ce8f4d61d4b12ae26d4bd0f50257a40e92` | P3 fallback for rank 1 if active P6 exact wave exposes runtime drift. |
| 8 | `c067_pr75_actions_top67_custompose125_p5` | `276706` | `d5b3061cfb31fdaf954cb2b4f8ee3db3d1a7565196495fa15e99b3d0102d8c3f` | Charged custom-dictionary nonlinear probe; rate-expensive and lower priority. |
| blocked | `cmg3_yf_top1base_top0064` | `129257` | `97d0bbb02ef1c19f0db907054e3af94f2d25bf5a1264d2c520caa46bbaef5663` | Saves `147129` bytes vs C088, but same-family exact CUDA CMG3/CMG3A evidence collapsed badly. Do not dispatch without a new geometry-escape proof. |

## Verification

- JSON validation passed for `candidate_matrix.json` and
  `validation_summary.json`.
- All nine matrix archives have exactly one ZIP member, `p`.
- All archive stat sizes and SHA-256 digests match their manifests.
- The robust payload parser accepts all nine matrix entries. Action candidates
  parse as PR75 P3/P5/P6 payloads; the CMG3/YF archive parses as a Brotli-wrapped
  `RPK1` payload containing `renderer.bin`, `masks.cmg3`, and
  `optimized_poses.bin`.
- CMG3/YF deterministic rebuild check:
  `97d0bbb02ef1c19f0db907054e3af94f2d25bf5a1264d2c520caa46bbaef5663` before
  and after rebuild.
- `git diff --check -- experiments/results/pr75_qp1_nextwave_sub314_20260503_worker .omx/research/pr75_qp1_nextwave_sub314_20260503_worker.md` passed.

## Decision

No local deterministic archive in this pass is both exact-eval-ready and
plausibly sub-0.314 on current evidence. The actionable next wave is therefore
component-improvement probes that may beat C088, led by
`positive_poseharm_ampminus1_p6`, while the only archive with sub-0.314
rate-only headroom remains blocked by exact-negative CMG3/YF geometry evidence.
