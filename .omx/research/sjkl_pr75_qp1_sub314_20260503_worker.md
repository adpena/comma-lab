# SJ-KL PR75/QP1/C088 sub-0.314 screen - 2026-05-03

Scope: local-only deterministic SJ-KL residual stacking screen against the
current C088 PR75/QP1 frontier. No remote GPU job was dispatched.

## Parent

- Label: `C088_PR75_top40_p3_T4`.
- Archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/archive.zip`.
- Bytes: `276386`.
- SHA-256: `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a`.
- Exact T4 score recomputed from components: `0.3155226919767294`.
- Score source:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/contest_auth_eval.json`.

## Built Candidate Archives

Generated under:
`experiments/results/sjkl_pr75_qp1_sub314_20260503_worker/`.

Primary artifact:
`experiments/results/sjkl_pr75_qp1_sub314_20260503_worker/candidate_matrix.json`.

Validation artifact:
`experiments/results/sjkl_pr75_qp1_sub314_20260503_worker/validation_summary.json`.

The existing `experiments/build_sjkl_c067_archive.py` builder supports the
needed PR75 stack path through `--archive-layout top_level_sibling`: it
preserves the parent `p` payload byte-for-byte and adds charged top-level
`sjkl.bin`. Fourteen archives were built from seven existing SJ-KL payloads
using `stored` and `deflated` top-level `sjkl.bin` variants.

Top byte candidates:

| candidate | bytes | SHA-256 | delta vs C088 | formula-only score if components unchanged |
| --- | ---: | --- | ---: | ---: |
| `q6_minrpk1_stored` | `276728` | `f97e1dd1ef7e888b8ff72cbb94a534adc5403a5d677086de316c34930a7138d7` | `+342` | `0.31575041573869717` |
| `q6_minrpk1_deflated` | `276733` | `7ef4e7493678be6d9b94f74aa7851e91cd3c58b77bb3a5a38ee415708256a092` | `+347` | `0.31575374503346276` |
| `consensus24_g0046875_deflated` | `276773` | `db0196f6a07c9791f43868763d7bcbb5f767b2ab4e69908d57d4fdf0b982c7c5` | `+387` | `0.31578037939158765` |
| `consensus32_g0046875_deflated` | `276788` | `157e1a5393330d731e7de971b1521bbb2497704bf8365c67b85d82a97f89518c` | `+402` | `0.3157903672758845` |

## Changed-Payload And No-Op Proof

- Every candidate ZIP contains exactly `p` and `sjkl.bin`.
- Parent `p` SHA-256 is preserved byte-for-byte:
  `93250610cbb819241d1356554149738af7de55e83a006396883fc7a18f51eff5`.
- Every candidate `sjkl.bin` SHA-256 matches the selected source payload.
- Builder manifests record `sidecars_required=false` and
  `score_affecting_payload_charged_in_archive=true`.
- Runtime static proof found the strict-apply hooks in
  `submissions/robust_current/inflate_renderer.py`, including
  `SJKL_REQUIRE_APPLIED`, `_apply_sjkl_residual_to_pairs`, and
  `_finalize_sjkl_application_contract`.
- Prior exact/diagnostic CUDA logs show the payload class is not a silent no-op
  on C067. The strongest evidence is `q6_minrpk1` exact T4:
  `experiments/results/lightning_batch/exact_eval_sjkl_c067_q6_sibling_t4_20260502T2357Z/auth_eval.log`
  loaded 16 pairs and passed `SJ-KL strict contract`.

PR75/C088 component response remains unmeasured. The matrix exact-CUDA
templates therefore set `SJKL_REQUIRE_APPLIED=1`; a promotable run must fail
closed if the charged residual does not apply.

## Decision

No built SJ-KL stack warrants immediate T4 exact eval as a sub-0.314 candidate.
The best byte candidate, `q6_minrpk1_stored`, already adds `0.0002277237619677826`
rate score and would need about `0.0017504157386971642` component-score gain to
cross `0.314`. Existing exact T4 C067 SJ-KL evidence showed only about
`0.00000284` component-score gain and scored worse overall:
`0.3158419419767293` at `276556` bytes.

The archives are valid as future strict-apply controls or low-priority
component-response probes, but not as immediate contest-faithful sub-0.314
dispatches.

## Verification

- Built 14 deterministic archives with
  `experiments/build_sjkl_c067_archive.py --archive-layout top_level_sibling`.
- JSON validation:
  `jq -e '.schema == "sjkl_pr75_qp1_sub314_candidate_matrix_v1" and (.candidates|length)==14 and (.remote_gpu_jobs_dispatched==false)' experiments/results/sjkl_pr75_qp1_sub314_20260503_worker/candidate_matrix.json`
  returned `true`.
- Archive custody checks in
  `experiments/results/sjkl_pr75_qp1_sub314_20260503_worker/validation_summary.json`
  verify archive bytes/SHA-256, top-level member order, parent `p`
  preservation, and `sjkl.bin` payload SHA-256 for all 14 candidates.
- Focused tests:
  `.venv/bin/python -m pytest src/tac/tests/test_build_sjkl_c067_archive.py src/tac/tests/test_inflate_renderer_sjkl_runtime.py`
  passed: `15 passed`.

