# IAS1 Exact-Auth Calibrated Inverse Action Findings

## Scope

Codex converted the IAS1 runtime-parity top4 exact CPU/CUDA result pair into
planner-consumable inverse-steganalysis calibration signal. This is a
measured-config update only; it does not retire inverse-scorer or
inverse-steganalysis as a family.

## Exact Review Packets

- CPU packet:
  `.omx/research/ias1_runtime_parity_top4_exact_cpu_result_review_20260523_codex.json`
- CUDA packet:
  `.omx/research/ias1_runtime_parity_top4_exact_cuda_result_review_20260523_codex.json`
- Shared archive SHA:
  `2d0850789483e17c7ee68ae8bfe1e33489d1981416f71266cf8a66b19a87e549`
- Shared archive bytes: `181232`
- Shared runtime content tree SHA:
  `20e218243304dd2dd2ff44717ca96467a2f3acaf45f9dbaf50b2a1333087ae3f`
- CPU score vs current CPU frontier:
  `0.19380912393883232 - 0.19202828295713675 = +0.00178084098169557`
- CUDA score vs current CUDA frontier:
  `0.2279696105246996 - 0.20533002902019143 = +0.022639581504508166`
- Combined paired exact-auth regression penalty:
  `0.024420422463531932`

## Code Landed

- `tac.optimization.inverse_steganalysis_acquisition` now supports
  `paired_exact_auth_calibration_observations_from_review_packets(...)`.
- The adapter requires exactly one `contest_cpu` and one `contest_cuda`
  `tac_result_review_packet_v1`, shared archive SHA/bytes/sample count,
  shared runtime content tree SHA, score recomputation match, terminal dispatch
  claims, non-indeterminate status, and no family-retirement flags.
- The emitted observation keeps `axis="[paired exact-auth calibration]"`
  rather than a contest axis, so the existing contest-axis rejection guard stays
  intact.
- Exact calibration observations now win same-candidate observation selection
  over optimistic local/proxy rows.
- `tools/build_inverse_steganalysis_action_functional.py` accepts
  `--exact-auth-calibration-packet` and threads paired packet metadata into the
  action functional observation feedback.

## Calibration Artifact

Generated local artifact:

`experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_steganalysis_action_functional_ias1_exact_calibrated_20260523_codex.json`

Result:

- schema: `inverse_steganalysis_discrete_action_functional.v1`
- cells: `64`
- selected cells: `0`
- best observation kind: `paired_exact_auth_calibration`
- expected score gain for the measured IAS1 config: `0.0`
- `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py tools/build_inverse_steganalysis_action_functional.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`

## Next

- Feed future exact review packets through the same paired calibration adapter
  before promoting inverse-action water buckets to materialization queues.
- Extend the materializer planner only after a second measured-config pair
  proves which cells or bundles should inherit calibration penalties.
