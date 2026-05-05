# Shannon Floor Codex Sequential Resume - 2026-05-01T13:23Z

Operator requested sequential mode: limit subagents for now and proceed
without spawning more. Cloud Batch Jobs may still run when already queued or
when a single fast duplicate materially reduces wall-clock time with full
custody.

Current exact evidence:

- `exact_eval_alpha_crf63_grayscale_t4_20260501T125258Z` harvested exact T4
  CUDA: score `4.926778674301541`, bytes `458341`, PoseNet `1.34352684`,
  SegNet `0.00956173`, archive SHA-256
  `76bc850551269cad8bc32315959521cbd6f02c2e29f6c16e38f4c68ecd3f0eea`.
  Interpretation: scoped forensic negative for plain grayscale CRF63 mask
  replacement; byte cut is real but scorer geometry collapses. Not an
  Alpha-family kill.
- `exact_eval_direct_fd_m2_frontier_t4_20260501T1110Z` harvested exact T4 CUDA:
  score `1.0356355862798443`, bytes `686632`, PoseNet `0.00311747`, SegNet
  `0.00401872`, promotion-eligible under recorded gates.
- `exact_eval_owv3_0120_wave3_l40s_20260501T1322Z` submitted as a fast CUDA
  duplicate for archive SHA-256
  `06af57f770342cde494c37839200fdda79bdadd29826009e5e107ab296b4057a`, bytes
  `617410`. This is fast custody signal; T4 remains promotion-grade
  confirmation.
- `exact_eval_owv3_0119_wave3_t4_20260501T125143Z` harvested exact T4 CUDA:
  score `1.0025871157494655`, bytes `618443`, PoseNet `0.00357964`, SegNet
  `0.00401592`, archive SHA-256
  `75fc6c5eee02845f09296cda4854158d6663bb7533c2bf5f3c7a4a5b0638e802`.
  Strong total score, but non-promotable under strict adjudication because
  PoseNet is `1.050046494164029` relative to the OWV3 R7 T4 reference under a
  `1.002` gate. Treat as scoped forensic evidence and pose-protection target.
- `exact_eval_owv3_0120_wave3_rtxpro_20260501T1326Z` submitted as a second fast
  CUDA duplicate after the L40S duplicate remained pending and the T4 0120
  record showed a nonterminal status regression from Running back to Pending.

Live queue at last refresh:

- `exact_eval_owv3_0120_wave3_t4_20260501T130313Z`: status reconciliation
  issue; Lightning reported Running earlier then Pending later while cost
  increased.
- `component_response_alpha_primitive_pfp16_crf63_t4_20260501T130822Z`:
  Running on T4.
- `component_response_alpha_primitive_pfp16_crf63_l40s_20260501T131540Z`:
  Running on L40S at last refresh.
- `exact_eval_owv3_0120_wave3_l40s_20260501T1322Z`: Pending on L40S.
- `exact_eval_owv3_0120_wave3_rtxpro_20260501T1326Z`: Pending on RTX PRO.

Orthogonal optimization rule:

- Treat mask/video geometry, renderer weights, and pose stream as separate
  constrained streams. Freeze two streams while optimizing one, then build a
  new exact archive and run CUDA auth eval before composing deltas.
- Alpha mask compression must include PoseNet/SegNet geometry preservation:
  component-response-selected sparse repair, NeRV/INR geometry preservation,
  or pose regeneration with all bits charged in the archive.
- OWV3/direct-FD renderer work should keep mask geometry frozen until an exact
  stacked archive proves interaction.
- No additive score claims from separate deltas; stacks are their own archive.

Planning artifact:

- `experiments/alpha_lossy_repair_budget_planner.py` landed and a real run
  wrote `experiments/results/alpha_lossy_repair_budget_planner_20260501_r1/`
  with `91` empirical budget records and `12` non-promotable candidate specs.
  Wait for official CUDA component response before building any sparse-repair
  archive from those specs.

Update 2026-05-01T13:40Z:

- Current exact T4 frontier is `exact_eval_owv3_0120_wave3_t4_20260501T130313Z`:
  score `1.0021175309471926`, bytes `617410`, PoseNet `0.00356094`, SegNet
  `0.00402305`, SHA-256
  `06af57f770342cde494c37839200fdda79bdadd29826009e5e107ab296b4057a`,
  promotion eligible under recorded adjudication.
- Redundant OWV3 0120 L40S and RTX PRO duplicate exact evals were stopped after
  the T4 result was harvested.
- Alpha CRF63 primitive response was harvested from L40S and locally validated.
  It is a scoped forensic negative: nonzero points collapse to combined
  component response around `12.33` to `12.35`, PoseNet around `8.35` to
  `8.39`, SegNet around `0.0319`. Because the response is confounded by the
  CRF63 lossy base, it must not be used as a marginal sparse-repair selector.
- Built deterministic CRF60 and CRF62 grayscale replacement archives and staged
  them via manifest:
  `.omx/state/alpha_grayscale_crf60_62_exact_sweep_20260501T1336Z_manifest.json`.
  CRF60: `645623` bytes, SHA-256
  `f83e17f136afd651e866ca4b564ad28681c3db36f15fd911c0a07e45fe9ac8ae`.
  CRF62: `531222` bytes, SHA-256
  `90423f438273f4d5cae324023bb14fdc568f91950a6ddedf371067b8548b1dd8`.
- Submitted L40S exact CUDA threshold probes:
  `exact_eval_alpha_crf60_grayscale_l40s_20260501T1339Z` and
  `exact_eval_alpha_crf62_grayscale_l40s_20260501T1339Z`. If either is
  component-safe, rerun same archive bytes on T4/equivalent for promotion.
- One read-only xhigh reverse-engineering subagent (`Schrodinger`) reported
  that public top-leaderboard evidence points to scorer-aligned learned
  representations with charged tiny decoders. Use this as motivation only;
  local exact archive custody remains mandatory.
- Keep hard-pair/hard-zone/class/pose interval analysis, engineered
  corrections, adversarial/learned repair, and inverse-steg payload allocation
  in the search loop. They are valid only when charged inside the archive and
  verified by exact CUDA auth eval.
- Added durable execution synthesis:
  `.omx/research/shannon_floor_sub03_execution_plan_20260501_codex.md`.
  Core plan: sub-0.3 requires mask/geometry replacement or a learned
  topology-decoder class, not renderer polish alone; harvest CRF60/62 first,
  then either promote/repair a safe threshold archive or pivot immediately to
  NeRV/INR/SegMap/Q-FAITHFUL-class learned representation with deterministic
  charged decoder custody.
