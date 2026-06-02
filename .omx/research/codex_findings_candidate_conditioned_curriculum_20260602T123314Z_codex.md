# Codex Findings: Candidate-Conditioned NeRV Curriculum

UTC: 2026-06-02T12:33:14Z

Verdict: modelsize candidates now control more than launch geometry. The
compact runner binds selected HiNeRV/SNeRV candidates to curriculum/QAT/scorer
requirements and byte-feedback records. This is still false authority; exact
score movement requires full-video MLX prefilter, local CPU replay, and exact
CPU/CUDA only for true local winners.

What landed:

- `tac.analysis.nerv_candidate_curriculum` emits candidate-conditioned training
  plans for HiNeRV and SNeRV.
- HiNeRV candidate plans require PR95-style 8-stage minimum scheduling,
  real SegNet and PoseNet teachers, joint P18/P19 recon weights, coder-aware
  QAT, and trained archive-byte feedback.
- Low-bit HiNeRV modelsize candidates automatically enable coder-aware QAT and
  align quant bits to the candidate decoder codec. The runner still does not
  invent scorer loss weights; missing real scorer teachers remain blockers.
- SNeRV candidate plans carry LF/HF receiver grammar controls: levels,
  LF bits-per-coeff, waterfill step-map precision, decoder payload codec, and
  SNAR1 measured byte feedback.
- The candidate curriculum plan is recorded in runner reports. A sanitized copy
  is embedded in MLX training artifact metadata so canonical authority fields
  do not leak into substrate metadata.
- `nerv_stack_synergy_audit` now lists the curriculum planner as a shared
  synergy surface.

Smoke anchors:

- Focused tests:
  `pytest src/tac/tests/test_nerv_candidate_curriculum.py
  src/tac/tests/test_nerv_modelsize_budget.py
  src/tac/tests/test_nerv_stack_synergy_audit.py
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q`
  -> 54 passed.
- Plan report:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_nerv_candidate_curriculum_plan_20260602T123245Z/compact_renderer_mlx_spine_runner_report.json`.
- Manual HiNeRV smoke:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hinerv_candidate_curriculum_manual_smoke_20260602T123245Z/compact_renderer_mlx_spine_runner_report.json`.
- Auto HiNeRV smoke:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hinerv_candidate_curriculum_auto_smoke_20260602T123304Z/compact_renderer_mlx_spine_runner_report.json`.

Key auto-smoke signal:

- Auto selected `hinerv_np600_ld12_ed24_dc32_int4_mixed_ceil178000`.
- Launch codec became `int4_mixed`.
- Candidate curriculum enabled coder-aware QAT.
- Candidate curriculum quant bits became `4`.
- Byte feedback was ready from measured archive bytes.
- Sanitized artifact metadata did not carry nested `score_claim`.

Remaining blocker:

Candidate-conditioned curriculum is now wired for HiNeRV and advisory SNeRV,
but replay/posterior learning is not closed. Next step: write measured
candidate byte feedback plus local MLX/CPU component deltas into a reusable
posterior surface, then let candidate curriculum schedules update from those
posterior rows instead of static rules.
