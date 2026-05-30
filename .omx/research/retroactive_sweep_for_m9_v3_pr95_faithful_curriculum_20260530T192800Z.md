# Retroactive sweep for M9-v3 PR95-faithful 8-stage curriculum scaffold (Option A) 2026-05-30T19:28:00Z

Per CLAUDE.md "EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" Catalog #348
self-protection.

## 4-field contract

### 1. Bug-class symptom signature

The Option A optimizer stack research memo (commit `118ddb1a4`) landed
2026-05-30 with the explicit observation that the canonical PR95 8-stage
Muon+AdamW MLX kernel + per-stage descriptors + canonical apply step
ALREADY existed in the repo (at `src/tac/local_acceleration/pr95_hnerv_mlx.py`
and `src/tac/optimization/optimizer_scheduler_registry.py`) but had NO
production caller wiring them into the canonical MLX score-aware adapter
at `src/tac/substrates/_shared/mlx_score_aware/adapter.py:150`. The
canonical wiring point per the research memo was the bare AdamW line at
`adapter.py:150`. Per CLAUDE.md "Subagent coherence-by-default" the
orphan-signal-at-canonical-wiring-point bug class applies: a fully-
implemented canonical primitive that no caller wires into the canonical
substrate adapter is by definition non-functional from the substrate
trainer's perspective.

THIS landing (M9-v3) closes the orphan-signal at the adapter wiring point.
Sister waves close the per-substrate-trainer wiring (op-routable #1).

### 2. Pre-fix window

`118ddb1a4` (optimizer research memo landing 2026-05-30) → THIS commit.
The pre-fix window is approximately 1 day; the research memo IS the
explicit gap identification + Option A scaffold recommendation.

### 3. Historical KILL/DEFER/FALSIFY search results

**0 historical findings invalidated** by THIS landing because the
canonical PR95 MLX Muon+AdamW kernel + per-stage descriptors WERE
already empirically validated by sister landings:

- `tac.local_acceleration.pr95_hnerv_mlx.apply_pr95_mlx_optimizer_step` —
  empirically validated via `src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
  (16 test cases covering Muon+AdamW partition + per-stage descriptor lookup).
- `tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx` —
  empirically validated via PR95 timing smoke at sister
  `src/tac/tests/test_run_pr95_mlx_timing_smoke.py`.
- `tac.optimization.optimizer_scheduler_registry.default_optimizer_scheduler_descriptors` —
  empirically validated via sister `src/tac/tests/test_pr95_mlx_stage_7_sigma_sweep_curriculum_build.py`
  + `src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`.

No historical kill / defer / falsify verdict referenced ANY of these
canonical primitives because (a) they are first-class canonical helpers
landed as positive infrastructure; (b) sister tests already empirically
validate them; (c) the orphan-signal was at the ADAPTER CALLER surface,
NOT at the canonical-helper IMPLEMENTATION surface.

### 4. RE-EVAL-priority assignment per affected finding

**N/A** — this is a NEW scaffold landing (Option A MINIMUM-VIABLE) NOT
a fix that invalidates historical verdicts. The op-routables surfaced in
the landing memo are:

1. SISTER WAVE — RECOMMENDED-LAND-NEXT to wire `pr95_faithful_curriculum_enabled=True`
   into substrate trainers (z6 / atw_v2 / hnerv-family) per Catalog #270
   dispatch optimization protocol.
2. PAIRED-CUDA RATIFICATION DISPATCH WAVE at canonical 29,650-epoch budget
   per Catalog #246 once a contest-faithful substrate routes through the
   factory.
3. PHASE 2 SISTER LANDING wiring per-stage `cat_sigma` + `cat_lambda` +
   `qat_active` + `loss_family` through to the `score_aware_loss` surface.

All 3 op-routables are FORWARD-LOOKING extensions; no historical anchor
needs re-evaluation.

## Sister-extinction architecture per Catalog #299 quota brake

NO new Catalog # claimed (current 382 well under 400 quota; THIS landing
extends the existing canonical helper + adapter surface; no new STRICT
preflight gate needed because the canonical contract is enforced via the
opt-in kwarg semantics + the 26-test suite verifying NO FAKE per
CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable).

## Verbatim per Catalog #348 acceptance contract

- bug-class symptom signature: orphan-signal-at-canonical-wiring-point
  (PR95 MLX Muon+AdamW kernel + per-stage descriptors landed but no
  caller wired them into MlxScoreAwareAdapter at adapter.py:150).
- pre-fix window: 118ddb1a4 (2026-05-30 research memo landing) → THIS
  commit (1-day window).
- historical KILL/DEFER/FALSIFY search results: 0 historical findings
  invalidated; canonical primitives already empirically validated by
  sister tests.
- RE-EVAL-priority assignment: N/A (NEW scaffold landing, not a fix).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
