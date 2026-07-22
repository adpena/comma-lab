---
schema: codex_findings.v1
task: 603
feeds_task: 613
review_round: 1
reviewer: codex:gpt-5.6-sol
research_only: true
main_landing_review_required: true
---

# Round-1 findings — DDM v7 solved-plane tolerance waterfill

## Disposition

`PASS_AFTER_THREE_FIXES`, research-only. Ruff, pycompile, and 45 focused tests pass. The measured
formulation verdict is valid after report and resume repair; no contest score or promotion is valid.

## Finding 1 — handwritten waterfill was not byte-monotone

- Severity: high, decision-law bug.
- Observed: round-0 compared `q16_all -> waterfill_balanced` even though the latter was cheaper,
  emitting negative added bytes and a meaningless marginal.
- Fix: `ea058a361b` derives the route from the measured discrete Pareto envelope: exact byte sort,
  strict distortion-record retention, and positive-byte/positive-gain assertions.
- Regression: dominated states are excluded and every admitted marginal has positive byte and
  distortion deltas.

The invalid round-0 report remains preserved with SHA-256
`217bd957e2d1e24ab623f6bfc8464e4c135fadc3f07fb0f86070448d77511062`; it must not be consumed.

## Finding 2 — scorer-axis label leaked from v6

- Severity: high, evidence-custody bug.
- Observed: the v7 bridge measured both SegNet and PoseNet, but one candidate verdict scope used the
  older `[macOS-CPU frozen-SegNet advisory]` constant.
- Fix: `6d45f35a0c` centralizes the v7 scope, validates checkpoint bridge axis exactly, and normalizes
  resumed reporting only after that validation. v6 retains its separate SegNet-only label.
- Regression: the v7 joint label must be present and the v6 label absent.

## Finding 3 — volatile storage telemetry broke sealed replay

- Severity: high, P0 resumability bug.
- Observed: after n256 completed, the same argv reconstructed a receipt with a new
  `observed_free_bytes` value and correctly refused to overwrite the sealed receipt.
- Fix: `e16e25b025` validates and returns a completed receipt before dynamic preflight. Validation
  binds the typed config, DSL hash, committed producer, candidate-table hash, every candidate
  archive/checkpoint, and all 42 rung frames.
- Proof: final committed receipts regenerate from candidate stages in 7.85/25.62 seconds and sealed
  validation completes in 0.81/1.05 seconds for n64/n256.

## Re-derived measurement invariants

- Exact receivers match the solved-plane windows bit-for-bit.
- Only exact rows satisfy `d_seg <= 0.00116`.
- Exact archive bytes are 43,112,153 and 171,332,654, versus a 200,000-byte falsifier.
- Dominant rate homes are Undrivable, MyCar, and Road; dominant remaining exact d_seg strata are
  Lane and boundary.
- Every point retains joint d_seg/d_pose, six-stream bytes, class/topology/margin decomposition,
  and `score_claim=false`.
- n600 and contest CPU/CUDA remain unmeasured; pointer unchanged.

Canonical #603 remains 8/19 until MAIN reviews and registers the append-only draft.
