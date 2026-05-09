# HALT GHA promotion pending comprehensive adversarial review (2026-05-09)

<!-- generated_at: 2026-05-09T09:30:00Z, from_state_hash: operator_review_halt -->

## Operator directive (verbatim, 2026-05-09)

> "we need another round of comprehensive aggressive full adversarial grand council bug hunter and rigor review of all [the roadmap items] prior to more GPU spend"

## Implication for in-flight a8522fca (constrained coord search)

You are mid-execution on:
- **Phase 1**: generate 64 (or 343) candidate variants — CONTINUE (no spend)
- **Phase 2**: M5 Max sweep all candidates — CONTINUE ($0; M5 Max is local)
- **Phase 3**: GHA dispatch top-N winners — **HALT until adversarial review completes**

The M5 Max phase is fine; only the GHA promotion phase is gated on review.

## What to do

1. Complete Phase 1 + Phase 2 (M5 Max sweep) per original plan
2. Write Phase 2 results to `experiments/results/constrained_coord_search_m5max_<ts>/results.jsonl` per original plan
3. **DO NOT** invoke `tools/dispatch_cpu_eval_via_github_actions.py` until the operator + adversarial review subagent (a-NEW-spawned) clears the candidate selection
4. Surface in your landing memo:
   - Top-N M5 Max calibrated candidates (sub-0.190 / silver-band / above-frontier)
   - Recommended GHA dispatch list (1-5 candidates)
   - Predicted score ± ε bound per candidate
   - **EXPLICIT WAIT-FOR-OPERATOR** marker on the GHA dispatch step

## Coordination

- Adversarial review subagent (newly spawned) will produce per-item verdict matrix
- Operator approves GHA dispatch list AFTER reviewing both your M5 Max results + the adversarial review verdict
- This is the canonical "constrained coordinate search → review-before-spend" pattern

## References

- Original prompt: this subagent's launch turn (sequential #1 score-lowering)
- Adversarial review subagent: spawning now (a-NEW-id)
- Codex review: also being launched in parallel as cross-check
