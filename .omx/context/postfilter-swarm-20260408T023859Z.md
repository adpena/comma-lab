# Task Statement

Launch a coordinated swarm to push the honest Track B floor below `2.05` by running the next capped post-filter cycle with durable evidence and verification.

# Desired Outcome

- Keep `robust_current` healthy and honest under the current published workflow and the stricter rule-faithful view.
- Execute up to three post-filter follow-on lanes in parallel.
- Promote only if package, inflate, smoke, and local authoritative scorer evidence support a real gain.
- Leave durable state so a fresh agent can resume from disk.

# Known Facts / Evidence

- Current honest Track B floor: `2.05`
- Current-workflow bytes: `861,986`
- Rule-faithful estimate: `2.0778631822069484` at `896,432` bytes
- Distortions: PoseNet `0.07996829`, SegNet `0.00586716`
- Promoted config: `522x392 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / sharpness=1 / learned int8 post-filter`
- Canonical evidence:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-report.txt`
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-smoke.json`
- Broad preprocessing is repeatedly rejected; do not spend the next cycle there without a new causal hypothesis.
- BAT00 is useful for smoke/ranking only; local CPU scorer runs remain authoritative.

# Constraints

- Mutation frontier applies; do not touch forbidden upstream or `exact_current` files.
- Prefer small, reversible changes.
- Prefer at most 3 experiments this cycle.
- Never claim a win without a measured score.
- Keep `current_workflow` and `rule_faithful` accounting separate.
- Use isolated authoritative runs; do not assume shared-tree artifacts are solely ours.

# Unknowns / Open Questions

- Does modest capacity growth (`hidden=24` or `32`) beat `2.05` after byte cost?
- Can a smaller or luma-only variant preserve most of the gain while reducing runtime/bytes?
- Can a cheaper CPU-friendly block preserve the gain?
- Is there a faster proxy/ranking path that predicts the scorer well enough to prune before full eval?

# Likely Touchpoints

- `experiments/train_postfilter.py`
- `experiments/train_postfilter_canonical.py`
- `submissions/robust_current/inflate_postfilter.py`
- `submissions/robust_current/postfilter_int8.pt`
- `submissions/robust_current/config.av1-2.05-postfilter.env`
- `reports/raw/**`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.ralph/run_log.md`
- `reports/latest.md`
