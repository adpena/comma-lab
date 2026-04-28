# Tuna-2 Lane Methodology

## Controlled-Variant Pattern

Every Tuna-2 lane should pair the proposed mechanism with a minimal-change
baseline. The baseline uses the same profile, training length, artifacts,
evaluation path, and deployment script shape, changing only the mechanism
under test.

Lane A at 1.15 [contest-CUDA] and Lane G-V3 at 1.05 [contest-CUDA] are examples
of lanes that should have had explicit controlled baselines when follow-on
variants were launched. A new variant should be interpretable as "baseline plus
one mechanism," not as a bundle of architecture, schedule, data, and eval-path
changes.

## Isolation Rule

One comparison isolates one mechanism. If a lane changes the renderer, the
loss weights, the pose optimizer, and the archive builder at the same time, the
result is not actionable. Split those changes into separate lanes or declare a
controlled baseline for each mechanism.

## Provenance Template

Every future `provenance.json` for a controlled lane should include:

```json
{
  "lane_name": "lane_t2_example",
  "lane_script": "scripts/remote_lane_t2example_bootstrap.sh",
  "controlled_baseline_lane": "lane_t2_example_control",
  "controlled_baseline": "same profile and script, without the tested mechanism",
  "changed_mechanism": "single mechanism under test",
  "score_context": "[contest-CUDA] for authoritative scores, [advisory only] otherwise",
  "eval_roundtrip": true,
  "informational_only": false
}
```

`controlled_baseline_lane` is the stable key consumed by methodology checks.
`controlled_baseline` may be a human-readable label or script-local variable,
but it must name the comparator clearly enough that a later audit can reproduce
the lane comparison.
