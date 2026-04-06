# promotion gate

Every new honest Track B baseline candidate must clear this gate before canonicalization.

## scope rule

- Default assumption: every lane is rule-faithful and compliance-checked.
- The only explicitly non-rule-faithful lane is `submissions/exact_current` / Track A.

## mandatory promotion checklist

1. scorer-backed measurement on the local CPU path
2. pre-scorer smoke gate:
   - raw output exists
   - raw byte size matches expected `frame_count * width * height * 3`
   - file cardinality matches the video list
3. canonical default-config regression to confirm the live path reproduces the candidate
4. current_workflow vs rule_faithful separation recorded explicitly
5. critical senior engineer review against the contest rules and official scoring path
6. bug audit over:
   - raw byte layout / pixel format
   - colorspace / range / conversion path
   - frame count / ordering / parity-sensitive behavior
   - geometry / even-dimension handling
   - branch-specific differences across codec and ROI paths
7. written decision: promote or reject

## writeup requirement for each serious experiment

Each serious experiment should add a concise research note with:
- prior baseline
- hypothesis
- estimated directional improvement and why
- measured result
- reflection on why it won or lost
- whether the hypothesis held
