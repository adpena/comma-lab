# L5 v2 Latest Neural Video Codec Source Basis - 2026-05-16

## Summary

Refreshed the L5 v2 planning-only research basis with newer neural video
codec anchors that directly affect score-lowering design pressure:

- DCVC-RT / practical real-time neural video compression
- unified intra/inter neural video coding
- generative latent video compression
- generative neural video compression with video diffusion priors

These are source-basis records only. They do not authorize dispatch,
promotion, or score claims.

## Integration

- `src/tac/optimization/research_basis.py`
- `src/tac/tests/test_research_basis.py`
- `src/tac/tests/test_l5_staircase_v2.py`

## Design Consequences

- L5 v2 should explicitly consider runtime operational cost, not only model
  FLOPs, because practical NVC papers identify memory I/O/function-call
  overhead as a real speed bottleneck.
- Intra/inter adaptivity is a stronger fit for two-frame contest packets than
  a single inherited inter-frame scaffold; selectors must be charged or
  deterministic.
- Latent/generative priors remain proxy-only until mapped to SegNet/PoseNet
  component deltas and byte-closed inside the contest archive/runtime.

## Hardening

Every added source carries `charged_byte_contract` and `hardening_blockers`.
The basis remains `planning_only=true`, `score_claim=false`, and
`promotion_eligible=false`.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_research_basis.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```
