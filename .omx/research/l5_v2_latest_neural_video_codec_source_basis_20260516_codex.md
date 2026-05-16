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

## Primary Sources

Retrieved 2026-05-16.

- Official challenge repository:
  https://github.com/commaai/comma_video_compression_challenge
- Public frontier PR anchors used as contest-specific PacketIR/HNeRV context:
  - PR95: https://github.com/commaai/comma_video_compression_challenge/pull/95
  - PR100: https://github.com/commaai/comma_video_compression_challenge/pull/100
  - PR101: https://github.com/commaai/comma_video_compression_challenge/pull/101
  - PR103: https://github.com/commaai/comma_video_compression_challenge/pull/103
  - PR106: https://github.com/commaai/comma_video_compression_challenge/pull/106
- HNeRV, the nearest public leaderboard substrate family:
  - CVF open-access paper:
    https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html
  - arXiv: https://arxiv.org/abs/2304.02633
- DCVC-RT / practical real-time neural video compression:
  - arXiv: https://arxiv.org/abs/2502.20762
  - code family: https://github.com/microsoft/DCVC
- TeCoNeRV / temporal coherence for compressible neural representations:
  - arXiv: https://arxiv.org/abs/2602.16711
  - project page: https://namithap10.github.io/teconerv/

## Claim Scope

- HNeRV/TeCoNeRV sources justify neural representation and temporal-coherence
  design pressure only; they are not contest score evidence.
- DCVC-RT justifies treating runtime operational overhead as a first-class
  constraint; it does not authorize importing a production NVC architecture
  without a byte-closed contest grammar and scorer-aware training.
- PR101/PR106 links are public-frontier provenance anchors; any exact score
  claim still requires local archive SHA, runtime tree/content SHA, CPU/CUDA
  axis labels, logs, and paired exact-eval custody.

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
