# Codex Memory - OWV3 0120 Stack T4 Frontier - 2026-05-01T15:27Z

Context:

- Project: `/Users/adpena/Projects/pact`.
- Progress ledger:
  `.omx/research/shannon_floor_nextwave_telemetry_and_research_20260430_codex.md`.
- Claim matrix row: C-044 in
  `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`.

New active A++ frontier:

- Method: OWV3 0120 renderer/mask stack plus PFP16 pose representation.
- Archive bytes: `609963`.
- Archive SHA-256:
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- Exact eval path: canonical `archive.zip -> inflate.sh ->
  upstream/evaluate.py` through `experiments/contest_auth_eval.py --device
  cuda`, harvested from Lightning Batch artifacts.

Primary T4 evidence:

- Job: `exact_eval_owv3_0120_stack_t4_20260501T150652Z`.
- Local dir:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4_20260501T150652Z/`.
- Score: `0.9975405870574277`.
- Components: PoseNet `0.00357302`, SegNet `0.00402367`.
- Samples: `600`.
- Hardware: `Tesla T4`, `gpu_t4_match=true`.
- Adjudication: `A++ contest T4`, `promotion_eligible=true`,
  `lane_status=IN_PREDICTED_BAND`, component gates passed.

Hedge reproduction:

- Job: `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`.
- Local dir:
  `experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4aws_20260501T151050Z/`.
- Score: `0.9975385870574276`.
- Components: PoseNet `0.00357302`, SegNet `0.00402365`.
- Samples: `600`.
- Hardware: `Tesla T4`, `gpu_t4_match=true`.
- Adjudication: `A++ contest T4`, `promotion_eligible=true`,
  `lane_status=IN_PREDICTED_BAND`, component gates passed.

Queue cleanup:

- Hedge job completed and was harvested.
- Primary job had valid artifacts but SDK status regressed to `Pending`; it was
  stopped to prevent further spend. State refresh records `Stopped` and cost
  `0.04222222`.
- Hedge final cost was `0.17503889`.

Scientific status:

- This supersedes Direct-FD PFP16 C-043 as the active frontier but does not
  change the strategic conclusion: sub-0.3 still needs Alpha mask-payload
  collapse plus scorer-aware sparse repair and component-response gates.
