# DDM M7 relaxed receiver realization — canonical DAG feed

Date: 2026-07-23
`research_only=true`
MAIN landing review and MAIN-owned Task #381 dispatch are required.

## Executed trajectory

`authority/config`
→ exact archive bytes/SHA/ZIP member
→ PR110 `FrozenPacket.parse`
→ native `(600,28) uint8` lattice
→ byte-identical latent/member/archive parse-back
→ receiver `Renderer` including resize/clamp/round/selector
→ exact source pairs in evaluator order
→ frozen `DistortionNet.compute_distortion`
→ 38 immutable resumable checkpoints
→ exact per-pair aggregation
→ contest-objective decomposition
→ strict routing fork.

Every node above is satisfied for the named archive. No scorer proxy,
continuous re-solve, solve-then-round replacement, remote execution, or
dispatch node was used.

## Typed feed row

```yaml
schema: ddm_m7_relaxed_receiver_realization_feed.v1
lane_id: ddm_m7_relaxed_receiver_realize_byteclose
object:
  archive_bytes: 177169
  archive_sha256: cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f
  receiver_member: x
  native_lattice: [600, 28, uint8]
measurement:
  axis: "[macOS-CPU frozen-scorer advisory]"
  pair_count: 600
  d_seg: 0.000545578002735662
  d_pose: 0.00002930755865188909
  archive_score: 0.18964681567130603
  score_claim: false
fork:
  pointer: 0.1910828242
  strict_threshold: 0.19108
  clears_strict_threshold: true
  delta_vs_strict_threshold: -0.001433184328693965
  route: BYTE-CLOSED_CANDIDATE_FOR_MODAL_EXACT_EVAL
authority:
  promotion_eligible: false
  ready_for_exact_eval_dispatch: false
  dispatch_performed: false
  main_review_required: true
custody:
  implementation_git_head: e961b08fad12e2c7446efdb1db01a89fe0c497a2
  runtime_manifest_sha256: 0e0702e374546b00a0169b00c836e54bc2c4b7268e096bf1dd34776e64335d1b
  upstream_manifest_sha256: 826399224af95aec46bdc797b2ee93804d8b30574c05e95f50f9bce0c4688f69
  measurement_receipt_file_sha256: 429baa120a54af0246a7cb45bef53b326c40ac2a317e18e1f6081590ab7f8718
```

## Triality

- DSL: `.omx/research/ddm_m7_relaxed_receiver_realize_byteclose_20260723.config.json`
- DAG: this feed; no dispatch edge is active
- equation:
  `ddm_m7_solve_to_realized_transfer_receiver_closed_v1` in
  `src/tac/canonical_equations/ddm_m7_realization_transfer_20260723.py`
- durable measurement:
  `.omx/research/ddm_m7_relaxed_receiver_realize_byteclose_20260723_receipt.json`
  plus the SSD receipt and all 38 preserved checkpoints

## Downstream consumer rule

The next admissible edge is MAIN review. If MAIN accepts archive/runtime/source
custody, MAIN may claim Task #381 and dispatch exact contest-CPU/CUDA replay.
Until those axis-specific results land, consumers must retain:
`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`, and the unchanged canonical pointer.
