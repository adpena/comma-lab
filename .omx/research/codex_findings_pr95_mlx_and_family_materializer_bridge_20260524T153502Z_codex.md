# Codex Findings: PR95 MLX Status And Family Materializer Bridge

Timestamp UTC: 2026-05-24T15:35:02Z

## Findings

1. PR95/HNeRV MLX is not yet a full source-faithful PR95 reproduction. The
   current implementation has MLX decoder topology, PR95 archive grammar
   parsing/export primitives, PyTorch-vs-MLX forward parity probes, and
   optimizer descriptor plumbing for stages 1/5/8, but the executable training
   path is explicitly synthetic timing only.

2. The correct PR95 truth label is `implemented_mlx_synthetic_timing_only`, not
   source faithful. Remaining blockers are source video loader, scorer/eval
   roundtrip loss, exact stage hparams/cosine schedules, QAT/resume semantics,
   export forward parity, byte-closed contest archive export, and exact
   CPU/CUDA auth eval.

3. The family-agnostic materializer bridge was too chain-shaped. It now accepts
   harvestable family candidate manifests (`archive_section_entropy_recode`,
   `packet_member_recompress`, `tensor_factorize`) through the optimizer queue
   and scheduler harvest path without making score/dispatch authority claims.

4. Family-agnostic receiver proof state must remain visible to exact readiness:
   missing receiver proof now leaves `runtime_adapter_ready=false`,
   `candidate_runtime_adapter_blocker_cleared=false`, and the blocker
   `family_agnostic_receiver_contract_not_satisfied` on the planning row.

5. Inverse-steganalysis water-bucket economics were using a stale local
   denominator (`25/50_000_000`). The acquisition surface now imports the
   canonical contest denominator from `tac.score_composition`, so
   `lambda_rate = 25 / 37_545_489`. A boundary regression proves a cell that
   would have been selected under the stale denominator is rejected under the
   contest formula.

6. IAS1 exact-mode work queue rows could previously look executable without a
   real inflate parity context. Exact-mode inverse-scorer chains now refuse to
   queue unless they have either runtime parity context or paired source and
   candidate inflate-output dirs, and the chain command defaults to
   `--fail-if-inflate-parity-blocked`. Descriptor-only probing remains possible
   only through the non-exact descriptor path.

7. The real IAS1 materializer currently appends descriptor bytes. Its chain
   economics are now pinned in tests as `realized_cost`; exact-readiness already
   rejects byte-increasing full-frame-parity IAS1 rows unless explicitly marked
   `rate_only_control`.

8. Packet-member recompress probes against PR95/PR100/PR101/PR103/PR105/PR107
   archives emitted byte-closed candidates but were negative rate signal: the
   tried deflate-9 member recompress path increased archives by about 55 bytes
   and remains blocked by missing runtime-consumption proof. Probe artifacts are
   preserved under
   `.omx/research/codex_packet_member_recompress_probe_20260524T_local/`.

9. Inverse-steganalysis water buckets were still too easy to consume as
   isolated leaf cells. The bridge now emits an explicit
   `inverse_steganalysis_water_bucket_materialization_portfolio.v1` with an
   actuation mode per selected cell, preserves that portfolio through merged
   byte-shaving signal surfaces and campaign plans, and defaults bare cells to
   `high_level_operation_compiler_required`. The old IAS1 leaf-cell coordinate
   probe remains available only behind an explicit diagnostic opt-in flag.

10. Exact auth dispatch is intentionally not claimed green here. The bridge now
    carries candidate manifests and exact-readiness blockers forward, but there
    is still no lane-claiming exact-eval actuator stage for the new portfolio
    compiler output. The next implementation must lower a portfolio row into a
    byte-closed materializer result, prove runtime consumption/parity, promote
    through exact readiness, claim the lane, and dispatch exact auth eval.

11. Final-byte materializer context compilation no longer drops unsupported
    backlog rows. Unsupported rows now survive as blocked context rows with an
    explicit `materializer_context_compiler_missing:*` blocker, so future
    inverse-steg portfolio compiler gaps remain visible to the queue and cannot
    disappear as a silent unsupported count.

12. PacketIR/compiler integration is the correct next lowering target for
    portfolio-level inverse-steg operation sets. Unsupported context rows now
    carry a fail-closed `final_byte_packetir_compiler_bridge_hint.v1` pointing
    at `tac.packet_compiler.deterministic_compiler`. The PacketIR operation-set
    schema, canonical compiler order, and required proofs are owned by
    `packetir_operation_set_bridge_contract()` in the deterministic compiler;
    scheduler rows only add row context and blockers. This avoids adding more
    one-off materializer glue while preserving exact archive/runtime custody.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_family_agnostic_materializers.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py::test_pr95_optimizer_descriptor_drives_stage8_partition src/tac/tests/test_pr95_hnerv_mlx.py::test_synthetic_timing_smoke_emits_runtime_profile_and_refusal src/tac/tests/test_pr95_muon_local_training_integration.py::test_pr95_manifest_adapter_preserves_optimizer_descriptor_identity -q`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_family_agnostic_materializers.py -q`
- `.venv/bin/ruff check src/tac/optimizer/materializer_chain_harvest.py src/tac/optimizer/candidate_queue.py src/tac/optimizer/exact_readiness.py src/comma_lab/scheduler/materializer_chain_harvest.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_optimizer_exact_readiness.py tools/build_optimizer_candidate_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_inverse_scorer_exact_eval_queue.py -q` (233 passed, 1 duplicate-ZIP warning)
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_optimizer_scheduler_registry.py -q` (32 passed)
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py -q` (27 passed)
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py -q` (7 passed)
- `git diff --check`

## Next Step

Close PR95 source-faithful reproduction by adding the real source-video loader
and PR95 loss/schedule/QAT/resume semantics to the MLX training path, then run
same-checkpoint PyTorch/MLX forward parity and byte-closed archive export before
any compare-and-score claim.
