# INDEX — codec pipeline canonicalization session (2026-05-07)

This index threads the empirical + design artifacts produced 2026-05-07
into a navigable chain. Future sessions should read this first to find
the right entrypoint without re-deriving findings.

## TL;DR

- **Canonical Protocol**: `tac.codec_pipeline.CodecOp` + `CodecPipeline` orchestrator (CPL1 wire format). Six paradigm wraps + Op 4 full-stack runner.
- **Empirical PR106 frontier**: PR103-on-PR106 standalone candidate at **185,578 B** (-661 B vs source 186,239), candidate SHA `ec0890c2`. Gates 1-4 GREEN; gate 5 (contest-CUDA) blocked on GPU billing.
- **Mathematical headroom**: PR106 brotli = 170,096 B; H₀ floor = 167,570 B (sits at 1.015×); **H₂ floor = 88,983 B**. Current H₀-H₂ gap = 78,580 B (top 5 tensors capture 78.5%, all `blocks.*`).
- **All score claims tagged `[empirical]` or `[predicted-band]`** per CLAUDE.md "Forbidden score claims". No `[contest-CUDA]` claims this session.

## Canonical artifacts

### Code (committed)

- `src/tac/codec_pipeline.py` — Protocol + CPL1 wire format + Op 1/2 (`b4562092` + `33bef6d3`)
- `src/tac/codec_pipeline_apogee_int.py` — Op 3 substrate-transform (`4153b195` batch)
- `src/tac/codec_pipeline_mask.py` — α-paradigm 4-way bakeoff (`9e3ee043`)
- `src/tac/codec_pipeline_sensitivity.py` — β-paradigm preprocessing (`fc44d74a` + `5ef4e416`)
- `src/tac/codec_pipeline_joint_admm.py` — γ-paradigm Joint-ADMM wrap (`fc44d74a`)
- `src/tac/codec_pipeline_full_stack.py` — Op 4 composition matrix (`ba681d11`)
- `src/tac/codec_pipeline_deltaepszeta_callback.py` — δεζ training-time empirical signal (`ee2d5fcf`)
- `src/tac/shannon_h2_loss.py` — differentiable H₀/H₂ surrogates (`fcfb8861`)
- `src/tac/pr103_arithmetic_codec.py` (existing) + per-tensor AC auto-fallback (`569e5ca8`)
- `tools/per_tensor_shannon_analysis.py` — exact (numpy) Shannon floor (`31183b01`)
- `tools/build_deltaepszeta_training_targets.py` — per-tensor training-target weights (latest commit)
- `scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh` — pre-staged gate-5 executor (`fcfb8861`)

### Empirical evidence directories

- `experiments/results/pr103_repack_pr106_standalone_20260507/` — the production-frontier candidate (185,578 B, gates 1-4 GREEN)
- `experiments/results/lane_codec_pipeline_full_stack_pr106_20260507T172731Z/` — composition matrix on PR106
- `experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/` — per-tensor Shannon floor (Path B)
- `experiments/results/lane_op2_ac_fallback_pr106_20260507T174715Z/` — Op2 auto-fallback empirical
- `experiments/results/lane_council_fixes_pr106_20260507T174118Z/` — β-identity short-circuit + γ STUB
- `experiments/results/lane_deltaepszeta_targets_pr106_20260507T184921Z/` — δεζ per-tensor target weights

### Design memos (this session)

- `.omx/research/four_way_stack_composition_contract_20260507_claude.md` — canonical contract: CodecOp Protocol, CPL1 wire format, three composition modes (substitutional / substrate-transform / decorator), composition rule for `transforms_state_dict=True`.
- `.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md` — council deliberation + corrigendum. Authoritative Next-Move list (5 gates for PR103-on-PR106 promotion).
- `.omx/research/pr_top3_non_arbitrariness_paper_cross_reference_20260507_claude.md` — paper cross-reference for non-arbitrariness audit.
- `.omx/research/pr103_pr106_runtime_closure_20260507_codex.md` — runtime closure proof for the PR103-on-PR106 production candidate.

### Auto-memory entries (durable across sessions)

In `~/.claude/projects/-Users-adpena-Projects-pact/memory/`:

- `feedback_canonical_codec_pipeline_session_complete_20260507.md` — full session retrospective.
- `feedback_op3_op2_fallback_breakthrough_20260507.md` — corrigendum on the 135,940 B framing; authoritative current state.
- `feedback_pr106_substrate_composition_matrix_empirical_20260507.md` — composition matrix details.
- `feedback_top3_PRs_are_boltons_on_PR100_substrate_20260507.md` — strategic finding (substrate-mismatch).
- `feedback_canonical_codec_pipeline_orchestrator_20260507.md` — canonicalization pattern.

## Gate state for promotion ladder (PR103-on-PR106 standalone)

Per `grand_council_pr106_substrate_findings_zig_default_20260507.md` "Next Move":

| # | Gate | State | Evidence |
|---|---|---|---|
| 1 | runtime packet | ✅ | `submissions/pr103_pr106_final_runtime/`; `final_runtime_packet_proof.json` `passed: True` |
| 2 | brotli/constriction custody | ✅ | `pyproject_hard_dependency: True` for both; uv.lock entries; runtime versions captured |
| 3 | strict static compliance | ✅ | `pre_submission_compliance.static.json` `passed: True` |
| 4 | lane claim | ✅ | `pr103_pr106_standalone` active in `.omx/state/active_lane_dispatch_claims.md`, ttl 168h |
| 5 | contest-CUDA auth eval | ❌ | requires GPU billing; pre-staged executor at `scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh` |

## Strategic next moves (when GPU returns)

1. Run gate-5 dispatch playbook → contest-CUDA score on PR103-on-PR106
2. If score validates: promote to lane-maturity L3, update `reports/latest.md`
3. If score validates: open the actual contest PR (council "Next Move" #5)

## Strategic next moves (CPU-side, while GPU is blocked)

1. **Bug-hunter adversarial review** in flight (subagent `a9843a68d07ce87aa`) — sweeping all 9 session landings under 8-perspective council
2. **δεζ training scaffolding** — Phase 3 hyperprior init API design (per Ballé council position), wire shannon_h2_loss into actual training loops once a δεζ training lane exists
3. **HuggingFace dataset upload** — task #368 in_progress, separate session

## Council positions on session landings (one-line each)

- **Shannon (LEAD)**: brotli at 1.015× H₀ is near-optimal i.i.d.; H₂ headroom is the actual prize; training-direction question requires a coder that can EXPLOIT H₂.
- **Dykstra (CO-LEAD)**: composition matrix is the achievable region; Op2_alone is the vertex; CPL1 wire-format honesty preserves correctness over byte-savings.
- **Yousfi**: substrate-mismatch is the cover-source-mismatch problem; auto_select + auto_fallback are correct safety gates; production must ship one charged representation.
- **Fridrich**: zig as default is information-theoretically optimal under cover-source uncertainty; non-binding per operator.
- **Contrarian**: 135,940 B framing was over-claiming; CPL1 stores both blobs; production cannot double-spend; corrigendum properly re-anchored the rate-only frontier at 185,578 B.
- **Quantizr**: at 0.195 score band, contest rewards engineering velocity on a shared substrate; PR103-on-PR106 standalone is the actionable ship.
- **Hotz**: `auto_select=True` already on by default; per-tensor decisions delegated to algorithm; ship Op2_alone with auto-fallback.
- **Selfcomp**: production stack on PR106 is exactly Op2_alone + per-tensor auto-fallback; multiplicative gain is academic until δεζ training co-adapts substrate.
- **MacKay**: substrate-mismatch is MDL prior-mismatch; auto_select algorithms ARE the Bayes-optimal codec under empirical posterior.
- **Ballé**: γ hyperprior needs substrate-tuned init (deferred to Phase 2 GPU work); 1.91× brotli/H₂ ratio is the modern-codec target.

## How to verify findings yourself

```bash
# 1. Run all session tests:
.venv/bin/python -m pytest src/tac/tests/test_codec_pipeline*.py \
    src/tac/tests/test_pr101_split_brotli_codec*.py \
    src/tac/tests/test_pr103_arithmetic_codec.py \
    src/tac/tests/test_shannon_h2_loss.py \
    src/tac/tests/test_per_tensor_shannon_analysis.py \
    src/tac/tests/test_build_deltaepszeta_training_targets.py -q

# 2. Re-run composition matrix on PR106:
.venv/bin/python -c "import torch; from tac.codec_pipeline_full_stack import run_composition_matrix; \
    sd = torch.load('experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt', \
                    weights_only=True, map_location='cpu'); \
    run_composition_matrix(sd)"

# 3. Re-derive δεζ training targets:
.venv/bin/python tools/build_deltaepszeta_training_targets.py

# 4. Verify PR103-on-PR106 candidate bytes + SHA:
bash scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh --dry-run

# 5. Re-validate runtime closure:
.venv/bin/python tools/prove_pr103_pr106_runtime_closure.py
```

## Score-claim discipline

Every artifact in this index has `score_claim: false` or its tagged
equivalent. The only path to a `[contest-CUDA]` score is gate 5 of the
promotion ladder, which requires GPU billing. Future sessions reading
this index should NOT promote any of these numbers to score claims
without that empirical step.
