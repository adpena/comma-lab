# INDEX - codec pipeline canonicalization session (2026-05-07)

This index threads the empirical + design artifacts produced 2026-05-07
into a navigable chain. Future sessions should read this first to find
the right entrypoint without re-deriving findings.

## TL;DR

- **Canonical Protocol**: `tac.codec_pipeline.CodecOp` + `CodecPipeline`
  orchestrator (CPL1 wire format). Six paradigm wraps + Op 4 full-stack runner.
- **Promoted PR106 frontier anchor**: PR103-on-PR106 standalone candidate at
  **185,578 B** (-661 B vs source 186,239), candidate SHA `ec0890c2`, exact
  CUDA T4 strict formula score **0.2089810755823297**
  (report-reconstructed score **0.20898105277982337**). Static compliance and
  contest-final compliance are both preserved in
  `experiments/results/pr103_repack_pr106_standalone_20260507/`.
- **Mathematical headroom**: PR106 brotli = 170,096 B; H0 floor = 167,570 B
  (sits at 1.015x); H2 floor = 88,983 B. Current H0-H2 gap = 78,580 B
  (top 5 tensors capture 78.5%, all `blocks.*`). This is a conditional-entropy
  weighting signal, not a score claim.
- **Score-claim discipline**: PR103-on-PR106 now has A++ exact CUDA evidence.
  Delta-epsilon-zeta, codec-pipeline, mask-bakeoff, and composition artifacts
  remain `[empirical]` or `[predicted-band]`.

## Canonical artifacts

### Code (committed)

- `src/tac/codec_pipeline.py` - Protocol + CPL1 wire format + Op 1/2 (`b4562092` + `33bef6d3`)
- `src/tac/codec_pipeline_apogee_int.py` - Op 3 substrate-transform (`4153b195` batch)
- `src/tac/codec_pipeline_mask.py` - alpha-paradigm 4-way bakeoff (`9e3ee043`)
- `src/tac/codec_pipeline_sensitivity.py` - beta-paradigm preprocessing (`fc44d74a` + `5ef4e416`)
- `src/tac/codec_pipeline_joint_admm.py` - gamma-paradigm Joint-ADMM wrap (`fc44d74a`)
- `src/tac/codec_pipeline_full_stack.py` - Op 4 composition matrix (`ba681d11`)
- `src/tac/codec_pipeline_deltaepszeta_callback.py` - delta-epsilon-zeta training-time empirical signal (`ee2d5fcf`)
- `src/tac/shannon_h2_loss.py` - differentiable H0/H2 surrogates (`fcfb8861`, locally hardened this tranche)
- `src/tac/pr103_arithmetic_codec.py` (existing) + per-tensor AC auto-fallback (`569e5ca8`)
- `tools/per_tensor_shannon_analysis.py` - exact (numpy) Shannon floor (`31183b01`)
- `tools/build_deltaepszeta_training_targets.py` - per-tensor training-target weights with fixed-timestamp rebuild support

### Empirical evidence directories

- `experiments/results/pr103_repack_pr106_standalone_20260507/` - production-frontier candidate, contest-final compliance, and tracked score snapshot
- `experiments/results/lane_codec_pipeline_full_stack_pr106_20260507T172731Z/` - composition matrix on PR106
- `experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/` - per-tensor Shannon floor (Path B)
- `experiments/results/lane_op2_ac_fallback_pr106_20260507T174715Z/` - Op2 auto-fallback empirical
- `experiments/results/lane_council_fixes_pr106_20260507T174118Z/` - beta-identity short-circuit + gamma STUB
- `experiments/results/lane_deltaepszeta_targets_pr106_20260507T184921Z/` - delta-epsilon-zeta per-tensor target weights

### Design memos (this session)

- `.omx/research/four_way_stack_composition_contract_20260507_claude.md` - canonical contract: CodecOp Protocol, CPL1 wire format, three composition modes (substitutional / substrate-transform / decorator), composition rule for `transforms_state_dict=True`.
- `.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md` - council deliberation + corrigendum. Authoritative Next-Move list (5 gates for PR103-on-PR106 promotion).
- `.omx/research/pr_top3_non_arbitrariness_paper_cross_reference_20260507_claude.md` - paper cross-reference for non-arbitrariness audit.
- `.omx/research/pr103_pr106_runtime_closure_20260507_codex.md` - runtime closure proof for the PR103-on-PR106 production candidate.

### Auto-memory entries (durable across sessions)

In `~/.claude/projects/-Users-adpena-Projects-pact/memory/`:

- `feedback_canonical_codec_pipeline_session_complete_20260507.md` - full session retrospective.
- `feedback_op3_op2_fallback_breakthrough_20260507.md` - corrigendum on the 135,940 B framing; authoritative current state.
- `feedback_pr106_substrate_composition_matrix_empirical_20260507.md` - composition matrix details.
- `feedback_top3_PRs_are_boltons_on_PR100_substrate_20260507.md` - strategic finding (substrate-mismatch).
- `feedback_canonical_codec_pipeline_orchestrator_20260507.md` - canonicalization pattern.

## Gate state for promotion ladder (PR103-on-PR106 standalone)

Per `grand_council_pr106_substrate_findings_zig_default_20260507.md` "Next Move":

| # | Gate | State | Evidence |
|---|---|---|---|
| 1 | runtime packet | GREEN | `submissions/pr103_pr106_final_runtime/`; `final_runtime_packet_proof.json` `passed: True` |
| 2 | brotli/constriction custody | GREEN | `pyproject_hard_dependency: True` for both; uv.lock entries; runtime versions captured |
| 3 | strict static compliance | GREEN | `pre_submission_compliance.static.json` `passed: True` |
| 4 | lane claim | GREEN | exact-eval claim closed as `completed_score_0_2089810528`; stale duplicate closed as superseded |
| 5 | contest-CUDA auth eval | GREEN | `pre_submission_compliance.contest_final.json` records A++ T4 strict formula score `0.2089810755823297`, report-reconstructed score `0.20898105277982337`, 185,578 bytes, SHA `ec0890c2...` |

## Strategic next moves after PR103 anchor promotion

1. Route first-tranche score work to categorical byte-closed parity, JCSP
   runtime consumption, HNeRV entropy candidates below the 185,578 B byte floor,
   telescopic foveation charged-consumer proof, and LA-pose archive consumption.
2. Treat PR103-on-PR106 as the A++ current local anchor in Pareto/meta-Lagrangian
   selection. Do not spend exact eval on rate-only candidates that fail to beat
   the 185,578 B floor unless they also change scorer-relevant outputs.
3. Keep delta-epsilon-zeta work as training-target and substrate-design work
   until a byte-closed archive and exact CUDA run exist.

## Strategic next moves (CPU-side, while GPU is blocked)

1. **Bug-hunter adversarial review** - treat codec-pipeline, entropy, categorical,
   LA-pose/foveation, and frontier-routing changes as review-required before
   promotion.
2. **Delta-epsilon-zeta training scaffolding** - wire `shannon_h2_loss` into
   an actual training lane only after the objective states whether it exploits
   current H2 structure or reshapes the substrate.
3. **HuggingFace dataset upload** - task #368 in_progress, separate session.

## Council positions on session landings (one-line each)

- **Shannon (LEAD)**: brotli at 1.015x H0 is near-optimal i.i.d.; H2 headroom is the actual prize; training-direction question requires a coder that can exploit H2.
- **Dykstra (CO-LEAD)**: composition matrix is the achievable region; Op2_alone is the vertex; CPL1 wire-format honesty preserves correctness over byte-savings.
- **Yousfi**: substrate-mismatch is the cover-source-mismatch problem; auto_select + auto_fallback are correct safety gates; production must ship one charged representation.
- **Fridrich**: zig as default is information-theoretically optimal under cover-source uncertainty; non-binding per operator.
- **Contrarian**: 135,940 B framing was over-claiming; CPL1 stores both blobs; production cannot double-spend; corrigendum properly re-anchored the rate-only frontier at 185,578 B.
- **Quantizr**: at 0.195 score band, contest rewards engineering velocity on a shared substrate; PR103-on-PR106 standalone is the actionable ship.
- **Hotz**: `auto_select=True` already on by default; per-tensor decisions delegated to algorithm; ship Op2_alone with auto-fallback.
- **Selfcomp**: production stack on PR106 is exactly Op2_alone + per-tensor auto-fallback; multiplicative gain is academic until delta-epsilon-zeta training co-adapts substrate.
- **MacKay**: substrate-mismatch is MDL prior-mismatch; auto_select algorithms ARE the Bayes-optimal codec under empirical posterior.
- **Balle**: gamma hyperprior needs substrate-tuned init (deferred to Phase 2 GPU work); 1.91x brotli/H2 ratio is the modern-codec target.

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

# 3. Re-derive delta-epsilon-zeta training targets byte-reproducibly:
.venv/bin/python tools/build_deltaepszeta_training_targets.py \
    --shannon-json experiments/results/lane_per_tensor_shannon_pr106_20260507T173846Z/per_tensor_shannon.json \
    --output-dir experiments/results/lane_deltaepszeta_targets_pr106_20260507T184921Z \
    --started-at-utc 2026-05-07T18:49:21Z

# 4. Verify PR103-on-PR106 contest-final score snapshot:
jq '.auth_eval.record' \
    experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json

# 5. Re-validate runtime closure:
.venv/bin/python tools/prove_pr103_pr106_runtime_closure.py
```

## Score-claim discipline

PR103-on-PR106 has an A++ exact CUDA score snapshot. Every other artifact in
this index has `score_claim: false` or its tagged equivalent. Future sessions
should not promote delta-epsilon-zeta, mask-bakeoff, foveation, or composition
numbers to score claims without exact CUDA evidence on exact archive bytes.
