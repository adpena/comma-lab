# Codex routing directive: DP1+PR101 OP-1 + OP-2 — $0 local CPU probes BEFORE $3-5 Stage 2 smoke
# Date: 2026-05-18
# Operator: approved 2026-05-18 ("All are approved")
# Authority: .omx/research/dp1_pr101_composition_design_memo_20260518.md (commits 1d8f490fd + 427405d86)
#   §TOP-5 op-routables: 4-of-5 are $0 local CPU probes (Carmack-style validation cascade)
# Per CLAUDE.md "Frontier target" + premise-verification-before-edit discipline (Catalog #229)

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Forbidden device-selection defaults" + Catalog #192 macOS-CPU advisory + #313 probe outcomes ledger)
2. `.omx/research/dp1_pr101_composition_design_memo_20260518.md` (the AUTHORITY memo; read TOP-5 op-routables verbatim)
3. `src/tac/substrates/pretrained_driving_prior/` (DP1 canonical impl)
4. `src/tac/substrates/pr101_lc_v2_clone/` (PR101 GOLD consumer)
5. `upstream/videos/0.mkv` (contest video for OOD-similarity comparison)
6. Comma2k19 dataset (DP1's training corpus; canonical access via `tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache` per Catalog #213)

## STRATEGIC CONTEXT

DP1+PR101 composition design memo §13 Dykstra-feasibility identified TWO probes that disambiguate whether ΔS lands at optimistic -0.012 or pessimistic +0.002:

| Probe | Cost | Gates |
|---|---|---|
| **OP-1 OOD-similarity probe** | $0 local CPU | "DP1's prior is meaningful warm-start" assumption |
| **OP-2 Architecture-compatibility probe** | $0 local CPU | "DP1 init is structurally loadable into PR101 HNeRV" assumption |
| ↓ (both must pass to proceed) | | |
| OP-3 Stage-2 Modal T4 smoke (Path A) | $3-5 | first empirical ΔS measurement |

Per Carmack-Hotz "30-second reviewability": $0 probes BEFORE GPU spend. Total wall-clock for OP-1+OP-2 in parallel: 10-15 min.

## OP-1: OOD-similarity probe (Comma2k19 vs contest video)

### What Codex builds

`tools/probe_dp1_pr101_ood_similarity.py` (~250 LOC) — measures feature-distribution distance between Comma2k19 dashcam frames and `upstream/videos/0.mkv` contest frames:

1. Load N representative frames from Comma2k19 (use canonical `Comma2k19LocalCache` per Catalog #213)
2. Load N matching-resolution frames from `upstream/videos/0.mkv`
3. Compute feature-distribution distance via ONE of:
   - SegNet feature-map cosine distance (canonical scorer surrogate)
   - PoseNet feature-map MSE
   - Histogram intersection over RGB
   - FID-like score (Frechet Inception Distance approximation; CPU-only)
4. Emit verdict: `OOD_SIMILAR` (distance < threshold) / `OOD_DIFFERENT` (distance > threshold) / `INDETERMINATE` (band overlap)
5. Register probe outcome via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313
6. Emit canonical advisory result row per Catalog #192 (macOS-CPU advisory only; non-promotable; but actionable as design-decision input)

### Acceptance criteria

- Probe completes in < 10 min on M5 Max CPU
- Emits structured verdict + numeric distance + N-frame sample size + reproducibility seed
- Result row tagged `[macOS-CPU advisory]` per Catalog #192 (probe is INPUT to dispatch decision; not a contest score claim)
- Register canonical posterior anchor via `tac.probe_outcomes_ledger.register_probe_outcome` with `probe_id=dp1_pr101_ood_similarity_<utc>` + `verdict=<above>`

### Verdict consumption

- `OOD_SIMILAR`: Path A DP1+PR101 composition predicted ΔS shifts toward -0.012 (optimistic); recommend Stage-2 dispatch
- `OOD_DIFFERENT`: Path A predicted ΔS shifts toward +0.002 (pessimistic); recommend DEFER Stage-2 OR scope-shrink to Path A-narrow
- `INDETERMINATE`: more analysis required; recommend extending sample size N before dispatch decision

## OP-2: Architecture-compatibility probe (DP1 codebook vs PR101 HNeRV layer-init)

### What Codex builds

`tools/probe_dp1_pr101_architecture_compatibility.py` (~250 LOC) — verifies DP1 codebook shape + PR101 HNeRV layer dimensions are STRUCTURALLY compatible for Stage-1 weight init:

1. Load DP1 codebook from `src/tac/substrates/pretrained_driving_prior/` canonical archive (162164-byte decoder blob + 15387-byte latent blob per Catalog #210)
2. Load PR101 HNeRV decoder architecture from `src/tac/substrates/pr101_lc_v2_clone/`
3. Verify per-layer:
   - Tensor shape compatibility (DP1's per-layer tensors must fit into PR101's per-layer slots)
   - Activation function compatibility (both use same nonlinearity family per HNeRV parity)
   - Normalization layer compatibility (LayerNorm vs BatchNorm vs none)
   - Initialization scale compatibility (DP1 fp16 vs PR101 fp32 — needs explicit cast)
4. Emit verdict: `COMPATIBLE` (all layers shape-load) / `PARTIAL` (some layers need reshape/projection) / `INCOMPATIBLE` (structural mismatch; Path A blocked)
5. Register probe outcome per Catalog #313
6. Emit canonical advisory result row per Catalog #192

### Acceptance criteria

- Probe completes in < 5 min on M5 Max CPU
- Emits structured verdict + per-layer compatibility report + suggested reshape/projection operations if PARTIAL
- Result row tagged `[macOS-CPU advisory]` per Catalog #192
- Register canonical posterior anchor via `tac.probe_outcomes_ledger.register_probe_outcome` with `probe_id=dp1_pr101_architecture_compatibility_<utc>` + `verdict=<above>`

### Verdict consumption

- `COMPATIBLE`: Path A Stage-2 dispatch can proceed; predicted ΔS unmoved by this probe
- `PARTIAL`: Path A requires FORK-of-canonical weight-init bridge per design memo §11 (canonical-vs-unique decision per layer); estimated +2 LOC overhead; predicted ΔS preserved
- `INCOMPATIBLE`: Path A blocked; DP1+PR101 composition is structurally infeasible; recommend DEFER + design-memo update + alternative-substrate selection

## COMPOSITION DECISION TABLE (OP-1 × OP-2 verdict matrix)

| OP-1 \ OP-2 | COMPATIBLE | PARTIAL | INCOMPATIBLE |
|---|---|---|---|
| **OOD_SIMILAR** | PROCEED Stage 2; optimistic ΔS -0.012 | PROCEED Stage 2 with FORK weight-init; optimistic-but-reduced ΔS -0.008 | DEFER + redesign |
| **OOD_DIFFERENT** | PROCEED Stage 2 cautiously; pessimistic ΔS +0.002 | DEFER pending bigger N; pessimistic-but-actionable | DEFER + redesign |
| **INDETERMINATE** | Extend N for OP-1 | Extend N for OP-1 + plan FORK weight-init | DEFER + redesign |

## DISCIPLINE

- Catalog #229 premise verification (verify Comma2k19LocalCache canonical helper exists; verify DP1 canonical archive present; verify PR101 architecture loadable)
- Catalog #117/#157/#174 commit serializer with POST-EDIT shas
- Catalog #206 checkpoint discipline (probe scripts are short; one checkpoint each)
- Catalog #131 fcntl-locked writes to `.omx/state/probe_outcomes.jsonl`
- Catalog #192 macOS-CPU advisory only (probes emit advisory results; NEVER promote without paired Linux x86_64 verification)
- Catalog #287 evidence-tag discipline (every numeric output tagged)
- Catalog #213 Comma2k19 canonical helper required (no ad-hoc download)
- Catalog #313 register probe outcomes for both probes
- Catalog #314 absorption avoidance: scope is `tools/probe_dp1_pr101_*.py` + tests + `.omx/state/probe_outcomes.jsonl` registration

## EXIT CRITERIA

- [ ] `tools/probe_dp1_pr101_ood_similarity.py` runnable; emits structured verdict
- [ ] `tools/probe_dp1_pr101_architecture_compatibility.py` runnable; emits structured verdict
- [ ] Both probes complete in < 15 min combined on M5 Max CPU
- [ ] Both probe-outcome rows registered to `.omx/state/probe_outcomes.jsonl`
- [ ] Composition decision-table verdict emitted (cell from OP-1 × OP-2 matrix)
- [ ] If PROCEED verdict: append canonical_task_status row to queue OP-3 (Stage-2 Modal T4 smoke); leave operator-gated per Catalog #270 dispatch protocol
- [ ] If DEFER verdict: write `feedback_dp1_pr101_op1_op2_defer_<verdict>_20260518.md` per Catalog #229 evidence-tag discipline
- [ ] codex_persistent_session_state row appended

## OPERATOR-FACING NOTE

This routing directive implements the synthesis's "Carmack-style $0 validation cascade" pattern: cheap probes BEFORE GPU spend. Result lands in 10-15 min and disambiguates whether the $3-5 Stage-2 dispatch is worth firing.

After OP-1+OP-2 land, the operator can review the composition decision table verdict via:
```bash
.venv/bin/python tools/codex_to_claude_inbox.py summary --format=text   # if inbox channel deployed
# OR
cat .omx/state/probe_outcomes.jsonl | jq 'select(.probe_id | startswith("dp1_pr101"))'
```

— Main-Claude 2026-05-18 (DP1+PR101 design-memo-authorized routing per Carmack-style cascade)
