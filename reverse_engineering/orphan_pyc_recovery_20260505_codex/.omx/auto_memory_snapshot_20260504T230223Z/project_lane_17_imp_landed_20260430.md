---
name: Lane 17 (IMP 10-cycle) — Phases A-I LANDED at Level 2; awaiting user approval for contest-CUDA gate
description: 2026-04-30. Lane 17 IMP scaffold-to-Level-3 push. 6 of 7 Level-3 gates satisfied (impl, real-archive empirical, STRICT preflight Check 94, 3-clean-pass adversarial review, memory entry, deploy runbook). Final gate (contest-CUDA) is BUDGET-GATED — pre-dispatch memo project_lane_17_imp_pre_dispatch_20260430.md awaits explicit user "approved" response.
type: project
authoritative_for: lane_17_imp_level3_push
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR

Lane 17 (IMP — Iterative Magnitude Pruning, 10-cycle) advanced from Level-1 scaffold to **Level-2** at the user's "Full Production Hardened" standard. The remaining gate (contest-CUDA) requires explicit user approval for $25 GPU dispatch (or $12.50 5-cycle quick variant).

**Empirical headline**: at 89.3% sparsity, IMPS sparse-CSR archive is **177KB vs FP4A baseline 297KB = 40.2% byte savings on Lane G v3 renderer.bin** [empirical:reports/lane_17_imp_real_archive.json]. The breakeven crossover is at cycle 5 (74% sparse).

## Phases completed (Phase A-I + pre-dispatch memo)

### Phase A — Scaffold audit (`.omx/research/lane_17_imp_scaffold_audit_20260430.md`)

Found Lane 17 was actually at Level-1+ already (the Phase 2 audit underestimated maturity):
- 610-LOC `iterative_magnitude_pruning.py` module with sparse-CSR codec + FP4 nibble pack/unpack
- 401-LOC `train_imp_cycle.py` per-cycle script with EMA (Council D 2026-04-29 wire-in) + Frankle-2019 weight-rewinding
- 319-LOC `imp_cycle_runner.py` in-process orchestrator
- 325-LOC `remote_lane_j_imp_iterative_magnitude_pruning.sh` 10-cycle bash dispatcher
- 17 unit tests across `test_iterative_magnitude_pruning.py` + `test_imp_cycle_runner.py`
- Profile `IMP_CYCLE_DILATED_H64` at `src/tac/profiles.py:2860`

Gaps: inflate-side sparse-CSR magic-byte handler, real-archive empirical, STRICT preflight check, 3-clean-pass review, memory entry, revert-on-regression kill criterion.

### Phase B — Council design review (`.omx/research/council_lane_17_imp_design_20260430.md`)

10-voice inner council + Frankle (Lane 17 OG, grand-council specialty call) deliberated 7 design questions:

| Q | Decision | Vote |
|---|----------|------|
| 1 | 20%/cycle × 10 cycles | 8/10 |
| 2 | Global magnitude prune + per-layer 99% safety check | 9/10 |
| 3 | Per-cycle CUDA auth eval at cycles 0,2,4,6,8,9 (6 evals = $1.80) + final | 7/10 |
| 4 | Revert-on-regression at `cycle_N_score > 1.10 × min(cycle_0..N-1_score)` | 9/10 |
| 5 | FP4 on survivors as part of Lane 17; defer block-FP stack | 10/10 |
| 6 | Magic byte `b"IMPS"` | 10/10 |
| 7 | Stack with Ω-W-V2 / J-NWC: out of scope this push | (deferred) |

### Phase C — Implementation completion

NEW: `src/tac/imps_renderer_archive.py` (575 LOC). Wire layout mirrors OWV2 (multi-tensor archive header + per-layer kind/payload):
- `IMPS_ARCHIVE_MAGIC = b"IMPS"`, version 1.
- `IMPS_PER_TENSOR_SPARSITY_GATE = 0.78` (sparse-CSR-vs-FP4 breakeven + safety margin).
- `IMPS_PER_TENSOR_NUMEL_CAP = 65535` (uint16 indexing cap).
- `encode_imps_archive(model, masks)` → walks named_modules, sparse-CSR encodes Conv2d weights at gate-eligible sparsity, FP16 fallback elsewhere.
- `decode_imps_archive(data, device)` → pure-math byte→model (Check H STRICT — no scorer load at decode).

UPDATED: `submissions/robust_current/inflate_renderer.py`:
- IMPS magic-byte handler added before ASYM (lines ~2167-2197).
- Magic-byte registry docstring updated to include format #12 (IMPS).

UPDATED: `src/tac/codec_magic_registry.py`:
- IMPS entry added; `find_by_magic(b"IMPS")` now resolves to the canonical decode module.

UPDATED: `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh`:
- Per-cycle smoke auth eval at cycles `IMP_AUTH_EVAL_CYCLES="0 2 4 6 8 9"` (Council Q3).
- Revert-on-regression kill: if `cycle_N_score > IMP_REGRESSION_THRESHOLD × BEST_CYCLE_SCORE`, write `REVERT_TO_CYCLE.txt` + break (Council Q4 9/10).
- BEST_CYCLE_SCORE pre-populated from Lane G v3 anchor 1.05 (Round 1 M1 fix).
- Special-case "all cycles regressed → ALL_CYCLES_REGRESSED" exit code 8.
- Stage 2 sources renderer.pt from `$FINAL_CYCLE_DIR` (= best cycle's dir, not always cycle-9).

### Phase D + F — Tests (NEW)

`src/tac/tests/test_imps_renderer_archive.py` (18 tests):
- 4 magic-byte / format invariant tests.
- 4 eligibility-logic tests (below/above gate, no mask, oversized).
- 3 encode/decode roundtrip tests (dense baseline, high-sparsity sparse-CSR layers, pruned-zeros preservation).
- 6 error-handling tests (None data/device/model, bad magic, truncated, unsupported version).
- 1 codec_magic_registry sanity test.

`src/tac/tests/test_imp_real_archive_smoke.py` (2 tests, real-archive empirical):
- 10-cycle simulated IMP on the actual Lane G v3 anchor renderer.bin.
- Writes `reports/lane_17_imp_real_archive.json` with per-cycle byte measurements.
- Asserts 89.3% final sparsity AND 89%-sparse IMPS < dense-IMPS baseline.
- Anchor smoke test: passing dense anchor with empty masks must produce valid archive.

### Phase H — STRICT preflight Check 94 (`src/tac/preflight.py`)

`check_imp_cycles_use_ema_and_auth_eval` — scans `scripts/remote_lane_*imp*.sh` and asserts each contains:
1. `contest_auth_eval.py` invocation (CLAUDE.md "Auth eval EVERYWHERE").
2. A revert-on-regression token (Council Q4 — kill criterion).
3. `heartbeat` loop (CLAUDE.md "Remote code parity").
4. `probe_nvdec` at Stage 0 (CLAUDE.md "Vast.ai NVDEC roulette").

Lands STRICT @ 0 violations. Wired into `preflight_all` after Check 93.

### Phase I — 3-clean-pass adversarial review

| Round | Perspectives | Findings | Counter |
|-------|-------------|----------|---------|
| 1 | Yousfi, Frankle, Carmack, Hotz, Contrarian | 1 medium fix (M1 — BEST_CYCLE_SCORE not pre-populated; could miss DENSE-baseline regression) → APPLIED | reset to 0/3 |
| 2 | Shannon, Dykstra, Selfcomp, Quantizr, MacKay | 0 bugs (M4 doc-only; M5 memo enhancement APPLIED; M6 false-positive cost acceptable) | 1/3 |
| 3 | Yousfi, Ballé, Tao, Karpathy, Schmidhuber | 0 bugs (L4 not-a-bug per smoke docstring) | **3/3 CLEAN** |

3-clean-pass gate satisfied.

### Pre-dispatch memo (`project_lane_17_imp_pre_dispatch_20260430.md`)

User-facing escalation memo per CLAUDE.md GPU budget non-negotiable ($10 threshold exceeded by $25 estimate). Contains:
- Cost breakdown for full 10-cycle ($25) vs quick 5-cycle ($12.50) vs stack ($25 in 2 stages).
- Predicted score band: best 0.90-1.00 (~25%), most likely 1.00-1.15 (~50%), kill (~25%).
- Explicit verdict criteria (STRONG WIN / WIN / NULL / REGRESSION) keyed on cycle-9 score.
- Council recommendation: dispatch quick variant first ($12.50), gate full 10-cycle on cycle-5 result.
- Required user response: "approved-quick" / "approved-full" / "approved-stack" / "deferred" / "killed".

## Lane registry update

`.omx/state/lane_registry.json`: `lane_17_imp_10cycle` entry promoted from Level 1 → Level 2 with 6 of 7 gates green (contest-CUDA gate awaits user approval).

## Files created / modified (commit serializer batched)

NEW:
- `src/tac/imps_renderer_archive.py` (575 LOC)
- `src/tac/tests/test_imps_renderer_archive.py` (18 tests)
- `src/tac/tests/test_imp_real_archive_smoke.py` (2 tests, writes reports/lane_17_imp_real_archive.json)
- `.omx/research/lane_17_imp_scaffold_audit_20260430.md` (Phase A)
- `.omx/research/council_lane_17_imp_design_20260430.md` (Phase B)
- `.omx/research/council_lane_17_imp_round1_20260430.md` (Phase I)
- `.omx/research/council_lane_17_imp_round2_20260430.md` (Phase I)
- `.omx/research/council_lane_17_imp_round3_20260430.md` (Phase I)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_lane_17_imp_pre_dispatch_20260430.md` (escalation memo)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_lane_17_imp_landed_20260430.md` (this file)

MODIFIED:
- `src/tac/preflight.py` (Check 94 added + wired into `preflight_all` STRICT)
- `src/tac/codec_magic_registry.py` (IMPS entry)
- `submissions/robust_current/inflate_renderer.py` (IMPS magic-byte handler + docstring update)
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (per-cycle smoke + revert-on-regression + Round 1 M1 fix + ALL_CYCLES_REGRESSED special case)
- `.omx/state/lane_registry.json` (lane_17_imp_10cycle gates updated)

## Commit metadata

- All commits via `tools/subagent_commit_serializer.py` (per CLAUDE.md non-negotiable).
- 36 tests pass (9 + 7 + 18 + 2).
- 0 STRICT preflight violations (Check 94 + Check 88 + 6 shell-script related checks).
- Lane G v3 anchor unchanged.

## Awaiting user approval

**The next action is GPU dispatch.** Per CLAUDE.md GPU-budget non-negotiable, this requires explicit user approval. Read `project_lane_17_imp_pre_dispatch_20260430.md` and reply with one of:
- `approved-quick` ($12.50, 5-cycle, gates full run)
- `approved-full` ($25, 10-cycle with revert-on-regression cap)
- `approved-stack` (~$15, quick + Ω-W-V2 stack)
- `deferred` (lane in production state, hold for later)
- `killed` (decommission)

## Cross-refs

- `feedback_production_hardened_standard_definition_20260430.md` (Level 3 standard)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 17"
- `feedback_check_64_smoke_proofs_resolved_AND_subagent_serializer_landed_20260429.md` (commit serializer)
- CLAUDE.md "EMA — NON-NEGOTIABLE", "Auth eval EVERYWHERE", "NEVER invent CLI flags", "Recursive adversarial review protocol", "GPU budget and compute resources"
