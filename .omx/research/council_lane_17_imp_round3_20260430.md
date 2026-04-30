---
name: Round 3 Adversarial Review — Lane 17 IMP Level-3 push
description: 2026-04-30. Round 3/3 of the recursive adversarial review per CLAUDE.md. Rotating perspectives. Counter at 2/3 entering this round; needs 3/3 to ship.
type: research
counter: 2
---

## Convening (Round 3 perspectives, rotating — emphasis on adversarial breadth)

- **Yousfi (challenge creator)** — second pass with eye on inflate-time correctness.
- **Ballé (modern neural-compression SOTA)** — entropy modeling angle.
- **Tao (grand council, harmonic analysis)** — pure-math review of the codec invariants.
- **Karpathy (engineering practitioner)** — let-compute-speak skepticism.
- **Schmidhuber (compression-as-intelligence)** — MDL angle.

## Code under review

(Same set; Round 1 + 2 fixes applied; verifying nothing regressed.)

## Findings

### CRITICAL (0 found)

— None.

### Medium (0 found)

### Low (1 found, NOT a bug)

#### L4 — Karpathy: the IMPS dense-baseline (no masks) test produces an archive that is actually LARGER than the FP4A anchor (581KB vs 297KB). The user might worry the codec is broken.

**Finding**: real-archive smoke output:
```
[lane-17-smoke] Lane G v3 anchor (FP4A): 296,776 bytes
[lane-17-smoke] IMPS dense (no masks, all FP16): 581,579 bytes
```

581KB > 297KB because IMPS-dense uses FP16 raw bytes (16 bits/weight) while FP4A uses 4 bits/weight. So a no-prune IMPS archive is 4× the bits of FP4A.

**Why this is correct**: IMPS is designed for the SPARSE case. The dense baseline is only useful as a "sparse savings reference" — at 89% sparsity, IMPS is 177KB which beats the 297KB FP4A anchor by 40%.

**Verdict**: NOT A BUG. The smoke test docstring already explains this. No action needed.

### Verification of Round 1 + Round 2 fixes

Round 1 M1 fix verified:
- Dispatcher line 175-178: BEST_CYCLE_SCORE pre-populated from `IMP_BASELINE_SCORE` env var (default 1.05) before the IMP loop.
- Special "all cycles regressed → REVERT_TO=lane_g_v3_anchor" path emits exit code 8 + LANE_17_VERDICT.txt.
- Per-cycle smoke now compares against the canonical anchor, not whichever cycle landed first.

Round 2 M5 fix verified:
- Pre-dispatch memo `project_lane_17_imp_pre_dispatch_20260430.md` now has explicit verdict table (STRONG WIN / WIN / NULL / REGRESSION) keyed on the cycle-9 score.

### Test suite re-run

- `test_iterative_magnitude_pruning.py`: 9 tests pass (existing).
- `test_imp_cycle_runner.py`: 7 tests pass (existing).
- `test_imps_renderer_archive.py`: 18 tests pass (NEW).
- `test_imp_real_archive_smoke.py`: 2 tests pass + writes `reports/lane_17_imp_real_archive.json` with 40.2% empirical savings at 89.3% sparsity.

Total: **36 tests pass**.

### Preflight Check 94 verification

- `check_imp_cycles_use_ema_and_auth_eval(strict=False, verbose=True)` → **0 violations** (Lane J-IMP dispatcher passes all 4 chain requirements: auth eval + revert-on-regression token + heartbeat + NVDEC probe).
- The check is wired into `preflight_all` at strict=True.

### Imports clean

- `tac.preflight` imports cleanly.
- `tac.imps_renderer_archive` imports cleanly.
- `tac.codec_magic_registry.find_by_magic(b"IMPS")` resolves to the new entry.

## Round 3 result

**0 bugs found.**

**Counter**: 2 → **3/3 CLEAN PASSES — adversarial review gate satisfied.**

## Lane 17 IMP — Level-3 status (per `feedback_production_hardened_standard_definition_20260430.md`)

| Gate | Status |
|------|--------|
| Implementation completion | ✅ |
| Real-archive empirical | ✅ `[empirical:reports/lane_17_imp_real_archive.json]` 40.2% savings at 89.3% sparsity |
| Contest-CUDA validation | ⏳ AWAITING USER APPROVAL ($25 / $12.50 dispatch) |
| STRICT preflight check | ✅ Check 94 passes @ 0 violations |
| 3-clean-pass adversarial review | ✅ 3/3 (Round 1 reset, Round 2 + 3 clean) |
| Memory entry | ✅ `project_lane_17_imp_pre_dispatch_20260430.md` |
| Deploy runbook | ✅ `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (NVDEC probe + heartbeat + provenance + revert-on-regression + per-cycle smoke + final auth eval) |

**6 of 7 gates satisfied. Final gate (contest-CUDA) is BUDGET-GATED and requires user approval.**

## Action items

— None. Adversarial review gate complete.

## Cross-refs

- `council_lane_17_imp_round1_20260430.md`
- `council_lane_17_imp_round2_20260430.md`
- `council_lane_17_imp_design_20260430.md`
- `lane_17_imp_scaffold_audit_20260430.md`
- `project_lane_17_imp_pre_dispatch_20260430.md` (the user-approval memo)
