---
name: Round 1 Adversarial Review — Lane 17 IMP Level-3 push
description: 2026-04-30. Round 1/3 of the recursive adversarial review per CLAUDE.md. Rotating perspectives. Counter starts at 0/3.
type: research
counter: 0
---

## Convening (Round 1 perspectives, rotating)

- **Yousfi (challenge creator)** — every CRITICAL bug is one wasted GPU cycle.
- **Frankle (Lane 17 OG)** — rigor on the LTH protocol itself.
- **Carmack (engineering shortcuts)** — would shred 50KB of cruft from this if unchecked.
- **Hotz (raw engineering instinct)** — calls out over-engineering.
- **Contrarian (veto power)** — challenges weak arguments.

## Code under review

- `src/tac/imps_renderer_archive.py` (575 LOC, NEW)
- `src/tac/codec_magic_registry.py` (IMPS entry added)
- `submissions/robust_current/inflate_renderer.py` (IMPS handler added before ASYM, magic-byte registry docstring updated)
- `src/tac/preflight.py` (Check 94 added, wired in `preflight_all`)
- `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh` (per-cycle smoke + revert-on-regression added)
- `src/tac/tests/test_imps_renderer_archive.py` (18 tests, NEW)
- `src/tac/tests/test_imp_real_archive_smoke.py` (2 tests, real-archive empirical, NEW)

## Findings

### CRITICAL (0 found)

— None.

### Medium (3 found)

#### M1 — Yousfi: dispatcher's per-cycle smoke uses BEST score from PRIOR cycles, not from `cycle_0` baseline.

**Finding**: in the dispatcher's revert-on-regression block, `BEST_CYCLE_SCORE` is set on the FIRST cycle that lands a parseable RESULT_JSON (typically cycle 0). If cycle 0's auth eval crashes (e.g., NVDEC retry caused log loss), `BEST_CYCLE_SCORE` is set on cycle 2 instead, which is already 36% sparse. Now cycle 4's regression check is against cycle 2 (already-pruned) baseline, not the dense baseline.

**Why it matters**: in the failure mode where cycle 0 auth eval crashes silently, the dispatcher loses its "compare against dense baseline" anchor. A 30% regression from the dense renderer would not trigger if cycle 2 already absorbed 25% of the regression.

**Fix**: pre-populate `BEST_CYCLE_SCORE` from the Lane G v3 anchor's known [contest-CUDA] score (1.05) BEFORE the loop starts. This way, the regression check is always against the canonical baseline, not against whatever cycle happened to land first.

**Counter impact**: RESET to 0/3.

**Status**: FIX APPLIED in this round.

#### M2 — Frankle: the per-tensor sparsity gate (78%) is set for the SPARSE-CSR-vs-DENSE-FP4 breakeven, but the IMPS archive uses FP16 fallback (not FP4 dense). The breakeven-FP16 is at ~50% sparsity, not 78%.

**Finding**: my LTH paper reports lottery tickets emerge around 60-80% sparsity. If the IMPS gate is 78%, layers at 60-77% sparsity (which is still highly sparse + still meaningful as a lottery-ticket measurement) are encoded as FP16 raw, missing a ~25% byte savings vs sparse-CSR.

**Why it matters**: at intermediate cycles (5-7, sparsity 67-83%), the layer-by-layer eligibility wobbles right around the 78% gate. We're leaving compression on the table.

**Counter-argument (Carmack)**: lower the gate to 60% and you ship a regression on 40-60%-sparse tensors where sparse-CSR overhead exceeds savings. The 78% number is correct for sparse-CSR-vs-DENSE-FP4 (the original analysis); the FP16-fallback reframing is correct but doesn't change the gate.

**Verdict**: Frankle is half-right: the gate is correctly placed for sparse-CSR-vs-FP4, but the documentation should be clearer about why we're comparing to FP4 (not FP16) — because the alternative IMPS archive variant could ship FP4 for the dense fallback. This is an enhancement, not a bug.

**Counter impact**: documentation-only fix; not bug-class.

**Status**: NOT FIXED THIS ROUND (documentation enhancement, deferred).

#### M3 — Carmack: the `_eligible_for_sparse_csr` function is called once per Conv2d in encode, but the inflate-side decode never validates its match.

**Finding**: if a future encode change moves the gate to 50% (per M2), the decode will happily decode mixed sparse+FP16 layers without complaint. The header `kind` field is the only source of truth.

**Why it matters**: a hand-edited header could swap a `fp16_conv` to `imps_conv`, then the decoder calls `sparse_csr_decode` on FP16 raw bytes → crash with cryptic error.

**Verdict**: existing inflate-side code already raises `IMPSArchiveError` on bad magic + truncation. The kind-mismatch case raises a structured error path through `sparse_csr_decode`. Acceptable.

**Counter impact**: not a bug.

**Status**: NOT FIXED THIS ROUND.

### Low (2 found)

#### L1 — Hotz: 575-LOC `imps_renderer_archive.py` could be 100 LOC if we just compose `OWV2.encode_owv2_archive` on the masked weights.

**Finding**: the encode walk in IMPS is 90% identical to OWV2's. Differences: kind = `imps_conv` instead of `owv2_conv`, and the per-tensor codec is `sparse_csr_export` instead of `encode_omega_w_v2`.

**Counter-argument (Contrarian)**: factoring out the shared walk creates a base class / helper. CLAUDE.md "Strategic Secrecy Rule" + maintenance burden of cross-module refactor at $25 GPU spend on a deadline is a CLEAR loser. Defer to a Phase 2 cleanup.

**Verdict**: Hotz right about duplication, Contrarian right about NOT fixing it now. DEFER.

#### L2 — Contrarian: the empirical smoke test prints to stdout, fragile in CI.

**Finding**: `test_imp_real_archive_smoke.py` uses `print(...)` for the per-cycle progress output. CI runners may suppress stdout.

**Verdict**: pytest captures stdout by default; the report file is the source of truth. The print is for human-friendly local runs (`-s` flag). NOT A BUG.

## Round 1 result

**1 medium fix applied (M1: pre-populate BEST_CYCLE_SCORE from Lane G v3 anchor before loop).**

**Counter**: 1 fix → RESET to **0/3**.

## Action items

- [x] Fix M1 in dispatcher: set `BEST_CYCLE_SCORE` to "1.05" before the loop starts.
- [ ] M2 doc enhancement (deferred, not bug-class).
- [ ] L1 refactor (deferred to Phase 2).

## Cross-refs

- `council_lane_17_imp_design_20260430.md` (Phase B council vote)
- `lane_17_imp_scaffold_audit_20260430.md` (Phase A audit)
- `feedback_production_hardened_standard_definition_20260430.md` (Level 3 standard)
