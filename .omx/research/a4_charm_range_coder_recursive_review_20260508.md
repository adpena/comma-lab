# A4 ChARM Range Coder + Integration — Recursive Adversarial Review

**Date:** 2026-05-08
**Subject:** ChARM 2020 conditional-Gaussian range coder + CARM2 byte-tight wire format (commit `16a2d9d0`) — pre-dispatch gate before $15 Lightning T4 GPU dispatch fires.
**Operator gate:** "you can fire A4 but i want another round of adversarial review first" (/loop iteration 3, 2026-05-08).
**Reviewer:** code-reviewer subagent `a57765019dddb6536` (read-only profile).
**Cross-references:**
- `codex_finding_charm_high_a_b_recursive_review_20260508.md` — prior 3-clean-pass review on the math fix (HIGH-A: entropy `e` double-count) and context wiring (HIGH-B). Different scope from this review, which covers the NEW range coder module and integration.

---

## Files reviewed

- `src/tac/codec/charm_range_coder.py` — new ChARM 2020 conditional-Gaussian range coder
- `src/tac/tests/test_charm_range_coder.py` — 33 tests
- `experiments/train_charm_50k_toy_substrate.py` — modified `encode_weight_with_charm`/`decode_weight_with_charm` + CARM2 wire format
- `src/tac/lossless/range_coder.py` — underlying 32-bit integer-state range coder
- `src/tac/tests/test_tac_lossless_range_coder.py` — 9 underlying tests
- `src/tac/tests/test_train_charm_50k_toy_rate_math.py` — 7 rate-math tests

## Round 1 — Math/theory perspective (Shannon LEAD, Ballé, MacKay)

**Verifications:**
- `gaussian_pmf_int8` is mathematically correct: CDF differences with tail extension capture all probability mass; `floor_prob` epsilon-floor is canonical; normalization is exact.
- `shannon_bits_for_pmf` correctly computes self-information `-log2(p)`.
- The 1%-of-Shannon convergence claim is methodologically sound at n=8192 — the Witten/Neal/Cleary coder achieves O(1/n) bits-of-entropy convergence at block length n.
- ChARM 2020 fidelity: `ChARMRangeEncoder/Decoder` correctly implement per-symbol PMF range coding, the canonical Minnen & Singh 2020 conditional structure.
- Integer-state range coder bit-state machine (32-bit FULL_RANGE, HALF, QUARTER) verified for correctness in 9 underlying tests.

**Findings:**
- **R1-1 (Medium):** `CharmHyperprior.forward` rate formula uses continuous differential entropy `0.5*log2(2*pi*e*sigma^2)`. For sigma << 1 (tight predictions), this can be NEGATIVE while actual discrete entropy is bounded ≥ 0 bits/symbol. The `clamp(min=0.0)` catches it but suppresses the gradient signal in that regime. Falsification test at epoch 500 measures ACTUAL bytes via `encode_weight_with_charm`, so the discrepancy is visible in `rate_model_gap_bytes`. Not a blocker; document and proceed.
- **R1-2 (Low):** Tests at lines 240, 297, 317 use magic constant `s - (-128)` instead of `s - alphabet_lo`. Future-maintenance hazard.

**Round 1 verdict: NOT CLEAN (1 Medium, 1 Low). Counter: 0/3.**

## Round 2 — Adversarial implementation (Quantizr, Yousfi, Carmack)

**Verifications:**
- The fp16 sidecar correctness fix is correct: encoder and decoder both derive PMFs from byte-identical fp16-quantised `(mu, log_sigma)` values. Verified by `test_carm2_wire_roundtrips_from_canonical_sidecar_pmfs`.
- No invented CLI flags (no subprocess calls in reviewed scope).
- Symbol ordering (encoder writes channel 0 then channel 1 etc.) matches decoder iteration order.
- fp16 precision for log_sigma is tight enough: ~0.1% relative error in sigma → bin probability shift of ~0.2% → quantises to same integer with overwhelming probability at `pmf_total = 32768`.

**Findings:**
- **R2-1 (Medium):** Non-deterministic ZIP archive. `build_archive` uses `zf.write()` which bakes filesystem mtime → archive_sha256 not reproducible across rebuilds. Violates CLAUDE.md preflight rule R5-r6 #5 (`check_archive_builders_use_deterministic_zip`). Custody concern.
- **R2-2 (Medium):** CARM2 stores a self-contained CHRC blob (18-byte CHRC header + range-coded bits) inside `range_coded_payload`, but the docstring describes it as "raw range-coded bits." Documentation gap; functionality correct.
- **R2-3 (Low):** Dead `num_weight_channels` parameter in `CharmContextNet.__init__` — accepted positionally, never used. Dangling-helper-in-reverse pattern.

**Round 2 verdict: NOT CLEAN (2 Medium, 1 Low). Counter: 0/3.**

## Round 3 — Engineering / contrarian (Hotz, Hinton, the Contrarian)

**Verifications:**
- `payload_len` uint16 ceiling (65535 bytes) is fail-loud (raises in `finish()`) — not a data-corruption risk.
- For 50K-param toy at ~5 bits/symbol, payload is ~31KB — well under ceiling.
- Hyperprior `z` is conservatively coded at 8 bits/element (uniform assumption); pessimistic, never optimistic.
- Complexity is justified vs flat brotli baseline (Tier 0 finding established that bolt-on Ballé fails; co-training is required path).

**Findings:**
- **R3-1 (Low):** No test exercises the `payload_len` uint16 ceiling guard path. Coverage gap.
- **R3-2 (Medium, reinforces R2-2):** CARM2 format docstring's double-framing of CHRC inside range_coded_payload is independently surfaced from engineering angle.
- **R3-3 (Low):** `decode_symbols` silently ignores extra PMFs beyond `dec.num_symbols`. Documented but untested.

**Round 3 verdict: NOT CLEAN (1 Medium reinforcing R2-2, 2 Low). Counter: 0/3.**

## Aggregate findings table

| ID | Severity | Where | Description | Dispatch blocker? |
|----|----------|-------|-------------|-------------------|
| R1-1 | Medium | train_charm_50k.py:352-360 | Continuous differential entropy underestimates discrete entropy for tight σ; clamp(min=0) suppresses gradient | No — visible in rate_model_gap_bytes; acceptable for toy ablation |
| R1-2 | Low | test_charm_range_coder.py:240,297,317 | Magic `s - (-128)` instead of `s - alphabet_lo` | No |
| R2-1 | Medium | train_charm_50k.py:1102-1105 | Non-deterministic ZIP build (mtime baked) | No for single dispatch; Yes for replay-custody |
| R2-2 | Medium | train_charm_50k.py:397+506 | CARM2 stores CHRC blob (18-byte hdr + bits) in range_coded_payload but docstring says "raw range-coded bits" | No (doc gap only) |
| R2-3 | Low | train_charm_50k.py:232,276 | Dead `num_weight_channels` parameter in CharmContextNet | No |
| R3-1 | Low | test_charm_range_coder.py | No test for payload_len uint16 ceiling guard | No |
| R3-2 | Medium | (same as R2-2) | Reinforced finding | No |
| R3-3 | Low | charm_range_coder.py:498-505 | decode_symbols silently ignores extra PMFs (documented, untested) | No |

**Final review verdict:** REVIEW BLOCKED (3-clean-pass gate not satisfied) but **NOT a DISPATCH BLOCKER** — 0 CRITICAL, no correctness bug, no invented CLI flag, no /tmp paths in artifacts, no silent data corruption. Range coder roundtrip is byte-exact. Math is correct.

## Fixes applied this session (commit `83fb8e6a`)

Three Medium findings fixed in-place at `experiments/train_charm_50k_toy_substrate.py`:

1. **R2-1 (Medium):** `build_archive` now uses `zipfile.ZipInfo` + pinned epoch (2000-01-01) + `writestr()` for byte-deterministic archive output. `archive_sha256` reproducible across replays. Per CLAUDE.md preflight rule R5-r6 #5.

2. **R2-2 (Medium, reinforced by R3-2):** Encoder docstring + decoder docstring updated to document the CARM2 → CHRC double-framing explicitly. Format readers no longer surprised by the 18-byte CHRC header inside range_coded_payload.

3. **R2-3 (Low, but free with the file edit):** `CharmContextNet.__init__` no longer accepts the unused `num_weight_channels` parameter. Call site at `CharmHyperprior.__init__` updated.

Tests after fixes: **40/40 pass** (33 charm range-coder + 7 rate-math) plus 9 underlying range-coder tests still pass. No regressions.

## Findings deferred (Low, low-EV to fix now)

- R3-1: ceiling test missing — guard path is fail-loud-on-violation, not silent
- R3-3: extra-PMFs-ignored test missing — behavior is documented

These three Low findings can be addressed in a future cleanup pass; not gating dispatch.

## R1-1 Medium: not fixed, document-and-proceed

The continuous-vs-discrete entropy bias for σ << 1 is inherent to the matched-Gaussian rate-prediction formulation. CompressAI handles the same issue with the same `clamp(min=0)` guard. For the A4 toy ablation, the falsification criterion uses ACTUAL coded bytes, not predicted rate, so the bias is observable but not corrupting. The dispatch's `build_manifest.json` already reports `rate_model_gap_bytes` for direct comparison. Operator should expect a positive gap value when the hyperprior predicts tight distributions.

## Operator authorization for A4 dispatch

The 3-clean-pass gate is NOT satisfied per CLAUDE.md non-negotiable. However, the review found NO dispatch-blocking finding. The three Medium findings actionable as code fixes have been applied (R2-1, R2-2, R2-3); the fourth (R1-1) is documentation-and-proceed.

**Operator decision needed before $15 Lightning T4 fires:**
- Authorize dispatch with explicit acknowledgment of (a) `rate_model_gap_bytes` will be non-zero for tight-distribution channels, (b) the three Low findings are deferred to future cleanup, OR
- Require another full 3-clean-pass review cycle on the post-fix code before authorizing dispatch.

## Addendum — R1-2 fixed by codex carry-forward

R1-2 is no longer deferred. `src/tac/tests/test_charm_range_coder.py` now uses
`DEFAULT_ALPHABET_LO` instead of hard-coded `-128` in the Shannon-bit
indexing checks. Verification rerun:

```bash
.venv/bin/python -m pytest src/tac/tests/test_charm_range_coder.py \
  src/tac/tests/test_train_charm_50k_toy_rate_math.py -q
```

Result: `40 passed, 3 warnings`.
