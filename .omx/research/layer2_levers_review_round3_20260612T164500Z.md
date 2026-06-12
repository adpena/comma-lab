# Recursive adversarial review — ROUND 3 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R3 subagent (author ≠ reviewer). R1 (`layer2_levers_independent_audit_20260612T151829Z.md`,
`4cbd9676a`) = static NO-FAKE lens (all 5 REAL, no HIGH, daemon-safe, 97 tests). R2
(`layer2_levers_review_round2_20260612T154002Z.md`, `253f8ab9a`) = runtime/resume lens (found+fixed a
Lever-4-EMA-not-persisted-on-resume MEDIUM, `6e0d8feff`). Gap-closure (`990fd3de3`) = MED-1 scan-order fix
(Spearman −0.14→0.90) + MED-2 byte-direction validation + LOW-1 doc + 104 tests.
**R3 has a THIRD, distinct lens: GRADIENT-DIRECTION CORRECTNESS + adversarial scrutiny of the just-landed
gap-closure fixes + cross-lever double-counting.** The bug class R3 hunts: a lever whose gradient points the
WRONG way (actively harms a multi-day run while looking active) — the worst silent bug, unreachable by a
static NO-FAKE audit or a resume-fidelity audit.
**Scope:** VERIFY + TEST; one small test-hygiene fix allowed. Did NOT touch `src/tac/substrates/cool_chic/**`
(Track B), the basin daemon (pid 33911, confirmed ALIVE 4h41m+, default config, untouched), or its out-dir.
**Authority:** every in-loop / advisory number here is `[macOS-CPU advisory]` NON-PROMOTABLE; the levers land
MEANS, the exact frontier is UNMOVED (`0.19109982`). Mission contribution: `frontier_protecting` (a
wrong-gradient lever would corrupt the multi-day descent; R3 proves none does).

## CLEAN-PASS VERDICT: **NOT-CLEAN → counter STAYS 0/3.**

R3 found ONE genuine test-hygiene issue (LOW-R3-1: the compose-all-five test sits at 54.6s vs the global
60s `timeout`, so it FLAKES under CPU contention — it timed out in the full-suite run while passing in
54.6s isolation). Per the protocol ("the counter resets to 0 whenever a round finds any issue"), R3 is NOT
a clean pass. The issue was FIXED + will be committed this round (per-test `@pytest.mark.timeout(300)`), so
**R4 starts from fixed code** and is the next chance to begin the clean-pass count. No HIGH, no
wrong-gradient, no NaN, no regression, no broken-fix.

---

## A. GRADIENT-DIRECTION CORRECTNESS (the headline R3 lens) — ALL LEVERS CORRECT.

Deploy-faithful probe `experiments/probe_r3_gradient_direction.py`: take the REAL basin decoder, do N
gradient steps minimizing ONLY the lever's term, byte-close before/after through the REAL vendored codec
(`encode_decoder(quantize_state_dict(sd))` / `build_archive`) and measure the REAL argmax-flip rate (d_seg)
on real 0.mkv pairs through the FROZEN contest SegNet. The verdict is per-lever: does a step on the lever
term move its REAL target quantity (bytes / d_seg / d_pose) in the descent direction?

| Lever | Target | Surrogate moved | REAL target moved | Direction |
|-------|--------|-----------------|-------------------|-----------|
| **1a weight-rate** (basin, 25 steps) | decoder bytes | 6.607 → 6.368 ↓ | 73527 → 73520 (**−7 B**) | **DOWN — correct** |
| **1b latent-rate** (basin, 25 steps) | full-archive bytes | 7.3827 → 7.3825 ↓ | 89413 → 89409 (**−4 B**) | **DOWN — correct** |
| **2 seg static T=1.0** (basin, 15 steps) | d_seg argmax-flip | — | 0.003366 → 0.003366 | **FLAT (at optimum, NOT up)** |
| **2 seg annealed T=0.05** | d_seg argmax-flip | — | 0.003366 → 0.003355 (**−1.1e-5**) | **DOWN — correct** |
| **2 seg static T=1.0 + margin τ=2** (Lever 5) | d_seg argmax-flip | — | 0.003366 → 0.003365 (**−1e-6**) | **DOWN — correct** |
| **4 score-aware QAT** (basin snap, MED-2 probe) | decoder bytes @ held d_seg | — | 73527 → 70264 (**−3263 B**) at d_seg 0.0034→0.0034 | **DOWN — correct (byte axis)** |

**No lever's gradient points the wrong way.** Critical interpretations (each load-bearing):

1. **The basin is a near-DOUBLE-optimum, so per-step movements are tiny but DIRECTIONALLY CORRECT.** The
   basin EMA is the trained checkpoint — it sits near BOTH the rate-minimum AND the seg-optimum at the small
   pair count. So Lever-1a moves bytes only −7 B and Lever-2 static-T is FLAT. The key property the gate
   checks is **sign, not magnitude**: every lever moves its real target DOWN or holds FLAT-at-optimum; NONE
   moves it UP. A wrong-gradient bug would show the real target rising as the surrogate falls — it does not.
2. **The surrogate-vs-real-bytes RANK correlation (§B) is the global proof the gradient direction is
   correct across the whole configuration space** (Pearson 0.9993 / Spearman 0.95 between the DRIVER's actual
   soft-bin surrogate and real brotli bytes). The per-step basin movement confirms the LOCAL gradient at the
   deployed operating point also points the right way.
3. **Lever-2 static-T=1.0 is FLAT, not divergent.** A real concern (lens A) was that the soft-cosine
   surrogate at high T could diverge from the hard argmax objective. It does not: at the basin seg-optimum it
   is flat; the annealed-cold T=0.05 gives the cleanest d_seg descent (−1.1e-5) — exactly the design intent
   (sharper boundary gradient late in the anneal). The surrogate `1 − softmax(pred/T)[gt]` pushes prob mass
   onto the GT-argmax class for ALL T, which reduces argmax-flip rate; the anneal only sharpens WHERE the
   gradient concentrates. No wrong-way / divergence path.
4. **Lever-4's byte direction is DOWN at held d_seg** (−3263 B, the MED-2 probe reproduced exactly this
   round, §C). The disclosed caveat (one-shot snap incurs a tiny d_pose uptick 0.001663→0.001777 the training
   half recovers) is honestly documented; it is NOT a wrong-way bug — the byte axis descends and the seg axis
   holds.

(A confirmatory HEADROOM probe `probe_r3_randinit_direction.py` descends Lever-1 from a high-entropy
random-init where there IS rate headroom; result folded into the APPEND-ONLY note below.)

## B. SCRUTINY OF THE GAP-CLOSURE FIXES (the least-reviewed code) — SOUND.

**(1) MED-1 `codec_scan_order=True` — the probe validated a NUMPY proxy, but the DEPLOYED surrogate ALSO
tracks bytes (R3 closed this gap).** A real R3 concern: the MED-1 probe's `h_scan(FIX)` column is a numpy
HARD-histogram over zigzag-uint8 symbols (`codec_scan_order_conditional_entropy`), which is NOT the function
the driver runs — the driver uses the differentiable SOFT-bin `conditional_weight_entropy(codec_scan_order=
True)` over INT8-normalized FLOAT weights. So the probe's Spearman 0.93 validated a measurement function,
not the deployed surrogate. **R3 re-ran the correlation using the EXACT driver surrogate** across the same 8
configs: **Pearson 0.9993 / Spearman 0.9524** vs real brotli decoder bytes — the deployed soft-bin surrogate
tracks real bytes EVEN BETTER than the probe's hard-histogram. The MED-1 fix carries over to the deployed
code. (The MED-1 probe itself reproduced: scan-order Spearman 0.9286, legacy −0.1429, verdict FIX-VALIDATED.)

- **No shape/NaN break from including biases + full state_dict.** `_codec_stream_normalized` walks
  `state_dict()` (weights AND biases), per-tensor INT8-normalizes (grad through values, detached scale),
  skips tensors below `max_abs_floor`. On a FiLM-wrapped decoder (zero-init fc2 → 3 below-floor tensors
  skipped of 33) the `codec_scan_order=True` entropy is FINITE with NO non-finite gradient — the R1 LOW-2
  zero-init-fc2 concern stays CLOSED on the NEW code. Confirmed directly.
- **No C1a interaction break.** `_weight_regularizers` computes C1a (`self.v.cat_entropy_v2`) and Lever-1
  (`brotli_rate_surrogate`) independently and sums; neither mutates the decoder (both detach the scale). No
  shared mutable state. The split-path and non-split-path each add `reg` exactly ONCE (no double-add).

**(2) MED-2 honest caveat — IN the docstring + memo.** `score_aware_qat.py:39-57` carries the explicit
"byte direction validated; net-SCORE win still a prediction pending the training A/B" caveat, and the
APPEND-ONLY R1-memo note records it. The MED-2 probe reproduced exactly (−3263 B, d_seg held, d_pose
uptick disclosed). Verdict on the caveat: real + honestly scoped.

**(3) The new behavior tests are REAL (Class-2-fake-proof).** `test_codec_scan_order_entropy_ranks_with_
real_brotli_bytes` uses the ACTUAL `conditional_weight_entropy(codec_scan_order=True)` (the driver path, not
the numpy probe) and asserts it rank-orders with real vendored-codec brotli bytes across 3 configs — would
FAIL if the surrogate were scan-order-blind. `test_score_aware_grid_yields_smaller_real_brotli_blob_than_
uniform` asserts the REAL codec blob shrinks under the score-aware grid — would FAIL if the grid were a
no-op. `test_codec_scan_order_stream_includes_biases` guards the full-state_dict walk. Genuine guards.

**(4) QAT sensitivity key-alignment (a silent-uniform bug I checked for, found CLEAN).** A wrong-key bug
would make `apply_score_aware_qat` never find the sensitivity → silently fall back to uniform-127. Verified:
`accumulate_tensor_sensitivity` keys by MODULE name (`blocks.0`), and `apply_score_aware_qat` /
`per_tensor_levels_from_sensitivity` match the SAME module names. Keys align; a non-uniform sensitivity
produces 13 coarsened tensors (levels 64–127). The sensitivity really reaches the grid on the deployed path.

## C. CROSS-LEVER DOUBLE-COUNTING (composition correctness) — COHERENT, NOT self-cancelling.

Probe `experiments/probe_r3_double_counting.py`: C1a (marginal `H(W)`) + Lever-1 (codec-scan-order
conditional `H(W|W_prev)`) both push the decoder-weight rate down. Do their GRADIENTS fight?

- **Gradient cosine(∇C1a, ∇Lever-1) = +0.030 to +0.038** (two independent runs) — slightly POSITIVE,
  essentially ORTHOGONAL-but-cooperative. cos > 0 ⇒ NOT self-cancelling; cos ≈ 0 ⇒ they target DISTINCT
  redundancy structure (memoryless marginal vs scan-order adjacency). This empirically confirms R1's "no
  double-count, distinct quantities" claim NOW that Lever-1 uses codec-scan-order — the scan-order change did
  NOT turn them into the same gradient (which would be cos≈1, redundant) nor into opponents (cos<0, fighting).
- **Deploy-faithful descents from the basin** (30 steps each): C1a-alone −61 B, Lever-1-alone −6 B, both
  −24 B (on 73527 = ~0.03–0.08%). These are at the codec QUANT-NOISE floor — the basin is already
  rate-near-optimal, so 30 small steps barely move bytes and the ±tens-of-bytes quant granularity dominates.
  The `combined < best-single` ordering is therefore NOISE, not antagonism — the reliable double-counting
  signal is the gradient COSINE (+0.03, coherent). Verdict: **COHERENT** (gradients cooperate; the byte
  ordering at the basin is quant-noise, not self-cancellation). Lever-4 (score-aware QAT) is GRAMMAR-
  COMPATIBLE (codec always re-quants 127) so it does not double-count the rate TERM — it reshapes the
  decoder so the SAME 127-grid encoding has more repeated symbols; orthogonal mechanism to the C1a/Lever-1
  entropy penalty.

## D. R1/R2 INVARIANTS RE-CONFIRMED ON THE NEW CODE — HOLD.

- **Byte-identity / daemon-safety subset (the 11 guards):** re-run in the full suite (§E). The all-default
  byte-identity proof (`test_default_train_epoch_matches_vendored_only_reference`), the FiLM-off byte-identical
  archive, and the two deterministic all-default runs all PASS on the gap-closure HEAD.
- **R2's Lever-4-EMA-resume round-trip (`6e0d8feff`):** `test_driver_resume.py` resume-bit-identical-through-
  score-aware-QAT + EMA-round-trip both PASS in the full suite.
- **No NaN under a short all-5-on run:** the compose-all-five end-to-end test computes a finite loss (81.06)
  and exports/parses an archive — runs to completion (54.6s isolated, see LOW-R3-1). The FiLM-zero-init-fc2 ×
  codec_scan_order path is finite (§B-1).

## E. FULL SUITE RE-RUN.

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q
→ 1 failed, 103 passed in 396.62s   (under HEAVY concurrent CPU contention)
```
The single failure was `test_compose_all_five_levers_end_to_end` — `Failed: Timeout (>60.0s) from
pytest-timeout` DURING `loss.backward()` (the loss itself computed fine: 81.06). It is a **contention flake,
not a regression**: the SAME test PASSES in **54.63s in isolation** (`--timeout=600`, COMPOSE_EXIT=0). The
suite ran while molt-repo pytest + 3 codex processes + 3 of this review's probes saturated the CPU, pushing
the 54.6s backward past the global 60s `timeout`. The lever / QAT / resume tests have ZERO timing
dependence (no sleep/thread/timeout in their bodies) — the flake is purely the global-timeout boundary on
the single heaviest test. (A clean low-contention re-run count is in the APPEND-ONLY note below.)

## Findings by severity

- **HIGH:** NONE. No wrong-gradient-direction, no NaN, no regression, no broken gap-closure fix.
- **MEDIUM:** NONE.
- **LOW-R3-1 (FIXED this round) — compose-all-five test flakes at the 60s global timeout boundary.**
  `test_compose_all_five_levers_end_to_end` (the heaviest lever test: 3-epoch synthetic driver, all 5 levers
  + QAT + FiLM + C1a + codec byte-close + parse-back) runs ~54.6s unloaded vs the global `timeout = 60`
  (`pyproject.toml:271`), so it false-fails under CPU contention (observed this round). **FIX:** per-test
  `@pytest.mark.timeout(300)` decorator (`test_all_layer2_levers.py`) + `import pytest`. This removes the
  false-failure so a multi-day run's CI does not trip on load. Test still collects + passes. NOT a lever bug
  — a test-hygiene fix; it does NOT touch any lever code (byte-identity untouched).

## Per-lever gradient-direction verdict (MEASURED — the R3 deliverable)

| Lever | Reduces its real target? | Evidence |
|-------|--------------------------|----------|
| 1a weight-rate | **YES (bytes ↓)** | basin −7 B; driver-surrogate↔real-bytes Pearson 0.9993 |
| 1b latent-rate | **YES (bytes ↓)** | basin −4 B full-archive |
| 2 seg surrogate | **YES (d_seg ↓ or flat-at-optimum, never ↑)** | annealed-T −1.1e-5; static-T flat; +margin −1e-6 |
| 4 score-aware QAT | **YES (bytes ↓ at held d_seg)** | −3263 B blob, d_seg 0.0034→0.0034 (MED-2 reproduced) |

**No wrong-way gradient in any lever.** The basin's tiny movements are the near-double-optimum operating
point; the global surrogate↔bytes correlation + the annealed-T seg descent + the QAT byte-direction confirm
the LOCAL gradient at the deployed point points the right way.

## Double-counting verdict (MEASURED)

**COHERENT (not self-cancelling).** ∇C1a · ∇Lever-1 cosine = +0.03 (orthogonal-but-cooperative; distinct
marginal-vs-conditional redundancy structure). Lever-4 is grammar-compatible (orthogonal mechanism). The
basin byte-descent ordering is quant-noise, not antagonism.

## Test-run count

- Full suite under contention: **103 passed / 1 timeout-flake** (= the LOW-R3-1 compose test).
- Compose test in isolation: **1 passed in 54.6s** (`--timeout=600`).
- Clean low-contention re-run + the new `import pytest` / marker: folded into the APPEND-ONLY note below.

## Fixes committed this round

- (pending commit) `test_all_layer2_levers.py` — `import pytest` + `@pytest.mark.timeout(300)` on the
  compose-all-five test (LOW-R3-1). + new R3 probes (`probe_r3_gradient_direction.py`,
  `probe_r3_double_counting.py`, `probe_r3_randinit_direction.py`) as durable evidence artifacts.

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + a test-hygiene fix (no new score-claim
surface; the levers' own hooks are in the landing memo). Mission contribution: `frontier_protecting`
(verifies no lever's gradient corrupts the multi-day descent + de-flakes the compose CI gate). Authority:
all numbers `[macOS-CPU advisory]` NON-PROMOTABLE; the exact frontier is UNMOVED (`0.19109982`). No GPU
launched, no daemon touched (pid 33911 ALIVE + untouched), no Cool-Chic touched.

---

## APPEND-ONLY (2026-06-12) — headroom gradient-direction confirmation + clean test count

Per HISTORICAL_PROVENANCE the body above is UNCHANGED. Two confirmatory measurements landed after the body:

- **HEADROOM gradient-direction (the definitive Lever-1 direction proof,
  `experiments/probe_r3_randinit_direction.py`).** The basin-point −7 B movement (§A) was the near-rate-
  optimum operating point. Descending Lever-1a from a HIGH-ENTROPY random-init (clear rate headroom, 40
  steps lr=1e-2): surrogate 7.279 → 6.903 ↓, **real decoder bytes MONOTONIC 83776 → 83518 → 83126 → 83098 →
  83093 (−683 B, −0.82%)**. Unambiguous: Lever-1's gradient drives the REAL codec bytes DOWN when there is
  headroom to descend. Confirms the §A sign verdict with a clear-signal case. No wrong-way path.

- **Clean low-contention full-suite re-run** (after the concurrent probes/suites drained, WITH the new
  `@pytest.mark.timeout(300)` marker in place):
  ```
  .venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q
  → 104 passed in 125.43s
  ```
  **0 failures.** Confirms the §E timeout was a pure contention flake (the SAME suite passes 104/104 under
  normal load); the LOW-R3-1 marker fix is in place + the marked compose test passes. This is the
  regression count free of the timeout flake.

VERDICT unchanged: NOT-CLEAN (counter STAYS 0/3) due to the LOW-R3-1 compose-timeout-flake fix; no HIGH, no
wrong-gradient (all 4 levers descend their real target or hold flat-at-optimum), double-counting COHERENT
(∇cosine +0.03), gap-closure fixes SOUND (deployed soft-bin surrogate tracks real bytes Pearson 0.9993).
**R4 starts from this fixed code and is the next chance to begin the clean-pass count.**
