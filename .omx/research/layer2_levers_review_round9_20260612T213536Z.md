# Recursive adversarial review — ROUND 9 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** Partner-A (author ≠ reviewer). Per the orchestrator's binding directive, the SEAL
requires **3 FRESH consecutive clean rounds R9, R10, R11** (the R8 instrument-bug finding means the
counter is treated as starting fresh from 0/3 at R9). Prior rounds, all on disjoint lenses:
R1 static NO-FAKE; R2 runtime/resume (fixed Lever-4-EMA-not-persisted-on-RESUME); R3 gradient-direction;
R4 deployed-archive e2e (CLEAN); R5 determinism/Muon/long-run-stability (CLEAN); R6 multi-QAT-stage
integration (fixed Lever-4-EMA-reset-at-QAT→QAT-boundary, NOT-CLEAN); R7 lever-state-persistence MATRIX
(CLEAN); R8 real-scorer paired smoke — ZERO lever findings, but found+fixed a probe-instrument
measurement-validity bug (stale-DONE skip + EMA-lagged vacuous-D), so NOT a clean lever advance.

**R9 has the NINTH, distinct lens: NUMERICAL STABILITY UNDER THE ANNEAL TAIL (T → 0.05).** The live
distortion arm (PROVENANCE.json: `lever2_temperature_anneal: "1.0->0.05"`, `lever2_seg_surrogate:
soft_cosine`, `lever5_margin_weight_tau: 2.0`) anneals the seg-surrogate prediction-softmax temperature
to the **coldest tail T=0.05**. At that tail the surrogate computes `softmax(pred/0.05) = softmax(pred·20)`
and the margin lever computes `exp(−margin/τ)`. R9 asks: do these arithmetics stay FINITE — in both the
LOSS VALUE *and the GRADIENTS* — when the SegNet logits are adversarial (saturated / tied / huge-magnitude
/ near-tie), not just benign random? A NaN/Inf gradient at the tail would silently poison the live arm's
optimizer step and was NOT covered: the prior anneal-tail test (`test_lever2_anneal_at_t_min_over_long_
stage_is_clamped_and_surrogate_finite`) checked only the VALUE, only on CLEAN random logits, only on
soft_cosine, with NO margin lever and NO `.backward()`.

**Scope:** VERIFY + TEST (additive coverage — two new tests pinning an already-correct behavior; NOT a
defect fix). Did NOT touch any lever CODE file — `git diff --stat` shows the ONLY change under my
authorship is `src/tac/torch_vehicle/tests/test_all_layer2_levers.py` (+103). (A concurrent partner's
in-flight `driver.py` change exists in the working tree at the archive-build region lines ~284 + ~1437 +
`__all__` — NOT the lever region 93–178, NOT mine; the serializer commit stages ONLY my test file via
explicit `--files`, leaving the partner's working-tree change untouched.) Did not touch
`src/tac/substrates/cool_chic/**` (Track B), the live distortion arm (out-dir
`experiments/results/distortion_arm_l235_20260612T205102Z`, confirmed ALIVE at global_epoch 477+,
stage `stage1_v328_ce`, wall 2510 s+), or its out-dir. The daemon loaded code at its launch HEAD
`d75dee4ea` and a running Python process does NOT reload edited .py — my additive test change is
resume-bit-identical-compatible by construction (it adds NO lever code, changes NO lever code path).

**Authority:** every number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic logit fixtures +
fp32-arithmetic reasoning, RESEARCH-ONLY); the levers land MEANS, the exact frontier is UNMOVED
(`0.19109982419209975` contest-CPU per `.omx/state/canonical_frontier_pointer.json`). Mission
contribution: `frontier_protecting` (the anneal-tail finiteness proof guards the live multi-day arm's
optimizer against a silent NaN-poison at the T=0.05 tail it is descending toward).

## CLEAN-PASS VERDICT: **CLEAN → counter ADVANCES 0/3 → 1/3.**

R9 found **ZERO lever findings** (no HIGH, no MEDIUM, no LOW). Across all 3 selectable surrogates
(`soft_cosine` / `fisher_rao` / `sinkhorn`) × 6 adversarial logit families (tied / saturated / huge /
extreme / near-tie / normal) × 3 margin-tau values ({None, 2.0 (the live value), 1e-6 (the clamp floor)})
— **54 combinations** — the seg loss at the annealed tail T=0.05 produces a FINITE loss AND FINITE
gradients, in `[0,1]`, MEASURED. The only non-finite point found anywhere is fp32 INPUT-arithmetic
overflow (`|logit| ≳ 1.7e37` makes `pred/0.05 = pred·20` exceed fp32's 3.4e38 ceiling BEFORE softmax can
max-subtract) — that is **not a lever defect** (it is the intrinsic fp32 representable limit, ~10^36×
beyond any real SegNet logit, which R8 measured at O(10)). Per the protocol ("a round with zero issues is
a clean pass"), R9 advances the counter to **1/3**.

The two tests ADDED this round PIN an already-correct behavior (the behavior was MEASURED finite before the
tests were written), so they do NOT reset the counter — they are the durable guard the bug class ("a lever
arithmetic goes NaN/Inf at the annealed tail") cannot silently re-open against.

---

## A. THE ANNEAL-TAIL NUMERICAL-STABILITY MATRIX (the headline R9 lens) — MEASURED, CLEAN.

### A.0 The two arithmetic surfaces at T=0.05

| Surface | Formula at the tail | Overflow/NaN risk analyzed |
|---------|--------------------|-----------------------------|
| **Lever-2 surrogate softmax** | `softmax(pred / 0.05) = softmax(pred·20)` | `F.softmax` is internally max-stable (subtracts row-max) → finite for any finite input; the GT one-hot logits (`onehot·30/0.05 = onehot·600`) softmax to exactly one-hot (no overflow, onehot ∈ {0,1}). |
| **Lever-5 margin weight** | `exp(−margin / max(τ, 1e-6))`, `margin = (top1 − top2).clamp_min(0) ≥ 0` | `−margin/τ ≤ 0` always (margin ≥ 0 by `clamp_min`) → `exp(·) ∈ (0, 1]` → CANNOT overflow; underflows to 0 (finite) when `margin/τ` is huge. |
| **fisher_rao `acos`/`torch.where`** | `acos(bc.clamp(max=1−eps))`, masked by `torch.where(bc≥1−eps, 0, ·)` | the classic torch.where-NaN-grad trap: clamp keeps `bc_safe ≤ 1−1e-6` so `acos'` = `−1/sqrt(1−x²)` is finite (≈ −707 at the clamp) — MEASURED finite grad. |
| **sinkhorn log-domain** | log-sum-exp potentials at `ε=0.05` | log-domain form is overflow-safe at small ε by construction (Cuturi 2013) — MEASURED finite. |

### A.1 The adversarial logit × surrogate × margin-tau matrix (MEASURED)

A direct probe across the full cross-product (synthetic, CPU, `[macOS-CPU advisory]` NON-PROMOTABLE),
each row = `_seg_loss_for_spec(spec, x, t, temperature=0.05)` value + `.backward()` grad finiteness:

```
surrogate     family        tau      val            val_finite  grad_finite  verdict
soft_cosine   tied          1e-6     8.0000e-01     True        True         OK
soft_cosine   saturated     None     8.0729e-01     True        True         OK
soft_cosine   saturated     2.0      0.0000e+00     True        True         OK
soft_cosine   inf_adjacent  None     7.8125e-01     True        True         OK
soft_cosine   normal        2.0      5.6963e-01     True        True         OK
fisher_rao    huge          2.0      7.5900e-04     True        True         OK   <- acos/where NaN-trap CLEARED
fisher_rao    tied          1e-6     4.9679e-01     True        True         OK
fisher_rao    saturated     2.0      0.0000e+00     True        True         OK
sinkhorn      huge          2.0      7.5901e-04     True        True         OK   <- log-domain stable at eps=0.05
sinkhorn      tied          2.0      8.0000e-01     True        True         OK
sinkhorn      saturated     2.0      0.0000e+00     True        True         OK
   ... ALL 54 combinations: val_finite=True, grad_finite=True, val ∈ [0,1].
```

**Every one of the 54 (surrogate × family × tau) combinations gives a finite loss AND finite gradient at
the annealed tail T=0.05.** The saturated / huge families that drive the surrogate to its 0.0 / 0.8
extremes still backprop finite gradients; the margin weight `exp(−margin/τ)` at τ=1e-6 (the clamp floor)
underflows to 0 on large-margin pixels (finite) and is exactly 1 on tied/zero-margin pixels (finite).

### A.2 The ONE non-finite point — fp32 INPUT overflow, NOT a lever defect (characterized, not "fixed")

At `|logit| ≈ 1.2e37` the surrogate goes non-finite. Isolated to the cause:

```
x.abs().max()          = 3.41e37   (still a finite fp32 tensor)
pred/0.05 = pred·20    = inf        (3.41e37·20 = 6.8e38 > fp32 max 3.4e38)  <- overflow HERE
softmax(pred/0.05)     = non-finite (softmax cannot max-subtract an already-inf input)
```

This is the intrinsic fp32 representable ceiling: a logit tensor whose magnitude approaches fp32's limit
overflows during the `pred·20` arithmetic BEFORE softmax sees it. It is **not a lever-introduced
instability** — the lever cannot make `inf/inf` finite, and clamping the input would be a silent semantic
change to the surrogate. The headroom is ~10^36× over any real SegNet logit (R8 measured real
EfficientNet-B2 margins 2.06–5.72, logits O(10)); the test fixtures stop at `|logit| ≤ 1e8` (= ~10^7×
beyond real, the `extreme` family) so a NON-finite result inside the test domain WOULD be a genuine
lever defect. The 1.2e37 overflow is documented here as the characterized boundary, not flagged as a bug.

## B. NO-FAKE PROOF (the Class-2-fake bar) — the tests verify BEHAVIOR, not constants.

A finiteness-only test is vulnerable to the Class-2 fake ("returns a constant — still finite + zero grad").
R9 adds a dedicated guard `test_lever2_anneal_tail_surrogate_is_input_dependent_not_constant` that PROVES
the tail surrogate is a real function of the input:
- `normal` family loss ≠ `saturated` family loss (input-dependent value: 0.57 vs 0.0 — `not allclose`).
- `x.grad.abs().sum() > 0` (the surrogate genuinely backprops through `softmax(pred/0.05)`).

Validated that the guard CATCHES the fakes: a simulated constant-return makes `allclose(const,const)=True`
→ the `not allclose` assertion FAILS (correctly); a simulated zero-grad surrogate gives `grad.abs().sum()=0`
→ the `> 0` assertion FAILS (correctly). The finiteness parametrization ALSO spans values
(saturated→0.0, normal→~0.57) — a constant impl could not produce that spread.

## C. STANDARD CLEAN-CHECK (R9 lens) — all R1–R8 invariants hold on the post-R8 HEAD.

The full lever suite (run detached, SIGURG-proof per the bash-harness ~3-min kill):

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/test_all_layer2_levers.py -q --timeout=600
→ 83 passed in 472.84s (0:07:52)   (SUITE_EXIT=0)
```

No lever code file changed this round (`git diff --stat src/tac/torch_vehicle/ | grep -v tests` shows only
the concurrent partner's archive-build hunk, NOT the lever region), so no R1–R8 invariant *could* regress
from my change; the suite confirms they hold (R1 daemon byte-identity, R2 EMA-resume, MED-1 codec_scan_order,
R3 anneal+compose-timeout, R5 all-5-on determinism, R6 QAT→QAT EMA carry, R7 boundary matrix). The live
distortion arm is structurally untouched (the levers it reads are byte-unchanged; the test additions cannot
reach the running process).

## D. FRESH-EYES "QUESTION EVERYTHING" — the tail surfaces R1–R8 did NOT measure.

1. **Does `softmax(pred/0.05)` overflow on a saturated SegNet?** No — `F.softmax` max-subtracts internally;
   finite for all finite inputs up to ~1.7e37 (then fp32 input-overflow, not a lever issue).
2. **Does the margin weight `exp(−margin/τ)` overflow?** No — `margin ≥ 0` (clamp_min) forces the exponent
   `≤ 0`, so the result is in `(0, 1]` and CANNOT overflow; it underflows to 0 (finite) at huge margin/τ.
3. **Does the τ=1e-6 clamp floor (Lever-5 `max(τ, 1e-6)`) prevent a div-by-zero?** Yes — a τ=0 user input is
   clamped to 1e-6; `exp(−margin/1e-6)` underflows to 0 on any margin > ~1e-4 (finite), is 1 at margin 0.
4. **Does fisher_rao's `acos`/`torch.where` NaN-grad trap fire at the tail?** No — `bc` is clamped to
   `1−1e-6` before `acos`, so `acos'` is finite (≈ −707); MEASURED finite grad on every family.
5. **Does sinkhorn underflow at ε=0.05 with saturated inputs?** No — the log-domain (log-sum-exp) form is
   overflow/underflow-safe at small ε by construction (the docstring's Cuturi-2013 rationale); MEASURED finite.
6. **Does a NaN/Inf GRADIENT slip past a finite VALUE?** No — every combination's `.backward()` produces a
   finite `x.grad` (the prior test checked only the value; R9 checks the grad — the load-bearing addition).

No new finding. The anneal tail is numerically robust across the entire realistic-and-far-beyond input domain.

## Findings by severity

- **HIGH:** NONE.
- **MEDIUM:** NONE.
- **LOW:** NONE.
- (Coverage gap closed — the anneal tail's adversarial-logit × all-3-surrogates × margin-path × GRADIENT
  finiteness now has a regression guard. NOT a finding: the behavior was MEASURED correct before the tests
  were written; the tests pin a passing behavior + a Class-2-fake input-dependence guard.)

## Test-run count

- Full lever suite (detached, SIGURG-proof): **83 passed in 472.84s, 0 failures** (`--collect-only`: 83 collected = all run).
- New R9 anneal-tail tests in isolation: **55 passed in 0.89s** (`-k anneal_tail`: 54 parametrized
  finiteness + 1 input-dependence guard).
- Adversarial-tail probe (full 54-combo cross-product + extreme-magnitude + sub-clamp-T): CLEAN; the only
  non-finite is `|logit| ≳ 1.7e37` fp32 input-overflow (characterized as not-a-lever-defect).

## Tests this round (durable regression guards)

Added to `test_all_layer2_levers.py` (R9 lens):
- `test_lever2_lever5_anneal_tail_finite_loss_and_grad_adversarial` — parametrized over
  3 surrogates × 6 adversarial logit families × 3 margin-tau values (54 cases): asserts FINITE loss AND
  FINITE gradients in `[0,1]` at the live arm's tail T=0.05. Class-2-fake-proof via the value spread +
  the companion guard.
- `test_lever2_anneal_tail_surrogate_is_input_dependent_not_constant` — the Class-2-fake guard: the tail
  surrogate's value is input-dependent (normal ≠ saturated) and its gradient is non-zero (real backprop).
- `_adversarial_seg_logits(kind)` helper — the 6 adversarial logit families, all `|logit| ≤ 1e8` (inside
  the realistic-and-far-beyond domain; a non-finite here = genuine defect, not input overflow).

All ruff-clean.

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + additive numerical-stability regression
guards (no new score-claim surface; the levers' own hooks are in the landing memo). Mission contribution:
`frontier_protecting` (the anneal-tail finiteness proof guards the live multi-day arm's optimizer against a
silent NaN-poison at the T=0.05 tail it descends toward; the END remains a lower exact score, frontier
UNMOVED). Authority: all numbers `[macOS-CPU advisory]` synthetic NON-PROMOTABLE. No GPU launched, no
daemon touched (distortion arm ALIVE + untouched, out-dir separate), no Cool-Chic touched, no lever CODE
file modified.

**VERDICT: CLEAN (zero lever findings) → counter ADVANCES 0/3 → 1/3.** The 5 Layer-2 levers are
numerically robust at the annealed tail T=0.05 across all 3 surrogates × 6 adversarial logit families ×
the margin path, in both loss value AND gradients. R10 + R11 (two more distinct clean lenses) are required
to SEAL. The next lens (R10) is **lever-interaction SIGN/monotonicity on the REAL frozen scorer** (do
levers 2+3+5 compose, or does margin-weight fight the pose-FiLM gradient?).
