# `rel_err` Definition Inconsistency Audit and Canonicalization

**Date**: 2026-05-08
**Author**: Claude Opus 4.7 (1M context) [bug-class closure subagent]
**Surfacing source**: Codex Pattern A adversarial review, finding #1
(`/tmp/codex_runs/adversarial_review_session_designing_20260508.log`).
**Bug class**: `rel_err` mixes L1 / L2 / RMS conventions silently across
`tac.codec`, `tac.optimization`, and `tools/pr10[16]_*`. The Lagrangian cost
function `cost = bytes + λ · w · rel_err²` is mathematically inconsistent when
its inputs come from heterogeneous error definitions.
**Classification**: design decision per `CLAUDE.md` ("Design decisions —
non-negotiable"). Captured in this memo as the lightweight council
deliberation; if any unified change introduces ambiguity or perverse semantics,
it is surfaced and DEFERRED below rather than applied silently.

---

## 1. Audit results

### 1.1 Per-occurrence enumeration

#### `src/tac/codec/per_tensor_codecs.py` — primary canonical encoders

| Line | Function | Definition | Form |
|------|----------|------------|------|
| 70   | `encode_brotli_only` | `0.0` (lossless) | n/a |
| 108  | `encode_sparsity_alpha` | `‖diff‖₂ / (‖orig‖₂ + 1e-12)` | **L2 / Euclidean** |
| 144  | `encode_lossy_K_coarsen` | `Σ\|round-orig\| / Σ\|orig\|` | **L1 / Manhattan** |

**This is the central inconsistency.** Two co-located primitives sharing
identical signatures `(symbols, …) -> (bytes, rel_err)` use *different*
definitions of `rel_err`. The brotli-only/sparsity path uses L2 Euclidean
ratios; the lossy K-coarsen path uses L1 absolute-sum ratios. Both feed the
same Lagrangian curve dataset.

#### `src/tac/codec/cost_curves.py` — curve precompute + greedy selectors

| Line | Construct | Behavior |
|------|-----------|----------|
| 93,96 | `precompute_per_tensor_sparsity_curves` | Forwards the L2 number from `encode_sparsity_alpha` into the curve as `rel_err`. |
| 123  | `greedy_uniform_per_tensor_budget_sparsity` | Aggregates per-tensor errors as `sqrt(mean(e²))` (RMS). |
| 158-159 | `precompute_per_tensor_K_curves` | Forwards the L1 number from `encode_lossy_K_coarsen` into the curve as `rel_err`. |
| 184-193 | `find_best_K_for_tensor` | Recomputes L1 inline (`abs_err / abs_sum`); duplicate of `encode_lossy_K_coarsen`. |

Greedy aggregation in `cost_curves.py:123` assumes RMS-style composability
(`sqrt(mean(e²))`); but when fed the K-curve dataset the inputs are L1 ratios.
This is a category error: `sqrt(mean(L1²))` is not a meaningful aggregate
distortion.

#### `src/tac/optimization/lagrangian_per_tensor_allocation.py` — allocator

| Line | Construct | Behavior |
|------|-----------|----------|
| 24   | Docstring | `cost_t(c) = bytes_t(c) + λ · w_t · rel_err_t(c)²` (squared error penalty). |
| 74   | `_select_min_cost` | Squares whatever `rel_err` is in the curve. |
| 191  | Default aggregate | `sqrt(mean(e²))` of the per-tensor `rel_err`. |

The squared penalty `λ · w · rel_err²` is mathematically an MSE-style
Lagrangian (penalizing the L2² deviation). When fed L1 ratios, the penalty
becomes `λ · w · L1²`, which is **not** a Lagrangian dual on a meaningful
distortion. The allocator silently produces per-tensor decisions whose sum is
mis-priced relative to the actual archive geometry.

#### `tools/build_pr106_uniward_runtime_packet.py:290`

The joint-encoder hook at lines 308–317 reports `rel_err = sum_t |rounded - orig| / sum_t |orig|`
(global L1, not RMS). The docstring at line 333 says "global L1 not RMS,"
acknowledging the divergence. Result: when the allocator bisects to achieve
a `rms_target`, the achieved value the joint encoder reports back is L1, not
RMS — bisection is on apples vs. oranges.

#### `tools/pr101_lossy_coarsening_analytical.py`

Lines 109–123 (`find_best_K_for_tensor`) and 130–156 (`encode_with_per_tensor_K`):
both compute global L1 ratios. Header docstring (lines 12–14) labels them
`rel_err` interchangeably with the budget targets. This is the canonical
non-RMS path; the allocator that consumes its byte-proxy curves
(`tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py:205` and
`tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py`) inherits the L1
form silently.

#### `tools/build_admm_x_lossy_coarsening_path_b_step6.py` and `_no_dead_k.py`

| Line | Tool | Form |
|------|------|------|
| 248 (step6) | tensor reconstruction error | global L1 |
| 327 (step6) | post-codec smoke check | global L1 |
| 588, 956 (no_dead_k) | reconstruction error | global L1 |

#### `tools/pr101_omega_opt_per_tensor_codec_choice_empirical.py:124`

`total_rel_err = sqrt(sum(rel_err² for s in selections) / n)` — RMS aggregate
over selections. **Inputs to the sum come from `encode_brotli_only` /
`encode_sparsity_alpha` (L2) AND `encode_lossy_K_coarsen` (L1)** because this
tool ranges over the codec-CHOICE substrate that includes lossy_coarsening.
This is the most concrete instance of the bug: an RMS over mixed L1²/L2²
inputs.

#### `tools/pr101_omega_opt_linear_stack_empirical.py:170`

`rel_err = mean((|recon - orig| / |orig|))` over kept indices, then `× 100`
(percentage). This is yet another form: per-element L1-relative MEAN, not
sum/sum, not L2.

#### `tools/pr101_lossy_int4_*` (AWQ, GPTQ, QAT, mixed-precision)

`rel_err_pct` reported by these tools is `100 * abs_err[mask] / abs(orig[mask])`
elementwise, summarized as a "weighted average." These are **per-element L1
percentages**, not the same construct as the curves above. They are not fed
to the Lagrangian (they live in a separate per-channel-scale workflow), but
they share the name `rel_err` in evidence rows and reports.

#### `tools/pr101_balle_cross_tensor_hyperprior.py:611`,
`tools/pr101_compressai_balle_FIXED.py:355`,
`tools/pr101_compressai_balle_hyperprior_full.py:394`,
`tools/pr101_compressai_factorized_prior.py:375`,
`tools/pr101_neural_weight_codec_NWC.py:402`

All compute `rel_err = sum|abs_err| / sum|orig|` (global L1). Consistent
within the neural-codec family but inconsistent with the brotli/sparsity
primitives.

#### `tools/pr101_unified_winners_stack_empirical.py` (lines 239, 327, 871, 924)

Multiple call sites computing global L1 ratios. Aggregates them with `sqrt`
in some places, plain sum in others.

### 1.2 Summary counts

- **L2 (Euclidean) sites**: 1 (`encode_sparsity_alpha`).
- **L1 (sum-of-abs) sites**: ~22 distinct call sites across `src/tac/codec/`
  (1) and `tools/` (≥21).
- **Per-element L1 mean / percentage sites**: 6 (int4 family +
  linear_stack).
- **RMS-aggregate-over-something sites**: 4
  (`cost_curves.greedy_*`, `lagrangian_*.allocate` default aggregator,
  `omega_opt_per_tensor_codec_choice:124`, `unified_winners` partial).
- **Total occurrences of the literal token `rel_err`**: well over 200 across
  audited files (see grep output captured in session shell history).

### 1.3 Cross-module flow problems

1. `cost_curves.precompute_per_tensor_sparsity_curves` writes L2 numbers
   under the key `rel_err`.
2. `cost_curves.precompute_per_tensor_K_curves` writes L1 numbers under the
   same key.
3. The Lagrangian allocator squares whatever it finds and bisects on RMS.
4. `tools/build_pr106_uniward_runtime_packet.py` joint hook returns global
   L1 under the same key, contradicting the bisect target's RMS semantics.
5. Reports tag empirical anchors with `rel_err=0.0386` etc. without
   specifying which form — the same number means different things in
   different tools.

---

## 2. Council-style decision (in-memo deliberation)

### 2.1 Options enumerated

**Option A — Unify on RMS** (`sqrt(mean((q - q_orig)² / max(orig_norm, eps)))`).
Pros: matches the squared Lagrangian penalty; matches the contest score's pose
distortion form `sqrt(10 · d_pose)` (mean-squared L2 norm); aggregates
cleanly across tensors as `sqrt(mean(e²))`; differentiable everywhere;
standard signal-processing distortion.
Cons: every L1 site needs to either be retagged or recomputed. Some empirical
anchors (e.g. PR101 lossy_coarsening 0.0386 L1) lose direct comparability —
must be replayed if we want an apples-to-apples RMS number, or kept as
separate `l1_err` for backward reference.

**Option B — Unify on global L1** (`sum|q - q_orig| / sum|q_orig|`).
Pros: matches the historical PR101 lossy_coarsening/PR106 UNIWARD packet
convention; majority of tools already use it; existing empirical anchors
(0.0386, 0.0415) remain valid without retagging.
Cons: the Lagrangian penalty `λ · w · L1²` is **not** a meaningful dual on
either MSE or L2 distortion; squaring an L1 ratio loses its interpretation;
RMS aggregate `sqrt(mean(L1²))` over per-tensor L1 errors is mathematically
hollow. Composing codecs across stacks loses the additivity that L2² has.

**Option C — Forbid the unified `rel_err` key entirely; require explicit
form-tagged keys (`l1_err`, `rms_err`, `linf_err`)**.
Pros: zero ambiguity; type system carries the semantics.
Cons: 22+ call sites and dozens of evidence rows need migration; high
cosmetic churn for limited new safety beyond Option A + assertions.

**Option D — Keep both forms but tag them in the curve data structure**
(e.g. `rows.append({"rel_err": …, "rel_err_form": "l1"})`).
Pros: minimal churn; preserves all historical numbers.
Cons: doesn't fix the Lagrangian allocator's mathematical inconsistency
(it still squares heterogeneous inputs); only documents the bug.

### 2.2 Council positions

- **Shannon (lead)**: `cost = bytes + λ · ‖q-q_orig‖²` is the canonical
  rate-distortion Lagrangian where distortion = MSE = mean of squared L2.
  RMS² = MSE up to mean-vs-sum framing. The squared penalty form *itself*
  mandates RMS semantics. **Vote: Option A.**
- **Dykstra (co-lead)**: alternating-projections feasibility on convex
  rate/distortion sets requires distortion be a convex norm with consistent
  scale. L2² is a Euclidean projection-natural distortion; L1 is OK in
  isolation but its square (`L1²`) is not the squared-distortion of an inner
  product space. The Lagrangian dual of an L1 constraint would use `λ · L1`,
  not `λ · L1²`. **Vote: Option A** (with the caveat that L1 may be
  legitimately preferable for max-norm robustness in specific paths; expose
  `mode="l1"` as an opt-in).
- **Yousfi**: distortion in the contest is built from per-pixel argmax
  disagreement (SegNet) and squared-L2 (PoseNet). Neither is L1. RMS aligns
  with PoseNet's structure. **Vote: A.**
- **Fridrich**: steganalysis cares about *detectability* which behaves like
  energy (L2²) under whitening. UNIWARD weights are inverse local variance —
  variance is L2². The whole paradigm is L2-native. **Vote: A.**
- **Contrarian**: "unifying" can still be wrong. The 0.0386 number we have
  shipped 6 ways from Sunday is L1; if we silently switch the Lagrangian to
  RMS the empirical predictions retroactively shift. Demand:
  (a) the canonical implementation MUST default to RMS, but
  (b) every tool using L1 today MUST be explicitly opted-in with `mode="l1"`
  AND tagged in evidence rows so historical numbers are not silently
  reinterpreted. **Vote: A + explicit opt-in.**
- **Quantizr**: the 0.33 archive used `mode="l1"`-equivalent reasoning with
  per-tensor abs-error bounds for K coarsening; it works empirically because
  L1 is conservative for small K. Don't break working paths; let them tag
  themselves. **Vote: A with backward-compat `mode="l1"`.**
- **Hotz**: stop arguing, ship one definition and put a runtime assertion
  in the allocator. **Vote: A + assertion.**
- **MacKay**: MDL framing — the Lagrangian dual `λ · D(c)²` is the natural
  KKT condition only if D is L2; the L1 case has a different dual. We've
  been doing the wrong dual. **Vote: A.**
- **Ballé**: end-to-end neural codecs minimize MSE-Lagrangian. RMS is the
  right canonical. **Vote: A.**
- **Selfcomp**: practical caveat — the analytical lossy_coarsening tool
  searches L1-budgets per tensor which is the right shape for "cap the
  worst-case symbol error." Keep that operating mode under `mode="l1"`;
  don't delete it. **Vote: A with explicit mode parameter.**

### 2.3 Verdict

**OPTION A: RMS canonical (default), with explicit `mode="l1"` / `mode="max"`
opt-ins for paths that have load-bearing alternative semantics.** 10/10
council members endorse; the Contrarian's caveat is captured by making the
mode parameter mandatory at the call site (no silent default promotion).

### 2.4 Backward-compat plan

- New canonical function `compute_rel_err(q, q_orig, *, mode)`.
- Existing per-tensor encoders KEEP their current numerical behavior; their
  docstrings are upgraded to declare the form (`# REL_ERR_NON_CANONICAL_OK:
  K-coarsen uses L1 for legacy tools/pr101_lossy_coarsening_analytical.py
  parity (preserves 0.0386 anchor)`). They each gain a `rel_err_form` tag
  that is forwarded into curve rows.
- The Lagrangian allocator gains a runtime assertion that all curves it
  consumes carry a uniform `rel_err_form` (initially warn-only via a
  module-level flag; flip to `raise` after one tranche of evidence-row
  retagging).
- A new preflight check `check_rel_err_definition_canonical` greps for
  `rel_err =` definitions outside the canonical helper and asserts the
  presence of `# REL_ERR_NON_CANONICAL_OK:<reason>` waivers; initially
  `strict=False`.

This minimizes empirical churn while making the math crisp.

---

## 3. Fixes applied

(Filled in by the implementation step that follows this audit memo.)

- `src/tac/codec/rel_err.py` — new canonical helper + `RelErrForm` enum +
  `compute_rel_err(q, q_orig, *, mode)` implementing modes `rms`, `l1`,
  `max`, plus `aggregate_rel_err(values, mode)` for RMS/L1/max aggregation.
- `src/tac/codec/per_tensor_codecs.py` — encoders' docstrings updated to
  declare the form they emit (`L2 ratio` for sparsity, `L1 ratio` for
  K-coarsen). Each encoder also returns the same numerical value as before
  to preserve every empirical anchor.
- `src/tac/codec/cost_curves.py` — curve rows annotated with
  `rel_err_form` (`"l2_ratio"` or `"l1_ratio"`); aggregate function
  documented.
- `src/tac/optimization/lagrangian_per_tensor_allocation.py` — allocator
  docstring + entry assertion that the curve rows have a uniform
  `rel_err_form`. Default behavior unchanged when curves carry no form
  tag (warn once); raise is gated on `strict_rel_err_form=True` flag.
- `src/tac/tests/test_rel_err_canonicalization.py` — pinning tests for
  each mode + RMS-aggregate consistency + allocator-side assertion.
- `src/tac/preflight.py` — `check_rel_err_definition_canonical`
  registered with `strict=False`.

---

## 4. Deferred items

- **Mass migration of empirical anchors to RMS form**: 0.0386 / 0.0415
  numbers in shipped evidence rows are L1 and stay L1; new evidence rows
  using the canonical helper will be tagged `rel_err_form="l2_rms"` (or
  whichever explicit form is computed). No retroactive rewriting of
  evidence JSONL.
- **Strict-flip of the preflight check**: held warn-only until live
  violation count is confirmed at 0 in a follow-on tranche. The PR101
  Path-B and PR106 UNIWARD tools all carry legitimate L1 semantics; their
  waivers must be added before strict.
- **`pr101_lossy_int4_*` per-element percentages**: their semantics are
  legitimately different (per-channel-scale workflow, not allocator-fed)
  and they share only the name `rel_err`. Recommendation: rename the
  field in evidence rows to `rel_err_pct_per_element` in a follow-on
  cleanup; out of scope for this bug-class closure.

---

## 5. Cross-references

- Codex adversarial review #1 — `/tmp/codex_runs/adversarial_review_session_designing_20260508.log`
- `tac.optimization.lagrangian_per_tensor_allocation.LagrangianPerTensorAllocator`
- `tac.codec.cost_curves.precompute_per_tensor_K_curves`
- `tac.codec.per_tensor_codecs.encode_sparsity_alpha` (L2)
  vs `tac.codec.per_tensor_codecs.encode_lossy_K_coarsen` (L1)
- `tools/build_pr106_uniward_runtime_packet.py:290` global-L1 joint hook
- `tools/pr101_lossy_coarsening_analytical.py:146` global-L1
- `tools/pr101_omega_opt_per_tensor_codec_choice_empirical.py:124` mixed
  RMS over heterogeneous inputs
- CLAUDE.md: "Design decisions — non-negotiable", "Forbidden empirical-claim-
  without-evidence-tag", "Meta-bug class catalog (strict-mode preflight)"
