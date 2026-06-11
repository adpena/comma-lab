# canonical_kernels real-MLX gumbel fix (2026-06-11)

**Mission:** MLX-port adversarial-audit takeover HIGH item #1
(`.omx/research/mlx_port_adversarial_audit_and_takeover_20260611.md` §"Prioritized
takeover tasks" #1, citing `framework_agnostic/canonical_kernels.py:237-263`
`_gumbel_softmax_sample_mlx`). The canonical `gumbel_softmax_sample` MLX backend
was a FAKE-MLX path: it did `mlx → numpy → forward → mlx` (gradient-breaking,
forward-only), so substrate trainers could NOT route through it and kept LOCAL
duplicate gumbel impls — the Catalog #383 dedup goal was unfulfilled.

This landing makes the MLX path a REAL gradient-preserving native-`mx`
implementation and routes one local duplicate (`mdl_ibps_j`) through it.

## The bug (NO-FAKE class: fake-MLX / gradient-breaking)

`_gumbel_softmax_sample_mlx` (old, lines 237-263) computed the result in numpy via
`_gumbel_softmax_sample_numpy(np.asarray(logits), ...)` then wrapped it
`mx.array(result_np)`. `mx.array(constant)` has NO autograd edge back to the input
`logits` — the MLX computation graph was severed. Any trainer calling it through
`mx.grad`/`value_and_grad` got an identically-zero gradient. That is precisely why
the substrate trainers kept local gumbel impls. The numpy backend (the inflate
contract) and the PyTorch backend were already real; only the MLX path was fake.

## The fix

`_gumbel_softmax_sample_mlx` is now end-to-end native `mx`:
- Keep the input as an `mx.array` (only `mx.array(np.asarray(...))` if a non-MLX
  array is passed — no roundtrip when an mx.array comes in from a trainer).
- Gumbel(0,1) reparametrization noise via `mx.random.uniform` →
  `-mx.log(-mx.log(u))` (the noise is a constant w.r.t. logits, as it must be).
- `perturbed = (logits + g) / τ`, optional Hafner-2023 §3 unimix mixture in
  prob-space via a new native `_apply_unimix_to_logits_mlx` (`mx.softmax`/`mx.log`),
  then `mx.softmax(..., axis=-1)`.
- Determinism: a Python `int` seed maps to `mx.random.key(seed)`; `seed is None`
  defers to the global MLX rng. Same key-based determinism the canonical DreamerV3
  reference (`dreamer_v3_rssm/module.py::gumbel_softmax_sample`) uses.

The gradient now flows through `(logits + g)/τ → softmax` — the canonical
Gumbel-softmax reparametrization estimator. Public API unchanged; numpy + torch +
tinygrad backends untouched.

## Before/after gradient evidence (the no-fake gate)

Same loss `sum(gumbel_softmax(logits, τ=0.5, α=0.01, MLX, seed=7) · onehot)`,
`mx.grad` w.r.t. logits:

| Path | `max|grad|` | verdict |
|---|---|---|
| BEFORE (numpy roundtrip) | **0.0** | autograd severed — forward-only fake-MLX |
| AFTER (native `mx`) | **0.3549545** | real gradient flows to logits |

The new test `test_gumbel_softmax_mlx_gradient_flows_to_logits` asserts
`max|grad| > 1e-4` and `any(grad != 0)` — it FAILS on the old roundtrip
(grad ≡ 0) and PASSES on the fix. A test that would pass on the broken
forward-only version is forbidden; this one would not.

## Torch-parity result

Bit-for-bit numpy↔MLX (or torch↔MLX) parity is IMPOSSIBLE for a real native-MLX
gumbel because MLX's RNG ≠ numpy's `default_rng` for the same integer seed (the
prior same-seed parity test ONLY passed because the MLX path WAS numpy output —
it was a fake-parity test; it has been replaced). The real, backend-agnostic
parity invariants now tested:
- **Deterministic-functional parity (numpy↔MLX):** with identical *injected*
  Gumbel noise, the unimix+softmax functional agrees within Slot 16 atol=1e-5
  (`test_gumbel_softmax_deterministic_functional_parity_numpy_vs_mlx`).
- **Discrete-limit parity (numpy↔MLX, torch↔MLX):** at τ=0.01–0.05 both backends
  concentrate on the argmax-logit category (`p > 0.9`).
- **Gradient-direction parity (MLX↔torch):** averaged over 20 seeds, the
  Gumbel-softmax gradient on a `-log p(target)` loss pushes the target logit UP
  (negative grad) with the SAME sign in MLX and torch
  (`test_gumbel_softmax_mlx_gradient_direction_matches_torch`).

## Dedup: one local duplicate routed to the canonical

`tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer.
gumbel_softmax_sample_mlx` SOFT path (`hard=False`, the one its renderer's
`reconstruct_pair` calls) now delegates to the canonical
`framework_agnostic.canonical_kernels.gumbel_softmax_sample(..., backend=MLX,
unimix_alpha=0.0)`. `unimix_alpha=0.0` preserves the substrate's exact prior
categorical-posterior behaviour (no Hafner unimix here). The gradient-free HARD
straight-through one-hot branch stays local (the canonical returns a single soft
simplex; STE one-hot is not part of that contract). Verified: routed soft path is
a valid simplex, gradient flows (`any(grad!=0)`), hard path still one-hot. The
mdl trainer can now train THROUGH the canonical primitive — the thing the
numpy-roundtrip canonical made impossible.

### Duplicates NOT routed (cited follow-up)

- `dreamer_v3_rssm/module.py::gumbel_softmax_sample` (the canonical reference
  impl, returns `(soft_or_hard, indices)` tuple + STE flag + MLX key) — a
  PRINCIPLED FORK per Catalog #290 (`MLX_PRIMITIVE_UNIQUE_BECAUSE_` waiver on
  line 269): its tuple/STE/key signature is the substrate-optimal RSSM contract;
  the canonical single-tensor soft helper cannot serve it without losing the
  index + STE return. Left as-is. **Follow-up task:** consider extracting a
  shared `_gumbel_perturb_mlx` core that both the canonical soft helper and the
  dreamer tuple/STE variant call, so the noise+unimix math has ONE source even
  though the return contracts differ.
- `z8_hierarchical_predictive_coding/mlx_renderer.py::gumbel_softmax_sample`
  already delegates to the dreamer sister (line 294) — NOT a pure duplicate;
  it inherits the dedup transitively. No change needed.

## NO-FAKE / discipline accounting

- The MLX path now does REAL native-`mx` work on the REAL input tensor; no
  numpy roundtrip; gradient flows (empirically 0.0 → 0.355).
- The replaced same-seed parity test was a fake-parity test (passed only because
  of the roundtrip); the new tests verify real cross-backend invariants and a
  gradient that the broken version cannot produce.
- No score claims. MLX outputs remain `[macOS-MLX research-signal]`
  non-promotable per Catalog #192/#317. torch-CPU is the only authority cited.
- NO MPS used anywhere. Seeds pinned (`mx.random.key(seed)` /
  `np.random.RandomState`).
- `check_383` (`check_mlx_primitives_route_through_canonical_helper`) STRICT:
  PASS, 0 live violations (the canonical kernel is the recognized extractor;
  the mdl local def now delegates to it; dreamer keeps its #290 waiver).

## Tests

- `framework_agnostic/tests/test_canonical_kernels.py`: 49 passed, 1 skipped
  (tinygrad). The `TestCrossBackendParityMLX` class is 11 tests; 8 are
  gumbel-specific (gradient-flow gate, no-roundtrip structural guard, fixed-seed
  determinism, deterministic-functional parity, low-τ argmax parity numpy↔MLX,
  torch↔MLX discrete-limit, torch↔MLX gradient-direction).
- `mdl_ibps_j_discrete_categorical_mine_hybrid/tests/`: 43 passed (no regression
  from routing the soft path).
- `dreamer_v3_rssm` + `z8` regression: 558 passed, 1 failed —
  `test_real_teacher_loss_total_composes_all_three_axes`. **Confirmed
  PRE-EXISTING:** it FAILS identically with my changes git-stashed out; it
  exercises `score_aware_loss` real-teacher loss composition, imports neither
  canonical_kernels nor mdl_ibps_j, and is unrelated to gumbel. Not introduced
  by this landing.
- ruff: clean on all 3 edited files.

## Files

- `src/tac/framework_agnostic/canonical_kernels.py` — real native-MLX
  `_gumbel_softmax_sample_mlx` + new `_apply_unimix_to_logits_mlx`.
- `src/tac/framework_agnostic/tests/test_canonical_kernels.py` — replaced the
  fake-parity test; +6 net new MLX gumbel tests.
- `src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/mlx_renderer.py`
  — soft path routed to the canonical kernel.
