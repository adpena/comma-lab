# Codex Routing Directive — TOP-7 Arbitrariness Extinction: qint_max Grid {1,3,7,15,31} → Continuous Water-filling

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `qint_max_grid_1_3_7_15_31_arbitrary_water_filling`
**Resolution path**: `formula`
**Predicted ΔS**: [-0.003, -0.0005]
**Cost envelope**: $0
**Rank score per dollar**: 3.0

## Bug class

`src/tac/water_filling_codec.py:54` `_ALLOWED_QINT_MAX = (1, 3, 7, 15, 31)` — powers of 2 minus 1. Water-filling is mathematically OPTIMAL **for this grid**, but the grid itself is unprincipled. Information-theoretic optimum is CONTINUOUS bit budget per channel.

The {1,3,7,15,31} grid forces ~5-10% suboptimality at boundaries: a channel optimal at `qint_max=11` (would be ~3.46 bits) is forced to `qint_max=15` (4.95 bits, +43% bytes) or down to `qint_max=7` (3.91 bits, -29% bytes with potential distortion blow-up).

## 5-path analysis

1. **experimental** — sweep finer grids. Improves slightly; doesn't solve.
2. **analytical_solve** — Lloyd-max R-D theory: continuous bit budget per channel, then post-round.
3. **formula** [RECOMMENDED] — continuous water-filling: solve `b_i = max(0, (Σ S - bit_budget) / N - log(S_i))` then ROUND to nearest entropy-coded integer level (NOT power-of-2-minus-1 grid).
4. **learned** — Ballé hyperprior already does this (Catalog #319 sister); but for non-Ballé substrates the closed form applies.
5. **self_alien_tech** — composite QINT + Huffman coding (already used for fec6 entropy).

## Concrete next step ($0)

Extend `water_filling_codec.py`:

```python
def allocate_qint_continuous(
    *,
    sensitivity_per_channel: ndarray,
    total_bit_budget: float,
    integer_round_mode: str = "huffman",  # or "fixed_grid"
) -> ndarray:
    """Continuous water-filling + post-rounding to entropy-coded levels."""
    # b_i = max(0, lagrange_mu + log(sensitivity_i))
    # adjust mu so sum(b_i) = total_bit_budget
    # round b_i to nearest VARIABLE-LENGTH integer level (via Huffman-aware code)
```

`arithmetic_qint_codec.py` already handles variable-length integer codes per `num_symbols = 2*qint_max+1` — wire output here.

## Sister coordination

- Catalog #319 Ballé hyperprior uses learned hyperprior bits — orthogonal to this for non-Ballé codecs
- Couples with `block_fp_block_size_undeclared` row (sister bit-allocator decision)
- Cathedral autopilot bit-allocator hook #3 per Catalog #125

## Exit criteria

1. `allocate_qint_continuous` canonical function added
2. Per-codec wire-up to continuous + post-round path
3. Empirical anchor on PR106 / PR101 archives shows expected byte reduction
