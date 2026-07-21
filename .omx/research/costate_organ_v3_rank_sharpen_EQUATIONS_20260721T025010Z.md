# Equations: costate organ v3 rank sharpening

UTC: 2026-07-21T02:50:10Z  
Equation family: `costate_v3_rank_sharpen_composition_v1`

## Graded receiver survival

For a route whose deepest required stage has `k` surviving sites of `n`, use
the declared Jeffreys posterior mean

`p_survive = (k + 1/2) / (n + 1)`.

The measured r1b7 histogram has `n=498`: 289 clean survivors, 5 collateral
survivors, 204 killed at same-rival head resolution, and zero observed deaths
at uint8, resize, stem, or wrong-rival head stages. Graded prediction is

`lambda_survival = gap * visibility * p_survive(route) * byte_price`.

The exact source gap is used only as a structural prediction feature to order
otherwise identical route probabilities; realized targets never enter this
key.

## Pool-interaction/KKT marginal

For opportunity pool `g`, candidate raw marginals `m_i`, allocation ceiling
`C_g`, and reverse-waterfill threshold `nu_g`, the admitted pool allocation
satisfies

`x_i >= 0`, `sum_{i in g} x_i <= C_g`,

`m_i - nu_g <= 0` when `x_i=0`, and `m_i - nu_g = 0` for an interior active
coordinate. The finite receipt implementation sorts by raw marginal and applies
the measured pool ceiling, preventing multiple candidates from claiming the
same remaining debt. This reuses registered LawRef
`witness_measured_reverse_waterfill_v1`.

## EMA target de-noising and apparatus variance

For constant EMA decay `d` and integer observation horizon `h`,

`a_h = 1 - d^h`,

`DeltaS_EMA = a_h * DeltaS_live`,

`DeltaS_live_hat = DeltaS_EMA / a_h`,

`w_h proportional to 1 / Var(DeltaS_live_hat) = a_h^2 / sigma^2`.

The inverse is applied only to custodied #205 rows with `d=0.997` and explicit
integer horizons. C2 n=120 subset rows retain their measured target and receive
apparatus precision weight `120/600`. These operations affect the realized
target/metric side only, never prediction.

The evaluator and typed DSL consumer for `ema_decay_run_geometry_v1` exist, but
the canonical-equations JSONL row is absent. This note does not represent it as
canonically registered.

## Rank objectives

Weighted Pearson correlation of ranks is

`rho_w = sum_i w_i (R_i-Rbar_w)(Q_i-Qbar_w) /
         sqrt(sum_i w_i(R_i-Rbar_w)^2 * sum_i w_i(Q_i-Qbar_w)^2)`.

Decision-weighted NDCG@8 uses nonnegative realized score benefit `b_i`:

`DCG@8 = sum_{j=1}^8 w_(j) * b_(j) / log2(j+1)`,

`NDCG@8 = DCG@8 / IDCG@8`.

Top8 precision is the fraction of the eight highest predicted rows with
strictly positive realized benefit. Paired stratum-preserving bootstrap deltas
are the verdict surface; point changes inside a CI spanning zero are not
improvement claims.

## Receipt boundary

The typed row key is `id`; the exact canonical record hash includes the source
receipt SHA-256. Appends are exclusive-lock serialized. An exact duplicate is
a no-op; any same-ID content or custody conflict is refused.
The M1 bridge additionally requires a repo-relative source receipt, matching
SHA-256, a realized byte-close block, and nonzero byte delta. Thus identity seed
rows cannot masquerade as a first byte-paying proof.
