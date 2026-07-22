---
title: Canonical equations - per-stream optimal coder assignment
date_utc: 2026-07-22T04:47:13Z
task: 603
feeds_task: 613
lane_id: lane_truly_optimal_coder_survey_603_613_20260722
research_only: true
main_landing_review_required: true
---

# Total description, not entropy alone

For semantic stream `s`, transform/model `m`, channel coder `c`, and deterministic decoder `D`, define

`L_s(m,c) = |tag| + |model| + |parameters| + |termination| + |payload| + |stream framing|`.

The stream assignment is registerable only when a real encoded byte string `b_s` satisfies

`D_s(b_s; m,c) = s`, `SHA256(D_s(b_s)) = SHA256(s)`, and `L_s(m,c) = len(b_s)`.

The selected local coder is

`(m_s*, c_s*) = argmin_(m,c in C_s) L_s(m,c)`.

For final archive compiler `A`, isolated stream bytes are not additive authority. The admitted rate is

`Delta B_s = len(A(z with b_s*)) - len(A(z baseline))`.

Admit a lossless recode only if `Delta B_s < 0`, the receiver consumes the new bytes, every semantic
stream is preserved, Pose completeness remains one, and parse/re-encode determinism holds. The
contest rate marginal, once an exact archive delta exists, is

`Delta score_rate = 25 * Delta B_s / 37,545,489`.

# Classical exact lengths used

For nonnegative integer `n` and Golomb divisor `M`, with `q=floor(n/M)` and `r=n-qM`,

`l_G(n;M) = q + 1 + l_truncated_binary(r;M)`.

For Rice `M=2^k`, this becomes

`l_R(n;k) = floor(n/2^k) + 1 + k`.

Signed values use a sealed zigzag bijection before the length formula. The selected Rice parameter is

`k* = argmin_k [header(k) + sum_i l_R(zigzag(x_i);k)]`.

For positive integer `n`, the Elias lengths measured are

`l_gamma(n) = 2 floor(log2 n) + 1`,

`l_delta(n) = floor(log2 n) + 2 floor(log2(floor(log2 n)+1)) + 1`.

For a constant-weight support of `K` positions in universe `N`, Cover enumerative/colex length is

`l_enum(N,K) = ceil(log2 binomial(N,K))`,

plus the counted `N`, `K`, record, and stream framing. The colex rank itself is

`rank({a_1 < ... < a_K}) = sum_(i=1)^K binomial(a_i, i)`

under the zero-based convention sealed by the decoder.

# Universal-model ceilings

For alphabet counts `n_a` after prefix `x_<t`, the KT predictive probability is

`P_KT(x_t=a | x_<t) = (n_a + 1/2) / (t + |Sigma|/2)`.

The reported KT0/KT1 rate is

`L_KT = ceil(-log2 product_t P_KT(x_t | context_t) / 8)` bytes.

It is labeled `DERIVED` because this survey did not land a complete KT/CTW bitstream with model,
termination, and receiver framing. An arithmetic or ANS channel cannot improve a bad model's
`-log2 P`; it only approaches that modeled rate with finite-state overhead.

# Measured assignment vector

On the scoped real objects,

`c* = (Brotli_610, LZMA_204, Rice_k6_3509, LZMA_181904, Brotli_1086, Brotli_global_80478)`

ordered as `(static, xi trajectory, Pose ordinal proxy, events, entropy state, exceptions)`.

Its numeric subtotal

`610 + 204 + 3509 + 181904 + 1086 + 80478 = 267791 B`

is a noncomposable blocker diagnostic, not `len(A(z))`: it mixes legacy opaque aliases, omits final
framing, excludes the PPCS semantic remainder, and uses a Pose proxy. Adding the whole PPCS zlib
diagnostic only for budget pressure gives

`267791 + 78969 = 346760 B > 216223 B > 154524 B`.

The decisive inequality is already

`L_events* = 181904 B > B_strict = 154524 B`.

Therefore the coder-only formulation cannot reach the strict Task #613 box on these measured objects.
This negative is scoped to the current representation; a new event grammar may reduce `s` before the
lossless coder is selected.

# Xi and Pose selection tags

Because xi LZMA and Brotli differ by one byte on the current compact wire, the future selector pays
for itself only when

`min(L_lzma(s), L_brotli(s)) + L_tag < L_fixed(s)`.

The same rule applies to Pose `min(raw, Rice(k*))`. The tag, `k*`, length fields, checksum, and decoder
branch are never free unless already sealed in the receiver grammar.

0.1910828242 [contest-CPU] — unchanged by construction.
