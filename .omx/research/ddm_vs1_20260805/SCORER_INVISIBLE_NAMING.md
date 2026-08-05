# VS1 scorer-invisible naming convention

**Date:** 2026-08-05
**Scope:** naming hygiene for #839; no scorer run, no score claim.

The string "scorer-invisible" has been used for four different quantities. Use these names going forward,
and retag old rows on touch rather than sweeping unrelated history.

| canonical name | old ambiguous wording | quantity | rate admissibility |
|---|---|---|---|
| `RESIZE_KERNEL_NULLITY_DOF` | `ker(A) 80.67% scorer-invisible` | Dimension count of the bilinear resize kernel in camera space. | Not bytes. No rate credit. |
| `CERTIFIED_ZERO_WEIGHT_BLIND_MASK` | `22.70% blind/scorer-invisible pixels` | Exact camera pixels read by neither scorer's resize windows: 230,904 px/frame. | Not bytes by itself. It may define an actuator/support mask. |
| `RANGE_A_COMPLEMENT_RENDER_ENERGY` | `~52% scorer-invisible render energy/head-norm` | Measured energy in the range(A)-complement for a specific render/object. | Precision/gauge signal only; not a priced rate column. |
| `COUNTED_PAYLOAD_RATE_CREDIT` | `scorer-invisible bytes`, `ker(A) free bytes`, `rate-neutral null` | Actual archive bytes removed from a parser-consumed counted payload while preserving receiver/scorer behavior. | The only live rate-pricing quantity. Waterfill tables may consume this column; the other three may not. |

**Live retag:** `P_NULL_GAUGE` in `src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py`
now names `COUNTED_PAYLOAD_RATE_CREDIT` explicitly. The `pantheon_synergy` F5 row is reworded so the
~52% gauge/energy result cannot be consumed as a measured byte column.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
