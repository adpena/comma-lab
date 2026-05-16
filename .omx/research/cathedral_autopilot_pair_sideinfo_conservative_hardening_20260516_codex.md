# Cathedral autopilot pair side-info conservative hardening - 2026-05-16

## Trigger

L5/Cathedral adversarial review found that orthogonal pair candidates in
`tac.optimization.autopilot_dispatch_ranking` aggregated `sideinfo_consumed`
with boolean OR. That can mark a composed candidate as side-info-consumed when
only one substrate has consumption proof and the other is false or unknown.

## Landing

Pair aggregation now uses a conservative tri-state rule:

- all component substrates `True` -> pair `True`;
- any component substrate `False` -> pair `False`;
- otherwise -> pair `None` / unknown.

Singleton semantics are unchanged. This keeps Cathedral rows from overstating
proof quality before L5-v2 byte-closed gates and paired exact eval custody.

## Tests

- `test_orthogonal_pair_sideinfo_consumption_is_conservative_per_substrate`
- `test_orthogonal_pair_sideinfo_consumption_keeps_unknown_when_not_all_proven`

## Boundary

This is ranking/proof metadata only. Dispatch remains operator-gated and score
claims remain false until contest exact evidence lands.
