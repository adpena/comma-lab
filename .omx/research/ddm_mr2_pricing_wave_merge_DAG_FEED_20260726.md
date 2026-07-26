# DDM MR2 pricing-wave merge DAG feed — 2026-07-26

Status:
`CONTENT_CLEAN_SERIAL_MERGE_BLOCKED_AT_PF3B_NONTRIVIAL_CONFLICT`.

Inputs are pinned to PF3B `074955c6ad0900268b01b7be6e359677e254b0d0`,
WF7 `e3c2140d3a9f096082efa7dfb938fd6d7f9b31db`, and CB1
`2721704ab215806d34788b29bae227554a6b9b50`. MR2 base is
`5a55aa5914dc5675d2dd0fbe8bc225c77e2d9163`.

The independent review preserves three non-additive facts:

- PF3B: `delta_D_joint=-0.00010799244434957068`, `+860` bytes,
  `delta_S=+0.0004646462553354967`, integer break-even at most `162` bytes.
- WF7: exact lossless state restoration, `-1,797` payload bytes plus a
  21-byte directory, total `-1,776` bytes; state-container authority only.
- CB1: MyCar `delta_S=-0.051645614850883974` at `+319` bytes is admitted;
  Lane `delta_S=+9.21156940553832` at `+1,530` bytes remains
  `REJECTED/INSTANCE`.

The first required merge was aborted on `.omx/state/lane_registry.json` and
`tools/materialize_ddm_pf3_finite_prices.py`. WF7 and CB1 were not merged out
of order. MAIN must resolve PF3B on current main, greenup the resolved bytes,
then resume the serial wave. Only after all three land may c1 perform the
separate same-base composition measurement.

Competitive frontier: official leaderboard displayed `0.172`. Local
`0.1910828242` is custody-only. No launch, exact score, fire, reseal, paid
dispatch, promotion, or pointer movement.

MAIN landing review is required.
