Implemented and committed the charter as `8b8bb25e6fb2fc72f536051435a8024bf79562a9`.

Measured outcome:

- Full n600 native parity passed: 117,964,800 tokens, SHA `9ba2e52b…`, corrected-logit SHA `617e9fcf…`, CDF SHA `ba0d529b…`, RC64 bit position `921964`.
- `[M5-CPU 4-thread scorer-free]` token stage: `383.354385 s → 203.843359 s`, a measured `1.880632×` speedup.
- Derived Modal projection: `1,709.199804 s` total, still above the `1,600 s` fire gate by `109.199804 s`.
- The candidate remains unsealed, Python remains the default, and no scorer or exact row was dispatched.
- All token payloads, failed rungs, checkpoints, deterministic binaries, and build receipts were retained under `/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/`.

The implementation, custody bundle, measurements, and handoff are documented in the [final memo](/Users/adpena/Projects/pact/.omx/research/ddm_f26q_rc64_native_lowering_20260814.md:1). The executable equivalence check passed after commit.

## NEXT_IF_RESUMED

- `QUEUED_WITH_A_FIRE_ORDER` — owner: successor native-runtime arm; consumer store: the final memo and `receipts/result.json`; fire trigger: implement one hidden/logit optimization capable of removing at least `16.015113` measured M5 seconds, then repeat all six full-n600 identity gates.
- `QUEUED_WITH_A_FIRE_ORDER` — owner: MAIN; consumer store: `.omx/state/main_hot_state.md` and the Modal returned-artifact store; fire trigger: full parity plus derived Modal total ≤1,600 seconds, after which seal and dispatch with volume-backed raw retention.

## LIVE-HYPOTHESES

- An explicit NEON/AVX2 integer requantization kernel is plausible because sparse hidden/logit arithmetic still consumes `140.575280 s`.
- Producing int16 frame context directly is plausible because preparation and packing consume `30.169883 s`.
- Precomputed int16 conv-A class deltas may reduce the `22.297621 s` incremental-update stage.

## DEAD-ENDS

- RC64-only lowering is closed for this instance: probability plus RC64 is only `4.993713 s`; HPAC generation is dominant.
- Single-thread native execution passed parity but was slower than Python.
- Architecture-native compiler flags alone produced no M5 gain.
- Incremental conv-A alone was insufficient.
- One persistent OpenMP team was exact but slower than the adopted v13 baseline.
- Exact re-fire is closed on the current receipt because the projection remains `1,709.199804 s`. Own-vehicle frontier remains `S=0.7539807296911207` at `357,836 B` `[macOS-CPU advisory]` n600; this scorer-free arm did not move it.