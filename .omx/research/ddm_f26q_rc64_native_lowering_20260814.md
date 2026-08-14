# DDM F26Q — native F26 HPAC/RC64 lowering (2026-08-14)

## Verdict

`DECODE_ENGINEERING_GATED`, `INSTANCE(v13 native F26 on M5 plus the measured
Modal/M5 stage ratio)`.

The full native receiver is byte-identical, deterministic, resumable, and
retains every decoded token field. It lowers the measured M5 token stage from
383.354385 s to 203.843359 s, a 1.880632x speedup. That is real but not enough:
the measured 6.818547x cross-host stage ratio gives a **DERIVED**, not measured,
Modal token wall of 1,389.915578 s and a projected total of 1,709.199804 s.
The 1,600-second fire gate is missed by 109.199804 projected seconds, equivalent
to 16.015113 additional M5 token-stage seconds. The candidate therefore remains
unsealed, Python remains the default, and no exact row was dispatched.

This arm did not score a candidate, did not run a scorer, and did not move any
frontier.

## Corrected problem statement

The charter called the 2,613.920-second Modal stage “sequential Python RC64.”
The live call graph and earlier native-runtime evidence show that RC64 entropy
decode was already a compiled C call. The actual hot surface was Python/Torch
causal sparse integer HPAC probability generation around that call. The full
native grant made the admissible cure a fused native lowering of HPAC,
probability construction, and the same RC64 recurrence. Porting only RC64 again
would have repeated a closed dead end.

## RECALL EVIDENCE

Sources and queries consulted before implementation:

- Full corpus query:
  `.venv/bin/python tools/corpus_query.py --stores research,equations,memory,dag,council,tasks,docs --top 40 --json 'F26 RC64 Python sequential native lowering 2613.9 token_decode decoder_bit_position 921964'`.
- Canonical equations registry:
  `.venv/bin/python tools/list_canonical_equations.py --json`.
- Content search across `.omx/research/`, the live runtime, canonical index/DAG,
  specifications, and task surfaces for `F26`, `RC64`, `native`, `HPAC`, and the
  exact token SHA.
- Governing sources: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the
  charter, common contract, and parent memo
  `.omx/research/ddm_f26p_runtime_cpu_lift_20260814.md`.

Found beyond the charter seeds:

- `.omx/research/ddm_rc64p_native_cpu_decode_20260810.md` measured direct entropy
  calls at only 1.11–3.13 s and more than 99.5% of receiver wall elsewhere.
- `.omx/research/ddm_na6_arc_negative_audit_20260811.md` upheld that
  `INSTANCE` negative and explicitly left native probability-generation
  lowering open.
- The live F26 Python receiver already used `runtime/entropy/rc64_backend.c` via
  a ctypes wrapper; Python/Torch `selected_logits` produced the expensive
  corrected-CDF inputs.

Plan change: no entropy-only port was attempted as the deliverable. The native
implementation fused sparse HPAC evaluation, exact probability formation, and
RC64, while retaining the Python receiver as semantics authority.

## Element profile on the real stream

All prefix rows are contiguous prefixes of the pinned real archive and are
scope-only, not population verdicts. Payloads were retained. The full-field row
is the final native measurement.

| Axis / scope | Measured wall | Selected-logit or native kernel | Frame context | Probability + RC64 | Retained token payload |
|---|---:|---:|---:|---:|---|
| `[M5-CPU 4-thread scorer-free real-stream profile]`, Python n4 | 2.911862 s | 2.583645 s | 0.138857 s | 0.101406 s including correction | `reference_prefix_n4.u8`, 786,432 B, SHA `4a5047ee…` |
| same, Python n32 | 26.186863 s | 20.931370 s | 1.102007 s | 0.813304 s including correction | `reference_prefix_n32.u8`, 6,291,456 B, SHA `870cc4d3…` |
| `[M5-CPU 4-thread scorer-free native token decode]`, native n32 | 10.913828 s | 8.599735 s fused | 1.628736 s | 0.260473 s | `native_tokens_v13_n32.u8`, byte-identical SHA `870cc4d3…` |
| same, native n600 | **203.843359 s** | **168.277530 s fused** | **30.169883 s** | **4.993713 s** | `native_tokens_v13_n600.u8`, 117,964,800 B, SHA `9ba2e52b…` |

The real full-field operation census is 600 frames, 190 causal groups per
frame, 114,000 group calls, 48 patches per call, 64 channels, 23 active conv-A
offsets, and 117,964,800 output symbols. The Python n4 linear extrapolation
(436.779 s) and n32 linear extrapolation (491.004 s) remain explicitly
`DERIVED`; neither substitutes for the measured n600 wall.

## Native implementation and custody

The adopted v13 source is:

- `runtime-rs/native/f26-hpac/f26_hpac_native.c`: generic C receiver with exact
  round-to-even/clamp semantics, incremental causal conv-A state, vectorized
  depthwise channel reductions, persistent workspaces, exact probability
  frequencies, and the original RC64 recurrence.
- `experiments/ddm_f26q_f26_hpac_native.py`: archive-to-native binding,
  exact-integer model export, initial plus every-25-frame crash checkpoints,
  atomic receipts, retained token memmap, and trace digests.
- `experiments/ddm_f26q_rc64_native_lowering.py`: storage preflight, fresh
  candidate construction, deterministic repeat build, profiling, parity gates,
  gated sealing, and typed result.
- `experiments/ddm_f26p_f26_cpu_lift.py`: explicit
  `--token-decoder native-hpac` integration. Default remains `python`.

The retained repeat binaries are both 52,816 B and both SHA
`b791acf032c7f373beb329c3241323af04f8e939dd8c0195ac84ae908221779c`.
The binary/source audit found no learned/video-derived payload embedded in code.
All weights, residual values, causal plans, frame context, and stream bytes come
from the counted archive. Rebuild commands, source audit, embedded-constants
audit, payload manifest, and executable equivalence check are in
`runtime-rs/native/f26-hpac/`, with the executable Python equivalence check at
`experiments/ddm_f26q_python_reference_equivalence_test.py`.

Durable root:
`/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/`.
No payload from any completed or failed rung was deleted. The full n600 token
field has an initial checkpoint and 24 distinct through-stage receipts ending
at frame 599.

## Full-field byte-identity receipt

All required gates passed on the full n600 field:

| Gate | Expected | Observed | Result |
|---|---|---|---|
| token bytes | 117,964,800 | 117,964,800 | PASS |
| decoded token SHA-256 | `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` | exact | PASS |
| RC64 bit position | 921,964 | 921,964 | PASS |
| corrected quantized-logit SHA-256 | `617e9fcfc967c200f1ecc8bea93dd45a22f7af2a050092f982169b5f5e5a3523` | exact | PASS |
| corrected CDF-input SHA-256 | `ba0d529b7eaf6e16da1f62fc1cc7ca43ccc1b989356a68b8d37988088cb7c7ff` | exact | PASS |
| digest scope | full field | full field | PASS |

Primary receipt:
`/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/receipts/native_run_v13_n600.json`.

## Timing verdict and residual

| Quantity | Value | Type |
|---|---:|---|
| Python reference token stage, M5 | 383.354385 s | MEASURED |
| native v13 token stage, M5 | 203.843359 s | MEASURED |
| M5 speedup | 1.880632x | MEASURED ratio |
| Modal/M5 token-stage ratio | 6.818547x | MEASURED ancestor ratio |
| projected Modal native token stage | 1,389.915578 s | DERIVED |
| fixed non-token wall from failed Modal run | 319.284225 s | MEASURED ancestor subtraction |
| projected Modal total | **1,709.199804 s** | DERIVED |
| fire gate | ≤1,600 s | charter gate |

Measured native sub-stages on the M5 full field:

- sparse hidden/logits: 140.575280 s;
- frame-context preparation and exact int16 packing: 30.169883 s;
- incremental conv-A update: 22.297621 s;
- probability plus RC64: 4.993713 s;
- checkpoint persistence: 3.131017 s;
- trace digests: 1.503176 s;
- all other reported setup/boundary/finalization: below 1 s combined.

The focused remaining implementation surface is sparse hidden/logit arithmetic,
not RC64. A new full-field token receipt must remove at least 16.015113 measured
M5 seconds before the exact-row fire order can activate. No more broad
profiling is justified.

## Conditional exact-row fire order

Disposition: `QUEUED_WITH_A_FIRE_ORDER`; owner: MAIN; consumer store: this memo
plus `.omx/state/main_hot_state.md`; fire trigger: a new full-n600 receipt passes
all six identity gates and projects Modal total at or below 1,600 s.

When triggered, MAIN should use the same canonical
`experiments/modal_auth_eval_cpu.py` chain, the unchanged archive SHA
`f0ba4bb4…`, the sealed native stage, and four CPU threads. The Modal volume
must retain the complete 3.6 GB raw output or a volume-backed per-frame manifest
that identifies every retained frame payload; container-ephemeral raw is
forbidden. The returned bundle must contain the exact archive/runtime hashes,
stdout/stderr, result JSON, raw SHA/bytes, per-frame manifest, command/config,
hardware axis, and failure sentinel. Do not fire while the projection is above
1,600 s.

## NEXT_IF_RESUMED

- `QUEUED_WITH_A_FIRE_ORDER` — owner: successor native-runtime arm; consumer
  store: this memo and `receipts/result.json`; fire trigger: begin only from the
  retained v13 full profile, and implement one measured hidden/logit cure that
  can remove at least 16.015113 M5 seconds while preserving all six identity
  gates.
- `QUEUED_WITH_A_FIRE_ORDER` — owner: MAIN; consumer store:
  `.omx/state/main_hot_state.md` and the Modal returned-artifact store; fire
  trigger: full n600 parity plus derived Modal total ≤1,600 s; then seal v13's
  successor and dispatch the canonical contest-CPU chain with volume-backed raw
  retention.

## LIVE-HYPOTHESES

- Specializing the 64-channel hidden requantization into an explicit NEON/AVX2
  integer kernel is plausible because 140.575 s remains in that exact element
  and the compiler still cannot vectorize the clamp/round/context loop as a
  whole.
- Precomputing archive-derived conv-A class deltas as int16 is plausible because
  22.298 s remains in incremental updates that currently load two int8 vectors
  and subtract them for every nonzero causal symbol.
- Producing int16 context directly inside the quantized frame-context path is
  plausible because the present correctness check and float-to-int16 pack costs
  part of the measured 30.170 s, while the full-field trace proves those values
  are exactly integral for this pinned archive.

## DEAD-ENDS

- Entropy-only RC64 lowering as the CPU cure is closed for this instance: direct
  entropy work is only 4.994 s in the final full field and was 1.11–3.13 s in
  prior cells; the hot surface is HPAC hidden/logit generation.
- A single-thread native fused receiver is closed as a performance choice: n4
  parity passed, but its 6.554 s wall was slower than Python.
- Architecture-native compiler flags alone are closed on M5: they produced the
  same binary and no kernel speed gain.
- The first incremental conv-A formulation alone is closed as a sufficient
  cure: parity passed but n4 kernel time was unchanged until context and
  depthwise work were also lowered.
- One persistent OpenMP team across all 190 groups is closed for this instance:
  it preserved n32 parity but measured 11.093062 s versus v13's 10.913828 s.
- The exact-row re-fire is closed at the current receipt: projected total is
  1,709.199804 s, above the 1,600-second gate.

Own-vehicle frontier remains S=0.7539807296911207 at 357,836 B
`[macOS-CPU advisory]` n600; this scorer-free runtime arm did not move it.
