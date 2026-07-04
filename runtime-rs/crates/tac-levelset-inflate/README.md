# `tac-levelset-inflate`

Rust native port for the level-set (#171/#205 capstone) **inflate DECODE** primitives whose
Python oracle is **bit-exact-portable**. Task #282 first increment (2026-07-03).

## Status

`v0.1.0` — byte-for-byte parity **GREEN** on committed golden vectors derived from REAL contest
`gt_poses` ξ. Two primitives lowered, both bit-exact-by-construction:

| fn | Python oracle | golden vector | basis |
|----|---------------|---------------|-------|
| `range_decode::decode_static_symbols` | `tac.lossless.range_coder.decode_static_symbols` | `levelset_xi_range_decode_v1` | pure-integer CACM range decode |
| `xi_column::decode_xi_column` | `tac.boundary_math.xi_pose_coder._channel_delta_decode` | `levelset_xi_column_delta_v1` | arithmetic decode + `+lo` + prefix-sum + `seed` |

## Feasibility boundary (why only these two — MEASURED)

The level-set inflate's wall-clock hot path is the coord-INR neural forward (matmuls + hosc
`tanh(β·sin)` `_act`). It is **NOT bit-exact-portable** against the numpy-fp64 oracle:

- numpy `@` = Apple Accelerate `dgemm` — a portable fma-off Rust matmul diverges on **4497/6144**
  elements at ~1e-14 (Accelerate's blocked+FMA reduction order is not reproducible).
- numpy fp64 `tanh` ≠ scalar system libm `tanh` — **844/4096** at 1 ULP (numpy `sin`/`cos`/`exp`
  DO match libm; `tanh` is the blocker).

So the only genuinely bit-exact piece is the **#257 store-nothing ξ pose-carrier decode** (pure
integer) — this crate. Deferred, with the honest route recorded in `binary_source_audit.md`:

- the neural forward → **fp32-TOLERANT** Rust (gated on task #281's fp32 score-preserving verdict;
  not bit-exact → needs a score-equivalence gate, not `assert_sha256_parity`);
- the **lane AA-SDF render-band rasterizer** (`_lane_coverage`) → MEASURED matmul/transcendental-FREE
  ⇒ bit-exact-able AND a real per-pixel wall-clock contributor ⇒ the **highest-value NEXT increment**.

## Discipline

- Python oracle stays canonical; every primitive has a sha256 parity gate on the SAME real bytes.
  Negative control verified: a 1-bit stream flip fails the gate.
- Payload-clean: zero learned/video-derived constants in the binary (`embedded_constants_audit.txt`);
  the ξ payload is COUNTED in `archive.zip`, the decode algorithm is FREE (rule 118).
- MEANS: decode wall-clock only (ξ decode ~0.16 ms one-time; ~73× the Python oracle but negligible
  vs the ~16-min inflate). Pointer 0.19110 UNMOVED.

## Build / test / prove

```bash
cd runtime-rs
cargo test -p tac-levelset-inflate --offline           # 11 tests incl. 2 real-xi SHA parity gates
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/python_reference_equivalence_test.py
```

See `rebuild_instructions.md` for golden-vector regeneration + the bench + the negative control.
