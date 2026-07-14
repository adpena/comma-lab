# Codex session summary — 2026-07-14T11:22:42Z

**Lane:** `margin_adaptive_mixed_precision`  
**Status:** `BUILD_AND_LOCAL_PREFLIGHT_GREEN; MAIN_M5_MAX_N600_OWED`  
**Pointer:** unchanged; `score_claim=false`, `pointer_moved=false`

## Landed

- Composed the exact signed-int64/per-output-channel fixed-point suite into arbitrary, statically
  range-safe per-layer precision maps, native exact int8/int16/int32 operand storage buckets, and the
  existing custom-Metal SegNet lowering.
- Built strict classwise output-interval certification, frozen tie-rule composition, and exact
  finite-profile per-pixel margin waterfill.
- Built a resumable exact-pairs-0..599 probe with frozen 0..263 design selection, untouched
  264..599 validation, ten-process digest, measured CPU-vs-Metal latency, and durable stage receipts.
- Added a typed launch-inert DSL policy, registration-inert canonical equation, standalone DAG FEED,
  focused tests, and the exact M5-Max host command.

## Important correction

The executable win is a frame-independent **per-layer** profile. The per-region/per-pixel annulus
waterfill is a lower bound, not a native sparse speed path, because global SegNet dependencies close
exact support to the full frame. Exact int64 also describes accumulator semantics, not fp32-logit
equality; margin/tie certification remains load-bearing. Blind actual-model derivation found the
physical storage phase boundary: cap8 uses int8, caps10..16 use int16, and cap18+ uses int32. The
receipt therefore refuses to attribute a physical-width win to an exact cap18+ profile.

Verification closed at `49 passed` plus Ruff, `py_compile`, shell syntax, diff check, and three clean
review passes. Metal remains deliberately unmeasured here.

## MAIN action

```zsh
cd /Users/adpena/Projects/pact
./tools/run_margin_adaptive_mixed_precision_n600_host.command
```

Expected durable result:
`experiments/results/margin_adaptive_mixed_precision_20260714/margin_adaptive_mixed_precision_n600.json`.
Do not promote from bit count alone: require full n600 exact/certificate custody, one digest across ten
fresh processes, and measured positive SegNet wall-clock speedup. A byte-closed score row remains a
later, separate gate.

## Inbox consumed

- per-arm through `2026-07-14T11:12:34Z` (exact-int framing and Molt analogy)
- fleet broadcast through `2026-07-14T11:20:13Z` (latest surrogate-VJP handoff was orthogonal)
