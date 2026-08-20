# CHARTER — ddm_f26r_hpac_hot_stage_final_rung (2026-08-14, f26q's sealed successor)

PARENT: .omx/research/ddm_f26q_rc64_native_lowering_20260814.md (commit
8b8bb25e6fb2fc72f536051435a8024bf79562a9) — READ IT + its receipts/result.json
FIRST. This charter executes f26q's own QUEUED_WITH_A_FIRE_ORDER row verbatim.

## THE MEASURED STATE (f26q receipts, all retained under
/Volumes/VertigoDataTier/pact/ddm_f26q_rc64_native_20260814/)

- Native lowering PASSED all identity gates at full n600: 117,964,800 tokens,
  token sha 9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52,
  corrected-logit sha 617e9fcf…, CDF sha ba0d529b…, RC64 bit position 921964.
- Token stage [M5-CPU 4-thread scorer-free]: 383.354385 s → 203.843359 s
  (1.880632× measured).
- Derived Modal projection: 1,709.199804 s total — 109.199804 s ABOVE the
  1,600 s fire gate (contest budget 1,800 s; the 1,600 s gate is the derived
  safety margin because the projection is DERIVED from a single measured
  M5→Modal stage ratio, not measured on Modal).
- MECHANISM (the big f26q finding): probability+RC64 entropy coding is only
  4.993713 s — the wall is HPAC GENERATION: sparse hidden/logit arithmetic
  140.575280 s · int16 frame-context prep/packing 30.169883 s · conv-A
  incremental update 22.297621 s.

## THE TASK — remove ≥16.015113 measured M5 seconds with ONE (or more) of
f26q's own live hypotheses, then re-prove ALL SIX full-n600 identity gates:

1. NEON (M5) integer requantization kernel for the sparse hidden/logit
   arithmetic (140.6 s pool — 16 s = 11.4% of this stage; note the SHIPPED
   target is Modal x86, so any SIMD path MUST have a portable/AVX2-or-scalar
   twin proven bit-identical — the M5 NEON build is the local measuring
   instrument, the x86 build is what ships).
2. Direct int16 frame-context production (30.2 s pool).
3. Precomputed int16 conv-A class deltas (22.3 s pool).
Rungs may compose. After each rung: full-n600 parity (token sha 9ba2e52b… +
logit sha + CDF sha + bit position + repeat determinism + x86-twin parity via
Rosetta/emulation or scalar-reference equality) — a rung without all six
gates is telemetry, not a candidate.

## FIRE ORDER (MAIN executes; Standing GO covers)
When measured M5 total-stage time projects Modal ≤1,600 s (same derivation as
f26q's — cite it), emit SEALED_FIRE_ORDER.json for the contest-CPU exact row:
same canonical modal_auth_eval_cpu chain, archive
f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de @186,269 B,
runtime = the lifted F26 module + the native decoder integrated behind the
proven flag, AND volume-backed raw retention (the 08-14 failed run's 3.6 GB
raw was container-ephemeral — the payload law demands the re-fire persist raw
or its per-frame sha manifest to the comma-auth-eval-cache-artifacts volume).

## OPTIMAL FORM
PINS: f26q commit 8b8bb25e6fb2fc72f536051435a8024bf79562a9 · f26p commit
a5e1f6027018f001975619f1aff187c75777fc52 · archive sha f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de ·
token-parity oracle 9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52.
Reference form = f26q's v13 adopted baseline (its receipts name the failed
rungs — do NOT re-try its 5 dead-ends: RC64-only, single-thread native,
compiler-flags-only, incremental conv-A alone, persistent OpenMP team).
MECHANISM reductions FORBIDDEN (bit-identity or nothing). Payload law:
retain every rung's binary + build manifest + parity receipt. Decode-time
law binds: if the gate is still missed, report the typed residual seconds +
the next named stage — never close the CPU axis. Git-blocked ⇒ memo SHA
handoff.

## OUTPUT
Work dir /Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/.
Memo .omx/research/ddm_f26r_hpac_hot_stage_final_rung_20260814.md: per-rung
measured seconds + parity receipts + the projection arithmetic + sealed
fire-order (or typed decode-engineering-gated residual). Serializer commit,
[no-triality] [p0-ledger-ok], no co-author trailer.
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
