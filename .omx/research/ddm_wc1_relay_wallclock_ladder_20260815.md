# RELAY → ddm_wc1 + LADDER REGISTER — advisory wall-clock iterates PAST the wc1 bar (2026-08-15)

**Operator steer (binding, 2026-08-15, second in the series):** "The wall clock can be
further iterated and optimized from here." Wall-clock is an ITERATED LADDER, not a
one-shot arm. This memo registers the measured rung ladder and relays in-scope guidance
to the live wc1 arm.

## The measured wall (base leg r2 receipt, 4-thread CPU, hv1 archive 80d9c8c6…)

Receipt: `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json`

| Segment | Seconds | Share |
|---|---:|---:|
| Total advisory eval | 2,472.65 | 100% |
| Decode (inflate) | 1,907.24 | 77.1% |
| Scorer (evaluate) | 563.96 | 22.8% |

Decode sub-profile (launcher_r2/run.log:30): render 1,179.59 · token 642.14 · io 79.68 ·
setup 2.65. Native token stage sub-profile (f26r memo): sparse hidden+logits **114.39 s
of 147.01 s (78%)** · incremental conv 16.98 · context 2.43 · RC64 4.36.

## The ladder (each rung identity-gated; rungs are cumulative)

- **R0 = wc1 (LIVE, unchanged):** decode 1,907 → ≤700 bar / ~430 projected
  (parallel render + native tokens + section cache). Post-R0 total ≈ 430 + 564 ≈ ~995 s.
- **R1 (wc1 IN-SCOPE if cheap):** the native token residual — sparse hidden+logits is
  78% of the token stage. Further blocking/NEON/threading on that one loop; the f26r
  forced-scalar twin pattern is the portability gate. Do NOT delay the primary bar for
  this; take it only if the composition work leaves slack.
- **R2 (wc1 IN-SCOPE as a MEASUREMENT, build optional):** chunk-parallel token decode.
  The `.f26_decode_checkpoints` machinery implies chunked decode with checkpoint
  boundaries — MEASURE whether RC64/context state at boundaries permits independent
  chunk decode (context-free boundaries ⇒ chunks parallelize like frames). If yes and
  cheap, build; if no, one paragraph in the memo closes it honestly.
- **R3 (NAMED SUCCESSOR — the scorer half, 563.96 s):** pairs are independent; each
  pair's SegNet/PoseNet forward at pinned (threads, batch) is deterministic; per-pair
  components collected from N workers and aggregated IN CANONICAL PAIR ORDER reproduce
  the serial numbers bit-identically (same reassociation-avoidance as the render lever).
  563.96 → ~100–150 s at 4–6 workers. wc1: do NOT build this — but DESIGN the render
  worker-pool abstraction so it is reusable for a scorer pool (process-parallel + pinned
  per-worker instrument + ordered assembly + sha parity gate). MAIN charters wc2 at
  wc1's landing.
- **R4 (RESEARCH-GATED SUCCESSOR — GPU advisory instrument):** MLX/Metal render+scorer
  as a NEW advisory axis (runtime-lift grant). Discriminator BEFORE any adoption: does
  GPU drift CANCEL in same-instrument DELTAS below the ±3.5e-6 admission band? (pk4 law:
  0.55% rel pose drift ≈ 1e-4 S absolute — 30× the band — so only delta-cancellation
  can make this admissible. Measure: base ×2 + one known candidate on the GPU
  instrument; compare deltas vs the CPU-instrument deltas.) Potential: minutes per eval.
  Not started until R0+R3 land — those get ~4.7× total with ZERO instrument change.

## Arithmetic of the ladder

R0+R3 (no instrument change, bit-identical): 2,473 s → ~530–850 s ≈ **~9–14 min per
advisory n600 eval** (from 41 min). R4, if the delta-cancellation gate passes, takes the
loop to low single-digit minutes on a new advisory axis.

## Discipline reminders (unchanged from the wc1 charter)

Bit-identity fail-closed per rung · instrument pins in every receipt · ALWAYS KEEP THE
PAYLOAD (per-frame sha manifests + one retained reference payload per archive) ·
watched detached launches with success_receipts · serialize behind the live advisory
slot · shipping packet untouched.
