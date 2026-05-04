---
name: Lane Bit-level archive optimization — Level 0 → Level 1 SCAFFOLD landed (REDIRECTED SCOPE)
description: 2026-04-30. Phase 3 Lane 15 (Bit-level archive optimizer) advanced from Level 0 (sketch only) to Level 1 (SCAFFOLD). 26/26 synthetic tests passing. Council design doc landed with REDIRECTED SCOPE — original "Carmack 50 KB outer-container" target is dead per the empirical audit in src/tac/custom_binary_container.py (Lane A archive has only 328 B / 0.05% ZIP overhead). Redirected to PAYLOAD-side optimization: sub-FP16 bit-packing of poses.pt (~7 KB main win) + cross-stream shared Brotli dictionary (~1-2 KB). impl_complete gate satisfied.
type: project
authoritative_for: lane_bit_level_archive_opt_level1_scaffold
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR

Lane Bit-level archive optimization advanced from Level 0 (sketch only) to **Level 1 (SCAFFOLD)** with **REDIRECTED SCOPE** that the council had to confront honestly:

- **Original target (Carmack):** "Rip up ZIP container, save 50 KB" — **DEAD** per the empirical audit hardcoded in `src/tac/custom_binary_container.py` docstring: Lane A's reference archive (694,045 B) has only 328 B of ZIP overhead = 0.05% of file. Outer-container rewrites cannot plausibly save 50 KB.
- **Redirected target (Carmack revised + Hotz extension):** Payload-side bit-level optimization. Two main wins:
  1. **Sub-FP16 bit-packing of poses.pt** via per-dim affine quantizer (the BLPS wire format) — saves ~7 KB on Lane G v3 (1200 frames × 6 DoF: 14.4 KB FP16 → ~7.2 KB int8 + ~64 B header)
  2. **Shared Brotli dictionary** for renderer.bin + poses.pt — saves ~1–2 KB IF the streams share common subsequences (empirical TBD)

**Predicted band [prediction]:** **1–8 KB net savings on Lane G v3 archive** = -0.001 to -0.005 score. Honest tagging per Quantizr's adversarial objection.

## Files landed

- `src/tac/bit_level_archive_optimizer.py` — 412 LOC; payload-side scope only; outer ZIP container OUT OF SCOPE per the audit; pure CPU; required-keyword args
- `src/tac/tests/test_bit_level_archive_optimizer.py` — 26 tests; covers per-dim quantizer fit (recovers dynamic range, handles offsets), quantize/dequantize roundtrip with bounded LSB error, BLPS wire format roundtrip (int8 + int4 packed), CRC + magic + version validation (rejects bad/truncated/corrupt), shared Brotli dict (extracts cross-stream subsequences, skips singletons, respects max size), archive byte composition audit (synthetic ZIP), orchestrator predicted savings (~7 KB on Lane G v3-shape input)
- `.omx/research/council_lane_bit_level_archive_design_20260430.md` — full council deliberation; **Carmack channeled and TOOK THE HIT** ("My original 50 KB target was wrong; the audit module already proved it"); Hotz + Selfcomp + Boyd + Shannon + MacKay + Contrarian seats; **Contrarian conditional GREEN on honest scope tagging**

## Council verdict

**Adopt: Payload-side scope only. Outer ZIP container OUT OF SCOPE per the empirical audit.** All 7 voices signed off after Carmack acknowledged the original 50 KB target was infeasible. Lane MDL framework (just-landed, sibling Phase 3 lane) is the natural ranking tool for empirically comparing the bit-packed-poses + shared-dict-Brotli + baseline alternatives.

The Carmack confession is preserved in the design doc as a model for non-conservative-but-evidence-respecting council debate: when empirical evidence forces a scope reduction, the council updates the lane scope rather than denying the evidence.

## Tests

```
PYTHONPATH=src .venv/bin/python -m pytest src/tac/tests/test_bit_level_archive_optimizer.py
26 passed in 0.13s
```

Notable: `test_quantize_rejects_dim_mismatch` initially used `match="(5, 3)"` regex which collapses to a literal-comma group; fixed to `match=r"got \(5, 4\)"` (escaped parens).

## Lane registry status

```
lane_bit_level_archive_opt: level 0 → 1
gates: impl_complete=true; remaining 6 gates false
```

## Why the predicted band changed from "50 KB target" to "1–8 KB target"

The audit in `src/tac/custom_binary_container.py` was performed on Lane A's actual archive bytes. It found:
- Total: 694,045 B
- Three member payloads: 693,717 B compressed
- ZIP central directory + local file headers: 328 B = **0.05%** of file

There is no scenario where outer-container rewrites save 50 KB on this archive — the byte budget is simply not there. The honest predicted band is the payload-side achievable: 7 KB on poses.pt + 1–2 KB on shared Brotli dict = 8 KB ceiling.

This is exactly the council-conduct discipline CLAUDE.md mandates — "evidence forces, not conservative bias."

## Phase ordering ahead

- Phase B (Level 1) ✅ THIS COMMIT
- Phase C (Level 2 prep) — empirical audit on Lane G v3 reference archive; produce `reports/lane_bit_level_byte_composition.json`; tag `[empirical:...]`
- Phase D (Level 2) — wire `PoseStreamBitPacker` into compress.sh; inflate handles BLPS magic
- Phase E (Level 3 path) — STRICT preflight check (bit-packed pose has matching inflate handler) + 3-clean-pass adversarial review
- Phase F — measure on Lane G v3 contest-CUDA; tag result `[contest-CUDA]`

## Cross-references

- CLAUDE.md "Council conduct — non-negotiable" (council redirected scope; not conservative bias — empirical evidence forced it)
- `feedback_production_hardened_standard_definition_20260430.md`
- `.omx/research/council_lane_bit_level_archive_design_20260430.md`
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 15 Bit-level archive optimizer"
- `src/tac/custom_binary_container.py` (the HARD TRUTH BOMB audit that REDIRECTED scope; foundational evidence)
- `src/tac/archive_diet_pack.py` (Subagent L baseline; 14.7 KB savings on Lane A)
- `src/tac/mdl_bayesian_codec.py` (just-landed sibling Lane MDL — natural ranking tool)
- Shannon 1948 — source coding theorem; Brotli RFC 7932 (shared dictionary support)
