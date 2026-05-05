---
name: CORRECTION — Lane Ω-W-V2 byte savings 40.98% was conv-only, full renderer is 20.59%
description: 2026-04-30 ~04:00 CDT (post-#272 dispatch). Earlier memory files cited "40.98% byte savings on REAL Lane G v3 renderer.bin [empirical]" without qualifying — that number is the eligible-conv-layer aggregate from test_omega_w_v2_real_archive.py, NOT the full renderer with FP16 protected-layer fallback. The actual full-renderer saving is 20.59% (296,776B → 235,660B). The archive saves 50,985B (694,074B → 643,089B) → Δ rate −0.0339 [derivation] → predicted contest-CUDA score ~1.016 [derivation] (NOT ~0.97 as earlier predicted). Predicted band [0.95, 1.02] [contest-CUDA].
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What happened

When #272 actually built the byte-identical Lane Ω-W-V2 stacked archive, the empirical numbers came out:

| Component | Bytes baseline | Bytes OWV2 | Delta | % |
|---|---|---|---|---|
| ASYM renderer.bin | 296,776 | 235,660 | −61,116 | −20.59% |
| Full archive | 694,074 | 643,089 | −50,985 | −7.34% |
| Δ rate term | — | — | — | −0.034 |

## Why the discrepancy with the "40.98%" claim

`src/tac/tests/test_omega_w_v2_real_archive.py` measures bytes saved on the ELIGIBLE conv-layer subset (the layers OWV2 actually re-quantizes). Aggregated over those layers in isolation, the number is 40.98%.

When the codec is run on the full renderer.bin, MANY layers fall through to the FP16 protected-layer fallback (BN params, biases, embeddings, etc.). Those layers are unchanged by OWV2 → the eligible-conv-only 40.98% gets diluted to 20.59% across the full state-dict.

## Why the discrepancy with the "predicted ~0.97" claim

Earlier prediction extrapolated:
```
Lane G v3 = 1.05  
Δ score from 40.98% on full renderer ≈ −0.078  
→ predicted 0.97
```

But −0.078 was derived from the wrong base (conv-only aggregate generalized to full renderer). Corrected derivation:
```
Lane G v3 = 1.05  
Δ archive = −50,985 B  
Δ rate = 25 × 50,985 / 37,545,489 = −0.0339  
→ predicted ~1.016 [derivation]
Band [0.95, 1.02] [contest-CUDA] with ±0.02 for FP16 + per-channel block-FP round-trip variance
```

## What to do going forward

1. **Stop citing "40.98% byte savings on Lane G v3 renderer.bin" without qualification** — it is conv-only-aggregate, not full-renderer.
2. **The actual delta to use for stack predictions is 20.59% on renderer.bin or 7.34% on the archive.**
3. **Predicted Ω-W-V2 stack contest-CUDA score is ~1.016**, NOT ~0.97. Sub-Quantizr 0.33 stack-EV is correspondingly lower.
4. **Awaiting actual contest-CUDA result** from #272 dispatch — instance 35886609 is mid-eval as of this memory write.
5. **All future OWV2-class codec measurements MUST report (a) eligible-subset %, (b) full-renderer %, (c) archive % — with the eligible-subset clearly tagged as such.** Update Check 89 (`encode-then-discard`) to also flag eligibility-subset/full-renderer aliasing.

## Files affected by the misleading "40.98%" claim

- `feedback_production_hardened_standard_definition_20260430.md` — Lane Ω-W-V2 V2 description "(40.98% empirical + #272 inflate handler firing)" needs corrigendum
- `project_session_state_checkpoint_20260430.md` — same
- `project_codec_stacking_composition_canonical_orders_20260429.md` — Lane Ω-W-V2 EV bands need recomputation
- `.omx/research/council_chain_integrity_audit_20260430.md` — Ω-W-V2 stack alternative section
- The 33% prior-hit-rate anchor in chain-integrity audit was based on the conv-only number; the corrected full-renderer saving still satisfies the prior, but the EV is reduced

## Cross-refs

- #272 dispatch report (commit 232b24ec + eff3020c + 9fef3382)
- `experiments/build_lane_g_v3_omega_w_v2_stack.py` (the canonical stacked-archive builder; print byte breakdown for any future stack)
- `src/tac/owv2_renderer_archive.py` (the OWV2 encode/decode + 11 round-trip tests at `src/tac/tests/test_owv2_renderer_archive_inflate.py`)
- Vast.ai instance 35886609 (label `lane_g_v3_omega_w_v2_stack_2026-04-30_b_a2`) — auth eval result lands ~15-20 min after this memory file is written
