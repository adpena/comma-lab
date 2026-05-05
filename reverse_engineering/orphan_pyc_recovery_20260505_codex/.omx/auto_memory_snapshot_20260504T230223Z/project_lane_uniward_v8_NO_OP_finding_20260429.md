---
name: Lane UNIWARD v8 = 1.14 IS A NO-OP — bit-identical Lane A masks shipped, encoder bytes discarded
description: 2026-04-29 PM Council B audit: shipped masks.mkv is SHA-identical (c07bd465...) to Lane A's. Stage 3 computes 8.6MB UNIWARD SLI1 payload then Stage 4 `cp $ANCHOR_DIR/masks.mkv $ITER_DIR/` discards it. Score 1.14 vs Lane A 1.15 is PURE CPU-vs-CUDA PoseNet drift. UNIWARD lane is BROKEN as currently wired. Daubechies-8 kernels are wrong (2-tap gradient stencils, ~12% vs canonical ~91% diagonal energy capture).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What the Council B audit found (council_uniward_v8_fridrich_shannon_audit_20260429.md)

### The lane is a NO-OP
- `experiments/results/lane_uniward_v8_modal/harvested_artifacts/lane_uniward_results/eval_work/extracted/masks.mkv` SHA = `c07bd465...`
- `experiments/results/lane_a_landed/iter_0/masks.mkv` SHA = same `c07bd465...`
- Bit-identical. The shipped archive is Lane A masks.

### The encoded bytes never reach the archive
Per Council B's read of `scripts/remote_lane_uniward.sh`:
- Stage 3: computes `texture_probability.pt` (4MB) + UNIWARD SLI1 payload (8.6MB)
- Stage 4: `cp $ANCHOR_DIR/masks.mkv $ITER_DIR/` overwrites with the anchor's masks
- The SLI1 bytes are written to disk but NEVER inserted into the archive ZIP

### The 1.14 score is just Lane A measured on CPU
- Lane A: 1.15 [contest-CUDA] (true baseline)
- Lane UNIWARD v8: 1.14 [contest-CPU advisory] = 1.15 + (PoseNet CPU drift -9.5%) + (SegNet CPU drift ~0%)
- Per CLAUDE.md "MPS auth eval is NOISE", CPU has smaller drift than MPS but is non-zero

### The Fridrich math is wrong as implemented
- Module: `src/tac/uniward_texture.py`
- Kernels: 2-tap gradient stencils (Sobel-class), NOT canonical Daubechies-8 wavelets
- Energy capture: ~12% on diagonal vs canonical ~91%
- Docstring says "texture probability" but output is proportional to INVERSE embedding cost (a junior reviewer fixing the "bug" by negating would invert the encoder)
- `(1+local_var)` multiplier is non-canonical
- Dispatcher's `tex_bool = (tex_f >= median)` accidentally walks the orientation correctly end-to-end
- Tier: "approximation; would not be co-authored as published canonical UNIWARD; acceptable as a sprint-shortcut tagged `[uniward-2tap-approx]` but currently un-tagged"

### V7→V8 14000× PoseNet jump explained
- V7: anchor `submissions/baseline_dilated_h64_0_90/iter_0/masks.mkv` = 48x64 AV1 (verified ffprobe)
- V8: anchor `lane_a_landed/iter_0/masks.mkv` = 384x512
- Same disaster as 2026-04-21 "MASKS.MKV AT 48x64 DESTROYED THE SCORE" (CLAUDE.md catastrophic failures section)
- Check 76 STRICT already blocks this permanently — V7 was dispatched BEFORE Check 76 landed

## Implications

### For the harvested score
- Score 1.14 is NOT a Phase 1 contender — it's just Lane A under CPU eval
- Stacking with PD-V2/Ω-W-V2/LCT/etc would give ZERO additional bp from the UNIWARD direction (no archive bytes changed)
- Promotion to Phase 1 GREEN: REJECTED until V9 lands with proper wiring

### For the $0.50 CUDA re-eval the user authorized
- Council 7/7 APPROVE the spend, but ONLY to lock the CPU-vs-CUDA drift estimate at -0.01/+9.5% PoseNet
- It will NOT validate UNIWARD as a real lane — it will just measure Lane A on CUDA (which we already have at 1.15)
- Cheaper alternative: `sha256sum experiments/results/lane_uniward_v8_modal/.../masks.mkv vs experiments/results/lane_a_landed/iter_0/masks.mkv` proves no-op for free
- Recommendation: SKIP the $0.50 unless we want to formalize the CPU-CUDA drift number for documentation

### For Lane UNIWARD V9 (the proper rebuild)
The Council B report sketches V9:
- Build SLI1 inflate decoder so the bytes actually go into the archive
- Replace 2-tap gradient stencils with Daubechies-8 wavelets (canonical UNIWARD)
- Preserve class boundaries (current impl can corrupt class transitions)
- Retrain renderer on UNIWARD-weighted loss (not just bolt-on at archive build)
- Predicted band: 0.96-1.04 conservative stack (with PD-V2 + LCT + Ω-W-pose-only) [prediction]
- Moonshot 0.78-0.84 if Selfcomp-eligible renderer.bin lands (Ω-W-V2 -200bp on Selfcomp weights pool) [prediction]

## Permanent fix needed

Add a STRICT preflight check that scans `scripts/remote_lane_*.sh` for the pattern:
```
ENCODE_PAYLOAD ($PAYLOAD_FILE)
...
cp $ANCHOR_DIR/masks.mkv $ITER_DIR/    # ← THIS DISCARDS THE PAYLOAD
```
If a script computes a payload but the archive build doesn't include it, fail. Working name: Check 88 (`check_remote_lane_scripts_use_computed_payloads`).

## Cross-refs

- Council report: `/Users/adpena/Projects/pact/.omx/research/council_uniward_v8_fridrich_shannon_audit_20260429.md` (30.4K)
- Lane script: `scripts/remote_lane_uniward.sh`
- UNIWARD module: `src/tac/uniward_texture.py`
- Memory (now SUPERSEDED): `project_lane_uniward_v8_harvested_1_14_advisory_20260429.md` — the "competitive with Lane A" framing was wrong; it IS Lane A
- CLAUDE.md "MASKS.MKV AT 48x64 DESTROYED THE SCORE" (V7 reproduction)
- CLAUDE.md "Auth eval EVERYWHERE" (the rate term depends on archive bytes — wrong archive = wrong score)
