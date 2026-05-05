---
name: qat_finetune.py has CHAINED arch-drift bugs — deep refactor needed
description: 2026-04-27: Lane B FP4 QAT failed twice. (1) load_float_checkpoint built model from CLI defaults (motion.head=[4]) but loaded checkpoint had [6]. Patched today by returning loaded model directly. (2) load_asymmetric_checkpoint_fp4 reload at qat_finetune.py:995 has the SAME pattern — constructs new model from cfg + reads shape-mismatched FP4 binary. Three or more sites in qat_finetune need the same fix.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 on Lane B (Vast.ai 4090).**

**Bug chain:**

1. `load_float_checkpoint(model, path)` constructed `model` from CLI args (defaults: motion.head=[4]) then loaded the dilated-h64 .bin (motion.head=[6]) → shape mismatch.
   - **Fixed today** (commit pending): function now returns the loaded model for ASYM/FP4A paths instead of mutating the wrong-arch model.

2. `qat_finetune.py:995` calls `load_asymmetric_checkpoint_fp4(fp4_path.read_bytes(), device=...)` to verify the FP4 export. This reload constructs a fresh model from the binary header — but somewhere in renderer_export.py:1418 it does `module.weight.copy_(flat.reshape(shape))` where the underlying module's weight is shape [2] but the flat buffer is shape [6]. Same arch-drift class, different code path.
   - Likely cause: `load_asymmetric_checkpoint_fp4` reads the binary header for some fields but uses defaults for others, producing a hybrid arch that doesn't match either the binary OR the original arch.

3. There may be MORE — every "construct model, then load state into it" call site in qat_finetune.py is at risk.

**How to apply:**

1. **DON'T use qat_finetune.py on the dilated-h64 baseline (motion.head=[6])** until the chained bugs are fixed.
2. Quick alternative for FP4-style rate compression: brotli-compress renderer.bin INSIDE the zip archive. Saves ~12% (~35KB) without touching QAT logic. Rate -0.023 score points.
3. When fixing qat_finetune properly: audit every `load_*_checkpoint(...)` call, replace with "use loaded model directly" pattern. Add an integration test that round-trips the dilated-h64 baseline through QAT and verifies output shape matches input.
4. Until fixed, qat_finetune is OK for the 4-channel renderers (most other profiles) but not the 6-channel dilated-h64.

**Cost of this trap today:** ~$0.30 of GPU spend on Lane B + 90 min wall time discovering bug 2 after fixing bug 1. Multi-day arc absorbs this; not a session-ending issue.
