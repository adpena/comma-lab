---
name: Everything Configurable — No Hardcoded Architectural Choices
description: All architectural decisions must be configurable via profiles for sweeps and optimization. Nothing hardcoded.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Rule

EVERY architectural choice must be configurable via profile config, not hardcoded. This enables systematic sweeps across the entire design space.

**Why:** Hardcoded choices prevent experimentation. The winning configuration is found by sweeping, not by guessing. If a parameter can't be changed from the profile, it can't be swept.

**What must be configurable:**
- Noise mode (deterministic / shared / independent)
- Blend mode (scalar / spatial / none)
- Motion predictor type (depth_aware / learned_cnn / analytical / none)
- Attention type (patch_cross / full_spatial / per_pixel_gate / none)
- All dimension/channel counts
- All loss weights
- All learning rate schedules
- Depth initialization values
- Camera motion estimation method
- Phase transition epochs

**How to apply:** When implementing ANY new module, add its key parameters to the profile config schema. Use `cfg.get("param_name", default)` pattern. Never write `if True:` or magic numbers without a config key.
