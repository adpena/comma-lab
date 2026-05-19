# Codex Routing Directive — TOP-5 Arbitrariness Extinction: Inflate Device Pin Per Archive

**Subagent**: `lane_arbitrariness_extinction_meta_lens_systematic_audit_20260518`
**Value ID**: `inflate_device_fallback_policy_PACT_INFLATE_DEVICE_auto`
**Resolution path**: `formula`
**Predicted ΔS**: [-0.005, -0.001]
**Cost envelope**: $0
**Rank score per dollar**: 5.0

## Bug class

Per CLAUDE.md Catalog #205 + A1 PR Council Round 1 F1/F11: inflate-time device selection is now via `tac.substrates._shared.inflate_runtime.select_inflate_device` (canonical helper, MPS refused). However, the AUTO selection per-archive can yield DIFFERENT floating-point bytes:

- A1 archive `87ec7ca5f2f328a8...` → CPU 0.19285 / CUDA 0.22635 (Δ = **+0.0335**)
- Mechanism: bicubic kernel + clamp/round/uint8 cast non-bit-identical across devices

The submitted archive is currently scored on whichever device the contest runner picks. If our archive's training device was CPU but contest runs on CUDA (or vice versa), we lose 0.03+ score points to numerical drift alone.

## 5-path analysis

1. **experimental** — sweep per-archive device-pin policies. $0 (per-archive 2x eval).
2. **analytical_solve** — N/A (device drift is empirical).
3. **formula** [RECOMMENDED] — per-submission archive metadata declares its TRAINING device + inflate-time PIN to that device. The contest evaluator respects `PACT_INFLATE_DEVICE_PIN` env var.
4. **learned** — N/A (no learning needed).
5. **self_alien_tech** — N/A.

## Concrete next step ($0)

Extend `submissions/*/inflate.py` to:

1. Read `archive_metadata.json` (NEW canonical sub-file in archive.zip)
2. Field `training_device_pin: "cpu" | "cuda"` declares the device the archive was OPTIMIZED for
3. `select_inflate_device(pin=training_device_pin)` honors the pin
4. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": evaluate both axes BEFORE submission; submit the archive whose training device matches its WORST eval axis (so the worse axis is at-most-equal to its non-pin counterpart)

## Sister coordination

- Catalog #205 (inflate device fork) already canonical; this extends it with PIN metadata
- Catalog #316 (frontier scan) ensures both axes ranked
- Affects every submission archive going forward; particularly PR101 fec6 + PR106 format0d

## Exit criteria

1. `archive_metadata.json` schema declares `training_device_pin`
2. `select_inflate_device(pin=...)` parameter wired
3. Empirical: A1 anchor re-eval with pin shows reproducibility
