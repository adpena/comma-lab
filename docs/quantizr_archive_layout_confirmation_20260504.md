# Quantizr (0.33 leader) archive layout confirmation (2026-05-04)

## Layout discovery

Audited `experiments/results/public_pr100_intake_20260504_codex/source/submissions/quantizr/inflate.py:247-321` (the `main()` function). Quantizr's archive is **fundamentally different** from PR106 (HNeRV-family) — it's THREE separate brotli-compressed files, NOT a single 0.bin:

```
Quantizr archive (293KB total per CLAUDE.md intelligence):
  model.pt.br   ~64 KB  — brotli-compressed FP4-quantized state dict
  mask.obu.br  ~150 KB  — brotli-compressed AV1 OBU container (mask video)
  pose.npy.br   ~80 KB  — brotli-compressed numpy pose array
```

vs PR106 (HNeRV-family, 186KB):
```
  0.bin  186 KB  — single file: brotli decoder + brotli latents
```

## Why this matters for our roadmap

Quantizr is a **fundamentally different architecture** (not just a different encoding). Their inflate runs a `JointFrameGenerator` from `(mask_video, pose_vectors)` inputs — it does NOT use HNeRV-family latents. The mechanisms:

- **SegNet-mimicking encoder**: produces masks at compress time, stored as AV1 monochrome video
- **Explicit pose vectors**: stored as numpy array, brotli-compressed
- **JointFrameGenerator decoder**: takes (mask, pose) → renders frame pairs
- **FP4 weight quantization**: 88K params × 4 bits = ~44KB raw, brotli to ~64KB

Implications for our PR106-stacking lanes:
- **NOT a candidate for stacking on PR106** — PR106 has no mask channel, no pose vectors. Architecture mismatch.
- **IS the target of our Q-FAITHFUL clone effort** — separately tracked lane that aims to reproduce this paradigm in-house.
- **Quantizr ≠ apogee_intN** — apogee_intN works on PR106's 28-tensor HNeRV decoder; Quantizr's model.pt.br is JointFrameGenerator weights (different schema entirely).

## What this confirms

CLAUDE.md "Quantizr intelligence — verified competitive data (2026-04-21)" already documents:
- Archive 293KB (✓ confirmed)
- 88K param FP4 model (✓ confirmed via inflate code)
- AV1 mask encoding (✓ confirmed via load_encoded_mask_video)
- 600 odd-frame mask subset (✓ confirmed via "1 mask per generated pair")
- frame1 warped from frame2 (✓ implicit in JointFrameGenerator output)

The byte-level audit ADDS:
- Confirmed three-file split (not single 0.bin): model.pt.br + mask.obu.br + pose.npy.br
- Each file independently brotli'd (NOT one giant brotli stream)
- AV1 OBU format (not MKV container) for mask video

## Decision: NO new lane from this audit

The Quantizr layout doesn't translate to a PR106-stacking opportunity. The
existing Q-FAITHFUL clone lane (separately tracked, independently registered)
is the correct path for this paradigm. CLAUDE.md memory entries already
document Q-FAITHFUL state (e.g., `feedback_q_faithful_no_recoverable_checkpoint_history_20260501`).

This audit is **defensive validation**: the byte-level inspection confirms
the CLAUDE.md intelligence is accurate, no surprises, no missed angles.

## Cross-refs

- Quantizr inflate source: `experiments/results/public_pr100_intake_20260504_codex/source/submissions/quantizr/inflate.py`
- CLAUDE.md "Quantizr intelligence" section: cumulative reference
- Q-FAITHFUL clone lane: `feedback_q_faithful_*` memory entries
- Score-aware sidechannel paradigm (the cross-PR pattern audit): `docs/score_aware_sidechannel_paradigm_20260504.md`
- PR-comparison framework (where this audit fits): `docs/pr106_byte_layout_deconstruction_20260504.md` + sister memos
