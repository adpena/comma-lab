# DEFER — fec6 + PD-V2 pose codec (Ext 1)

**Date:** 2026-05-17
**Lane:** `lane_fec6_stacking_wave_5_grammar_extensions_20260517`
**Verdict:** DEFERRED-pending-separate-pose-slot-in-successor-substrate (NOT KILLED per CLAUDE.md "Forbidden premature KILL")
**Substrate compatibility:** verified incompatible (PR101 has no separate pose slot)

## Why this is deferred

Per the canonical premise verifier at `.omx/tmp/fec6_stacking_wave_premise_verifier.txt` PV-1 + PV-2:

- fec6 wraps the PR101 archive as opaque bytes via `OUTER_MAGIC FP11 | source_len | source_payload | selector_len | selector_payload`.
- PR101's archive is a single ZIP member `x` (178,158 bytes) containing a monolithic packet `decoder_blob | latent_blob | sidecar_blob` at fixed offsets per `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py` (DECODER_BLOB_LEN=162_164, LATENT_BLOB_LEN=15_387, sidecar~707).
- **There is NO separate `poses.pt` slot in either fec6 or PR101.** Pose information is implicit in PR101's HNeRV decoder + latents (the HNeRV per-frame embedding plus the decoder weights jointly produce the rendered RGB; from rendered RGB the contest scorer's PoseNet extracts pose distortion).

Therefore the PD-V2 pose-delta codec (`src/tac/pose_delta_codec_v2.py`) cannot replace a non-existent slot in fec6.

## Substrate-compatibility evidence

Per CLAUDE.md Catalog #301 (`check_kill_memos_have_substrate_compatibility_evidence`), kill memos must enumerate the substrate compatibility considered. This is NOT a kill memo (this is DEFERRED-pending-research per CLAUDE.md "Forbidden premature KILL"), but the same compatibility evidence applies:

- **Compatible substrates for PD-V2**: substrates that ship a `poses.pt` artifact alongside the renderer (per `src/tac/submission_archive.py::load_optimized_poses`). Examples: any substrate inheriting the PR95 baseline's `(renderer.bin + masks.mkv + poses.pt)` 3-file archive shape.
- **Incompatible substrates for PD-V2**: substrates wrapping monolithic-packet base archives like PR101, where the pose info is implicit in the renderer + latent jointly. fec6, PR106 LANES r2, and sister wraps over PR101 fall in this class.

## Reactivation criteria

DEFER reverses to LAND when ANY of:

1. **A successor substrate (PR95 Phase 2-4 family per `experiments/train_substrate_pr95_phase_2_4.py` queued at task #608) lands a separate `poses.pt` slot in its archive grammar.** PD-V2 can then drop in directly.

2. **Operator approves a substrate-engineering scope rebuild of the PR101 family with an exposed `poses.pt` slot.** This is large scope: requires retraining the HNeRV decoder with pose-conditioning factored out, re-encoding the latent blob, redesigning the archive grammar. Substrate-engineering, not bolt-on.

3. **An alternative substrate (e.g. PR106 r2's `format0d` latent-correction stream) is demonstrated to be a pose-equivalent encoding surface.** The PR106 format0d primitive corrects PR101 LATENTS at per-pair-and-per-dim granularity. If empirically the optimal corrections concentrate on a small subset of latent dims that map 1:1 to pose dimensions, then format0d-EXTRA (Ext 4 in this wave) IS the de-facto pose codec for the PR101 family and Ext 1 is subsumed.

## Per CLAUDE.md "Forbidden premature KILL without research exhaustion"

Default verdict for one-config failure is **DEFERRED-pending-research**, not KILLED. The PD-V2 codec is well-validated for substrates with a separate `poses.pt` slot (per `src/tac/tests/test_pose_delta_codec_v2.py`). The deferral is about ARCHIVE-COMPATIBILITY with the PR101 family specifically, not about the PD-V2 method itself.

## Alternative probe methodologies per Catalog #308

Per the alternative-probe enumeration discipline (N≥3 required):

1. **Alternative 1**: SPLIT-VERDICT — apply PD-V2 to a different substrate's `poses.pt` slot (e.g. the PR95 Phase 2 successor when it lands).

2. **Alternative 2**: REQUEST-REINVESTIGATION — explore whether PR106 r2's `format0d` correction stream is the de-facto pose codec for the PR101 family (see Reactivation Criteria #3).

3. **Alternative 3**: SUBSTRATE-PIVOT — if no PR101-family pose-slot becomes available, ratify PD-V2 as falsified-for-PR101-family and reactivate ONLY on a future substrate-engineering scope rebuild.

## Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW"

**Operating-within assumption**: this DEFER memo assumes that the operator's framing of "PD-V2 applied to fec6's poses.pt" was based on a stale inventory entry (`tac.codec.pd_v2` per `meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md` §3.B) that hadn't been verified against the actual fec6 + PR101 grammar.

**Assumption-Adversary verdict**: HARD-EARNED — the inventory entry IS the source of the stale assumption; the premise-verification step caught it; the deferral is the correct response per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode (don't force-fit a primitive to a slot that doesn't exist).

## Predicted paradigm-vs-implementation classification per Catalog #307

**IMPLEMENTATION-CARGO-CULT** — the PD-V2 paradigm (entropy-coded pose deltas with V1 fallback) is intact and demonstrably works on substrates with a separate `poses.pt` slot. The specific implementation-mapping "apply PD-V2 to fec6" cargo-culted the assumption that fec6 has a `poses.pt` slot from the inventory framing. The paradigm reactivates the moment a compatible substrate ships.

## Cross-references

- `src/tac/pose_delta_codec_v2.py` — PD-V2 canonical implementation
- `src/tac/submission_archive.py::load_optimized_poses` — the consumer that dispatches V2 vs V1 vs raw
- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py` — PR101 archive grammar
- `tools/build_pr101_frame_exploit_selector_packet.py` — fec6 builder (wraps PR101 opaquely)
- `experiments/train_substrate_pr95_phase_2_4.py` — task #608 PR95 Phase 2-4 substrate (future home for PD-V2 application)
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE"
- Catalog #229 premise verification
- Catalog #301 kill-memo substrate-compatibility evidence
- Catalog #307 paradigm-vs-implementation classification
- Catalog #308 alternative-probe-methodologies enumeration
