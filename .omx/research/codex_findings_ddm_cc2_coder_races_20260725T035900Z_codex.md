---
title: DDM CC2 coder races adversarial findings
utc: 2026-07-25T03:59:00Z
lane_id: ddm_cc2_coder_races
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Verdict

`COOL_CHIC_V5_TERMINAL_PROXY` wins the bounded Race 2 instance on the same landed J8F theta and exact counted PC1 wrapper. Race 3 identifies a lossless `-3422 B` mixed-coder price-table opportunity, but it is **not yet a receiver-closed archive**. Pointer `0.1910828242 [contest-CPU]` is unchanged.

# Race 2 — exact receiver/R/frozen-scorer n600

All arms compile through the real W_joint receiver and use the same active zero-effect 40-byte PC1 packet. Each full composition is `139538 B`.

| Arm | d_seg | d_pose | delta bytes | delta d_seg | delta d_pose | delta advisory action |
|---|---:|---:|---:|---:|---:|---:|
| `CAMERA_Q8_EXACT` | 0.0702156745062934 | 36.37587755493872 | 0 | 0 | 0 | 0 |
| `C3_ORIGINAL_TERMINAL_PROXY` | 0.0702156745062934 | 36.37600106210136 | 0 | 0 | +0.00012350716264108996 | +0.00003237837440650537 |
| `COOL_CHIC_V5_TERMINAL_PROXY` | 0.0702132076687283 | 36.369476781438415 | 0 | -0.0000024668375651071273 | -0.006400773500303103 | -0.001924772136713493 |

Epistemic labels:

- **MEASURED:** Q8 is the landed J8F n600 result reused only after exact parent SHA equality. C3 and v5 are fresh `camera -> uint8/R -> frozen SegNet/PoseNet` n600 passes.
- **DERIVED:** component action deltas are recomputed from `100*delta_d_seg + delta sqrt(10*d_pose) + 25*delta_bytes/37545489`.
- **NOT CLAIMED:** C3/v5 are terminal single-pass source-schedule proxies. They were not retrained. One seeded proxy instance neither promotes v5 nor kills C3.

# Race 3 — exact five-arm per-stream pricing

The exact Q8+PC1 composition expands to 27 physical counted leaves plus `5292 B` of recursive ZIP overhead. Every leaf races:

1. current raw bytes;
2. fully counted static tiny ARM/IFCE;
3. G4 decoder-derived causal context;
4. depth-8 Willems CTW;
5. Bellard-class online Bayesian KT mixing.

All 135 frame instances parse back exactly. Video-derived static ARM parameters and all switching headers are counted. G4/CTW/mixing carry zero model bytes because state is derived only from the already-decoded prefix.

The independently selected menu prices `134246 -> 130824` leaf bytes and `139538 -> 136116` total estimated archive bytes: `-3422 B`, rate-action delta `-0.0022785693375840703`, with zero distortion by lossless identity. Winners: raw `19`, G4 `1`, Bellard-class mixing `7`; ARM/IFCE and CTW `0`.

The 40-byte PC1 pose home and 29819-byte G1 stream both stay raw. The largest state-stream saving is the 1233-byte Q8 receiver program, priced at 649 bytes by G4 (`-584 B`). Larger savings occur on counted JSON manifests.

**Hard boundary:** this is a typed c1/costate SENSE price table, not a shippable archive. The mixed context-frame interpreter has not been integrated into inflate/receiver composition, so `-3422 B` is **DERIVED** at archive level even though each frame byte count and parse-back is **MEASURED**. MAIN must review before any receiver integration or candidate claim.

# Apparatus changes

- Extended the existing MS7 coder harness; no parallel race framework.
- Added strict canonical frames and decoders for G4, Willems CTW, and Bellard-class mixing.
- Added recursive counted-leaf inventory, explicit model/header accounting, c1 waterfill order, and costate SENSE rows.
- Added a typed SSD-only runner with atomic per-stage checkpoints, exact source hashes, storage waterfall, and byte-identical resume replay.

# Next executable edge

MAIN review may admit only the eight negative-delta leaf rows into a receiver integration change. That integration must preserve member identity, install the generic decoder in free interpreter code, reconstruct an exact mixed archive, and then repeat exact parse-back before any score evaluation. Separately, v5 requires multi-seed or actual schedule retraining evidence before adoption.

# Verification

- Focused tests: `49 passed, 1 skipped` (optional dependency).
- Randomized coder audit: 57 exact deterministic round-trips and 114 malformed-frame rejections.
- Ruff check: clean.
- Formatting check: clean after formatter.
- Resume replay: `1.0 s`; full receipt remained byte-identical at SHA-256 `f9432959d9c8711276379ef681f5b6985157f49bdd4b2f4a401bfb35ce737ec1`.
- Two independent clean review passes per modified Python file: correctness/parse-back, then math/provenance/resume custody.
- Full receipt: `/Volumes/VertigoDataTier/pact/experiments/results/ddm_cc2_coder_races_20260725T030606Z/ddm_cc2_coder_races_receipt.json`.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; v7.5 operating contract; top project MEMORY entries; latest Codex findings/session summary, council and design memos; canonical frontier, lane, subagent, cost-band, continual-learning, task-status, probe, and dispatch surfaces; live arm/broadcast inbox through `2026-07-24T23:09:25Z`; J8F Step-4 receipt/checkpoint; PC1 packet/receipt; MS7 harness/receipt; SHA-bound C3 and Cool-Chic source harvest; exact n600 target cache; frozen upstream scorer.
