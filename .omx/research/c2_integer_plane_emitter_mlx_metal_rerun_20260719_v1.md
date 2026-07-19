# C2 integer-plane emitter — MLX Metal parity rerun + basis A/B data (Task #571 items 1–2)

**UTC:** 2026-07-19
**Host:** M5 Max, Metal GPU present (`Device(gpu, 0)`) — the codex build host had NO Metal device; this closes its `BLOCKED_ENVIRONMENT_NO_METAL` gate.
**Axis:** `[macOS-MLX research-signal]` / `[macOS-CPU advisory, untrained]` · `score_claim=false` · `promotion_authority=false`
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` UNMOVED. No training, dispatch, score, promotion, or sacred-run write occurred.
**Verdict scope:** C2 fixture + six frozen n24 real pairs (IDs 0–5), fresh untrained emitter only. NOT a trained vehicle, byte-closed archive, contest score, or basis winner.

## Stores consulted
- `.omx/research/c2_integer_plane_emitter_build_20260719_codex.md` (build memo; owed the Metal rerun) + `.omx/research/c2_integer_plane_emitter_implementation_spec_20260719.md`.
- `CLAUDE.md` (MLX is NEVER score authority; `MLXTorchParityThresholds` is the charged gate) + `docs/operating_manual_craft_handoff.md`.
- Frozen `gt_n600.npz` SHA `cf8d83…`, SegNet SHA `68956e32…`, PoseNet SHA `0f3a0874…`, VJP manifest SHA `3d1218a5…` (all custody re-validated by the tool, PASS).

## Receipts (append-only, content-SHA)
- Fixture: `c2_integer_plane_emitter_fixture_metal_rerun_20260719_v1.json` — SHA `da91738d63aaf25c45d4e6976c8be9e6ce43ba425ce3b72f892a5fb96ee4083d`.
- n24-advisory (n6 + MLX Metal twin): `c2_integer_plane_emitter_n24_advisory_metal_rerun_20260719_v1.json` — SHA `ebf7459d2d1076a927dcb0dc559d6967c132324b44fc4529f6700b7e86a7204c`.
- Tool: `tools/measure_c2_integer_plane_emitter.py` (unmodified — ran as-built; no production-code edit needed on Metal).

---

## Item 1 — MLX Metal parity rerun

### Fixture (emitter level) — ALL PASS
- **EXACT emitted-byte parity (charged): PASS.** MLX-Metal emitter bytes SHA `ac317fcd…` == NumPy-fp32 SHA `ac317fcd…` (bit-identical; also matches the codex build's recorded byte SHA).
- Saturation-aware STE gradient parity: in-range-nonzero PASS, saturated-zero PASS.
- MLX fixture wall-clock: median **0.115 s/iteration** (setup 0.0025 s; single iteration).
- `status: MEASURED_MLX_METAL_FIXTURE_PARITY_PASS`. U4 singular values reproduce the sealed `[3.1284, 2.1543, 2.0247, 1.7963]`.

### n24-advisory scorer parity (n6, MLXTorchParityThresholds) — split verdict
Emitter bytes through Metal re-verified bit-identical to NumPy per pair (PASS). Scorer parity over 12 manifests (6 pairs × {candidate, reference}):

| Gate | Threshold | Measured worst-case | Verdict |
|---|---|---|---|
| **PoseNet component \|Δ\|** (charged) | ≤ 2e-5 | **3.88e-11** | **PASS** (≈14 orders of margin; 0 pose blockers) |
| PoseNet output \|Δ\| | ≤ 2e-3 | 1.53e-5 | PASS |
| **SegNet argmax-diff pixels** (charged) | == 0 | **up to 3 px/frame** (fraction ~1.5e-5 of 196 608) | **FAIL** |
| SegNet logit \|Δ\| | ≤ 1e-2 | 1.49e-2 | FAIL (marginal) |

`status: MEASURED_MLX_METAL_SCORER_PARITY_FAIL` — 10/12 manifests failed; **all 17 blocker instances are SegNet, zero PoseNet.**

**Precise attribution (a FINDING, not papered over):** the FAIL fires on BOTH the `candidate` (C2-emitted) AND the `reference` (raw source-pair) surfaces. Since the reference surface is source-vs-source and never touches the C2 emitter, the mismatch is a **pure MLX-vs-Torch SegNet numeric drift, independent of the C2 emitter** (whose bytes are bit-identical). The mismatched pixels sit exactly on the codim-1 argmax separatrix: mismatch top-2 margins ~1.1e-4–1.7e-4, flipped by SegNet logit deltas up to ~3.7e-3 (worst-frame logit delta 1.49e-2). This is the known MLX SegNet argmax-boundary sensitivity, consistent with the standing CLAUDE.md rule that **MLX is NEVER a d_seg authority** — the exact factor-2 lattice + CPU-Torch remains the sole authority row. PoseNet on Metal is clean to 2e-5.

### Real seconds/iteration (n6, cpu-threads=1)
| Loop | setup s | total s | median s/pair | p95 s/pair |
|---|---|---|---|---|
| MLX Metal twin | 0.288 | 4.731 | **0.739** | 0.752 |
| CPU-Torch authority | 3.512 | 7.682 | **0.686** | 0.714 |

MLX per-pair iteration is not faster than CPU-Torch here (the exact factor-2 lattice realization stays on CPU by design; MLX only carries the scorer forward), but setup is ~12× cheaper.

---

## Item 2 — Basis A/B data (raw_centered vs sign_fixed_u4_pair_margin)

The tool enforces byte-identity across arms at fixed capacity. Measured (fixture `basis_ab_build_invariant`, independent state objects):

| Field | raw arm | u4 arm |
|---|---|---|
| objective | `raw_centered` | `sign_fixed_u4_pair_margin` |
| emitted-bytes SHA256 | `ac317fcd…` | `ac317fcd…` |
| capacity SHA256 | `8e01e5b5…` | `8e01e5b5…` |
| `emitted_bytes_equal` | **true** | |
| `capacity_equal` | **true** | |

**Per-arm d_seg/d_pose trajectory (n6 hard-oracle, cpu-threads=1).** Because both arms emit byte-identical planes at fixed capacity and the tool's loop applies **no optimization steps** (fresh seeded residual, untrained — as-built), the per-arm trajectories are **IDENTICAL by construction**. The shared trajectory (both arms) is:

| pair | d_seg | d_pose | s/pair |
|---:|---:|---:|---:|
| 0 | 0.0001475016 | 0.0000100852 | 0.718 |
| 1 | 0.0001068115 | 0.0000827029 | 0.677 |
| 2 | 0.0000966390 | 0.0000394644 | 0.675 |
| 3 | 0.0001271566 | 0.0000183378 | 0.688 |
| 4 | 0.0001525879 | 0.0000260173 | 0.684 |
| 5 | 0.0000762939 | 0.0000098116 | 0.702 |

(CPU-Torch authority; reproduces the codex build receipt within ~1e-8 d_pose CPU-nondeterminism jitter. n6 mean d_seg 0.0001178, mean d_pose 3.11e-5.)

**Data-only note for MAIN (verdict is MAIN's):** at fixed capacity + zero training, the raw vs U4/pair-margin basis choice is a *training-objective* selector that is provably byte-inert here (identical bytes ⇒ identical d_seg/d_pose ⇒ ΔS = 0). This measurement therefore does NOT discriminate the two bases; a discriminating verdict requires identical optimization steps under each objective, which the as-built tool does not perform (no training loop). The memo's "residual width sealed at 4 pending the basis verdict" remains OWED — this rerun supplies the parity + byte-identity substrate, not the basis winner.

## Triality / pointer honesty
- **Pointer:** UNMOVED (`0.1910828242 [contest-CPU]`). Means, not ends.
- **DAG:** Metal rerun closes the codex `BLOCKED_ENVIRONMENT_NO_METAL` gate → emitter byte parity PASS, PoseNet parity PASS, SegNet argmax parity FAIL (MLX-vs-Torch boundary drift, emitter-independent), basis A/B byte-inert-untrained.
- **Equation leg:** N/A (no new law; a parity/measurement rerun).
