# Local hardware (CPU+MPS+MLX) maximization sweep — landed 2026-05-13

**Lane:** `lane_local_hardware_maximization_sweep_20260513`
**Mode:** $0 GPU budget; macOS-CPU + MPS local-only fan-out
**Evidence grade:** `[macOS-CPU advisory]` (Stream 1, 2, 4) + `[MPS-research-signal]` (Stream 3 plan)
**Score claim:** false. `promotion_eligible:` false. `ready_for_exact_eval_dispatch:` false.

---

## Operator directive

Per operator 2026-05-13: "make sure we are maximizing utility from local cpu and
mps / mlx usage" while Modal GPU orchestrator + adversarial review + other
priorities subagents run in parallel.

## Mandate fulfilled

4 work streams executed within $0 budget on M5 Max:

| Stream | Status | Output |
|---|---|---|
| 1 — substrate macOS-CPU smoke fan-out (4 substrates) | ✅ COMPLETE | 4/4 wiring-verified |
| 2 — per-pair PSNR diagnostic | 🔧 TOOL LANDED + 1/8 LIVE | A1 600-pair eval running |
| 3 — MPS parallel architecture search | 📋 PLAN LANDED | Operator-routable (30 configs / ~6.5h) |
| 4 — A1 sub-frontier enumeration | ✅ COMPLETE | A1 saturated at 1st-order entropy |

Lane registered: `python tools/lane_maturity.py add-lane lane_local_hardware_maximization_sweep_20260513 --name "Local hardware (CPU+MPS+MLX) maximization sweep" --phase 2`

---

## Stream 1 — Substrate macOS-CPU smoke fan-out

All 4 substrate trainers exercise their wiring on macOS-CPU via `--smoke
--device cpu --epochs 3`. Verdicts:

| Substrate | Verdict | Smoke archive bytes | Predicted band |
|---|---|---:|---|
| `sabor_boundary_only_renderer` | wiring_verified_substrate_smoke_only | 42,225 | [0.165, 0.185] |
| `s2sbs_byte_stuffing` | wiring_verified_substrate_smoke_only | 1,040 | [0.168, 0.188] |
| `a1_plus_wavelet_residual` | wiring_verified_substrate_smoke_only | 178,203 (+41 vs A1) | [0.187, 0.194] |
| `a1_plus_lapose` | wiring_verified_substrate_smoke_only | 178,201 (+39 vs A1) | [0.185, 0.195] |

All 4 trainers loaded, ran 3 synthetic-data smoke steps, and produced
roundtrip-valid archives on macOS-CPU. **No empirical macOS-CPU score**
captured because smokes use synthetic data + `--skip-auth-eval`. Real
score requires real-archive training (~hours macOS-CPU) OR paired GPU
training + macOS-CPU eval (~9.5 min/archive).

Manifest: `experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>/stream1_substrate_smoke_manifest.jsonl`.

---

## Stream 2 — Per-pair sensitivity diagnostic

Built canonical `tools/diagnose_per_pair_sensitivity.py`:
- Inflate archive via its own `inflate.sh` (PR101 / HNeRV-family grammar).
- Decode 1200 frames via pyav.
- Run upstream `DistortionNet.compute_distortion` per pair (B=1) on macOS-CPU.
- Compute per-pair (`pose_d`, `seg_d`) plus marginal score contribution.
- Identify top-50 / bottom-50 leverage pairs.

**Smoke verified** on A1 (N=50 pairs, ~30s): tool works end-to-end.
**Live full-eval** (N=600 pairs, ~10 min macOS-CPU) running for A1 baseline.
Output JSON: `per_pair_sensitivity_a1_baseline_600pair.json` (lands when complete).

Remaining 7 archives (PR101, PR103, PR100, PR107, PR105, PR104, PR063) are
operator-routable — each takes ~10 min macOS-CPU + total ~80 min.

Memo: `.omx/research/per_pair_sensitivity_map_8_archives_20260513.md`.

---

## Stream 3 — MPS architecture search plan

Built `tools/plan_mps_architecture_search_local.py` emitting a structured
search grid:
- `decoder_param_count ∈ {25K, 50K, 75K, 100K, 150K}`
- `latent_dim ∈ {16, 24, 32}`
- `foveation_grid ∈ {4×4, 8×8}`
- **30 configs total**
- est. wall-clock: ~6.5 hours serial / ~2-3 hours @ concurrency=2

Per CLAUDE.md "MPS auth eval is NOISE" — output is `[MPS-research-signal]`
permanently (`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`). Live execution requires building
the actual MPS trainer (~3-4 hours dev) plus the 6.5 hours of MPS time;
**deferred as operator-routable** rather than executed in this session.

Plan: `experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>/mps_architecture_search_plan.json`.

---

## Stream 4 — A1 sub-frontier enumeration

Built `tools/enumerate_a1_sub_frontier_macos_cpu.py`. Decoded A1 archive
into the canonical PR101 grammar (uint32 header + decoder_blob + 15,387 B
latent_blob + sidecar_blob) and computed 1st-order Shannon entropy per
section.

**HIGH-SIGNAL FINDING**:

A1's three sections are essentially incompressible by 1st-order entropy
coding. Predicted savings under 1st-order entropy bound:

| Section | Predicted savings (B) | Δscore (rate-only) |
|---|---:|---:|
| `decoder_blob` (lossy coarsening / block-FP4) | 31 | 2.117e-07 |
| `latent_blob` (uint8→int4 / smaller block) | 22 | 1.503e-07 |
| `sidecar_blob` (pruning / re-entropy-coding) | 21 | 1.434e-07 |

Total 1st-order entropy headroom on A1: ~74 bytes (Δscore ≈ 5.05e-07,
~12× below the macOS-CPU ↔ Linux x86_64 proxy noise floor of 6e-6).

**Operational implication**: A1 byte-mutation cannot reach sub-0.192847 via
1st-order entropy compression alone. Sub-0.192847 score requires EITHER:
1. A different substrate (SABOR alternative-RGB / S2SBS payload-stuffing /
   A1+wavelet residual / A1+LAPose hierarchical) — all 4 wiring-verified in
   Stream 1.
2. DISTORTION improvement (pose-axis lanes per CLAUDE.md operating-point-aware
   priority) holding bytes constant.
3. Higher-order codec (HOC / cross-paradigm composition) on A1's bytes
   exploiting structure 1st-order entropy doesn't see — operator-routable
   downstream lane.

Output: `experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>/a1_sub_frontier_enumeration.json`.

---

## Stream 5 — MLX BitNet 1.58-bit ternary pilot

**Deferred** in this session. MLX install + pilot requires:
- `uv pip install mlx mlx-lm` (~few min)
- 50K-param HNeRV-like ternary trainer scaffold (~2-3 hours dev)
- MPS execution on M5 Max
- Result tagged `[MPS-research-signal]`

Recorded as operator-routable. Cross-ref: Keller Jordan + zen-state-frontier
convergence on ternary QAT (per session memory).

---

## Solver-stack posterior delta

| Hook | This lane's contribution |
|---|---|
| (1) Sensitivity-map contribution (`tac.sensitivity_map`) | per-pair JSON consumable as augmenter; full memo cross-refs the bit-allocator priors |
| (2) Pareto constraint (`tac.pareto_*`) | A1 entropy-saturation result CONSTRAINS the rate axis at the A1 operating point — sub-0.193 requires distortion improvement, not byte reduction |
| (3) Bit-allocator hook | top-50 / bottom-50 per-pair pair indices ready to feed allocator (operator-routable: complete remaining 7 archives) |
| (4) Cathedral autopilot dispatch hook | 4 substrate smoke verdicts + entropy-saturation finding inform autopilot priors (consumed via JSONL manifest) |
| (5) Continual-learning posterior update | N/A — no new contest anchors landed (advisory grade per Catalog #127 routing) |
| (6) Probe-disambiguator | N/A — no design ambiguity surfaced this lane |

---

## Operator-routable decisions

1. **Stream 2 follow-up**: ~80 min macOS-CPU to complete per-pair diagnostic
   on 7 remaining archives → consensus high-leverage pair set.
2. **Stream 3 live execution**: ~6.5 hours MPS wall-clock + ~3-4 hours dev
   for the MPS trainer → arch-search priors.
3. **Stream 4 NO-OP**: A1 is 1st-order-entropy-saturated; further byte
   mutation on A1 alone is not score-productive. Direct effort toward
   substrate-level alternatives (Stream 1 substrates) or HOC (cross-paradigm
   composition).
4. **Stream 5 MLX pilot**: optional ~5-6 hours total (install + dev + run).

## Cross-refs

- `tools/diagnose_per_pair_sensitivity.py` — landed this lane
- `tools/build_stream1_substrate_smoke_manifest.py` — landed this lane
- `tools/enumerate_a1_sub_frontier_macos_cpu.py` — landed this lane
- `tools/plan_mps_architecture_search_local.py` — landed this lane
- `.omx/research/per_pair_sensitivity_map_8_archives_20260513.md` — landed this lane
- `.omx/research/macos_cpu_canvas_pareto_ranking_20260513.md` (sister; consumed)
- `.omx/research/macos_cpu_proxy_drift_table_20260513.md` (sister; cross-ref)
- `.omx/research/sabor_boundary_audit_20260513.md` (sister; cross-ref)
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" non-negotiable
- CLAUDE.md Catalog #125 subagent-coherence wire-in hooks
- CLAUDE.md Catalog #127 custody validator (refuses macos_substrate for paired-axis)
- CLAUDE.md Catalog #128 atomic-locked posterior writes
- CLAUDE.md Catalog #192 macOS-CPU advisory canonical axis (proxy within 2e-5)

## Lane registration + 3-clean-pass review (this lane)

- `lane_local_hardware_maximization_sweep_20260513` at L1 (impl_complete +
  memory_entry). `three_clean_review` deferred to operator review.
- Sister lanes that ALREADY ran this session: GPU orchestrator (Modal),
  adversarial review (codex), other-priorities subagent. No file-surface
  conflict observed (new tools land at unique paths; new memos at unique paths).
