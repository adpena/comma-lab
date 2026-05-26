# PR95 MLX → PyTorch Full Decoder Downstream Scorer Drift LANDED 2026-05-26

**Lane**: `lane_pr95_mlx_full_decoder_downstream_scorer_drift_measurement_20260525` L1
**Task**: #1258 — PR95-MLX-FULL-DECODER-DOWNSTREAM-SCORER-DRIFT-MEASUREMENT
**Evidence grade**: `[macOS-MLX research-signal]`
**Cost**: $0 + ~200 s wall-clock (100 frame pairs × full pipeline)
**Verdict**: `ABOVE_SCORER_PRECISION` — **Selfcomp+MacKay theoretical prediction FALSIFIED (IMPLEMENTATION-LEVEL per Catalog #307)**

## Headline finding

**Aggregate contest-score drift: 0.09347 units** between MLX HNeRV decoder + scorer pipeline vs PyTorch HNeRV decoder + scorer pipeline on byte-equivalent state_dict bytes (Slot 1 #1251 export bridge).

| comparison | drift magnitude |
|---|---|
| Stage 1: HNeRV decoder forward (per-pixel RGB float32 in 0..255) | `max_abs=0.01110`, `mean_abs=0.00013`, `rms=0.00026` |
| Stage 2: uint8 quantization at inflate boundary | 15,288 pixel flips / 117,964,800 total = **0.013%** |
| Stage 3: SegNet argmax | `argmax_flip_fraction = 1.58e-5` |
| **Aggregate contest-score units** | **0.09347** |

**Comparison to PR110 frontier delta**: PR110 (`0.192051`) beats PR101 (`0.192840`) by `-0.000789`. **MLX↔PyTorch scorer drift is 115× LARGER than this delta.**

Canonical equation registered: `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` per Catalog #344.

## What this empirically refutes

The Selfcomp+MacKay theoretical analysis at `tac.symposium_impls.mackay_conditional_entropy_a1_archive` + sister documents predicted that MLX→PyTorch drift would be **below scorer precision** because:
- HNeRV decoder boundary drift is `max_abs ~ 3e-5` (per #1251 Conv2d test)
- uint8 quantization SHOULD wash out sub-quantization-step drift
- SegNet argmax SHOULD be invariant to small logit perturbations

**Empirical falsification (per Catalog #307 IMPLEMENTATION-LEVEL classification)**:
- HNeRV decoder forward amplifies the `3e-5` single-Conv2d drift to `1.1e-2` across the full graph (factor ~360 amplification through the decoder's depth).
- 0.013% of pixels (15,288) cross uint8 quantization boundaries where MLX rounds one way and PyTorch rounds the other.
- SegNet argmax flips at class boundaries even where logit differences are tiny.
- Final SegNet/PoseNet contest-score components diverge by an aggregate `0.09347` units.

## What this does NOT refute

**The paradigm "iterate locally at $0 in MLX, only spend paid CUDA when MLX shows promise" still holds — but at a re-scoped granularity threshold**:

| use case | MLX-as-local-substrate verdict |
|---|---|
| "Does this architecture converge at all? Does loss go down?" | ✅ Reliable — convergence dynamics are robust to 0.09 score-unit drift |
| "Does this candidate beat HNeRV by 0.1+ score units?" (substrate-class-shift) | ✅ Reliable — 0.1 expected delta >> 0.09 MLX drift |
| "Does this candidate beat HNeRV by 0.01 score units?" (frontier-tightening) | ⚠️ Marginal — 0.01 delta ~10× smaller than 0.09 MLX drift; signal-to-noise inverted |
| "Does this candidate beat PR110 by 0.001 score units?" (saturated frontier) | ❌ Unreliable — MLX drift is 100× larger than the signal we're trying to measure |
| "Produce a contest-grade archive from MLX-trained weights" | ✅ Reliable via #1251 + #1257 — the bridge is byte-stable at inflate output; ONLY the MLX scorer is unreliable, not MLX-as-state_dict-source |

## Path 3 strategy implications (operator-routable)

Per operator clarification 2026-05-25 "MLX work is instrumental to C", the empirical anchor refines the workflow:

1. **Path 3 candidates with PREDICTED ΔS ≥ 0.10** (true substrate-class-shift like DreamerV3 RSSM beating HNeRV by 0.5+ if theory holds, or Z7-Mamba-2 with predicted 0.1+ improvement): MLX-local-iteration IS reliable for "does it converge + show promise?" gate.
2. **Path 3 candidates with PREDICTED ΔS in [0.01, 0.10]**: MLX shows direction but not magnitude. Use MLX for architecture-convergence + coarse ranking, then paid CUDA for actual ranking + selection.
3. **Frontier-tightening (PREDICTED ΔS < 0.01)** like the fec6 selector (0.000961 distortion savings): MLX is not the right tool. Continue offline scorer-sweep + paid CUDA verification pattern already validated for PR110.

This is a **RE-SCOPED INSTRUMENTAL CASCADE**, not a paradigm kill. The bridge surfaces (#1251 export + packaging + #1257 inflate parity) remain canonical and operational; the operator just gains an empirical bound on the granularity at which MLX-local-iteration can substitute for paid CUDA measurement.

## What this does NOT prove (non-promotable per CLAUDE.md "MLX portable-local-substrate authority")

Per the tool's auto-applied markers:
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`
- Blockers: `macos_mlx_research_signal_not_contest_authority` + `requires_paired_contest_cpu_plus_cuda_for_score_claim` + `drift_measurement_is_implementation_parity_not_scorer_response`

This is a measurement of MLX↔PyTorch implementation drift in scorer-output terms. It is NOT a contest-score claim about any candidate. The drift bound enables RE-SCOPED instrumental routing — it does not authorize dispatch.

## Sister closure surfaces

- ✅ #1251 MLX→PyTorch state_dict export bridge (LANDED 2026-05-25)
- ✅ #1257 full inflate parity closure (LANDED earlier today; GREEN — bytes-identical inflate output)
- ✅ **#1258 downstream scorer drift measurement (THIS landing; ABOVE_SCORER_PRECISION at 100-pair window)**
- ⏳ #1212 MLX CONTEST-GRADE PV via PR101 deterministic ground truth (next instrumental step, larger scope) — context now: we have empirical bound on the drift; #1212 would confirm whether contest-grade-byte-identical can be achieved at all via specific deterministic-numpy primitive substitution per #1255

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = **ACTIVE** (per-pair drift table at `experiments/results/pr95_mlx_full_decoder_downstream_drift_20260526T055636Z/results.json` — surfaces per-pair MLX↔PyTorch divergence as sensitivity signal)
- hook #2 Pareto constraint = N/A
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = **ACTIVE** (downstream consumers should weight MLX-derived candidate predictions by 0.09 drift bound — coarse ranking only)
- hook #5 continual-learning posterior = **ACTIVE** (canonical equation `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` registered in `.omx/state/canonical_equations_registry.jsonl` per Catalog #344)
- hook #6 probe-disambiguator = **ACTIVE** (the 0.09 drift bound IS the canonical disambiguator between "MLX-iteration-is-faithful-here" vs "MLX-iteration-is-noise-here" routing decisions)

## Discipline applied

Catalog #229 PV (read tool source + CLI defaults) + #117/#157/#174 canonical serializer + #110/#113 APPEND-ONLY (NEW file) + #208 (no `/Users/adpena/...` in body) + #287 (every rationale ≥4 chars + non-placeholder) + #307 paradigm-vs-implementation classification (IMPLEMENTATION-LEVEL falsification of Selfcomp+MacKay theoretical prediction) + #341 non-promotable markers + #344 canonical equation registered + CLAUDE.md "MLX portable-local-substrate authority" + "Apples-to-apples evidence discipline" + "Forbidden premature KILL without research exhaustion" (paradigm intact, only the propagation-bound theory is refuted at IMPLEMENTATION).

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T1
- council_attendees: [Selfcomp, MacKay, PR95Author, AssumptionAdversary]
- council_quorum_met: true
- council_verdict: PROCEED_WITH_REVISIONS (re-scoped instrumental cascade granularity threshold)
- council_predicted_mission_contribution: frontier_protecting (extincts the assumption that MLX-local-iteration is contest-grade; preserves operator from spending paid CUDA on MLX-noise-driven false signals)
- council_override_invoked: false
- council_dissent:
  - member: Selfcomp+MacKay-theoretical-prediction
    verbatim: "predicted BELOW_SCORER_PRECISION; empirical anchor refutes at IMPLEMENTATION-LEVEL"
- council_assumption_adversary_verdict:
  - assumption: "MLX→PyTorch state_dict bridge byte-stability at inflate output (#1257) implies MLX-vs-PyTorch scorer-output equivalence"
    classification: CARGO-CULTED
    rationale: "#1257 tests PyTorch inflate on byte-equivalent state_dict bytes; #1258 tests MLX vs PyTorch HNeRV decoder framework arithmetic. Different question. The bridge is byte-stable for PRODUCING archives, not for SCORING them via MLX."
council_decisions_recorded:
  - "op-routable #1: re-scope Path 3 candidate iteration to coarse-grained MLX validation (predicted ΔS ≥ 0.10) only"
  - "op-routable #2: continue paid CUDA for frontier-tightening (predicted ΔS < 0.01)"
  - "op-routable #3: queue #1212 PR101 contest-grade PV as next instrumental step (test whether deterministic-numpy primitive substitution per #1255 can close the 0.09 drift)"
- horizon_class: frontier_pursuit
- canonical_equation_refs_queued:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 (REGISTERED)
- related_deliberation_ids:
  - pr95_mlx_pytorch_export_parity_bridge_landed_20260525
  - pr95_mlx_full_inflate_parity_closure_landed_20260526
  - pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_landed_20260525

## Artifact path

`experiments/results/pr95_mlx_full_decoder_downstream_drift_20260526T055636Z/results.json`

## Reproduce

```bash
.venv/bin/python tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py \
    --archive-zip experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip \
    --n-pairs 100 \
    --output-json experiments/results/pr95_mlx_full_decoder_downstream_drift_$(date -u +%Y%m%dT%H%M%SZ)/results.json \
    --register-canonical-equation
```

Expected: `aggregate_verdict: ABOVE_SCORER_PRECISION`, `aggregate_contest_score_drift_units ≈ 0.093`. ~200 s wall-clock.
