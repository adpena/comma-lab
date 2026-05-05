# INDEX: Score-aware sidechannel paradigm — landing page (2026-05-04)

This is the master index for a 6-memo research thread documenting the
**score-aware sidechannel paradigm** discovered during the 2026-05-04
PR-comparison audit session. New readers start here.

## TL;DR

The Shannon-floor mechanism for the comma video compression contest is
**small parameter-sparse correction sidechannels trained against the scorer
at compress time**. Six independent implementations exist across PR100-era
public submissions + in-house code, spanning the full granularity spectrum
(per-pair to per-tile). Two of them are **portable to PR106** without
architecture change and are pre-registered as launch-ready lane leads.

## Reading order

| # | Memo | Length | What it covers |
|---:|---|---:|---|
| 0 | (this file) | – | Index + reading order + decision pipeline |
| 1 | `score_aware_sidechannel_paradigm_20260504.md` | 105 | Paradigm definition + 3 initial implementations cross-PR |
| 2 | `pr100_latent_sidecar_porting_proposal_20260504.md` | 160 | Lead lane: PR100 sidecar → PR106 (council-gated) |
| 3 | `codex_metric_yshift_audit_20260504.md` | 93 | Variant #3: per-frame YUV + (dy, dx) pixel translation |
| 4 | `qpose14_seg_tile_actions_paradigm_extension_20260504.md` | 89 | Variant #5: per-tile YUV codebook (maximalist extreme) |
| 5 | `codex_metric_lrl1_audit_20260504.md` | 103 | Variant #6: luma-only low-rank pixel residual |

Plus 3 sister memos that defined what's NOT this paradigm:

| Memo | Conclusion |
|---|---|
| `apogee_intN_single_stream_brotli_DEFERRED_20260504.md` | Codec optimization, ~0.0006 score Δ, BELOW noise — DEFER |
| `pr106_latent_optimization_FALSIFIED_20260504.md` | 3 angles tried, ALL falsified — latent at Shannon floor for given representation |
| `quantizr_archive_layout_confirmation_20260504.md` | Architecture mismatch — Quantizr is Q-FAITHFUL territory, not PR106-stacking |

## Six variants of the paradigm

| # | Implementation | Granularity | Mechanism | Sidechannel size | Status |
|---:|---|---|---|---:|---|
| 1 | PR100 hnerv_lc_v2 | per-pair (1 dim) | latent additive correction | ~1.2 KB | **lane_pr106_latent_sidecar L1** |
| 2 | codex_metric_yshift Y_SAT | per-frame (2 ch) | Y offset + saturation | ~600 B | reserve |
| 3 | codex_metric_yshift Y_SHIFT | per-frame (3 ch) | Y offset + (dy, dx) translation | ~1-2 KB | **lane_pr106_yshift_sidechannel L1** |
| 4 | Lane SJ-KL (in-house) | per-frame (K coefs) | RGB low-rank pixel residual | ~10-30 KB | already implemented |
| 5 | qpose14 seg_tile_actions | per-tile (codebook) | YUV codebook per tile | 24-120 KB | not portable to PR106 (no tiles) |
| 6 | codex_metric LRL1 | per-frame (K coefs) | LUMA low-rank + bilinear upsample | ~3-5 KB | reserve (3rd stack-on candidate) |

## Decision pipeline

The two pre-registered lanes form an explicit STACKED dispatch sequence:

```
TICK 1: Operator approves lane_pr106_latent_sidecar dispatch (~$0.60)
        - Predicted: PR106 + sidecar → 0.20728 (Δ -0.00218)
        - Lands EMPIRICAL gate; informs whether sidechannel paradigm
          actually works on our PR106 base

TICK 2 (only if TICK 1 wins as predicted):
        Operator approves lane_pr106_yshift_sidechannel dispatch (~$0.60)
        - Predicted ORTHOGONAL gain: -0.0005 to -0.0015 score Δ standalone
        - STACKED on top of latent-sidecar: ~-0.003 score Δ total
        - Lands the second representation-extension; first sub-0.205 candidate

TICK 3 (only if TICK 2 wins):
        Reactivate variant #6 (LRL1 luma residual) as 3rd stack-on
        - Or pivot to apogee_intN bit-width axis if the sidechannel
          ladder is exhausted
```

The pipeline is gated on EMPIRICAL evidence at each step. NOT a
multi-lane parallel dispatch — sequential validation prevents wasting
GPU spend on stacking lanes that interact unexpectedly.

## What's NOT in this thread

This paradigm thread covers only **PR106-portable** sidechannel work.
NOT included:
- Q-FAITHFUL clone of Quantizr's full architecture (separate lane family)
- apogee_intN bit-width sweep (already-launch-ready Pareto-frontier work)
- Lane Ω-W-V3 sensitivity-weighted water-filling (already-launch-ready)

For those, see `reference_dispatch_tooling_stack_20260504.md` (in user-memory).

## Cross-refs

- Lane registry: `tools/lane_maturity.py audit | grep pr106_` shows both pre-registered lanes
- Operator briefing (one command, full state): `.venv/bin/python tools/operator_briefing.py`
- Pareto + dashboard + reconciler trio: `tools/apogee_intN_pareto.py`, `tools/score_dashboard.py`, `tools/predicted_vs_actual_reconciler.py`
- Dispatch dry-runs ($0 cost): `tools/dispatch_dryrun_apogee_intN.py`, `tools/dispatch_dryrun_omega_w_v3.py`, `tools/all_lanes_preflight.py`
- PR106 byte-layout deconstruction (foundation): `docs/pr106_byte_layout_deconstruction_20260504.md`

## How this thread evolved

Started this session at 0 understanding of the paradigm — just had the
intuition that "PR106 has 8.5% latent payload, maybe leverage there."
After 4 hours of audit-driven exploration:

- 3 codec-optimization angles on PR106 latents → all FALSIFIED below noise
- Cross-PR comparison (PR100 vs PR105) revealed the sidecar trick
- Audit of 32 sibling submissions found 5 independent implementations
- Pattern recognition: "encode different bits, not same bits better"
- 2 pre-registered lanes ready for council gate + GPU dispatch
- 0 LOC of speculative code; 0 GPU dollars spent

This is what the durable-record discipline produces over many small ticks.
