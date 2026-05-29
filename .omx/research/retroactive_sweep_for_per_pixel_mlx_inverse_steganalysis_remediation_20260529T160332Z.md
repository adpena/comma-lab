# Retroactive sweep for per-pixel MLX inverse-steganalysis remediation (Slot EEE)

Per Catalog #348 retroactive sweep discipline + canonical 4-field contract.

## 1. Bug-class symptom signature

Inverse-steganalysis L0 SCAFFOLDS that:
- Implement a per-pair scalar aggregation of a paper-canonical per-pixel cost matrix
- Run cost-function smokes on synthetic 32x32-96x64 random noise inputs
- Claim canonical paper fidelity in docstrings while the implementation is
  simplified/abstracted (per-pixel → per-pair scalar, Wiener filter → box blur,
  matrix-distance SPAM-delta → cell-counting heuristic, directional wavelet →
  scalar 1/(eps+r), etc.)
- Strategy enums with 4 values where 2-3 of 4 fall through to the canonical
  baseline strategy

The Slot EEE audit empirically classified 5 of 7 L0 SCAFFOLDs from today's
session as PARTIAL via this pattern and 1 of 7 (Slot RR) as FAKE.

## 2. Pre-fix window

The 6-axis honesty audit pattern was canonicalized in Slot EEE 2026-05-29.
The remediation pattern (canonical shared helper + per-pixel MLX + real video
frame bind) lands today (2026-05-29 ~16:00 UTC).

## 3. Historical KILL/DEFER/FALSIFY search results

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": NO
historical KILL/FALSIFIED verdicts on HILL/MiPOD/HUGO/UNIWARD/SegNet-waterfill/
pose-axis-null-projection are invalidated by this remediation. The Slot EEE
audit verdicts were classified IMPLEMENTATION-LEVEL per Catalog #307; the
canonical Fridrich-Yousfi inverse-steganalysis PARADIGM remained INTACT for
all 7 audited scaffolds.

Sister related anchors searched:
- OVERNIGHT-EEE 2026-05-21 HILL probe NULL_SIGNAL_DEFER per the existing
  Slot YY HILL package design memo: structurally DISJOINT from THIS
  remediation (OVERNIGHT-EEE used CCC/DDD reciprocal-weight framing on a
  canonical 3x3 L1 kernel; THIS remediation uses canonical Li-Wang Step 2
  L1 kernel default 7 and the canonical per-pixel-cost-as-sparse-K-priority
  interpretation per the canonical Slot YY landing memo)
- Slot QQ phantom-score class anti-pattern (Catalog #321 sister): NOT
  triggered by this remediation (no phantom_provenance composition_alpha;
  no research-sidecar score claim; the bind helper returns macOS-CPU
  advisory smoke results with explicit promotable=False)
- Slot WW HONEST 4→2 reconciliation anti-pattern: NOT triggered (the
  canonical bind helper does not add a paired_cuda_ratification_targets
  list; it routes the operator to the existing canonical PR110 archive
  surfaces)

## 4. Per-finding RE-EVAL-priority assignment

| Finding | Source | Pre-fix verdict | Post-fix status | RE-EVAL priority |
|---------|--------|-----------------|------------------|-------------------|
| Slot YY HILL Axis A PARTIAL (per-pair row-band vs per-pixel paper formulation) | Slot EEE audit | PARTIAL | REMEDIATED via new bind helper `apply_hill_canonical_per_pixel_mlx_to_real_video_frames` | LOW (no paired-CUDA RATIFICATION change required; the bind helper is observability-only) |
| Slot YY HILL Axis C FAIL (synthetic noise smoke) | Slot EEE audit | FAIL | REMEDIATED via canonical shared helper that decodes real `upstream/videos/0.mkv` frames per Catalog #213 | LOW |
| Slot AAA MiPOD Axis A PARTIAL (`_wiener_filter_canonical` is box-blur not Wiener) | Slot EEE audit | PARTIAL | OPERATOR-ROUTABLE: canonical shared helper provides REAL Wiener filter via `wiener_filter_canonical_mlx` per Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1; sister wave to migrate Slot AAA `apply_*` to use canonical shared helper | MEDIUM |
| Slot CCC HUGO Axis A PARTIAL (per-pixel SPAM-delta heuristic vs matrix-distance) | Slot EEE audit | PARTIAL | OPERATOR-ROUTABLE: canonical shared helper provides `compute_hugo_per_pixel_spam_delta_mlx` (canonical 4-direction truncated-residual change magnitude per the paper first-order approximation); sister wave to migrate Slot CCC | MEDIUM |
| Slot FF UNIWARD Axis A PARTIAL (per-pair scalar collapse vs per-pixel directional wavelet) | Slot EEE audit | PARTIAL | OPERATOR-ROUTABLE: canonical shared helper provides `compute_uniward_per_pixel_directional_wavelet_mlx` (canonical Holub-Fridrich-Denemark 2014 directional sub-band filter); sister wave to migrate Slot FF | MEDIUM |
| Slot TT SegNet-boundary-waterfill Axis C FAIL (synthetic per-class SegNet response) | Slot EEE audit | PARTIAL | OPERATOR-ROUTABLE: canonical shared helper provides real-video frame ingestion; SegNet inference delegated to `tac.scorer.load_differentiable_scorers` per existing pattern in `tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py`; sister wave to migrate Slot TT | LOW (the existing Slot TT cost computation is real for per-class UNIWARD; the bug is at the frame ingestion surface) |
| Slot RR pose-axis null-projection Axis A FAKE (`apply_*` is no-op design-memo serialization) | Slot EEE audit | FAKE | OPERATOR-ROUTABLE: the operator-routable from Slot EEE landing memo task #1 (rename `apply_pose_axis_null_projection_to_pr110_archive` → `build_pose_axis_null_projection_menu_for_pr110_archive` OR implement actual frame perturbation) is unblocked by the canonical shared helper's frame ingestion primitive; sister wave to migrate Slot RR | HIGH (FAKE classification was most severe per Slot EEE) |

## Sister-extinction architecture rationale (Catalog #299 quota brake)

NO new Catalog # gate claimed per Slot CC STRATEGIC RESET #1 self-application
+ the canonical 13th OPTIMAL-TRIO standing directive (current count 382 well
under 400 quota). The remediation uses existing Catalog surfaces:

- Catalog #192 (macOS-CPU advisory NEVER promotable) — canonical Provenance + Tier A markers
- Catalog #213 (Comma2k19 canonical / upstream video canonical) — real frame ingestion enforcement
- Catalog #287 (placeholder-rationale rejection) — canonical helper rejects placeholder rationales
- Catalog #305 (observability surface) — canonical 6-facet smoke result
- Catalog #323 (canonical Provenance umbrella) — every smoke output carries canonical Provenance
- Catalog #325 (per-substrate symposium 6-step contract) — symposium memo emitted
- Catalog #335 (cathedral consumer canonical contract) — new consumer auto-discovered
- Catalog #341 (Tier A canonical-routing markers) — bind helper returns Tier A
- Catalog #348 (retroactive sweep) — THIS memo
- Catalog #356 (AxisDecomposition per-axis emission) — existing apply preserved
- Catalog #357 (dual-tier consumer architecture) — Tier A new consumer

## Empirical receipts

| Smoke | Frames | Resolution | Dynamic range (dB) | Wall-clock | Used MLX |
|-------|--------|-----------|---------------------|------------|----------|
| HILL per-pixel real-video | 4 | 96x128 | 27.83 | 0.165s | yes |
| MiPOD per-pixel real-video (REAL Wiener) | 4 | 96x128 | 1.23 | 0.094s | yes |
| UNIWARD per-pixel real-video | 4 | 96x128 | 1.69 | 0.087s | yes |
| HUGO per-pixel real-video | 4 | 96x128 | 6.02 | 0.083s | yes |

Notes:
- HILL is the canonical highest-cost-discrimination paradigm at this scale (27.83 dB), consistent with Li-Wang-Li-Huang 2014 expectations for natural-image cover sources.
- MiPOD shows low dynamic range at small resolution because the Fisher-information clip saturates; operator-routable to test at the canonical 384x512 renderer resolution.
- UNIWARD low dynamic range is consistent with the canonical Holub scale invariance.
- HUGO 6 dB band is consistent with 4-direction residual coverage at 4-truncation.

## Sister-cascade migration pattern for the other 5 targets

Each sister wave follows the canonical Slot YY HILL pattern landed in this work:

```python
def apply_<paradigm>_canonical_per_pixel_mlx_to_real_video_frames(
    *, num_frames=4, target_resolution=(128, 96), use_mlx=True, **paradigm_kwargs
) -> dict[str, Any]:
    from tac.inverse_steganalysis_real_video_mlx import (
        compute_<paradigm>_per_pixel_cost_mlx,
        run_macos_cpu_advisory_smoke,
    )
    smoke_result = run_macos_cpu_advisory_smoke(
        target_name="<paradigm>_canonical_per_pixel_mlx_real_video",
        cost_function=compute_<paradigm>_per_pixel_cost_mlx,
        num_frames=num_frames,
        target_resolution=target_resolution,
        use_mlx=use_mlx,
        cost_function_kwargs=paradigm_kwargs,
    )
    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "score_claim": False,
        "axis_tag": "[predicted]",
        "smoke_result": smoke_result.to_dict(),
        "verdict": "PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN_DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        "per_pixel_real_video_remediation_anchor": {
            "slot_eee_audit_axis_a_verdict": "PARTIAL_remediated",
            "slot_eee_audit_axis_c_verdict": "FAIL_remediated",
            "remediation_pattern": (...),
            "canonical_helper_module": "tac.inverse_steganalysis_real_video_mlx",
            "canonical_helper_function": "compute_<paradigm>_per_pixel_cost_mlx",
            "operator_binding_5_invariant_standing_directive_anchor": (...),
            "slot_eee_audit_anchor": (...),
        },
    }
```

Add `<paradigm>_canonical_per_pixel_mlx_real_video_frames` to the package's `__all__`. Add 7-9 tests mirroring `TestApplyHillPerPixelMlxRealVideoFrames`. Total: ~140 LOC + ~70 LOC tests per sister wave.

## Recursive self-reflection (Catalog #363)

This remediation is Round 1 of the canonical recursive self-reflection on the
Slot EEE audit findings. Round 2 (council self-reflection) is the per-substrate
symposium memo `.omx/research/council_per_pixel_remediation_hill_li_wang_li_huang_2014_*.md`.
Round 3 (resolution) — material unverified-assumption findings (the
ASSUMED_AWAITING_VERIFICATION on score-impact prediction) are gated to
PROVISIONAL-PENDING-VERIFICATION per CLAUDE.md canonical 4-state taxonomy;
operator-routable resolution via paired-CUDA RATIFICATION per Catalog #246.
