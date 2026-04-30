"""Lane 17 — Real-archive empirical smoke (CPU-only, deterministic).

[empirical:reports/lane_17_imp_real_archive.json]

Runs a 2-cycle IMP simulation on the actual Lane G v3 anchor renderer.bin
and measures sparse-CSR byte savings vs the dense FP16 / FP4A baseline. No
GPU. Skips if the anchor file is missing (deploy-only).

Per Council design (.omx/research/council_lane_17_imp_design_20260430.md):
- Predicted-band [0.85, 1.20] 90% CI; this test does NOT predict the score.
- It measures the codec savings ONLY — score requires a contest-CUDA dispatch.

The test writes its measurement to ``reports/lane_17_imp_real_archive.json``
so other artifacts (memory entries, dispatcher provenance) can reference
the empirical bytes alongside the [empirical:...] tag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tac.imps_renderer_archive import (
    IMPS_PER_TENSOR_NUMEL_CAP,
    decode_imps_archive,
    encode_imps_archive,
)
from tac.iterative_magnitude_pruning import (
    apply_mask_to_model,
    compute_actual_sparsity,
    iter_prunable_parameters,
    prune_lowest_magnitude,
)


_ROOT = Path(__file__).resolve().parents[3]
_ANCHOR = (
    _ROOT / "experiments" / "results" / "lane_g_v3_landed" / "iter_0" / "renderer.bin"
)
_REPORT_PATH = _ROOT / "reports" / "lane_17_imp_real_archive.json"


def _load_anchor():
    """Load the Lane G v3 ASYM/FP4A renderer.bin into an
    AsymmetricPairGenerator. Returns (model, raw_bytes)."""
    raw = _ANCHOR.read_bytes()
    magic = raw[:4]
    if magic == b"FP4A":
        from tac.renderer_export import load_asymmetric_checkpoint_fp4

        model = load_asymmetric_checkpoint_fp4(raw, device="cpu")
    elif magic == b"ASYM":
        from tac.renderer_export import load_asymmetric_checkpoint

        model = load_asymmetric_checkpoint(raw, device="cpu")
    else:
        raise pytest.skip.Exception(
            f"Lane G v3 anchor has unexpected magic {magic!r} — only FP4A/ASYM "
            f"supported by this smoke. Update the test if Lane G ships a new "
            f"format."
        )
    return model, raw


@pytest.mark.skipif(
    not _ANCHOR.exists(),
    reason=f"Lane G v3 anchor missing at {_ANCHOR}; smoke is deploy-only.",
)
def test_imp_real_archive_savings_at_89_percent_sparsity() -> None:
    """[empirical:reports/lane_17_imp_real_archive.json]

    Loads the Lane G v3 anchor, simulates 2 IMP cycles to ~36% sparsity
    (cycle-1 of the 10-cycle plan), 6 cycles to ~74% (cycle-5), and 10
    cycles to ~89% (cycle-9), encodes each as IMPS archive, and reports
    bytes vs the FP4A baseline.

    Asserts the 89%-sparsity archive is STRICTLY SMALLER than the
    dense-mask (no-prune) IMPS baseline. The dense baseline comparison
    isolates the sparse-CSR codec contribution from the FP16-vs-FP4
    container overhead.
    """
    torch.manual_seed(2026)
    model, raw_baseline = _load_anchor()
    n_baseline = len(raw_baseline)
    print(f"\n[lane-17-smoke] Lane G v3 anchor (FP4A): {n_baseline:,} bytes")

    # IMPS-encode the dense model (no masks) → all FP16 fallback. Used as
    # the codec-overhead baseline so the comparison isolates sparsity
    # savings from the FP16-vs-FP4 container difference.
    blob_dense = encode_imps_archive(model=model, masks={})
    n_dense_imps = len(blob_dense)
    print(f"[lane-17-smoke] IMPS dense (no masks, all FP16): {n_dense_imps:,} bytes")

    # Run 10 IMP cycles at 20%/cycle → 89% sparsity.
    masks = None
    cycle_results = []
    for cycle_idx in range(10):
        masks = prune_lowest_magnitude(
            model, sparsity_increment=0.20, current_mask=masks
        )
        apply_mask_to_model(model, masks)
        sparsity = compute_actual_sparsity(model, masks)
        # Encode at this cycle.
        blob_imps = encode_imps_archive(model=model, masks=masks)
        n_imps = len(blob_imps)
        sparse_layers = sum(
            1 for n, p in iter_prunable_parameters(model)
            if p.numel() <= IMPS_PER_TENSOR_NUMEL_CAP
        )
        cycle_results.append({
            "cycle": cycle_idx,
            "cumulative_sparsity": sparsity,
            "expected_sparsity": 1.0 - 0.8 ** (cycle_idx + 1),
            "imps_archive_bytes": n_imps,
            "savings_vs_dense_imps_pct": (
                (n_dense_imps - n_imps) / n_dense_imps * 100.0
            ),
            "savings_vs_fp4a_anchor_pct": (
                (n_baseline - n_imps) / n_baseline * 100.0
            ),
            "sparse_eligible_layer_count": sparse_layers,
        })
        print(
            f"[lane-17-smoke] cycle {cycle_idx}: sparsity={sparsity:.3f} → "
            f"{n_imps:,} bytes (vs dense IMPS {n_dense_imps:,}: "
            f"{cycle_results[-1]['savings_vs_dense_imps_pct']:+.1f}%)"
        )

    # Verify the IMPS round-trip after the final 89%-sparse cycle.
    final_blob = encode_imps_archive(model=model, masks=masks)
    decoded = decode_imps_archive(data=final_blob, device="cpu")
    # Architecture should match
    n_orig_params = sum(p.numel() for p in model.parameters())
    n_dec_params = sum(p.numel() for p in decoded.parameters())
    assert n_orig_params == n_dec_params, (
        f"param count mismatch after IMPS roundtrip: "
        f"{n_orig_params} vs {n_dec_params}"
    )
    final = cycle_results[-1]

    # Write the empirical report.
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "anchor_path": str(_ANCHOR.relative_to(_ROOT)),
        "anchor_fp4a_bytes": n_baseline,
        "imps_dense_baseline_bytes": n_dense_imps,
        "cycle_results": cycle_results,
        "final_sparsity": final["cumulative_sparsity"],
        "final_imps_bytes": final["imps_archive_bytes"],
        "savings_vs_dense_imps_pct": final["savings_vs_dense_imps_pct"],
        "savings_vs_fp4a_anchor_pct": final["savings_vs_fp4a_anchor_pct"],
        "council_predicted_band": [0.85, 1.20],
        "tag": "[empirical] CPU-only codec measurement; score still pending [contest-CUDA].",
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[lane-17-smoke] wrote {_REPORT_PATH}")

    # Final assertions.
    # 1. After 10 cycles, sparsity is approximately 89.3% (1 - 0.8^10).
    assert abs(final["cumulative_sparsity"] - 0.893) < 0.02, (
        f"final sparsity {final['cumulative_sparsity']:.3f} should be ~0.893"
    )
    # 2. The 89%-sparse IMPS archive must be SMALLER than the dense IMPS
    # baseline — the sparse-CSR codec MUST extract savings.
    assert final["imps_archive_bytes"] < n_dense_imps, (
        f"89%-sparse IMPS {final['imps_archive_bytes']:,} should be smaller "
        f"than dense-IMPS baseline {n_dense_imps:,}"
    )
    # 3. We must observe a positive savings PERCENTAGE.
    assert final["savings_vs_dense_imps_pct"] > 0.0


@pytest.mark.skipif(
    not _ANCHOR.exists(),
    reason=f"Lane G v3 anchor missing at {_ANCHOR}; smoke is deploy-only.",
)
def test_imp_archive_handles_dense_anchor_no_imp() -> None:
    """[empirical:reports/lane_17_imp_real_archive.json]

    Sanity: passing the dense anchor with empty masks must always succeed
    and produce a valid IMPS archive (all-FP16 fallback). This is the
    pre-cycle-0 baseline measurement.
    """
    model, _raw = _load_anchor()
    blob = encode_imps_archive(model=model, masks={})
    assert blob[:4] == b"IMPS"
    decoded = decode_imps_archive(data=blob, device="cpu")
    assert decoded is not None
    # Param count match
    assert (
        sum(p.numel() for p in model.parameters())
        == sum(p.numel() for p in decoded.parameters())
    )
