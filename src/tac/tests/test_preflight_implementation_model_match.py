"""Tests for tools/check_evidence_implementation_matches_model_spec.py.

Bug class: implementation-vs-model gap (audit memo
``feedback_implementation_vs_model_gap_audit_20260508.md``). The scanner
must catch the four documented gap classes:

  - CAPACITY mismatch: declared "<=200 params" vs evidence at rank=8
    (~5K params)
  - SUBSTRATE mismatch: declared 2d_natural_image vs 1D weight reshape
  - SHAPE-FAMILY mismatch: declared kaiming/laplace+outliers/spike-and-
    slab vs generic Gaussian/Laplace/Cauchy
  - VARIANT partial-exhaustion: lane-class lists multiple variants but
    aggregate evidence touches only one

Plus the baseline (faithful implementation should pass) and a
multi-criteria case (one missing dimension).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCANNER_PATH = REPO_ROOT / "tools" / "check_evidence_implementation_matches_model_spec.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        "_pact_evidence_impl_model_match_test", SCANNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_synthetic_repo(
    tmp_path: Path,
    *,
    catalog_rows: list[dict],
    evidence_rows: list[dict],
) -> tuple[Path, Path, Path]:
    """Write synthetic catalog + evidence files under tmp_path.

    Returns (repo_root, catalog_path, evidence_jsonl_path).
    """
    catalog_dir = tmp_path / "tools"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "synthetic_catalog.py"
    encoder_rows = [r for r in catalog_rows if r.get("_section", "encoder") == "encoder"]
    arch_rows = [r for r in catalog_rows if r.get("_section") == "arch"]
    # Strip the synthetic _section key before serializing.
    enc_clean = [{k: v for k, v in r.items() if k != "_section"} for r in encoder_rows]
    arch_clean = [{k: v for k, v in r.items() if k != "_section"} for r in arch_rows]
    catalog_text = (
        "ENCODER_TECHNIQUES = "
        + json.dumps(enc_clean, indent=2)
        + "\nARCH_TECHNIQUES = "
        + json.dumps(arch_clean, indent=2)
        + "\n"
    )
    catalog_path.write_text(catalog_text, encoding="utf-8")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = reports_dir / "cathedral_autopilot_evidence.jsonl"
    evidence_path.write_text(
        "\n".join(json.dumps(r) for r in evidence_rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path, catalog_path, evidence_path


def test_baseline_matching_spec_passes(tmp_path: Path) -> None:
    """A faithful evidence row that matches all model_spec dimensions passes."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "lossy_coarsening_analytical",
                "predicted_archive_bytes": 156_344,
                "cost_hours": 1.0,
                "cost_dollars": 0.0,
                "model_spec": {
                    "capacity_constraint": "per_tensor_K_search",
                    "architecture_class": "analytical_coarsening",
                    "substrate_constraint": "1d_quantized_symbols",
                    "canonical_shape_family": "per_tensor_step_size",
                    "variant_required": ["analytical_K_search"],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "lossy_coarsening_analytical",
                "empirical_archive_bytes": 156344,
                "source": (
                    "[CPU-prep empirical] reports/raw/pr101_lossy_"
                    "coarsening/manifest.json (per-tensor analytical K "
                    "coarsening)"
                ),
                "timestamp": "2026-05-08T01:17:45Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    assert findings == [], f"unexpected findings: {[f.as_str() for f in findings]}"


def test_capacity_mismatch_5k_params_for_200_param_mlp_fails(tmp_path: Path) -> None:
    """tiny_nn pattern: rank=8 factorized softmax tested against ~200-param MLP."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "tiny_nn_pmf_predictor",
                "predicted_archive_bytes": 167_000,
                "cost_hours": 3.0,
                "cost_dollars": 0.0,
                "model_spec": {
                    "capacity_constraint": "<=200_params",
                    "architecture_class": "MLP",
                    "substrate_constraint": "1d_quantized_symbols",
                    "canonical_shape_family": "tensor_id_layer_class_features",
                    "variant_required": ["mlp_under_200_params"],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "tiny_nn_pmf_predictor",
                "empirical_archive_bytes": 179_276,
                "source": (
                    "[CPU-prep empirical] reports/raw/pr101_tiny_nn_"
                    "20260508T001420Z.json (tensor_only variant; "
                    "rank=8 factorized softmax)"
                ),
                "timestamp": "2026-05-08T00:14:20Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    capacity_findings = [
        f for f in findings if f.spec_field == "capacity_constraint"
    ]
    assert capacity_findings, (
        f"expected CAPACITY_MISMATCH; got: {[f.as_str() for f in findings]}"
    )
    assert any(
        "200" in str(f.spec_value) for f in capacity_findings
    ), "capacity finding should reference the 200-param ceiling"


def test_substrate_mismatch_1d_for_scalehyperprior_fails(tmp_path: Path) -> None:
    """compressai_balle pattern: ScaleHyperprior tested on 1D weight symbols."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "compressai_balle_hyperprior",
                "predicted_archive_bytes": 158_000,
                "cost_hours": 4.0,
                "cost_dollars": 5.0,
                "model_spec": {
                    "capacity_constraint": "5KB_to_10KB_compressed_hyperprior",
                    "architecture_class": "ScaleHyperprior",
                    "substrate_constraint": "2d_natural_image",
                    "canonical_shape_family": "hyperprior_side_info_GDN_nonlinearity",
                    "variant_required": [
                        "scale_hyperprior",
                        "mean_scale_hyperprior",
                    ],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "compressai_balle_hyperprior",
                "empirical_archive_bytes": 207_065,
                "source": (
                    "[MPS-research-signal] reports/raw/pr101_balle_"
                    "hyperprior/manifest.json (ScaleHyperprior on PR101 "
                    "INT8 symbols reshape 1×1×448×512 pseudo-image)"
                ),
                "timestamp": "2026-05-08T00:22:37Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    substrate_findings = [
        f for f in findings if f.spec_field == "substrate_constraint"
    ]
    assert substrate_findings, (
        f"expected SUBSTRATE_MISMATCH; got: {[f.as_str() for f in findings]}"
    )
    assert any(
        "2d_natural_image" in str(f.spec_value)
        for f in substrate_findings
    )


def test_variant_partial_exhaustion_for_lossy_int4_fails(tmp_path: Path) -> None:
    """lossy_int4 pattern: only naive_ptq tested; QAT/LSQ/GPTQ/AWQ missing."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "_section": "arch",
                "name": "lossy_int4_quantization",
                "predicted_archive_bytes": 105_440,
                "cost_hours": 6.0,
                "cost_dollars": 8.0,
                "model_spec": {
                    "capacity_constraint": "n_quant=15_int4",
                    "architecture_class": "low_bit_quantization",
                    "substrate_constraint": "renderer_weights",
                    "canonical_shape_family": "int4_per_block_or_per_channel_scales",
                    "variant_required": [
                        "naive_ptq",
                        "qat",
                        "lsq",
                        "per_channel_scales",
                        "mixed_precision_int4_int6_int8",
                        "gptq",
                        "awq",
                    ],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "lossy_int4_quantization",
                "empirical_archive_bytes": 100_799,
                "source": (
                    "synthetic naive PTQ only — QAT/LSQ/etc not exercised"
                ),
                "timestamp": "2026-05-08T00:34:49Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    variant_findings = [
        f for f in findings if f.spec_field == "variant_required"
    ]
    assert variant_findings, (
        f"expected VARIANT_PARTIAL_EXHAUSTION; got: "
        f"{[f.as_str() for f in findings]}"
    )
    missing = variant_findings[0].spec_value
    assert isinstance(missing, list)
    # naive_ptq variant tokens should appear in the source; the other
    # variants (qat/lsq/gptq/awq/etc.) should be reported as missing.
    assert any("qat" in m or "lsq" in m or "gptq" in m or "awq" in m for m in missing), (
        f"missing variants should include unexercised alternatives, got {missing}"
    )


def test_shape_family_mismatch_for_kalle_fold_fails(tmp_path: Path) -> None:
    """kalle_fold pattern: generic Gaussian/Laplace/Cauchy vs canonical NN-shapes."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "kalle_fold_mixture_canonical_shapes",
                "predicted_archive_bytes": 173_500,
                "cost_hours": 2.0,
                "cost_dollars": 0.0,
                "model_spec": {
                    "capacity_constraint": "<=8_components",
                    "architecture_class": "mixture_of_canonical_NN_PMF_shapes",
                    "substrate_constraint": "1d_quantized_symbols",
                    "canonical_shape_family": (
                        "kaiming+laplace_with_outliers+spike_and_slab+truncated_normal"
                    ),
                    "variant_required": ["nn_weight_distribution_basis"],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "kalle_fold_mixture_canonical_shapes",
                "empirical_archive_bytes": 205_963,
                "source": (
                    "[CPU-prep empirical] reports/raw/pr101_kalle_fold/"
                    "manifest.json (4-comp mixture: Gaussian+Laplace+"
                    "delta+uniform; 8-comp adds Cauchy)"
                ),
                "timestamp": "2026-05-08T00:13:04Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    shape_findings = [
        f for f in findings if f.spec_field == "canonical_shape_family"
    ]
    assert shape_findings, (
        f"expected SHAPE_FAMILY_MISMATCH; got: "
        f"{[f.as_str() for f in findings]}"
    )


def test_multi_criteria_one_missing_fails(tmp_path: Path) -> None:
    """Hybrid case: capacity matches but substrate diverges (independent dims)."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "compressai_balle_hyperprior",
                "predicted_archive_bytes": 158_000,
                "cost_hours": 4.0,
                "cost_dollars": 5.0,
                "model_spec": {
                    "capacity_constraint": "5KB_to_10KB_compressed_hyperprior",
                    "architecture_class": "ScaleHyperprior",
                    "substrate_constraint": "2d_natural_image",
                    "canonical_shape_family": "hyperprior_side_info_GDN_nonlinearity",
                    "variant_required": [
                        "scale_hyperprior",
                        "mean_scale_hyperprior",
                    ],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "compressai_balle_hyperprior",
                # Capacity fingerprint is in range but substrate says
                # the test reshaped weights into a pseudo-image; only
                # the substrate dimension should fail.
                "empirical_archive_bytes": 200_000,
                "source": (
                    "subagent built proper 7KB compressed scale_hyperprior "
                    "on 1D weight symbols reshaped to 1×1×448×512 "
                    "pseudo-image (substrate-mismatched)"
                ),
                "timestamp": "2026-05-08T05:00:00Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    fields = {f.spec_field for f in findings}
    assert "substrate_constraint" in fields, (
        f"expected substrate_constraint mismatch in multi-criteria case; "
        f"got fields={fields}"
    )
    # Capacity should NOT be flagged here (no rank=K / factorized fingerprint).
    capacity_findings = [
        f for f in findings if f.spec_field == "capacity_constraint"
    ]
    assert not capacity_findings, (
        f"capacity should not flag in this case; got "
        f"{[f.as_str() for f in capacity_findings]}"
    )


def test_strict_mode_returns_nonzero_when_findings_present(tmp_path: Path) -> None:
    """The CLI in --strict mode must exit non-zero on any finding."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "tiny_nn_pmf_predictor",
                "predicted_archive_bytes": 167_000,
                "cost_hours": 3.0,
                "cost_dollars": 0.0,
                "model_spec": {
                    "capacity_constraint": "<=200_params",
                    "architecture_class": "MLP",
                    "substrate_constraint": "1d_quantized_symbols",
                    "canonical_shape_family": "tensor_id_layer_class_features",
                    "variant_required": ["mlp_under_200_params"],
                },
            },
        ],
        evidence_rows=[
            {
                "technique": "tiny_nn_pmf_predictor",
                "empirical_archive_bytes": 179_276,
                "source": "rank=32 factorized softmax (capacity-mismatched)",
                "timestamp": "2026-05-08T07:00:00Z",
            },
        ],
    )
    rc = mod.main([
        "--repo-root", str(repo),
        "--evidence-jsonl", str(ev),
        "--catalog-source", str(cat),
        "--strict",
    ])
    assert rc == 1, f"strict mode should return 1 on findings, got {rc}"


def test_no_model_spec_in_catalog_row_is_flagged(tmp_path: Path) -> None:
    """A catalog row that lacks a model_spec is itself a finding."""
    mod = _load_scanner()
    repo, cat, ev = _make_synthetic_repo(
        tmp_path,
        catalog_rows=[
            {
                "name": "legacy_no_spec_technique",
                "predicted_archive_bytes": 200_000,
                "cost_hours": 1.0,
                "cost_dollars": 0.0,
            },
        ],
        evidence_rows=[
            {
                "technique": "legacy_no_spec_technique",
                "empirical_archive_bytes": 195_000,
                "source": "any source",
                "timestamp": "2026-05-08T08:00:00Z",
            },
        ],
    )
    findings = mod.scan(
        repo_root=repo,
        evidence_jsonl=ev,
        catalog_source=cat,
    )
    spec_missing = [f for f in findings if f.spec_field == "model_spec"]
    assert spec_missing, (
        f"missing model_spec on catalog row should be flagged; got "
        f"{[f.as_str() for f in findings]}"
    )


def test_self_protection_layer_attaches_warning_metadata(tmp_path: Path) -> None:
    """update_catalog_from_evidence flags mismatches, sets non-promotable."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        import cathedral_autopilot as ca  # noqa: WPS433
    finally:
        try:
            sys.path.remove(str(REPO_ROOT / "tools"))
        except ValueError:
            pass

    catalog = [
        {
            "name": "tiny_nn_pmf_predictor",
            "predicted_archive_bytes": 167_000,
            "cost_hours": 3.0,
            "cost_dollars": 0.0,
            "risk": "lossless",
            "evidence_grade": "[predicted]",
            "description": "200-param MLP",
            "model_spec": {
                "capacity_constraint": "<=200_params",
                "architecture_class": "MLP",
                "substrate_constraint": "1d_quantized_symbols",
                "canonical_shape_family": "tensor_id_layer_class_features",
                "variant_required": ["mlp_under_200_params"],
            },
        },
    ]
    evidence = [
        ca.TechniqueEvidence(
            technique="tiny_nn_pmf_predictor",
            empirical_archive_bytes=179_276,
            source=(
                "[CPU-prep empirical] rank=8 factorized softmax — "
                "5K params not 200"
            ),
            timestamp="2026-05-08T07:30:00Z",
        ),
    ]
    updated = ca.update_catalog_from_evidence(
        catalog, evidence, log_warnings=False,
    )
    row = updated[0]
    assert row.get("model_spec_mismatch_count", 0) >= 1, (
        f"expected mismatch_count >= 1; got row={row}"
    )
    assert row.get("empirical_anchor_promotable") is False, (
        "mismatched evidence should never be flagged promotable"
    )
    blockers = row.get("dispatch_blockers") or []
    assert "model_spec_mismatch_pending_faithful_implementation" in blockers, (
        f"expected mismatch blocker; got blockers={blockers}"
    )
