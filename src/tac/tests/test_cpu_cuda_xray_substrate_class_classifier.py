# SPDX-License-Identifier: MIT
"""Tests for ``tools/cpu_cuda_xray_substrate_class_classifier.py``.

Per CLAUDE.md "Beauty, simplicity, and DX" + the substrate-class-boundary
hypothesis (council Insight 1), this test module covers:

* ``InputSpec`` parsing (well-formed and ill-formed forms);
* schema validation of input ``layer_drift.json`` files (missing keys,
  bad evidence_grade, malformed JSON);
* ``extract_signature`` produces a per-substrate signature with the
  expected fields;
* ``cosine_similarity`` matches numpy's reference for known vectors;
* class-centroid computation aggregates correctly across multiple
  substrates per class;
* classifier verdict logic (CONFIRMS / CONTRADICTS / INSUFFICIENT-DATA);
* ``_validate_output_dir`` refuses /tmp paths;
* CLI dry-run produces JSON to stdout without touching disk;
* CLI requires --output-dir unless --dry-run;
* deterministic-bytes guarantee (same inputs → same manifest body).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.cpu_cuda_xray_substrate_class_classifier import (
    InputSpec,
    SubstrateSignature,
    _SUBSTRATE_CLASSES,
    _VALID_EVIDENCE_GRADES,
    _l2_value_for_signature,
    _load_and_validate,
    _truncate_to_min_length,
    _validate_output_dir,
    build_manifest,
    classify_substrates,
    compute_class_centroids,
    compute_pairwise_similarity,
    cosine_similarity,
    extract_signature,
    main,
    parse_args,
    parse_input_spec,
)


def _write_minimal_layer_drift_json(
    path: Path,
    *,
    label: str,
    layer_l2_values: list[float] | None = None,
    cpu_substrate: str = "linux_x86_64",
) -> None:
    """Write a minimal valid layer_drift.json for fixture use."""
    if layer_l2_values is None:
        layer_l2_values = [1e-7, 2e-7, 3e-7, 4e-7]
    rows = []
    for i, v in enumerate(layer_l2_values):
        rows.append(
            {
                "fingerprint_only_l2_proxy": float(v),
                "fingerprint_only_max_proxy": float(v) * 2.0,
                "has_full_tensors": False,
                "kl_divergence": None,
                "l2_relative_error": None,
                "layer_name": f"layer_{i}",
                "max_abs_error": None,
                "mean_abs_error": None,
                "module_type": "TestModule",
                "note": "test fixture",
                "output_index": i,
                "rank_top1_disagreement": None,
            }
        )
    payload = {
        "label": label,
        "layer_drift_rows": rows,
        "evidence_grade": "diagnostic_not_score",
        "cpu_record_path": f"experiments/results/{label}/cpu_record.pt",
        "cuda_record_path": f"experiments/results/{label}/cuda_record.pt",
        "first_divergence": {
            "first_argmax_divergence": None,
            "first_l2_relative_exceedance": None,
            "l2_relative_threshold": 0.01,
        },
        "stage_compounding": {
            "by_stage": [
                {
                    "compound_factor": 1.000001,
                    "eps_sources": ["fingerprint_proxy"],
                    "max_eps": 0.0,
                    "mean_eps": 0.0,
                    "num_layers": 1,
                    "stage_key": "test_stage",
                }
            ]
        },
        "cpu_capture_host": {
            "is_linux_x86_64": cpu_substrate == "linux_x86_64",
            "is_macos_darwin": cpu_substrate == "macos_darwin",
            "machine": "x86_64" if cpu_substrate == "linux_x86_64" else "arm64",
            "platform": cpu_substrate,
            "system": "Linux" if cpu_substrate == "linux_x86_64" else "Darwin",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ── parse_input_spec tests ──────────────────────────────────────────────────


def test_parse_input_spec_valid():
    spec = parse_input_spec("hnerv_family:experiments/results/foo/layer_drift.json")
    assert spec.substrate_class == "hnerv_family"
    assert spec.path == Path("experiments/results/foo/layer_drift.json")


def test_parse_input_spec_strips_whitespace():
    spec = parse_input_spec(" hnerv_family : foo/bar.json ")
    assert spec.substrate_class == "hnerv_family"
    assert spec.path == Path("foo/bar.json")


def test_parse_input_spec_missing_colon_raises():
    with pytest.raises(SystemExit, match="must be of the form"):
        parse_input_spec("hnerv_family_no_colon_path.json")


def test_parse_input_spec_unknown_class_raises():
    with pytest.raises(SystemExit, match="unknown substrate_class"):
        parse_input_spec("not_a_real_class:foo/bar.json")


def test_substrate_classes_constant():
    assert "hnerv_family" in _SUBSTRATE_CLASSES
    assert "non_hnerv_family" in _SUBSTRATE_CLASSES


def test_valid_evidence_grades_constant():
    assert "diagnostic_not_score" in _VALID_EVIDENCE_GRADES


# ── _load_and_validate tests ────────────────────────────────────────────────


def test_load_and_validate_missing_file_raises(tmp_path: Path):
    with pytest.raises(SystemExit, match="not found"):
        _load_and_validate(tmp_path / "nope.json")


def test_load_and_validate_invalid_json_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("not-json{{{")
    with pytest.raises(SystemExit, match="invalid JSON"):
        _load_and_validate(bad)


def test_load_and_validate_non_object_raises(tmp_path: Path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(SystemExit, match="must be a JSON object"):
        _load_and_validate(bad)


def test_load_and_validate_missing_keys_raises(tmp_path: Path):
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps({"layer_drift_rows": []}))
    with pytest.raises(SystemExit, match="missing required keys"):
        _load_and_validate(bad)


def test_load_and_validate_bad_evidence_grade_raises(tmp_path: Path):
    bad = tmp_path / "bad_grade.json"
    bad.write_text(
        json.dumps(
            {
                "layer_drift_rows": [],
                "evidence_grade": "contest_cuda",  # forbidden authoritative grade
                "cpu_record_path": "x",
                "cuda_record_path": "y",
                "first_divergence": {},
            }
        )
    )
    with pytest.raises(SystemExit, match="unexpected evidence_grade"):
        _load_and_validate(bad)


def test_load_and_validate_minimal_valid(tmp_path: Path):
    p = tmp_path / "ok.json"
    _write_minimal_layer_drift_json(p, label="test")
    data = _load_and_validate(p)
    assert data["label"] == "test"


# ── _l2_value_for_signature tests ───────────────────────────────────────────


def test_l2_value_for_signature_uses_l2_relative_when_present():
    row = {
        "l2_relative_error": 0.5,
        "fingerprint_only_l2_proxy": 0.1,
    }
    assert _l2_value_for_signature(row) == 0.5


def test_l2_value_for_signature_falls_back_to_proxy_on_none():
    row = {
        "l2_relative_error": None,
        "fingerprint_only_l2_proxy": 0.1,
    }
    assert _l2_value_for_signature(row) == 0.1


def test_l2_value_for_signature_falls_back_on_nan():
    row = {
        "l2_relative_error": float("nan"),
        "fingerprint_only_l2_proxy": 0.1,
    }
    assert _l2_value_for_signature(row) == 0.1


def test_l2_value_for_signature_zero_when_both_missing():
    row = {"l2_relative_error": None, "fingerprint_only_l2_proxy": None}
    assert _l2_value_for_signature(row) == 0.0


# ── extract_signature tests ─────────────────────────────────────────────────


def test_extract_signature_basic(tmp_path: Path):
    p = tmp_path / "x.json"
    _write_minimal_layer_drift_json(p, label="x", layer_l2_values=[1e-6, 2e-6])
    data = _load_and_validate(p)
    sig = extract_signature(substrate_class="hnerv_family", path=p, data=data)
    assert sig.substrate_class == "hnerv_family"
    assert sig.label == "x"
    assert sig.n_layers == 2
    assert len(sig.drift_vector) == 2
    assert sig.evidence_grade == "diagnostic_not_score"
    assert sig.layer_drift_sha256 != ""


def test_extract_signature_records_substrate_host(tmp_path: Path):
    p = tmp_path / "linux.json"
    _write_minimal_layer_drift_json(p, label="x", cpu_substrate="linux_x86_64")
    data = _load_and_validate(p)
    sig = extract_signature(substrate_class="hnerv_family", path=p, data=data)
    assert sig.cpu_capture_substrate == "linux_x86_64"


def test_extract_signature_records_macos(tmp_path: Path):
    p = tmp_path / "mac.json"
    _write_minimal_layer_drift_json(p, label="x", cpu_substrate="macos_darwin")
    data = _load_and_validate(p)
    sig = extract_signature(substrate_class="non_hnerv_family", path=p, data=data)
    assert sig.cpu_capture_substrate == "macos_darwin"


def test_extract_signature_aggregates_stage_compounding(tmp_path: Path):
    p = tmp_path / "stages.json"
    _write_minimal_layer_drift_json(p, label="x")
    sig = extract_signature(
        substrate_class="hnerv_family", path=p, data=_load_and_validate(p)
    )
    assert sig.n_stages == 1
    assert math.isclose(sig.stage_compounding_total, 1.000001, rel_tol=1e-6)


# ── cosine_similarity tests ─────────────────────────────────────────────────


def test_cosine_similarity_identity_is_one():
    a = [1.0, 2.0, 3.0]
    assert math.isclose(cosine_similarity(a, a), 1.0, abs_tol=1e-9)


def test_cosine_similarity_orthogonal_is_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


def test_cosine_similarity_anti_parallel_is_minus_one():
    assert math.isclose(
        cosine_similarity([1.0, 2.0], [-1.0, -2.0]), -1.0, abs_tol=1e-9
    )


def test_cosine_similarity_zero_norm_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_truncates_to_min_length():
    sim = cosine_similarity([1.0, 1.0, 1.0, 99.0], [1.0, 1.0, 1.0])
    assert math.isclose(sim, 1.0, abs_tol=1e-9)


def test_cosine_similarity_empty_returns_zero():
    assert cosine_similarity([], [1.0, 2.0]) == 0.0


# ── _truncate_to_min_length ─────────────────────────────────────────────────


def test_truncate_to_min_length():
    truncated = _truncate_to_min_length([[1, 2, 3, 4], [1, 2], [1, 2, 3]])
    assert truncated == [[1, 2], [1, 2], [1, 2]]


def test_truncate_to_min_length_empty():
    assert _truncate_to_min_length([]) == []


# ── compute_class_centroids ─────────────────────────────────────────────────


def test_compute_class_centroids_single_class():
    sigs = [
        SubstrateSignature(
            substrate_class="hnerv_family",
            label="a",
            layer_drift_path="/a",
            layer_drift_sha256="aa",
            n_layers=3,
            mean_l2_relative_error=0.0,
            max_l2_relative_error=0.0,
            median_l2_relative_error=0.0,
            first_divergence={},
            stage_compounding_total=1.0,
            n_stages=0,
            cpu_capture_substrate="linux_x86_64",
            cuda_record_path="",
            cpu_record_path="",
            evidence_grade="diagnostic_not_score",
            drift_vector=[1.0, 2.0, 3.0],
        ),
        SubstrateSignature(
            substrate_class="hnerv_family",
            label="b",
            layer_drift_path="/b",
            layer_drift_sha256="bb",
            n_layers=3,
            mean_l2_relative_error=0.0,
            max_l2_relative_error=0.0,
            median_l2_relative_error=0.0,
            first_divergence={},
            stage_compounding_total=1.0,
            n_stages=0,
            cpu_capture_substrate="linux_x86_64",
            cuda_record_path="",
            cpu_record_path="",
            evidence_grade="diagnostic_not_score",
            drift_vector=[3.0, 4.0, 5.0],
        ),
    ]
    centroids = compute_class_centroids(sigs)
    assert centroids["hnerv_family"] == [2.0, 3.0, 4.0]


def test_compute_class_centroids_truncates_to_min_length():
    sigs = [
        SubstrateSignature(
            substrate_class="hnerv_family",
            label="a", layer_drift_path="/a", layer_drift_sha256="x",
            n_layers=4,
            mean_l2_relative_error=0.0, max_l2_relative_error=0.0,
            median_l2_relative_error=0.0, first_divergence={},
            stage_compounding_total=1.0, n_stages=0,
            cpu_capture_substrate="x", cuda_record_path="",
            cpu_record_path="", evidence_grade="diagnostic_not_score",
            drift_vector=[1.0, 1.0, 1.0, 1.0],
        ),
        SubstrateSignature(
            substrate_class="hnerv_family",
            label="b", layer_drift_path="/b", layer_drift_sha256="x",
            n_layers=2,
            mean_l2_relative_error=0.0, max_l2_relative_error=0.0,
            median_l2_relative_error=0.0, first_divergence={},
            stage_compounding_total=1.0, n_stages=0,
            cpu_capture_substrate="x", cuda_record_path="",
            cpu_record_path="", evidence_grade="diagnostic_not_score",
            drift_vector=[2.0, 2.0],
        ),
    ]
    centroids = compute_class_centroids(sigs)
    # Truncated to length 2; mean of [1.0, 1.0] and [2.0, 2.0].
    assert centroids["hnerv_family"] == [1.5, 1.5]


# ── compute_pairwise_similarity ─────────────────────────────────────────────


def test_compute_pairwise_similarity_off_diagonal_only():
    sigs = [
        SubstrateSignature(
            substrate_class="hnerv_family",
            label="a", layer_drift_path="/a", layer_drift_sha256="x",
            n_layers=2,
            mean_l2_relative_error=0.0, max_l2_relative_error=0.0,
            median_l2_relative_error=0.0, first_divergence={},
            stage_compounding_total=1.0, n_stages=0,
            cpu_capture_substrate="x", cuda_record_path="",
            cpu_record_path="", evidence_grade="diagnostic_not_score",
            drift_vector=[1.0, 1.0],
        ),
        SubstrateSignature(
            substrate_class="non_hnerv_family",
            label="b", layer_drift_path="/b", layer_drift_sha256="x",
            n_layers=2,
            mean_l2_relative_error=0.0, max_l2_relative_error=0.0,
            median_l2_relative_error=0.0, first_divergence={},
            stage_compounding_total=1.0, n_stages=0,
            cpu_capture_substrate="x", cuda_record_path="",
            cpu_record_path="", evidence_grade="diagnostic_not_score",
            drift_vector=[1.0, -1.0],
        ),
    ]
    rows = compute_pairwise_similarity(sigs)
    assert len(rows) == 1  # 2 substrates → 1 off-diagonal pair
    assert rows[0]["substrate_a_label"] == "a"
    assert rows[0]["substrate_b_label"] == "b"
    assert not rows[0]["same_class_declared"]


# ── classify_substrates ─────────────────────────────────────────────────────


def _make_sig(substrate_class: str, label: str, vec: list[float]) -> SubstrateSignature:
    return SubstrateSignature(
        substrate_class=substrate_class,
        label=label, layer_drift_path=f"/{label}", layer_drift_sha256="x",
        n_layers=len(vec),
        mean_l2_relative_error=0.0, max_l2_relative_error=0.0,
        median_l2_relative_error=0.0, first_divergence={},
        stage_compounding_total=1.0, n_stages=0,
        cpu_capture_substrate="linux_x86_64", cuda_record_path="",
        cpu_record_path="", evidence_grade="diagnostic_not_score",
        drift_vector=vec,
    )


def test_classify_substrates_insufficient_data_when_class_has_one_member():
    sigs = [_make_sig("hnerv_family", "alone", [1.0, 1.0, 1.0])]
    centroids = compute_class_centroids(sigs)
    rows = classify_substrates(sigs, centroids)
    assert rows[0]["verdict"] == "INSUFFICIENT-DATA"


def test_classify_substrates_confirms_when_aligned_with_centroid():
    sigs = [
        _make_sig("hnerv_family", "a", [1.0, 1.0, 1.0]),
        _make_sig("hnerv_family", "b", [1.0, 1.0, 1.0]),
        _make_sig("hnerv_family", "c", [1.0, 1.0, 1.0]),
        _make_sig("non_hnerv_family", "x", [-1.0, -1.0, -1.0]),
        _make_sig("non_hnerv_family", "y", [-1.0, -1.0, -1.0]),
    ]
    centroids = compute_class_centroids(sigs)
    rows = classify_substrates(sigs, centroids)
    # All hnerv_family substrates align perfectly with their centroid.
    hnerv_verdicts = [
        r["verdict"] for r in rows if r["declared_substrate_class"] == "hnerv_family"
    ]
    assert all(v == "CONFIRMS" for v in hnerv_verdicts)


def test_classify_substrates_contradicts_when_misaligned():
    sigs = [
        _make_sig("hnerv_family", "a_correct", [1.0, 1.0, 1.0]),
        _make_sig("hnerv_family", "b_correct", [1.0, 1.0, 1.0]),
        _make_sig("non_hnerv_family", "x_correct", [-1.0, -1.0, -1.0]),
        _make_sig("non_hnerv_family", "y_correct", [-1.0, -1.0, -1.0]),
        # Mislabeled: declared hnerv but anti-aligned with hnerv centroid
        _make_sig("hnerv_family", "wrong_class", [-1.0, -1.0, -1.0]),
    ]
    centroids = compute_class_centroids(sigs)
    rows = classify_substrates(sigs, centroids)
    wrong_row = next(r for r in rows if r["substrate_label"] == "wrong_class")
    assert wrong_row["verdict"] == "CONTRADICTS"


# ── _validate_output_dir tests ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "forbidden_path",
    [Path("/tmp/x"), Path("/private/tmp/x")],
)
def test_validate_output_dir_refuses_tmp(forbidden_path: Path):
    with pytest.raises(SystemExit, match=r"forbidden_/tmp_paths"):
        _validate_output_dir(forbidden_path)


def test_validate_output_dir_accepts_canonical(tmp_path: Path):
    _validate_output_dir(tmp_path)


# ── parse_args + main tests ─────────────────────────────────────────────────


def test_parse_args_requires_input_spec():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_basic(tmp_path: Path):
    args = parse_args(
        [
            "--input-spec",
            "hnerv_family:foo.json",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert args.input_spec == ["hnerv_family:foo.json"]
    assert args.output_dir == tmp_path
    assert args.dry_run is False
    assert math.isclose(args.confirms_threshold, 0.85)
    assert math.isclose(args.contradicts_threshold, 0.30)


def test_parse_args_dry_run_flag():
    args = parse_args(["--input-spec", "hnerv_family:x.json", "--dry-run"])
    assert args.dry_run is True


def test_parse_args_custom_thresholds():
    args = parse_args(
        [
            "--input-spec",
            "hnerv_family:x.json",
            "--dry-run",
            "--confirms-threshold",
            "0.95",
            "--contradicts-threshold",
            "0.10",
        ]
    )
    assert math.isclose(args.confirms_threshold, 0.95)
    assert math.isclose(args.contradicts_threshold, 0.10)


def test_main_dry_run_outputs_json(
    capsys: pytest.CaptureFixture, tmp_path: Path
):
    p = tmp_path / "input.json"
    _write_minimal_layer_drift_json(p, label="x")
    rc = main(
        [
            "--input-spec",
            f"hnerv_family:{p}",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert (
        payload["schema"]
        == "cpu_cuda_xray_substrate_class_classifier_manifest.v1"
    )
    assert payload["score_claim"] is False


def test_main_requires_output_dir_unless_dry_run(tmp_path: Path):
    p = tmp_path / "input.json"
    _write_minimal_layer_drift_json(p, label="x")
    with pytest.raises(SystemExit, match="--output-dir is required"):
        main(["--input-spec", f"hnerv_family:{p}"])


def test_main_writes_manifest_to_disk(tmp_path: Path):
    p_a = tmp_path / "a.json"
    p_b = tmp_path / "b.json"
    _write_minimal_layer_drift_json(p_a, label="a")
    _write_minimal_layer_drift_json(p_b, label="b")
    out = tmp_path / "out"
    rc = main(
        [
            "--input-spec",
            f"hnerv_family:{p_a}",
            "--input-spec",
            f"hnerv_family:{p_b}",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    manifest_path = (
        out / "cpu_cuda_xray_substrate_class_classifier_manifest.json"
    )
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["n_input_substrates"] == 2


def test_build_manifest_records_score_claim_disciplines(tmp_path: Path):
    p = tmp_path / "x.json"
    _write_minimal_layer_drift_json(p, label="x")
    manifest = build_manifest(
        input_specs=[InputSpec(substrate_class="hnerv_family", path=p)],
        operator=None,
        confirms_threshold=0.85,
        contradicts_threshold=0.30,
    )
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["evidence_grade"] == "diagnostic_not_score"


def test_main_deterministic_bytes(tmp_path: Path):
    p = tmp_path / "x.json"
    _write_minimal_layer_drift_json(p, label="x")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    main(
        [
            "--input-spec",
            f"hnerv_family:{p}",
            "--input-spec",
            f"non_hnerv_family:{p}",
            "--output-dir",
            str(out_a),
        ]
    )
    main(
        [
            "--input-spec",
            f"hnerv_family:{p}",
            "--input-spec",
            f"non_hnerv_family:{p}",
            "--output-dir",
            str(out_b),
        ]
    )
    body_a = (out_a / "cpu_cuda_xray_substrate_class_classifier_manifest.json").read_text()
    body_b = (out_b / "cpu_cuda_xray_substrate_class_classifier_manifest.json").read_text()
    payload_a = json.loads(body_a)
    payload_b = json.loads(body_b)
    payload_a.pop("generated_at_utc")
    payload_b.pop("generated_at_utc")
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(
        payload_b, sort_keys=True
    )
