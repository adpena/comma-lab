"""Tests for ``tools/generate_synthetic_substrate_anchors.py``.

Per CLAUDE.md "Recursive adversarial review protocol" + "Forbidden score
claims" + "synthetic-not-empirical" separation, these tests verify the
synthetic anchor generator produces:

- 16 canonical specs (matches the QQ landing's non-HNeRV substrate count)
- NN-classifier-compatible JSON schema
- Deterministic anchors (seed-controlled)
- Per-fine-grained-class drift kernels
- score_claim=False permanent invariants
- /tmp path refusal
- CLI smoke

Coverage:
- canonical_synthetic_substrate_specs returns 16 specs
- per-spec drift vector deterministic by seed
- synthetic_anchor flag set true on every anchor
- anchor schema matches NN classifier required keys
- NaN handling (synthetic-anchor JSON parses via NN's classifier)
- emit_anchors filesystem layout
- /tmp output dir refused
- CLI smoke
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.generate_synthetic_substrate_anchors import (
    DEFAULT_SEED,
    SCHEMA_VERSION,
    SubstrateSyntheticSpec,
    build_synthetic_anchor,
    canonical_synthetic_substrate_specs,
    emit_anchors,
    main,
    parse_args,
)


# ── Spec inventory ───────────────────────────────────────────────────────


def test_canonical_inventory_has_16_specs():
    specs = canonical_synthetic_substrate_specs()
    assert len(specs) == 16


def test_canonical_inventory_substrate_ids_unique():
    specs = canonical_synthetic_substrate_specs()
    ids = [s.substrate_id for s in specs]
    assert len(set(ids)) == len(ids)


def test_canonical_inventory_coarse_class_all_non_hnerv():
    specs = canonical_synthetic_substrate_specs()
    for s in specs:
        assert s.coarse_class == "non_hnerv_family"


def test_canonical_inventory_classes_represented():
    specs = canonical_synthetic_substrate_specs()
    classes = {s.fine_grained_class for s in specs}
    assert {"residual", "pose_axis_sidechannel", "self_compression", "nerv_family"} <= classes


def test_canonical_inventory_class_counts():
    specs = canonical_synthetic_substrate_specs()
    counts: dict[str, int] = {}
    for s in specs:
        counts[s.fine_grained_class] = counts.get(s.fine_grained_class, 0) + 1
    assert counts["residual"] == 5
    assert counts["pose_axis_sidechannel"] == 3
    assert counts["self_compression"] == 3
    assert counts["nerv_family"] == 5


# ── build_synthetic_anchor ───────────────────────────────────────────────


def test_build_synthetic_anchor_top_level_keys_present():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    required = {
        "schema",
        "synthetic_anchor",
        "synthetic_anchor_seed",
        "label",
        "tag",
        "tool",
        "generated_at_utc",
        "mode",
        "evidence_grade",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "first_divergence",
        "stage_compounding",
        "layer_drift_rows",
        "cpu_record_path",
        "cuda_record_path",
        "cpu_capture_host",
    }
    assert required <= set(anchor.keys())


def test_build_synthetic_anchor_score_claim_false_permanent():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    assert anchor["score_claim"] is False
    assert anchor["promotion_eligible"] is False
    assert anchor["ready_for_exact_eval_dispatch"] is False
    assert anchor["score_claim_valid"] is False
    assert anchor["rank_or_kill_eligible"] is False
    assert anchor["dispatch_attempted"] is False


def test_build_synthetic_anchor_synthetic_flag_true():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    assert anchor["synthetic_anchor"] is True
    assert anchor["mode"] == "synthetic_not_empirical"


def test_build_synthetic_anchor_has_synthetic_tag_in_string():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    assert "synthetic-not-empirical" in anchor["tag"]
    assert "synthetic_not_empirical" in anchor["mixed_substrate_advisory"]


def test_build_synthetic_anchor_evidence_grade_valid():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    # NN classifier accepts both `diagnostic_not_score` and `diagnostic-not-score`.
    assert anchor["evidence_grade"] in ("diagnostic_not_score", "diagnostic-not-score")


def test_build_synthetic_anchor_layer_drift_rows_match_spec_n_layers():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    assert len(anchor["layer_drift_rows"]) == spec.n_layers


def test_build_synthetic_anchor_first_divergence_has_threshold():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    fd = anchor["first_divergence"]
    assert "first_l2_relative_exceedance" in fd
    assert "l2_relative_threshold" in fd
    assert fd["l2_relative_threshold"] == 1e-2


def test_build_synthetic_anchor_stage_compounding_has_by_stage():
    spec = canonical_synthetic_substrate_specs()[0]
    anchor = build_synthetic_anchor(spec, seed=DEFAULT_SEED)
    sc = anchor["stage_compounding"]
    assert "by_stage" in sc
    assert isinstance(sc["by_stage"], list)
    assert len(sc["by_stage"]) >= 1


# ── Determinism ──────────────────────────────────────────────────────────


def test_build_synthetic_anchor_deterministic_by_seed():
    spec = canonical_synthetic_substrate_specs()[0]
    a1 = build_synthetic_anchor(spec, seed=42)
    a2 = build_synthetic_anchor(spec, seed=42)
    # generated_at_utc varies; compare on the deterministic bits.
    assert [r["fingerprint_only_l2_proxy"] for r in a1["layer_drift_rows"]] == [
        r["fingerprint_only_l2_proxy"] for r in a2["layer_drift_rows"]
    ]


def test_build_synthetic_anchor_different_seed_changes_drift():
    spec = canonical_synthetic_substrate_specs()[0]
    a1 = build_synthetic_anchor(spec, seed=1)
    a2 = build_synthetic_anchor(spec, seed=2)
    v1 = [r["fingerprint_only_l2_proxy"] for r in a1["layer_drift_rows"]]
    v2 = [r["fingerprint_only_l2_proxy"] for r in a2["layer_drift_rows"]]
    assert v1 != v2


def test_build_synthetic_anchor_different_substrate_changes_drift():
    specs = canonical_synthetic_substrate_specs()
    a = build_synthetic_anchor(specs[0], seed=DEFAULT_SEED)
    b = build_synthetic_anchor(specs[5], seed=DEFAULT_SEED)
    va = [r["fingerprint_only_l2_proxy"] for r in a["layer_drift_rows"]]
    vb = [r["fingerprint_only_l2_proxy"] for r in b["layer_drift_rows"]]
    assert va != vb


# ── Per-fine-grained-class drift kernels ─────────────────────────────────


def test_residual_drift_decays_with_depth():
    specs = canonical_synthetic_substrate_specs()
    res_spec = next(s for s in specs if s.fine_grained_class == "residual")
    anchor = build_synthetic_anchor(res_spec, seed=DEFAULT_SEED)
    drift = [r["fingerprint_only_l2_proxy"] for r in anchor["layer_drift_rows"]]
    # Compare first quartile mean to last quartile mean; first should be larger
    # (entropy coder front-loads drift).
    n = len(drift)
    first_q = drift[: n // 4]
    last_q = drift[3 * n // 4 :]
    first_q_mean = sum(first_q) / len(first_q)
    last_q_mean = sum(last_q) / len(last_q)
    assert first_q_mean > last_q_mean


def test_nerv_family_drift_grows_with_depth():
    specs = canonical_synthetic_substrate_specs()
    nerv_spec = next(s for s in specs if s.fine_grained_class == "nerv_family")
    anchor = build_synthetic_anchor(nerv_spec, seed=DEFAULT_SEED)
    drift = [r["fingerprint_only_l2_proxy"] for r in anchor["layer_drift_rows"]]
    n = len(drift)
    first_q = drift[: n // 4]
    last_q = drift[3 * n // 4 :]
    first_q_mean = sum(first_q) / len(first_q)
    last_q_mean = sum(last_q) / len(last_q)
    # NeRV family compounds drift across depth — last quartile > first quartile.
    assert last_q_mean > first_q_mean


# ── emit_anchors ─────────────────────────────────────────────────────────


def test_emit_anchors_writes_per_substrate_layer_drift(tmp_path: Path):
    specs = canonical_synthetic_substrate_specs()[:3]
    manifest = emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    assert manifest["n_anchors"] == 3
    for spec in specs:
        sub_dir = tmp_path / "out" / spec.substrate_id
        assert (sub_dir / "layer_drift.json").exists()


def test_emit_anchors_writes_jsonl_index(tmp_path: Path):
    specs = canonical_synthetic_substrate_specs()[:3]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    jsonl = (tmp_path / "out" / "anchors_index.jsonl").read_text()
    rows = [json.loads(line) for line in jsonl.splitlines() if line]
    assert len(rows) == 3
    for r in rows:
        assert r["score_claim"] is False
        assert r["synthetic"] is True


def test_emit_anchors_writes_set_manifest(tmp_path: Path):
    specs = canonical_synthetic_substrate_specs()[:3]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    manifest = json.loads(
        (tmp_path / "out" / "anchors_set_manifest.json").read_text()
    )
    assert manifest["schema"] == SCHEMA_VERSION
    assert manifest["score_claim"] is False
    assert manifest["synthetic_set"] is True


def test_emit_anchors_refuses_tmp_dir():
    specs = canonical_synthetic_substrate_specs()[:1]
    with pytest.raises(SystemExit):
        emit_anchors(Path("/tmp/should_not_be_allowed"), specs, seed=DEFAULT_SEED)


def test_emit_anchors_jsonl_anchor_path_resolvable(tmp_path: Path):
    specs = canonical_synthetic_substrate_specs()[:2]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    jsonl = (tmp_path / "out" / "anchors_index.jsonl").read_text()
    for line in jsonl.splitlines():
        if not line:
            continue
        row = json.loads(line)
        anchor_path = Path(row["anchor_path"])
        assert anchor_path.exists()


# ── NN-classifier compatibility ──────────────────────────────────────────


def test_synthetic_anchors_match_nn_classifier_required_keys(tmp_path: Path):
    """The synthetic anchor JSON should pass NN classifier's _load_and_validate."""
    specs = canonical_synthetic_substrate_specs()[:1]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    anchor_path = tmp_path / "out" / specs[0].substrate_id / "layer_drift.json"
    data = json.loads(anchor_path.read_text())
    # Required keys per NN classifier:
    required_keys = (
        "layer_drift_rows",
        "evidence_grade",
        "cpu_record_path",
        "cuda_record_path",
        "first_divergence",
    )
    for k in required_keys:
        assert k in data, f"missing required key for NN classifier: {k!r}"
    assert data["evidence_grade"] in ("diagnostic_not_score", "diagnostic-not-score")


def test_synthetic_anchor_layer_rows_have_required_subkeys(tmp_path: Path):
    specs = canonical_synthetic_substrate_specs()[:1]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    anchor_path = tmp_path / "out" / specs[0].substrate_id / "layer_drift.json"
    data = json.loads(anchor_path.read_text())
    rows = data["layer_drift_rows"]
    assert rows
    for r in rows[:5]:
        # NN classifier reads `fingerprint_only_l2_proxy` when
        # `l2_relative_error` is null/NaN (synthetic case).
        assert "fingerprint_only_l2_proxy" in r
        assert "module_type" in r


# ── CLI smoke ────────────────────────────────────────────────────────────


def test_main_cli_smoke(tmp_path: Path):
    args = ["--output-dir", str(tmp_path / "cli_smoke"), "--seed", "7"]
    rc = main(args)
    assert rc == 0
    assert (tmp_path / "cli_smoke" / "anchors_set_manifest.json").exists()


def test_main_cli_substrate_subset(tmp_path: Path):
    args = [
        "--output-dir",
        str(tmp_path / "subset_smoke"),
        "--substrate-ids",
        "wavelet_residual,c3_residual",
    ]
    rc = main(args)
    assert rc == 0
    manifest = json.loads(
        (tmp_path / "subset_smoke" / "anchors_set_manifest.json").read_text()
    )
    assert manifest["n_anchors"] == 2


def test_main_cli_unknown_substrate_returns_2(tmp_path: Path):
    args = [
        "--output-dir",
        str(tmp_path / "no_match_smoke"),
        "--substrate-ids",
        "nonexistent_substrate",
    ]
    rc = main(args)
    assert rc == 2


def test_parse_args_required_output_dir():
    with pytest.raises(SystemExit):
        parse_args(["--seed", "1"])


# ── Determinism across emit ──────────────────────────────────────────────


def test_emit_anchors_byte_deterministic_by_seed(tmp_path: Path):
    """Two emit calls with the same seed produce byte-identical anchors."""
    specs = canonical_synthetic_substrate_specs()[:2]
    out1 = tmp_path / "first"
    out2 = tmp_path / "second"
    emit_anchors(out1, specs, seed=42)
    emit_anchors(out2, specs, seed=42)
    for s in specs:
        a1 = (out1 / s.substrate_id / "layer_drift.json").read_text()
        a2 = (out2 / s.substrate_id / "layer_drift.json").read_text()
        # Strip generated_at_utc lines (timestamp varies).
        a1_lines = [
            line for line in a1.splitlines() if "generated_at_utc" not in line
        ]
        a2_lines = [
            line for line in a2.splitlines() if "generated_at_utc" not in line
        ]
        assert a1_lines == a2_lines


# ── Clean separation: no real anchor leakage ─────────────────────────────


def test_synthetic_anchor_paths_carry_synthetic_token(tmp_path: Path):
    """Synthetic anchor SHA256 fields begin with 'synthetic_' to prevent
    collision with real empirical anchors in downstream consumers."""
    specs = canonical_synthetic_substrate_specs()[:1]
    emit_anchors(tmp_path / "out", specs, seed=DEFAULT_SEED)
    data = json.loads(
        (tmp_path / "out" / specs[0].substrate_id / "layer_drift.json").read_text()
    )
    assert data["cpu_record_sha256"].startswith("synthetic_")
    assert data["cuda_record_sha256"].startswith("synthetic_")
    assert data["shared_input_tensor_sha256"].startswith("synthetic_")
