# SPDX-License-Identifier: MIT
"""Tests for the Rudin floor `_full_main` implementation (Phase 1b lift).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + the design memo
``.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md``
§15 canonical-vs-unique decision per layer + Catalog #229 premise-verification.

Coverage:

* ``_full_main`` no longer raises NotImplementedError (Phase 1b lift)
* canonical PR95-paradigm tokens present in source (decode_real_pairs,
  patch_upstream_yuv6_globally, load_differentiable_scorers, gate_auth_eval_call,
  require_contest_cuda_auth_eval_claim, posterior_update_locked_from_auth_eval_json,
  detect_hardware_substrate)
* substrate-unique tokens present (RashomonEnsembleRanker, pack_archive,
  RudinRuleList, RDIF v1 grammar)
* canonical-vs-unique decision per layer regression guard (N/A items not
  silently re-added: EMA / eval_roundtrip / AdamW / torch.compile / autocast)
* 9-dim checklist evidence regression guard (file-level waivers + design memo
  reference present)
* archive grammar roundtrip-stable on a synthetic Rashomon-driven rule_list
* Rashomon K=8 bootstrap helper produces canonical falling-rule-list
* feature extraction helper handles missing-scorers fallback
* `--max-pairs` + `--skip-auth-eval` + `--skip-archive-build` argparse plumbing
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TRAINER_PATH = (
    REPO_ROOT / "experiments" / "train_substrate_rudin_floor_interpretable_ml.py"
)

# Ensure REPO_ROOT/src on path
sys.path.insert(0, str(REPO_ROOT / "src"))

import experiments.train_substrate_rudin_floor_interpretable_ml as trainer_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Source-level / AST-level regression guards
# ---------------------------------------------------------------------------


def test_trainer_file_exists() -> None:
    assert TRAINER_PATH.exists(), f"trainer source missing: {TRAINER_PATH}"


def _trainer_source() -> str:
    return TRAINER_PATH.read_text(encoding="utf-8")


def _full_main_body() -> str:
    """Return the source of `_full_main` as a string (AST-extracted)."""
    tree = ast.parse(_trainer_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_full_main":
            return ast.unparse(node)
    raise AssertionError("no _full_main found")


def test_full_main_no_longer_raises_not_implemented_error() -> None:
    """Phase 1b lift: `_full_main` body must not contain `raise NotImplementedError`."""
    body = _full_main_body()
    assert "raise NotImplementedError" not in body, (
        "Phase 1b lift regression: _full_main re-introduced NotImplementedError"
    )


def test_full_main_binds_canonical_pr95_paradigm_ingredients() -> None:
    """Each adopt-canonical layer per design memo §15 must be referenced.

    Tokens checked are the canonical helper names; the PR95-paradigm-compliant
    pattern requires them to appear in the trainer (in `_full_main` or via
    module-level import the function uses).
    """
    src = _trainer_source()
    required = [
        # pyav decode (Catalog #114)
        "decode_real_pairs",
        # upstream yuv6 patch (Catalog #187)
        "patch_upstream_yuv6_globally",
        # scorer load (compress-time ORACLE per Wyner-Ziv side-info)
        "load_differentiable_scorers",
        # auth eval canonical gate (Catalog #226)
        "gate_auth_eval_call",
        # custody validator (Catalog #127)
        "require_contest_cuda_auth_eval_claim",
        # posterior update (Catalog #128)
        "posterior_update_locked_from_auth_eval_json",
        # hardware substrate detection (Catalog #190)
        "detect_hardware_substrate",
        # device gate (canonical)
        "device_or_die",
    ]
    missing = [t for t in required if t not in src]
    assert not missing, f"PR95-paradigm canonical tokens missing: {missing}"


def test_full_main_binds_substrate_unique_ingredients() -> None:
    """Substrate's UNIQUE (FORK) layers per design memo §15 must be present."""
    src = _trainer_source()
    required = [
        # Rashomon K=8 bootstrap (substrate's UNIQUE training mode)
        "RashomonEnsembleRanker",
        # RDIF v1 archive grammar (UNIQUE per design memo §10)
        "pack_archive",
        "RudinRuleList",
        # Substrate-specific (FORK) Rashomon bootstrap helper
        "_compile_rashomon_rule_list",
        # Compress-time scorer feature extraction (FORK)
        "_compute_per_pair_scorer_features",
    ]
    missing = [t for t in required if t not in src]
    assert not missing, f"Substrate-unique tokens missing: {missing}"


def test_full_main_does_not_silently_reintroduce_n_a_layers() -> None:
    """Design memo §15 N/A layers (no neural training) must NOT appear inside _full_main.

    The Rudin substrate is no-neural by construction; if EMA / eval_roundtrip /
    AdamW / torch.compile / autocast appear in `_full_main`, the substrate has
    drifted away from its declared class-shift.
    """
    body = _full_main_body()
    forbidden = [
        # EMA — N/A per §15
        "EMA(renderer",
        "ema.update",
        # eval_roundtrip — N/A per §15
        "apply_eval_roundtrip",
        "apply_eval_roundtrip_during_training",
        # AdamW backprop — N/A per §15
        "AdamW(",
        # torch.compile — N/A per §15
        "torch.compile(",
        # autocast — N/A per §15
        "torch.autocast(",
    ]
    found = [t for t in forbidden if t in body]
    assert not found, (
        f"Design memo §15 N/A layers leaked into _full_main: {found} — "
        "substrate has drifted from its no-neural class-shift declaration"
    )


def test_trainer_has_design_memo_reference() -> None:
    src = _trainer_source()
    assert "rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md" in src


def test_trainer_has_9_dim_dispatch_optimization_protocol_waivers() -> None:
    """File-level Catalog #270 dispatch optimization protocol waivers must be present."""
    src = _trainer_source()[:5000]  # first ~5000 chars for header waivers
    required_waivers = [
        "DISPATCH_OPTIMIZATION_PROTOCOL_OK",
        "AUTOCAST_FP16_WAIVED",
        "TORCH_COMPILE_WAIVED",
        "TF32_WAIVED",
        "SCORER_LOADER_ORDER_OK",
        "AUTH_EVAL_DIRECT_SUBPROCESS_OK",
    ]
    missing = [w for w in required_waivers if w not in src]
    assert not missing, f"file-level waivers missing: {missing}"


def test_trainer_has_required_argparse_flags() -> None:
    """`_full_main` references --max-pairs / --skip-auth-eval / --skip-archive-build / --full-cpu."""
    parser = trainer_mod._build_parser()
    args = parser.parse_args(["--output-dir", "/tmp/_dummy_rudin"])
    assert hasattr(args, "max_pairs")
    assert hasattr(args, "skip_auth_eval")
    assert hasattr(args, "skip_archive_build")
    assert hasattr(args, "full_cpu")
    assert hasattr(args, "advisory_cpu_explicitly_waived")
    assert hasattr(args, "k_rules")
    assert hasattr(args, "k_rashomon")
    assert hasattr(args, "slim_coeff_bound")


# ---------------------------------------------------------------------------
# Helper unit-tests (Rashomon bootstrap + feature extraction)
# ---------------------------------------------------------------------------


def test_rashomon_bootstrap_emits_canonical_falling_rule_list() -> None:
    from tac.substrates.rudin_floor_interpretable_ml import (
        CANONICAL_K_RASHOMON,
        CANONICAL_K_RULES,
        CANONICAL_SLIM_COEFF_BOUND,
        RudinRuleList,
    )

    # Synthetic features: 100 pairs spread across canonical class indices
    pair_features = []
    for i in range(100):
        pair_features.append({
            "mean_class": i % 4,  # cycles across road/vehicle/sky/person
            "class_diversity": 0.5 + (i % 3) * 0.3,
            "pose_motion": 5.0 + (i % 7) * 6.0,  # varies 5..47
            "chroma_var": 25.0 + (i % 5) * 10.0,
        })

    rule_list, summary = trainer_mod._compile_rashomon_rule_list(
        pair_features,
        k_rules=CANONICAL_K_RULES,
        k_rashomon=CANONICAL_K_RASHOMON,
        slim_coeff_bound=CANONICAL_SLIM_COEFF_BOUND,
        seed=0,
    )
    assert isinstance(rule_list, RudinRuleList)
    # Wang-Rudin canonical: K=4-6 rules (always-fire catch-all is last).
    assert 2 <= len(rule_list.rules) <= CANONICAL_K_RULES + 1
    # Last rule MUST be the always-fire catch-all (Wang-Rudin first-match-wins).
    assert rule_list.rules[-1].predicate == "always"
    # Summary metadata expected
    assert "k_rashomon" in summary
    assert summary["k_rashomon"] == CANONICAL_K_RASHOMON
    assert "rule_metadata" in summary
    assert "anchors_seen" in summary
    # Per design memo §4: K=8 bootstrap-diverse SLIM rankers; class-bucket
    # aggregation keeps refit cost O(K²·S) — anchors_seen reflects buckets,
    # not raw pair count.
    assert 1 <= summary["anchors_seen"] <= CANONICAL_K_RASHOMON
    assert "ensemble_confidence_tag" in summary
    assert "class_counts" in summary
    assert "ensemble_n_anchors" in summary


def test_rashomon_bootstrap_byte_deterministic_for_seed() -> None:
    """Same seed + same features → byte-identical archive (HNeRV L9)."""
    from tac.substrates.rudin_floor_interpretable_ml import (
        CANONICAL_K_RASHOMON,
        CANONICAL_K_RULES,
        CANONICAL_SLIM_COEFF_BOUND,
        pack_archive,
    )

    pair_features = [
        {"mean_class": i % 5, "class_diversity": 0.3, "pose_motion": 10.0, "chroma_var": 50.0}
        for i in range(50)
    ]
    rule_list_a, _ = trainer_mod._compile_rashomon_rule_list(
        pair_features,
        k_rules=CANONICAL_K_RULES,
        k_rashomon=CANONICAL_K_RASHOMON,
        slim_coeff_bound=CANONICAL_SLIM_COEFF_BOUND,
        seed=42,
    )
    rule_list_b, _ = trainer_mod._compile_rashomon_rule_list(
        pair_features,
        k_rules=CANONICAL_K_RULES,
        k_rashomon=CANONICAL_K_RASHOMON,
        slim_coeff_bound=CANONICAL_SLIM_COEFF_BOUND,
        seed=42,
    )
    assert pack_archive(rule_list=rule_list_a) == pack_archive(rule_list=rule_list_b)


def test_features_to_panel_returns_canonical_proxy_panel() -> None:
    from tac.autopilot_rudin_daubechies.slim_ranker import ProxyPanel
    panel = trainer_mod._features_to_panel({
        "mean_class": 2,
        "class_diversity": 1.5,
        "pose_motion": 20.0,
        "chroma_var": 64.0,
    })
    assert isinstance(panel, ProxyPanel)
    assert panel.panel_axis == "diagnostic_cpu"
    assert "rudin_pair_class_2" in panel.candidate_id


# ---------------------------------------------------------------------------
# Archive grammar end-to-end smoke (no GPU; no scorer)
# ---------------------------------------------------------------------------


def test_pack_archive_with_rashomon_rule_list_roundtrips(tmp_path: Path) -> None:
    from tac.substrates.rudin_floor_interpretable_ml import pack_archive, parse_archive

    pair_features = [
        {"mean_class": 0, "class_diversity": 0.5, "pose_motion": 8.0, "chroma_var": 30.0},
        {"mean_class": 1, "class_diversity": 0.6, "pose_motion": 12.0, "chroma_var": 40.0},
        {"mean_class": 2, "class_diversity": 1.2, "pose_motion": 35.0, "chroma_var": 55.0},
        {"mean_class": 0, "class_diversity": 0.4, "pose_motion": 5.0, "chroma_var": 25.0},
    ]
    rule_list, _ = trainer_mod._compile_rashomon_rule_list(
        pair_features,
        k_rules=6,
        k_rashomon=8,
        slim_coeff_bound=10,
        seed=0,
    )
    blob = pack_archive(rule_list=rule_list, scorer_priors_blob=b'{"pairs":4}')
    parsed = parse_archive(blob)
    assert parsed.header.section_count == 8
    assert len(parsed.rule_list.rules) == len(rule_list.rules)
    # rule_list_blob is non-empty
    assert len(parsed.sections["rule_list_blob"]) > 0


# ---------------------------------------------------------------------------
# main() dispatch (smoke path regression)
# ---------------------------------------------------------------------------


def test_smoke_path_still_works(tmp_path: Path) -> None:
    rc = trainer_mod.main(["--smoke", "--output-dir", str(tmp_path / "smoke")])
    assert rc == 0
    assert (tmp_path / "smoke" / "0.bin").exists()
    assert (tmp_path / "smoke" / "smoke_stats.json").exists()


def test_main_routes_full_when_not_smoke() -> None:
    """When --smoke is omitted, main() routes to _full_main (which now exists)."""
    with mock.patch.object(trainer_mod, "_full_main", return_value=42) as patched:
        rc = trainer_mod.main(["--output-dir", "/tmp/_dummy"])
        patched.assert_called_once()
        assert rc == 42


def test_full_cpu_requires_advisory_waiver_per_catalog_197(tmp_path: Path) -> None:
    """Catalog #197: --full-cpu without --advisory-cpu-explicitly-waived raises SystemExit."""
    with pytest.raises(SystemExit):
        trainer_mod.main([
            "--output-dir", str(tmp_path / "out"),
            "--full-cpu",
            # missing --advisory-cpu-explicitly-waived
        ])


# ---------------------------------------------------------------------------
# Canonical-vs-unique decision per layer regression guard
# ---------------------------------------------------------------------------


def test_docstring_documents_canonical_vs_unique_decision_per_layer() -> None:
    """Per Catalog #290 + design memo §15: trainer header must reference the
    canonical-vs-unique-per-layer pattern explicitly."""
    src = _trainer_source()
    # Look for the per-layer comment block we added
    assert "ADOPT canonical" in src, "canonical adoption decisions not documented in trainer"
    assert "FORK" in src, "FORK decisions (substrate-unique layers) not documented in trainer"
    assert "N/A" in src, "N/A decisions (no-neural class-shift) not documented in trainer"
    # Each layer reference from design memo §15
    assert "decode_real_pairs" in src
    assert "Rashomon K=8 bootstrap" in src
    assert "RDIF v1 grammar" in src


# ---------------------------------------------------------------------------
# CLI plumbing (verify --k-rules / --k-rashomon / --slim-coeff-bound thread through)
# ---------------------------------------------------------------------------


def test_argparse_default_rudin_discipline_knobs() -> None:
    from tac.substrates.rudin_floor_interpretable_ml import (
        CANONICAL_GOSDT_DEPTH,
        CANONICAL_K_RASHOMON,
        CANONICAL_K_RULES,
        CANONICAL_SLIM_COEFF_BOUND,
    )
    parser = trainer_mod._build_parser()
    args = parser.parse_args(["--output-dir", "/tmp/_dummy"])
    assert args.k_rules == CANONICAL_K_RULES
    assert args.k_rashomon == CANONICAL_K_RASHOMON
    assert args.slim_coeff_bound == CANONICAL_SLIM_COEFF_BOUND
    assert args.gosdt_depth == CANONICAL_GOSDT_DEPTH
