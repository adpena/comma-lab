# SPDX-License-Identifier: MIT
"""Tests for Catalog #250-#255 Rudin-Daubechies autopilot ranker self-protect gates."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_compressive_landscape_canonical_use,
    check_falling_rule_list_canonical_use,
    check_gosdt_dispatcher_whiteboard_discipline,
    check_rashomon_ensemble_continual_update_locked,
    check_slim_ranker_consumes_canonical_taylor_proxies,
    check_wavelet_multi_scale_ranker_contract,
)


# ── live-repo regression guards (all 6 gates must report 0) ───────────────


def test_check_250_live_repo_zero_violations():
    assert check_slim_ranker_consumes_canonical_taylor_proxies() == []


def test_check_251_live_repo_zero_violations():
    assert check_falling_rule_list_canonical_use() == []


def test_check_252_live_repo_zero_violations():
    assert check_rashomon_ensemble_continual_update_locked() == []


def test_check_253_live_repo_zero_violations():
    assert check_compressive_landscape_canonical_use() == []


def test_check_254_live_repo_zero_violations():
    assert check_wavelet_multi_scale_ranker_contract() == []


def test_check_255_live_repo_zero_violations():
    assert check_gosdt_dispatcher_whiteboard_discipline() == []


# ── #250: SLIM float-coef detection ───────────────────────────────────────


def test_check_250_flags_float_coef_token(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad_module.py").write_text(
        "float_coef = 1.5  # constructing a non-integer SLIM coefficient\n",
        encoding="utf-8",
    )
    violations = check_slim_ranker_consumes_canonical_taylor_proxies(
        repo_root=repo
    )
    assert len(violations) == 1
    assert "float_coef" in violations[0]


def test_check_250_waiver_with_rationale_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok_module.py").write_text(
        "float_coef = 1.5  # SLIM_FLOAT_COEF_OK:test fixture intentionally floats\n",
        encoding="utf-8",
    )
    violations = check_slim_ranker_consumes_canonical_taylor_proxies(
        repo_root=repo
    )
    assert violations == []


def test_check_250_placeholder_rationale_rejected(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "float_coef = 1.5  # SLIM_FLOAT_COEF_OK:<rationale>\n",
        encoding="utf-8",
    )
    violations = check_slim_ranker_consumes_canonical_taylor_proxies(
        repo_root=repo
    )
    assert len(violations) == 1


def test_check_250_strict_raises(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "non_integer_coef = 0.5\n", encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="Catalog #250"):
        check_slim_ranker_consumes_canonical_taylor_proxies(
            repo_root=repo, strict=True
        )


def test_check_250_self_exempt_path(tmp_path):
    """Files inside the canonical autopilot package itself are exempt."""
    repo = tmp_path
    (repo / "src" / "tac" / "autopilot_rudin_daubechies").mkdir(parents=True)
    (repo / "src" / "tac" / "autopilot_rudin_daubechies" / "self.py").write_text(
        "float_coef = 1.5\n", encoding="utf-8"
    )
    violations = check_slim_ranker_consumes_canonical_taylor_proxies(
        repo_root=repo
    )
    assert violations == []


# ── #251: Falling-rule-list non-falling pattern detection ─────────────────


def test_check_251_flags_ascending_rule_list(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "ascending_rule_list = []\n", encoding="utf-8"
    )
    violations = check_falling_rule_list_canonical_use(repo_root=repo)
    assert len(violations) == 1


def test_check_251_waiver_with_rationale_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok.py").write_text(
        "ascending_rule_list = []  # NON_FALLING_RULE_LIST_OK:legacy test fixture\n",
        encoding="utf-8",
    )
    violations = check_falling_rule_list_canonical_use(repo_root=repo)
    assert violations == []


def test_check_251_strict_raises(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "rising_rule_list = []\n", encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="Catalog #251"):
        check_falling_rule_list_canonical_use(repo_root=repo, strict=True)


# ── #252: Rashomon ensemble persistence enforcement ──────────────────────


def test_check_252_flags_bare_construction(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "e = RashomonEnsembleRanker(rng_seed=0).update_all(0.20, panel, "
        "store_path=None)\n",
        encoding="utf-8",
    )
    violations = check_rashomon_ensemble_continual_update_locked(repo_root=repo)
    # The simple regex matches the literal pattern in the file.
    assert len(violations) >= 1


def test_check_252_waiver_with_rationale_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok.py").write_text(
        "e = RashomonEnsembleRanker(rng_seed=0).update_all(0.20, panel, "
        "store_path=None)  # RASHOMON_NON_PERSISTED_OK:smoke-only ephemeral\n",
        encoding="utf-8",
    )
    violations = check_rashomon_ensemble_continual_update_locked(repo_root=repo)
    assert violations == []


# ── #253: Dense-anchor reconstruction refusal ────────────────────────────


def test_check_253_flags_dense_anchor_token(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "dense_anchor_reconstruction = True\n", encoding="utf-8"
    )
    violations = check_compressive_landscape_canonical_use(repo_root=repo)
    assert len(violations) == 1


def test_check_253_waiver_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok.py").write_text(
        "full_landscape_grid_search = True  # DENSE_ANCHOR_OK:diagnostic eval\n",
        encoding="utf-8",
    )
    violations = check_compressive_landscape_canonical_use(repo_root=repo)
    assert violations == []


def test_check_253_strict_raises(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "dense_anchor_reconstruction = True\n", encoding="utf-8"
    )
    with pytest.raises(PreflightError, match="Catalog #253"):
        check_compressive_landscape_canonical_use(repo_root=repo, strict=True)


# ── #254: Multi-scale contract enforcement ───────────────────────────────


def test_check_254_flags_single_scale_token(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "single_scale_rule_list_for_multi_scale_problem = True\n",
        encoding="utf-8",
    )
    violations = check_wavelet_multi_scale_ranker_contract(repo_root=repo)
    assert len(violations) == 1


def test_check_254_flags_fine_overrides_coarse(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "fine_rule_overrides_coarse_gate = True\n", encoding="utf-8"
    )
    violations = check_wavelet_multi_scale_ranker_contract(repo_root=repo)
    assert len(violations) == 1


def test_check_254_waiver_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok.py").write_text(
        "single_scale_rule_list_for_multi_scale_problem = True  "
        "# SINGLE_SCALE_OK:legacy migration backlog\n",
        encoding="utf-8",
    )
    violations = check_wavelet_multi_scale_ranker_contract(repo_root=repo)
    assert violations == []


# ── #255: GOSDT whiteboard discipline ────────────────────────────────────


def test_check_255_flags_auto_promote_token(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "auto_promote_whiteboard_rule(rule_id='r1')\n", encoding="utf-8"
    )
    violations = check_gosdt_dispatcher_whiteboard_discipline(repo_root=repo)
    assert len(violations) == 1


def test_check_255_waiver_passes(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "ok.py").write_text(
        "auto_promote_whiteboard_rule(rule_id='r1')  "
        "# AUTO_PROMOTE_WHITEBOARD_OK:offline replay tool\n",
        encoding="utf-8",
    )
    violations = check_gosdt_dispatcher_whiteboard_discipline(repo_root=repo)
    assert violations == []


def test_check_255_placeholder_rationale_rejected(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "auto_promote_whiteboard_rule(rule_id='r1')  "
        "# AUTO_PROMOTE_WHITEBOARD_OK:<rationale>\n",
        encoding="utf-8",
    )
    violations = check_gosdt_dispatcher_whiteboard_discipline(repo_root=repo)
    assert len(violations) == 1


def test_check_255_strict_raises(tmp_path):
    repo = tmp_path
    (repo / "src" / "tac").mkdir(parents=True)
    (repo / "src" / "tac" / "bad.py").write_text(
        "promote_whiteboard_rule_without_operator_review()\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="Catalog #255"):
        check_gosdt_dispatcher_whiteboard_discipline(
            repo_root=repo, strict=True
        )


# ── orchestrator wire-in regression guards ────────────────────────────────


def test_preflight_all_wires_check_250_strict_true():
    """Confirm Catalog #250 is wired strict=True in preflight_all."""
    import tac.preflight as p

    src = Path(p.__file__).read_text(encoding="utf-8")
    assert "check_slim_ranker_consumes_canonical_taylor_proxies" in src
    assert "strict=True" in src.split(
        "check_slim_ranker_consumes_canonical_taylor_proxies"
    )[1].split(")")[0]


def test_preflight_all_wires_all_six_gates_strict_true():
    """All 6 Rudin-Daubechies gates wired strict=True in preflight_all."""
    import tac.preflight as p

    src = Path(p.__file__).read_text(encoding="utf-8")
    for fn_name in [
        "check_slim_ranker_consumes_canonical_taylor_proxies",
        "check_falling_rule_list_canonical_use",
        "check_rashomon_ensemble_continual_update_locked",
        "check_compressive_landscape_canonical_use",
        "check_wavelet_multi_scale_ranker_contract",
        "check_gosdt_dispatcher_whiteboard_discipline",
    ]:
        # Find the wire-in invocation; verify strict=True nearby.
        idx = src.find(fn_name + "(\n")
        assert idx > 0, f"{fn_name} not wired into preflight_all"
        # Look in next 100 chars for strict=True.
        window = src[idx : idx + 200]
        assert "strict=True" in window, f"{fn_name} not strict=True"
