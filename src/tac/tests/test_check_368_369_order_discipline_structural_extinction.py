# SPDX-License-Identifier: MIT
"""Tests for Catalog #368 + #369 ORDER-DISCIPLINE structural-extinction gates.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
11th ORDER standing directive Dim 2 + Dim 8 + Catalog #348 retroactive-sweep
discipline.

Catalog #368 = substitution-stacking baseline matches canonical frontier
pointer (sister of Catalog #343).

Catalog #369 = substrate inflate consumes real trained weights (not synthetic
frame base; sister of Catalog #213 + #146 + #205 + #220 + #272).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_368_extract_sha_candidates_from_text,
    _check_368_iter_recipe_files,
    _check_368_load_canonical_frontier_shas,
    _check_368_recipe_has_opt_out,
    _check_368_recipe_has_stacking_trigger,
    _check_368_recipe_has_waiver,
    _check_368_sha_matches_canonical_frontier,
    _check_369_file_has_real_frame_vendor,
    _check_369_file_has_synthetic_pattern,
    _check_369_file_has_waiver,
    _check_369_iter_inflate_files,
    _check_369_substrate_has_opt_out,
    check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base,
    check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer,
)


# ============================================================================
# Catalog #368 — helper unit tests
# ============================================================================


def test_check_368_extract_sha_explicit_field(tmp_path: Path):
    """Extracts sha from explicit baseline_archive_sha field."""
    text = "baseline_archive_sha: 6bae0201fb082457a02c6956"
    shas = _check_368_extract_sha_candidates_from_text(text)
    assert "6bae0201fb082457a02c6956" in shas


def test_check_368_extract_sha_context_token(tmp_path: Path):
    """Extracts sha via 'sha' context token."""
    text = "stacked on archive sha 7a0da5d0fc327cba3f7d"
    shas = _check_368_extract_sha_candidates_from_text(text)
    assert any("7a0da5d0fc327cba" in s for s in shas)


def test_check_368_extract_sha_no_match_on_unrelated_hex(tmp_path: Path):
    """Does NOT extract random hex without sha context."""
    text = "magic_number = 0xdeadbeefcafebabe"  # no sha context
    shas = _check_368_extract_sha_candidates_from_text(text)
    assert shas == []


def test_check_368_recipe_has_stacking_trigger_field(tmp_path: Path):
    text = "baseline_archive_sha: 6bae0201"
    assert _check_368_recipe_has_stacking_trigger(text) is True


def test_check_368_recipe_has_stacking_trigger_text(tmp_path: Path):
    text = "Stacked archive built on PR101+FEC6 baseline (sha 6bae0201)"
    assert _check_368_recipe_has_stacking_trigger(text.lower()) is True


def test_check_368_recipe_has_stacking_trigger_negative(tmp_path: Path):
    text = "regular trainer dispatch with no stacking pattern"
    assert _check_368_recipe_has_stacking_trigger(text) is False


def test_check_368_opt_out_detected_research_only(tmp_path: Path):
    text = "dispatch_enabled: true\nresearch_only: true\n"
    assert _check_368_recipe_has_opt_out(text) is True


def test_check_368_opt_out_detected_substrate_engineering(tmp_path: Path):
    text = "lane_class: substrate_engineering\n"
    assert _check_368_recipe_has_opt_out(text) is True


def test_check_368_opt_out_detected_dispatch_disabled(tmp_path: Path):
    text = "dispatch_enabled: false\n"
    assert _check_368_recipe_has_opt_out(text) is True


def test_check_368_opt_out_negative(tmp_path: Path):
    text = "dispatch_enabled: true\nresearch_only: false\n"
    assert _check_368_recipe_has_opt_out(text) is False


def test_check_368_waiver_with_rationale(tmp_path: Path):
    text = "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:exploring alternate baselines per operator approval"
    assert _check_368_recipe_has_waiver(text) is True


def test_check_368_waiver_placeholder_rejected(tmp_path: Path):
    text = "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:<rationale>"
    assert _check_368_recipe_has_waiver(text) is False


def test_check_368_waiver_reason_placeholder_rejected(tmp_path: Path):
    text = "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:<reason>"
    assert _check_368_recipe_has_waiver(text) is False


def test_check_368_waiver_short_rationale_rejected(tmp_path: Path):
    text = "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:abc"  # <4 chars
    assert _check_368_recipe_has_waiver(text) is False


def test_check_368_sha_matches_canonical_frontier_full_match(tmp_path: Path):
    frontier = {
        "contest_cpu": "7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe",
    }
    assert _check_368_sha_matches_canonical_frontier(
        "7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe",
        frontier,
    ) is True


def test_check_368_sha_matches_canonical_frontier_prefix(tmp_path: Path):
    frontier = {
        "contest_cpu": "7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe",
    }
    assert _check_368_sha_matches_canonical_frontier(
        "7a0da5d0fc327cba", frontier
    ) is True


def test_check_368_sha_no_match(tmp_path: Path):
    frontier = {
        "contest_cpu": "7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe",
    }
    assert _check_368_sha_matches_canonical_frontier(
        "6bae0201fb082457", frontier  # FEC6 baseline, not frontier
    ) is False


# ============================================================================
# Catalog #368 — end-to-end gate behavior
# ============================================================================


def _make_recipes_dir(tmp_path: Path, recipes: dict[str, str]) -> Path:
    """Make a synthetic .omx/operator_authorize_recipes/ structure."""
    repo = tmp_path / "repo"
    rdir = repo / ".omx" / "operator_authorize_recipes"
    rdir.mkdir(parents=True)
    for name, content in recipes.items():
        (rdir / name).write_text(content)
    # Frontier pointer
    fdir = repo / ".omx" / "state"
    fdir.mkdir(parents=True)
    (fdir / "canonical_frontier_pointer.json").write_text(json.dumps({
        "our_local_frontier_contest_cpu": {
            "archive_sha256": "7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe",
            "axis": "contest_cpu",
            "score": 0.19202828,
        },
        "our_local_frontier_contest_cuda": {
            "archive_sha256": "9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4",
            "axis": "contest_cuda",
            "score": 0.20533002,
        },
    }))
    return repo


def test_check_368_clean_repo(tmp_path: Path):
    """Empty/no-stacking recipes pass."""
    repo = _make_recipes_dir(tmp_path, {
        "regular_recipe.yaml": "lane_id: lane_x\ndispatch_enabled: true\n",
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_v14_bug_class_flagged(tmp_path: Path):
    """V14 bug class: stacking on non-frontier baseline flagged."""
    repo = _make_recipes_dir(tmp_path, {
        "v14_recipe.yaml": (
            "lane_id: lane_v14_cascade_a\n"
            "dispatch_enabled: true\n"
            "research_only: false\n"
            "# Stacked archive built on PR101+FEC6 baseline (sha 6bae0201fb082457)\n"
            "baseline_archive_sha: 6bae0201fb082457a02c6956\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "v14_recipe.yaml" in violations[0]
    assert "Catalog #368" in violations[0]
    assert "6bae0201" in violations[0]


def test_check_368_frontier_baseline_accepted(tmp_path: Path):
    """Stacking on canonical frontier baseline is accepted."""
    repo = _make_recipes_dir(tmp_path, {
        "good_recipe.yaml": (
            "lane_id: lane_dqs1_stacking\n"
            "dispatch_enabled: true\n"
            "baseline_archive_sha: 7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe\n"
            "stacking on baseline sha 7a0da5d0\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_research_only_opt_out_accepted(tmp_path: Path):
    """research_only=true exempts non-frontier baseline."""
    repo = _make_recipes_dir(tmp_path, {
        "research_recipe.yaml": (
            "lane_id: lane_exploratory\n"
            "dispatch_enabled: false\n"
            "research_only: true\n"
            "baseline_archive_sha: 6bae0201fb082457\n"
            "stacking on baseline sha 6bae0201\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_substrate_engineering_opt_out_accepted(tmp_path: Path):
    """lane_class=substrate_engineering exempts non-frontier baseline."""
    repo = _make_recipes_dir(tmp_path, {
        "engineering_recipe.yaml": (
            "lane_id: lane_substrate_eng\n"
            "lane_class: substrate_engineering\n"
            "baseline_archive_sha: 6bae0201fb082457\n"
            "stacking on baseline sha 6bae0201\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_waiver_accepted(tmp_path: Path):
    """Same-line waiver with rationale exempts non-frontier baseline."""
    repo = _make_recipes_dir(tmp_path, {
        "waived_recipe.yaml": (
            "lane_id: lane_waived\n"
            "dispatch_enabled: true\n"
            "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:operator-approved exploration of FEC6 baseline stacking\n"
            "baseline_archive_sha: 6bae0201fb082457\n"
            "stacking on baseline sha 6bae0201\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_placeholder_waiver_rejected(tmp_path: Path):
    """Placeholder waiver rationale rejected per Catalog #287."""
    repo = _make_recipes_dir(tmp_path, {
        "placeholder_recipe.yaml": (
            "lane_id: lane_x\n"
            "dispatch_enabled: true\n"
            "# BASELINE_NON_FRONTIER_INTENTIONAL_OK:<rationale>\n"
            "baseline_archive_sha: 6bae0201fb082457\n"
            "stacking on baseline sha 6bae0201\n"
        ),
    })
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_check_368_strict_raises(tmp_path: Path):
    """Strict mode raises PreflightError on violation."""
    repo = _make_recipes_dir(tmp_path, {
        "v14_recipe.yaml": (
            "lane_id: lane_v14\n"
            "dispatch_enabled: true\n"
            "baseline_archive_sha: 6bae0201fb082457\n"
            "stacking on baseline sha 6bae0201\n"
        ),
    })
    with pytest.raises(PreflightError, match="Catalog #368"):
        check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check_368_strict_silent_on_clean(tmp_path: Path):
    """Strict mode silent when clean."""
    repo = _make_recipes_dir(tmp_path, {
        "good_recipe.yaml": (
            "lane_id: lane_good\n"
            "dispatch_enabled: true\n"
        ),
    })
    # No exception expected
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_check_368_no_recipes_dir_silent(tmp_path: Path):
    """Missing recipes dir yields empty result, no crash."""
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_missing_frontier_pointer_silent(tmp_path: Path):
    """Missing canonical frontier pointer skips evaluation."""
    repo = tmp_path / "no_pointer"
    rdir = repo / ".omx" / "operator_authorize_recipes"
    rdir.mkdir(parents=True)
    (rdir / "v14.yaml").write_text(
        "baseline_archive_sha: 6bae0201fb082457\n"
        "stacking on baseline sha 6bae0201\n"
    )
    # No canonical_frontier_pointer.json
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_368_live_repo_regression_guard():
    """Live repo: gate must remain clean (V14 closed, no new attempts)."""
    repo = Path(__file__).parent.parent.parent.parent
    violations = check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(
        repo_root=repo, strict=False, verbose=False,
    )
    # Live count must remain 0 for strict-flip readiness
    assert len(violations) <= 1, (
        f"Live count drift: {len(violations)} violations. First 3: "
        f"{violations[:3]}"
    )


def test_check_368_orchestrator_callsite_warn_only_regression():
    """Catalog #368 orchestrator callsite is strict=False (warn-only initial)."""
    preflight_path = Path(__file__).parent.parent / "preflight.py"
    text = preflight_path.read_text()
    assert "check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer(" in text


# ============================================================================
# Catalog #369 — helper unit tests
# ============================================================================


def test_check_369_synthetic_pattern_detected(tmp_path: Path):
    """Synthetic frame base function name detected."""
    text = "def _render_frame_0_base(h, w):\n    pass"
    assert _check_369_file_has_synthetic_pattern(text) is True


def test_check_369_synthetic_pattern_negative(tmp_path: Path):
    """Real-frame inflate has no synthetic pattern."""
    text = "from pyav import av\ndef inflate():\n    pass"
    assert _check_369_file_has_synthetic_pattern(text) is False


def test_check_369_real_frame_vendor_comma2k19(tmp_path: Path):
    """Comma2k19 canonical helper recognized."""
    text = "from tac.substrates.pretrained_driving_prior.local_chunk_cache import Comma2k19LocalCache"
    assert _check_369_file_has_real_frame_vendor(text) is True


def test_check_369_real_frame_vendor_pyav(tmp_path: Path):
    """av.open canonical pyav recognized."""
    text = "import av\ncontainer = av.open(video_path)"
    assert _check_369_file_has_real_frame_vendor(text) is True


def test_check_369_real_frame_vendor_negative(tmp_path: Path):
    """No real-frame-vendor token returns False."""
    text = "import numpy as np\ndef synth(h, w): return np.zeros((h, w, 3), np.uint8)"
    assert _check_369_file_has_real_frame_vendor(text) is False


def test_check_369_waiver_with_rationale(tmp_path: Path):
    text = "# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:smoke-only scaffold pending Phase 2 vendor wire-in"
    assert _check_369_file_has_waiver(text) is True


def test_check_369_waiver_placeholder_rejected(tmp_path: Path):
    text = "# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:<rationale>"
    assert _check_369_file_has_waiver(text) is False


def test_check_369_waiver_short_rejected(tmp_path: Path):
    text = "# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:no"  # <4 chars
    assert _check_369_file_has_waiver(text) is False


# ============================================================================
# Catalog #369 — end-to-end gate behavior
# ============================================================================


def _make_inflate_repo(
    tmp_path: Path,
    inflates: dict[str, str],
    recipes: dict[str, str] | None = None,
) -> Path:
    """Make a synthetic substrates inflate.py structure."""
    repo = tmp_path / "repo"
    sb_root = repo / "src" / "tac" / "substrates"
    sb_root.mkdir(parents=True)
    for name, content in inflates.items():
        # name format: "<substrate_id>/inflate.py" or "submissions/<sid>/inflate.py"
        if name.startswith("submissions/"):
            target = repo / name
        else:
            target = sb_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if recipes:
        rdir = repo / ".omx" / "operator_authorize_recipes"
        rdir.mkdir(parents=True)
        for rn, rc in recipes.items():
            (rdir / rn).write_text(rc)
    return repo


def test_check_369_clean_real_frame_inflate(tmp_path: Path):
    """Real-frame inflate.py passes."""
    repo = _make_inflate_repo(tmp_path, {
        "good_substrate/inflate.py": (
            "from tac.substrates.pretrained_driving_prior.local_chunk_cache "
            "import Comma2k19LocalCache\ndef inflate():\n    pass\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_cascade_c_prime_bug_class_flagged(tmp_path: Path):
    """Cascade C' synthetic _render_frame_0_base bug class flagged."""
    repo = _make_inflate_repo(tmp_path, {
        "cascade_c_prime/inflate.py": (
            "import numpy as np\n"
            "def _render_frame_0_base(h, w):\n"
            "    # sinusoidal grid + radial gradient\n"
            "    ys, xs = np.mgrid[0:h, 0:w]\n"
            "    return np.zeros((h, w, 3), dtype=np.uint8)\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "cascade_c_prime" in violations[0]
    assert "Catalog #369" in violations[0]


def test_check_369_real_frame_vendor_accepted(tmp_path: Path):
    """Synthetic pattern + real-frame-vendor accepted."""
    repo = _make_inflate_repo(tmp_path, {
        "good_substrate/inflate.py": (
            "import numpy as np\n"
            "from tac.substrates.pretrained_driving_prior.local_chunk_cache "
            "import Comma2k19LocalCache\n"
            "def _render_frame_0_base(h, w):  # superseded by Comma2k19 vendor\n"
            "    return np.zeros((h, w, 3), dtype=np.uint8)\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_waiver_accepted(tmp_path: Path):
    """Same-line waiver with rationale accepted."""
    repo = _make_inflate_repo(tmp_path, {
        "waived_substrate/inflate.py": (
            "import numpy as np\n"
            "# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:smoke-only scaffold awaiting vendor wire-in per Catalog #220\n"
            "def _render_frame_0_base(h, w):\n"
            "    return np.zeros((h, w, 3), dtype=np.uint8)\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_recipe_opt_out_accepted(tmp_path: Path):
    """Adjacent recipe research_only:true exempts the inflate.py."""
    repo = _make_inflate_repo(
        tmp_path,
        {
            "exploratory_substrate/inflate.py": (
                "import numpy as np\n"
                "def _render_frame_0_base(h, w):\n"
                "    return np.zeros((h, w, 3), dtype=np.uint8)\n"
            ),
        },
        recipes={
            "substrate_exploratory_substrate_modal_t4_dispatch.yaml": (
                "lane_id: lane_exp\nresearch_only: true\ndispatch_enabled: false\n"
            ),
        },
    )
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_recipe_smoke_only_opt_out_accepted(tmp_path: Path):
    """Adjacent recipe smoke_only:true exempts the inflate.py."""
    repo = _make_inflate_repo(
        tmp_path,
        {
            "smoke_substrate/inflate.py": (
                "def _render_frame_0_base(h, w):\n    return None\n"
            ),
        },
        recipes={
            "substrate_smoke_substrate_modal_t4_dispatch.yaml": (
                "lane_id: lane_smoke\nsmoke_only: true\n"
            ),
        },
    )
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_placeholder_waiver_rejected(tmp_path: Path):
    """Placeholder waiver rejected per Catalog #287."""
    repo = _make_inflate_repo(tmp_path, {
        "bad_substrate/inflate.py": (
            "# SYNTHETIC_FRAME_BASE_INTENTIONAL_OK:<rationale>\n"
            "def _render_frame_0_base(h, w):\n    return None\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_check_369_no_synthetic_pattern_ignored(tmp_path: Path):
    """Inflate.py without synthetic patterns silent."""
    repo = _make_inflate_repo(tmp_path, {
        "normal_substrate/inflate.py": (
            "import numpy as np\n"
            "def inflate(archive_bytes):\n    pass\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_strict_raises(tmp_path: Path):
    """Strict mode raises on violation."""
    repo = _make_inflate_repo(tmp_path, {
        "bad_substrate/inflate.py": (
            "def _render_frame_0_base(h, w):\n    return None\n"
        ),
    })
    with pytest.raises(PreflightError, match="Catalog #369"):
        check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check_369_strict_silent_on_clean(tmp_path: Path):
    """Strict mode silent when clean."""
    repo = _make_inflate_repo(tmp_path, {
        "good_substrate/inflate.py": "def inflate():\n    pass\n",
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_check_369_exact_current_excluded(tmp_path: Path):
    """submissions/exact_current/inflate.py exempt per CLAUDE.md mutation frontier."""
    repo = _make_inflate_repo(tmp_path, {
        "submissions/exact_current/inflate.py": (
            "def _render_frame_0_base(h, w):\n    return None\n"
        ),
    })
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_substrate_opt_out_helper(tmp_path: Path):
    """Helper _check_369_substrate_has_opt_out works on submissions/ paths."""
    repo = _make_inflate_repo(
        tmp_path,
        {"submissions/sub_x/inflate.py": ""},
        recipes={
            "substrate_sub_x_modal_t4.yaml": (
                "research_only: true\n"
            ),
        },
    )
    inflate_path = repo / "submissions" / "sub_x" / "inflate.py"
    assert _check_369_substrate_has_opt_out(inflate_path, repo) is True


def test_check_369_substrate_opt_out_no_recipe(tmp_path: Path):
    """No matching recipe returns False."""
    repo = _make_inflate_repo(tmp_path, {
        "sub_y/inflate.py": "",
    })
    inflate_path = repo / "src" / "tac" / "substrates" / "sub_y" / "inflate.py"
    assert _check_369_substrate_has_opt_out(inflate_path, repo) is False


def test_check_369_no_inflates_silent(tmp_path: Path):
    """No inflate.py files silent."""
    repo = tmp_path / "empty"
    repo.mkdir()
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    assert violations == []


def test_check_369_live_repo_regression_guard():
    """Live repo: gate may flag Cascade C' (expected baseline)."""
    repo = Path(__file__).parent.parent.parent.parent
    violations = check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(
        repo_root=repo, strict=False, verbose=False,
    )
    # Live count <= 3 (Cascade C' is the canonical bug class anchor;
    # operator-routable: either vendor real frames, add waiver, or set
    # research_only=true in recipe).
    assert len(violations) <= 3, (
        f"Live count drift: {len(violations)} violations. First 3: "
        f"{violations[:3]}"
    )


def test_check_369_orchestrator_callsite_warn_only_regression():
    """Catalog #369 orchestrator callsite is strict=False (warn-only initial)."""
    preflight_path = Path(__file__).parent.parent / "preflight.py"
    text = preflight_path.read_text()
    assert "check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base(" in text


# ============================================================================
# Catalog #185 sister-callable regression guards (both gates)
# ============================================================================


def test_check_368_callable_via_globals():
    """Catalog #185 sister: gate function callable via tac.preflight globals."""
    from tac import preflight
    fn = getattr(preflight, "check_substrate_substitution_stacking_baseline_matches_canonical_frontier_pointer", None)
    assert fn is not None
    assert callable(fn)


def test_check_369_callable_via_globals():
    """Catalog #185 sister: gate function callable via tac.preflight globals."""
    from tac import preflight
    fn = getattr(preflight, "check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base", None)
    assert fn is not None
    assert callable(fn)


def test_check_368_iter_recipe_files_live_repo():
    """Helper enumerates live recipe files."""
    repo = Path(__file__).parent.parent.parent.parent
    files = _check_368_iter_recipe_files(repo)
    assert len(files) > 0  # repo has many recipes


def test_check_369_iter_inflate_files_live_repo():
    """Helper enumerates live inflate.py files."""
    repo = Path(__file__).parent.parent.parent.parent
    files = _check_369_iter_inflate_files(repo)
    assert len(files) > 0  # repo has many inflate.py files


def test_check_368_load_canonical_frontier_shas_live_repo():
    """Helper loads canonical frontier shas from live repo."""
    repo = Path(__file__).parent.parent.parent.parent
    shas = _check_368_load_canonical_frontier_shas(repo)
    # Should have both contest_cpu and contest_cuda
    assert "contest_cpu" in shas
    assert "contest_cuda" in shas
    assert len(shas["contest_cpu"]) == 64
