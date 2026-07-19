# SPDX-License-Identifier: MIT
"""Tests for tac.witness_dsl.spec_v10_structured_init_defaults (SPEC_v10 P1, C3).

Behavior-verifying: the structured-init defaults are ALL default-on (P1: seeded statics are
birth defaults, not opt-in flags); machinery + self-detect entries actually import; the
fail-closed blocker fires on a missing required seed artifact; and the fold surface composes.
"""

from __future__ import annotations

from pathlib import Path

from tac.witness_dsl import spec_v10_structured_init_defaults as sid

REPO = "/Users/adpena/Projects/pact"


def test_defaults_are_all_birth_on():
    defs = sid.structured_init_defaults()
    assert len(defs) == 5
    # P1: seeded statics are the DEFAULT of the v10 compile path (not opt-in).
    assert all(d.default_on for d in defs)
    keys = {d.key for d in defs}
    assert keys == {"hood_static", "sky_undrivable_static", "lane_polynomial",
                    "per_dash_anchors", "hood_tex_seed"}


def test_machinery_modules_import_and_self_detect_present():
    st = sid.structured_init_status(REPO)
    assert all(st.machinery_present.values()), st.machinery_present
    assert all(st.self_detect_present.values()), st.self_detect_present


def test_every_default_names_a_self_detect_entry_not_a_class_index():
    # NEVER hardcode a class index — each default names a class-SELF-DETECTING entry point.
    for d in sid.structured_init_defaults():
        assert d.self_detect_entry
        assert isinstance(d.self_detect_entry, str)


def test_fail_closed_blocker_on_missing_required_seed():
    st = sid.structured_init_status(REPO)
    # hood_tex_seed is a REAL required counted artifact (may be absent on this host):
    # its absence MUST produce exactly one typed blocker (fail-closed, like the spec skeleton).
    seed_blockers = [b for b in st.blockers if b["id"].startswith("structured_init_seed:")]
    hood_present = (Path(REPO) /
                    "experiments/results/necessity_dseg_calibration_20260715/hood_tex_seed.npz").exists()
    if hood_present:
        assert not seed_blockers
    else:
        assert any(b["id"] == "structured_init_seed:hood_tex_seed" for b in seed_blockers)


def test_blockers_fold_surface_matches_status():
    blockers = sid.structured_init_blockers(REPO)
    st = sid.structured_init_status(REPO)
    assert blockers == st.blockers


def test_optional_seed_defaults_do_not_block():
    # defaults with seed_artifact=None or seed_required=False must never block on a seed.
    for d in sid.structured_init_defaults():
        if d.seed_artifact is None or not d.seed_required:
            ids = [b["id"] for b in sid.structured_init_blockers(REPO)]
            assert f"structured_init_seed:{d.key}" not in ids


def test_missing_machinery_produces_blocker(monkeypatch, tmp_path):
    # If a machinery module cannot import, a fail-closed blocker fires (never silent).
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "tac.boundary_math.hood_static_component":
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    st = sid.structured_init_status(REPO)
    assert any(b["id"].startswith("structured_init_machinery:") for b in st.blockers)
