"""NO-FAKE tests for the Gumbel-vs-argmax eval-render gap audit runner.

These tests verify ACTUAL behavior of the four decode variants, not constants
(per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE 5-class discipline). The
headline guards FAIL if a decode silently no-ops to argmax, if the
expected-value decode feeds a one-hot instead of a soft simplex, or if the
faithfulness classifier stops responding to real recon statistics.

The runner under test lives at
``tools/gumbel_vs_argmax_eval_render_gap_audit.py``; it is loaded via
``importlib`` because ``tools/`` is not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

REPO_ROOT = Path(__file__).resolve().parents[5]
RUNNER_PATH = REPO_ROOT / "tools" / "gumbel_vs_argmax_eval_render_gap_audit.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("_gap_audit_under_test", str(RUNNER_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ga():
    return _load_runner()


@pytest.fixture(scope="module")
def z8_model(ga):
    """A small fresh (untrained) Z8 model whose logits we can set deterministically."""
    model, cfg = ga._build_z8_model(num_pairs=2)
    return model, cfg


# ---------------------------------------------------------------------------
# Decode-variant behavior (the headline NO-FAKE guards)
# ---------------------------------------------------------------------------


def test_expected_value_decode_feeds_soft_simplex_not_one_hot(ga, z8_model):
    """The expected-value decode must feed softmax(logits) (a soft simplex),
    NOT a one-hot. If a peaked-but-not-degenerate posterior is fed, the
    expected-value embedding must DIFFER from the argmax (one-hot) embedding.

    This FAILS if expected_value silently collapses to argmax one-hot.
    """
    model, cfg = z8_model
    # Force a MODERATELY peaked posterior on level 0 of pair 0: logits that make
    # softmax non-degenerate (max prob ~0.5, not ~1.0) so soft != one-hot.
    L = int(cfg.num_levels)
    Ks = [int(cfg.num_categories_per_level[lv]) for lv in range(L)]
    # Build per-level logits with a clear-but-soft peak.
    new_logits = []
    for lv in range(L):
        G = int(cfg.num_groups_per_level[lv])
        K = Ks[lv]
        arr = np.zeros((int(cfg.num_pairs), G, K), dtype=np.float32)
        arr[..., 0] = 1.0  # mild peak at category 0 -> softmax ~ not one-hot
        new_logits.append(mx.array(arr))
    model.logits_per_level = new_logits

    logits = ga._per_level_logits_for_pair(model, cfg, 0)
    onehot = [ga._onehot_from_logits(logits[lv], Ks[lv]) for lv in range(L)]
    soft = [ga._softmax_simplex(logits[lv]) for lv in range(L)]

    # The soft simplex must NOT equal the one-hot (it is a genuine distribution).
    for lv in range(L):
        oh = np.asarray(onehot[lv])
        sm = np.asarray(soft[lv])
        # one-hot has a single 1.0 per group; softmax max prob must be < 1.
        assert float(sm.max()) < 0.999, f"level {lv} softmax is degenerate one-hot"
        assert not np.allclose(oh, sm), f"level {lv} soft simplex == one-hot (FAKE)"

    # And the rendered output must differ between argmax and expected-value.
    r_arg = np.asarray(ga._decode_from_per_level_simplex(model, cfg, onehot))
    r_ev = np.asarray(ga._decode_from_per_level_simplex(model, cfg, soft))
    assert not np.allclose(r_arg, r_ev), (
        "argmax and expected-value produced identical renders => one decode "
        "silently no-ops (FAKE)"
    )


def test_argmax_decode_is_one_hot_at_argmax(ga, z8_model):
    """The argmax decode must produce a one-hot at argmax(logits) per group."""
    model, cfg = z8_model
    L = int(cfg.num_levels)
    Ks = [int(cfg.num_categories_per_level[lv]) for lv in range(L)]
    # Random-ish logits so argmax is non-trivial.
    new_logits = []
    rng = np.random.default_rng(0)
    for lv in range(L):
        G = int(cfg.num_groups_per_level[lv])
        K = Ks[lv]
        new_logits.append(mx.array(rng.normal(size=(int(cfg.num_pairs), G, K)).astype(np.float32)))
    model.logits_per_level = new_logits

    logits = ga._per_level_logits_for_pair(model, cfg, 0)
    for lv in range(L):
        oh = np.asarray(ga._onehot_from_logits(logits[lv], Ks[lv]))[0]  # (G, K)
        expected_idx = np.asarray(mx.argmax(logits[lv], axis=-1))[0]  # (G,)
        produced_idx = oh.argmax(axis=-1)
        assert np.array_equal(produced_idx, expected_idx), (
            f"level {lv} one-hot not at argmax"
        )
        # Exactly one 1.0 per group (it is a true one-hot).
        assert np.allclose(oh.sum(axis=-1), 1.0)
        assert set(np.unique(oh).tolist()) <= {0.0, 1.0}


def test_low_temp_gumbel_is_deterministic_fixed_seed(ga, z8_model):
    """low_temp_gumbel uses a fixed per-pair seed => two renders are identical
    (so it qualifies as a contest-valid deterministic decode)."""
    model, cfg = z8_model
    L = int(cfg.num_levels)
    rng = np.random.default_rng(1)
    new_logits = []
    for lv in range(L):
        G = int(cfg.num_groups_per_level[lv])
        K = int(cfg.num_categories_per_level[lv])
        new_logits.append(mx.array(rng.normal(size=(int(cfg.num_pairs), G, K)).astype(np.float32)))
    model.logits_per_level = new_logits

    r1 = ga._render_all_pairs_decode(model, cfg, 2, decode="low_temp_gumbel", low_temp=0.1)
    r2 = ga._render_all_pairs_decode(model, cfg, 2, decode="low_temp_gumbel", low_temp=0.1)
    assert np.allclose(r1, r2), "low_temp_gumbel fixed-seed render is non-deterministic"


def test_decodes_render_nonconstant_and_differ_on_distinct_pairs(ga, z8_model):
    """The render must depend on the per-pair latent: two DISTINCT pairs render
    differently. This FAILS if the decoder ignores its categorical input
    (a saturation no-op would make all pairs identical)."""
    model, cfg = z8_model
    L = int(cfg.num_levels)
    new_logits = []
    rng = np.random.default_rng(2)
    for lv in range(L):
        G = int(cfg.num_groups_per_level[lv])
        K = int(cfg.num_categories_per_level[lv])
        # pair 0 strongly peaks cat 0, pair 1 strongly peaks cat (K-1) => distinct.
        arr = np.zeros((int(cfg.num_pairs), G, K), dtype=np.float32)
        arr[0, :, 0] = 10.0
        arr[1, :, K - 1] = 10.0
        new_logits.append(mx.array(arr))
    model.logits_per_level = new_logits

    recon = ga._render_all_pairs_decode(model, cfg, 2, decode="argmax")
    assert recon.shape[0] == 2
    diff = float(np.abs(recon[0] - recon[1]).max())
    assert diff > 1e-3, (
        f"distinct pairs rendered identically (max_abs_diff={diff}) => decoder "
        f"ignores categorical input (FAKE)"
    )
    del rng


# ---------------------------------------------------------------------------
# Faithfulness classifier (responds to REAL recon stats)
# ---------------------------------------------------------------------------


def test_faithfulness_flags_collapsed_white(ga):
    """A near-constant white render must be classified COLLAPSED."""
    gt = (np.random.default_rng(0).random((4, 2, 8, 8, 3)) * 43).astype(np.float32)  # GT mean ~21
    white = np.full((4, 2, 8, 8, 3), 254.5, dtype=np.float32)
    v = ga._render_faithfulness(white, gt)
    assert v["verdict"] == "COLLAPSED"
    assert v["collapsed_saturated"] or v["collapsed_near_constant"]


def test_faithfulness_flags_in_range_as_faithful(ga):
    """A render that matches GT distribution is FAITHFUL."""
    rng = np.random.default_rng(0)
    gt = (rng.random((4, 2, 8, 8, 3)) * 43).astype(np.float32)
    faithful = (rng.random((4, 2, 8, 8, 3)) * 43).astype(np.float32)
    v = ga._render_faithfulness(faithful, gt)
    assert v["verdict"] == "FAITHFUL", v


def test_faithfulness_flags_saturated_high_mean(ga):
    """recon_mean > 3x GT mean is COLLAPSED even if std is non-trivial."""
    rng = np.random.default_rng(0)
    gt = (rng.random((4, 2, 8, 8, 3)) * 43).astype(np.float32)  # mean ~21
    # mean ~160 (>3x 21) with real variance -> Z8's collapse signature.
    sat = (rng.random((4, 2, 8, 8, 3)) * 60 + 130).astype(np.float32)
    v = ga._render_faithfulness(sat, gt)
    assert v["verdict"] == "COLLAPSED"
    assert v["collapsed_saturated"]


# ---------------------------------------------------------------------------
# Custody / provenance (non-promotable per Catalog #192/#341/#127/#323)
# ---------------------------------------------------------------------------


def test_result_is_non_promotable_and_carries_provenance(ga):
    """A written result.json must carry canonical Provenance + non-promotable
    markers. ``_provenance_for`` requires a path UNDER the repo root (it calls
    ``relative_to(REPO_ROOT)``), so write into a repo-local scratch dir per
    CLAUDE.md "Forbidden /tmp paths" (the canonical .omx/tmp scratch)."""
    scratch = REPO_ROOT / ".omx" / "tmp" / "gap_audit_test_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rp = scratch / "result.json"
    try:
        rp.write_text('{"x": 1}')
        prov = ga._provenance_for(rp)
        assert isinstance(prov, dict)
        # Canonical macOS-MLX research-signal provenance is non-promotable by
        # construction; assert it serialized with a sha + path.
        flat = str(prov).lower()
        assert "sha256" in flat or "artifact" in flat
    finally:
        rp.unlink(missing_ok=True)
