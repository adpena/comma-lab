# SPDX-License-Identifier: MIT
"""FIX-ALL WAVE A trainer-side tests (levelset witness trainer).

Covers:
  * F2 resume-safety: ``_resume_lever_divergences`` FAIL-CLOSED logic (the render-side lever-drift
    guard) + the wired raise + the --resume-allow-lever-drift escape (source-level).
  * F4 (#222): the --adam-beta2 flag exists, is threaded into the MLX AdamW construction(s), and the
    default is byte-identical (0.999 == MLX default); + a small MLX AdamW betas smoke (finite update).
  * F3: _bnd_band is OR'd into _stage_boundary_now (band engagement gets the stage-transition treatment).
  * review MED-1: ``__cfg_film_stiefel`` is in the resume-lever-drift checks (the param-key guard
    cannot see it — film_stiefel constrains the EXISTING film.weight, no new keys).
  * review MED-3: ``_validate_aa_compose_compat`` fail-closed pure fn (AA supersample vs the
    base-grid composers: lane band / residual bulk / island seed) + its wired call site.

Pure-CPU: the trainer module imports fast (no GPU); the MLX smoke is a 2-element one-step AdamW update.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_TRAINER_PATH = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO / "experiments"))


def _load_trainer():
    spec = importlib.util.spec_from_file_location("tl_fixall_wave_a", str(_TRAINER_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TRAINER_SRC = _TRAINER_PATH.read_text()


def _args(**over):
    """A minimal args namespace with the F2-tracked lever defaults (== trainer argparse defaults)."""
    base = dict(
        mod_dim=19, lane_render_band=False, lane_band_start_epoch=300,
        persistence_loss_weight=0.0, amplify_weight=0.0, render_aa="none", hosc_beta_end=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _cfg(**over):
    """A resume sidecar cfg dict (the ``__cfg_*`` keys _build_resume_state_arrays persists)."""
    base = {
        "__cfg_mod_dim": 19, "__cfg_lane_render_band": 0, "__cfg_lane_band_start_epoch": 300,
        "__cfg_persistence_loss_weight": 0.0, "__cfg_amplify_weight": 0.0,
        "__cfg_render_aa": "none", "__cfg_hosc_beta_end": -1.0,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# F2: _resume_lever_divergences fail-closed logic
# --------------------------------------------------------------------------
def test_resume_no_divergence_when_matching():
    tl = _load_trainer()
    assert tl._resume_lever_divergences(_cfg(), _args()) == []


@pytest.mark.parametrize("cfg_over,arg_over,key", [
    ({"__cfg_lane_render_band": 1}, {"lane_render_band": False}, "lane_render_band"),
    ({"__cfg_persistence_loss_weight": 1.0}, {"persistence_loss_weight": 0.0}, "persistence_loss_weight"),
    ({"__cfg_amplify_weight": 1.0}, {"amplify_weight": 0.0}, "amplify_weight"),
    ({"__cfg_render_aa": "ipe"}, {"render_aa": "none"}, "render_aa"),
    ({"__cfg_hosc_beta_end": 4.0}, {"hosc_beta_end": None}, "hosc_beta_end"),
    ({"__cfg_mod_dim": 26}, {"mod_dim": 19}, "mod_dim"),
])
def test_resume_flags_each_lever_drop(cfg_over, arg_over, key):
    tl = _load_trainer()
    div = tl._resume_lever_divergences(_cfg(**cfg_over), _args(**arg_over))
    assert any(key in d for d in div), f"{key} divergence not flagged: {div}"


def test_resume_hosc_beta_end_set_matches_no_divergence():
    tl = _load_trainer()
    # both set to 4.0 -> no divergence
    assert tl._resume_lever_divergences(_cfg(__cfg_hosc_beta_end=4.0), _args(hosc_beta_end=4.0)) == []


def test_resume_lane_band_start_epoch_inert_when_band_off_both():
    tl = _load_trainer()
    # band OFF in both, start-epoch differs -> INERT (not flagged; start-epoch does nothing while off)
    div = tl._resume_lever_divergences(
        _cfg(__cfg_lane_band_start_epoch=200), _args(lane_band_start_epoch=300))
    assert div == []


def test_resume_lane_band_start_epoch_flagged_when_band_on():
    tl = _load_trainer()
    div = tl._resume_lever_divergences(
        _cfg(__cfg_lane_render_band=1, __cfg_lane_band_start_epoch=200),
        _args(lane_render_band=True, lane_band_start_epoch=300))
    assert any("lane_band_start_epoch" in d for d in div)


def test_resume_pre_f2_sidecar_backward_compatible():
    tl = _load_trainer()
    # a pre-F2 sidecar lacks the lever keys -> NO spurious divergence (only present keys are checked)
    old_cfg = {"__cfg_n_hidden": 4, "__cfg_hidden_dim": 96}
    assert tl._resume_lever_divergences(old_cfg, _args(lane_render_band=True,
                                                       persistence_loss_weight=1.0)) == []


def test_resume_multiple_divergences_all_listed():
    tl = _load_trainer()
    div = tl._resume_lever_divergences(
        _cfg(__cfg_lane_render_band=1, __cfg_persistence_loss_weight=1.0, __cfg_mod_dim=26),
        _args(lane_render_band=False, persistence_loss_weight=0.0, mod_dim=19))
    assert len(div) == 3


def test_resume_guard_wired_fail_closed_with_escape():
    # source-level: the guard is actually WIRED into the resume block (raise) + the escape flag exists.
    assert 'ap.add_argument("--resume-allow-lever-drift"' in _TRAINER_SRC
    assert "_resume_lever_divergences(resume_cfg, args)" in _TRAINER_SRC
    assert "resume_allow_lever_drift" in _TRAINER_SRC
    # the divergence path raises (fail-closed), not warns
    guard = _TRAINER_SRC[_TRAINER_SRC.index("_lever_div = _resume_lever_divergences"):]
    assert "raise ValueError(" in guard[:900]
    # the render-side lever cfg keys are persisted into the resume sidecar
    for k in ("__cfg_lane_render_band", "__cfg_persistence_loss_weight", "__cfg_amplify_weight",
              "__cfg_lane_band_start_epoch", "__cfg_render_aa", "__cfg_hosc_beta_end"):
        assert f'out["{k}"]' in _TRAINER_SRC, f"{k} not persisted in the resume sidecar"


# --------------------------------------------------------------------------
# review MED-1: __cfg_film_stiefel resume-drift guard (no param keys -> lever check must catch it)
# --------------------------------------------------------------------------
def test_resume_film_stiefel_drop_flagged():
    tl = _load_trainer()
    # trained WITH --film-stiefel, resume argv drops it -> flagged
    div = tl._resume_lever_divergences(_cfg(__cfg_film_stiefel=1), _args(film_stiefel=False))
    assert any("film_stiefel" in d for d in div), f"film_stiefel drop not flagged: {div}"
    # trained WITHOUT, resume argv adds it -> flagged (constraint added onto a foreign trajectory)
    div = tl._resume_lever_divergences(_cfg(__cfg_film_stiefel=0), _args(film_stiefel=True))
    assert any("film_stiefel" in d for d in div), f"film_stiefel add not flagged: {div}"


def test_resume_film_stiefel_matching_and_pre_fix_sidecar_no_divergence():
    tl = _load_trainer()
    # both ON -> no divergence
    assert tl._resume_lever_divergences(_cfg(__cfg_film_stiefel=1), _args(film_stiefel=True)) == []
    # pre-fix sidecar lacks the key -> NO spurious divergence (only present keys are checked)
    assert tl._resume_lever_divergences(_cfg(), _args(film_stiefel=True)) == []
    # sidecar persists the key (source-level; the R2a-MED-1 persistence this check reads)
    assert 'out["__cfg_film_stiefel"]' in _TRAINER_SRC


# --------------------------------------------------------------------------
# review MED-3 -> #220 UNBLOCK (2026-07-07): _validate_aa_compose_compat pure fn + wired call site.
# compose-after-downsample landed in aa_sdf_observation_render (compose_fn now runs at the BASE
# grid, after box_downsample), so the three tracked base-grid composers COMPOSE with AA
# supersample — the guard accepts every tracked combination (kept, same signature/call site, as
# the fail-closed home for any future fine-grid-only composer).
# --------------------------------------------------------------------------
def test_aa_compose_compat_noop_when_aa_off():
    tl = _load_trainer()
    # AA off => always compatible, even with every base-grid composer engaged
    tl._validate_aa_compose_compat(False, True, True, True)


def test_aa_compose_compat_ok_when_aa_on_no_composers():
    tl = _load_trainer()
    tl._validate_aa_compose_compat(True, False, False, False)


@pytest.mark.parametrize("band,residual,seed", [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, True),
])
def test_aa_compose_compat_accepts_base_grid_composers_post_220_unblock(band, residual, seed):
    # #220 unblock: compose_fn runs AFTER box_downsample (base grid) inside
    # render_aa_{batch_,}through_R_mlx, so band/residual/seed compose with AA by construction.
    tl = _load_trainer()
    tl._validate_aa_compose_compat(True, band, residual, seed)  # must NOT raise


def test_aa_compose_compat_wired_at_render_path():
    # source-level: the pure validator is actually CALLED at the unified-render-path site with all
    # three base-grid composers threaded (band + residual + seed).
    # the indented occurrence is the CALL (the def is at column 0 with a "def " prefix)
    call = _TRAINER_SRC[_TRAINER_SRC.index("\n    _validate_aa_compose_compat("):]
    head = call[:300]
    assert "_aa_on" in head and "_band_active" in head and "residual_mode" in head \
        and "seed_islands" in head, head


# --------------------------------------------------------------------------
# F4 (#222): --adam-beta2 flag + threading + byte-identical default
# --------------------------------------------------------------------------
def test_adam_beta2_flag_declared_default_bit_identical():
    m = re.search(r'add_argument\(\s*"--adam-beta2",\s*type=float,\s*default=([0-9.]+)', _TRAINER_SRC)
    assert m, "--adam-beta2 not declared as a float argparse flag"
    assert float(m.group(1)) == pytest.approx(0.999), "default must be 0.999 (== MLX AdamW default)"


def test_adam_beta2_threaded_into_adamw_constructions():
    # every optim.AdamW(...) that trains the witness must thread betas=[0.9, adam_beta2].
    # (structured-init pretrain AdamW is a separate short phase; the main + reset opts are the scope.)
    assert _TRAINER_SRC.count('betas=[0.9, float(getattr(args, "adam_beta2"') >= 2, \
        "adam_beta2 must be threaded into the main + stage-transition-reset AdamW constructions"


def test_mlx_adamw_betas_smoke_finite_step():
    mx = pytest.importorskip("mlx.core")
    import mlx.optimizers as optim
    # a 2-element param, one AdamW step with the all-levers extreme beta2 -> finite, moved param.
    p = {"w": mx.array([1.0, -2.0])}
    g = {"w": mx.array([0.5, -0.5])}
    opt = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4, betas=[0.9, 0.9999999])
    opt.init(p)
    new = opt.apply_gradients(g, p)
    mx.eval(new["w"])
    import numpy as np
    arr = np.asarray(new["w"])
    assert np.all(np.isfinite(arr)), "extreme beta2 must still produce a finite AdamW update"
    assert not np.allclose(arr, np.asarray(p["w"])), "the param must actually move"


# --------------------------------------------------------------------------
# F3: _bnd_band OR'd into _stage_boundary_now
# --------------------------------------------------------------------------
def test_bnd_band_in_stage_boundary():
    # (updated for the _lever_epoch remap: band engagement keys off the LEVER epoch, not raw ep)
    assert "_bnd_band = (_band_active and (_lever_epoch(ep) >= _band_start) and not band_gate[" \
        in _TRAINER_SRC
    m = re.search(r"_stage_boundary_now = \((.*?)\)", _TRAINER_SRC, re.S)
    assert m and "_bnd_band" in m.group(1), "_bnd_band must be OR'd into _stage_boundary_now"
    # the moment-reset telemetry records the band-engage cause
    assert '"from_lane_render_band_engage": bool(_bnd_band)' in _TRAINER_SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
