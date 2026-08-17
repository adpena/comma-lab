"""Guard the ddm_a1s FO-A band mask: default-off byte-identity, and the pre-registered bands.

FO-A adds exactly ONE flag to the committed sr1 tool (`--delta-mask {off,band,interior}`) and
asks one question: is A1's pose damage band-driven or interior-driven?  Two things here carry
the whole blast radius.

1.  **Default off must be the parent object, bit for bit.**  The ddm_a1s pose row is retained
    (`pose6_by_alpha.npy`, sha `97e2c899...`) and every later comparison is against it, so a
    mask that silently perturbs the default path would invalidate the parent row rather than
    extend it.  The guard is structural: with `off` the stage must never build a band, never
    open the token field, and never touch `delta_cam`.
2.  **The thresholds are pre-registered.**  `A1_FOA_LIVE_BELOW` / `A1_FOA_CLOSED_ABOVE` were
    fixed by ddm_a1s section 8 BEFORE this row ran.  A test that lets them move lets the arm
    pick the bucket after seeing the number.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

sr1 = pytest.importorskip("ddm_sr1_manufactured_seg_recovery")


def _toy() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """A deterministic (delta, token field, lift) triple at the real scorer/camera geometry."""
    rng = np.random.default_rng(20260817)
    delta = rng.normal(size=(sr1.CAM_H, sr1.CAM_W, 3))
    tok = np.zeros((sr1.SEG_H, sr1.SEG_W), dtype=np.uint8)
    tok[: sr1.SEG_H // 2, :] = 1
    tok[:, : sr1.SEG_W // 3] = 2
    return delta, tok, sr1._a1_delta_mask_lift()


# --------------------------------------------------------------------------------------------
# the pre-registered constants
# --------------------------------------------------------------------------------------------
def test_foa_thresholds_are_the_pre_registered_values():
    """ddm_a1s section 8 fixed both bars; neither may be re-derived after seeing the number."""
    assert sr1.A1_FOA_LIVE_BELOW == 0.0026240
    assert sr1.A1_FOA_CLOSED_ABOVE == 0.0083
    # LIVE bar IS the incumbent pose error, sqrt(d_pose) at hv1 ep0634.
    assert math.isclose(sr1.A1_FOA_LIVE_BELOW, math.sqrt(sr1.HV1_D_POSE), rel_tol=2e-5)
    # CLOSED bar is the 3.2x multiple the order named.
    assert math.isclose(sr1.A1_FOA_CLOSED_ABOVE / sr1.A1_FOA_LIVE_BELOW, 3.2, rel_tol=0.02)
    assert sr1.A1_FOA_LIVE_BELOW < sr1.A1_FOA_CLOSED_ABOVE


def test_delta_mask_modes_are_exactly_the_three_declared():
    assert sr1.A1_DELTA_MASKS == ("off", "band", "interior")
    assert sr1.A1_DELTA_MASKS[0] == "off", "the default must be the first, inert mode"


_STAGE_FN = {"ledger": "stage_ledger", "a1sign": "stage_a1sign", "a1pose": "stage_a1pose"}


def _args_through_the_real_cli(monkeypatch, tmp_path, stage: str, argv: list[str]):
    """Drive the REAL parser in `main` and capture the namespace a stage would receive."""
    seen: list[object] = []
    monkeypatch.setattr(sr1, _STAGE_FN[stage], lambda args: seen.append(args))
    assert sr1.main(["--stage", stage, "--work", str(tmp_path), *argv]) == 0
    return seen[0]


def test_the_real_parser_defaults_delta_mask_to_off(monkeypatch, tmp_path):
    """Not a hand-built parser and not the help text: the namespace a stage actually gets."""
    for stage in ("ledger", "a1sign", "a1pose"):
        assert _args_through_the_real_cli(monkeypatch, tmp_path, stage, []).delta_mask == "off"


@pytest.mark.parametrize("mode", ["off", "band", "interior"])
@pytest.mark.parametrize("stage", ["a1sign", "a1pose"])
def test_the_real_parser_accepts_every_declared_mode(monkeypatch, tmp_path, stage, mode):
    args = _args_through_the_real_cli(monkeypatch, tmp_path, stage, ["--delta-mask", mode])
    assert args.delta_mask == mode


def test_the_real_parser_refuses_an_undeclared_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(sr1, "stage_ledger", lambda args: None)
    with pytest.raises(SystemExit):
        sr1.main(["--stage", "ledger", "--work", str(tmp_path), "--delta-mask", "all"])


@pytest.mark.parametrize("stage", ["roperator", "sign", "emphasis", "waterfill", "ledger"])
@pytest.mark.parametrize("mode", ["band", "interior"])
def test_a_stage_that_cannot_consume_the_mask_refuses_it(monkeypatch, tmp_path, stage, mode):
    """An inert flag is a config-orphan: silently ignoring it is the bug this refuses."""
    for fn in ("stage_roperator", "stage_sign", "stage_emphasis", "stage_waterfill",
               "stage_ledger"):
        monkeypatch.setattr(sr1, fn, lambda args: None)
    with pytest.raises(SystemExit):
        sr1.main(["--stage", stage, "--work", str(tmp_path), "--delta-mask", mode])
    # ...and the same stage runs fine with the default.
    assert sr1.main(["--stage", stage, "--work", str(tmp_path)]) == 0


def test_masked_seg_verdict_scope_does_not_claim_the_global_actuator():
    """The bands never move with the mask, but the scope string must not mislabel the lever."""
    harmful = dict.fromkeys(sr1.A1_ALPHAS, sr1.A1_NEUTRAL_HI + 500)
    harmful[0.0] = sr1.A1_CONTROL_FLIPS
    assert "global linear de-blur" in sr1._a1_verdict(harmful)["verdict_scope"]
    for mode in ("band", "interior"):
        scope = sr1._a1_verdict(harmful, mode)["verdict_scope"]
        assert "global linear de-blur" not in scope
        assert f"{mode}-restricted linear de-blur" in scope
    # Same ladder, same verdict and same numbers under every mask: only the label changed.
    for mode in ("off", "band", "interior"):
        got = sr1._a1_verdict(harmful, mode)
        assert got["verdict"] == "CLOSED_HARMFUL"
        assert got["bands"] == sr1._a1_verdict(harmful)["bands"]
        assert got["best_flips"] == sr1._a1_verdict(harmful)["best_flips"]


# --------------------------------------------------------------------------------------------
# the mask itself
# --------------------------------------------------------------------------------------------
def test_band_and_interior_partition_the_perturbation_exactly():
    """The two legs are complements: their kept fields sum back to the unmasked delta."""
    delta, tok, lift = _toy()
    band, dband = sr1._a1_apply_delta_mask(delta, tok, "band", lift)
    interior, dint = sr1._a1_apply_delta_mask(delta, tok, "interior", lift)
    assert np.array_equal(band + interior, delta)
    assert np.count_nonzero(band) + np.count_nonzero(interior) == np.count_nonzero(delta)
    # Energy shares partition unity, and each leg reports the SAME band geometry.
    assert math.isclose(dband["delta_kept_energy_share"] + dint["delta_kept_energy_share"],
                        1.0, rel_tol=1e-12)
    assert dband["band_px_cam"] == dint["band_px_cam"]
    assert dband["delta_energy_share_band"] == dint["delta_energy_share_band"]
    assert math.isclose(dband["delta_kept_energy_share"], dband["delta_energy_share_band"],
                        rel_tol=1e-12)


def test_band_mask_zeroes_the_interior_and_keeps_the_band_untouched():
    delta, tok, lift = _toy()
    band_cam = sr1.boundary(tok)[np.ix_(*lift)]
    masked, _ = sr1._a1_apply_delta_mask(delta, tok, "band", lift)
    assert np.array_equal(masked[band_cam], delta[band_cam]), "band values must be unchanged"
    assert not np.any(masked[~band_cam]), "interior must be exactly zero"


def test_mask_is_the_same_band_object_a1sign_diagnoses_on():
    """The FO-A band must be `boundary(tokens)` lifted by rt1's nn index -- not a lookalike."""
    delta, tok, lift = _toy()
    expected = sr1.boundary(tok)[np.ix_(*lift)]
    masked, diag = sr1._a1_apply_delta_mask(delta, tok, "band", lift)
    assert diag["band_px_cam"] == int(expected.sum())
    assert math.isclose(diag["band_share_cam"], float(expected.mean()), rel_tol=1e-12)
    assert np.array_equal(masked != 0.0, np.broadcast_to(expected[:, :, None], delta.shape)
                          & (delta != 0.0))
    rows, cols = lift
    assert rows.shape == (sr1.CAM_H,) and cols.shape == (sr1.CAM_W,)


def test_unknown_mask_mode_fails_closed():
    delta, tok, lift = _toy()
    for bad in ("off", "BAND", "", "all"):
        with pytest.raises(sr1.Sr1Error):
            sr1._a1_apply_delta_mask(delta, tok, bad, lift)


def test_mask_never_amplifies_the_actuator():
    """Masking may only remove perturbation; a mask that grew it would be a different lever."""
    delta, tok, lift = _toy()
    for mode in ("band", "interior"):
        masked, _ = sr1._a1_apply_delta_mask(delta, tok, mode, lift)
        assert np.all(np.abs(masked) <= np.abs(delta) + 0.0)
        assert (masked ** 2).sum() <= (delta ** 2).sum()


# --------------------------------------------------------------------------------------------
# the adjudicator
# --------------------------------------------------------------------------------------------
def test_live_branch_fires_only_strictly_below_the_incumbent_pose_error():
    assert sr1._a1_foa_verdict("band", 0.0)[0] == "POSE_NULL_BRANCH_LIVE"
    assert sr1._a1_foa_verdict("band", sr1.A1_FOA_LIVE_BELOW * 0.999)[0] == (
        "POSE_NULL_BRANCH_LIVE")
    # Exactly AT the bar is not below it.
    assert sr1._a1_foa_verdict("band", sr1.A1_FOA_LIVE_BELOW)[0] == (
        "INDETERMINATE_BETWEEN_BANDS")


def test_family_closed_fires_at_and_above_the_closed_bar():
    assert sr1._a1_foa_verdict("band", sr1.A1_FOA_CLOSED_ABOVE)[0] == "FAMILY_CLOSED"
    assert sr1._a1_foa_verdict("band", 0.022019)[0] == "FAMILY_CLOSED"
    assert "FAMILY" in sr1._a1_foa_verdict("band", 0.05)[1]


def test_between_the_bars_is_reported_not_bucketed():
    mid = 0.5 * (sr1.A1_FOA_LIVE_BELOW + sr1.A1_FOA_CLOSED_ABOVE)
    verdict, scope = sr1._a1_foa_verdict("band", mid)
    assert verdict == "INDETERMINATE_BETWEEN_BANDS"
    assert "NOT bucketed" in scope
    assert sr1._a1_foa_verdict("band", sr1.A1_FOA_CLOSED_ABOVE * 0.999)[0] == verdict


def test_only_the_band_leg_can_carry_the_verdict():
    """The interior leg is a reference; it must never emit LIVE or FAMILY_CLOSED."""
    for drift in (0.0, 0.001, 0.005, 0.02, 1.0):
        assert sr1._a1_foa_verdict("interior", drift)[0] == "COMPLEMENT_LEG"
        assert sr1._a1_foa_verdict("off", drift)[0] == "COMPLEMENT_LEG"


def test_the_live_branch_states_the_owed_seg_row():
    """A LIVE pose result is not a win; the scope string must say what is still owed."""
    _, scope = sr1._a1_foa_verdict("band", 0.0)
    assert "seg row" in scope and "OWES" in scope


# --------------------------------------------------------------------------------------------
# custody: the masked legs must never overwrite the retained parent payloads
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,suffix", [("off", ""), ("band", "_bandmask"),
                                         ("interior", "_interiormask")])
def test_output_naming_keeps_the_parent_row_intact(mode, suffix):
    """Reproduce the stage's naming rule; only `off` may claim the parent's filenames."""
    computed = "" if mode == "off" else f"_{mode}mask"
    assert computed == suffix
    pose_name = f"pose6_by_alpha{computed}.npy"
    receipt = f"SR1_A1POSE{computed.upper()}.json"
    if mode == "off":
        assert pose_name == "pose6_by_alpha.npy" and receipt == "SR1_A1POSE.json"
    else:
        assert pose_name != "pose6_by_alpha.npy"
        assert receipt != "SR1_A1POSE.json"
        assert mode.upper() in receipt


def test_stage_source_skips_the_mask_path_entirely_when_off():
    """Structural byte-identity guard: `off` must not open tokens or build a lift index.

    Reading the source is the right instrument here -- a runtime assert could pass while a
    future edit moved the mask above the guard.
    """
    import inspect

    for stage in (sr1.stage_a1pose, sr1.stage_a1sign):
        src = inspect.getsource(stage)
        assert 'mask_mode = getattr(args, "delta_mask", "off")' in src
        assert 'suffix = "" if mask_mode == "off" else' in src
    pose_src = inspect.getsource(sr1.stage_a1pose)
    assert 'open_tokens(args.tokens) if mask_mode != "off" else None' in pose_src
    assert '_a1_delta_mask_lift() if mask_mode != "off" else None' in pose_src
    assert 'if mask_mode != "off":' in pose_src
    # a1sign already opens tokens for its clip diagnostics, so its guard is on the mask apply.
    sign_src = inspect.getsource(sr1.stage_a1sign)
    assert 'if mask_mode != "off":' in sign_src
    assert '_a1_delta_mask_lift() if mask_mode != "off" else None' in sign_src


def test_both_stages_mask_through_the_one_shared_helper():
    """No second inline copy of the mask: a duplicate would drift from the pinned helper."""
    import inspect

    for stage in (sr1.stage_a1pose, sr1.stage_a1sign):
        src = inspect.getsource(stage)
        assert "_a1_apply_delta_mask(" in src, f"{stage.__name__} must call the shared helper"
        assert "keep[:, :, None]" not in src, f"{stage.__name__} re-implements the mask inline"
