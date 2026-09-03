"""P0 FORCES phase-2 build tests (task #360) — the three in-trunk forces as DSL Lever factories.

Proves, per force: (1) the DSL Lever factory constructs + emits the expected trainer flags + validates
its params fail-closed; (2) the trainer flags default OFF (byte-identical when unset); (3) the loss
off-path guards are structurally present (byte-identical when the weight is 0 / source uniform);
(4) lever_registry auto-derives all three (mapped, not unmapped/stale — triality) + the activation
ledger holds them with duty-to-measure (default-off is a tracked queue, not a grave); (5) the resume
divergence guard flags a changed force + tolerates absent force keys (a pre-force sidecar); (6) the
GROUND-class masking (#1), the m_safe >= delta_R threshold (#2), and the flip-density edge weighting
(#3) mechanisms; (7) NONE of the three is composed ON in crucible_v7 (§9 scope discipline).

Derivation authority: .omx/research/p0_forces_derivation_20260708.md. Pointer 0.19110 UNMOVED — means.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import activation_ledger as _al
from tac.witness_dsl import lever_registry as _lr
from tac.witness_dsl.curriculum_dsl import (
    MarginBandSatisficing,
    TemporalScrewConsistency,
    TieLocusDisplacement,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _trainer_src() -> str:
    return _TRAINER.read_text(encoding="utf-8")


def _load_trainer():
    spec = importlib.util.spec_from_file_location("_p0forces_trainer_under_test", _TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- FORCE 1: temporal screw
def test_dsl_temporal_screw_factory_emits_flags():
    lev = TemporalScrewConsistency()
    assert lev.name == "temporal_screw_consistency"
    assert lev.overrides["--seg-temporal-screw-weight"] == pytest.approx(0.1)
    assert lev.overrides["--seg-temporal-screw-xi-source"] == "ground_gt"  # confound-safe default arm
    assert lev.overrides["--seg-temporal-screw-classes"] == "0,1,2"        # GROUND only
    assert lev.overrides["--seg-temporal-screw-band"] == pytest.approx(2.0)
    # (v7.5 B.4) start_event default None => NOT emitted (byte-identical OFF; fixed-epoch gate)
    assert "--seg-temporal-screw-start-event" not in lev.overrides


def test_dsl_temporal_screw_start_event_wiring():
    """v7.5 B.4: start_event='annulus_plateau' co-emits the sensor flag so engagement is EVENT-governed
    (the unify-τ replacement for the dissolved-l7 formed-partition gate; start_epoch is the backstop cap)."""
    lev = TemporalScrewConsistency(weight=0.1, start_epoch=450, start_event="annulus_plateau")
    assert lev.overrides["--seg-temporal-screw-start-event"] == "annulus_plateau"
    assert lev.overrides["--seg-temporal-screw-start-epoch"] == 450
    # only the one wired formed-boundary sensor is legal
    with pytest.raises(ValueError, match="annulus_plateau"):
        TemporalScrewConsistency(start_event="plateau_trigger")


def test_dsl_temporal_screw_sky_rotation_only():
    """v7.5 B.5: sky_rotation_only emits the store_true + sky-row-hi flags (the sky is at infinity ⇒
    rotation-only warp, ξ translation zeroed). Default OFF => not emitted (byte-identical single warp)."""
    off = TemporalScrewConsistency(weight=0.1)
    assert "--seg-temporal-screw-sky-rotation-only" not in off.overrides
    on = TemporalScrewConsistency(weight=0.1, sky_rotation_only=True, sky_row_hi=96)
    assert on.overrides["--seg-temporal-screw-sky-rotation-only"] is True
    assert on.overrides["--seg-temporal-screw-sky-row-hi"] == 96
    with pytest.raises(ValueError, match="sky_row_hi"):
        TemporalScrewConsistency(weight=0.1, sky_rotation_only=True, sky_row_hi=0)


def test_dsl_temporal_screw_rejects_nonground_classes():
    # Movable(3)/MyCar(4) are NON-ground — the plane homography is wrong for them (GROUND masking #1).
    for bad in ("0,1,3", "4", "3,4", "5", ""):
        with pytest.raises(ValueError, match="GROUND"):
            TemporalScrewConsistency(classes=bad)
    # a valid GROUND subset is accepted
    assert TemporalScrewConsistency(classes="1").overrides["--seg-temporal-screw-classes"] == "1"


def test_dsl_temporal_screw_rejects_bad_xi_source():
    with pytest.raises(ValueError, match="xi_source"):
        TemporalScrewConsistency(xi_source="mps_live")
    assert TemporalScrewConsistency(xi_source="carrier_live").overrides[
        "--seg-temporal-screw-xi-source"] == "carrier_live"


# --------------------------------------------------------------------------- FORCE 2: margin satisfice
def test_dsl_margin_satisfice_factory_emits_flags():
    lev = MarginBandSatisficing()
    assert lev.name == "margin_band_satisficing"
    delta_r = lev.overrides["--seg-margin-satisfice-delta-r"]
    headroom = lev.overrides["--seg-margin-satisfice-headroom"]
    assert delta_r == pytest.approx(0.021881818771362305)  # MEASURED n600 artifact floor (ddm_dr1)
    assert headroom == pytest.approx(2.0)  # DERIVED minimum covering full-R annulus p95
    assert lev.overrides["--seg-margin-satisfice-msafe"] == pytest.approx(
        headroom * delta_r
    )
    lawref = lev.lawrefs["--seg-margin-satisfice-msafe"]
    row = lev.constant_manifest["--seg-margin-satisfice-msafe"]
    assert lawref.equation_id == "margin_band_satisficing_threshold_v1"
    assert row["single_value_owner"] == "margin_band_satisficing_threshold_v1"
    assert row["value"] == pytest.approx(lev.overrides["--seg-margin-satisfice-msafe"])


def test_delta_r_artifact_has_one_source():
    # ddm_dr1 2026-09-04: the n96 -> n600 repoint surfaced THREE copies of the artifact
    # path (law, hg1 lever, two DSL factory defaults). One bank: every consumer must
    # resolve the law's constant, and the DSL default must emit the law's delta_R.
    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        DELTA_R_ARTIFACT,
        resolve_margin_band_threshold,
    )
    from tac.witness_dsl import hg1_ring0_margin_hinge_levers_20260816 as hg1

    assert DELTA_R_ARTIFACT.endswith("delta_R_noise_floor_n600.json")
    assert hg1.DELTA_R_ARTIFACT == DELTA_R_ARTIFACT
    lev = MarginBandSatisficing()
    assert lev.overrides["--seg-margin-satisfice-delta-r"] == pytest.approx(
        resolve_margin_band_threshold().delta_r
    )


def test_dsl_margin_satisfice_rejects_msafe_not_derived_from_delta_r():
    # Any independently supplied m_safe must equal headroom*delta_R; merely
    # sitting above/below the floor is no longer sufficient provenance.
    with pytest.raises(ValueError, match="canonical invariant"):
        MarginBandSatisficing(msafe=0.01)
    with pytest.raises(ValueError, match="canonical invariant"):
        MarginBandSatisficing(msafe=0.019590163230895963)


# --------------------------------------------------------------------------- FORCE 3: tie-locus + w_e
def test_dsl_tie_locus_wraps_subpix_and_adds_edge_weight():
    lev = TieLocusDisplacement()
    assert lev.name == "tie_locus_displacement"
    # WRAPS the existing subpix flags
    assert lev.overrides["--seg-subpix-boundary-weight"] == pytest.approx(0.3)
    assert "--seg-subpix-boundary-v-band" in lev.overrides
    # ADDS the flip-density edge weighting + ref-domain (the missing pieces)
    assert lev.overrides["--seg-subpix-edge-weight-source"] == "pa_flipmass"
    assert lev.overrides["--seg-subpix-ref-domain"] == "seg384"


def test_dsl_tie_locus_rejects_bad_edge_source_and_ref_domain():
    with pytest.raises(ValueError, match="edge_weight_source"):
        TieLocusDisplacement(edge_weight_source="guess")
    with pytest.raises(ValueError, match="ref_domain"):
        TieLocusDisplacement(ref_domain="camera874")


# --------------------------------------------------------------------------- trainer default-OFF (byte-identical)
def test_trainer_flags_default_off_byte_identical():
    src = _trainer_src()
    # FORCE 1/2 weights default 0.0 => branch skipped + provider None => byte-identical
    assert 'ap.add_argument("--seg-temporal-screw-weight", type=float, default=0.0,' in src
    assert 'ap.add_argument("--seg-margin-satisfice-weight", type=float, default=0.0,' in src
    # FORCE 3 edge-weight source default 'uniform' => _subpix_ew_prov None => pre-existing subpix path
    assert 'ap.add_argument("--seg-subpix-edge-weight-source", type=str, default="uniform",' in src
    # ref-domain default seg384 (correct for the training loss, post-R)
    assert 'ap.add_argument("--seg-subpix-ref-domain", type=str, default="seg384",' in src
    # FORCE 1 default xi source = ground_gt (pure seg regularizer, ZERO pose coupling)
    assert 'ap.add_argument("--seg-temporal-screw-xi-source", type=str, default="ground_gt",' in src


def test_trainer_loss_off_path_guards_present():
    src = _trainer_src()
    # each force branch is gated by its weight>0 AND provider-not-None => skipped when off
    assert 'if ms_w > 0.0 and ms_gate["on"] and _ms_ann_prov is not None:' in src
    assert "if ts_w > 0.0 and ts_gate[\"on\"] and _ts_ann_prov is not None and _slog is not None:" in src
    # FORCE 3: the edge-weight multiply is behind `_subpix_ew_prov is not None`; the else preserves the
    # EXACT pre-existing subpix mean => byte-identical when uniform/off
    assert "if _subpix_ew_prov is not None:" in src
    assert "subpix_term = mx.sum(_sq) / (mx.sum(_active) + 1e-6)" in src


def test_trainer_msafe_below_delta_r_fails_closed():
    src = _trainer_src()
    # the param-extraction guard raises when m_safe < delta_R (only when the lever is active, ms_w>0)
    assert "if ms_w > 0.0 and not (ms_msafe >= ms_delta_r):" in src


def test_trainer_imports_clean():
    m = _load_trainer()
    assert hasattr(m, "run_train")
    assert hasattr(m, "_resume_lever_divergences")


# --------------------------------------------------------------------------- triality: lever_registry
def test_lever_registry_auto_derives_all_three():
    facs = _lr.lever_factories()
    for name in ("TemporalScrewConsistency", "MarginBandSatisficing", "TieLocusDisplacement"):
        assert name in facs, f"{name} not auto-derived by lever_registry AST scan"
    comp = _lr.completeness()
    new_flags = {
        "--seg-temporal-screw-weight", "--seg-temporal-screw-start-epoch",
        "--seg-temporal-screw-xi-source", "--seg-temporal-screw-classes", "--seg-temporal-screw-band",
        "--seg-margin-satisfice-weight", "--seg-margin-satisfice-msafe",
        "--seg-margin-satisfice-delta-r", "--seg-margin-satisfice-headroom",
        "--seg-margin-satisfice-start-epoch", "--seg-margin-satisfice-band",
        "--seg-subpix-edge-weight-source", "--seg-subpix-edge-weight-path", "--seg-subpix-ref-domain",
    }
    # every new trainer flag is HELD by the DSL (not an unmapped gap) — #363 passes by construction
    assert new_flags.isdisjoint(set(comp.unmapped)), \
        f"new force flags unmapped: {new_flags & set(comp.unmapped)}"
    # no new emitted flag is stale (all exist in the trainer argparse — never-invent-flags)
    assert new_flags.isdisjoint(set(comp.stale)), \
        f"new force flags stale (trainer rejects): {new_flags & set(comp.stale)}"


def test_activation_ledger_holds_all_three_with_duty_to_measure():
    known = _al.known_levers()
    duty = _al.duty_to_measure()
    for name in ("TemporalScrewConsistency", "MarginBandSatisficing", "TieLocusDisplacement"):
        assert name in known, f"{name} absent from activation ledger known_levers"
        assert name in duty, f"{name} not owed a measurement (duty-to-measure) — orphan risk"


# --------------------------------------------------------------------------- resume-divergence guard
def _min_args(**over) -> types.SimpleNamespace:
    """A namespace with the P0-force fields at their OFF defaults (the guard reads via getattr+default)."""
    base = dict(
        seg_temporal_screw_weight=0.0, seg_temporal_screw_start_epoch=0,
        seg_temporal_screw_xi_source="ground_gt", seg_temporal_screw_classes="0,1,2",
        seg_temporal_screw_band=2.0,
        seg_margin_satisfice_weight=0.0, seg_margin_satisfice_msafe=0.06,
        seg_margin_satisfice_delta_r=0.0196, seg_margin_satisfice_headroom=2.0,
        seg_margin_satisfice_start_epoch=0, seg_margin_satisfice_band=2.0,
        seg_subpix_boundary_weight=0.0, seg_subpix_edge_weight_source="uniform",
        seg_subpix_ref_domain="seg384",
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_resume_divergence_flags_changed_force_weight():
    m = _load_trainer()
    # sidecar trained WITH the forces active; resume argv drops them => fail-closed
    ckpt = {
        "__cfg_seg_temporal_screw_weight": np.asarray(0.1),
        "__cfg_seg_margin_satisfice_weight": np.asarray(0.2),
        "__cfg_seg_subpix_boundary_weight": np.asarray(0.3),
    }
    div = m._resume_lever_divergences(ckpt, _min_args())  # all three OFF now
    joined = " ".join(div)
    assert "seg_temporal_screw_weight" in joined
    assert "seg_margin_satisfice_weight" in joined
    assert "seg_subpix_boundary_weight" in joined


def test_resume_divergence_tolerates_absent_force_keys():
    m = _load_trainer()
    # a pre-force sidecar lacks the keys entirely => NO spurious divergence (only present keys checked)
    div = m._resume_lever_divergences({"__cfg_mod_dim": np.asarray(16)},
                                      _min_args(seg_temporal_screw_weight=0.1))
    assert all("seg_temporal_screw" not in d for d in div)


def test_resume_divergence_ignores_ref_domain():
    m = _load_trainer()
    # ref_domain is decode-consumer provenance (inert for the training trajectory) => NOT guarded:
    # a resume that only changes it must NOT fail-close.
    ckpt = {"__cfg_seg_subpix_ref_domain": np.asarray("camera874_dphase")}
    div = m._resume_lever_divergences(ckpt, _min_args(seg_subpix_ref_domain="seg384"))
    assert all("ref_domain" not in d for d in div)


# --------------------------------------------------------------------------- FORCE 3 edge-weight math
def test_edge_weight_map_concentrates_on_road_lane():
    """Replicates the inline W_e provider math: a Road(0)<->Lane(1) straddle gets W_e[0,1], an inactive
    pixel gets 0, a non-Road edge gets its own (light) weight. This is the FORCE-3 flip-density lever."""
    # W_e stamped Road-hub (Road<->Lane heaviest); mean-normalized shape is irrelevant to the lookup.
    W_e = np.ones((5, 5), np.float32)
    W_e[0, 1] = W_e[1, 0] = 3.0     # Road<->Lane heaviest
    W_e[2, 3] = W_e[3, 2] = 0.2     # a light non-Road edge
    # a 2x3 lstar with a Road|Lane RIGHT straddle at (0,0)->(0,1) and an Undriv|Movable at (1,0)->(1,1)
    lst = np.array([[0, 1, 1], [2, 3, 3]], np.int64)
    t_map = np.array([[0.5, -1.0, -1.0], [0.4, -1.0, -1.0]], np.float32)  # active at (0,0) and (1,0)
    dir_map = np.zeros((2, 3), np.float32)  # all RIGHT
    act = t_map >= 0.0
    cb_r = np.zeros_like(lst); cb_r[:, :-1] = lst[:, 1:]
    cb_d = np.zeros_like(lst); cb_d[:-1, :] = lst[1:, :]
    c_b = np.where(dir_map < 0.5, cb_r, cb_d)
    wmap = np.zeros((2, 3), np.float32)
    ai, aj = np.nonzero(act)
    wmap[ai, aj] = W_e[lst[ai, aj], c_b[ai, aj]]
    assert wmap[0, 0] == pytest.approx(3.0)   # Road<->Lane straddle => heavy
    assert wmap[1, 0] == pytest.approx(0.2)   # Undriv<->Movable straddle => light
    assert wmap[0, 1] == pytest.approx(0.0)   # inactive pixel => 0


def test_temporal_screw_ground_class_mask_subset():
    """The GROUND include mask over channels {0,1,2}: 1.0 for selected, 0 else (#1 masking)."""
    for classes, expect in (("0,1,2", [1, 1, 1]), ("1", [0, 1, 0]), ("0,2", [1, 0, 1])):
        sel = {int(c.strip()) for c in classes.split(",") if c.strip() != ""}
        mask = [1.0 if c in sel else 0.0 for c in (0, 1, 2)]
        assert mask == pytest.approx(expect)


# --------------------------------------------------------------------------- scope: none composed ON in v7
def test_none_of_the_three_forces_composed_on_in_crucible_v7():
    from tac import witness_autoconfig as wa
    # the sealed v6/v7 DSL lever list must NOT contain any P0-force factory (§9: default-off, activate
    # ONE per crucible increment with a measured A/B; turning them on by default is a SPEC VIOLATION).
    force_names = {"TemporalScrewConsistency", "MarginBandSatisficing", "TieLocusDisplacement"}
    assert force_names.isdisjoint(set(wa._CRUCIBLE_V6_DSL_LEVERS)), \
        "a P0 force is composed ON in the crucible lever list — SPEC VIOLATION (§9)"
    # and the sealed constants module text does not activate the force flags with a non-off value
    wa_src = Path(wa.__file__).read_text(encoding="utf-8")
    for flag in ("--seg-temporal-screw-weight", "--seg-margin-satisfice-weight",
                 "--seg-subpix-edge-weight-source"):
        assert flag not in wa_src, f"crucible autoconfig activates {flag} — must stay default-off (§9)"


# ─────────────────────────────────────────────────────── v7.5 B.5: horizon-weighted margin (#169)
def test_dsl_horizon_weighted_margin_emits_flags():
    """v7.5 B.5 (#169): the DSL factory holds the 0-byte horizon-weighted margin flags (all REAL
    levelset-trainer argparse flags; never-invent-flags)."""
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin
    lev = HorizonWeightedMargin(weight=0.1)
    assert lev.name == "horizon_weighted_margin"
    ov = lev.overrides
    assert ov["--seg-horizon-margin-weight"] == pytest.approx(0.1)
    assert ov["--seg-horizon-margin-target"] == pytest.approx(0.5)     # the safe-margin ceiling
    assert ov["--seg-horizon-margin-lo"] == pytest.approx(0.3)          # reducible band lower (excl label-noise)
    assert ov["--seg-horizon-margin-hi"] == pytest.approx(0.5)          # reducible band upper
    assert ov["--seg-horizon-row-lo"] == 96 and ov["--seg-horizon-row-hi"] == 288  # horizon band


def test_dsl_horizon_weighted_margin_rejects_degenerate_bands():
    """The reducible GT-margin band [lo,hi] and the horizon row band must both be non-empty (the #169
    [0.3,0.5] band EXCLUDES the <lo irreducible label-noise; an empty band would be a silent no-op)."""
    from tac.witness_dsl.curriculum_dsl import HorizonWeightedMargin
    with pytest.raises(ValueError, match="margin_lo"):
        HorizonWeightedMargin(weight=0.1, margin_lo=0.5, margin_hi=0.3)
    with pytest.raises(ValueError, match="row_lo"):
        HorizonWeightedMargin(weight=0.1, row_lo=300, row_hi=96)


def test_horizon_weighted_margin_registered_off_with_exit_criterion():
    """v7.5 B.5 disposition: BUILT + HELD + REGISTERED default-OFF in the duty-to-measure queue (an A/B
    arm, NOT composed ON — it would perturb the sealed config + carries the label-noise risk). The
    registered trigger names the exit-criterion n600 A/B (the honest 'off is a tracked queue' state)."""
    import tac.witness_autoconfig as wac
    roff = wac.crucible_v7_registered_off_levers()
    assert "horizon_weighted_margin_169" in roff
    e = roff["horizon_weighted_margin_169"]
    assert e["default"] == "off" and e["state"] == "registered_duty_to_measure"
    # the exit criterion (distinguish real recovery from chasing label-noise) is named in the trigger
    assert "measure_dseg_reducibility_gt_margin" in e["trigger"]
    assert "label-noise" in e["trigger"] and "HIGHER GT margin" in e["trigger"]
