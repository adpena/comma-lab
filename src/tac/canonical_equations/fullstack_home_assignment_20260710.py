# SPDX-License-Identifier: MIT
"""Canonical equation: the FULL-STACK unique-home assignment law (#398 thread B, 2026-07-10).

One DERIVED clause-A design law, the equations leg of the full-stack recursive-fractal-optimal
synthesis (``.omx/research/fullstack_fractal_optimal_synthesis_20260710.md``).

LAW — ``fullstack_unique_home_assignment_v1``.
Every scored degree of freedom of the stack ``W = (G, xi, T)`` has exactly ONE carrier home,
DERIVED from the measured frozen-scorer structure (never assigned by convention):

* frame_0 (all bands)  -> the xi/pose channel ONLY — its seg obligation is identically 0
  (MEASURED n600: d_seg 8.48e-9 with random-noise f0 = 1 argmax-tie pixel / 117,964,800);
  efficient operating point = luma-only (chroma removal frees 67% bytes at sqrt(10*d_pose)
  <= 0.180 naive UPPER bound). frame_0 x chroma-HF is a DEAD subspace (no obligation either way).
* frame_1 luma          -> G (separatrix/partition placement) + T's luma FALLBACK — the only
  doubly-priced block (pose 11.1x per plane MEASURED); joint design lives here and only here.
* frame_1 chroma-HF at the yuv6/384 plane (grid-aligned, luma-null) -> T's PREFERENTIAL home —
  exactly pose-null (op-level 3.4e-6 MEASURED) with seg authority Δd_seg ~2.73e-3; accessible
  ONLY by pre-imaging through the exact bilinear-down kernel D (a naive camera-res chroma dither
  leaks 50% into luma and pays ~luma pose price).
* T's frequency support -> the stem passband [period-4 .. 8] render-px (stride-2 Nyquist wall);
  out-of-band texture is dead (aliased or read-flat).
* per-class fills       -> Road/Lane FILL = FLAT scene colour (region-wide period-4 grating fill
  is ANTAGONISTIC in composition: +0.228 agg d_seg, Road/Lane -> ~1.0, MEASURED n600); texture
  enters ONLY as learned modulation (T). Undrivable/MyCar/Movable are flat-winnable basins
  (+8.33 / +11.85 / +0.31 margins); Movable/MyCar geometry rides sparse-site/static carriers.
* xi                    -> the 6-dim se(3) twist per pair (the clause-B minimal instance),
  banked dxi d_pose 0.001610 -> contribution 0.127 at xi_eff 7.2 KB (n600 byte-close authority).

NORMATIVE CLAUSE (clause A + clause B jointly): a byte stored OUTSIDE its home is either a
DUPLICATE (reconstructible from {other sections + the rule-118-free generator} — a pure rate loss)
or it pays a pose/seg price the obligation matrix proves it need not pay. Dedup therefore FALLS OUT
of the home map; the pairwise non-derivability audit at byte-close is the law's enforcement gate.

Every row above is anchored by an already-registered MEASURED law
(``scorer_obligation_matrix_factorization_v1`` · ``posenet_luma_chroma_sensitivity_asymmetry_v1``
(Unit C anchors) · ``segnet_stem_nyquist_alias_wall_v1`` · ``segnet_through_r_texture_price_list_v1``
· ``roadlane_grating_composition_refuted_v1``). The chroma-HF basin-LEGIBILITY conjecture (whether
384-band chroma alone cuts the 0.0416 flat floor >=2x) is EXCLUDED from this law — it stays a
pre-registered falsifier (Fable E-1). means != ends: this is a DESIGN law; pointer contest-CPU
0.19110 UNMOVED — it moves only through a byte-closed ``upstream/evaluate.py`` n600 exact row.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

_UTC = "2026-07-10T12:30:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/fullstack_fractal_optimal_synthesis_20260710.md"
_HW = "apple_m5_max_cpu"

FULLSTACK_HOME_ASSIGNMENT_EQUATION_ID = "fullstack_unique_home_assignment_v1"


def _sidecar_prov(reactivation: str) -> object:
    return build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=reactivation,
        measurement_axis=_ADVISORY,
        hardware_substrate=_HW,
    )


def build_fullstack_unique_home_assignment_v1() -> CanonicalEquation:
    """Every scored DOF of W=(G, xi, T) has exactly one measured carrier home (clause A/B)."""

    anchor = EmpiricalAnchor(
        anchor_id="fullstack_home_map_component_anchor_composition_20260710",
        measurement_utc=_UTC,
        inputs={
            "cited_laws": [
                "scorer_obligation_matrix_factorization_v1 (2x2 frames x bands pricing)",
                "posenet_luma_chroma_sensitivity_asymmetry_v1 "
                "(frame0_segfree_n600_20260710: 8.48e-9; "
                "chroma_hf_posenull_transfer_384only_20260710: op-null 3.4e-6, "
                "camera-res leak 50%, ideal-lever seg authority 2.73e-3)",
                "segnet_stem_nyquist_alias_wall_v1 (period-4 = 2*stride Nyquist)",
                "segnet_through_r_texture_price_list_v1 (flat basins +8.33/+11.85/+0.31; "
                "Road/Lane flat floors -3.50/-5.00)",
                "roadlane_grating_composition_refuted_v1 (grating FILL +0.228 ANTAGONISTIC n600 "
                "=> Road/Lane fill PINNED flat; texture = learned modulation only)",
            ],
            "banked_pose": "R1 dxi d_pose 0.001610 -> 0.127 contribution, xi_eff 7.2 KB "
            "(n600 byte-close authority, MEMORY L68)",
            "method": "composition of the cited measured anchors into the one-home map; "
            "no new measurement — each row inherits its component's authority",
        },
        predicted_output={
            "home_map_rows": 6,
            "conjectures_inside_the_law": 0,
            "excluded_conjecture": "chroma-HF basin LEGIBILITY (>=2x flat-floor cut) — "
            "pre-registered falsifier Fable E-1, NOT part of the law",
        },
        empirical_output={
            "frame0_seg_obligation": 8.48e-9,
            "chroma_hf_384_pose_null_op_level": 3.4e-6,
            "chroma_hf_camera_res_luma_leak_ratio": 1.075,
            "stem_passband_period_px": [4.0, 8.0],
            "grating_fill_delta_agg_dseg": 0.22778,
            "banked_pose_contribution": 0.127,
            "xi_dim": 6,
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="composition of registered component anchors (each n600/op-level "
        "MEASURED at its own registration; see cited_laws)",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=_sidecar_prov(
            "re-derive any row whose component law gains a contradicting anchor; the chroma-HF "
            "PREFERENTIAL-home row weakens to chroma-first/luma-fallback if Fable E-1 refutes "
            "384-band basin legibility"
        ),
    )
    return CanonicalEquation(
        equation_id=FULLSTACK_HOME_ASSIGNMENT_EQUATION_ID,
        name=(
            "Full-stack unique-home assignment: every scored DOF of W=(G, xi, T) has exactly one "
            "carrier home derived from the measured scorer obligation matrix + stem passband + "
            "per-class basin price list (clause A/B; dedup falls out of the map)"
        ),
        one_line_summary=(
            "frame_0 -> xi only; frame_1 luma -> G+T-fallback; frame_1 chroma-HF@384 -> T "
            "pose-free; T band = [4,8] px; Road/Lane fill = flat; xi = 6-dim banked twist."
        ),
        latex_form=(
            r"\mathrm{home}: \{f_0\}\to\xi,\ \{f_1,\mathrm{luma}\}\to G\oplus T_{\mathrm{fb}},\ "
            r"\{f_1,\mathrm{chroma\text{-}HF}@384\}\to T,\ \mathrm{supp}(T)=[4,8]\,\mathrm{px},\ "
            r"\mathrm{fill}(\mathrm{Road,Lane})=\mathrm{flat}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.fullstack_home_assignment_20260710:"
            "build_fullstack_unique_home_assignment_v1"
        ),
        domain_of_validity={
            "vehicle": [
                "softmax_of_sdf_levelset_witness (v7.5.x single trunk)",
                "v8 per-class geometric carriers",
            ],
            "perceiver": [
                "frozen upstream SegNet (efficientnet_b2, x[:,-1]) + PoseNet (fastvit_t12, yuv6)"
            ],
            "measurement_axis": ["macOS-CPU advisory / through-R n600 component anchors"],
            "excluded": [
                "chroma-HF basin-legibility (conjecture, Fable E-1)",
                "any scorer-checkpoint change (re-measure the components)",
            ],
        },
        units_in={"scorer_blocks": "frames x frequency-bands", "class_basins": "signed margin"},
        units_out={"home": "carrier assignment", "obligation": "d_seg / d_pose price"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"fullstack_home_assignment": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_autoconfig (v7.5.3 skeleton)",
            "tac.boundary_math.texture_trunk",
            "SPEC_v8.1 increment sequence",
        ),
        canonical_producers=(".omx/research/fullstack_fractal_optimal_synthesis_20260710.md",),
        provenance=_sidecar_prov(
            "recalibrate when any component law gains a new anchor; the law is a composition — "
            "it is exactly as strong as its weakest component anchor"
        ),
    )


def populate_fullstack_unique_home_assignment_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the full-stack unique-home assignment law (#398
    thread B equations leg; the DAG leg is FEED-fractal-synth; the DSL leg is
    N/A-with-rationale — the v7.5.3 typed builder lands at BUILD time, see the memo)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_fullstack_unique_home_assignment_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="fullstack_home_assignment_20260710 (#398B: clause-A one-home map of W=(G,xi,T), "
        "composed strictly of already-measured component anchors; chroma-legibility conjecture "
        "EXCLUDED — pre-registered falsifier Fable E-1)",
    )
    return eq


__all__ = [
    "FULLSTACK_HOME_ASSIGNMENT_EQUATION_ID",
    "build_fullstack_unique_home_assignment_v1",
    "populate_fullstack_unique_home_assignment_equation",
]
