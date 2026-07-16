# SPDX-License-Identifier: MIT
"""Warm-start schedule reconstruction law — the DERIVED form for lineage schedule boundaries.

Registered 2026-07-16 for the ``c2_surgical_warm`` GO chain (operator GO 2026-07-16), surfaced by
the REAL schedule-provenance gate: the c2 composed config emits six positive ``--*-start-epoch``
triggers, NO recognised event sensor is co-emitted (the c1 ``label_floor`` sensor is MEASURED DEAD
on the warm path — its band [0.00496, 0.00700] sits ABOVE the trunk's resume d_seg ~0.0034,
adverse finding A2), so the EVENT and CAP legal forms are structurally unavailable and DERIVED is
the only honest form. This equation IS the derivation those constants rows cite.

THE LAW (the trainer's own warm-start contract, not a new physics claim):

  A weights-only warm start of checkpoint C MUST reproduce C's schedule plant — the resumed
  weights are conditioned on C's schedule (L18 config-conditional constants). Therefore every
  schedule boundary of the warm run is a FUNCTION of named inputs, never a hand-typed epoch:

    * ``config_of_record``      value = C's own launch.sh value        (tau@300, muon@726)
    * ``run_length_exclusion``  value = run_epochs                     (l7 NEVER runs: start==epochs,
                                                                        the trainer's documented
                                                                        exclusion pattern; A3)
    * ``resume_plus_window``    value = resume_epoch + re_anchor_window (the surgical engage
                                                                        boundary: fresh-moment
                                                                        settling ~= 2x the
                                                                        stage-transition rewarmup
                                                                        scale)
    * ``original_plant_end``    value = original_schedule_epochs        (the pose-finish backstop
                                                                        cap rides the plant end)

The evaluator RECOMPUTES each value from its inputs at gate time (``resolve_equation_value``), so
the DERIVED claim is machine-checked, and the ``config_of_record`` rows carry the checkpoint
launch.sh as a SHA-verified authority artifact (the #351 custody surface).

Verdict scope: FORMULATION — this is a config-derivation law (value provenance), not a score
claim; the pointer moves only through a byte-closed n600 ``upstream/evaluate.py`` row.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import (  # noqa: F401  (re-exported; the SoT evaluator)
    eval_warm_start_schedule_reconstruction,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "warm_start_schedule_reconstruction_v1"

#: the c2_surgical_warm config of record (the warm-start trunk's own launch.sh).
CONFIG_OF_RECORD = ("experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/"
                    "launch.sh")
CONFIG_OF_RECORD_SHA256 = "dd7921bce67da7fdb0982c5c951cfbc69156e5092b7a454576d5a4b4acfce94a"


def build_warm_start_schedule_reconstruction_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        CONFIG_OF_RECORD,
        reactivation_criteria=(
            "config-derivation law; re-derive if the trainer's --anneal-epochs warm-start "
            "contract or the l7 start==epochs exclusion pattern changes"
        ),
        measurement_axis="[config-of-record source inspection]",
        hardware_substrate="n/a (derivation)",
        captured_at_utc="2026-07-16T13:45:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="warm_start_schedule_reconstruction_config_of_record_20260716",
            measurement_utc="2026-07-16T13:45:00Z",
            inputs={"checkpoint": "mod32cap EMA-best ep650 (d_seg 0.003146 n600 through-R)",
                    "config_of_record": CONFIG_OF_RECORD,
                    "sha256": CONFIG_OF_RECORD_SHA256,
                    "read_values": {"tau_softplus_start_epoch": 300, "muon_start_epoch": 726,
                                    "l7_start_epoch_original": 1001, "anneal_epochs_plant": 1000}},
            predicted_output={"schedule_continuity": "resume epoch sees the checkpoint's exact "
                                                     "schedule state under --anneal-epochs 1000"},
            empirical_output={"tau_softplus_start_epoch": 300, "muon_start_epoch": 726,
                              "l7_never_runs_start": 1400, "surgical_engage_epoch": 700,
                              "pose_finish_backstop": 1000},
            residual=0.0,
            source_artifact=CONFIG_OF_RECORD,
            measurement_method=(
                "source inspection of the warm-start checkpoint's own launch.sh (SHA-verified); "
                "boundaries recomputed by eval_warm_start_schedule_reconstruction"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Warm-start schedule reconstruction (lineage boundaries are functions, not epochs)",
        one_line_summary=(
            "warm start reproduces the checkpoint's schedule plant: boundary = f(config-of-record "
            "| run_epochs | resume+window | plant_end); evaluator recomputes at gate time"
        ),
        latex_form=(
            r"e_{\mathrm{boundary}} = \begin{cases}"
            r" e_{\mathrm{record}} & \text{config-of-record}\\"
            r" E_{\mathrm{run}} & \text{l7 exclusion (start==epochs)}\\"
            r" e_{\mathrm{resume}} + w_{\mathrm{re-anchor}} & \text{surgical engage}\\"
            r" E_{\mathrm{plant}} & \text{backstop cap}\end{cases}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.warm_start_schedule_reconstruction_20260716:"
            "eval_warm_start_schedule_reconstruction"
        ),
        domain_of_validity={
            "vehicle": "weights-only warm start of a same-architecture checkpoint "
                       "(--warm-start-weights-only + --anneal-epochs <original plant>)",
            "contract": "--anneal-epochs docstring: a warm-start arm MUST pin the anneals to the "
                        "ORIGINAL schedule length so the resume epoch reproduces the plant",
            "sensor_context": "applies when NO recognised event sensor is co-emittable (c2: the "
                              "label_floor band is unreachable from resume d_seg ~0.0034 — "
                              "adverse finding A2); warm-path sensor recalibration stays OWED",
            "verdict_scope": "FORMULATION — config-derivation law (value provenance), not a "
                             "score claim",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"mode": "derivation selector", "config_of_record_value": "epoch",
                  "run_epochs": "epoch", "resume_epoch": "epoch", "re_anchor_window": "epochs",
                  "original_schedule_epochs": "epoch"},
        units_out={"boundary_epoch": "epoch"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"boundary_recompute": 0.0},
        last_calibration_utc="2026-07-16T13:45:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.spec_c2_surgical_20260716 (six DERIVED schedule constants rows)",
            "tools/schedule_provenance_gate.py (DERIVED classification of the emitted triggers)",
            "tools/launch_witness_run.py (the #332/#351 LawRef self-recompile)",
        ),
        canonical_producers=(
            CONFIG_OF_RECORD,
            "experiments/train_levelset_witness_realized_through_R_mlx.py (--anneal-epochs + "
            "l7 start==epochs contracts)",
        ),
        provenance=provenance,
    )


def populate_warm_start_schedule_reconstruction_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of FEED-c2-golaunch)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_warm_start_schedule_reconstruction_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "CONFIG_OF_RECORD",
    "CONFIG_OF_RECORD_SHA256",
    "EQUATION_ID",
    "build_warm_start_schedule_reconstruction_v1",
    "eval_warm_start_schedule_reconstruction",
    "populate_warm_start_schedule_reconstruction_equation",
]
