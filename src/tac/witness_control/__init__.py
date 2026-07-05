"""tac.witness_control — the θ* COSTATE shadow controller (task #303, Phase A).

The Hamiltonian/optimal-control meta-layer made concrete: the campaign triality
(DAG=state x(t) · DSL=control u(t) · equations=law S) is completed by its missing
fourth object — the COSTATE λ = ∂S/∂x, the measured marginal-ΔS shadow-price field
that flows measurement → decision (memory
``project_meta_layer_above_triality_hamiltonian_control_costate_20260703``).

Phase A is SHADOW MODE ONLY: observe → estimate → recommend (JSONL rows) → STOP.

CONTAINMENT (structural, not a config flag): no module in this package imports
``subprocess``, ``os.system``-style exec surfaces, or any trainer entry point.
The controller CANNOT actuate anything — it emits advisory rows to
``<run_dir>/costate_shadow.jsonl`` with ``actuation="NONE"`` and every number
axis-tagged ``[macOS advisory] NON-PROMOTABLE``. Actuation (Phase B) is DESIGN-ONLY
(``.omx/research/costate_controller_design_20260705.md``), gated on operator GO.
The no-actuation invariant is enforced by a source-scan test
(``src/tac/tests/test_witness_control_costate.py::test_no_actuation_capability``).

NO-FAKE: estimates are MEASURED (finite differences over real n600 verdict rows,
with least-squares standard errors honestly propagated) or ANALYTIC (exact partials
of the contest score law); anything else is returned as ``UNIDENTIFIABLE`` with the
probe that would identify it — never a canned/heuristic guess.
"""
from tac.witness_control.costate_estimator import (  # noqa: F401
    ANALYTIC,
    MEASURED,
    PARTIAL,
    UNIDENTIFIABLE,
    CostateEstimate,
    SlopeFit,
    analytic_costates,
    chain_ds_depoch,
    cross_run_lever_costate,
    rollback_gain,
    slope_with_stderr,
    stage_epoch_costates,
    sweep_finite_difference,
    transition_jump_costate,
)
from tac.witness_control.shadow_controller import (  # noqa: F401
    ShadowReport,
    build_shadow_report,
    load_run_inputs,
    write_shadow_row,
)
