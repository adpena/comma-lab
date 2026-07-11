"""tac.witness_control — the θ* COSTATE shadow controller (task #303, Phase A) +
the learned Pontryagin adjoint organ (task #426).

The Hamiltonian/optimal-control meta-layer made concrete: the campaign triality
(DAG=state x(t) · DSL=control u(t) · equations=law S) is completed by its missing
fourth object — the COSTATE λ = ∂S/∂x, the measured marginal-ΔS shadow-price field
that flows measurement → decision (memory
``project_meta_layer_above_triality_hamiltonian_control_costate_20260703``).

THE LOOP THIS PACKAGE SITS AT THE CENTER OF (2026-07-11, the Amortized-Operator
Pontryagin Loop — memo ``.omx/research/amortized_operator_pontryagin_loop_cluster_20260711.md``):
#211 FORWARD operator (clip→witness params) → **#426 ADJOINT operator** (state→λ field;
``lambda_net`` + ``costate_panel``) → **#247 CONTROLLER** (u*=argmin H; ``control_alphabet``
+ the CostateAgent DSL ``tac.witness_dsl.costate_agent_dsl``) → #396 TERMINAL solver
(gradient-free exact-argmax finisher) → measured rows → retrain the operators. The
analytic law is ``cgauge_master_action_v1``; λ IS ``costate_lambda_marginal_ds_v1``.

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
from tac.witness_control.campaign_repl import (  # noqa: F401
    CampaignWorldModel,
    ProposedAction,
    action_efficiency,
    campaign_world_model,
    propose_argv,
    write_world_model_row,
)
from tac.witness_control.costate_posterior import (  # noqa: F401
    CostatePosterior,
    all_posteriors,
    posterior_for,
    record_costate_observation,
    record_run_costates,
)
from tac.witness_control.producer_bridge import (  # noqa: F401
    ProducerSignal,
    duty_to_measure_as_candidates,
    read_producer_signals,
)
from tac.witness_control.shadow_controller import (  # noqa: F401
    VERDICT_RISING_DECOUPLING,
    ShadowReport,
    build_shadow_report,
    load_run_inputs,
    write_shadow_row,
)
from tac.witness_control.control_alphabet import (  # noqa: F401
    HEAVY_ACTIONS,
    SAFE_ACTIONS,
    ControlDecision,
    OperatorGoTicket,
    SpawnTicket,
    compose_spawn_ticket,
    feasibility_projection,
    hamiltonian_decide,
)
from tac.witness_control.costate_panel import (  # noqa: F401
    LENSES,
    ROUTING_MODES,
    PanelVerdict,
    routing_benchmark,
    run_panel,
)
from tac.witness_control.lambda_net import (  # noqa: F401
    BacktestReport,
    CampaignTrajectory,
    LambdaField,
    RashomonLambdaSet,
    backtest,
    benchmark_all,
    rashomon_lambda_set,
    read_trajectory,
)
from tac.witness_control.verdict_trend_alarm import (  # noqa: F401
    NO_ALARM,
    RISING_VERDICT,
    TRAIN_VERDICT_DECOUPLING,
    VerdictTrendAlarm,
    format_verdict_trend_line,
    verdict_trend_alarm,
)
