# SPDX-License-Identifier: MIT
"""The CostateAgent DSL — the CONTROLLER's typed program (task #426; sibling of the
WitnessProgram DSL, which defines the CONTROLLED SYSTEM).

The triality extended into a closed-loop control system: witness DSL = the plant
(what to train) · **CostateAgent DSL = the controller** (how the costate organ senses,
thinks, and acts) · equations = the law (λ = ∂S/∂x binds to ``cgauge_master_action_v1``
+ ``costate_lambda_marginal_ds_v1`` via EquationBinding, resolved fail-closed against
the canonical registry) · DAG = the state. A sensor/actuator/expert is not BUILT until
it is declared here (the same config-must-be-DSL-defined discipline that made every
lever a ``Lever`` factory).

``CostateAgentProgram.validate()`` fail-closes on: unknown expert architectures
(never-invent — checked against the REAL lens registry), unknown actuator actions
(closed alphabet from ``control_alphabet``), HEAVY actuators not operator-GO-gated,
unknown routing modes, equation ids absent from the canonical registry, and spawn
actuators without inherited containment. ``compile()`` returns a
``CompiledCostateOrgan`` whose methods are the REAL wiring (sense → adjoint → decide →
act), not a paper spec — NO-FAKE: the compiled organ is exercised by the test suite
against the live run directory's telemetry.

CONTAINMENT is a first-class typed field (``ContainmentSpec``), not a comment: the
compiled ``act()`` routes through ``control_alphabet.feasibility_projection`` — SAFE
actions pass through as advisory artifacts, HEAVY actions RETURN OperatorGoTickets
structurally (there is no execute path), and spawned agents inherit the containment
clause verbatim inside their composed prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tac.witness_control.control_alphabet import (
    HEAVY_ACTIONS,
    INHERITED_CONTAINMENT_CLAUSE,
    SAFE_ACTIONS,
    ControlDecision,
    OperatorGoTicket,
    SpawnTicket,
    compose_spawn_ticket,
    feasibility_projection,
    hamiltonian_decide,
)
from tac.witness_control.costate_panel import LENSES, ROUTING_MODES

#: the real lens architecture names (never-invent: an ExpertSpec must name one)
KNOWN_EXPERT_ARCHITECTURES = tuple(spec.architecture for spec in LENSES)


class SensorSpec(BaseModel):
    """One read-only observation channel. Score-neutral ⇒ DEFAULT-ON (the
    default-off-is-orphaned-signal rule): a sensor may only be off with a reason."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    source: Literal["run_telemetry", "graph_memory", "activation_ledger",
                    "cached_probe", "canonical_equations"]
    fields: tuple[str, ...] = ()
    enabled: bool = True
    disabled_reason: str = ""

    @model_validator(mode="after")
    def _off_needs_reason(self) -> "SensorSpec":
        if not self.enabled and len(self.disabled_reason.strip()) < 8:
            raise ValueError(f"sensor {self.name!r}: score-neutral sensors default ON; "
                             "'off' requires a recorded substantive reason")
        return self


class ActuatorSpec(BaseModel):
    """One tool-alphabet member with its cost tier. HEAVY ⇒ operator-GO gated."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    action: str
    cost_tier: Literal["SAFE", "HEAVY"]
    operator_go_gated: bool = False
    inherits_containment: bool = False       # REQUIRED True for spawn_reasoning_agent
    witness_lever_refs: tuple[str, ...] = () # composition: levers this actuator may propose

    @model_validator(mode="after")
    def _closed_alphabet_and_gates(self) -> "ActuatorSpec":
        if self.cost_tier == "SAFE" and self.action not in SAFE_ACTIONS:
            raise ValueError(f"unknown SAFE action {self.action!r} — closed alphabet "
                             f"{SAFE_ACTIONS}; never-invent-tools")
        if self.cost_tier == "HEAVY":
            if self.action not in HEAVY_ACTIONS:
                raise ValueError(f"unknown HEAVY action {self.action!r} — closed "
                                 f"alphabet {HEAVY_ACTIONS}")
            if not self.operator_go_gated:
                raise ValueError(f"HEAVY actuator {self.action!r} MUST be "
                                 "operator_go_gated=True (containment is typed, "
                                 "not a comment)")
        if self.action == "spawn_reasoning_agent" and not self.inherits_containment:
            raise ValueError("spawn_reasoning_agent MUST declare "
                             "inherits_containment=True")
        return self


class ExpertSpec(BaseModel):
    """One panel lens: architecture + the QUESTION it answers + the TOOLS it exercises."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    architecture: str
    question: str
    tools_exercised: tuple[str, ...]
    predictive: bool = True
    activation_note: str = ""   # e.g. "prior-weighted until measured skill accrues"

    @model_validator(mode="after")
    def _known_architecture(self) -> "ExpertSpec":
        if self.architecture not in KNOWN_EXPERT_ARCHITECTURES:
            raise ValueError(f"expert {self.name!r}: architecture "
                             f"{self.architecture!r} not in the real lens registry "
                             f"{KNOWN_EXPERT_ARCHITECTURES}; never-invent")
        return self


class RoutingSpec(BaseModel):
    """The routing grain — a typed choice over the measured spectrum."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: str = "EVIDENCE_SHRUNK_STACKING"
    single_best_lens: str = "flow"    # SINGLE_BEST's lens — arbitrated, not hardwired
    stacking_prior_strength: float = Field(default=3.0, ge=0.0)
    self_activation_cap: float = Field(default=0.5, gt=0.0, le=1.0)
    escalation: Literal["rashomon_action_disagreement", "dispersion", "both"] = "both"
    spread_gate: bool = True          # pre-decision Rashomon-spread routing (2607.08046 §3)
    interpretable_head: bool = True   # HARD requirement (Rudin): PRISM prototype router head
    interpretable_provenance: str = ""
    provenance: str = ""   # the benchmark row that justified the mode choice

    @model_validator(mode="after")
    def _known_mode(self) -> "RoutingSpec":
        if self.mode not in ROUTING_MODES:
            raise ValueError(f"routing mode {self.mode!r} not in {ROUTING_MODES}")
        if not self.interpretable_head:
            raise ValueError("interpretable_head=False violates the interpretable-by-design "
                             "non-negotiable (Rudin co-lead): the router head MUST be the "
                             "PRISM prototype router (explanations are contracts, not retrofits)")
        return self


class AcquisitionSpec(BaseModel):
    """The self-inventing curriculum (Schmidhuber PowerPlay/curiosity) as a typed policy.

    curiosity = innovation signal (derived-λ − measured-binding = compression progress);
    the Gödel-machine proof-gate = backtest-passed AND predicted-ΔS<0 AND containment."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    policy: Literal["powerplay_curiosity"] = "powerplay_curiosity"
    rank_key: Literal["curiosity_x_blast_over_cost"] = "curiosity_x_blast_over_cost"
    proof_gate: Literal["godel_backtest_and_never_regress"] = "godel_backtest_and_never_regress"
    provenance: str = ""


class TrainingStageSpec(BaseModel):
    """One stage of the organ's training pipeline (operator 2026-07-11: pre-training /
    post-training / RL — "how you were born"). Every stage is Gödel-gated: EXECUTED
    only with a measured backtest win; a stage that needs compute the advisory
    envelope cannot spend stays PROPOSED with its named blocker (NO-FAKE: a designed
    stage is never narrated as a built one)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    stage: Literal["pretrain", "posttrain_sft", "posttrain_alignment", "rl"]
    corpus: tuple[str, ...]            # data sources (trajectory-independent for pretrain)
    status: Literal["PROPOSED", "EXECUTED_$0", "BLOCKED"] = "PROPOSED"
    blocker: str = ""                  # named+measured blocker when BLOCKED/PROPOSED
    measured: str = ""                 # the backtest row when EXECUTED

    @model_validator(mode="after")
    def _blocked_needs_blocker(self) -> "TrainingStageSpec":
        if self.status == "BLOCKED" and len(self.blocker.strip()) < 8:
            raise ValueError(f"stage {self.name!r}: BLOCKED requires a named blocker")
        if self.status == "EXECUTED_$0" and len(self.measured.strip()) < 8:
            raise ValueError(f"stage {self.name!r}: EXECUTED requires the measured row")
        return self


#: the organ's training pipeline AS SHIPPED (honest statuses; the $0-executable piece
#: — the scorer-geometry prior injected into the solve (G arm) — is the first
#: pretrain-vs-scratch datapoint; everything needing scorer forwards / real training
#: compute carries its named blocker and rides the duty-to-measure queue)
DEFAULT_TRAINING_PIPELINE: tuple[TrainingStageSpec, ...] = (
    TrainingStageSpec(
        name="scorer_geometry_prior", stage="pretrain",
        corpus=("gt_n96.npz margins/lstars (cached Fisher-surrogate field)",),
        status="EXECUTED_$0",
        measured="G_ridge_scorerprior arm in the architecture tournament (LOO+WF vs "
                 "A_ridge_solve = the pretrain-vs-scratch A/B; see the organ ledger)"),
    TrainingStageSpec(
        name="uniward_cost_prior", stage="pretrain",
        corpus=("gt_n96.npz frames (Holub-Fridrich inverse-steg cost geometry)",),
        status="EXECUTED_$0",
        measured="uniward_cost_susceptibility sensor (scorer_geometry module); "
                 "tournament arm rides G with prior= swap"),
    TrainingStageSpec(
        name="scorer_adversarial_boundary", stage="pretrain",
        corpus=("frozen SegNet/PoseNet min-flip probing", "margin-saliency #141",
                "evaluator null-space #47"),
        status="BLOCKED",
        blocker="needs bounded scorer forwards; the live #205 run (pid-held slot) owns "
                "scorer compute — queued on the duty-to-measure ledger"),
    TrainingStageSpec(
        name="scorer_distilled_surrogate", stage="pretrain",
        corpus=("frozen SegNet/PoseNet distillation (Hinton)", "comma10k (SegNet's "
                "actual training distribution; 0 contest frames per L80)"),
        status="BLOCKED",
        blocker="multi-hour training job; operator-GO ticket for compute; never "
                "modifies the pinned scorers, never ships"),
    TrainingStageSpec(
        name="trajectory_sft", stage="posttrain_sft",
        corpus=("#205 trajectory intervals (the measured campaign data)",),
        status="EXECUTED_$0",
        measured="the tournament fits ARE the SFT stage at current data scale "
                 "(ridge/prototype solves; nets lose per the measured curse of "
                 "sample complexity — re-run per record accrual)"),
    TrainingStageSpec(
        name="constitution_alignment", stage="posttrain_alignment",
        corpus=("advisory-only containment (typed)", "interpretable-by-design (PRISM)",
                "honesty stamps (SPECULATIVE-UNTIL-BACKTESTED)", "Gödel proof-gate",
                "self-activation awareness (meta-λ)"),
        status="EXECUTED_$0",
        measured="structurally enforced by the DSL validators + containment "
                 "source-scan tests (the constitution is the type system)"),
    TrainingStageSpec(
        name="gepa_reflective_rl", stage="rl",
        corpus=("measured backtest records (reward = walk-forward improvement)",
                "PowerPlay curiosity (innovation signal = intrinsic reward)"),
        status="EXECUTED_$0",
        measured="first GEPA cycle 2026-07-11: plateau-fallback candidate PROPOSED "
                 "from the walk-forward reflection and REFUSED by measurement "
                 "(hybrid 0.002609 vs pure E_prototype 0.002513) — the gate works"),
)


class EquationBinding(BaseModel):
    """λ = ∂S/∂x bound to the canonical registry (single SoT, resolved fail-closed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    master_action_id: str = "cgauge_master_action_v1"
    costate_law_id: str = "costate_lambda_marginal_ds_v1"

    def resolve(self) -> None:
        from tac.canonical_equations import query_equations
        ids = {getattr(r, "equation_id", None) for r in query_equations()}
        for eq in (self.master_action_id, self.costate_law_id):
            if eq not in ids:
                raise ValueError(f"EquationBinding: {eq!r} absent from the canonical "
                                 "equations registry (fail-closed)")


class ContainmentSpec(BaseModel):
    """The operator-GO boundary as a typed field."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    autonomous_envelope: Literal["advisory-$0"] = "advisory-$0"
    heavy_requires_operator_go: Literal[True] = True
    spawned_agents_inherit: Literal[True] = True
    containment_clause: str = INHERITED_CONTAINMENT_CLAUSE


class CostateAgentProgram(BaseModel):
    """The controller's single source of truth (validate → compile → the real organ)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    run_dir: str
    sensors: tuple[SensorSpec, ...]
    actuators: tuple[ActuatorSpec, ...]
    experts: tuple[ExpertSpec, ...]
    routing: RoutingSpec
    acquisition: AcquisitionSpec = AcquisitionSpec()
    equations: EquationBinding = EquationBinding()
    containment: ContainmentSpec = ContainmentSpec()
    training_pipeline: tuple[TrainingStageSpec, ...] = DEFAULT_TRAINING_PIPELINE

    def validate_program(self) -> list[str]:
        """Fail-closed cross-field validation. Returns [] when clean."""
        problems: list[str] = []
        if not any(s.source == "run_telemetry" and s.enabled for s in self.sensors):
            problems.append("no enabled run_telemetry sensor — the organ would be blind")
        if not any(a.action == "spawn_reasoning_agent" for a in self.actuators):
            problems.append("no spawn_reasoning_agent actuator — System-2 escalation "
                            "is part of the #426 contract")
        preds = [e for e in self.experts if e.predictive]
        if not preds:
            problems.append("no predictive expert — the adjoint has no λ source")
        if self.routing.interpretable_head and not any(
                e.architecture == "E_prototype" for e in self.experts):
            problems.append("interpretable_head=True requires the E_prototype (PRISM) expert "
                            "in the panel — the interpretable router head is a hard requirement")
        try:
            self.equations.resolve()
        except Exception as exc:  # registry-missing is a validation failure, not a crash
            problems.append(str(exc))
        return problems

    def compile(self) -> "CompiledCostateOrgan":
        problems = self.validate_program()
        if problems:
            raise ValueError("CostateAgentProgram invalid: " + "; ".join(problems))
        return CompiledCostateOrgan(program=self)

    # ---- the generic CONSUMER surface (digest/dashboard render by introspection;
    # the consumer-leg discipline: a DSL nobody renders is orphaned signal) ----
    def describe(self) -> dict:
        """Machine-readable introspection of the whole program (pydantic dump +
        derived summaries) — the generic render surface consumers introspect."""
        d = self.model_dump()
        d["_derived"] = {
            "n_sensors_enabled": sum(1 for s in self.sensors if s.enabled),
            "n_actuators_safe": sum(1 for a in self.actuators if a.cost_tier == "SAFE"),
            "n_actuators_heavy_gated": sum(
                1 for a in self.actuators
                if a.cost_tier == "HEAVY" and a.operator_go_gated),
            "n_experts_predictive": sum(1 for e in self.experts if e.predictive),
            "expert_architectures": sorted({e.architecture for e in self.experts}),
            "routing_mode": self.routing.mode,
            "single_best_lens": self.routing.single_best_lens,
            "containment_envelope": self.containment.autonomous_envelope,
            "training_stages": {s.name: s.status for s in self.training_pipeline},
        }
        return d

    def render_lines(self, *, max_width: int = 110) -> list[str]:
        """Human-readable one-screen render (the costate-digest / dashboard surface)."""
        dd = self.describe()["_derived"]
        lines = [
            f"costate program {self.name!r} @ {self.run_dir}",
            f"  sensors: {dd['n_sensors_enabled']} enabled "
            f"({', '.join(s.name for s in self.sensors if s.enabled)})",
            f"  actuators: {dd['n_actuators_safe']} SAFE + "
            f"{dd['n_actuators_heavy_gated']} HEAVY (all operator-GO-gated; "
            "containment typed)",
            f"  experts: {dd['n_experts_predictive']} predictive of {len(self.experts)} "
            f"({', '.join(e.name for e in self.experts)})",
            f"  routing: {self.routing.mode} (single_best={self.routing.single_best_lens}; "
            f"spread_gate={'on' if self.routing.spread_gate else 'off'}; "
            f"interpretable_head={'on' if self.routing.interpretable_head else 'OFF'})",
            f"  acquisition: {self.acquisition.policy} / {self.acquisition.proof_gate}",
            f"  laws: {self.equations.master_action_id} + {self.equations.costate_law_id}",
            f"  containment: {self.containment.autonomous_envelope}; heavy⇒operator-GO; "
            "spawn inherits clause verbatim",
        ]
        return [ln[:max_width] for ln in lines]


@dataclass(frozen=True)
class CompiledCostateOrgan:
    """The REAL wiring the program compiles to (sense → adjoint → decide → act).

    Every method delegates to the actual built surfaces (lambda_net / costate_panel /
    control_alphabet) — nothing here is a stub; the test suite exercises the compiled
    organ end-to-end on the live run directory's telemetry."""

    program: CostateAgentProgram

    # SENSE — the observation operator (read-only)
    def sense(self):
        from tac.witness_control.lambda_net import read_trajectory
        return read_trajectory(self.program.run_dir)

    # ADJOINT — the λ-panel (fused per the declared routing mode)
    def adjoint(self, *, inner_skill: dict | None = None, seed: int = 0,
                status: str = "SPECULATIVE-UNTIL-BACKTESTED"):
        from tac.witness_control.costate_panel import run_panel
        return run_panel(self.sense(), routing_mode=self.program.routing.mode,
                         seed=seed, status=status, inner_skill=inner_skill,
                         single_best_lens=self.program.routing.single_best_lens,
                         spread_gate=self.program.routing.spread_gate)

    # DECIDE — closed-form pointwise Hamiltonian minimization given λ (ANR recipe)
    def decide(self, marginal_ds: dict[str, float], current_shares: dict[str, float],
               *, sealed: tuple[str, ...] = (), budget: float = 0.10,
               tier: str = "SPECULATIVE-UNTIL-BACKTESTED") -> ControlDecision:
        return hamiltonian_decide(marginal_ds, current_shares, sealed=sealed,
                                  budget=budget, tier=tier)

    # ACT — the act-time convex projection (the containment governor)
    def act(self, action: str, **kwargs) -> str | OperatorGoTicket:
        declared = {a.action for a in self.program.actuators}
        if action not in declared:
            raise ValueError(f"action {action!r} not declared by program "
                             f"{self.program.name!r} — the DSL is the closed alphabet")
        return feasibility_projection(action, **kwargs)

    # SPAWN — bounded System-2 (returns the ticket; the harness executes)
    def spawn(self, question: str, trigger: str, *, agent_kind: str = "claude_code",
              context_lines: tuple[str, ...] = ()) -> SpawnTicket:
        if not any(a.action == "spawn_reasoning_agent" for a in self.program.actuators):
            raise ValueError("program declares no spawn actuator")
        return compose_spawn_ticket(question, trigger, agent_kind=agent_kind,
                                    context_lines=context_lines)

    # OBSERVE — the OBSERVATORY: which prototype regimes fired, weights, uncertainty,
    # traceable to the trajectory neighborhood (the interpretable-by-design telemetry).
    def observe(self, *, seed: int = 0):
        from tac.witness_control.lambda_net import (
            build_intervals, fit_score_composition, lever_features)
        from tac.witness_control.prototype_router import PrototypeRouterLens
        import numpy as np
        traj = self.sense()
        comp = fit_score_composition(traj.verdicts)
        intervals = build_intervals(traj)
        phis = np.stack([lever_features(n) for n in traj.lever_names])
        lens = PrototypeRouterLens()
        lens.fit(intervals, phis, seed=seed)
        last = traj.verdicts[-1]
        x = np.concatenate([np.asarray(last["d_seg_by_class"]),
                            [np.log(max(float(last["blob_bytes"]), 1.0))]])
        return lens.attribute(x, float(last["epoch"]), comp.grad_s_wrt_state())

    # CURIOSITY — the self-inventing curriculum (PowerPlay) + the Gödel proof-gate
    def acquire(self, lam_field: dict[str, float],
                measured_binding: dict[str, float] | None = None,
                ever_fired: dict[str, bool] | None = None,
                costs: dict[str, float] | None = None):
        from tac.witness_control.control_alphabet import powerplay_acquisition
        return powerplay_acquisition(lam_field, measured_binding, ever_fired, costs)

    def prove_improvement(self, change: str, backtest_passed: bool,
                          predicted_delta_s: float | None):
        from tac.witness_control.control_alphabet import GodelProofGate
        return GodelProofGate.evaluate(change, backtest_passed, predicted_delta_s)

    # SELF-MONITOR — the meta-λ head (the organ probes its OWN activation state and
    # flags when the primary λ is untrustworthy; typed corrections, advisory only)
    def self_monitor(self, *, seed: int = 0):
        from tac.witness_control.self_monitor import self_activation_probe
        return self_activation_probe(self.sense(), seed=seed)

    # REFLECT — one GEPA cycle (reflection proposes, backtest disposes; Pareto
    # frontier over measured candidates; adoption only through the Gödel gate)
    def reflect(self, arch_reports: dict, *, incumbent: str = "E_prototype",
                seed: int = 0):
        from tac.witness_control.gepa_reflection import run_gepa_cycle
        return run_gepa_cycle(self.sense(), arch_reports, incumbent=incumbent,
                              seed=seed)

    # COMPOUND — persist this trajectory's organ fit into the triality ledger
    # (graph-memory indexes it; the next session reconstructs a smarter organ)
    def compound(self, arch_reports: dict, prototypes: list,
                 duty_top: list[dict] | None = None) -> str:
        from tac.witness_control.continual_costate import (
            append_trajectory_record, compose_trajectory_record)
        rec = compose_trajectory_record(self.sense(), arch_reports, prototypes,
                                        duty_top)
        return append_trajectory_record(rec)


# ─────────────────────────────────────────────────────────────────────────────
# The canonical #426 program — the organ AS SHIPPED (mirrors the measured verdicts:
# routing mode justified by the routing benchmark; experts = the real lens panel).
# ─────────────────────────────────────────────────────────────────────────────
def derive_costate_agent_v1(run_dir: str) -> CostateAgentProgram:
    return CostateAgentProgram(
        name="costate_agent_v1",
        run_dir=run_dir,
        sensors=(
            SensorSpec(name="verdict_stream", source="run_telemetry",
                       fields=("d_seg", "d_seg_by_class", "flip_share_by_class",
                               "d_pose", "blob_bytes", "ep_loss")),
            SensorSpec(name="loss_term_stream", source="run_telemetry",
                       fields=("terms", "gnorm", "softmax_temp", "hosc_beta")),
            SensorSpec(name="lever_precedent", source="activation_ledger"),
            SensorSpec(name="law_lookup", source="canonical_equations"),
            SensorSpec(name="scorer_geometry", source="cached_probe",
                       fields=("margins", "lstars", "flip_susceptibility",
                               "uniward_cost")),
            SensorSpec(name="organ_ledger_memory", source="graph_memory",
                       fields=("regime_nodes", "arch_records", "graduation")),
        ),
        actuators=(
            ActuatorSpec(action="emit_recommendation", cost_tier="SAFE"),
            ActuatorSpec(action="rank_duty_to_measure", cost_tier="SAFE"),
            ActuatorSpec(action="run_cached_probe", cost_tier="SAFE"),
            ActuatorSpec(action="compile_dsl_config", cost_tier="SAFE",
                         witness_lever_refs=("*",)),
            ActuatorSpec(action="query_canonical_equations", cost_tier="SAFE"),
            ActuatorSpec(action="write_memo", cost_tier="SAFE"),
            ActuatorSpec(action="spawn_reasoning_agent", cost_tier="SAFE",
                         inherits_containment=True),
            ActuatorSpec(action="launch_training", cost_tier="HEAVY",
                         operator_go_gated=True),
            ActuatorSpec(action="stop_run", cost_tier="HEAVY", operator_go_gated=True),
            ActuatorSpec(action="mutate_live_config", cost_tier="HEAVY",
                         operator_go_gated=True),
            ActuatorSpec(action="spend_paid_gpu", cost_tier="HEAVY",
                         operator_go_gated=True),
        ),
        experts=tuple(
            ExpertSpec(name=s.name, architecture=s.architecture, question=s.question,
                       tools_exercised=s.tools_exercised, predictive=s.predictive,
                       activation_note=("prior-favored (low complexity)" if s.name == "flow"
                                        else "prior-weighted until measured skill accrues"))
            for s in LENSES),
        acquisition=AcquisitionSpec(
            provenance="Schmidhuber self-improvement lineage (2026-07-11): PowerPlay "
                       "1112.5309 self-inventing curriculum · curiosity=innovation "
                       "signal (derived-λ − measured-binding = compression progress) · "
                       "Gödel-machine cs/0309048 proof-of-improvement = backtest-passed "
                       "AND predicted-ΔS<0 AND containment · LADDER ⊂ costate (L56)"),
        routing=RoutingSpec(
            mode="SINGLE_BEST",
            single_best_lens="prototype",
            escalation="both",
            spread_gate=True,
            interpretable_head=True,
            interpretable_provenance="PRISM 2607.00510 prototype router (E_prototype "
                       "lens): interpretable-by-design HARD requirement (Rudin co-lead). "
                       "RE-ARBITRATED 2026-07-11 adversarial-review pass on the EXTENDED "
                       "#205 trajectory (7 intervals): E_prototype is now the measured "
                       "WINNER outright — LOO fMAE 0.002744 (flow 0.002911) AND "
                       "walk-forward 0.002479 vs persistence 0.003111 (flow LOSES "
                       "walk-forward 0.003370) AND per-class 0.009744. Interpretability "
                       "now costs NOTHING — the interpretable head is also the best "
                       "predictor at this n. Faithfulness additionally AUDITED via the "
                       "suppression counterfactual (2607.08046 probe-vs-story).",
            provenance="re-arbitration 2026-07-11 (adversarial-review round 1; "
                       "costate_organ_backtest_20260711T113716Z.json + walk-forward "
                       "re-derivation): the earlier n=5 verdict (SINGLE_BEST=flow "
                       "0.002881, ridge 4.2x per-class) did NOT survive 2 new "
                       "intervals — flow FAILS the extended gate (binding AUROC 0.96; "
                       "walk-forward LOSES to persistence). verdict_scope: instance "
                       "(one trajectory, re-arbitrated per record accrual via "
                       "continual_costate.arbitrate_architecture — the organ ledger "
                       "is the compounding memory)"),
    )


def derive_costate_agent_arbitrated(run_dir: str) -> CostateAgentProgram:
    """The COMPOUNDING derivation: consult the organ's triality ledger and set the
    SINGLE_BEST lens from the accumulated cross-trajectory arbitration (measured
    graduation; falls back to the v1 program when no records exist)."""
    prog = derive_costate_agent_v1(run_dir)
    try:
        from tac.witness_control.continual_costate import (
            arbitrate_architecture, load_organ_memory)
        arb = arbitrate_architecture(load_organ_memory())
        if arb.n_records == 0:
            return prog
        arch_to_lens = {e.architecture: e.name for e in prog.experts}
        lens = arch_to_lens.get(arb.recommended)
        if lens is None:
            return prog
        routing = prog.routing.model_copy(update={
            "single_best_lens": lens,
            "provenance": (f"organ-ledger arbitration over {arb.n_records} record(s), "
                           f"{arb.total_intervals} intervals: recommended "
                           f"{arb.recommended} ({arb.recommendation_basis}). "
                           + prog.routing.provenance)})
        return prog.model_copy(update={"routing": routing})
    except Exception:
        return prog                      # fail-open to the static program (advisory)
