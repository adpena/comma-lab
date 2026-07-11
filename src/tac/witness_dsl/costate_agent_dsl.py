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
    stacking_prior_strength: float = Field(default=3.0, ge=0.0)
    self_activation_cap: float = Field(default=0.5, gt=0.0, le=1.0)
    escalation: Literal["rashomon_action_disagreement", "dispersion", "both"] = "both"
    provenance: str = ""   # the benchmark row that justified the mode choice

    @model_validator(mode="after")
    def _known_mode(self) -> "RoutingSpec":
        if self.mode not in ROUTING_MODES:
            raise ValueError(f"routing mode {self.mode!r} not in {ROUTING_MODES}")
        return self


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
    equations: EquationBinding = EquationBinding()
    containment: ContainmentSpec = ContainmentSpec()

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
                         seed=seed, status=status, inner_skill=inner_skill)

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
        routing=RoutingSpec(
            mode="SINGLE_BEST",
            escalation="both",
            provenance="routing benchmark 2026-07-11 FINAL (nested LOO, 5 folds, "
                       "#205 trajectory, routing_benchmark_final_20260711.json): "
                       "SINGLE_BEST=flow 0.002881 WINNER (=QUESTION_ROUTER) · "
                       "COMPONENT_FUSION 0.003023 (best per-class 0.085805) · "
                       "EVIDENCE_SHRUNK_STACKING 0.005060 (beats persistence "
                       "heuristic 0.005632; the declared GROWTH FORM — re-arbitrate "
                       "as folds accrue per 2101.08954) · SELF_ACTIVATION (raw "
                       "train-confidence) 0.068 FALSIFIED. Parsimony wins at n=5 "
                       "(kitchen_sink anti-pattern honored by measurement); the "
                       "panel stays for question/tool coverage + disagreement + "
                       "duty ranking, not forecast fusion"),
    )
