# SPDX-License-Identifier: MIT
"""The gated control alphabet — task #426 ACT side of the Amortized-Operator Pontryagin Loop.

Realizes the Adjoint-Neural-Regulator act-time split (arXiv 2606.16303, harvested
2026-07-11): DECIDE is the closed-form pointwise Hamiltonian minimization given λ
(u* = argmin_u H(x,u,λ) — trivially cheap once λ is what's modeled), and FEASIBILITY
is an act-time convex PROJECTION *outside* λ-learning. Our projection IS the
containment governor: the feasible set is {SAFE $0/score-neutral actions executable
now} × {HEAVY actions → OperatorGoTicket}, plus sealed-config knobs projected to zero.

CONTAINMENT (structural, verified by the package source-scan test): this module has
NO process/exec/signal surface. A HEAVY action's ``execute`` path does not exist —
requesting one RETURNS an ``OperatorGoTicket`` (a typed refusal-with-routing: the
governed launcher + operator GO are the only paths that can realize it). The
agent-spawn actuator returns a ``SpawnTicket`` — a fully-composed prompt artifact
(via ``tac.subagent_contract.standard_contract``) with the containment clause
INHERITED verbatim; the harness (a human-supervised Claude Code session) is the only
executor. The recursion is bounded: every spawned agent lives inside the same
advisory/$0 envelope, and any heavy action IT would take routes through the same
ticket type.

Every artifact is advisory: ``[macOS advisory] NON-PROMOTABLE``, score_claim=False.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from datetime import UTC, datetime


# ── cost tiers (the act-time feasibility split) ──
SAFE = "SAFE"          # $0, score-neutral, advisory-artifact-producing — executable
HEAVY = "HEAVY"        # paid / run-mutating / stop / live-config — operator-GO ONLY

#: the SAFE-native alphabet (each names the artifact it produces; no side effects
#: beyond advisory rows/files written by the CALLER, never by this module)
SAFE_ACTIONS = (
    "emit_recommendation",        # ranked advisory rows (costate-shadow style)
    "rank_duty_to_measure",       # order the never-fired lever queue by |λ-field|
    "run_cached_probe",           # $0 probes on cached tensors (no scorer forward)
    "compile_dsl_config",         # WitnessProgram compile+validate (no launch)
    "query_canonical_equations",  # law lookup
    "byte_close_on_cached",       # rate rows from cached artifacts
    "write_memo",                 # durable .omx/research note text
    "spawn_reasoning_agent",      # SpawnTicket (bounded System-2; see SpawnTicket)
)
#: the HEAVY alphabet — every member returns an OperatorGoTicket, never executes
HEAVY_ACTIONS = (
    "launch_training", "stop_run", "mutate_live_config", "spend_paid_gpu",
    "dispatch_remote", "signal_process",
)


@dataclass(frozen=True)
class OperatorGoTicket:
    """A typed refusal-with-routing for a HEAVY action (a REFUSE is information).

    The ticket carries everything the operator needs to say GO: the action, its
    λ-justification (predicted ΔS with its honesty tier), the exact command the
    governed launcher would run, and the gates still owed. It cannot execute."""

    action: str
    justification: str            # λ-readout + tier (ANALYTIC/MEASURED/PARTIAL/SPECULATIVE)
    predicted_delta_s: float | None
    governed_command: str         # what the operator would run (launcher path), verbatim
    gates_owed: tuple[str, ...]
    created_utc: str = _dc_field(default_factory=lambda: datetime.now(UTC).isoformat())
    actuation: str = "NONE"       # structural: the ticket is the terminal artifact
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"

    def __post_init__(self) -> None:
        if self.action not in HEAVY_ACTIONS:
            raise ValueError(f"OperatorGoTicket is only for HEAVY actions; got {self.action!r}")


@dataclass(frozen=True)
class SpawnTicket:
    """A bounded reasoning-agent spawn request (the System-2 escalation artifact).

    INHERITED CONTAINMENT (non-negotiable, embedded in the prompt itself): the spawned
    agent is advisory-only autonomous — read/reason/$0-probe/propose; ANY heavy/paid/
    stop/live-config action it would take is operator-GO through the governed launcher.
    The prompt composes ``tac.subagent_contract.standard_contract()`` so the spawned
    agent carries the full grounded-progress + serializer + triality discipline.
    Execution belongs to the harness (the supervising Claude Code session / Agent tool)
    — this dataclass is inert by construction."""

    question: str                 # the routed System-2 question (novel/uncertain/high-stakes)
    trigger: str                  # why the panel escalated (disagreement/low-confidence/OOS)
    prompt: str                   # the FULL composed prompt (contract + containment + question)
    agent_kind: str = "claude_code"   # claude_code | codex
    envelope: str = "advisory-$0"     # the spawned agent's whole authority envelope
    requires_harness: bool = True     # structural: nothing in-process can execute this
    created_utc: str = _dc_field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        # containment is TYPE-ENFORCED, not composer-enforced: a directly-constructed
        # ticket whose prompt lacks the inherited-containment clause is refused
        # (adversarial-review hardening 2026-07-11 — the composer embedded the clause
        # but the type did not require it; that was a bypass).
        if INHERITED_CONTAINMENT_CLAUSE not in self.prompt:
            raise ValueError("SpawnTicket.prompt MUST embed INHERITED_CONTAINMENT_CLAUSE "
                             "verbatim (compose via compose_spawn_ticket)")
        if self.requires_harness is not True:
            raise ValueError("SpawnTicket.requires_harness is structurally True")


#: the containment clause every spawned agent inherits VERBATIM (prompt-embedded)
INHERITED_CONTAINMENT_CLAUSE = (
    "INHERITED CONTAINMENT (binding, from the #426 costate organ that spawned you): you "
    "are advisory-only autonomous. You may read, reason, run $0 cached/through-R probes, "
    "and propose. ANY heavy/paid/stop/live-config action (training launch, paid GPU, "
    "stopping a run, mutating a live config) is operator-GO through the governed "
    "launcher — emit an OperatorGoTicket-style request instead of acting. Do NOT disturb "
    "any live training run. If you spawn further agents, this clause propagates to them "
    "verbatim. Pointer moves only through byte-closed exact eval; nothing you produce is "
    "a score."
)


def compose_spawn_ticket(question: str, trigger: str, *, agent_kind: str = "claude_code",
                         context_lines: tuple[str, ...] = ()) -> SpawnTicket:
    """Compose a fully-formed, containment-inheriting spawn request (does NOT spawn)."""
    from tac.subagent_contract import standard_contract  # canonical composer (no dup)
    body = "\n".join((
        "You are a bounded reasoning agent spawned by the #426 costate organ "
        "(the Amortized-Operator Pontryagin Loop's System-2 expert).",
        "",
        INHERITED_CONTAINMENT_CLAUSE,
        "",
        "ESCALATION TRIGGER: " + trigger,
        "",
        "QUESTION: " + question,
        *(("", "CONTEXT:") + context_lines if context_lines else ()),
        "",
        standard_contract(review=True, triality=True),
    ))
    return SpawnTicket(question=question, trigger=trigger, prompt=body,
                       agent_kind=agent_kind)


# ─────────────────────────────────────────────────────────────────────────────
# DECIDE — closed-form pointwise Hamiltonian minimization given λ (the ANR recipe).
# With the response model linear in the control shares u (see lambda_net), the
# Hamiltonian H = λ_xᵀ(base + Σ_i Λ_i u_i) is LINEAR in u on the share simplex —
# so argmin is closed-form: allocate the movable budget onto the most-negative
# marginal-ΔS levers (a vertex / waterfill), then PROJECT onto the feasible set.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ControlDecision:
    """u* + the act-time projection outcome. Advisory: proposed shares, never applied."""

    proposed_shares: dict[str, float]     # the closed-form u* (post-projection)
    frozen: tuple[str, ...]               # sealed knobs projected to zero change
    marginal_ds: dict[str, float]         # the λ-field used (per lever)
    hamiltonian_value: float              # λᵀ f(x, u*) — predicted dS/dep at u*
    tier: str                             # honesty tier of the λ inputs
    actuation: str = "NONE"


def hamiltonian_decide(marginal_ds: dict[str, float], current_shares: dict[str, float],
                       *, sealed: tuple[str, ...] = (), budget: float = 0.10,
                       tier: str = "SPECULATIVE-UNTIL-BACKTESTED") -> ControlDecision:
    """Closed-form u* on the share simplex with an act-time feasibility projection.

    Moves at most ``budget`` of total share mass from the worst (most-positive
    marginal-ΔS) unsealed levers onto the best (most-negative) unsealed lever —
    the linear-program vertex of min_u λᵀf s.t. Δu ∈ budget-ball ∩ simplex ∩
    {sealed components frozen}. Everything is a PROPOSAL (actuation NONE)."""
    unsealed = {k: v for k, v in marginal_ds.items()
                if k not in sealed and k in current_shares}
    if not unsealed:
        return ControlDecision(proposed_shares=dict(current_shares), frozen=tuple(sealed),
                               marginal_ds=dict(marginal_ds), hamiltonian_value=0.0, tier=tier)
    best = min(unsealed, key=lambda k: unsealed[k])
    donors = sorted((k for k in unsealed if k != best),
                    key=lambda k: unsealed[k], reverse=True)
    out = dict(current_shares)
    remaining = float(budget)
    for d in donors:
        if remaining <= 0 or unsealed[d] <= unsealed[best]:
            break
        take = min(out.get(d, 0.0), remaining)
        out[d] = out.get(d, 0.0) - take
        out[best] = out.get(best, 0.0) + take
        remaining -= take
    h = sum(marginal_ds.get(k, 0.0) * v for k, v in out.items())
    return ControlDecision(proposed_shares=out, frozen=tuple(sealed),
                           marginal_ds=dict(marginal_ds), hamiltonian_value=float(h),
                           tier=tier)


def feasibility_projection(action: str, *, justification: str = "",
                           predicted_delta_s: float | None = None,
                           governed_command: str = "tools/launch_witness_run.py (operator-GO)",
                           gates_owed: tuple[str, ...] = ()) -> str | OperatorGoTicket:
    """The act-time convex projection: SAFE actions pass through (caller executes the
    advisory artifact); HEAVY actions project onto the operator-GO boundary (ticket)."""
    if action in SAFE_ACTIONS:
        return action
    if action in HEAVY_ACTIONS:
        return OperatorGoTicket(action=action, justification=justification,
                                predicted_delta_s=predicted_delta_s,
                                governed_command=governed_command, gates_owed=gates_owed)
    raise ValueError(f"unknown action {action!r} — the alphabet is closed "
                     f"(SAFE={SAFE_ACTIONS}, HEAVY={HEAVY_ACTIONS}); never-invent-tools")


# ─────────────────────────────────────────────────────────────────────────────
# SCHMIDHUBERIAN SELF-IMPROVEMENT (the self-improving half of the loop, 2026-07-11).
# The costate organ is a Schmidhuberian self-improver; three principles encoded:
#   * PowerPlay (arXiv 1112.5309): the duty-to-measure queue is a SELF-INVENTING
#     curriculum — generate the SIMPLEST not-yet-solved probe that provably extends
#     competence without breaking what works. Rank by curiosity × blast-radius / cost.
#   * Gödel machine (arXiv cs/0309048): self-modify/actuate ONLY with a proof-of-
#     improvement = our backtest-IS-the-gate + the containment governor (already built).
#     `godel_proof_gate` names the correspondence and enforces it structurally.
#   * Artificial curiosity / compression progress: novelty = where the current λ-model
#     is SURPRISED = the innovation signal (derived-λ − measured-binding). That signal
#     IS the acquisition function — high-innovation regions rank UP.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AcquisitionRow:
    """One PowerPlay-ranked probe candidate with its full curiosity decomposition."""

    lever: str
    curiosity: float               # innovation |derived-λ − measured-binding| (surprise)
    blast_radius: float            # |λ-field| (marginal-ΔS magnitude = how much it could move S)
    cost: float                    # measurement cost (cheaper = simpler = PowerPlay-preferred)
    acquisition: float             # curiosity × blast_radius / cost (the rank key)
    ever_fired: bool
    kind: str                      # "exploit-surprise" | "explore-unknown"
    tier: str
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"


def powerplay_acquisition(lam_field: dict[str, float],
                          measured_binding: dict[str, float] | None = None,
                          ever_fired: dict[str, bool] | None = None,
                          costs: dict[str, float] | None = None,
                          *, default_cost: float = 1.0) -> list[AcquisitionRow]:
    """The self-inventing curriculum: rank probes by curiosity × blast-radius / cost.

    Curiosity = the INNOVATION SIGNAL (Schmidhuber's compression-progress / surprise):
    for a fired lever, |predicted λ-field − measured realized binding| (the model was
    WRONG by this much = it will LEARN this much). For a never-fired lever there is no
    measured binding → curiosity is the field magnitude itself (pure exploration; the
    unknown is maximally surprising by construction, PowerPlay's frontier). Blast-radius
    = |λ-field| (how much S it could move). Cost = measurement cost (PowerPlay prefers the
    SIMPLEST — cheapest — not-yet-solved probe). NO actuation — a ranked ADVISORY queue."""
    ever_fired = ever_fired or {}
    costs = costs or {}
    measured = measured_binding or {}
    rows: list[AcquisitionRow] = []
    for lever, lam in lam_field.items():
        blast = abs(float(lam))
        cost = float(costs.get(lever, default_cost))
        fired = bool(ever_fired.get(lever, False))
        if fired and lever in measured:
            curiosity = abs(float(lam) - float(measured[lever]))
            kind = "exploit-surprise"
            tier = "MEASURED innovation (predicted-vs-realized gap = compression progress)"
        else:
            curiosity = blast
            kind = "explore-unknown"
            tier = "PARTIAL frontier (never-fired/unmeasured — pure exploration; the " \
                   "unknown is maximally surprising by PowerPlay construction)"
        acq = curiosity * blast / max(cost, 1e-9)
        rows.append(AcquisitionRow(lever=lever, curiosity=curiosity, blast_radius=blast,
                                   cost=cost, acquisition=acq, ever_fired=fired,
                                   kind=kind, tier=tier))
    rows.sort(key=lambda r: -r.acquisition)
    return rows


@dataclass(frozen=True)
class GodelProofGate:
    """Gödel-machine correspondence: a self-change is admissible ONLY with a proof of
    improvement. Our 'proof' = the backtest verdict (held-out forecast beats the incumbent)
    AND the never-regress guard (predicted ΔS < 0) AND containment (HEAVY ⇒ operator-GO).
    This gate NAMES and enforces the correspondence; it authorizes nothing on its own —
    a True verdict still only unlocks emitting an advisory rec / OperatorGoTicket."""

    change: str
    backtest_passed: bool          # the held-out forecast gate (lambda_net.backtest)
    predicted_delta_s: float | None  # never-regress: must be < 0 to be an improvement
    admissible: bool
    reason: str
    actuation: str = "NONE"

    @staticmethod
    def evaluate(change: str, backtest_passed: bool,
                 predicted_delta_s: float | None) -> "GodelProofGate":
        improves = (predicted_delta_s is not None and predicted_delta_s < 0.0)
        admissible = bool(backtest_passed and improves)
        if admissible:
            reason = (f"proof-of-improvement HOLDS: backtest passed AND predicted ΔS "
                      f"{predicted_delta_s:+.5f} < 0. Advisory rec / OperatorGoTicket "
                      "may be emitted (still operator-GO for HEAVY).")
        elif not backtest_passed:
            reason = "REFUSED: no backtest proof (held-out forecast did not beat incumbent)."
        else:
            reason = (f"REFUSED: predicted ΔS {predicted_delta_s} is not an improvement "
                      "(never-regress guard).")
        return GodelProofGate(change=change, backtest_passed=backtest_passed,
                              predicted_delta_s=predicted_delta_s, admissible=admissible,
                              reason=reason)

    @staticmethod
    def evaluate_from_report(change: str, report, predicted_delta_s: float | None,
                             *, law_ref: str = "cgauge_master_action_v1",
                             require_law: bool = True) -> "GodelProofGate":
        """Provenance-typed proof path (adversarial-review hardening 2026-07-11): the
        backtest verdict is READ from a ``BacktestReport`` (LOO ∧ walk-forward when the
        walk-forward gate was computable), never caller-asserted; and the proof must
        REFERENCE a registered law (triality-native: proofs cite equations, not inline
        heuristics — fail-closed if the law is absent from the canonical registry)."""
        if require_law:
            from tac.canonical_equations import query_equations
            ids = {getattr(r, "equation_id", None) for r in query_equations()}
            if law_ref not in ids:
                return GodelProofGate(
                    change=change, backtest_passed=False, predicted_delta_s=predicted_delta_s,
                    admissible=False,
                    reason=f"REFUSED: law_ref {law_ref!r} not in the canonical equations "
                           "registry (a proof must reference a registered law)")
        passed = bool(getattr(report, "passed", False))
        return GodelProofGate.evaluate(change, passed, predicted_delta_s)


@dataclass(frozen=True)
class ScheduleBundleRec:
    """A COHERENT curriculum-schedule recommendation (task #430): coordinated lever
    BUNDLES in a state-gated cascade — not independent per-lever recommendations.

    Operator binding 2026-07-11: the hand-built curriculum's independently-clocked
    levers DESYNCHRONIZE (τ frozen while the boundary erodes); the organ's control
    output is the joint schedule. Advisory: the bundle is emitted INSIDE an
    OperatorGoTicket (mutate_live_config is HEAVY) — the organ recommends the whole,
    the operator launches. SPECULATIVE-UNTIL-BACKTESTED until the #205-replay A/B
    (hand-schedule vs organ-schedule) is measured — stated on the artifact."""

    stages: tuple[dict, ...]        # ordered [{gate: <state condition>, bundle: [levers], why}]
    joint_objective: str            # what the cascade jointly minimizes
    tier: str
    status: str = "SPECULATIVE-UNTIL-BACKTESTED (replay A/B owed)"
    actuation: str = "NONE"
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"


def compose_schedule_bundle_ticket(marginal_ds: dict[str, float],
                                   state_flags: dict[str, bool], *,
                                   tier: str = "SPECULATIVE-UNTIL-BACKTESTED",
                                   ) -> OperatorGoTicket:
    """Compose the coherent state-gated cascade rec + wrap it in the HEAVY ticket.

    The cascade template is the measured #205 flow (island-birth → boundary-form →
    τ-sharpen⊕repair → finish); the λ-field ORDERS the bundles (most-negative
    marginal-ΔS levers lead their stage) and the state flags gate the stages."""
    lam_sorted = sorted(marginal_ds, key=lambda k: marginal_ds[k])
    repair = [n for n in lam_sorted if n in
              ("chroma_boundary", "lane_edge", "thin_lane", "subpix", "margin_saliency")]
    birth = [n for n in lam_sorted if n in
             ("island_amplify", "area_constraint", "persistence")]
    stages = (
        {"gate": "per-class islands unborn (Movable/Lane d_seg near init)",
         "bundle": birth or ["island_amplify"],
         "why": "nucleate before any boundary work (measured island-birth precedence)"},
        {"gate": "islands born ∧ boundary forming (annulus share rising)",
         "bundle": ["seg", "eikonal"],
         "why": "form the level-set boundary before sharpening it"},
        {"gate": "boundary formed ∧ NOT eroding (no TRAIN_VERDICT_DECOUPLING)",
         "bundle": ["tau_advance", *(repair[:2] or ["chroma_boundary"])],
         "why": "advance τ TOGETHER WITH matched repair — the desynchronization the "
                "hand-schedule's independent clocks caused (operator diagnosis)"},
        {"gate": "plateau-slope < threshold ∧ verdict/train coupled",
         "bundle": ["muon_finish"],
         "why": "finishing optimizer only on a coherent boundary"},
    )
    rec = ScheduleBundleRec(
        stages=stages,
        joint_objective="min d_seg × train-time over the WHOLE cascade "
                        "(joint synergy, not per-lever EV)", tier=tier)
    active = [s for s in stages
              if not state_flags.get(f"stage_done:{s['bundle'][0]}", False)]
    return OperatorGoTicket(
        action="mutate_live_config",
        justification=(f"#430 coherent schedule bundle ({len(active)} stage(s) "
                       f"pending; tier {tier}): " +
                       " → ".join("+".join(s["bundle"]) for s in rec.stages) +
                       f" | objective: {rec.joint_objective} | status: {rec.status}"),
        predicted_delta_s=None,
        governed_command="tools/launch_witness_run.py --resume-from <ckpt> "
                         "(operator-GO; schedule via the witness DSL compile)",
        gates_owed=("#205-replay A/B: organ-schedule vs hand-schedule (backtest gate)",
                    "witness DSL compile of the bundle (never-invent-flags)"))


def rank_duty_to_measure(lam_field: dict[str, float], identified: dict[str, bool],
                         ever_fired: dict[str, bool]) -> list[dict]:
    """SAFE actuator: order never-fired levers by |λ-field| (the what-if shadow price).

    The duty-to-measure queue is the costate's EXPLORATION set (master action §2);
    field values for never-fired levers are FEATURE-STRUCTURED (PARTIAL tier) — the
    ranking is a measurement PRIORITY, never a measured effect."""
    rows = []
    for name, lam in lam_field.items():
        if ever_fired.get(name, False):
            continue
        rows.append({
            "lever": name, "lambda_field": float(lam),
            "abs_lambda": abs(float(lam)),
            "tier": "PARTIAL (feature-structured; never-fired — this row is a "
                    "measurement priority, not a measured effect)",
            "identified": bool(identified.get(name, False)),
        })
    rows.sort(key=lambda r: -r["abs_lambda"])
    return rows
