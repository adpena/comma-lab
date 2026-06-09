"""FrozenEvaluatorContract — the evaluator-PLUGGABLE abstraction (operator 2026-06-09).

The operator's question: "isn't there a program that ingests a file like evaluate.py,
verifies it satisfies a contract (or exposes one engineered against on the other end),
then applies the FROZEN evaluator to the FROZEN overfit-authorized input, leveraging
infinite compress/training compute EXCEPT eval-time compute, to realize optimal?"

Answer: that program is V6 (the proof-carrying evaluator-equivalent program compiler).
Its defining generalization over the contest is that the EVALUATOR is pluggable. Today
`tac.contest_eval_contract` documents the comma evaluator's semantics, but it is
comma-specific. This module factors out the GENERAL contract so the SAME compiler
(V3 waterfiller + the Evidence Constitution) optimizes against ANY frozen
(evaluator, input) pair — the thousand-year "E generalizes to any downstream task".

The deep computer-science framing: this is a SPECIALIZING SUPEROPTIMIZER UNDER A FROZEN
ORACLE. Given a frozen evaluator E, a frozen overfit-authorized input X, an eval-runtime
bound T, and a byte cost, search (unbounded compress-time) for the SHORTEST witness
program p with E(I(p)) in E's equivalence class of X. The contract is the oracle
interface; the comma `evaluate.py` is one instance; overfit-authorization is what makes
the search well-posed (optimize for THIS input, not generalization). Bytes prevent
trivial memorization -> it is task-conditioned MDL for a fixed (X, E).

NO score authority lives here: this is the CONTRACT, not the scorer. Running the
evaluator + producing authority-tiered rows stays with the V3 ingest path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

COMPRESS_TIME_UNBOUNDED = "unbounded"  # infinite compress/training compute is authorized


@runtime_checkable
class FrozenEvaluator(Protocol):
    """The interface a pluggable evaluator MUST expose (or be wrapped to expose).

    A concrete evaluator (the comma `upstream/evaluate.py` behind the inflate pipeline,
    OR a synthetic/literature-derived goal) is engineered against this on the other end.
    The compiler only ever talks to this surface. ``candidate`` is the PROGRAM in
    whatever representation the evaluator defines: for comma it is the archive.zip path;
    for a synthetic goal it may be an in-memory array / blob. Type-agnostic by design so
    the SAME compiler drives any frozen (evaluator, input) pair off the shelf.
    """

    def score(self, *, source_ref: str, candidate: Any) -> float:
        """The scalar objective on the frozen input vs a candidate program."""

    def score_terms(self, *, source_ref: str, candidate: Any) -> dict[str, float]:
        """The DECOMPOSED objective (named terms, e.g. d_seg / d_pose / rate) so the
        waterfiller can attribute marginal effect per term. Decomposability is required
        (a black-box scalar alone forbids per-term waterfilling)."""

    def within_eval_budget(self, *, candidate: Any) -> bool:
        """True iff inflating + scoring the candidate fits the eval-runtime bound."""


@dataclass(frozen=True)
class FrozenEvaluatorContract:
    """The declarative contract a frozen (evaluator, input) pair must satisfy.

    The compiler is agnostic to the specific evaluator; it consumes this contract +
    the FrozenEvaluator surface. Overfit is AUTHORIZED on the frozen input by design
    (the contest authorizes overfitting the one scored video; future tasks may too)."""

    name: str
    source_ref: str  # the frozen, overfit-authorized input (e.g. upstream/videos/0.mkv)
    objective_formula: str  # human-readable; the EXACT scalar the evaluator computes
    objective_terms: tuple[str, ...]  # the decomposable named terms (e.g. d_seg/d_pose/rate)
    minimize: bool  # True => lower objective is better
    eval_runtime_budget_seconds: float  # the ONLY bounded compute (e.g. 1800 = 30 min)
    submission_boundary: str  # what constitutes the program (e.g. "archive.zip + inflate.sh")
    rate_denominator_bytes: int | None = None  # if the objective charges archive bytes / N
    compress_time_budget: str = COMPRESS_TIME_UNBOUNDED  # infinite train/search authorized
    overfit_authorized: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_terms:
            raise ValueError(
                f"{self.name}: objective_terms must be non-empty — a black-box scalar "
                "forbids per-term waterfilling (the compiler needs the decomposition)."
            )
        if self.eval_runtime_budget_seconds <= 0:
            raise ValueError(
                f"{self.name}: eval_runtime_budget_seconds must be > 0 (the bounded compute)."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "frozen_evaluator_contract.v1",
            "name": self.name,
            "source_ref": self.source_ref,
            "objective_formula": self.objective_formula,
            "objective_terms": list(self.objective_terms),
            "minimize": self.minimize,
            "eval_runtime_budget_seconds": self.eval_runtime_budget_seconds,
            "submission_boundary": self.submission_boundary,
            "rate_denominator_bytes": self.rate_denominator_bytes,
            "compress_time_budget": self.compress_time_budget,
            "overfit_authorized": self.overfit_authorized,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


def verify_frozen_evaluator_contract(contract: FrozenEvaluatorContract) -> dict[str, Any]:
    """Verify a contract is well-formed for the compiler. Returns a verdict dict; raises
    nothing (the caller decides). Checks the structural invariants the compiler relies on."""
    issues: list[str] = []
    if not contract.source_ref:
        issues.append("missing source_ref (the frozen input)")
    if not contract.objective_terms:
        issues.append("missing objective_terms (decomposition required)")
    if contract.eval_runtime_budget_seconds <= 0:
        issues.append("eval_runtime_budget_seconds must be > 0")
    if "rate" in " ".join(contract.objective_terms).lower() and contract.rate_denominator_bytes is None:
        issues.append("objective charges rate but rate_denominator_bytes is unset")
    if not contract.submission_boundary:
        issues.append("missing submission_boundary (what is the program?)")
    return {
        "schema": "frozen_evaluator_contract_verdict.v1",
        "contract": contract.name,
        "well_formed": not issues,
        "issues": issues,
    }


def verify_evaluator_satisfies_contract(evaluator: object) -> dict[str, Any]:
    """Verify a candidate evaluator object exposes the FrozenEvaluator surface (the
    'engineered against on the other end' check). Uses the runtime-checkable Protocol."""
    ok = isinstance(evaluator, FrozenEvaluator)
    missing = [
        m for m in ("score", "score_terms", "within_eval_budget")
        if not callable(getattr(evaluator, m, None))
    ]
    return {
        "schema": "evaluator_satisfies_contract_verdict.v1",
        "satisfies": ok and not missing,
        "missing_methods": missing,
    }


@dataclass(frozen=True)
class FrontierResult:
    """The frontier the harness achieved for a (contract, evaluator) pair — the typed
    output primitive. The authority of these numbers is INHERITED from the underlying
    evaluator (a synthetic evaluator => research-only; the comma evaluator => whatever
    axis evaluate.py ran on). The harness NEVER manufactures authority."""

    contract_name: str
    best_objective: float
    best_terms: dict[str, float]
    best_candidate_id: str
    n_candidates_evaluated: int
    n_within_budget: int
    minimize: bool
    trajectory: tuple[dict[str, Any], ...] = ()
    authority_note: str = "authority inherited from the underlying evaluator; harness adds none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "frozen_evaluator_frontier_result.v1",
            "contract_name": self.contract_name,
            "best_objective": self.best_objective,
            "best_terms": dict(self.best_terms),
            "best_candidate_id": self.best_candidate_id,
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "n_within_budget": self.n_within_budget,
            "minimize": self.minimize,
            "authority_note": self.authority_note,
        }


def synthesize_frontier(
    contract: FrozenEvaluatorContract,
    evaluator: FrozenEvaluator,
    candidates: Any,
    *,
    candidate_id=repr,
    max_candidates: int | None = None,
) -> FrontierResult:
    """THE general harness op: given a frozen (evaluator, input) pair + an iterable of
    candidate programs, produce the frontier (best objective within the eval budget) +
    its per-term decomposition + trajectory. Evaluator-agnostic: the SAME op drives the
    comma contest, a synthetic goal, or a literature-extracted task off the shelf.

    Fail-closed: refuses a malformed contract or an evaluator missing the surface (so a
    frontier is never reported against an unverified objective).
    """
    cv = verify_frozen_evaluator_contract(contract)
    if not cv["well_formed"]:
        raise ValueError(f"contract {contract.name} not well-formed: {cv['issues']}")
    ev = verify_evaluator_satisfies_contract(evaluator)
    if not ev["satisfies"]:
        raise ValueError(f"evaluator missing FrozenEvaluator surface: {ev['missing_methods']}")

    best_obj: float | None = None
    best_terms: dict[str, float] = {}
    best_id = ""
    n_eval = 0
    n_budget = 0
    traj: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        if max_candidates is not None and i >= max_candidates:
            break
        if not evaluator.within_eval_budget(candidate=cand):
            continue
        n_budget += 1
        terms = dict(evaluator.score_terms(source_ref=contract.source_ref, candidate=cand))
        obj = float(evaluator.score(source_ref=contract.source_ref, candidate=cand))
        n_eval += 1
        cid = candidate_id(cand)
        traj.append({"candidate_id": cid, "objective": obj, "terms": terms})
        better = (best_obj is None) or (obj < best_obj if contract.minimize else obj > best_obj)
        if better:
            best_obj, best_terms, best_id = obj, terms, cid
    if best_obj is None:
        raise ValueError(
            f"{contract.name}: no candidate fit the eval budget — no frontier to report."
        )
    return FrontierResult(
        contract_name=contract.name,
        best_objective=best_obj,
        best_terms=best_terms,
        best_candidate_id=best_id,
        n_candidates_evaluated=n_eval,
        n_within_budget=n_budget,
        minimize=contract.minimize,
        trajectory=tuple(traj),
    )


def comma_video_compression_contract() -> FrozenEvaluatorContract:
    """The CANONICAL first instance: the comma video-compression challenge. The
    objective + frame-incidence + rate denominator mirror `tac.contest_eval_contract`
    (the comma-specific semantics module); this wraps them in the general contract so
    the compiler treats comma as ONE pluggable (evaluator, input) pair among many."""
    # Import lazily to avoid a hard dependency when the contract is used for other tasks.
    try:
        from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES

        denom = int(CONTEST_ORIGINAL_BYTES)
    except Exception:  # pragma: no cover - defensive; the literal is the pinned value
        denom = 37_545_489
    return FrozenEvaluatorContract(
        name="comma_video_compression_challenge",
        source_ref="upstream/videos/0.mkv",
        objective_formula="100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/N",
        objective_terms=("d_seg", "d_pose", "rate"),
        minimize=True,
        eval_runtime_budget_seconds=1800.0,  # 30-minute official limit
        submission_boundary="archive.zip + inflate.sh",
        rate_denominator_bytes=denom,
        overfit_authorized=True,
        notes=(
            "SegNet reads frame1 argmax (semantic); PoseNet reads both frames via "
            "RGB->YUV6 (temporal); rate = archive.zip bytes / N. The frozen evaluator is "
            "upstream/evaluate.py; never run here (this is the contract, not the scorer)."
        ),
        extra={"semantics_module": "tac.contest_eval_contract"},
    )
