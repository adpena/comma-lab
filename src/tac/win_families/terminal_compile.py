# SPDX-License-Identifier: MIT
"""F2 -- TERMINAL JOINT COMPILE: the downstream refit an upstream edit forces.

THE PATTERN
-----------
An edit never lands alone.  ``ddm_jg1`` measured the shape of it: editing seg tokens
damages the pose carrier, and a carrier RE-SOLVE against the edited frames absorbs that
damage at ~0 bytes.  The actuators compose -- but only if the downstream stages are
actually re-run against the NEW upstream state.  The compile order is a topological sort::

    seg edits -> carrier re-solve (vs the EDITED frames) -> compensation -> rate re-encode -> container search

THE BUG THIS EXTINCTS
---------------------
The failure is not getting the order wrong; it is running the order once and then changing
an upstream input without re-running what depended on it.  A carrier solved against the
PRE-edit frames is a carrier for a body that no longer exists, and nothing in the numbers
says so -- the stale result is a perfectly well-formed float.

This module makes that structural.  Each :class:`CompileStage` declares what it CONSUMES
and what it PRODUCES; the pipeline hashes every artifact; and
:meth:`CompilePipeline.stale_stages` reports any stage whose inputs changed after it last
ran.  :meth:`CompilePipeline.assert_fresh` refuses to certify a compile with stale stages,
so freshness is checked AT CONSUMPTION rather than trusted from ordering.

TWO MEASURED CORRECTIONS THIS MODULE CARRIES
--------------------------------------------
**The rate leg must be MEASURED, not modelled.**  ``ddm_jg1`` modelled the token-edit rate
cost at **+4.718 bits per changed token**; ``ddm_jg2`` then measured the real archive delta
and got **+30 B at 4.1379 bits/changed token**, of which 0.135 bits/token is coder tax over
ideal.  jg1's modelled headline (-0.0104) and the honest measured gap (0.006526) are not
the same number.  So :class:`RateStage` requires a MEASURED archive delta and refuses a
modelled one on the certifying path.

**Compensation must consume the shipping GT lineage.**  ``ddm_qs5``'s Schur compensation
reads a ``GT_POSE`` table built from the PyAV decode.  The shipped object is scored on the
CUDA axis against the DALI table, and those are different objectives separated by a fixed
additive ``d_pose`` floor of 1.4061e-04.  :class:`CompensationStage` therefore declares its
GT lineage and is checked against the target axis through
:mod:`tac.local_contest_instruments`, so a PyAV-fed compensation cannot silently certify a
CUDA-axis compile.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tac import local_contest_instruments as lci

__all__ = [
    "CompilePipeline",
    "CompileStage",
    "CompileStaleness",
    "TerminalCompileError",
    "artifact_digest",
    "canonical_compile_order",
]

#: The topological order ``ddm_jg1``/``ddm_jg2`` established.  A pipeline may omit stages
#: but may not reorder them: each entry consumes the previous one's output.
canonical_compile_order: tuple[str, ...] = (
    "seg_edit",
    "carrier_resolve",
    "compensation",
    "rate_reencode",
    "container_search",
)


class TerminalCompileError(RuntimeError):
    """A compile-pipeline precondition failed.  Always fail closed."""


def artifact_digest(value: Any) -> str:
    """A stable content digest for an artifact.

    ``bytes`` hash directly.  Anything else is hashed through its ``repr`` after a
    ``json`` attempt, so numpy arrays and dataclasses both get a stable key without this
    module needing to know their types.
    """
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(bytes(value)).hexdigest()
    try:
        payload = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):  # pragma: no cover - default=repr covers most cases
        payload = repr(value)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CompileStage:
    """One stage of the terminal compile, with an explicit input/output contract.

    ``consumes`` and ``produces`` are artifact NAMES.  The pipeline uses them to detect
    staleness; it never inspects the artifacts themselves.
    """

    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    #: For a stage that reads ground truth, the lineage it reads.  ``None`` = reads none.
    gt_lineage: str | None = None
    #: True when the stage's own output number is MEASURED through the real decode.
    measured: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise TerminalCompileError("a compile stage needs a name")
        if not self.produces:
            raise TerminalCompileError(
                f"stage {self.name!r} produces nothing; a stage that produces nothing "
                "cannot be depended on and cannot be checked for staleness"
            )
        overlap = set(self.consumes) & set(self.produces)
        if overlap:
            raise TerminalCompileError(
                f"stage {self.name!r} both consumes and produces {sorted(overlap)}; "
                "that makes its own freshness undecidable"
            )


@dataclass(frozen=True)
class CompileStaleness:
    """Which stages are stale, and why."""

    stage: str
    stale_inputs: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"stage": self.stage, "stale_inputs": list(self.stale_inputs)}


class CompilePipeline:
    """A staleness-tracking terminal compile.

    Args:
        stages: in execution order.
        target_axis: the axis the finished object will be scored on.  Every stage that
            declares a ``gt_lineage`` is checked against it.
    """

    def __init__(
        self, stages: Sequence[CompileStage], *, target_axis: str = lci.AXIS_CONTEST_CUDA
    ) -> None:
        if not stages:
            raise TerminalCompileError("a compile pipeline needs at least one stage")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise TerminalCompileError(f"duplicate stage names: {names}")
        self._assert_canonical_order(names)
        lci.required_lineage_for_axis(target_axis)  # refuses an unknown axis
        self.stages = tuple(stages)
        self.target_axis = target_axis
        self.artifacts: dict[str, Any] = {}
        self._digests: dict[str, str] = {}
        #: stage name -> the input digests it last ran against
        self._ran_with: dict[str, dict[str, str]] = {}

    @staticmethod
    def _assert_canonical_order(names: Sequence[str]) -> None:
        """Stages may be omitted, never reordered."""
        known = [name for name in names if name in canonical_compile_order]
        positions = [canonical_compile_order.index(name) for name in known]
        if positions != sorted(positions):
            raise TerminalCompileError(
                f"stages are out of canonical compile order: {known}. "
                f"the order is {list(canonical_compile_order)} and each entry consumes "
                "the previous one's output"
            )

    # -- artifacts -----------------------------------------------------------

    def set_artifact(self, name: str, value: Any) -> None:
        """Set or REPLACE an artifact.  Replacing one makes its consumers stale."""
        self.artifacts[name] = value
        self._digests[name] = artifact_digest(value)

    def digest_of(self, name: str) -> str:
        try:
            return self._digests[name]
        except KeyError as error:
            raise TerminalCompileError(f"no artifact named {name!r}") from error

    # -- lineage -------------------------------------------------------------

    def assert_stage_lineages(self) -> None:
        """Refuse any stage reading a GT lineage the target axis is not scored against."""
        required = lci.required_lineage_for_axis(self.target_axis)
        for stage in self.stages:
            if stage.gt_lineage is None:
                continue
            if stage.gt_lineage != required:
                raise TerminalCompileError(
                    f"stage {stage.name!r} reads {stage.gt_lineage} ground truth but the "
                    f"compile targets {self.target_axis!r}, which is scored against "
                    f"{required}. ddm_qs5's Schur compensation had exactly this defect: "
                    "its GT_POSE table is the PyAV decode, and the shipped object is "
                    "scored on the CUDA axis against DALI. The two are separated by a "
                    f"fixed additive d_pose floor of "
                    f"{lci.ADVISORY_POSE_ADDITIVE_FLOOR:.4e}."
                )

    # -- running -------------------------------------------------------------

    def run_stage(self, stage: CompileStage) -> dict[str, Any]:
        """Run one stage, recording the input digests it ran against."""
        missing = [name for name in stage.consumes if name not in self.artifacts]
        if missing:
            raise TerminalCompileError(
                f"stage {stage.name!r} consumes {missing} which no earlier stage produced"
            )
        inputs = {name: self.artifacts[name] for name in stage.consumes}
        produced = stage.run(inputs)
        if not isinstance(produced, dict):
            raise TerminalCompileError(
                f"stage {stage.name!r} returned {type(produced).__name__}, expected a "
                "dict of produced artifacts"
            )
        missing_outputs = [name for name in stage.produces if name not in produced]
        if missing_outputs:
            raise TerminalCompileError(
                f"stage {stage.name!r} declared it produces {list(stage.produces)} but "
                f"did not return {missing_outputs}"
            )
        for name, value in produced.items():
            self.set_artifact(name, value)
        self._ran_with[stage.name] = {
            name: self._digests[name] for name in stage.consumes
        }
        return produced

    def run(self) -> dict[str, Any]:
        """Run every stage in order, after checking lineages."""
        self.assert_stage_lineages()
        for stage in self.stages:
            self.run_stage(stage)
        return dict(self.artifacts)

    # -- staleness -----------------------------------------------------------

    def stale_stages(self) -> tuple[CompileStaleness, ...]:
        """Stages whose consumed artifacts changed after the stage last ran.

        This is freshness checked AT CONSUMPTION.  A carrier solved against the pre-edit
        frames is a carrier for a body that no longer exists, and the stale number looks
        exactly like a fresh one.
        """
        stale: list[CompileStaleness] = []
        for stage in self.stages:
            ran_with = self._ran_with.get(stage.name)
            if ran_with is None:
                continue  # never ran; not stale, just absent
            changed = tuple(
                name
                for name, digest in ran_with.items()
                if self._digests.get(name) != digest
            )
            if changed:
                stale.append(CompileStaleness(stage=stage.name, stale_inputs=changed))
        return tuple(stale)

    def never_ran(self) -> tuple[str, ...]:
        """Stages that have not run at all."""
        return tuple(
            stage.name for stage in self.stages if stage.name not in self._ran_with
        )

    def assert_fresh(self) -> None:
        """Refuse to certify a compile with stale or unrun stages."""
        absent = self.never_ran()
        if absent:
            raise TerminalCompileError(
                f"cannot certify: stages {list(absent)} never ran. An unrun stage is not "
                "a passing stage -- a skipped instrument reading as green is the "
                "vacuity-equals-pass failure."
            )
        stale = self.stale_stages()
        if stale:
            detail = ", ".join(
                f"{row.stage} (inputs {list(row.stale_inputs)})" for row in stale
            )
            raise TerminalCompileError(
                f"cannot certify: stale stages {detail}. An upstream artifact changed "
                "after these stages ran, so their outputs describe a body that no longer "
                "exists. Re-run them in canonical order."
            )

    def unmeasured_stages(self) -> tuple[str, ...]:
        """Stages whose own number is modelled rather than measured through the decode."""
        return tuple(stage.name for stage in self.stages if not stage.measured)

    def certify(self) -> dict[str, Any]:
        """The compile receipt.  Refuses unless every stage is fresh and ran."""
        self.assert_stage_lineages()
        self.assert_fresh()
        modelled = self.unmeasured_stages()
        return {
            "pipeline": [stage.name for stage in self.stages],
            "target_axis": self.target_axis,
            "gt_lineage": lci.required_lineage_for_axis(self.target_axis),
            "artifact_digests": dict(self._digests),
            "modelled_stages": list(modelled),
            "all_stages_measured": not modelled,
            "score_claim": False,
            "promotable": False,
            "note": (
                "a modelled stage is not a measured one: ddm_jg1 modelled +4.718 "
                "bits/changed token and ddm_jg2 measured 4.1379 with a +30 B archive "
                "delta. Only upstream/evaluate.py on contest hardware is a score."
            ),
        }


# ---------------------------------------------------------------------------
# Stage builders for the two corrections this module exists to carry.
# ---------------------------------------------------------------------------


def compensation_stage(
    *,
    name: str = "compensation",
    gt_lineage: str,
    run: Callable[[dict[str, Any]], dict[str, Any]],
    consumes: tuple[str, ...] = ("carrier_codes",),
    produces: tuple[str, ...] = ("compensated_codes",),
) -> CompileStage:
    """A compensation stage that must DECLARE the GT lineage it reads.

    ``gt_lineage`` is required rather than defaulted: ``ddm_qs5`` shipped a compensation
    fed by the PyAV ``GT_POSE`` table without anywhere to say so, and a default would
    reproduce that silence.
    """
    if gt_lineage is None:
        raise TerminalCompileError(
            "a compensation stage must declare the GT lineage it reads"
        )
    return CompileStage(
        name=name,
        consumes=consumes,
        produces=produces,
        run=run,
        gt_lineage=gt_lineage,
        measured=True,
    )


@dataclass(frozen=True)
class RateLeg:
    """A rate result, refusing to certify a MODELLED archive delta.

    ``ddm_jg1`` modelled 4.718 bits/changed token; ``ddm_jg2`` measured 4.1379 with a
    +30 B archive delta.  The difference is not rounding -- it changed the headline from
    -0.0104 to an honest gap of 0.006526.
    """

    changed_tokens: int
    archive_delta_bytes: int | None = None
    modelled_bits_per_token: float | None = None
    measured: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "measured", self.archive_delta_bytes is not None)
        if self.changed_tokens < 0:
            raise TerminalCompileError("changed_tokens must be >= 0")
        if self.archive_delta_bytes is None and self.modelled_bits_per_token is None:
            raise TerminalCompileError(
                "a rate leg needs either a MEASURED archive delta or a modelled rate"
            )

    @property
    def realized_bits_per_token(self) -> float | None:
        """Measured bits per changed token, or ``None`` when only a model exists."""
        if self.archive_delta_bytes is None or self.changed_tokens == 0:
            return None
        return 8.0 * self.archive_delta_bytes / self.changed_tokens

    def assert_measured(self) -> None:
        if not self.measured:
            raise TerminalCompileError(
                "this rate leg is MODELLED, not measured. ddm_jg1's modelled "
                "+4.718 bits/token became a measured 4.1379 with a +30 B archive delta "
                "once ddm_jg2 built the real thing. Measure the archive."
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "changed_tokens": self.changed_tokens,
            "archive_delta_bytes": self.archive_delta_bytes,
            "modelled_bits_per_token": self.modelled_bits_per_token,
            "realized_bits_per_token": self.realized_bits_per_token,
            "measured": self.measured,
        }
