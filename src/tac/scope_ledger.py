"""Canonical scope ledger — make VACUITY a distinct symbol from PASS.

Source: operator-derived law, 2026-08-01 (memory
``vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801``,
artifact ``.omx/research/ddm_rt1_counted_ledger_test_adjudication_and_ci_blind_silence_20260801.md``):

    An instrument that evaluated an EMPTY SCOPE emits the same symbol as one
    that evaluated a full scope cleanly. **Vacuity is indistinguishable from
    PASS.**

This is the GENUS of the 2026-08-01 silent-instrument family, not a sibling of
the usual confound signature. The usual signature (DEFAULT-HARMFUL x SILENT x
MEASUREMENT-CORRUPTING) describes an instrument giving a WRONG answer. This one
gives NO answer while looking like a right one, so it defeats every check that
reads the VERDICT. It is caught only by asking **"how many things did this
actually examine?"** — i.e. by reading the DENOMINATOR.

THREE MEASURED INSTANCES (2026-08-01, all live at derivation time):

===========================  ==================================  ==============
surface                      the empty scope                     symbol emitted
===========================  ==================================  ==============
pytest / CI                  ``mlx`` has no Linux wheel =>       GREEN
                             57 MLX-gated modules SKIPPED
preflight CLI (#842)         ``--no-codebase`` => 0 gates run    ``PREFLIGHT
                                                                 PASSED``
codex-findings scan          ``mtime < 3d`` => 0 of 1260 files   CLEAN
===========================  ==================================  ==============

The preflight instance is MEASURED by this landing's probe, and it is the pure
form: ``python -m tac.preflight --no-codebase`` completed in **0.52 s** having
executed **0 of 27 declared** developer gates (0 of 448 for ``--scope all``) and
printed ``PREFLIGHT PASSED``.

THE CURE this module implements
-------------------------------
A verdict is not a symbol; it is a symbol PLUS a denominator. ``ScopeLedger``
carries three numbers so the shape of an empty scope is legible:

``population``
    the universe BEFORE any scope filter (optional). This is the number that
    makes a filter's damage visible: the codex-findings scan is
    ``population=1260, declared=0, examined=0`` — "you filtered out 1260 things",
    not the far more innocent-looking "nothing to do".
``declared``
    what the instrument intended to examine after its scope filter.
``examined``
    what it actually examined. THE NUMERATOR.

and derives a verdict on a three-value ladder — ``COMPLETE`` / ``PARTIAL`` /
``VACUOUS`` — where **``VACUOUS`` is never rendered as a pass word**.

SEVERITY IS THE CALLER'S DECISION, DELIBERATELY
-----------------------------------------------
``VACUOUS`` is a *symbol*, not automatically an error. Some scopes are
legitimately empty (a gate globbing ``experiments/results/**/launch.sh`` on a
fresh clone examines nothing and nothing is wrong). Hard-failing those would be
a false alarm, and an apparatus that cries wolf gets overridden — which
reproduces the silence it was built to end. So the ledger ALWAYS tells the
truth about the denominator, and the caller chooses severity via
:meth:`ScopeLedger.require_non_vacuous`.

Corollary carried from the source memo: **an override that silences a loud
instrument reproduces the original silence with extra steps.** Hence
:meth:`acknowledged` — an acknowledged vacuity stays ``VACUOUS`` in the verdict
and stays visible in :meth:`render`; acknowledgement suppresses only the raise,
never the report.

MEASURED/DERIVED/INFERRED/ASSUMED labels for the claims above: the three
instances are MEASURED (rt1's artifact + this landing's probe); the ladder and
the severity split are DERIVED from the law; the false-alarm argument is
INFERRED from the rt1 corollary.

Sisters: ``tac.confound_gates`` (Layer 2 STRICT gates for the DEFAULT-HARMFUL
confound family) · CLAUDE.md "Confound self-protection" · CLAUDE.md "Bugs must
be permanently fixed AND self-protected against" (this module is landing 1 of
2; landing 2 is ``check_verdict_surfaces_report_examined_count``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "COMPLETE",
    "PARTIAL",
    "VACUOUS",
    "ScopeLedger",
    "ScopeVacuityError",
]

# --- the three-value verdict ladder ---------------------------------------
# VACUOUS is deliberately NOT a member of any "pass" set anywhere in this
# module. The whole point is that it can never be spelled with a pass word.
VACUOUS = "VACUOUS"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"

#: Verdicts that mean "this instrument actually looked at something".
_NON_VACUOUS = frozenset({PARTIAL, COMPLETE})


class ScopeVacuityError(RuntimeError):
    """Raised when a caller required a non-empty scope and got an empty one."""


@dataclass(frozen=True)
class ScopeLedger:
    """How many things an instrument actually examined, and out of how many.

    Construct one wherever a verdict is emitted over an enumerated scope, then
    print :meth:`render` instead of a bare pass word.

    Args:
        surface: the instrument's name, as a human would refer to it.
        examined: THE NUMERATOR — items actually examined.
        declared: items the instrument intended to examine after its scope
            filter. Defaults to ``examined`` when the instrument has no
            separate notion of intent.
        population: the universe BEFORE any scope filter, when known. This is
            what exposes a filter that silently emptied the scope.
        skipped_scopes: named sub-scopes that were not entered at all (e.g.
            ``("codebase",)`` when ``--no-codebase`` was passed). These are the
            *reason* for a gap and belong in the report.
        acknowledged: the caller has explicitly, in code, accepted that this
            scope may be empty. Suppresses the raise in
            :meth:`require_non_vacuous` — and NOTHING else.
        note: free-text context appended to :meth:`render`.

    Examples:
        A genuine clean pass over a real scope::

            >>> ScopeLedger(surface="gate", examined=25, declared=27).verdict
            'PARTIAL'

        The bug this module exists to make impossible to miss::

            >>> led = ScopeLedger(surface="preflight", examined=0, declared=27)
            >>> led.verdict
            'VACUOUS'
            >>> "PASS" in led.render()
            False
    """

    surface: str
    examined: int
    declared: int | None = None
    population: int | None = None
    skipped_scopes: tuple[str, ...] = field(default=())
    acknowledged: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.examined < 0:
            raise ValueError(f"examined must be >= 0, got {self.examined}")
        if self.declared is not None and self.declared < 0:
            raise ValueError(f"declared must be >= 0, got {self.declared}")
        if self.population is not None and self.population < 0:
            raise ValueError(f"population must be >= 0, got {self.population}")

    # -- derived numbers ---------------------------------------------------

    @property
    def declared_count(self) -> int:
        """``declared``, defaulting to ``examined`` when intent is unstated."""
        return self.examined if self.declared is None else self.declared

    @property
    def unexamined(self) -> int:
        """Declared-but-not-examined. The size of the blind spot."""
        return max(0, self.declared_count - self.examined)

    @property
    def filtered_out(self) -> int:
        """Population-but-not-declared: what the scope FILTER removed."""
        if self.population is None:
            return 0
        return max(0, self.population - self.declared_count)

    # -- the verdict -------------------------------------------------------

    @property
    def verdict(self) -> str:
        """``VACUOUS`` / ``PARTIAL`` / ``COMPLETE``.

        ``examined == 0`` is ``VACUOUS`` unconditionally — including the
        ``declared == 0`` case. "There was nothing to look at" and "I looked at
        nothing" are the same fact from the reader's side, and the reader is
        entitled to know it rather than to be told PASS.
        """
        if self.examined == 0:
            return VACUOUS
        if self.unexamined or self.skipped_scopes or self.filtered_out:
            return PARTIAL
        return COMPLETE

    @property
    def is_vacuous(self) -> bool:
        return self.verdict == VACUOUS

    @property
    def examined_something(self) -> bool:
        return self.verdict in _NON_VACUOUS

    # -- reporting ---------------------------------------------------------

    def render(self) -> str:
        """One line carrying the verdict AND the denominator.

        Never contains a pass word when the verdict is ``VACUOUS`` — asserted by
        the module's tests, because that property is the entire cure.
        """
        parts = [f"{self.surface}: {self.verdict}"]
        parts.append(f"examined {self.examined} of {self.declared_count} declared")
        if self.population is not None and self.population != self.declared_count:
            parts.append(f"({self.population} in population before scope filter)")
        if self.skipped_scopes:
            parts.append("scopes SKIPPED: " + ", ".join(self.skipped_scopes))
        if self.is_vacuous:
            parts.append(
                "NOTHING WAS EXAMINED — this is not a pass, it is an empty scope"
            )
            if self.acknowledged:
                parts.append("(empty scope explicitly acknowledged by the caller)")
        if self.note:
            parts.append(self.note)
        return " — ".join(parts)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable row for JSON/JSONL observability surfaces."""
        return {
            "surface": self.surface,
            "verdict": self.verdict,
            "examined": self.examined,
            "declared": self.declared_count,
            "population": self.population,
            "unexamined": self.unexamined,
            "filtered_out": self.filtered_out,
            "skipped_scopes": list(self.skipped_scopes),
            "acknowledged": self.acknowledged,
            "note": self.note,
        }

    # -- severity, chosen by the caller ------------------------------------

    def require_non_vacuous(self, *, hint: str = "") -> "ScopeLedger":
        """Raise :class:`ScopeVacuityError` if nothing was examined.

        ``acknowledged=True`` suppresses the raise ONLY. The verdict stays
        ``VACUOUS`` and :meth:`render` still says so, because an override that
        silences a loud instrument reproduces the original silence with extra
        steps.

        Returns ``self`` so it can be chained at a call site.
        """
        if not self.is_vacuous or self.acknowledged:
            return self
        msg = self.render()
        if hint:
            msg = f"{msg}\n  {hint}"
        raise ScopeVacuityError(msg)
