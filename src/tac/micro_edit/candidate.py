# SPDX-License-Identifier: MIT
"""Typed edit candidates and the generator plugin protocol (charter section 1).

ONE proposal interface over every measured edit family. A generator plugin emits
:class:`EditCandidate` objects; it never scores them, never compiles them, and never
decides anything. Scoring happens in the evaluate stage against the real
compile -> decode -> advisory chain; ordering happens in the ranker. Keeping the three
apart is what stops a generator's own optimism from becoming a verdict.

A candidate is a SPECIFICATION, not a result. It carries:

* the family and the support it touches;
* a persisted spec payload (ALWAYS KEEP THE PAYLOAD -- the spec is bytes we
  materialised, so it is written to the SSD tier and hashed before it is queued);
* the base it was generated against, because an edit spec is base-relative;
* an ESTIMATE, explicitly flagged as an estimate and never usable for admission.

STORES CONSULTED
----------------
* ``.omx/research/ddm_eu4_fresh_eyes_fractal_composition_20260813.md`` -- the edit
  object must bind semantic tokens, lattice, rendered frame, and compensation
  together (lens 2: the missing fingerprint was the qs4 defect).
* ``.omx/research/ddm_qs4_collateral_suppression_20260813.md`` -- compensation is
  never a reusable asset across edit objects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tac.micro_edit.score_model import ScoreState

__all__ = [
    "EditCandidate",
    "GeneratorPlugin",
    "candidate_fingerprint",
    "persist_candidate_spec",
]


@dataclass(frozen=True)
class EditCandidate:
    """One proposed edit object, fully specified and base-bound.

    ``spec`` is the family-specific payload the compiler consumes. Its exact shape is
    the generator's business; the engine only requires that it be JSON-serialisable and
    that it fully determine the edit (so the fingerprint is meaningful).
    """

    candidate_id: str
    family: str
    support_desc: str
    support_size: int
    spec: dict[str, Any]

    # the base this spec was generated against -- specs are NOT base-portable
    base_label: str
    base_archive_sha256: str | None
    coder_regime: str

    # estimates: ordering only, never admission
    est_net_seg_flips: str = "0"
    est_bytes_delta: int = 0
    est_rationale: str = ""

    # custody, filled by persist_candidate_spec
    spec_path: str | None = None
    spec_sha256: str | None = None

    generator: str = ""
    notes: str = ""
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return candidate_fingerprint(self)

    def est_flips_per_byte(self) -> Decimal | None:
        """Estimated flips bought per byte spent, or None when it spends nothing.

        Compared against ``BREAKEVEN_FLIPS_PER_BYTE`` this is the cheapest possible
        pre-filter: a candidate that cannot clear break-even even on its own optimistic
        estimate can be dropped without paying for a compile.
        """
        if self.est_bytes_delta <= 0:
            return None
        return -Decimal(self.est_net_seg_flips) / Decimal(self.est_bytes_delta)

    def bound_to(self, base: ScoreState) -> bool:
        """True when this candidate was generated against ``base``."""
        if self.base_archive_sha256 and base.archive_sha256:
            return self.base_archive_sha256 == base.archive_sha256
        return self.base_label == base.label


def candidate_fingerprint(candidate: EditCandidate) -> str:
    """Stable hash over the identity-bearing fields.

    Deliberately EXCLUDES estimates, notes, and custody paths: two candidates that
    specify the same edit against the same base are the same object even if one
    carries a more optimistic estimate. This is what makes de-duplication safe across
    generator plugins that overlap.
    """
    identity = {
        "family": candidate.family,
        "spec": candidate.spec,
        "support_desc": candidate.support_desc,
        "base_archive_sha256": candidate.base_archive_sha256,
        "base_label": candidate.base_label,
        "coder_regime": candidate.coder_regime,
    }
    blob = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def persist_candidate_spec(candidate: EditCandidate, root: Path) -> EditCandidate:
    """Write the spec to the SSD tier and return the candidate with custody filled.

    ALWAYS KEEP THE PAYLOAD applies at generation time, not only at measurement time:
    a spec we generated is bytes we materialised, and a queue of specs we cannot
    re-materialise is exactly the measure-and-discard defect one stage earlier.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{candidate.family}__{candidate.candidate_id}.json"
    body = dict(asdict(candidate))
    body.pop("spec_path", None)
    body.pop("spec_sha256", None)
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(blob)
    return EditCandidate(
        **{
            **asdict(candidate),
            "spec_path": str(path),
            "spec_sha256": hashlib.sha256(blob).hexdigest(),
        }
    )


@runtime_checkable
class GeneratorPlugin(Protocol):
    """A family's proposal source.

    Implementations MUST be pure proposal: no compiling, no scoring, no admission.
    ``generate`` may return an empty list -- an honest empty family is a valid output
    and gets its own row in the per-family asymptote table.
    """

    family: str

    def generate(self, base: ScoreState, budget: int) -> list[EditCandidate]:
        """Propose at most ``budget`` candidates against ``base``."""
        ...

    def describe(self) -> str:
        """One line: what this family edits and what receipt licenses it."""
        ...
