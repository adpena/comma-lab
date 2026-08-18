"""Freshness of the au1 corrections index, measured at read time — silence made loud.

THE FAILURE THIS CURES.  The corrections index is the coarse net a charter is linted
against.  It was built once, by hand, on 2026-08-05, and nothing ever ran it again: there
is no hook, no preflight gate, no scheduler, no caller.  Thirteen days later it still
answered every query, still returned plausible hits, and still read as comprehensive —
while seeing **none** of the live vehicle.  *An instrument that is comprehensive over a
window you cannot see reads exactly like an instrument that is comprehensive.*

THE CURE IS A DERIVED BANNER, NOT A STAMP.  Nothing here trusts a recorded build date: a
stamp is one more constant that can rot, and the rot would again be invisible.  The horizon
is measured from the index's own rows at the moment of use, and compared against the corpus
that exists at the moment of use.  So the banner cannot silently disagree with reality.

IT REPORTS ITS DENOMINATORS.  Rows, sources, sources carrying no parsable date, corpus files
scanned, corpus files beyond the horizon.  A reader can tell "clean" from "did not look".

IT ALSO REPORTS WHETHER THE CONSUMER IS LIVE.  The stale-number lint leg has been fail-closed
since 2026-08-17 because the schema cannot say *which quantity* a number is; it returns
nothing until the index carries a ``quantity`` field.  A freshly rebuilt but still inert
index would otherwise read as healthy — the vacuity-passes disease one layer up.  The banner
therefore prints the consumer's own state beside the horizon.

AXIS.  Apparatus observability.  No score, no promotion, no claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "DEFAULT_INDEX_PATH",
    "IndexFreshness",
    "freshness_banner",
    "measure_index_freshness",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = REPO_ROOT / ".omx" / "research" / "ddm_au1_20260805" / "au1_corrections_index.jsonl"
DEFAULT_RESEARCH_ROOT = REPO_ROOT / ".omx" / "research"

#: Sources are repo-relative paths carrying a ``YYYYMMDD`` token, e.g.
#: ``.omx/research/ddm_na9_gestalt_negative_audit_20260818.md``.
_DATE_TOKEN = re.compile(r"(20\d{6})")

#: The field the stale-number lint leg requires before it will emit anything.
_QUANTITY_FIELD = "quantity"

DEFAULT_STALE_AFTER_HOURS = 48.0


def _parse_date_token(text: str) -> str | None:
    match = _DATE_TOKEN.search(text)
    return match.group(1) if match else None


def _token_to_datetime(token: str) -> datetime | None:
    try:
        return datetime.strptime(token, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


@dataclass(frozen=True)
class IndexFreshness:
    """What the index can and cannot see, measured now, with the denominators kept."""

    index_path: Path
    exists: bool
    rows: int = 0
    sources: int = 0
    sources_without_date_token: int = 0
    min_source_date: str | None = None
    max_source_date: str | None = None
    horizon_age_hours: float | None = None
    stale: bool = False
    identifies_quantities: bool = False
    corpus_files_scanned: int | None = None
    corpus_max_date: str | None = None
    corpus_files_beyond_horizon: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blind_days(self) -> float | None:
        """How far the live corpus runs past what the index can see."""
        if not (self.max_source_date and self.corpus_max_date):
            return None
        index_at = _token_to_datetime(self.max_source_date)
        corpus_at = _token_to_datetime(self.corpus_max_date)
        if index_at is None or corpus_at is None:
            return None
        return max(0.0, (corpus_at - index_at).total_seconds() / 86400.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "index_path": str(self.index_path),
            "exists": self.exists,
            "rows": self.rows,
            "sources": self.sources,
            "sources_without_date_token": self.sources_without_date_token,
            "min_source_date": self.min_source_date,
            "max_source_date": self.max_source_date,
            "horizon_age_hours": self.horizon_age_hours,
            "stale": self.stale,
            "identifies_quantities": self.identifies_quantities,
            "corpus_files_scanned": self.corpus_files_scanned,
            "corpus_max_date": self.corpus_max_date,
            "corpus_files_beyond_horizon": self.corpus_files_beyond_horizon,
            "blind_days": self.blind_days,
            "warnings": list(self.warnings),
        }


def _scan_corpus(research_root: Path, horizon: str | None) -> tuple[int, str | None, int]:
    """Filename-only scan of the corpus the indexer globs. Cheap: no file is opened."""
    scanned = 0
    beyond = 0
    max_date: str | None = None
    for path in research_root.rglob("*.md"):
        if ".git" in path.parts:
            continue
        scanned += 1
        token = _parse_date_token(str(path))
        if token is None:
            continue
        if max_date is None or token > max_date:
            max_date = token
        if horizon is not None and token > horizon:
            beyond += 1
    return scanned, max_date, beyond


def measure_index_freshness(
    index_path: Path | None = None,
    research_root: Path | None = None,
    now: datetime | None = None,
    stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
    scan_corpus: bool = True,
) -> IndexFreshness:
    """Measure what the corrections index can see, right now, from the index itself."""
    index_path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
    research_root = Path(research_root) if research_root else DEFAULT_RESEARCH_ROOT
    now = now or datetime.now(UTC)

    if not index_path.is_file():
        return IndexFreshness(
            index_path=index_path,
            exists=False,
            stale=True,
            warnings=(
                f"corrections index ABSENT at {index_path} — every lint leg reading it is "
                "silently answering from nothing",
            ),
        )

    rows = 0
    sources: set[str] = set()
    identifies_quantities = False
    with index_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rows == 0:
                identifies_quantities = _QUANTITY_FIELD in row
            rows += 1
            source = row.get("source")
            if isinstance(source, str):
                sources.add(source)

    dated = [token for token in (_parse_date_token(source) for source in sources) if token]
    undated = len(sources) - len(dated)
    min_source_date = min(dated) if dated else None
    max_source_date = max(dated) if dated else None

    horizon_age_hours: float | None = None
    horizon_at = _token_to_datetime(max_source_date) if max_source_date else None
    if horizon_at is not None:
        horizon_age_hours = max(0.0, (now - horizon_at).total_seconds() / 3600.0)

    corpus_scanned: int | None = None
    corpus_max_date: str | None = None
    corpus_beyond: int | None = None
    if scan_corpus and research_root.is_dir():
        corpus_scanned, corpus_max_date, corpus_beyond = _scan_corpus(research_root, max_source_date)

    warnings: list[str] = []
    stale = horizon_age_hours is None or horizon_age_hours > stale_after_hours
    if horizon_age_hours is None:
        warnings.append(
            f"corrections index carries no parsable source date across {len(sources)} sources — "
            "its horizon cannot be established, so it must not be read as comprehensive"
        )
    elif stale:
        warnings.append(
            f"corrections index horizon is {max_source_date} ({horizon_age_hours / 24.0:.1f} days old, "
            f"bar {stale_after_hours / 24.0:.1f} days) — REBUILD before trusting a negative from it"
        )
    if corpus_beyond:
        warnings.append(
            f"{corpus_beyond} corpus file(s) of {corpus_scanned} are dated past the index horizon "
            f"{max_source_date} (corpus reaches {corpus_max_date}) — the index cannot see them at all"
        )
    if not identifies_quantities:
        warnings.append(
            "the stale-number lint leg is FAIL-CLOSED: index rows carry no 'quantity' field, so that "
            "leg emits nothing regardless of how fresh the index is (a rebuild alone will not revive it)"
        )

    return IndexFreshness(
        index_path=index_path,
        exists=True,
        rows=rows,
        sources=len(sources),
        sources_without_date_token=undated,
        min_source_date=min_source_date,
        max_source_date=max_source_date,
        horizon_age_hours=horizon_age_hours,
        stale=stale,
        identifies_quantities=identifies_quantities,
        corpus_files_scanned=corpus_scanned,
        corpus_max_date=corpus_max_date,
        corpus_files_beyond_horizon=corpus_beyond,
        warnings=tuple(warnings),
    )


def freshness_banner(freshness: IndexFreshness, rebuild_command: str | None = None) -> str:
    """Render the banner. It prints UNCONDITIONALLY — a healthy index says so out loud too."""
    if not freshness.exists:
        lines = [f"[corrections-index] ABSENT {freshness.index_path}"]
        lines.extend(f"[corrections-index]   WARN {warning}" for warning in freshness.warnings)
        return "\n".join(lines)

    blind = freshness.blind_days
    head = (
        f"[corrections-index] horizon {freshness.max_source_date or 'UNKNOWN'} · "
        f"{freshness.rows:,} rows over {freshness.sources:,} sources"
    )
    if freshness.sources_without_date_token:
        head += f" ({freshness.sources_without_date_token} undated)"
    if freshness.corpus_files_scanned is not None:
        head += f" · corpus reaches {freshness.corpus_max_date or 'UNKNOWN'} across {freshness.corpus_files_scanned:,} files"
    if blind:
        head += f" · BLIND {blind:.0f} day(s)"
    head += f" · {'STALE' if freshness.stale else 'FRESH'}"

    lines = [head]
    lines.extend(f"[corrections-index]   WARN {warning}" for warning in freshness.warnings)
    if freshness.warnings and rebuild_command:
        lines.append(f"[corrections-index]   REBUILD: {rebuild_command}")
    return "\n".join(lines)
