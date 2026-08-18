"""Controls for the corrections-index freshness banner — the instrument that watches the instrument.

Both directions are executed: a STALE fixture must warn (red on the disease) and a FRESH
fixture must not (green on the cure).  A freshness check that warns on everything is as
useless as one that warns on nothing, so the green leg is not optional.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tac.corrections_index_freshness import (
    freshness_banner,
    measure_index_freshness,
)

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)


def _write_index(path: Path, sources: list[str], quantity: bool = False) -> None:
    lines = []
    for source in sources:
        row = {"source": source, "phrase": "corrected", "numeric_literals": [1, 2]}
        if quantity:
            row["quantity"] = "archive_bytes"
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")


def _corpus(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text("# memo\n")


def test_red_a_stale_index_warns_and_names_the_horizon(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/a_20260805.md", ".omx/research/b_20260804.md"])
    research = tmp_path / "research"
    _corpus(research, ["a_20260805.md", "live_20260818.md", "later_20260817.md"])

    freshness = measure_index_freshness(index, research_root=research, now=NOW)

    assert freshness.stale is True
    assert freshness.max_source_date == "20260805"
    assert freshness.corpus_max_date == "20260818"
    assert freshness.corpus_files_beyond_horizon == 2
    assert freshness.blind_days == 13.0
    joined = " ".join(freshness.warnings)
    assert "REBUILD" in joined
    assert "cannot see them at all" in joined

    banner = freshness_banner(freshness, rebuild_command="rebuild-me")
    assert "STALE" in banner
    assert "BLIND 13 day(s)" in banner
    assert "rebuild-me" in banner


def test_green_a_fresh_index_does_not_warn_about_staleness(tmp_path: Path) -> None:
    """The green leg. Without it, a check that always warns would look like it works."""
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/live_20260818.md"], quantity=True)
    research = tmp_path / "research"
    _corpus(research, ["live_20260818.md"])

    freshness = measure_index_freshness(index, research_root=research, now=NOW)

    assert freshness.stale is False
    assert freshness.warnings == ()
    assert freshness.corpus_files_beyond_horizon == 0
    assert freshness.identifies_quantities is True

    banner = freshness_banner(freshness)
    assert "FRESH" in banner
    assert "WARN" not in banner


def test_the_banner_prints_even_when_healthy(tmp_path: Path) -> None:
    """Silence made loud: a clean index still announces its horizon and denominators."""
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/live_20260818.md"], quantity=True)
    research = tmp_path / "research"
    _corpus(research, ["live_20260818.md"])

    banner = freshness_banner(measure_index_freshness(index, research_root=research, now=NOW))

    assert "horizon 20260818" in banner
    assert "1 rows over 1 sources" in banner
    assert "corpus reaches 20260818" in banner


def test_a_fresh_but_inert_index_is_not_reported_as_healthy(tmp_path: Path) -> None:
    """The vacuity trap one layer up: rebuilt, current, and still emitting nothing."""
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/live_20260818.md"], quantity=False)
    research = tmp_path / "research"
    _corpus(research, ["live_20260818.md"])

    freshness = measure_index_freshness(index, research_root=research, now=NOW)

    assert freshness.stale is False  # freshness alone says everything is fine ...
    assert freshness.identifies_quantities is False
    assert any("FAIL-CLOSED" in warning for warning in freshness.warnings)  # ... and it is not


def test_an_absent_index_is_loud_not_silent(tmp_path: Path) -> None:
    freshness = measure_index_freshness(tmp_path / "nope.jsonl", research_root=tmp_path, now=NOW)

    assert freshness.exists is False
    assert freshness.stale is True
    assert any("ABSENT" in warning for warning in freshness.warnings)
    assert "ABSENT" in freshness_banner(freshness)


def test_denominators_are_reported_including_undated_sources(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/dated_20260818.md", ".omx/research/undated_memo.md"])
    research = tmp_path / "research"
    _corpus(research, ["dated_20260818.md"])

    freshness = measure_index_freshness(index, research_root=research, now=NOW)

    assert freshness.sources == 2
    assert freshness.sources_without_date_token == 1
    assert "(1 undated)" in freshness_banner(freshness)


def test_horizon_is_derived_from_rows_not_from_file_mtime(tmp_path: Path) -> None:
    """A stamp can rot invisibly; a derived horizon cannot disagree with the rows it came from."""
    index = tmp_path / "index.jsonl"
    _write_index(index, [".omx/research/old_20260701.md"])
    research = tmp_path / "research"
    _corpus(research, ["old_20260701.md"])

    # Touching the file (a "rebuild" that indexed nothing new) must not make it look fresh.
    index.touch()

    freshness = measure_index_freshness(index, research_root=research, now=NOW)
    assert freshness.max_source_date == "20260701"
    assert freshness.stale is True


def test_live_index_is_measurable_and_carries_its_denominators() -> None:
    """The real store must answer; whatever it says, it says with its denominators attached."""
    freshness = measure_index_freshness()

    assert freshness.exists is True
    assert freshness.rows > 0
    assert freshness.sources > 0
    assert freshness.max_source_date is not None
    assert "horizon" in freshness_banner(freshness)
