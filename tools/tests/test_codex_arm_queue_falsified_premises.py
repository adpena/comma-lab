# SPDX-License-Identifier: MIT
"""Charter-time falsified-premise advisory (ddm_pu3).

WHY THIS EXISTS. The premise "the pose GN solve was still descending 13-23% per
iteration when it stopped" originates in ONE n=1 STALE_REHEARSAL receipt whose own
``authority_blocker`` field reads "Pinned frozen PoseNet was evaluated on exactly one
stale composed pair". It was falsified four times at larger n (1.2%, 1.2%, 0.1549%,
0.07%) and still propagated into seven charters across three weeks -- two of them
written after the third falsification.

The sister leg ``_lint_stale_numbers`` could not catch it: that leg consumes an
auto-scraped 11,840-row index whose ``refuted_value`` entries go down to "0" and "6",
so it must require >=4-digit literals as a precision guard. A falsified PERCENTAGE is
structurally invisible to it. Hence a separate curated store.

These tests assert BEHAVIOUR (does the leg fire on the dead premise, stay silent
otherwise, and never block a spawn when its store is broken?), never constants.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "codex_arm_queue_under_test", _REPO / "tools" / "codex_arm_queue.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="caq")
def _caq():
    return _load_module()


def _point_registry_at(module, tmp_path: Path, rows: list[dict]) -> None:
    path = tmp_path / "falsified_premise_registry.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    module.FALSIFIED_PREMISE_REGISTRY = path


_ROW = {
    "premise_id": "demo_premise",
    "topic": "demo descent rate",
    "claim_patterns": ["13-23%"],
    "origin": {"path": "a/origin.json", "n_pairs": 1, "authority_mode": "STALE_REHEARSAL",
               "score_claim": False},
    "falsifications": [{"path": "b/later.md", "measured": "0.15%", "scale": "n=50"}],
    "verdict_scope": "formulation",
}


def test_fires_on_the_dead_premise(caq, tmp_path):
    _point_registry_at(caq, tmp_path, [_ROW])
    out = caq._lint_falsified_premises("still descending 13-23% per iteration when it stops")
    assert len(out) == 1
    warning = out[0]
    # The advisory must carry the three things that make it actionable: that the
    # premise is dead, the origin's SCALE, and what was measured instead.
    assert "FALSIFIED" in warning
    assert "n=1" in warning
    assert "STALE_REHEARSAL" in warning
    assert "0.15%" in warning


def test_unicode_dashes_do_not_smuggle_the_premise_past_the_leg(caq, tmp_path):
    """A charter written with an en- or em-dash is the same claim."""
    _point_registry_at(caq, tmp_path, [_ROW])
    assert len(caq._lint_falsified_premises("descending 13–23% per iteration")) == 1
    assert len(caq._lint_falsified_premises("descending 13—23% per iteration")) == 1


def test_silent_on_a_charter_that_does_not_restate_it(caq, tmp_path):
    _point_registry_at(caq, tmp_path, [_ROW])
    assert caq._lint_falsified_premises("seg flips and rate, no dead premise here") == []


def test_missing_store_is_silence_not_a_block(caq, tmp_path):
    caq.FALSIFIED_PREMISE_REGISTRY = tmp_path / "absent.jsonl"
    assert caq._lint_falsified_premises("descending 13-23% per iteration") == []


def test_malformed_rows_are_skipped_without_raising(caq, tmp_path):
    path = tmp_path / "falsified_premise_registry.jsonl"
    path.write_text(
        "{not json\n"
        + json.dumps({"claim_patterns": "not-a-list"}) + "\n"
        + json.dumps({"premise_id": "no_patterns"}) + "\n"
        + json.dumps(_ROW, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    caq.FALSIFIED_PREMISE_REGISTRY = path
    out = caq._lint_falsified_premises("descending 13-23% per iteration")
    assert len(out) == 1


def test_advisory_is_capped_so_one_charter_cannot_flood_the_spawn_site(caq, tmp_path):
    rows = [dict(_ROW, premise_id=f"p{i}", claim_patterns=[f"dead-claim-{i}"]) for i in range(9)]
    _point_registry_at(caq, tmp_path, rows)
    text = " ".join(f"dead-claim-{i}" for i in range(9))
    assert len(caq._lint_falsified_premises(text)) == 5


def test_leg_is_wired_into_the_recall_advisories(caq, tmp_path):
    """A leg that exists but is never called is the orphan class this repo names P0."""
    _point_registry_at(caq, tmp_path, [_ROW])
    charter = tmp_path / "charter.md"
    charter.write_text("still descending 13-23% per iteration\n", encoding="utf-8")
    out = caq.lint_charter_recall_advisories(str(charter))
    assert any("FALSIFIED" in line for line in out)


def test_a_broken_leg_never_blocks_a_spawn(caq, tmp_path):
    """Advisory contract: legs report their own failure, they do not raise."""
    caq.FALSIFIED_PREMISE_REGISTRY = tmp_path  # a directory, not a file
    charter = tmp_path / "charter.md"
    charter.write_text("descending 13-23% per iteration\n", encoding="utf-8")
    assert isinstance(caq.lint_charter_recall_advisories(str(charter)), list)


def test_the_live_registry_parses_and_every_row_is_actionable(caq):
    """The shipped store must be readable and complete, not just present."""
    module = _load_module()
    path = module.FALSIFIED_PREMISE_REGISTRY
    assert path.is_file(), "curated registry is missing"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert row.get("claim_patterns"), f"{row.get('premise_id')} has no claim pattern to match"
        assert row.get("falsifications"), f"{row.get('premise_id')} asserts death with no receipt"
        assert row.get("verdict_scope"), f"{row.get('premise_id')} has no verdict_scope"
        assert row["origin"].get("path"), f"{row.get('premise_id')} has no origin receipt"


def test_a_non_dict_origin_does_not_crash_the_leg(caq, tmp_path):
    """Round-1 self-review: `origin` was dereferenced without a type guard."""
    _point_registry_at(caq, tmp_path, [dict(_ROW, origin=["not", "a", "dict"])])
    out = caq._lint_falsified_premises("descending 13-23% per iteration")
    assert len(out) == 1
    assert "FALSIFIED" in out[0]


def test_patterns_are_anchored_enough_not_to_fire_on_a_plain_range(caq):
    """A bare '13-23' would fire on 'pairs 13-23'. Warnings block here; noise is a cost."""
    module = _load_module()
    rows = [
        json.loads(line)
        for line in module.FALSIFIED_PREMISE_REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        for pattern in row["claim_patterns"]:
            assert not pattern.replace("-", "").replace("–", "").isdigit(), (
                f"{pattern!r} is a bare numeric range and will false-positive"
            )
    assert caq._lint_falsified_premises("solved pairs 13-23 of the shard") == []
