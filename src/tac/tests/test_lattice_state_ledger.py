# SPDX-License-Identifier: MIT
"""Dedicated tests for tac.lattice_state_ledger canonical helper.

Mirrors the test depth of sister Catalog #245 ``test_modal_call_id_ledger.py``
+ Catalog #313 ``test_probe_outcomes_ledger.py``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from tac.lattice_state_ledger import (
    CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
    CLASSIFICATION_PARADIGM_INTACT,
    CLASSIFICATION_TBD,
    EVENT_DEFERRED,
    EVENT_PROMOTED,
    EVENT_REGISTERED,
    HORIZON_ASYMPTOTIC_PURSUIT,
    HORIZON_FRONTIER_PURSUIT,
    HORIZON_PLATEAU_ADJACENT,
    HORIZON_WON,
    LATTICE_STATE_LEDGER_LOCK,
    LATTICE_STATE_LEDGER_PATH,
    NERV_FAMILY_ARCHITECTURAL_CLASSES,
    RULE_1,
    RULE_2,
    RULE_3,
    RULE_4,
    SCHEMA_VERSION,
    STATUS_DEFERRED_OPERATOR,
    STATUS_DISPATCHED_EVIDENCE,
    STATUS_LIFTED_DISPATCH_READY,
    STATUS_LIFTED_PENDING_COUNCIL,
    STATUS_NOT_YET_LIFTED,
    VALID_HORIZON_CLASSES,
    VALID_LATTICE_RULES,
    VALID_STATUSES,
    LatticeStateLedgerCorruptError,
    compute_coverage_report,
    latest_node_state,
    load_nodes,
    load_nodes_strict,
    query_by_architectural_class,
    query_by_rule,
    query_by_substrate,
    query_outside_nerv_family,
    query_uncovered_rules,
    register_lattice_node,
    update_lattice_node,
)


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> tuple[Path, Path]:
    p = tmp_path / "lattice_state.jsonl"
    l = tmp_path / "lattice_state.jsonl.lock"
    return p, l


# -----------------------------------------------------------------------
# Schema constants
# -----------------------------------------------------------------------

def test_schema_version_pinned() -> None:
    assert SCHEMA_VERSION == 1


def test_valid_rules_taxonomy() -> None:
    assert VALID_LATTICE_RULES == frozenset(
        {RULE_1, RULE_2, RULE_3, RULE_4, "rule_5_request_operator_review"}
    )


def test_valid_horizon_classes_taxonomy() -> None:
    assert "plateau_adjacent" in VALID_HORIZON_CLASSES
    assert "frontier_pursuit" in VALID_HORIZON_CLASSES
    assert "asymptotic_pursuit" in VALID_HORIZON_CLASSES
    assert "won" in VALID_HORIZON_CLASSES
    assert "n_a" in VALID_HORIZON_CLASSES


def test_valid_statuses_includes_canonical_set() -> None:
    for s in [
        STATUS_LIFTED_DISPATCH_READY,
        STATUS_LIFTED_PENDING_COUNCIL,
        STATUS_NOT_YET_LIFTED,
        STATUS_DISPATCHED_EVIDENCE,
        STATUS_DEFERRED_OPERATOR,
    ]:
        assert s in VALID_STATUSES


def test_nerv_family_architectural_classes_canonical() -> None:
    # Operator binding constraint anchor — must include the canonical NeRV-family entries
    assert "nerv_family" in NERV_FAMILY_ARCHITECTURAL_CLASSES
    assert "ego_motion_focused_renderer" in NERV_FAMILY_ARCHITECTURAL_CLASSES
    assert "sane_hnerv_family" in NERV_FAMILY_ARCHITECTURAL_CLASSES


# -----------------------------------------------------------------------
# register_lattice_node — happy path + validation
# -----------------------------------------------------------------------


def test_register_writes_one_row(tmp_ledger: tuple[Path, Path]) -> None:
    p, l = tmp_ledger
    record = register_lattice_node(
        lattice_node_id="nscs01",
        substrate="nscs01_nullspace_split_renderer",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="pr95_paradigm_nullspace_split",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
        path=p,
        lock_path=l,
    )
    assert record["lattice_node_id"] == "nscs01"
    assert record["lattice_rule"] == RULE_2
    assert record["event_type"] == EVENT_REGISTERED
    rows = load_nodes(p)
    assert len(rows) == 1


def test_register_persists_to_disk(tmp_ledger: tuple[Path, Path]) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="z6",
        substrate="z6_predictive_coding",
        lattice_rule=RULE_4,
        horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
        architectural_class="predictive_coding_hierarchical",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    text = p.read_text()
    parsed = json.loads(text.strip())
    assert parsed["lattice_node_id"] == "z6"
    assert parsed["architectural_class"] == "predictive_coding_hierarchical"


def test_register_rejects_empty_lattice_node_id(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="lattice_node_id"):
        register_lattice_node(
            lattice_node_id="",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_register_rejects_empty_substrate(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="substrate"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_register_rejects_invalid_rule(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="lattice_rule"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="x",
            lattice_rule="rule_99_fake",
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_register_rejects_invalid_horizon(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="horizon_class"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class="orbital",
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_register_rejects_invalid_status(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="status"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status="cooking",
            path=p,
            lock_path=l,
        )


def test_register_rejects_invalid_classification(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="paradigm_vs_implementation_classification"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification="vibes",
            path=p,
            lock_path=l,
        )


def test_register_rejects_newline_in_id(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="newlines"):
        register_lattice_node(
            lattice_node_id="bad\nid",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_register_extra_kwarg_collision_rejected(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="collides"):
        register_lattice_node(
            lattice_node_id="x",
            substrate="x",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            written_at_utc="2026-05-16T00:00:00Z",  # reserved
            path=p,
            lock_path=l,
        )


def test_register_extra_kwarg_attached(tmp_ledger) -> None:
    p, l = tmp_ledger
    record = register_lattice_node(
        lattice_node_id="x",
        substrate="x",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="x",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        custom_priority="high",
        path=p,
        lock_path=l,
    )
    assert record["custom_priority"] == "high"


# -----------------------------------------------------------------------
# update_lattice_node — event-type transitions
# -----------------------------------------------------------------------


def test_update_appends_new_row(tmp_ledger) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="x",
        substrate="x",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="x",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    update_lattice_node(
        lattice_node_id="x",
        event_type=EVENT_PROMOTED,
        status=STATUS_LIFTED_DISPATCH_READY,
        path=p,
        lock_path=l,
    )
    rows = load_nodes(p)
    assert len(rows) == 2
    assert rows[0]["status"] == STATUS_LIFTED_PENDING_COUNCIL
    assert rows[1]["status"] == STATUS_LIFTED_DISPATCH_READY


def test_update_inherits_unmodified_fields(tmp_ledger) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="x",
        substrate="orig_substrate",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="orig_class",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        evidence_score=0.19,
        evidence_score_axis="contest-CPU",
        path=p,
        lock_path=l,
    )
    update_lattice_node(
        lattice_node_id="x",
        event_type=EVENT_PROMOTED,
        status=STATUS_DISPATCHED_EVIDENCE,
        path=p,
        lock_path=l,
    )
    latest = latest_node_state("x", path=p)
    assert latest["substrate"] == "orig_substrate"
    assert latest["architectural_class"] == "orig_class"
    assert latest["evidence_score"] == 0.19


def test_update_rejects_unknown_probe_id(tmp_ledger) -> None:
    p, l = tmp_ledger
    with pytest.raises(ValueError, match="no prior registered"):
        update_lattice_node(
            lattice_node_id="ghost",
            event_type=EVENT_PROMOTED,
            path=p,
            lock_path=l,
        )


def test_update_does_not_mutate_prior_row(tmp_ledger) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="x",
        substrate="x",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="x",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        notes="initial",
        path=p,
        lock_path=l,
    )
    update_lattice_node(
        lattice_node_id="x",
        event_type=EVENT_DEFERRED,
        status=STATUS_DEFERRED_OPERATOR,
        notes="deferred",
        path=p,
        lock_path=l,
    )
    rows = load_nodes(p)
    assert rows[0]["notes"] == "initial"
    assert rows[1]["notes"] == "deferred"


# -----------------------------------------------------------------------
# load_nodes / load_nodes_strict
# -----------------------------------------------------------------------


def test_load_empty_when_missing(tmp_path) -> None:
    p = tmp_path / "nope.jsonl"
    assert load_nodes(p) == []
    assert load_nodes_strict(p) == []


def test_load_lenient_skips_malformed(tmp_ledger) -> None:
    p, l = tmp_ledger
    p.write_text("not-json\n" + json.dumps({"a": 1}) + "\n")
    rows = load_nodes(p)
    assert rows == [{"a": 1}]


def test_load_strict_raises_on_malformed(tmp_ledger) -> None:
    p, l = tmp_ledger
    p.write_text("not-json\n")
    with pytest.raises(LatticeStateLedgerCorruptError):
        load_nodes_strict(p)


def test_load_strict_raises_on_non_dict_root(tmp_ledger) -> None:
    p, l = tmp_ledger
    p.write_text('"a string"\n')
    with pytest.raises(LatticeStateLedgerCorruptError):
        load_nodes_strict(p)


# -----------------------------------------------------------------------
# query helpers
# -----------------------------------------------------------------------


def _seed_canonical_corpus(p: Path, l: Path) -> None:
    """Seed 6 substrates spanning all 4 active rules + NeRV-family + outside-NeRV."""
    register_lattice_node(
        lattice_node_id="nscs06_v7",
        substrate="nscs06_v7",
        lattice_rule=RULE_1,
        horizon_class=HORIZON_WON,
        architectural_class="chroma_preserving_no_neural",
        status=STATUS_DEFERRED_OPERATOR,
        path=p,
        lock_path=l,
    )
    register_lattice_node(
        lattice_node_id="nscs01",
        substrate="nscs01_nullspace_split_renderer",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="pr95_paradigm_nullspace_split",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    register_lattice_node(
        lattice_node_id="a_stack",
        substrate="stack_of_stacks",
        lattice_rule=RULE_3,
        horizon_class=HORIZON_FRONTIER_PURSUIT,
        architectural_class="stack_composition",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    register_lattice_node(
        lattice_node_id="z6",
        substrate="z6_predictive_coding",
        lattice_rule=RULE_4,
        horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
        architectural_class="predictive_coding_hierarchical",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    register_lattice_node(
        lattice_node_id="sane_hnerv",
        substrate="sane_hnerv",
        lattice_rule=RULE_4,
        horizon_class=HORIZON_FRONTIER_PURSUIT,
        architectural_class="sane_hnerv_family",
        status=STATUS_LIFTED_DISPATCH_READY,
        path=p,
        lock_path=l,
    )
    register_lattice_node(
        lattice_node_id="hi_nerv",
        substrate="hi_nerv",
        lattice_rule=RULE_4,
        horizon_class=HORIZON_FRONTIER_PURSUIT,
        architectural_class="nerv_family",
        status=STATUS_LIFTED_DISPATCH_READY,
        path=p,
        lock_path=l,
    )


def test_query_by_substrate(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    rows = query_by_substrate("nscs01_nullspace_split_renderer", path=p)
    assert len(rows) == 1
    assert rows[0]["lattice_node_id"] == "nscs01"


def test_query_by_rule_returns_active_substrates(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    rule_4_nodes = query_by_rule(RULE_4, path=p)
    assert {n["lattice_node_id"] for n in rule_4_nodes} == {"z6", "sane_hnerv", "hi_nerv"}


def test_query_by_architectural_class(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    nodes = query_by_architectural_class("nerv_family", path=p)
    assert len(nodes) == 1
    assert nodes[0]["lattice_node_id"] == "hi_nerv"


def test_query_outside_nerv_family_excludes_nerv(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    outside = query_outside_nerv_family(path=p)
    ids = {n["lattice_node_id"] for n in outside}
    # Outside NeRV: nscs06_v7 + nscs01 + a_stack + z6
    assert ids == {"nscs06_v7", "nscs01", "a_stack", "z6"}
    # NeRV family excluded: sane_hnerv + hi_nerv
    assert "sane_hnerv" not in ids
    assert "hi_nerv" not in ids


def test_query_uncovered_rules_excludes_active_rules(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    uncovered = query_uncovered_rules(path=p)
    # Rule 1 is deferred (NSCS06 v7) — NOT counted as active coverage
    # Rules 2, 3, 4 all have at least one lifted_* substrate
    # Rule 5 has no coverage
    assert RULE_2 not in uncovered
    assert RULE_3 not in uncovered
    assert RULE_4 not in uncovered
    assert "rule_5_request_operator_review" in uncovered


def test_latest_node_state_chronological(tmp_ledger) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="x",
        substrate="x",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="x",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    update_lattice_node(
        lattice_node_id="x",
        event_type=EVENT_PROMOTED,
        status=STATUS_DISPATCHED_EVIDENCE,
        path=p,
        lock_path=l,
    )
    latest = latest_node_state("x", path=p)
    assert latest["status"] == STATUS_DISPATCHED_EVIDENCE


def test_compute_coverage_report(tmp_ledger) -> None:
    p, l = tmp_ledger
    _seed_canonical_corpus(p, l)
    report = compute_coverage_report(path=p)
    assert report.total_nodes == 6
    assert report.nerv_family_count == 2  # sane_hnerv + hi_nerv
    assert report.outside_nerv_count == 4
    assert report.rule_counts[RULE_4] == 3
    assert "rule_5_request_operator_review" in report.uncovered_rules
    # Rule 1 is uncovered because the only node is deferred (not active)
    assert RULE_1 in report.uncovered_rules


# -----------------------------------------------------------------------
# 4-process concurrent-append stress (per Catalog #245 exemplar)
# -----------------------------------------------------------------------


def _worker_append(args) -> None:
    p, l, lattice_node_id_base, count = args
    from tac.lattice_state_ledger import register_lattice_node as _r
    for i in range(count):
        _r(
            lattice_node_id=f"{lattice_node_id_base}_{i}",
            substrate=f"sub_{lattice_node_id_base}_{i}",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="x",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            path=p,
            lock_path=l,
        )


def test_concurrent_append_4proc_spawn_pool(tmp_ledger) -> None:
    p, l = tmp_ledger
    ctx = mp.get_context("spawn")
    rows_per_proc = 5
    with ctx.Pool(4) as pool:
        pool.map(
            _worker_append,
            [(p, l, f"w{i}", rows_per_proc) for i in range(4)],
        )
    rows = load_nodes(p)
    # 4 processes × 5 rows = 20 rows; all must survive
    assert len(rows) == 4 * rows_per_proc
    ids = {r["lattice_node_id"] for r in rows}
    assert len(ids) == 4 * rows_per_proc


# -----------------------------------------------------------------------
# JSONL byte-stability (sort_keys=True)
# -----------------------------------------------------------------------


def test_jsonl_byte_stable_sort_keys(tmp_ledger) -> None:
    p, l = tmp_ledger
    register_lattice_node(
        lattice_node_id="x",
        substrate="x",
        lattice_rule=RULE_2,
        horizon_class=HORIZON_PLATEAU_ADJACENT,
        architectural_class="x",
        status=STATUS_LIFTED_PENDING_COUNCIL,
        path=p,
        lock_path=l,
    )
    text = p.read_text()
    # The first line's keys should appear in sorted order
    first_line = text.splitlines()[0]
    keys_in_order = []
    pos = 0
    while True:
        start = first_line.find('"', pos)
        if start == -1:
            break
        end = first_line.find('"', start + 1)
        if end == -1:
            break
        key = first_line[start + 1 : end]
        # Heuristic: a "key" is followed by ':'
        if end + 1 < len(first_line) and first_line[end + 1] == ":":
            keys_in_order.append(key)
        pos = end + 1
    assert keys_in_order == sorted(keys_in_order)


# -----------------------------------------------------------------------
# Module-level constants exported
# -----------------------------------------------------------------------


def test_canonical_ledger_path_under_omx_state() -> None:
    assert ".omx/state/lattice_state.jsonl" in str(LATTICE_STATE_LEDGER_PATH)


def test_canonical_lock_path_sibling() -> None:
    assert LATTICE_STATE_LEDGER_LOCK == LATTICE_STATE_LEDGER_PATH.with_suffix(".jsonl.lock")
