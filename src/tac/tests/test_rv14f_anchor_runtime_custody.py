"""rv13 F4 + F7 + F12 — anchor custody, harvested-row facts, claim closure.

Every test here has a NAMED defect behind it, measured on live artifacts on
2026-08-19 and recorded in ``.omx/research/ddm_rv13_landing_wave_review_20260819.md``.
The controls are executed, not asserted-by-construction: each gate test is paired
with a case that makes the gate FIRE, because a detector that never fired is not
a detector.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pytest

from tac.frontier_scan import (
    RUNTIME_CUSTODY_MISSING,
    RUNTIME_CUSTODY_PINNED,
    RuntimeCustodyWarning,
    load_experiments_results_anchors,
)

REPO = Path(__file__).resolve().parents[3]


def _load_poller():
    """Import tools/modal_harvest_poller.py by path (tools/ is not a package)."""
    path = REPO / "tools" / "modal_harvest_poller.py"
    spec = importlib.util.spec_from_file_location("_rv14f_poller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_rv14f_poller"] = module
    spec.loader.exec_module(module)
    return module


poller = _load_poller()


def _write_anchor(repo_root: Path, name: str, payload: dict) -> Path:
    d = repo_root / "experiments" / "results" / "modal_auth_eval_mirror"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"contest_auth_eval_{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _base_row(**over) -> dict:
    row = {
        "schema": "modal_auth_eval_anchor_mirror.v2",
        "score": 0.15652626435208142,
        "score_axis": "contest_cuda",
        "evidence_grade": "contest-CUDA",
        "archive_sha256": "7ce46fd7" + "0" * 56,
        "archive_size_bytes": 176420,
        "runtime_tree_sha256": "d829ff29" + "1" * 56,
        "hardware_substrate": "linux_x86_64_t4",
        "lane_id": "lane_ddm_up3_thirteenth_move_t4_20260819",
        "n_samples": 600,
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# F12 — the pointer's CUDA leg carried null archive_bytes / lane_id
# ---------------------------------------------------------------------------


def test_f12_anchor_carries_archive_bytes_from_mirror_spelling(tmp_path):
    """The mirror writes ``archive_size_bytes``; the pointer reads ``archive_bytes``.

    The scanner dropped the field entirely, so a populated size on disk became a
    null in the pointer, which made ``read_frontier_archive_identity()`` refuse.
    """
    _write_anchor(tmp_path, "f12", _base_row())
    (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["archive_bytes"] == 176420
    assert anchor.extra["lane_id"] == "lane_ddm_up3_thirteenth_move_t4_20260819"


def test_f12_accepts_the_other_two_spellings(tmp_path):
    _write_anchor(
        tmp_path, "alt", _base_row(archive_size_bytes=None, archive_bytes=176525)
    )
    (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["archive_bytes"] == 176525


def test_f12_zero_bytes_is_a_measurement_not_a_missing_value(tmp_path):
    """CONTROL for the ``or``-chain bug: 0 must not fall through to the next key.

    An ``or``-chained lookup would skip a real 0 and silently read a different
    field. This is the shape that made the original defect invisible.
    """
    _write_anchor(tmp_path, "zero", _base_row(archive_size_bytes=0, archive_bytes=None))
    (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["archive_bytes"] == 0


def test_f12_non_integral_size_is_not_a_size(tmp_path):
    _write_anchor(tmp_path, "bad", _base_row(archive_size_bytes="not-a-number"))
    (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["archive_bytes"] is None


def test_f12_live_pointer_cuda_leg_now_carries_bytes():
    """The live surface, not a fixture. This is the field that was null tonight."""
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    leg = pointer.get("our_local_frontier_contest_cuda") or {}
    assert isinstance((leg.get("extra") or {}).get("archive_bytes"), int), (
        "the CUDA leg lost archive_bytes again; tac.candidate_seal."
        "read_frontier_archive_identity() refuses without it"
    )


# ---------------------------------------------------------------------------
# F4 — one archive sha carried two contradictory contest-CUDA scores
# ---------------------------------------------------------------------------


def test_f4_pinned_runtime_tree_is_carried_and_stamped(tmp_path):
    _write_anchor(tmp_path, "pinned", _base_row())
    (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["runtime_tree_sha256"] == "d829ff29" + "1" * 56
    assert anchor.extra["runtime_custody"] == RUNTIME_CUSTODY_PINNED


def test_f4_missing_custody_FIRES_the_warning(tmp_path):
    """POSITIVE CONTROL. The gate must actually fire on a v1-shaped row."""
    _write_anchor(
        tmp_path,
        "legacy",
        _base_row(schema="modal_auth_eval_anchor_mirror.v1", runtime_tree_sha256=None),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.extra["runtime_custody"] == RUNTIME_CUSTODY_MISSING
    assert any(issubclass(w.category, RuntimeCustodyWarning) for w in caught), (
        "a custody-less anchor row must be LOUD; a silent instrument cannot be "
        "distinguished from a checked one"
    )


def test_f4_legacy_row_is_still_readable_and_still_qualifies(tmp_path):
    """Additive, not invalidating: legacy rows keep their score and qualification.

    Refusing them outright would have invalidated the four honest rows that
    already moved the pointer.
    """
    _write_anchor(
        tmp_path,
        "legacy2",
        _base_row(schema="modal_auth_eval_anchor_mirror.v1", runtime_tree_sha256=None),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeCustodyWarning)
        (anchor,) = load_experiments_results_anchors(tmp_path)
    assert anchor.score == pytest.approx(0.15652626435208142)
    assert anchor.is_qualifying()


def test_f4_pinned_row_does_NOT_fire_the_warning(tmp_path):
    """The other half of the control: no false positive on a good row."""
    _write_anchor(tmp_path, "quiet", _base_row())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_experiments_results_anchors(tmp_path)
    assert not [w for w in caught if issubclass(w.category, RuntimeCustodyWarning)]


def test_f4_the_measured_collision_becomes_distinguishable(tmp_path):
    """The live hazard, reproduced: ONE archive sha, two passing scores.

    0.15710198 and 79.40216174 are the real values measured on archive
    ``35c318d5…``; the only differing input was the runtime tree. Keyed on the
    archive alone the two rows are identical, which is what made the pointer's
    safety accidental rather than designed.
    """
    sha = "35c318d5" + "a" * 56
    _write_anchor(
        tmp_path, "collide_good",
        _base_row(score=0.15710198138050818, archive_sha256=sha,
                  runtime_tree_sha256="da91e067" + "b" * 56),
    )
    _write_anchor(
        tmp_path, "collide_broken",
        _base_row(score=79.40216174747616, archive_sha256=sha,
                  runtime_tree_sha256="71c75468" + "c" * 56),
    )
    anchors = load_experiments_results_anchors(tmp_path)
    assert len({a.archive_sha256 for a in anchors}) == 1
    keys = {(a.archive_sha256, a.extra["runtime_tree_sha256"]) for a in anchors}
    assert len(keys) == 2, "keyed on (archive, runtime_tree) the two rows must differ"


def test_f4_live_mirror_rows_all_carry_runtime_custody():
    """The live mirror directory, after the backfill."""
    d = REPO / "experiments/results/modal_auth_eval_mirror"
    rows = sorted(d.glob("contest_auth_eval*.json"))
    assert rows, "the live mirror is empty; the scanner would see no CUDA anchors"
    missing = [p.name for p in rows if not json.loads(p.read_text()).get("runtime_tree_sha256")]
    assert not missing, f"mirror rows without runtime-tree custody: {missing}"


# ---------------------------------------------------------------------------
# F4 writer half — build_anchor_mirror emits the tree it was scored under
# ---------------------------------------------------------------------------


def _receipt(**over) -> dict:
    r = {
        "score_recomputed_from_components": 0.15652626435208142,
        "final_score": 0.16,
        "expected_archive_sha256": "7ce46fd7" + "0" * 56,
        "expected_archive_size_bytes": 176420,
        "archive_size_bytes": 176420,
        "expected_runtime_tree_sha256": "d829ff29" + "1" * 56,
        "inflate_sh_rel": "inflate.sh",
        "inflate_device_policy": "auto",
        "score_axis": "contest_cuda",
        "evidence_grade": "contest-CUDA",
        "gpu_t4_match": True,
        "gpu_model": "Tesla T4",
        "n_samples": 600,
        "passed": True,
    }
    r.update(over)
    return r


def test_f4_writer_emits_v2_with_runtime_tree(tmp_path):
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_receipt()))
    payload, blocker = poller.build_anchor_mirror(
        _receipt(), lane_id="lane_x", source_receipt=receipt
    )
    assert blocker is None
    assert payload["schema"] == "modal_auth_eval_anchor_mirror.v2"
    assert payload["runtime_tree_sha256"] == "d829ff29" + "1" * 56
    assert payload["archive_size_bytes"] == 176420
    assert payload["lane_id"] == "lane_x"


def test_f4_writer_never_falls_back_to_rounded_final_score(tmp_path):
    """CONTROL: a receipt with only the 2dp ``final_score`` must be REFUSED.

    A fallback is how a rounded 0.16 becomes an anchor.
    """
    receipt = tmp_path / "r.json"
    receipt.write_text("{}")
    payload, blocker = poller.build_anchor_mirror(
        _receipt(score_recomputed_from_components=None),
        lane_id="l",
        source_receipt=receipt,
    )
    assert payload is None
    assert "score_recomputed_from_components" in blocker


# ---------------------------------------------------------------------------
# F4 backfill — recovery from a hash-pinned receipt, never assertion
# ---------------------------------------------------------------------------


def test_backfill_recovers_from_a_verified_receipt(tmp_path):
    import hashlib

    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_receipt()))
    mirror = tmp_path / "m.json"
    mirror.write_text(json.dumps({
        "schema": "modal_auth_eval_anchor_mirror.v1",
        "source_receipt": str(receipt),
        "source_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }))
    verdict = poller.backfill_runtime_custody(mirror)
    assert verdict["action"] == "backfilled"
    after = json.loads(mirror.read_text())
    assert after["runtime_tree_sha256"] == "d829ff29" + "1" * 56
    assert after["schema"] == "modal_auth_eval_anchor_mirror.v2"


def test_backfill_REFUSES_when_the_receipt_sha_moved(tmp_path):
    """POSITIVE CONTROL. The pin is the whole point.

    A receipt whose bytes changed is not the receipt the row was written from;
    copying from it would forge custody rather than recover it.
    """
    receipt = tmp_path / "MODAL_REMOTE_RESULT.json"
    receipt.write_text(json.dumps(_receipt()))
    mirror = tmp_path / "m.json"
    mirror.write_text(json.dumps({
        "schema": "modal_auth_eval_anchor_mirror.v1",
        "source_receipt": str(receipt),
        "source_receipt_sha256": "0" * 64,  # deliberately wrong
    }))
    verdict = poller.backfill_runtime_custody(mirror)
    assert verdict["action"] == "refused"
    assert "sha mismatch" in verdict["reason"]
    assert "runtime_tree_sha256" not in json.loads(mirror.read_text())


def test_backfill_refuses_when_the_receipt_is_gone(tmp_path):
    mirror = tmp_path / "m.json"
    mirror.write_text(json.dumps({
        "schema": "modal_auth_eval_anchor_mirror.v1",
        "source_receipt": str(tmp_path / "absent.json"),
        "source_receipt_sha256": "a" * 64,
    }))
    assert poller.backfill_runtime_custody(mirror)["action"] == "refused"


def test_backfill_dry_run_does_not_write(tmp_path):
    import hashlib

    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps(_receipt()))
    mirror = tmp_path / "m.json"
    before = json.dumps({
        "schema": "modal_auth_eval_anchor_mirror.v1",
        "source_receipt": str(receipt),
        "source_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    })
    mirror.write_text(before)
    assert poller.backfill_runtime_custody(mirror, dry_run=True)["action"] == "backfilled"
    assert mirror.read_text() == before


# ---------------------------------------------------------------------------
# F7 — harvested rows dropped lane_id, score, archive sha
# ---------------------------------------------------------------------------


def test_f7_harvest_facts_copy_the_recomputed_score(tmp_path):
    facts = poller._harvest_outcome_facts(_receipt())
    assert facts["score"] == pytest.approx(0.15652626435208142)
    assert facts["score_axis"] == "contest_cuda"
    assert facts["archive_sha256"] == "7ce46fd7" + "0" * 56
    assert facts["archive_bytes"] == 176420
    assert facts["evidence_grade"] == "contest-CUDA"


def test_f7_harvest_facts_REFUSE_the_rounded_final_score():
    """POSITIVE CONTROL. ``final_score`` is 2dp and must never become the score."""
    facts = poller._harvest_outcome_facts(
        _receipt(score_recomputed_from_components=None)
    )
    assert "score" not in facts, "the rounded final_score leaked into the ledger"


def test_f7_claim_close_without_lane_id_says_so(tmp_path):
    note = poller._close_terminal_claim(
        lane_id=None, call_id="fc-x", result=_receipt(), out_dir=tmp_path
    )
    assert "CLAIM NOT CLOSED" in note


def test_f7_claim_close_failure_never_loses_the_harvest(tmp_path, monkeypatch):
    """POSITIVE CONTROL for the fail-open path.

    Bookkeeping must never destroy a real paid harvest, and the failure must
    land on disk rather than only in a return value.
    """
    import tac.deploy.claims as claims

    def boom(**_):
        raise RuntimeError("claims ledger locked")

    monkeypatch.setattr(claims, "terminal_dispatch_claim", boom)
    note = poller._close_terminal_claim(
        lane_id="lane_x", call_id="fc-y", result=_receipt(), out_dir=tmp_path
    )
    assert "CLAIM CLOSE FAILED" in note
    written = json.loads((tmp_path / "CLAIM_CLOSE_FAILED.json").read_text())
    assert written["lane_id"] == "lane_x"
    assert written["call_id"] == "fc-y"


def test_f7_poller_threads_lane_id_into_the_outcome_row():
    """The defect was structural: the CLI accepted --lane-id and used it only
    for the mirror, so every harvested ledger row was lane-less and every
    completed fire looked like a phantom to a lane-keyed reconcile."""
    src = (REPO / "tools" / "modal_harvest_poller.py").read_text()
    head, _, _ = src.partition("if not args.no_anchor_mirror")
    _, _, update_block = head.rpartition("update_call_id_outcome(")
    assert "lane_id=args.lane_id" in update_block, (
        "the harvested ledger row must carry lane_id or the claim reconcile is blind"
    )
