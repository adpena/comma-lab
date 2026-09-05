"""A refused best-per-axis anchor must not shadow an admissible twin at the same score.

2026-09-05, move 28: two T4 rows on the same 176,448 B archive — lane ``…_t4_v3_lattice…``
(refused by the checkpoint-maturity gate: variant token ``v3``) and the compliant custody
twin ``ddm_pc1_t4_lattice_x4_on_rc1_20260905``. The refresh took only the scan's single
best row, refused it, and kept the prior pointer. The refresh now walks the ranked list.
"""

from __future__ import annotations

from tac.checkpoint_maturity import pointer_promotion_verdict


def test_the_two_move_28_lane_ids_are_gated_as_recorded() -> None:
    refused, _ = pointer_promotion_verdict("ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905")
    allowed, _ = pointer_promotion_verdict("ddm_pc1_t4_lattice_x4_on_rc1_20260905")
    assert refused is False and allowed is True


def test_refresh_walks_top5_and_takes_the_first_admissible(monkeypatch, tmp_path) -> None:
    import tac.canonical_frontier_pointer as cfp

    def fake_payload(_root):
        row = {
            "score": 0.1451981569076111,
            "axis": "contest_cuda",
            "archive_sha256": "891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40",
            "archive_bytes": 176448,
            "source_path": "x",
        }
        refused = dict(row, extra={"lane_id": "ddm_pc1_t4_v3_lattice_x4_on_rc1_20260905", "evidence_grade": "contest-CUDA"})
        twin = dict(row, extra={"lane_id": "ddm_pc1_t4_lattice_x4_on_rc1_20260905", "evidence_grade": "contest-CUDA"})
        return {
            "best_per_axis": {"contest_cuda": refused},
            "top_5_per_axis": {"contest_cuda": [refused, twin]},
        }

    import tac.frontier_scan as fs

    monkeypatch.setattr(fs, "build_frontier_scan_payload", fake_payload)
    monkeypatch.setattr(cfp, "load_canonical_frontier_pointer_lenient", lambda **_: None)
    pointer = cfp.refresh_canonical_frontier_from_local_state(repo_root=tmp_path, write=False)
    anchor = pointer.our_local_frontier_contest_cuda
    assert anchor is not None
    assert anchor.lane_id == "ddm_pc1_t4_lattice_x4_on_rc1_20260905"
    refusals = (pointer.refresh_provenance or {}).get("checkpoint_maturity_refusals") or []
    assert any("v3" in str(r.get("reason", "")) for r in refusals)
