from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr85_qrgb_transfer_actions.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_pr85_qrgb_transfer_actions_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _zip_single(path: Path, member: str, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(member, data)
    return path


def _sparse_raw(assignments: dict[int, int], total: int) -> bytes:
    expanded = dict(assignments)
    positions = sorted(assignments)
    prev = -1
    for pos in positions:
        while pos - prev - 1 > 255:
            prev += 256
            expanded.setdefault(prev, 0)
        prev = pos
    positions = sorted(expanded)
    gaps = []
    prev = -1
    for pos in positions:
        gap = pos - prev - 1
        assert 0 <= gap <= 255
        gaps.append(gap)
        prev = pos
    return bytes(gaps) + bytes(np.array([expanded[pos] for pos in positions], dtype=np.int8))


def _ordered_position_for_semantic_plane(semantic_plane: int, pair: int) -> int:
    source_plane = module.RGB_COMPACT_PLANE_ORDER.index(semantic_plane)
    return source_plane * module.PAIR_COUNT + pair


def test_qrgb_decoder_restores_semantic_plane_order() -> None:
    raw = _sparse_raw(
        {
            _ordered_position_for_semantic_plane(3, 2): 4,
            _ordered_position_for_semantic_plane(22, 2): -5,
        },
        48 * module.PAIR_COUNT,
    )

    planes, stats = module._decode_qrgb_residual(brotli.compress(raw))

    assert int(planes[3, 2]) == 4
    assert int(planes[22, 2]) == -5
    assert int(np.count_nonzero(planes[:, 2])) == 2


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _sparse_randmulti(groups: list[tuple[int, int, int]]) -> bytes:
    rows = []
    wanted = {(lh, lw, amp): value for lh, lw, amp, value in groups}
    for lh, lw, amp, selection_rows in module.PR85_HEADERLESS_RANDMULTI_SPECS:
        for _ in range(selection_rows):
            value = wanted.get((lh, lw, amp), 0)
            if value:
                rows.append(bytes([1]) + _varint(0) + bytes([value]))
            else:
                rows.append(bytes([0]))
    return b"".join(rows)


def _pr85_archive(path: Path) -> Path:
    sys.path.insert(0, str(REPO / "src"))
    from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle

    segments = {name: b"x" for name in SEGMENT_ORDER}
    segments["mask"] = b"QMA9" + b"\0" * 16
    segments["model"] = brotli.compress(b"QH0" + b"\0" * 8)
    segments["pose"] = brotli.compress(b"P1D1" + b"\0" * 8)
    segments["post"] = brotli.compress(bytes([0]) * (module.PAIR_COUNT * 4))
    segments["shift"] = brotli.compress(b"SD4" + bytes([0]) * module.PAIR_COUNT)
    segments["frac"] = brotli.compress(b"FV1" + (0).to_bytes(2, "little"))
    segments["frac2"] = brotli.compress(b"FH2" + bytes([4]) * module.PAIR_COUNT)
    segments["frac3"] = brotli.compress(b"FD3" + bytes([0]) * module.PAIR_COUNT)
    segments["bias"] = brotli.compress(b"BD1" + bytes([0]) * module.PAIR_COUNT)
    segments["region"] = brotli.compress(b"RH1" + bytes([0]) * module.PAIR_COUNT)
    segments["randmulti"] = brotli.compress(
        _sparse_randmulti(
            [
                (222, 222, 4, 3),
                (224, 222, 4, 5),
            ]
        )
    )
    return _zip_single(path, "x", pack_pr85_bundle(segments, header_mode="explicit_30"))


def _pr90_archive_and_probe(tmp_path: Path) -> tuple[Path, Path]:
    assignments = {
        _ordered_position_for_semantic_plane(3, 0): 3,
        _ordered_position_for_semantic_plane(0, 1): -2,
        _ordered_position_for_semantic_plane(2, 1): 5,
        _ordered_position_for_semantic_plane(18, 2): 4,
        _ordered_position_for_semantic_plane(33, 3): -3,
    }
    qrgb = brotli.compress(_sparse_raw(assignments, 48 * module.PAIR_COUNT))
    payload = b"a" * 100 + qrgb + b"z" * 10
    archive = _zip_single(tmp_path / "pr90.zip", "p", payload)
    probe = _write_json(
        tmp_path / "payload_probe.json",
        {
            "payload_sha256": module._sha256(payload),
            "payload_len": len(payload),
            "slices": {"qrgb_residual_br": {"offset": 100, "len": len(qrgb)}},
        },
    )
    return archive, probe


def _pair_planning(path: Path) -> Path:
    scorer = _write_json(
        path.parent / "scorer.json",
        {
            "schema": "pr85_scorer_gradient_atom_opportunity_v1",
            "stable_plan_digest_sha256": "f" * 64,
            "atom_ranking": [
                {
                    "atom_id": f"fixture:pair_{pair:04d}",
                    "pair_index": pair,
                    "frame_indices": [pair * 2, pair * 2 + 1],
                    "ranking_score": 0.001 - pair * 0.00001,
                    "byte_break_even": {
                        "combined": {
                            "max_charged_bytes_for_zero_net_change": 500.0 - pair,
                            "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
                        },
                        "pose_only": {
                            "max_charged_bytes_for_zero_net_change": 300.0 - pair,
                            "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
                        },
                        "seg_only": {
                            "max_charged_bytes_for_zero_net_change": 200.0,
                            "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
                        },
                    },
                }
                for pair in range(4)
            ],
        },
    )
    return _write_json(
        path,
        {
            "schema": "pr85_pair_atom_candidate_readiness_v1",
            "score_claim": False,
            "dispatch_unlocked": False,
            "candidate_archive_count": 0,
            "scorer_gradient_plan": {
                "path": str(scorer),
                "sha256": module._sha256_file(scorer),
                "top_atoms": [],
            },
        },
    )


def test_build_plan_emits_locked_pair_action_evidence(tmp_path: Path) -> None:
    pr85 = _pr85_archive(tmp_path / "pr85.zip")
    pr90, probe = _pr90_archive_and_probe(tmp_path)
    pair = _pair_planning(tmp_path / "pair.json")

    plan = module.build_plan(
        pr85_archive=pr85,
        pr90_archive=pr90,
        pr90_probe_json=probe,
        pair_planning_json=pair,
        top_n=4,
        max_candidates=4,
    )

    assert plan["schema"] == module.SCHEMA
    assert plan["dispatch_unlocked"] is False
    assert plan["score_claim"] is False
    assert plan["candidate_count"] >= 3
    assert plan["ready_for_exact_eval_after_lane_claim_count"] == 0
    evidence = plan["pair_action_evidence"]
    assert evidence["schema"] == module.PAIR_ACTION_EVIDENCE_SCHEMA
    assert evidence["dispatch_performed"] is False
    assert evidence["candidates"]
    first = evidence["candidates"][0]
    assert first["archive_changing_path"] is None
    assert first["charged_bytes_proxy"]["candidate_action_bytes"] >= 1
    assert first["actions"][0]["source_artifact_sha256"] == "f" * 64
    assert first["actions"][0]["source_value"] != first["actions"][0]["candidate_value"]


def test_plan_digest_is_deterministic(tmp_path: Path) -> None:
    pr85 = _pr85_archive(tmp_path / "pr85.zip")
    pr90, probe = _pr90_archive_and_probe(tmp_path)
    pair = _pair_planning(tmp_path / "pair.json")

    one = module.build_plan(
        pr85_archive=pr85,
        pr90_archive=pr90,
        pr90_probe_json=probe,
        pair_planning_json=pair,
        top_n=4,
        max_candidates=4,
    )
    two = module.build_plan(
        pr85_archive=pr85,
        pr90_archive=pr90,
        pr90_probe_json=probe,
        pair_planning_json=pair,
        top_n=4,
        max_candidates=4,
    )

    assert one["stable_plan_digest_sha256"] == two["stable_plan_digest_sha256"]
    assert [row["candidate_id"] for row in one["ranked_candidates"]] == [
        row["candidate_id"] for row in two["ranked_candidates"]
    ]
