from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_bridge_sparse_action_candidates.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("build_pr85_bridge_sparse_action_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _zip_info(name: str, *, compress_type: int = zipfile.ZIP_DEFLATED) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_source_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x", compress_type=zipfile.ZIP_STORED), b"fixture-pr85-x")


def _randmulti_qrm1() -> bytes:
    row0 = bytearray(600)
    row0[0] = 3
    row0[17] = 1
    group0 = module._encode_sparse_row(bytes(row0))
    group0 += b"".join(module._encode_sparse_row(bytes(600)) for _ in range(11))
    row1 = bytearray(600)
    row1[1] = 2
    group1 = module._encode_sparse_row(bytes(row1))
    raw = (
        b"QRM1"
        + (2).to_bytes(2, "little")
        + (0).to_bytes(2, "little")
        + group0
        + (1).to_bytes(2, "little")
        + group1
    )
    return brotli.compress(raw, quality=5)


def _fixture_qpost() -> bytes:
    post = b"".join(
        bytes([(stage * 13 + pair) % 251 for pair in range(600)])
        for stage in range(4)
    )
    streams = {
        "post": brotli.compress(post, quality=5),
        "shift": brotli.compress(
            module.post_motion.recode._encode_delta_choice(
                b"SD4",
                bytes([40 if pair % 3 else 41 for pair in range(600)]),
                default_choice=40,
            ),
            quality=5,
        ),
        "frac": brotli.compress(
            module.post_motion.recode._encode_sparse_choice(
                b"FV1",
                bytes([4 if pair % 5 else 6 for pair in range(600)]),
                default_choice=4,
            ),
            quality=5,
        ),
        "frac2": brotli.compress(
            b"FH2" + bytes([4 if pair % 7 else 2 for pair in range(600)]),
            quality=5,
        ),
        "frac3": brotli.compress(
            module.post_motion.recode._encode_delta_choice(
                b"FD3",
                bytes([4 if pair % 11 else 7 for pair in range(600)]),
                default_choice=4,
            ),
            quality=5,
        ),
        "bias": brotli.compress(b"BD1" + bytes([0]) * 600, quality=5),
        "region": brotli.compress(b"RH1" + bytes([0]) * 600, quality=5),
        "randmulti": _randmulti_qrm1(),
    }
    return module._pack_qpost(streams)


def _write_bridge_archive(path: Path) -> None:
    members = {
        "masks.qma9": b"QMA9" + b"M" * 32,
        "renderer.bin": b"QH0" + b"R" * 64,
        "optimized_poses.bin": b"\x00" * (600 * 6 * 2),
        "qpost.bin": _fixture_qpost(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in module.MEMBER_ORDER:
            zf.writestr(
                _zip_info(name),
                members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _write_policy_json(path: Path, selected_group_ids: list[int]) -> None:
    payload = {
        "schema": "fixture_pr85_randmulti_group_policy_candidates_v1",
        "score_claim": False,
        "dispatch_performed": False,
        "policies": [
            {
                "candidate_policy_id": "fixture_keep_selected",
                "selected_group_ids": selected_group_ids,
                "planning_only": True,
            }
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_exact_evidence_override(blockers: list[str]) -> dict[str, object]:
    return {
        "allowed_preflight_blockers": blockers,
        "archive_sha256": "a" * 64,
        "auth_json": "experiments/results/fixture/contest_auth_eval.adjudicated.json",
        "evidence_grade": "A++ exact CUDA fixture",
        "override_id": "fixture_exact_evidence_override",
        "rationale": "fixture exact evidence for dispatch-preflight override",
        "score_delta_vs_pr85": -0.001,
    }


def _add_fixture_action_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy_id: str = "fixture_all_qpost_keep_selected",
    selected_qpost_groups: tuple[str, ...] = module.POST_MOTION_GROUPS,
    selected_randmulti_group_ids: tuple[int, ...] | None = None,
    exact_evidence_override: dict[str, object] | None = None,
) -> str:
    policy: dict[str, object] = {
        "qpost_policy_id": "fixture_qpost",
        "selected_qpost_groups": selected_qpost_groups,
        "basis": "fixture policy",
    }
    if selected_randmulti_group_ids is None:
        policy["randmulti_policy_id"] = "fixture_keep_selected"
    else:
        policy["selected_randmulti_group_ids"] = selected_randmulti_group_ids
    if exact_evidence_override is not None:
        policy["exact_evidence_override"] = exact_evidence_override
    monkeypatch.setitem(
        module.ACTION_POLICIES,
        policy_id,
        policy,
    )
    return policy_id


def test_bridge_sparse_action_candidates_are_deterministic_and_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0])
    policy_id = _add_fixture_action_policy(monkeypatch)

    first = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out1",
        policy_ids=[policy_id],
    )
    second = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out2",
        policy_ids=[policy_id],
    )

    first_manifest = first["candidates"][0]
    second_manifest = second["candidates"][0]
    assert first_manifest["score_claim"] is False
    assert first_manifest["dispatch_performed"] is False
    assert first_manifest["ready_for_exact_eval_dispatch_claim"] is True
    assert first_manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert first_manifest["dispatch_preflight"]["status"] == "passed"
    assert first_manifest["dispatch_preflight"]["blocker_ids"] == []
    assert first_manifest["candidate_archive"]["archive_sha256"] == second_manifest["candidate_archive"]["archive_sha256"]
    assert first_manifest["charged_byte_deltas"]["member_deltas"]["qpost.bin"]["candidate_sha256"] == second_manifest["charged_byte_deltas"]["member_deltas"]["qpost.bin"]["candidate_sha256"]
    assert first_manifest["safe_archive_members"] == {
        "status": "passed",
        "expected_order": list(module.MEMBER_ORDER),
        "observed_order": list(module.MEMBER_ORDER),
        "zip_slip_safe": True,
        "duplicate_members": False,
    }
    assert first_manifest["selected_action_policy"]["selected_randmulti_group_ids"] == [0]
    randmulti = next(row for row in first_manifest["transforms"] if row["segment"] == "randmulti")
    assert randmulti["selected_group_ids"] == [0]
    assert randmulti["omitted_group_ids"] == [1]

    with zipfile.ZipFile(tmp_path / "out1" / policy_id / "archive.zip", "r") as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == list(module.MEMBER_ORDER)
        assert all(info.date_time == module.FIXED_ZIP_TIMESTAMP for info in infos)
        qpost = zf.read("qpost.bin")
    streams, _meta = module._parse_qpost(qpost)
    groups, _report = module._parse_qrm1_groups(streams["randmulti"])
    assert sorted(groups) == [0]


def test_dispatch_preflight_blocks_whole_randmulti_deletion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0])
    policy_id = _add_fixture_action_policy(
        monkeypatch,
        policy_id="fixture_delete_all_randmulti",
        selected_randmulti_group_ids=(),
    )

    summary = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out",
        policy_ids=[policy_id],
    )

    manifest = summary["candidates"][0]
    preflight = manifest["dispatch_preflight"]
    assert manifest["ready_for_exact_eval_dispatch_claim"] is False
    assert manifest["dispatch_gate"] == "planning_only/preflight_blocked"
    assert preflight["status"] == "blocked"
    assert preflight["fail_closed"] is True
    assert preflight["blocker_ids"] == ["whole_randmulti_deletion"]
    assert preflight["exact_evidence_override"]["required"] is True
    assert preflight["exact_evidence_override"]["present"] is False
    assert summary["dispatchable_candidate_count"] == 0
    assert summary["dispatch_preflight_blocked_candidate_count"] == 1


@pytest.mark.parametrize(
    ("policy_id", "selected_qpost_groups", "expected_blocker"),
    [
        ("fixture_delete_all_post", module.MOTION_GROUPS, "whole_post_deletion"),
        ("fixture_delete_all_motion", module.POST_GROUPS, "whole_motion_deletion"),
    ],
)
def test_dispatch_preflight_blocks_whole_qpost_family_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_id: str,
    selected_qpost_groups: tuple[str, ...],
    expected_blocker: str,
) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0])
    policy_id = _add_fixture_action_policy(
        monkeypatch,
        policy_id=policy_id,
        selected_qpost_groups=selected_qpost_groups,
    )

    summary = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out",
        policy_ids=[policy_id],
    )

    preflight = summary["candidates"][0]["dispatch_preflight"]
    assert preflight["status"] == "blocked"
    assert expected_blocker in preflight["blocker_ids"]
    assert "protected_qpost_group_deletion" in preflight["blocker_ids"]
    assert summary["candidates"][0]["ready_for_exact_eval_dispatch_claim"] is False


def test_dispatch_preflight_blocks_protected_qpost_group_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0])
    policy_id = _add_fixture_action_policy(
        monkeypatch,
        policy_id="fixture_delete_post_stage4",
        selected_qpost_groups=(
            "post_stage1",
            "post_stage2",
            "post_stage3",
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
    )

    summary = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out",
        policy_ids=[policy_id],
    )

    preflight = summary["candidates"][0]["dispatch_preflight"]
    assert preflight["status"] == "blocked"
    assert preflight["blocker_ids"] == ["protected_qpost_group_deletion"]
    blocker = preflight["blockers"][0]
    assert blocker["deleted_protected_qpost_groups"] == ["post_stage4"]
    assert summary["candidates"][0]["dispatch_gate"] == "planning_only/preflight_blocked"


def test_exact_evidence_override_allows_protected_qpost_deletion_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0])
    policy_id = _add_fixture_action_policy(
        monkeypatch,
        policy_id="fixture_delete_post_stage4_with_override",
        selected_qpost_groups=(
            "post_stage1",
            "post_stage2",
            "post_stage3",
            "motion_shift",
            "motion_frac",
            "motion_frac2",
            "motion_frac3",
        ),
        exact_evidence_override=_fixture_exact_evidence_override(["protected_qpost_group_deletion"]),
    )

    summary = module.build_candidates(
        source_archive=source,
        bridge_archive=bridge,
        randmulti_policy_json=policies,
        out_dir=tmp_path / "out",
        policy_ids=[policy_id],
    )

    manifest = summary["candidates"][0]
    preflight = manifest["dispatch_preflight"]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch_claim"] is True
    assert manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert preflight["status"] == "passed_with_exact_evidence_override"
    assert preflight["exact_evidence_override"]["valid"] is True
    assert preflight["exact_evidence_override"]["covers_blockers"] is True


def test_bridge_archive_member_validation_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    policies = tmp_path / "policies.json"
    bad_bridge = tmp_path / "bad_bridge.zip"
    _write_source_archive(source)
    _write_policy_json(policies, [0])
    with zipfile.ZipFile(bad_bridge, "w") as zf:
        zf.writestr(_zip_info("../qpost.bin"), b"bad")

    with pytest.raises(module.BridgeSparseActionError, match="unsafe ZIP member"):
        module.build_candidates(
            source_archive=source,
            bridge_archive=bad_bridge,
            randmulti_policy_json=policies,
            out_dir=tmp_path / "out",
        )


def test_policy_validation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.zip"
    bridge = tmp_path / "bridge.zip"
    policies = tmp_path / "policies.json"
    _write_source_archive(source)
    _write_bridge_archive(bridge)
    _write_policy_json(policies, [0, 0])
    policy_id = _add_fixture_action_policy(monkeypatch)

    with pytest.raises(module.BridgeSparseActionError, match="duplicate selected randmulti"):
        module.build_candidates(
            source_archive=source,
            bridge_archive=bridge,
            randmulti_policy_json=policies,
            out_dir=tmp_path / "out",
            policy_ids=[policy_id],
        )

    monkeypatch.setitem(
        module.ACTION_POLICIES,
        "bad_qpost_group",
        {
            "selected_qpost_groups": ("post_stage1", "randmulti_group0"),
            "selected_randmulti_group_ids": (0,),
            "basis": "bad fixture",
        },
    )
    _write_policy_json(policies, [0])
    with pytest.raises(module.BridgeSparseActionError, match="unknown selected qpost"):
        module.build_candidates(
            source_archive=source,
            bridge_archive=bridge,
            randmulti_policy_json=policies,
            out_dir=tmp_path / "out",
            policy_ids=["bad_qpost_group"],
        )
