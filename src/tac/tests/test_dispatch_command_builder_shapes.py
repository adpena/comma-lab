# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

from tac.optimizer.exact_readiness import runtime_dependency_manifest

REPO = Path(__file__).resolve().parents[3]


def _load_tool(module_name: str):
    path = REPO / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ready_lightning_candidate() -> dict:
    return {
        "candidate_id": "candidate42",
        "lane_id": "pr106_latent_sidecar",
        "archive_path": "experiments/results/candidate42/archive.zip",
        "predicted_band": [0.19, 0.24],
        "ready_for_exact_eval_dispatch": True,
        "target_modes": ["contest_exact_eval"],
        "evidence_semantics": "contest_cuda_exact_eval_positive",
        "score_axis": "contest_cuda",
        "score_claim": False,
    }


def _write_ranked_input(tmp_path: Path, candidates: list[dict]) -> Path:
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps({"dispatch_ready": candidates}), encoding="utf-8")
    return ranked


def _recent_claim_timestamp() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _write_claims(
    path: Path,
    *,
    lane_id: str,
    platform: str,
    job_id: str,
    status: str = "running",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _recent_claim_timestamp()
    lines = [
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
        "|---|---|---|---|---|---|---|---|",
        (
            f"| {timestamp} | codex | {lane_id} | {platform} | {job_id} | "
            f"{timestamp} | {status} | active claim policy dispatch test |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_archive(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    return path


def _write_runtime_custody_files(archive: Path) -> str:
    runtime_root = archive.parent
    fixture_repo_root = runtime_root.parent
    (fixture_repo_root / "upstream").mkdir(parents=True, exist_ok=True)
    (fixture_repo_root / "upstream" / "evaluate.py").write_text(
        "# fixture upstream evaluator\n",
        encoding="utf-8",
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_size = archive.stat().st_size
    (runtime_root / "inflate.py").write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2] if len(sys.argv) > 2 else sys.argv[-1]).write_bytes(b'')\n",
        encoding="utf-8",
    )
    (runtime_root / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "python \"$SCRIPT_DIR/inflate.py\" \"$@\"\n",
        encoding="utf-8",
    )
    (runtime_root / "inflate.sh").chmod(0o755)
    (runtime_root / "report.txt").write_text(
        f"archive_sha256={archive_sha} archive_bytes={archive_size}\n",
        encoding="utf-8",
    )
    (runtime_root / "archive_manifest.json").write_text(
        json.dumps(
            {
                "candidate_archive_sha256": archive_sha,
                "candidate_archive_bytes": archive_size,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inflate_sh_sha = hashlib.sha256((runtime_root / "inflate.sh").read_bytes()).hexdigest()
    inflate_py_sha = hashlib.sha256((runtime_root / "inflate.py").read_bytes()).hexdigest()
    runtime_packet_manifest = runtime_root / "runtime_packet_manifest.json"
    runtime_packet_manifest.write_text(
        json.dumps(
            {
                "schema": "pr101_kaggle_proxy_runtime_packet_v1",
                "packet_dir": str(runtime_root),
                "runtime_custody": {
                    "runtime_files": [
                        {"relpath": "inflate.sh", "sha256": inflate_sh_sha},
                        {"relpath": "inflate.py", "sha256": inflate_py_sha},
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_root / "runtime_consumption_proof.json").write_text(
        json.dumps(
            {
                "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
                "proof_kind": "fixture_runtime_bound_pr101_proof",
                "manifest_path": str(runtime_packet_manifest),
                "manifest_sha256": hashlib.sha256(
                    runtime_packet_manifest.read_bytes()
                ).hexdigest(),
                "packet_dir": str(runtime_root),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "inflate_sh_routes_to_packet_inflate_py": True,
                "runtime_consumption_proven_for_supported_bias_params": True,
                "archive_unchanged_proof": {"archive_sha256": archive_sha},
                "inflate_wrapper_route_proof": {
                    "wrapper_invoked_packet_inflate_py": True,
                    "inflate_sh_sha256": inflate_sh_sha,
                    "packet_inflate_py_sha256": inflate_py_sha,
                },
                "inflate_static_bias_patch_proof": {
                    "inflate_sha256": inflate_py_sha,
                },
                "inflate_runtime_bias_logic_proof": {
                    "packet_inflate_function_executed": True,
                    "inflate_py_sha256": inflate_py_sha,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(
        runtime_dependency_manifest(runtime_root, fixture_repo_root)[
            "runtime_tree_sha256"
        ]
    )


def _write_archive_with_members(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, name in enumerate(names):
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                zf.writestr(info, f"payload-{idx}".encode("ascii"))
    return path


def _ready_custody_candidate(tmp_path: Path, **overrides) -> dict:
    archive = _write_archive(tmp_path / "submission" / "archive.zip")
    runtime_tree_sha256 = _write_runtime_custody_files(archive)
    runtime_manifest = runtime_dependency_manifest(archive.parent, archive.parent.parent)
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": archive.as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": (
            archive.parent / "runtime_consumption_proof.json"
        ).as_posix(),
    }
    candidate.update(overrides)
    return candidate


def test_parallel_dispatch_lightning_command_uses_stack_dispatcher_flags() -> None:
    tool = _load_tool("parallel_dispatch_top_k")

    cmd = tool._build_dispatch_cmd(
        _ready_lightning_candidate(),
        provider="lightning",
        lane_script="scripts/legacy_should_be_ignored_for_lightning.sh",
        label_prefix="batch",
        estimated_cost=0.11,
        max_dph=0.50,
    )

    joined = " ".join(cmd)
    assert "tools/lightning_dispatch_pr106_stack.py" in joined
    assert "--lane pr106_latent_sidecar" in joined
    assert "--archive experiments/results/candidate42/archive.zip" in joined
    assert "--predicted-low 0.19" in joined
    assert "--predicted-high 0.24" in joined
    assert "--job-name batch_candidate42" in joined
    assert "--dispatch-lane-id pr106_latent_sidecar" in joined
    assert "--lane-script" not in cmd
    assert "--predicted-band" not in cmd


def test_parallel_dispatch_lightning_existing_claim_command_uses_required_job(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")

    cmd = tool._build_dispatch_cmd(
        _ready_custody_candidate(tmp_path),
        provider="lightning",
        lane_script="scripts/legacy_should_be_ignored_for_lightning.sh",
        label_prefix="batch",
        estimated_cost=0.11,
        max_dph=0.50,
        dispatch_claims_path=tmp_path / "claims.md",
        claim_policy="require_active_claim",
        required_claim_instance_job_ids=("claimed_job_1",),
    )

    joined = " ".join(cmd)
    assert "--job-name claimed_job_1" in joined
    assert "--dispatch-lane-id pr106_latent_sidecar" in joined
    assert "--dispatch-claims-path" in cmd
    assert "--use-existing-dispatch-claim" in cmd


def test_parallel_dispatch_rejects_above_active_floor_archive_by_default(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(
            ranked,
            k=None,
            active_floor_archive_bytes=candidate["archive_size_bytes"] - 1,
        )
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "above_active_floor_archive_bytes" in message
    assert str(candidate["archive_size_bytes"]) in message


def test_parallel_dispatch_allows_above_active_floor_with_operator_reason(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(
        ranked,
        k=None,
        active_floor_archive_bytes=candidate["archive_size_bytes"] - 1,
        allow_above_active_floor_dispatch=True,
        operator_override_reason="calibration dispatch for non-rate-axis candidate",
    )

    assert loaded == [candidate]


def test_parallel_dispatch_require_active_claim_policy_allows_claim_then_dispatch(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    ranked = _write_ranked_input(tmp_path, [candidate])
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    _write_claims(
        claims,
        lane_id=str(candidate["lane_id"]),
        platform="lightning",
        job_id="job-1",
    )

    try:
        tool._load_top_k(ranked, k=None, dispatch_claims_path=claims)
    except tool.DispatchInputError as exc:
        preclaim_message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected active claim conflict in preclaim mode")

    assert "same_lane_active_dispatch_claim" in preclaim_message
    loaded = tool._load_top_k(
        ranked,
        k=None,
        dispatch_claims_path=claims,
        claim_policy="require_active_claim",
        required_claim_platform="lightning",
        required_claim_instance_job_ids=("job-1",),
    )

    assert loaded == [candidate]


def test_parallel_dispatch_exact_ready_queue_refuses_top_k_fallback(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    ranked = tmp_path / "exact_ready_queue.json"
    ranked.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_exact_eval_ready_queue_v1",
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": [candidate],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected exact-ready queue to refuse top_k fallback")

    assert "refusing top_k fallback" in message


def test_parallel_dispatch_accepts_candidate_archive_schema_with_exact_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive(tmp_path / "submission" / "pr101_candidate.zip", b"pr101")
    runtime_tree_sha256 = _write_runtime_custody_files(archive)
    runtime_manifest = runtime_dependency_manifest(archive.parent, archive.parent.parent)
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": None,
        "candidate_archive_path": archive.as_posix(),
        "candidate_archive_bytes": archive.stat().st_size,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": (
            archive.parent / "runtime_consumption_proof.json"
        ).as_posix(),
    }
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(ranked, k=None)
    cmd = tool._build_dispatch_cmd(
        loaded[0],
        provider="lightning",
        lane_script="scripts/ignored.sh",
        label_prefix="batch",
        estimated_cost=0.11,
        max_dph=0.50,
    )

    assert loaded == [candidate]
    assert "--archive" in cmd
    assert archive.as_posix() in cmd


def test_parallel_dispatch_rejects_ready_candidate_without_exact_archive_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    del candidate["archive_sha256"]
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "archive_custody:archive_sha256_missing_or_invalid" in message


def test_parallel_dispatch_rejects_ready_candidate_without_runtime_tree_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    del candidate["runtime_tree_sha256"]
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "runtime_custody:runtime_tree_sha256_missing_or_invalid" in message


def test_parallel_dispatch_rejects_predicted_codecop_score_fields_even_with_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_semantics="cpu_substrate_predicted_band",
        predicted_score=0.207,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_evidence_semantics:cpu_substrate_predicted_band" in message
    assert "predicted_score_field_present:predicted_score" in message


def test_parallel_dispatch_rejects_spoofed_mps_ready_candidate(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_grade="[MPS-research-signal]",
        evidence_semantics="mps_proxy_curve_shape_only",
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_evidence_semantics:" in message
    assert "mps" in message


def test_parallel_dispatch_rejects_spoofed_cpu_build_ready_candidate(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_grade="[CPU-build]",
        evidence_semantics="contest_cuda_exact_eval_positive",
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_evidence_semantics:" in message
    assert "cpu-build" in message.lower()


def test_parallel_dispatch_rejects_spoofed_evidence_marker_cpu_only_candidate(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_semantics="contest_cuda_exact_eval_positive",
        evidence_marker="[CPU-only]",
        promotion_eligible=True,
        rank_or_kill_eligible=True,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_evidence_semantics:" in message
    assert "cpu-only" in message.lower()


def test_parallel_dispatch_rejects_spoofed_source_text_local_only_candidate(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_semantics="contest_cuda_exact_eval_positive",
        source_text="local only CPU prep artifact; no contest CUDA replay yet",
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_evidence_semantics:" in message
    assert "local only" in message.lower()
    assert "cpu prep" in message.lower()


def test_parallel_dispatch_rejects_deferred_research_dispatch_verdict(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        evidence_semantics="contest_cuda_exact_eval_positive",
        contest_dispatch_verdict="DEFERRED-pending-research",
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "blocked_contest_dispatch_verdict:" in message
    assert "deferred-pending-research" in message.lower()


def test_parallel_dispatch_rejects_missing_evidence_semantics_with_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    del candidate["evidence_semantics"]
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "evidence_semantics_missing" in message


def test_parallel_dispatch_rejects_candidate_archive_bytes_above_floor_by_default(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive(tmp_path / "submission" / "candidate_archive.zip")
    runtime_tree_sha256 = _write_runtime_custody_files(archive)
    runtime_manifest = runtime_dependency_manifest(archive.parent, archive.parent.parent)
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": None,
        "candidate_archive_path": archive.as_posix(),
        "candidate_archive_bytes": archive.stat().st_size,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_content_tree_sha256": runtime_manifest["runtime_content_tree_sha256"],
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": (
            archive.parent / "runtime_consumption_proof.json"
        ).as_posix(),
    }
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(
            ranked,
            k=None,
            active_floor_archive_bytes=candidate["candidate_archive_bytes"] - 1,
        )
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "above_active_floor_archive_bytes" in message


def test_parallel_dispatch_rejects_mismatched_archive_byte_fields(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        expected_archive_size_bytes=123,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "archive_custody:archive_bytes_field_mismatch" in message


def test_parallel_dispatch_rejects_zip_slip_archive_member(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive_with_members(tmp_path / "zip_slip.zip", ["../escape"])
    candidate = _ready_custody_candidate(
        tmp_path,
        archive_path=archive.as_posix(),
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "archive_custody:archive_zip_unsafe_member:" in message
    assert "zip-slip archive member path" in message


def test_parallel_dispatch_rejects_duplicate_archive_members(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive_with_members(tmp_path / "duplicate.zip", ["x", "x"])
    candidate = _ready_custody_candidate(
        tmp_path,
        archive_path=archive.as_posix(),
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "archive_custody:archive_zip_duplicate_members" in message


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        ("data_descriptor", "archive_zipwire:x:data_descriptor_member_not_supported"),
        ("encrypted", "archive_zipwire:x:encrypted_member"),
        ("unsupported_method", "archive_zipwire:x:unsupported_zip_method:99"),
        ("local_crc", "archive_zipwire:x:local_central_crc32_mismatch"),
        ("local_size", "archive_zipwire:x:local_central_compressed_size_mismatch"),
        ("local_method", "archive_zipwire:x:local_central_compress_type_mismatch"),
    ],
)
def test_parallel_dispatch_rejects_zipwire_strict_blockers(
    tmp_path: Path,
    mutator: str,
    expected: str,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive(tmp_path / f"{mutator}.zip")
    _mutate_zip_header(archive, mutator)
    candidate = _ready_custody_candidate(
        tmp_path,
        archive_path=archive.as_posix(),
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert expected in message


def test_parallel_dispatch_rejects_directory_archive_member(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive_with_members(tmp_path / "directory.zip", ["dir/"])
    candidate = _ready_custody_candidate(
        tmp_path,
        archive_path=archive.as_posix(),
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "archive_zipwire:dir/:directory_member_not_supported_for_contest_packet" in message


def _mutate_zip_header(archive: Path, mutation: str) -> None:
    raw = bytearray(archive.read_bytes())
    central = raw.index(b"PK\x01\x02")
    if mutation == "data_descriptor":
        raw[6] |= 0x08
        raw[central + 8] |= 0x08
    elif mutation == "encrypted":
        raw[6] |= 0x01
        raw[central + 8] |= 0x01
    elif mutation == "unsupported_method":
        raw[8] = 99
        raw[central + 10] = 99
    elif mutation == "local_crc":
        raw[14] ^= 0x01
    elif mutation == "local_size":
        raw[18] += 1
    elif mutation == "local_method":
        raw[8] = 99
    else:  # pragma: no cover - defensive
        raise AssertionError(mutation)
    archive.write_bytes(raw)


def test_parallel_dispatch_rejects_production_only_target_for_contest_dispatch(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="openpilot_edge_prod_probe",
        target_modes=["openpilot_edge"],
        optimization_target="openpilot_edge",
        deployment_target="comma_ai_production",
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "non_contest_target_mode:" in message
    assert "parallel_dispatch_top_k only dispatches contest exact-eval archives" in message


def test_parallel_dispatch_requires_explicit_contest_target_metadata(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(tmp_path)
    candidate.pop("target_modes", None)
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "target_modes_missing" in message
    assert "requires explicit contest_exact_eval target metadata" in message


def test_parallel_dispatch_allows_dual_contest_and_production_target_with_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="dual_target_archive",
        target_modes=["contest_exact_eval", "openpilot"],
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(ranked, k=None)

    assert loaded == [candidate]


def test_parallel_dispatch_rejects_contest_generalized_without_exact_eval_marker(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="contest_generalized_native_rewrite",
        target_modes=["contest_generalized"],
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "contest_exact_eval_target_mode_missing:contest_generalized" in message
    assert "requires explicit contest_exact_eval target metadata" in message


def test_parallel_dispatch_rejects_edge_adaptive_profile_without_contest_marker(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="production_edge_adaptive_native_probe",
        target_modes=["production_edge_adaptive"],
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "non_contest_target_mode:production_edge_adaptive" in message


def test_parallel_dispatch_rejects_self_neural_edge_probe_without_bit_change_proof(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="self_compress_edge_learning_probe",
        target_modes=["contest_exact_eval", "openpilot"],
        score_affecting_payload_changed=False,
        charged_bits_changed=False,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "self_neural_edge_candidate_missing_charged_bits_changed_proof" in message


def test_parallel_dispatch_accepts_self_neural_edge_probe_with_sha_diff_proof(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="neural_compression_contest_probe",
        target_modes=["contest_exact_eval", "openpilot"],
        source_payload_sha256="0" * 64,
        candidate_payload_sha256="1" * 64,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(ranked, k=None)

    assert loaded == [candidate]


def test_parallel_dispatch_rejects_self_neural_edge_declared_no_op(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="neural_compression_noop_probe",
        target_modes=["contest_exact_eval"],
        source_archive_sha256="0" * 64,
        candidate_archive_sha256="1" * 64,
        no_op_payload=True,
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    try:
        tool._load_top_k(ranked, k=None)
    except tool.DispatchInputError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected DispatchInputError")

    assert "self_neural_edge_candidate_missing_charged_bits_changed_proof" in message


def test_feedback_loop_lightning_command_uses_stack_dispatcher_flags() -> None:
    tool = _load_tool("feedback_loop_sweep")

    cmd = tool._build_dispatch_cmd(
        _ready_lightning_candidate(),
        provider="lightning",
        lane_script="scripts/legacy_should_be_ignored_for_lightning.sh",
        label_prefix="cycle",
        max_dph=0.50,
        estimated_cost=0.11,
    )

    joined = " ".join(cmd)
    assert "tools/lightning_dispatch_pr106_stack.py" in joined
    assert "--lane pr106_latent_sidecar" in joined
    assert "--archive experiments/results/candidate42/archive.zip" in joined
    assert "--predicted-low 0.19" in joined
    assert "--predicted-high 0.24" in joined
    assert "--job-name cycle_candidate42" in joined
    assert "--lane-script" not in cmd
    assert "--predicted-band" not in cmd
