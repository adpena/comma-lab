from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

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
        "evidence_semantics": "contest_cuda_exact_eval_positive",
        "score_claim": False,
    }


def _write_ranked_input(tmp_path: Path, candidates: list[dict]) -> Path:
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps({"dispatch_ready": candidates}), encoding="utf-8")
    return ranked


def _write_archive(path: Path, payload: bytes = b"fixture") -> Path:
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    return path


def _ready_custody_candidate(tmp_path: Path, **overrides) -> dict:
    archive = _write_archive(tmp_path / "archive.zip")
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": archive.as_posix(),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
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
    assert "--lane-script" not in cmd
    assert "--predicted-band" not in cmd


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


def test_parallel_dispatch_accepts_candidate_archive_schema_with_exact_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive(tmp_path / "pr101_candidate.zip", b"pr101")
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": None,
        "candidate_archive_path": archive.as_posix(),
        "candidate_archive_bytes": archive.stat().st_size,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
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


def test_parallel_dispatch_rejects_candidate_archive_bytes_above_floor_by_default(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    archive = _write_archive(tmp_path / "candidate_archive.zip")
    candidate = {
        **_ready_lightning_candidate(),
        "archive_path": None,
        "candidate_archive_path": archive.as_posix(),
        "candidate_archive_bytes": archive.stat().st_size,
        "candidate_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
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


def test_parallel_dispatch_rejects_production_only_target_for_contest_dispatch(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="openpilot_edge_prod_probe",
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


def test_parallel_dispatch_allows_contest_generalized_profile_with_custody(
    tmp_path: Path,
) -> None:
    tool = _load_tool("parallel_dispatch_top_k")
    candidate = _ready_custody_candidate(
        tmp_path,
        candidate_id="contest_generalized_native_rewrite",
        target_modes=["contest_generalized"],
    )
    ranked = _write_ranked_input(tmp_path, [candidate])

    loaded = tool._load_top_k(ranked, k=None)

    assert loaded == [candidate]


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
