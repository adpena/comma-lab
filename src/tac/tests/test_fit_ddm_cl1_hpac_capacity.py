from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools/fit_ddm_cl1_hpac_capacity.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("fit_ddm_cl1_hpac_capacity", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(label: str | bytes) -> str:
    raw = label if isinstance(label, bytes) else label.encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_identity(tool, trainer, rate_lambda: float) -> dict:
    config = dict(trainer.PREREGISTERED_CONFIG)
    config["rate_lambda"] = rate_lambda
    ema_policy = {
        "equation_id": "ema_decay_run_geometry_v1",
        "decay": 0.999,
        "warmup": True,
    }
    local_causal = trainer._local_causal_sha256()
    source_identity = {
        "trainer_sha256": tool._sha256_file(tool.TRAINER_PATH),
        "intake_source_sha256": trainer.EXPECTED_INTAKE_SHA256,
        "local_causal_source_sha256": local_causal,
    }
    schedule = {key: value for key, value in config.items() if key != "rate_lambda"}
    return {
        "schema": "ddm_cl1_hpac_capacity_run_identity.v1",
        "launch_git_sha": "1" * 40,
        "trainer": str(tool.TRAINER_PATH),
        "trainer_sha256": source_identity["trainer_sha256"],
        "training_config": config,
        "seed_schedule_identity_sha256": tool._canonical_json_sha256(
            {"schedule_config": schedule, "ema_policy": ema_policy}
        ),
        "trainer_source_identity_sha256": tool._canonical_json_sha256(source_identity),
        "cache_path": str(tool.CANONICAL_CACHE_PATH),
        "cache_sha256": tool.EXPECTED_CACHE_SHA256,
        "init_path": str(tool.CANONICAL_INIT_PATH),
        "init_sha256": tool.EXPECTED_INIT_SHA256,
        "intake_code_root": str(tool.INTAKE_CODE_ROOT),
        "intake_source_sha256": trainer.EXPECTED_INTAKE_SHA256,
        "local_causal_source_sha256": local_causal,
        "ema_policy": ema_policy,
        "software": {"python": "test", "torch": "test", "numpy": "test"},
        "hardware": {
            "system": "Darwin",
            "mps_built": True,
            "mps_available": True,
        },
    }


def _checkpoint(tool, trainer, rate_lambda: float) -> dict:
    identity = _run_identity(tool, trainer, rate_lambda)
    value = torch.tensor([rate_lambda], dtype=torch.float32)
    payload = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "epoch": 60,
        "phase": "discrete_qat",
        "qat_start": 31,
        "state_dict": {"weight": value.clone()},
        "deployment_weights": "ema_shadow",
        "live_state_dict": {"weight": value.clone()},
        "ema": {
            "decay": 0.999,
            "warmup": True,
            "num_updates": 4500,
            "shadow": {"weight": value.clone()},
        },
        "ema_policy": identity["ema_policy"],
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 60},
        "rng": {"python": (3, (1, 2, 3), None)},
        "best": {"epoch": 60},
        "history": [{"epoch": 60}],
        "run_identity": identity,
        "run_identity_sha256": tool._canonical_json_sha256(identity),
        "resume_lineage": [],
    }
    payload["causal_state_sha256"] = trainer._causal_state_sha256(payload)
    return payload


def _safe_receipt(
    argv: list[str],
    *,
    elapsed: float = 10.0,
    exit_code: int = 0,
    observed_peak: bool = False,
) -> dict:
    child_pid = 1_000 + int(_sha("\0".join(argv))[:6], 16)
    return {
        "schema": "safe_run_status_receipt.v1",
        "generated_utc": "2026-08-09T12:00:10Z",
        "start_utc": "2026-08-09T12:00:00Z",
        "label": "test-artifact-run",
        "status": "ok",
        "exit": exit_code,
        "elapsed_s": elapsed,
        "argv": argv,
        "child_pid": child_pid,
        "pgid": child_pid,
        "rss_limit_mib": 4096,
        "timeout_s": 1800,
        "peak_rss_observed": observed_peak,
        "peak_rss_mib": 128.0 if observed_peak else 0.0,
        "kill_action": None,
    }


def _record(tool, path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": tool._sha256_file(path),
    }


def _attestation(
    tool,
    operation: str,
    child_argv: list[str],
    *,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> dict:
    return {
        "schema": tool.ATTESTATION_SCHEMA,
        "operation": operation,
        "runner_sha256": tool._sha256_file(Path(tool.__file__).resolve()),
        "runtime_identity": tool._artifact_runtime_identity(),
        "causal_source_sha256": tool._verified_causal_sources(operation),
        "child_argv": child_argv,
        "inputs": {name: _record(tool, path) for name, path in inputs.items()},
        "outputs": {name: _record(tool, path) for name, path in outputs.items()},
    }


def _row(
    tool,
    trainer,
    root: Path,
    rung_id: str,
    model_bytes: int,
    ideal_bytes: int,
    range_bytes: int,
) -> dict:
    rate_lambda = tool.RUNG_LAMBDAS[rung_id]
    rung = root / rung_id
    is_resume_control = rung_id == "lambda_1p0_resume_control"
    if is_resume_control:
        interrupted_save = rung / "interrupted" / "best_ema.pt"
        interrupted_out = rung / "interrupted" / "trainer.json"
        resume_parent = tool._epoch_one_checkpoint_path(interrupted_save)
        resume_parent.parent.mkdir(parents=True, exist_ok=True)
        parent_payload = _checkpoint(tool, trainer, rate_lambda)
        parent_payload["epoch"] = 1
        parent_payload["phase"] = "continuous"
        parent_payload["causal_state_sha256"] = trainer._causal_state_sha256(parent_payload)
        torch.save(parent_payload, resume_parent)
        final_save = rung / "resumed" / "best_ema.pt"
        final_out = rung / "resumed" / "trainer.json"
        preserved_parent = (
            final_save.with_name(final_save.stem + ".checkpoints")
            / "resume_parents"
            / f"{tool._sha256_file(resume_parent)}.pt"
        )
        preserved_parent.parent.mkdir(parents=True, exist_ok=True)
        preserved_parent.write_bytes(resume_parent.read_bytes())
        lineage = [
            {
                "source_path": str(resume_parent.resolve()),
                "preserved_path": str(preserved_parent.resolve()),
                "bytes": resume_parent.stat().st_size,
                "sha256": tool._sha256_file(resume_parent),
                "epoch": 1,
            }
        ]
        interrupted_receipt = rung / "run" / "interrupted.safe_run.json"
        resumed_receipt = rung / "run" / "resumed.safe_run.json"
        _write_json(
            interrupted_receipt,
            _safe_receipt(
                tool._expected_training_argv(
                    rate_lambda=rate_lambda,
                    save=interrupted_save,
                    out=interrupted_out,
                ),
                elapsed=60.0,
                exit_code=-9,
                observed_peak=True,
            ),
        )
        final_training_argv = tool._expected_training_argv(
            rate_lambda=rate_lambda,
            save=final_save,
            out=final_out,
            resume_from=resume_parent,
        )
        _write_json(resumed_receipt, _safe_receipt(final_training_argv))
        training_receipts = [interrupted_receipt, resumed_receipt]
    else:
        final_save = rung / "training" / "best_ema.pt"
        final_out = rung / "training" / "trainer.json"
        lineage = []
        training_receipt = rung / "run" / "training.safe_run.json"
        final_training_argv = tool._expected_training_argv(
            rate_lambda=rate_lambda,
            save=final_save,
            out=final_out,
        )
        _write_json(training_receipt, _safe_receipt(final_training_argv))
        training_receipts = [training_receipt]
    checkpoint_path = tool._terminal_checkpoint_path(final_save)
    packed_model_path = rung / "model.bin.xz"
    range_token_path = rung / "tokens.range.bin"
    decoded_raw_path = rung / "tokens.raw.u8"
    pack_report_path = rung / "pack.json"
    encode_report_path = rung / "encode.json"
    decode_report_path = rung / "decode.json"
    pack_attestation_path = rung / "pack.attestation.json"
    encode_attestation_path = rung / "encode.attestation.json"
    decode_attestation_path = rung / "decode.attestation.json"
    pack_receipt_path = rung / "pack.safe_run.json"
    encode_receipt_path = rung / "encode.safe_run.json"
    decode_receipt_path = rung / "decode.safe_run.json"
    rung.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = _checkpoint(tool, trainer, rate_lambda)
    checkpoint_payload["resume_lineage"] = lineage
    checkpoint_payload["causal_state_sha256"] = trainer._causal_state_sha256(checkpoint_payload)
    torch.save(checkpoint_payload, checkpoint_path)
    trainer_result = {
        "schema": "ddm_cl1_hpac_capacity_trainer_result.v1",
        "run_identity": checkpoint_payload["run_identity"],
        "score_claim": False,
    }
    final_save.parent.mkdir(parents=True, exist_ok=True)
    # This fixture mirrors the trainer's real on-disk layout, so it mirrors the
    # trainer's write ORDER too: `trainer.json` (cheap, irreplaceable) before
    # `best_ema.pt` (large, rebuildable). See the matching cure in
    # `tools/train_ddm_cl1_hpac_capacity.py` (ddm_pl1 incident A2 / ddm_ql2).
    _write_json(final_out, trainer_result)
    torch.save(
        {"state_dict": checkpoint_payload["state_dict"], "result": trainer_result},
        final_save,
    )
    selected_record = _record(tool, checkpoint_path)
    _write_json(
        final_save.with_suffix(".artifacts.json"),
        {
            "schema": trainer.MANIFEST_SCHEMA,
            "score_claim": False,
            "run_identity": checkpoint_payload["run_identity"],
            "argv": final_training_argv,
            "artifacts": [selected_record],
        },
    )
    packed_model_path.write_bytes(b"m" * model_bytes)
    range_token_path.write_bytes(b"t" * range_bytes)
    raw = b"r" * tool.PIXELS
    decoded_raw_path.write_bytes(raw)
    _write_json(
        pack_report_path,
        {
            "raw_model_bytes": model_bytes + 100,
            "compressed_model_bytes": model_bytes,
            "verified_exact": True,
            "max_logit_diff": 0.0,
        },
    )
    logit_hash = _sha(f"logits:{rung_id}")
    _write_json(
        encode_report_path,
        {
            "frames": 600,
            "token_bytes": range_bytes,
            "token_bpp": range_bytes * 8.0 / tool.PIXELS,
            "ideal_bpp": (ideal_bytes - 0.25) * 8.0 / tool.PIXELS,
            "logit_hash_encode": logit_hash,
        },
    )
    _write_json(
        decode_report_path,
        {
            "frames": 600,
            "verified_exact": True,
            "raw_token_sha256": tool.EXPECTED_RAW_TOKEN_SHA256,
            "logit_hash_decode": logit_hash,
        },
    )
    _write_json(
        pack_attestation_path,
        _attestation(
            tool,
            "pack",
            tool._expected_pack_argv(checkpoint_path, packed_model_path, pack_report_path),
            inputs={"checkpoint": checkpoint_path},
            outputs={
                "packed_model": packed_model_path,
                "report": pack_report_path,
            },
        ),
    )
    _write_json(
        encode_attestation_path,
        _attestation(
            tool,
            "encode",
            tool._expected_encode_argv(checkpoint_path, range_token_path, encode_report_path),
            inputs={
                "checkpoint": checkpoint_path,
                "cache": tool.CANONICAL_CACHE_PATH,
            },
            outputs={
                "range_tokens": range_token_path,
                "report": encode_report_path,
            },
        ),
    )
    _write_json(
        decode_attestation_path,
        _attestation(
            tool,
            "decode",
            tool._expected_decode_argv(
                checkpoint_path,
                range_token_path,
                decoded_raw_path,
                decode_report_path,
            ),
            inputs={
                "checkpoint": checkpoint_path,
                "cache": tool.CANONICAL_CACHE_PATH,
                "range_tokens": range_token_path,
            },
            outputs={
                "decoded_raw_tokens": decoded_raw_path,
                "report": decode_report_path,
            },
        ),
    )
    _write_json(
        pack_receipt_path,
        _safe_receipt(
            tool._expected_runner_argv(
                "pack",
                checkpoint=checkpoint_path,
                blob=packed_model_path,
                report=pack_report_path,
                attestation=pack_attestation_path,
            )
        ),
    )
    _write_json(
        encode_receipt_path,
        _safe_receipt(
            tool._expected_runner_argv(
                "encode",
                checkpoint=checkpoint_path,
                tokens=range_token_path,
                report=encode_report_path,
                attestation=encode_attestation_path,
            )
        ),
    )
    _write_json(
        decode_receipt_path,
        _safe_receipt(
            tool._expected_runner_argv(
                "decode",
                checkpoint=checkpoint_path,
                tokens=range_token_path,
                raw=decoded_raw_path,
                report=decode_report_path,
                attestation=decode_attestation_path,
            ),
            elapsed=1000.0,
        ),
    )
    return {
        "rung_id": rung_id,
        "training_receipt_paths": [str(path) for path in training_receipts],
        "selected_checkpoint_path": str(checkpoint_path),
        "packed_model_path": str(packed_model_path),
        "range_token_path": str(range_token_path),
        "decoded_raw_token_path": str(decoded_raw_path),
        "pack_report_path": str(pack_report_path),
        "encode_report_path": str(encode_report_path),
        "decode_report_path": str(decode_report_path),
        "pack_attestation_path": str(pack_attestation_path),
        "encode_attestation_path": str(encode_attestation_path),
        "decode_attestation_path": str(decode_attestation_path),
        "pack_receipt_path": str(pack_receipt_path),
        "encode_receipt_path": str(encode_receipt_path),
        "decode_receipt_path": str(decode_receipt_path),
    }


def _rows(
    tool,
    trainer,
    root: Path,
    *,
    half_model: int = 16_164,
    half_range: int = 114_800,
    quarter_range: int = 114_000,
    model_offset: int = 0,
    twin_model_offset: int = 0,
) -> list[dict]:
    return [
        _row(
            tool,
            trainer,
            root,
            "lambda_1p0_resume_control",
            15_164 + model_offset,
            114_852,
            116_980,
        ),
        _row(
            tool,
            trainer,
            root,
            "lambda_1p0_uninterrupted_twin",
            15_164 + model_offset + twin_model_offset,
            114_852,
            116_980,
        ),
        _row(
            tool,
            trainer,
            root,
            "lambda_0p5",
            half_model + model_offset,
            112_800,
            half_range,
        ),
        _row(
            tool,
            trainer,
            root,
            "lambda_0p25",
            17_164 + model_offset,
            112_100,
            quarter_range,
        ),
    ]


@pytest.fixture
def artifact_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = _load_tool()
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    root = tmp_path / "custody"
    intake = tmp_path / "intake"
    intake.mkdir()
    source_hashes = {}
    for name in tool.EXPECTED_CAUSAL_SOURCE_SHA256:
        source = intake / name
        source.write_bytes(f"test source:{name}".encode())
        source_hashes[name] = _sha(source.read_bytes())
    cache = tmp_path / "inputs" / "cache.pt"
    init = tmp_path / "inputs" / "init.pt"
    cache.parent.mkdir()
    cache.write_bytes(b"canonical cache fixture")
    init.write_bytes(b"canonical init fixture")
    monkeypatch.setattr(tool, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(tool, "INTAKE_CODE_ROOT", intake)
    monkeypatch.setattr(tool, "EXPECTED_CAUSAL_SOURCE_SHA256", source_hashes)
    monkeypatch.setattr(tool, "CANONICAL_CACHE_PATH", cache)
    monkeypatch.setattr(tool, "CANONICAL_INIT_PATH", init)
    monkeypatch.setattr(tool, "EXPECTED_CACHE_SHA256", _sha(cache.read_bytes()))
    monkeypatch.setattr(tool, "EXPECTED_INIT_SHA256", _sha(init.read_bytes()))
    monkeypatch.setattr(tool, "PIXELS", 64)
    monkeypatch.setattr(tool, "EXPECTED_RAW_TOKEN_SHA256", _sha(b"r" * 64))
    return tool, tool._load_trainer_tool(), root


def test_final_fit_uses_unique_lambdas_and_brackets_only_secants(
    artifact_tool,
) -> None:
    tool, trainer, root = artifact_tool
    result = tool.fit({"schema": tool.INPUT_SCHEMA, "rows": _rows(tool, trainer, root)})
    assert result["range_fit"]["n_unique_lambdas"] == 3
    assert result["range_fit"]["degrees_freedom"] == 1
    assert result["verdict"]["knee_status"] == "BRACKETED_BY_ADJACENT_SECANTS"
    assert result["verdict"]["knee_lambda"] is None
    assert result["verdict"]["best_observed_lambda"] == 0.5
    assert result["verdict"]["knee_bracket_lambda"] == [0.25, 1.0]
    assert result["verdict"]["all_observed_intervals_pay"] is False
    assert result["control_repeat_floor"]["causal_state_equal"] is True
    assert result["aggregated_by_lambda"][0]["replicates"] == 2
    assert result["verdict"]["best_experimental_beats_pr130"] is True


def test_comparison_identity_removes_only_rate_lambda(artifact_tool) -> None:
    tool, trainer, _ = artifact_tool
    reference = _run_identity(tool, trainer, 1.0)
    treatment = _run_identity(tool, trainer, 0.5)
    assert tool._comparison_identity_sha256(reference) == (tool._comparison_identity_sha256(treatment))
    treatment["hardware"]["system"] = "DifferentHost"
    assert tool._comparison_identity_sha256(reference) != (tool._comparison_identity_sha256(treatment))


def test_paying_endpoint_is_unbracketed_not_a_knee(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    result = tool.fit(
        {
            "schema": tool.INPUT_SCHEMA,
            "rows": _rows(tool, trainer, root, quarter_range=113_400),
        }
    )
    assert result["verdict"]["all_observed_intervals_pay"] is True
    assert result["verdict"]["knee_status"] == "UNBRACKETED_LOWER_LAMBDA"
    assert result["verdict"]["knee_lambda"] is None
    assert result["verdict"]["best_observed_lambda"] == 0.25


def test_interim_shape_emits_secant_but_never_ols(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root, half_range=116_480)[:3]
    result = tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})
    assert result["measurement_shape"] == "INTERIM_EXACT_SECANT"
    assert result["range_fit"] is None
    assert result["adjacent_slopes"][0]["range_token_bytes_per_model_byte"] == -0.5
    assert result["verdict"]["knee_status"] == "REFERENCE_BOUNDARY"
    assert result["verdict"]["knee_lambda"] is None


def test_final_fit_refuses_off_order_lambda_quarter(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root, half_range=116_480)
    with pytest.raises(tool.CL1FitError, match="conditional fire order"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_refuses_missing_or_divergent_controls(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    missing = [rows[0], rows[2], rows[3]]
    with pytest.raises(tool.CL1FitError, match="both named controls"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": missing})
    rows = _rows(tool, trainer, root / "divergent", twin_model_offset=1)
    with pytest.raises(tool.CL1FitError, match="controls diverge"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_opens_artifacts_and_refuses_tampered_pack(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    packed = Path(rows[2]["packed_model_path"])
    packed.write_bytes(b"x" * packed.stat().st_size)
    with pytest.raises(tool.CL1FitError, match="attestation output bytes changed"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_refuses_same_length_range_stream_substitution(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    stream = Path(rows[2]["range_token_path"])
    stream.write_bytes(b"y" * stream.stat().st_size)
    with pytest.raises(tool.CL1FitError, match="attestation output bytes changed"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_artifact_runner_executes_child_and_attests_exact_bytes(artifact_tool, monkeypatch: pytest.MonkeyPatch) -> None:
    tool, _, root = artifact_tool
    monkeypatch.setattr(tool, "assert_governed_admission", lambda *_args: True)
    monkeypatch.setenv("TAC_ADMISSION_ENFORCE", "1")
    tool.PYTHON_PATH = Path(sys.executable)
    packer = tool.INTAKE_CODE_ROOT / "pack_hpac_self_compress.py"
    packer.write_text(
        """import json
import sys
from pathlib import Path

def value(flag):
    return Path(sys.argv[sys.argv.index(flag) + 1])

blob = value('--blob')
report = value('--report')
blob.parent.mkdir(parents=True, exist_ok=True)
blob.write_bytes(b'actual-child-packed-bytes')
report.write_text(json.dumps({
    'raw_model_bytes': 99,
    'compressed_model_bytes': blob.stat().st_size,
    'verified_exact': True,
    'max_logit_diff': 0.0,
}) + '\\n')
""",
        encoding="utf-8",
    )
    tool.EXPECTED_CAUSAL_SOURCE_SHA256["pack_hpac_self_compress.py"] = _sha(packer.read_bytes())
    checkpoint = root / "runner" / "terminal.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint fixture")
    blob = root / "runner" / "model.bin.xz"
    report = root / "runner" / "pack.json"
    attestation = root / "runner" / "pack.attestation.json"
    args = SimpleNamespace(
        command="pack",
        checkpoint=checkpoint,
        blob=blob,
        report=report,
        attestation=attestation,
    )
    payload = tool._run_artifact_operation(args)
    assert payload["outputs"]["packed_model"]["sha256"] == _sha(blob.read_bytes())
    tool._verify_attestation(
        attestation,
        operation="pack",
        child_argv=tool._expected_pack_argv(checkpoint, blob, report),
        inputs={"checkpoint": checkpoint},
        outputs={"packed_model": blob, "report": report},
    )
    blob.write_bytes(b"z" * blob.stat().st_size)
    with pytest.raises(tool.CL1FitError, match="attestation output bytes changed"):
        tool._verify_attestation(
            attestation,
            operation="pack",
            child_argv=tool._expected_pack_argv(checkpoint, blob, report),
            inputs={"checkpoint": checkpoint},
            outputs={"packed_model": blob, "report": report},
        )


def test_fit_refuses_impossible_range_smaller_than_ideal(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    token_path = Path(rows[2]["range_token_path"])
    token_path.write_bytes(b"x")
    report_path = Path(rows[2]["encode_report_path"])
    report = json.loads(report_path.read_text())
    report["token_bytes"] = 1
    report["token_bpp"] = 8.0 / tool.PIXELS
    _write_json(report_path, report)
    with pytest.raises(tool.CL1FitError, match="smaller than its own ideal"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_refuses_changed_checkpoint_cache_even_with_rehashed_payload(
    artifact_tool,
) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    checkpoint_path = Path(rows[2]["selected_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["run_identity"]["cache_sha256"] = "a" * 64
    checkpoint["run_identity_sha256"] = tool._canonical_json_sha256(checkpoint["run_identity"])
    checkpoint["causal_state_sha256"] = trainer._causal_state_sha256(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(tool.CL1FitError, match="pinned DALI cache"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_refuses_wrong_receipt_argv(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    receipt_path = Path(rows[2]["encode_receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["argv"][receipt["argv"].index("--checkpoint") + 1] = "/wrong.pt"
    _write_json(receipt_path, receipt)
    with pytest.raises(tool.CL1FitError, match="argv differs"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_nonmonotone_model_bytes_emit_scoped_nonmonotone_verdict(
    artifact_tool,
) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root, half_model=15_000)
    result = tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows[:3]})
    assert result["range_fit"] is None
    assert result["verdict"]["capacity_order_status"] == ("NON_MONOTONE_MODEL_CAPACITY")
    assert result["verdict"]["knee_status"] == "NON_MONOTONE_MODEL_CAPACITY"
    with pytest.raises(tool.CL1FitError, match="conditional fire order"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_fit_refuses_nonterminal_selected_checkpoint(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    checkpoint_path = Path(rows[2]["selected_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["epoch"] = 1
    checkpoint["phase"] = "continuous"
    checkpoint["causal_state_sha256"] = trainer._causal_state_sha256(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(tool.CL1FitError, match="terminal epoch-60"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_resume_control_requires_interruption_receipt_and_parent_lineage(
    artifact_tool,
) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root)
    missing_receipt = [dict(row) for row in rows]
    missing_receipt[0]["training_receipt_paths"] = missing_receipt[0]["training_receipt_paths"][1:]
    with pytest.raises(tool.CL1FitError, match="2 ordered training receipt"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": missing_receipt})

    checkpoint_path = Path(rows[0]["selected_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["resume_lineage"] = []
    checkpoint["causal_state_sha256"] = trainer._causal_state_sha256(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(tool.CL1FitError, match="lineage does not bind"):
        tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})


def test_exact_minus_one_does_not_pass_break_even(artifact_tool) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root, half_range=115_980)[:3]
    result = tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})
    assert result["adjacent_slopes"][0]["range_token_bytes_per_model_byte"] == -1.0
    assert result["adjacent_slopes"][0]["real_break_even_pass"] is False
    assert result["verdict"]["knee_status"] == "REFERENCE_BOUNDARY"


def test_pr130_reference_is_immutable_and_can_defeat_experiment(
    artifact_tool,
) -> None:
    tool, trainer, root = artifact_tool
    rows = _rows(tool, trainer, root, model_offset=5_000)
    result = tool.fit({"schema": tool.INPUT_SCHEMA, "rows": rows})
    assert result["verdict"]["best_experimental_beats_pr130"] is False
    assert result["verdict"]["selected_section_candidate"] == (tool.PR130_REFERENCE["name"])
    result["immutable_pr130_reference"]["range_joint_bytes"] = 1
    assert tool.PR130_REFERENCE["range_joint_bytes"] == 132_144
    with pytest.raises(TypeError):
        tool.PR130_REFERENCE["range_joint_bytes"] = 1
