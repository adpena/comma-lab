# SPDX-License-Identifier: MIT
"""Static hardening checks for experiments/modal_train_lane.py."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SOURCE = REPO_ROOT / "experiments" / "modal_train_lane.py"


def _load_timeout_contract():
    tree = ast.parse(SOURCE.read_text())
    selected = []
    names = {
        "MODAL_STATIC_MAX_SECONDS",
        "MODAL_FINALIZATION_GRACE_SECONDS",
    }
    for node in tree.body:
        is_contract_constant = isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        is_contract_function = (
            isinstance(node, ast.FunctionDef)
            and node.name == "_validated_modal_timeouts"
        )
        if is_contract_constant or is_contract_function:
            selected.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_validated_modal_timeouts"]


def _load_head_contract():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_expected_mounted_head"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validate_expected_mounted_head"]


def _load_cost_contract():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validated_expected_cost_usd"
    )
    namespace: dict[str, object] = {"V9_MAX_PLAN_COST_USD": 5.0}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validated_expected_cost_usd"]


def _load_provider_cost_contract():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validated_provider_cost_ceiling"
    )
    namespace: dict[str, object] = {
        "MODAL_GPU_USD_PER_HOUR_CEILINGS": {
            "T4": 0.75,
            "A10G": 1.50,
            "A100": 4.00,
            cloud.EXACT_H100_GPU: cloud.H100_GPU_USD_PER_HOUR,
        },
        "V9_CPU_CORES": cloud.CPU_CORES,
        "V9_CPU_USD_PER_CORE_SECOND": cloud.CPU_USD_PER_CORE_SECOND,
        "V9_MEMORY_GIB": cloud.MEMORY_GIB,
        "V9_MEMORY_USD_PER_GIB_SECOND": cloud.MEMORY_USD_PER_GIB_SECOND,
        "V9_CPU_PREFLIGHT_SECONDS": cloud.CPU_PREFLIGHT_SECONDS,
        "V9_IMAGE_STAGING_ALLOWANCE_USD": cloud.BUDGETED_IMAGE_STAGING_ALLOWANCE_USD,
        "V9_MAX_PLAN_COST_USD": cloud.MAX_PLAN_COST_USD,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validated_provider_cost_ceiling"], cloud


def _load_gpu_request_contract():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validated_modal_gpu_request"
    )
    namespace: dict[str, object] = {"V9_EXACT_H100_GPU": cloud.EXACT_H100_GPU}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validated_modal_gpu_request"]


def _load_remote_dispatch_stage_contract():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    wanted = {"_derive_cpu_preflight_receipt", "_validate_remote_dispatch_stage"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace: dict[str, object] = {"V9_EXACT_H100_GPU": cloud.EXACT_H100_GPU}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return (
        namespace["_derive_cpu_preflight_receipt"],
        namespace["_validate_remote_dispatch_stage"],
    )


def _load_env_parser():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_env_overrides"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_parse_env_overrides"]


def _load_v9_lane_classifier():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_classify_v9_lane_script"
    )
    namespace: dict[str, object] = {
        "Path": Path,
        "V9_REMOTE_DRIVER": cloud.REMOTE_DRIVER,
        "V9_EXACT_H100_GPU": cloud.EXACT_H100_GPU,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_classify_v9_lane_script"], cloud


def _load_image_rebuild_guard():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_refuse_forced_modal_image_rebuild"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_refuse_forced_modal_image_rebuild"]


def _load_v9_contract():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "_classify_v9_lane_script",
        "_classify_v9_trainer_path",
        "_lane_script_invokes_v9_trainer",
        "_validate_v9_env_controls",
        "_validate_v9_direct_dispatch_policy",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace: dict[str, object] = {
        "V9_REMOTE_DRIVER": cloud.REMOTE_DRIVER,
        "V9_EXACT_H100_GPU": cloud.EXACT_H100_GPU,
        "V9_LANE_ID": cloud.LANE_ID,
        "V9_TRAINER": cloud.TRAINER,
        "V9_UNIQUE_ENV_KEYS": frozenset(
            {
                "WITNESS_GT_CACHE_SHA256",
                "WITNESS_SEGNET_SHA256",
                "WITNESS_POSENET_SHA256",
                "WITNESS_RESUME_SHA256",
                "WITNESS_STOP_AFTER_EPOCHS",
            }
        ),
        "V9_CUDA_ENV": cloud.CUDA_ENV,
        "V9_TYPED_EPOCH_HORIZON": cloud.TYPED_EPOCH_HORIZON,
        "V9_STOP_AFTER_EPOCHS": cloud.DEFAULT_STOP_AFTER_EPOCHS,
        "V9_DSL_CONFIGS": cloud.V9_DSL_CONFIGS,
        "V9_LABEL_RE": cloud.LABEL_RE,
        "V9_MAX_LABEL_LENGTH": cloud.MAX_LABEL_LENGTH,
        "V9_CHILD_TIMEOUT_SECONDS": cloud.CHILD_TIMEOUT_SECONDS,
        "V9_INVOCATION_TIMEOUT_SECONDS": cloud.INVOCATION_TIMEOUT_SECONDS,
        "V9_REVIEWED_SENTINELS": cloud.REVIEWED_SENTINELS,
        "V9_H100_GPU_USD_PER_HOUR": cloud.H100_GPU_USD_PER_HOUR,
        "V9_CPU_CORES": cloud.CPU_CORES,
        "V9_CPU_USD_PER_CORE_SECOND": cloud.CPU_USD_PER_CORE_SECOND,
        "V9_MEMORY_GIB": cloud.MEMORY_GIB,
        "V9_MEMORY_USD_PER_GIB_SECOND": cloud.MEMORY_USD_PER_GIB_SECOND,
        "V9_CPU_PREFLIGHT_SECONDS": cloud.CPU_PREFLIGHT_SECONDS,
        "V9_IMAGE_STAGING_ALLOWANCE_USD": cloud.BUDGETED_IMAGE_STAGING_ALLOWANCE_USD,
        "_validate_v9_modal_custody_path": cloud._validate_modal_custody_path,
        "Path": Path,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validate_v9_direct_dispatch_policy"], cloud


def _load_v9_remote_contract():
    from tac.deploy import witness_cloud_launcher as cloud

    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "_classify_v9_lane_script",
        "_validate_v9_env_controls",
        "_validate_v9_remote_h100_invocation",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace: dict[str, object] = {
        "V9_REMOTE_DRIVER": cloud.REMOTE_DRIVER,
        "V9_EXACT_H100_GPU": cloud.EXACT_H100_GPU,
        "V9_REVIEWED_SENTINELS": cloud.REVIEWED_SENTINELS,
        "V9_CUDA_ENV": cloud.CUDA_ENV,
        "V9_TYPED_EPOCH_HORIZON": cloud.TYPED_EPOCH_HORIZON,
        "V9_STOP_AFTER_EPOCHS": cloud.DEFAULT_STOP_AFTER_EPOCHS,
        "V9_DSL_CONFIGS": cloud.V9_DSL_CONFIGS,
        "V9_LABEL_RE": cloud.LABEL_RE,
        "V9_MAX_LABEL_LENGTH": cloud.MAX_LABEL_LENGTH,
        "V9_CHILD_TIMEOUT_SECONDS": cloud.CHILD_TIMEOUT_SECONDS,
        "V9_INVOCATION_TIMEOUT_SECONDS": cloud.INVOCATION_TIMEOUT_SECONDS,
        "MODAL_V9_DEADLINE_TOLERANCE_SECONDS": 5,
        "_validate_v9_modal_custody_path": cloud._validate_modal_custody_path,
        "Path": Path,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_validate_v9_remote_h100_invocation"], cloud


def test_modal_train_lane_keeps_wrapper_non_promotional_but_allows_inline_custody_eval() -> None:
    text = SOURCE.read_text()
    assert '"T1_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"SCPP_RUN_CONTEST_CUDA_AUTH_EVAL": "0"' in text
    assert '"RUN_CONTEST_EVAL": "0"' in text
    assert "refusing exact CUDA auth-eval from modal_train_lane.py" in text
    assert "Lane-local auth-eval subprocesses are allowed only when" in text
    assert "wrapper_score_claim" in text
    assert "inline_auth_eval_contract_required" in text
    assert "MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL" not in text
    assert "env.update({str(k): _modal_workspace_env(v)" in text
    assert 'readonly = "/workspace/pact"' in text
    assert 'if env.get("MODAL_ALLOW_EXACT_CUDA_AUTH_EVAL"' not in text


def test_modal_env_sh_also_fails_closed_for_sourced_lane_scripts() -> None:
    text = SOURCE.read_text()
    assert "export T1_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0" in text
    assert "export RUN_CONTEST_EVAL=0" in text
    assert "export DALI_DISABLE_NVML=" in text
    assert "export PYTORCH_CUDA_ALLOC_CONF=" in text


def test_modal_train_lane_threads_shared_dali_nvml_env_to_image_and_subprocess() -> None:
    text = SOURCE.read_text()
    assert "DALI_DISABLE_NVML_VALUE" in text
    assert "PYTORCH_CUDA_ALLOC_CONF_VALUE" in text
    assert '"DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE' in text
    assert '"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE' in text


def test_modal_env_sh_pins_cublas_workspace_for_gpu_determinism() -> None:
    text = SOURCE.read_text()
    expected = 'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"'
    assert expected in text
    assert '"CUBLAS_WORKSPACE_CONFIG": os.environ.get(' in text


def test_modal_train_lane_copies_dispatch_claim_ledger_to_remote_workspace() -> None:
    text = SOURCE.read_text()
    assert "claim_ledger_bytes: bytes" in text
    assert 'workspace / ".omx/state/active_lane_dispatch_claims.md"' in text
    assert "claim_path.parent.mkdir(parents=True, exist_ok=True)" in text
    assert "claim_path.write_bytes(claim_ledger_bytes)" in text
    assert '"T1_DISPATCH_CLAIMS_PATH": str(claim_path)' in text
    assert '"SCPP_DISPATCH_CLAIMS_PATH": str(claim_path)' in text
    assert "claims_path = repo_root / \".omx/state/active_lane_dispatch_claims.md\"" in text
    assert "claim_ledger_bytes = claims_path.read_bytes()" in text
    assert "fn.with_options(" in text
    assert ").spawn(" in text


def test_modal_train_lane_claims_before_spawn_and_records_lane_id() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]

    assert '"scripts/remote_lane_t1_balle_endtoend.sh": "t1_balle_128k_endtoend"' in text
    assert '"scripts/remote_lane_scpp_stage1.sh": "lane_scpp_stage1_smoke_anchor"' in text
    assert "def _ensure_dispatch_claim(" in text
    assert "tools/claim_lane_dispatch.py" in text
    assert "--status" in text
    assert "active_dispatching" in text
    assert "aborting before Modal GPU spawn" in text
    assert main_src.index("_ensure_dispatch_claim(") < main_src.index(
        "fn_call = fn.with_options("
    )
    assert '"lane_id": resolved_lane_id' in text
    assert "from tac.deploy.modal.auth_eval import function_call_id" in text
    assert "call_id = function_call_id(fn_call)" in text
    assert "fn_call.object_id" not in text


def test_modal_train_lane_timeout_contract_rejects_unsafe_values_and_adds_grace() -> None:
    validate = _load_timeout_contract()

    assert validate(1500 / 3600) == (1500, 1800)
    assert validate(14) == (14 * 3600 - 300, 14 * 3600)
    for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan):
        try:
            validate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe timeout accepted: {invalid!r}")


def test_expected_mounted_head_must_match_before_provider_calls() -> None:
    validate = _load_head_contract()
    head = "a" * 40
    validate(head, head)
    for invalid in ("", "A" * 40, "b" * 39, "b" * 40):
        try:
            validate(invalid, head)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe expected HEAD accepted: {invalid!r}")

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    custody = main_src.index("_validate_expected_mounted_head(")
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    spawn = main_src.index("fn_call = fn.with_options(", preflight)
    assert "expected_mounted_code_git_head: str = \"\"" in main_src
    assert "require_clean_head: bool = False" in main_src
    assert custody < preflight < spawn


def test_expected_cost_is_validated_and_recorded_before_provider_calls() -> None:
    validate = _load_cost_contract()
    assert validate(3.0, preflight_first=True) == 3.0
    assert validate(3.0, preflight_first=False) == 3.0
    for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan):
        try:
            validate(invalid, preflight_first=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe expected cost accepted: {invalid!r}")
    try:
        validate(100.0, preflight_first=True)
    except ValueError:
        pass
    else:
        raise AssertionError("preflight-first dispatch accepted cost above $5")

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    validation = main_src.index("validated_expected_cost_usd =")
    custody = main_src.index("_validate_expected_mounted_head(", validation)
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    spawn = main_src.index("fn_call = fn.with_options(", preflight)
    register = main_src.index("register_dispatched_call_id_fail_closed(", spawn)
    assert "expected_cost_usd: float = 0.0" in main_src
    assert validation < custody < preflight < spawn < register
    assert "expected_cost_usd=validated_expected_cost_usd" in main_src[register:]
    assert '"expected_cost_usd": validated_expected_cost_usd' in main_src
    assert "expected cost ceiling:" in main_src


def test_global_provider_cost_floor_covers_cpu_gpu_aliases_and_rejects_14h() -> None:
    validate, cloud = _load_provider_cost_contract()
    v9 = validate(
        requested_gpu=cloud.EXACT_H100_GPU,
        hard_timeout_seconds=cloud.INVOCATION_TIMEOUT_SECONDS,
        preflight_first=True,
        expected_cost_usd=3.296256,
    )
    assert v9 == pytest.approx(3.296256)
    for gpu in ("T4", "A10G", "A100", cloud.EXACT_H100_GPU):
        assert 0 < validate(
            requested_gpu=gpu,
            hard_timeout_seconds=1800,
            preflight_first=True,
            expected_cost_usd=5.0,
        ) <= 5.0
        try:
            validate(
                requested_gpu=gpu,
                hard_timeout_seconds=14 * 3600,
                preflight_first=True,
                expected_cost_usd=5.0,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"14h generic {gpu} provider plan was accepted")

    try:
        validate(
            requested_gpu=None,
            hard_timeout_seconds=14 * 3600,
            preflight_first=False,
            expected_cost_usd=5.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("14h 4-core/32GiB CPU provider plan was accepted")

    try:
        validate(
            requested_gpu=cloud.EXACT_H100_GPU,
            hard_timeout_seconds=cloud.INVOCATION_TIMEOUT_SECONDS,
            preflight_first=True,
            expected_cost_usd=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("underdeclared H100 provider plan was accepted")

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    global_cost = main_src.index("_validated_provider_cost_ceiling(")
    custody = main_src.index("_validate_expected_mounted_head(", global_cost)
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    assert global_cost < custody < preflight
    assert '"T4": 0.75' in text
    assert '"A10G": 1.50' in text
    assert '"A100": 4.00' in text
    assert "V9_EXACT_H100_GPU: V9_H100_GPU_USD_PER_HOUR" in text
    assert '"provider_resource_ceiling_usd": provider_resource_ceiling_usd' in text


def test_named_provider_endpoint_is_cpu_bounded_and_gpu_is_post_gate_dynamic() -> None:
    validate_gpu = _load_gpu_request_contract()
    assert validate_gpu("CPU", preflight_first=False) is None
    for gpu, expected in (
        ("T4", "T4"),
        ("A10G", "A10G"),
        ("A100-80GB", "A100"),
        ("H100", "H100"),
    ):
        assert validate_gpu(gpu, preflight_first=True) == expected
        try:
            validate_gpu(gpu, preflight_first=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"paid GPU bypassed CPU preflight: {gpu}")
    for unsafe_h100 in ("H100!", "H100-80GB"):
        with pytest.raises(ValueError):
            validate_gpu(unsafe_h100, preflight_first=True)

    tree = ast.parse(SOURCE.read_text())
    modal_functions: list[tuple[ast.FunctionDef, ast.Call]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.func.attr == "function"
            ):
                modal_functions.append((node, decorator))
    assert [node.name for node, _ in modal_functions] == ["run_lane_training_cpu"]
    endpoint, decorator = modal_functions[0]
    assert all(keyword.arg != "gpu" for keyword in decorator.keywords)
    timeout = next(keyword.value for keyword in decorator.keywords if keyword.arg == "timeout")
    assert isinstance(timeout, ast.Name)
    assert timeout.id == "MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS"
    assert any(argument.arg == "requested_gpu" for argument in endpoint.args.args)
    assert any(argument.arg == "dispatch_stage" for argument in endpoint.args.args)
    assert any(argument.arg == "cpu_preflight_receipt" for argument in endpoint.args.args)

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    policy = main_src.index("_validate_v9_direct_dispatch_policy(")
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    claim = main_src.index("_ensure_dispatch_claim(", preflight)
    dynamic_gpu = main_src.index('invocation_options["gpu"] = requested_modal_gpu')
    spawn = main_src.index("fn_call = fn.with_options(", dynamic_gpu)
    assert policy < preflight < claim < dynamic_gpu < spawn
    assert "Function.from_name(...).with_options(...)" in text
    assert "Provider IAM/budget controls" in text


def test_remote_dispatch_stage_requires_preflight_receipt_before_source_copy() -> None:
    derive_receipt, validate_stage = _load_remote_dispatch_stage_contract()
    nonce = "a" * 64
    head = "b" * 40
    lane_script = "scripts/generic.sh"
    label = "generic-safe"
    receipt = derive_receipt(
        dispatch_nonce=nonce,
        lane_script=lane_script,
        label=label,
        mounted_code_git_head=head,
    )
    validate_stage(
        dispatch_stage="cpu_preflight",
        dispatch_nonce=nonce,
        cpu_preflight_receipt="",
        requested_gpu="CPU",
        lane_script=lane_script,
        label=label,
        mounted_code_git_head=head,
        visible_gpu=False,
    )
    validate_stage(
        dispatch_stage="gpu_dispatch",
        dispatch_nonce=nonce,
        cpu_preflight_receipt=receipt,
        requested_gpu="H100",
        lane_script=lane_script,
        label=label,
        mounted_code_git_head=head,
        visible_gpu=True,
    )
    for changes in (
        {"dispatch_stage": ""},
        {"dispatch_stage": "unknown"},
        {"cpu_preflight_receipt": ""},
        {"requested_gpu": ""},
        {"visible_gpu": False},
    ):
        kwargs = {
            "dispatch_stage": "gpu_dispatch",
            "dispatch_nonce": nonce,
            "cpu_preflight_receipt": receipt,
            "requested_gpu": "H100!",
            "lane_script": lane_script,
            "label": label,
            "mounted_code_git_head": head,
            "visible_gpu": True,
            **changes,
        }
        try:
            validate_stage(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe remote dispatch stage accepted: {changes}")

    text = SOURCE.read_text()
    endpoint = text[
        text.index("def run_lane_training_cpu("):text.index("def _compact_stamp(")
    ]
    stage_guard = endpoint.index("_validate_remote_dispatch_stage(")
    source_copy = endpoint.index("result = _run_lane_inner(")
    assert stage_guard < source_copy
    assert "unknown or omitted dispatch_stage" in text


def test_results_volume_lookup_is_fail_closed_and_not_dynamically_mutated() -> None:
    text = SOURCE.read_text()
    assert (
        "results_vol = modal.Volume.from_name(RESULTS_VOL, create_if_missing=False)"
        in text
    )
    assert "create_if_missing=True" not in text
    main_src = text[text.index("@app.local_entrypoint()"):]
    dynamic_options = main_src[
        main_src.index("invocation_options = {") : main_src.index(
            "fn_call = fn.with_options(",
        )
    ]
    assert "volume" not in dynamic_options.lower()


def test_v9_direct_dispatch_policy_refuses_ten_hour_bypass(tmp_path: Path) -> None:
    validate, cloud = _load_v9_contract()
    digest = "a" * 64
    label = "v9-policy-test"
    overrides = dict(cloud.CUDA_ENV)
    overrides.update(
        {
            "WITNESS_GT_CACHE": f"/modal_results/assets/v9_cgauge/gt_{digest}.npz",
            "WITNESS_OUT_DIR": f"/modal_results/{label}/output",
            "WITNESS_EPOCHS": str(cloud.TYPED_EPOCH_HORIZON),
            "WITNESS_DSL_CONFIG": cloud.DEFAULT_DSL_CONFIG,
            "WITNESS_TORCH_COMPILE_MODE": cloud.DEFAULT_TORCH_COMPILE_MODE,
            "WITNESS_STOP_AFTER_EPOCHS": str(cloud.DEFAULT_STOP_AFTER_EPOCHS),
            "WITNESS_CHILD_TIMEOUT_SECONDS": str(cloud.CHILD_TIMEOUT_SECONDS),
            "WITNESS_NUM_PAIRS": "600",
            "WITNESS_RESUME_FROM": "",
            "WITNESS_RESUME_SHA256": "",
            "WITNESS_GT_CACHE_SHA256": digest,
            "WITNESS_SEGNET_SHA256": digest,
            "WITNESS_POSENET_SHA256": digest,
        }
    )
    kwargs = {
        "lane_script": cloud.REMOTE_DRIVER,
        "lane_id": cloud.LANE_ID,
        "label": label,
        "gpu": cloud.EXACT_H100_GPU,
        "requested_gpu": cloud.EXACT_H100_GPU,
        "mounted_code_git_branch": "main",
        "preflight_first": True,
        "require_clean_head": True,
        "max_seconds": cloud.CHILD_TIMEOUT_SECONDS,
        "hard_timeout_seconds": cloud.INVOCATION_TIMEOUT_SECONDS,
        "expected_cost_usd": 3.5,
        "sentinel_files": ",".join(cloud.REVIEWED_SENTINELS),
        "overrides": overrides,
        "trainer_module_path": cloud.TRAINER,
        "repo_root": REPO_ROOT,
    }
    validate(**kwargs)
    resume_path = f"/modal_results/{label}/checkpoints/stage-1.pt"
    resumed_overrides = {
        **overrides,
        "WITNESS_RESUME_FROM": resume_path,
        "WITNESS_RESUME_SHA256": digest,
    }
    validate(**{**kwargs, "overrides": resumed_overrides})
    for changes in (
        {"lane_script": f"scripts/../{cloud.REMOTE_DRIVER}"},
        {"max_seconds": 10 * 3600 - 300, "hard_timeout_seconds": 10 * 3600},
        {"gpu": "A100"},
        {"gpu": "H100!"},
        {"requested_gpu": "H100!"},
        {"mounted_code_git_branch": "feature"},
        {"preflight_first": False},
        {"require_clean_head": False},
        {"sentinel_files": cloud.REMOTE_DRIVER},
        {"expected_cost_usd": 1.0},
        {"overrides": {k: v for k, v in overrides.items() if k != "WITNESS_NUM_PAIRS"}},
        {"overrides": {**overrides, "WITNESS_EPOCHS": "10"}},
        {"overrides": {**overrides, "WITNESS_RESUME_SHA256": digest}},
        {
            "overrides": {
                **overrides,
                "WITNESS_RESUME_FROM": resume_path,
                "WITNESS_RESUME_SHA256": "",
            }
        },
    ):
        candidate = {**kwargs, **changes}
        try:
            validate(**candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe V9 direct dispatch accepted: {changes}")

    for invalid_label in ("foo/bar", "a" * (cloud.MAX_LABEL_LENGTH + 1)):
        invalid_overrides = {
            **overrides,
            "WITNESS_OUT_DIR": f"/modal_results/{invalid_label}/output",
        }
        try:
            validate(
                **{
                    **kwargs,
                    "label": invalid_label,
                    "overrides": invalid_overrides,
                }
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe V9 label accepted: {invalid_label!r}")

    wrapper = tmp_path / "scripts" / "generic_wrapper.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "T=experiments/train_levelset_witness_realized_through_R\n"
        'python "${T}_torch.py"\n'
    )
    for disguised_v9 in (
        {
            "lane_script": "scripts/copied_v9_driver.sh",
            "lane_id": "generic-lane",
            "trainer_module_path": "experiments/generic_trainer.py",
        },
        {
            "lane_script": "scripts/remote_lane_scpp_stage1.sh",
            "lane_id": "generic-lane",
            "trainer_module_path": cloud.TRAINER,
            "overrides": {},
        },
        {
            "lane_script": wrapper.relative_to(tmp_path).as_posix(),
            "lane_id": "generic-lane",
            "trainer_module_path": "experiments/generic_trainer.py",
            "overrides": {},
            "repo_root": tmp_path,
        },
    ):
        try:
            validate(**{**kwargs, **disguised_v9})
        except ValueError:
            pass
        else:
            raise AssertionError(f"disguised V9 intent accepted: {disguised_v9}")

    validate(
        **{
            **kwargs,
            "lane_script": "scripts/remote_lane_scpp_stage1.sh",
            "lane_id": "generic-lane",
            "gpu": "A100",
            "requested_gpu": "A100",
            "trainer_module_path": "experiments/generic_trainer.py",
            "overrides": {},
        }
    )

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    policy = main_src.index("_validate_v9_direct_dispatch_policy(")
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    spawn = main_src.index("fn_call = fn.with_options(", preflight)
    assert policy < preflight < spawn
    assert "v9_env_intent = bool(" in main_src
    assert "v9_trainer_intent = _classify_v9_trainer_path(" in main_src
    assert "v9_wrapper_intent = _lane_script_invokes_v9_trainer(" in main_src
    assert 'invocation_options["gpu"] = requested_modal_gpu' in main_src
    assert '"gpu": gpu' in main_src


def test_v9_env_parser_rejects_malformed_and_duplicate_segments() -> None:
    parse = _load_env_parser()
    assert parse("A=1,B=2", strict=True) == {"A": "1", "B": "2"}
    for raw in ("BROKEN", "=empty", "A=1,A=2"):
        try:
            parse(raw, strict=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe V9 env string accepted: {raw!r}")


def test_v9_lane_script_classifier_rejects_path_aliases(tmp_path: Path) -> None:
    classify, cloud = _load_v9_lane_classifier()

    assert classify(cloud.REMOTE_DRIVER, repo_root=REPO_ROOT) is True
    assert classify("scripts/remote_lane_scpp_stage1.sh", repo_root=REPO_ROOT) is False
    for alias in (
        f"scripts/../{cloud.REMOTE_DRIVER}",
        f"./{cloud.REMOTE_DRIVER}",
        f"{cloud.REMOTE_DRIVER} ",
        str(REPO_ROOT / cloud.REMOTE_DRIVER),
    ):
        try:
            classify(alias, repo_root=REPO_ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe V9 path alias accepted: {alias!r}")

    canonical = tmp_path / cloud.REMOTE_DRIVER
    canonical.parent.mkdir(parents=True)
    canonical.write_text("#!/bin/sh\n")
    alias = canonical.parent / "v9-alias.sh"
    alias.symlink_to(canonical.name)
    try:
        classify(alias.relative_to(tmp_path).as_posix(), repo_root=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("symlink alias to the V9 driver was accepted")

    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    assert "is_v9_direct = _classify_v9_lane_script(" in main_src
    assert "is_v9 = _classify_v9_lane_script(" in text
    assert "is_canonical_v9_lane = _classify_v9_lane_script(" in text


def test_modal_image_definition_refuses_forced_rebuild_controls() -> None:
    refuse = _load_image_rebuild_guard()
    refuse({})
    refuse({"MODAL_FORCE_BUILD": "", "MODAL_IGNORE_CACHE": ""})
    for env in (
        {"MODAL_FORCE_BUILD": "1"},
        {"MODAL_IGNORE_CACHE": "yes"},
    ):
        try:
            refuse(env)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"forced Modal rebuild control accepted: {env!r}")

    text = SOURCE.read_text()
    guard = text.index("_refuse_forced_modal_image_rebuild()")
    image_definition = text.index("image = (", guard)
    assert guard < image_definition


def test_remote_h100_v9_guard_refuses_direct_fourteen_hour_call() -> None:
    validate, cloud = _load_v9_remote_contract()
    digest = "a" * 64
    label = "v9-remote-guard"
    overrides = dict(cloud.CUDA_ENV)
    overrides.update(
        {
            "WITNESS_GT_CACHE": f"/modal_results/assets/v9_cgauge/gt_{digest}.npz",
            "WITNESS_OUT_DIR": f"/modal_results/{label}/output",
            "WITNESS_EPOCHS": str(cloud.TYPED_EPOCH_HORIZON),
            "WITNESS_DSL_CONFIG": cloud.DEFAULT_DSL_CONFIG,
            "WITNESS_TORCH_COMPILE_MODE": cloud.DEFAULT_TORCH_COMPILE_MODE,
            "WITNESS_STOP_AFTER_EPOCHS": str(cloud.DEFAULT_STOP_AFTER_EPOCHS),
            "WITNESS_CHILD_TIMEOUT_SECONDS": str(cloud.CHILD_TIMEOUT_SECONDS),
            "WITNESS_NUM_PAIRS": "600",
            "WITNESS_RESUME_FROM": "",
            "WITNESS_RESUME_SHA256": "",
            "WITNESS_GT_CACHE_SHA256": digest,
            "WITNESS_SEGNET_SHA256": digest,
            "WITNESS_POSENET_SHA256": digest,
        }
    )
    now = 1000.0
    kwargs = {
        "lane_script": cloud.REMOTE_DRIVER,
        "label": label,
        "env_overrides": overrides,
        "mounted_code_git_head": "b" * 40,
        "mounted_code_git_branch": "main",
        "sentinel_sha256_local": dict.fromkeys(cloud.REVIEWED_SENTINELS, digest),
        "max_seconds": cloud.CHILD_TIMEOUT_SECONDS,
        "billing_deadline_unix_s": now + cloud.INVOCATION_TIMEOUT_SECONDS,
        "invocation_started_unix_s": now,
        "requested_gpu": cloud.EXACT_H100_GPU,
    }
    validate(**kwargs)
    validate(
        **{
            **kwargs,
            "lane_script": "scripts/generic_wrapper.sh",
            "requested_gpu": "A100",
        }
    )
    resume_path = f"/modal_results/{label}/checkpoints/stage-1.pt"
    validate(
        **{
            **kwargs,
            "env_overrides": {
                **overrides,
                "WITNESS_RESUME_FROM": resume_path,
                "WITNESS_RESUME_SHA256": digest,
            },
        }
    )
    for changes in (
        {"lane_script": f"scripts/../{cloud.REMOTE_DRIVER}"},
        {"lane_script": "scripts/generic_wrapper.sh"},
        {"max_seconds": 14 * 3600},
        {"requested_gpu": "A100"},
        {"requested_gpu": "H100!"},
        {"billing_deadline_unix_s": now + 14 * 3600},
        {"billing_deadline_unix_s": None},
        {"env_overrides": {**overrides, "WITNESS_EPOCHS": "10"}},
        {"env_overrides": {**overrides, "WITNESS_RESUME_SHA256": digest}},
        {
            "env_overrides": {
                **overrides,
                "WITNESS_RESUME_FROM": resume_path,
                "WITNESS_RESUME_SHA256": "not-a-digest",
            }
        },
    ):
        try:
            validate(**{**kwargs, **changes})
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe direct H100 V9 call accepted: {changes}")

    text = SOURCE.read_text()
    endpoint = text[
        text.index("def run_lane_training_cpu("):text.index("def _compact_stamp(")
    ]
    guard = endpoint.index("_validate_v9_remote_h100_invocation(")
    inner = endpoint.index("result = _run_lane_inner(")
    assert guard < inner


def test_modal_train_lane_refuses_duplicate_before_spawn() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]

    assert "active duplicate Modal claim" in text
    assert "refusing second spawn" in text
    assert main_src.index("_ensure_dispatch_claim(") < main_src.index("fn_call = fn.with_options(")


def test_modal_train_lane_dynamic_timeout_resources_and_absolute_deadline() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]

    assert "MODAL_GPU_CPU_REQUEST_LIMIT = (4.0, 4.0)" in text
    assert "MODAL_GPU_MEMORY_REQUEST_LIMIT_MB = (32768, 32768)" in text
    assert text.count("cpu=MODAL_GPU_CPU_REQUEST_LIMIT") == 2
    assert text.count("memory=MODAL_GPU_MEMORY_REQUEST_LIMIT_MB") == 2
    assert "billing_deadline_unix_s = time.time() + hard_timeout_seconds" in main_src
    assert '"timeout": hard_timeout_seconds' in main_src
    assert '"cpu": MODAL_GPU_CPU_REQUEST_LIMIT' in main_src
    assert '"memory": MODAL_GPU_MEMORY_REQUEST_LIMIT_MB' in main_src
    assert "retries=0" in main_src
    assert "remaining_total_before_child - MODAL_FINALIZATION_GRACE_SECONDS" in text
    assert "timeout=remaining_child_seconds" in text
    assert '"billing_deadline_unix_s": billing_deadline_unix_s' in text


def test_worker_watchdog_precedes_setup_and_child_budget_is_recomputed() -> None:
    text = SOURCE.read_text()
    worker = text[text.index("def _run_lane_inner("):text.index("@app.function(", text.index("def _run_lane_inner("))]
    watchdog = worker.index("billing_watchdog.start()")
    source_copy = worker.index('image_workspace = Path("/workspace/pact")')
    recompute = worker.index("remaining_total_before_child = int(")
    child = worker.index("proc = subprocess.run(", recompute)
    entry_prefix = worker[:source_copy]

    assert "os._exit(124)" in entry_prefix
    assert watchdog < source_copy < recompute < child
    assert "remaining_child_seconds =" not in entry_prefix
    assert "billing_watchdog.cancel()" in worker


def test_modal_train_lane_cpu_preflight_precedes_claim_and_gpu_spawn() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    duplicate_check = main_src.index("if _active_claim_exists(")
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    claim = main_src.index("_ensure_dispatch_claim(", preflight)
    gpu_spawn = main_src.index("fn_call = fn.with_options(", claim)

    assert "preflight_first: bool = False" in main_src
    assert '"WITNESS_PREFLIGHT_ONLY": "1"' in main_src
    assert "MODAL_PREFLIGHT_HARD_TIMEOUT_SECONDS = 600" in text
    assert "MODAL_PREFLIGHT_CHILD_TIMEOUT_SECONDS = 300" in text
    assert "preflight_deadline_unix_s" in main_src
    assert ").remote(" in main_src[preflight:claim]
    assert "retries=0" in main_src[preflight:claim]
    assert duplicate_check < preflight < claim < gpu_spawn


def test_local_dispatch_guard_spans_preflight_claim_spawn_and_registration() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    acquire = main_src.index("dispatch_guard_fh = _acquire_local_dispatch_guard(")
    duplicate = main_src.index("if _active_claim_exists(", acquire)
    preflight = main_src.index("preflight_result = run_lane_training_cpu.with_options(")
    claim = main_src.index("_ensure_dispatch_claim(", preflight)
    spawn = main_src.index("fn_call = fn.with_options(", claim)
    register = main_src.index("register_dispatched_call_id_fail_closed(", spawn)
    release = main_src.index("_release_local_dispatch_guard(dispatch_guard_fh)", register)

    assert "modal_train_lane_dispatch_guards" in text
    assert "fcntl.flock(guard_fh.fileno(), fcntl.LOCK_EX)" in text
    assert "fcntl.flock(guard_fh.fileno(), fcntl.LOCK_UN)" in text
    assert "identity = lane_id if lane_global else" in text
    assert "lane_global=is_v9_direct" in main_src
    assert "instance_job_id=None if is_v9_direct else label" in main_src
    assert acquire < duplicate < preflight < claim < spawn < register < release


def test_modal_train_lane_registers_before_convenience_writes_and_cancels_on_failure() -> None:
    text = SOURCE.read_text()
    main_src = text[text.index("@app.local_entrypoint()"):]
    spawn = main_src.index("fn_call = fn.with_options(")
    extract = main_src.index("call_id = function_call_id(fn_call)", spawn)
    register = main_src.index("register_dispatched_call_id_fail_closed(", extract)
    sentinel = main_src.index("sentinel_dir =", register)

    assert spawn < extract < register < sentinel
    assert "fn_call.cancel(terminate_containers=True)" in text
    assert "call-id extraction failure" in main_src
    assert "call-id ledger registration failure" in main_src
    assert "--from-ledger " in main_src
    assert "--call-id {call_id} --execute" in main_src
    assert "modal.FunctionCall.from_id('{call_id}')" in main_src
    assert ".cancel(terminate_containers=True)" in main_src
    assert '"exact_cancel_command": cancel_command' in main_src


def test_modal_train_lane_passes_mounted_git_custody_to_remote_scripts() -> None:
    text = SOURCE.read_text()

    assert "mounted_code_git_head: str" in text
    assert "mounted_code_git_branch: str" in text
    assert '"T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head' in text
    assert '"T1_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch' in text
    assert '"SCPP_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head' in text
    assert '"SCPP_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch' in text
    assert 'mounted_code_git_head = _git_value(repo_root, "rev-parse", "HEAD")' in text
    assert (
        'mounted_code_git_branch = _git_value(repo_root, "branch", "--show-current")'
        in text
    )
    assert "unable to resolve mounted git custody for Modal training" in text


def test_modal_train_lane_records_cost_band_metadata_without_score_authority() -> None:
    text = SOURCE.read_text()

    assert "cost_band_trainer: str = \"\"" in text
    assert "cost_band_epochs: int = 0" in text
    assert "cost_band_batch_size: int = 0" in text
    assert "FATAL: --cost-band-trainer is required" in text
    assert "FATAL: --cost-band-epochs must be positive" in text
    assert "FATAL: --cost-band-batch-size must be positive" in text
    assert '"schema": "modal_training_cost_anchor_metadata_v1"' in text
    assert '"score_claim": False' in text
    assert '"promotion_eligible": False' in text
    assert 'metadata["cost_band_anchor"] = cost_band_anchor' in text


def test_modal_train_lane_returns_experiments_results_artifacts() -> None:
    text = SOURCE.read_text()

    assert 'workspace / "experiments" / "results"' in text


def test_modal_train_lane_preserves_volume_output_submission_runtime_paths() -> None:
    text = SOURCE.read_text()

    assert "modal_training_artifact_relative_path" in text
    assert "fp.relative_to(volume_dir)" in text
    assert "modal_training_artifact_should_collect" in text
    assert 'parts[0] == "output" and parts[1] == "submission"' in text
    assert "MODAL_TRAINING_ARTIFACT_EXTENSIONS" in text
    assert 'rel = modal_training_artifact_relative_path(' in text
    assert "modal_training_artifact_should_collect(rel)" in text


def test_modal_train_lane_image_carries_hard_entropy_runtime_deps() -> None:
    text = SOURCE.read_text()

    assert '"brotli"' in text
    assert '"constriction>=0.4,<0.5"' in text
    assert '"pyppmd>=1.3,<2.0"' in text


def _load_gpu_visibility_detector():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_gpu_visible_in_container"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["_gpu_visible_in_container"]


def test_gpu_detector_accepts_measured_modal_h100_container_signals(monkeypatch) -> None:
    """MEASURED 2026-07-15 gpu_probe_438: Modal H100 containers leave
    CUDA_VISIBLE_DEVICES unset, set NVIDIA_VISIBLE_DEVICES to the GPU UUID,
    and expose /dev/nvidia3 (host index, NOT /dev/nvidia0). The pre-fix
    detector refused this healthy allocation with rc=13 twice (07-12, 07-15)."""
    detector = _load_gpu_visibility_detector()
    measured_env = {
        "NVIDIA_VISIBLE_DEVICES": "GPU-60936b7e-a57a-60e4-ab35-1ca0fe6104ed",
    }
    assert detector(measured_env) is True
    # Device-node-only visibility (env stripped) must also count.
    import glob as glob_module

    monkeypatch.setattr(
        glob_module, "glob", lambda pattern: ["/dev/nvidia3"] if "nvidia" in pattern else []
    )
    assert detector({}) is True
    # Classic CUDA_VISIBLE_DEVICES still counts.
    monkeypatch.setattr(glob_module, "glob", lambda pattern: [])
    assert detector({"CUDA_VISIBLE_DEVICES": "0"}) is True


def test_gpu_detector_refuses_cpu_container_signals(monkeypatch) -> None:
    """CPU containers (no env, no /dev/nvidia*) must stay False so the CPU
    preflight/dispatch stages keep their GPU-free refusal; sentinel 'void'
    and 'none' values never count as GPUs."""
    detector = _load_gpu_visibility_detector()
    import glob as glob_module

    monkeypatch.setattr(glob_module, "glob", lambda pattern: [])
    assert detector({}) is False
    assert detector({"CUDA_VISIBLE_DEVICES": ""}) is False
    assert detector({"CUDA_VISIBLE_DEVICES": "-1"}) is False
    assert detector({"NVIDIA_VISIBLE_DEVICES": "void"}) is False
    assert detector({"NVIDIA_VISIBLE_DEVICES": "none"}) is False
