"""Contract tests for the Lane G v3 + OWV3 Fisher stack launcher."""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_g_v3_owv3_fisher_stack.sh"
PROFILER = REPO / "experiments" / "profile_hessian_per_weight.py"
CONVERTER = REPO / "experiments" / "convert_fisher_to_owv3_sensitivity_map.py"
BUILDER = REPO / "experiments" / "build_lane_g_v3_owv3_stack.py"
LAUNCHER = REPO / "scripts" / "launch_lane_on_vastai.py"


def _argparse_flags(path: Path) -> set[str]:
    src = path.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


def _invocation_block(text: str, marker: str) -> str:
    start = text.index(f'"$PYBIN" -u {marker}')
    tail = text[start:]
    end = tail.find("2>&1")
    assert end > 0, f"could not find end of invocation for {marker}"
    return tail[:end]


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"missing OWV3/Fisher lane script: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


def test_script_has_strict_shell_and_cuda_nvdec_preflight() -> None:
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "torch.cuda.is_available()" in text
    assert "CUDA unavailable; Fisher artifacts from CPU/MPS are smoke-only" in text
    assert "probe_nvdec.sh" in text
    assert "--ensure-dali" in text
    assert "NVDEC/DALI probe failed" in text


def test_script_requires_lane_g_v3_anchors_and_owv3_inflate_dispatch() -> None:
    text = SCRIPT.read_text()
    for required in (
        "experiments/results/lane_g_v3_landed",
        "iter_0/renderer.bin",
        "iter_0/masks.mkv",
        "iter_0/optimized_poses.pt",
        "archive_lane_g_v3.zip",
        "experiments/profile_hessian_per_weight.py",
        "experiments/convert_fisher_to_owv3_sensitivity_map.py",
        "experiments/build_lane_g_v3_owv3_stack.py",
        "experiments/contest_auth_eval.py",
        "submissions/robust_current/inflate_renderer.py",
        "magic == b\"OWV3\"",
    ):
        assert required in text
    assert 'ANCHOR_MAGIC="$("$PYBIN" -c "print(open(' in text
    assert '[ "$ANCHOR_MAGIC" = "ASYM" ]' in text


def test_profiler_invocation_uses_real_cuda_flags() -> None:
    text = SCRIPT.read_text()
    block = _invocation_block(text, "experiments/profile_hessian_per_weight.py")
    flags = set(re.findall(r"--([a-z][a-z0-9-]+)", block))
    parser_flags = _argparse_flags(PROFILER)
    invented = flags - parser_flags
    assert not invented, f"profile_hessian_per_weight.py invented flags: {invented}"
    assert "--device cuda" in block
    assert "--pair-batch" in block
    assert 'PAIR_WEIGHT_ARGS=(--all-pairs)' in text
    assert 'PAIR_WEIGHT_ARGS=(--pair-weights "$PAIR_WEIGHTS")' in text


def test_converter_and_builder_invocations_use_real_flags() -> None:
    text = SCRIPT.read_text()

    convert_block = _invocation_block(
        text,
        "experiments/convert_fisher_to_owv3_sensitivity_map.py",
    )
    convert_flags = set(re.findall(r"--([a-z][a-z0-9-]+)", convert_block))
    invented_convert = convert_flags - _argparse_flags(CONVERTER)
    assert not invented_convert, (
        f"convert_fisher_to_owv3_sensitivity_map.py invented flags: {invented_convert}"
    )
    assert "--missing-policy protect" in convert_block
    assert "--metadata-json" in convert_block

    build_block = _invocation_block(text, "experiments/build_lane_g_v3_owv3_stack.py")
    build_flags = set(re.findall(r"--([a-z][a-z0-9-]+)", build_block))
    invented_build = build_flags - _argparse_flags(BUILDER)
    assert not invented_build, (
        f"build_lane_g_v3_owv3_stack.py invented flags: {invented_build}"
    )
    assert "--sensitivity-map" in build_block
    assert "--bit-budget-ratio" in build_block


def test_exact_eval_is_cuda_by_default_and_non_cuda_is_smoke_only() -> None:
    text = SCRIPT.read_text()
    assert 'RUN_CONTEST_EVAL="${RUN_CONTEST_EVAL:-1}"' in text
    assert 'AUTH_DEVICE="${AUTH_EVAL_DEVICE:-cuda}"' in text
    assert 'if [ "$RUN_CONTEST_EVAL" != "1" ]; then' in text
    assert "Archive requires CUDA eval before promotion" in text
    assert 'if [ "$AUTH_DEVICE" != "cuda" ]' in text
    assert "would be advisory-only" in text
    assert 'ALLOW_NON_CUDA_EVAL:-0' in text

    eval_block = _invocation_block(text, "experiments/contest_auth_eval.py")
    assert '--archive "$STACKED_ARCHIVE"' in eval_block
    assert '--device "$AUTH_DEVICE"' in eval_block
    assert '--work-dir "$EVAL_WORK_DIR"' in eval_block
    assert '"lane_status": "COMPLETE_CONTEST_CUDA"' in text


def test_hardened_vast_tarball_discovers_lane_g_v3_anchor() -> None:
    spec = importlib.util.spec_from_file_location("_vast_launcher_under_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    anchors = set(mod._discover_anchor_paths_from_lane_scripts())
    assert "experiments/results/lane_g_v3_landed" in anchors
    assert "experiments/results/lane_g_v3_landed/iter_0/renderer.bin" in anchors
    assert "experiments/results/lane_g_v3_landed/iter_0/masks.mkv" in anchors
    assert "experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt" in anchors
