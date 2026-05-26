#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove local runtime consumption for the PR101 Kaggle proxy packet.

This is a local-only proof. It reads the runtime packet manifest emitted by
``tools/build_pr101_kaggle_proxy_runtime_packet.py``, verifies the unchanged
archive custody, verifies that ``inflate.py`` contains only the three supported
bias-param replacements, and runs a no-scorer wrapper-route probe proving that
``inflate.sh`` invokes the packet-local ``inflate.py`` entrypoint.

It does not run inflate, invoke scorers, dispatch jobs, or claim score.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/prove_pr101_kaggle_proxy_runtime_consumption.py"
PACKET_SCHEMA = "pr101_kaggle_proxy_runtime_packet_v1"
PROOF_SCHEMA = "pr101_kaggle_proxy_runtime_consumption_proof_v1"
PROOF_KIND = "static_bias_patch_plus_local_wrapper_route_plus_no_scorer_bias_runtime_v1"
CANDIDATE_PARAM_SCHEMA = "pr101_kaggle_proxy_bias_runtime_params_v1"
DEFAULT_MANIFEST = Path(
    "experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/"
    "pr101_proxy_sweep/proxy_runtime_packet/runtime_packet_manifest.json"
)
PROOF_NAME = "runtime_consumption_proof.json"
SUPPORTED_BIAS_SLOTS = {
    "bias_r": "up[:, 0, 0]",
    "bias_b": "up[:, 0, 2]",
    "bias_g": "up[:, 1, 1]",
}
OLD_PR101_LINES = {
    "bias_r": "up[:, 0, 0].sub_(1.0)",
    "bias_b": "up[:, 0, 2].sub_(1.0)",
    "bias_g": "up[:, 1, 1].sub_(1.0)",
}
UNSUPPORTED_PARAMS = ("delta_scale", "latent_delta_scale", "smooth_weight")
REMOVED_UNSUPPORTED_BLOCKERS = (
    "unsupported_proxy_params_not_runtime_consumed",
    "delta_scale_not_runtime_consumed",
    "latent_delta_scale_not_runtime_consumed",
    "smooth_weight_not_runtime_consumed",
)
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}
ROUTE_PROBE_FILE_LIST_ENTRY = "0.mkv"
ROUTE_PROBE_TIMEOUT_SECONDS = 10
BIAS_RUNTIME_PROBE_CAMERA_SHAPE = (2, 2)
BIAS_RUNTIME_PROBE_BASE_VALUE = 100.0
FORBIDDEN_RUNTIME_PROBE_IMPORT_MARKERS = (
    "posenet",
    "scorer",
    "segmentation_models_pytorch",
    "segnet",
    "upstream",
)


class RuntimeConsumptionProofError(ValueError):
    """Raised when the local runtime-consumption proof must fail closed."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConsumptionProofError(f"{field} must be a JSON object")
    return value


def _require_false(payload: Mapping[str, Any], field: str) -> None:
    if payload.get(field) is not False:
        raise RuntimeConsumptionProofError(f"{field} must be false")


def _canonical_json_sha256(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_manifest_self_hash(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("manifest_sha256_excluding_self")
    if not isinstance(expected, str):
        raise RuntimeConsumptionProofError("manifest_sha256_excluding_self must be present")
    basis = dict(manifest)
    basis.pop("manifest_sha256_excluding_self", None)
    actual = _canonical_json_sha256(basis)
    if actual != expected:
        raise RuntimeConsumptionProofError("runtime packet manifest self-hash mismatch")


def _resolve_packet_file(packet_dir: Path, record: Mapping[str, Any], field: str) -> Path:
    relpath = record.get("relpath")
    if not isinstance(relpath, str) or not relpath:
        raise RuntimeConsumptionProofError(f"{field}.relpath must be a non-empty string")
    path = Path(relpath)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeConsumptionProofError(f"{field}.relpath must stay inside packet_dir")
    return packet_dir / path


def _verify_archive_unchanged(manifest: Mapping[str, Any], packet_dir: Path) -> dict[str, Any]:
    if manifest.get("archive_changed") is not False:
        raise RuntimeConsumptionProofError("archive_changed must be false")
    source = _require_mapping(manifest.get("source_archive"), "source_archive")
    packet = _require_mapping(manifest.get("packet_archive"), "packet_archive")
    source_sha = source.get("sha256")
    packet_sha = packet.get("sha256")
    unchanged_sha = manifest.get("archive_unchanged_sha256")
    if not all(isinstance(value, str) and len(value) == 64 for value in (source_sha, packet_sha, unchanged_sha)):
        raise RuntimeConsumptionProofError("archive SHA fields must be 64-char hex strings")
    if source_sha != packet_sha or packet_sha != unchanged_sha:
        raise RuntimeConsumptionProofError("source, packet, and unchanged archive SHA fields must match")
    archive_path = _resolve_packet_file(packet_dir, packet, "packet_archive")
    if not archive_path.is_file():
        raise RuntimeConsumptionProofError(f"packet archive missing: {archive_path}")
    actual_sha = sha256_file(archive_path)
    if actual_sha != packet_sha:
        raise RuntimeConsumptionProofError("packet archive file SHA does not match manifest")
    return {
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_sha,
        "manifest_archive_sha256": packet_sha,
    }


def _runtime_file_record(manifest: Mapping[str, Any], relpath: str) -> Mapping[str, Any]:
    custody = _require_mapping(manifest.get("runtime_custody"), "runtime_custody")
    files = custody.get("runtime_files")
    if not isinstance(files, list):
        raise RuntimeConsumptionProofError("runtime_custody.runtime_files must be a list")
    matches = [
        _require_mapping(row, "runtime_custody.runtime_files[]")
        for row in files
        if isinstance(row, Mapping) and row.get("relpath") == relpath
    ]
    if len(matches) != 1:
        raise RuntimeConsumptionProofError(f"expected exactly one runtime file record for {relpath}")
    return matches[0]


def _verify_runtime_file_sha(manifest: Mapping[str, Any], packet_dir: Path, relpath: str) -> dict[str, Any]:
    record = _runtime_file_record(manifest, relpath)
    path = packet_dir / relpath
    if not path.is_file():
        raise RuntimeConsumptionProofError(f"{relpath} missing: {path}")
    actual_sha = sha256_file(path)
    if record.get("sha256") != actual_sha:
        raise RuntimeConsumptionProofError(f"{relpath} SHA does not match runtime custody manifest")
    return {
        "path": _repo_rel(path),
        "sha256": actual_sha,
        "bytes": path.stat().st_size,
    }


def _verify_inflate_file(manifest: Mapping[str, Any], packet_dir: Path) -> dict[str, Any]:
    patch = _require_mapping(manifest.get("runtime_patch"), "runtime_patch")
    if patch.get("patched_file") != "inflate.py":
        raise RuntimeConsumptionProofError("runtime_patch.patched_file must be inflate.py")
    inflate_record = _runtime_file_record(manifest, "inflate.py")
    inflate_path = packet_dir / "inflate.py"
    if not inflate_path.is_file():
        raise RuntimeConsumptionProofError(f"inflate.py missing: {inflate_path}")
    actual_sha = sha256_file(inflate_path)
    if inflate_record.get("sha256") != actual_sha:
        raise RuntimeConsumptionProofError("inflate.py SHA does not match runtime custody manifest")

    text = inflate_path.read_text(encoding="utf-8")
    rows_raw = patch.get("runtime_consumed_params")
    if not isinstance(rows_raw, list):
        raise RuntimeConsumptionProofError("runtime_patch.runtime_consumed_params must be a list")
    rows = [_require_mapping(row, "runtime_patch.runtime_consumed_params[]") for row in rows_raw]
    params_seen: set[str] = set()
    consumed_rows: list[dict[str, Any]] = []
    for row in rows:
        param = row.get("param")
        replacement = row.get("replacement")
        slot = row.get("slot")
        if not isinstance(param, str) or param not in SUPPORTED_BIAS_SLOTS:
            raise RuntimeConsumptionProofError(f"unsupported runtime-consumed param in manifest: {param!r}")
        if param in params_seen:
            raise RuntimeConsumptionProofError(f"duplicate runtime-consumed param in manifest: {param}")
        params_seen.add(param)
        if slot != SUPPORTED_BIAS_SLOTS[param]:
            raise RuntimeConsumptionProofError(f"slot mismatch for {param}")
        if not isinstance(replacement, str) or not replacement:
            raise RuntimeConsumptionProofError(f"replacement missing for {param}")
        if text.count(replacement) != 1:
            raise RuntimeConsumptionProofError(f"inflate.py must contain exactly one manifest replacement for {param}")
        consumed_rows.append(
            {
                "param": param,
                "slot": slot,
                "replacement": replacement,
                "value": row.get("value"),
                "occurrences": 1,
            }
        )
    if params_seen != set(SUPPORTED_BIAS_SLOTS):
        missing = sorted(set(SUPPORTED_BIAS_SLOTS) - params_seen)
        raise RuntimeConsumptionProofError(f"missing supported bias replacements: {missing}")

    for param, old_line in OLD_PR101_LINES.items():
        if old_line in text:
            raise RuntimeConsumptionProofError(f"old PR101 bias line remains in inflate.py for {param}")

    for param in UNSUPPORTED_PARAMS:
        if param in text:
            raise RuntimeConsumptionProofError(f"unsupported param name appears in inflate.py: {param}")

    return {
        "inflate_path": _repo_rel(inflate_path),
        "inflate_sha256": actual_sha,
        "supported_bias_params_consumed": sorted(consumed_rows, key=lambda row: row["param"]),
        "old_pr101_sub_lines_absent": sorted(OLD_PR101_LINES.values()),
        "unsupported_param_names_absent_from_inflate_py": list(UNSUPPORTED_PARAMS),
    }


def _sentinel_inflate_py() -> str:
    return """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

sentinel = Path(os.environ["PR101_INFLATE_ROUTE_SENTINEL"])
sentinel.parent.mkdir(parents=True, exist_ok=True)
if len(sys.argv) >= 3:
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[2]).write_bytes(b"")
sentinel.write_text(
    json.dumps(
        {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "probe_kind": "pr101_packet_inflate_py_route_probe_v1",
        },
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
"""


def _normalize_probe_path(path_value: str, *, cwd: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _verify_inflate_wrapper_routes_to_packet_inflate_py(
    manifest: Mapping[str, Any],
    packet_dir: Path,
    inflate_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a no-scorer local probe proving ``inflate.sh`` calls packet ``inflate.py``."""

    wrapper_record = _verify_runtime_file_sha(manifest, packet_dir, "inflate.sh")
    original_inflate_sha = inflate_proof.get("inflate_sha256")
    if not isinstance(original_inflate_sha, str) or len(original_inflate_sha) != 64:
        raise RuntimeConsumptionProofError("inflate static proof missing inflate_sha256")

    with tempfile.TemporaryDirectory(prefix="pr101_inflate_route_probe_") as tmp:
        root = Path(tmp)
        probe_packet = root / "packet"
        probe_packet.mkdir()
        probe_wrapper = probe_packet / "inflate.sh"
        probe_inflate = probe_packet / "inflate.py"
        probe_bin = root / "bin"
        probe_bin.mkdir()
        python_shim = probe_bin / "python"
        try:
            python_shim.symlink_to(sys.executable)
        except OSError:
            shutil.copyfile(sys.executable, python_shim)
            shutil.copymode(sys.executable, python_shim)
        shutil.copyfile(packet_dir / "inflate.sh", probe_wrapper)
        shutil.copymode(packet_dir / "inflate.sh", probe_wrapper)
        probe_inflate.write_text(_sentinel_inflate_py(), encoding="utf-8")
        os.chmod(probe_inflate, 0o755)

        data_dir = root / "data"
        output_dir = root / "out"
        data_dir.mkdir()
        output_dir.mkdir()
        (data_dir / "x").write_bytes(b"route-probe-payload")
        file_list = root / "file_list.txt"
        file_list.write_text(f"{ROUTE_PROBE_FILE_LIST_ENTRY}\n", encoding="utf-8")
        sentinel = root / "sentinel.json"

        env = os.environ.copy()
        env["PR101_INFLATE_ROUTE_SENTINEL"] = str(sentinel)
        env["PATH"] = f"{probe_bin}{os.pathsep}{env.get('PATH', '')}"
        try:
            proc = subprocess.run(
                [str(probe_wrapper), str(data_dir), str(output_dir), str(file_list)],
                cwd=probe_packet,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=ROUTE_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeConsumptionProofError("inflate.sh wrapper-route probe timed out") from exc
        except OSError as exc:
            raise RuntimeConsumptionProofError(f"inflate.sh wrapper-route probe could not run: {exc}") from exc

        if proc.returncode != 0:
            raise RuntimeConsumptionProofError(
                "inflate.sh wrapper-route probe failed "
                f"(returncode={proc.returncode}, stderr={proc.stderr.strip()!r})"
            )
        if not sentinel.is_file():
            raise RuntimeConsumptionProofError("inflate.sh did not invoke packet inflate.py")

        sentinel_payload = _require_mapping(read_json(sentinel), "inflate route sentinel")
        argv = sentinel_payload.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise RuntimeConsumptionProofError("inflate route sentinel argv must be a string list")
        cwd_raw = sentinel_payload.get("cwd")
        if not isinstance(cwd_raw, str):
            raise RuntimeConsumptionProofError("inflate route sentinel cwd must be a string")
        if len(argv) != 3:
            raise RuntimeConsumptionProofError("inflate.sh invoked packet inflate.py with unexpected argv shape")

        observed_inflate = _normalize_probe_path(argv[0], cwd=Path(cwd_raw))
        if observed_inflate != probe_inflate.resolve():
            raise RuntimeConsumptionProofError("inflate.sh did not route to packet-local inflate.py")
        observed_src = _normalize_probe_path(argv[1], cwd=Path(cwd_raw))
        observed_dst = _normalize_probe_path(argv[2], cwd=Path(cwd_raw))
        if observed_src != (data_dir / "x").resolve():
            raise RuntimeConsumptionProofError("inflate.sh route probe did not pass the packet x payload")
        if observed_dst != (output_dir / "0.raw").resolve():
            raise RuntimeConsumptionProofError("inflate.sh route probe did not pass the expected output path")

        stdout = proc.stdout
        stderr = proc.stderr

    return {
        "proof_kind": "local_no_scorer_wrapper_route_probe_v1",
        "inflate_sh_path": wrapper_record["path"],
        "inflate_sh_sha256": wrapper_record["sha256"],
        "packet_inflate_py_path": inflate_proof["inflate_path"],
        "packet_inflate_py_sha256": original_inflate_sha,
        "wrapper_invoked_packet_inflate_py": True,
        "probe_entry": ROUTE_PROBE_FILE_LIST_ENTRY,
        "observed_argv_shape": ["inflate.py", "src_bin", "dst_raw"],
        "observed_src_basename": "x",
        "observed_dst_basename": "0.raw",
        "returncode": 0,
        "stdout_sha256": _sha256_text(stdout),
        "stderr_sha256": _sha256_text(stderr),
        "timeout_seconds": ROUTE_PROBE_TIMEOUT_SECONDS,
        "scorers_invoked": False,
        "gpu_required": False,
    }


def _supported_bias_values(inflate_proof: Mapping[str, Any]) -> dict[str, float]:
    rows = inflate_proof.get("supported_bias_params_consumed")
    if not isinstance(rows, list):
        raise RuntimeConsumptionProofError("inflate static proof missing supported_bias_params_consumed")
    values: dict[str, float] = {}
    for raw in rows:
        row = _require_mapping(raw, "supported_bias_params_consumed[]")
        param = row.get("param")
        value = row.get("value")
        if param not in SUPPORTED_BIAS_SLOTS:
            raise RuntimeConsumptionProofError(f"unexpected supported bias param in static proof: {param!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeConsumptionProofError(f"supported bias value for {param} must be numeric")
        values[str(param)] = float(value)
    if set(values) != set(SUPPORTED_BIAS_SLOTS):
        raise RuntimeConsumptionProofError("static proof did not include every supported bias value")
    return values


def _guarded_import_factory(blocked_imports: list[str]):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        lowered = str(name).lower()
        if any(marker in lowered for marker in FORBIDDEN_RUNTIME_PROBE_IMPORT_MARKERS):
            blocked_imports.append(str(name))
            raise RuntimeConsumptionProofError(
                f"runtime bias probe refuses scorer/upstream import: {name}"
            )
        return real_import(name, globals, locals, fromlist, level)

    return real_import, guarded_import


def _verify_bias_params_execute_in_inflate_logic(
    manifest: Mapping[str, Any],
    packet_dir: Path,
    inflate_proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute packet ``inflate()`` under tiny stubs and observe bias-slot mutation."""

    import torch

    inflate_record = _verify_runtime_file_sha(manifest, packet_dir, "inflate.py")
    inflate_path = packet_dir / "inflate.py"
    expected_biases = _supported_bias_values(inflate_proof)
    module_name = f"_pr101_bias_runtime_probe_{os.getpid()}_{abs(hash(str(inflate_path)))}"
    captured: dict[str, Any] = {}
    blocked_imports: list[str] = []
    original_sys_path = list(sys.path)
    saved_modules = {
        name: sys.modules.pop(name, None)
        for name in ("codec", "model", module_name)
    }
    real_import, guarded_import = _guarded_import_factory(blocked_imports)
    interpolate_owner: Any | None = None
    interpolate_was_present = False
    original_interpolate: Any | None = None

    try:
        builtins.__import__ = guarded_import
        sys.path.insert(0, str(packet_dir))
        sys.path.insert(0, str(packet_dir / "src"))
        spec = importlib.util.spec_from_file_location(module_name, inflate_path)
        if spec is None or spec.loader is None:
            raise RuntimeConsumptionProofError(f"could not import packet inflate.py: {inflate_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        def fake_parse_archive(archive_bytes: bytes):  # type: ignore[no-untyped-def]
            captured["archive_bytes_read"] = len(archive_bytes)
            return (
                {},
                torch.zeros((1, 1), dtype=torch.float32),
                {
                    "latent_dim": 1,
                    "base_channels": 1,
                    "eval_size": (1, 1),
                    "n_pairs": 1,
                },
            )

        class FakeDecoder:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                captured["decoder_init_kwargs"] = dict(kwargs)

            def to(self, device):  # type: ignore[no-untyped-def]
                captured["decoder_device"] = str(device)
                return self

            def load_state_dict(self, state_dict):  # type: ignore[no-untyped-def]
                captured["load_state_dict_called"] = state_dict == {}

            def eval(self):  # type: ignore[no-untyped-def]
                captured["decoder_eval_called"] = True

            def __call__(self, latents):  # type: ignore[no-untyped-def]
                captured["latents_shape"] = list(latents.shape)
                batch = int(latents.shape[0])
                return torch.zeros((batch, 2, 3, 1, 1), dtype=torch.float32, device=latents.device)

        def fake_interpolate(flat, size, mode=None, align_corners=None):  # type: ignore[no-untyped-def]
            captured["interpolate_size"] = list(size)
            captured["interpolate_mode"] = mode
            captured["interpolate_align_corners"] = align_corners
            probe = torch.full(
                (int(flat.shape[0]), 3, int(size[0]), int(size[1])),
                BIAS_RUNTIME_PROBE_BASE_VALUE,
                dtype=torch.float32,
                device=flat.device,
            )
            captured["interpolate_tensor"] = probe
            return probe

        module.parse_archive = fake_parse_archive
        module.HNeRVDecoder = FakeDecoder
        if not hasattr(module, "F"):
            module.F = type("_FakeFunctional", (), {})()
        interpolate_owner = module.F
        interpolate_was_present = hasattr(interpolate_owner, "interpolate")
        original_interpolate = getattr(interpolate_owner, "interpolate", None)
        module.F.interpolate = fake_interpolate
        module.CAMERA_H, module.CAMERA_W = BIAS_RUNTIME_PROBE_CAMERA_SHAPE

        runtime_stdout = io.StringIO()
        runtime_stderr = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="pr101_bias_runtime_probe_") as tmp:
            root = Path(tmp)
            src_bin = root / "x"
            dst_raw = root / "0.raw"
            src_bin.write_bytes(b"tiny-no-scorer-runtime-probe")
            with contextlib.redirect_stdout(runtime_stdout), contextlib.redirect_stderr(runtime_stderr):
                n_frames = module.inflate(str(src_bin), str(dst_raw))
            output_bytes = dst_raw.stat().st_size if dst_raw.is_file() else 0

        probe_tensor = captured.get("interpolate_tensor")
        if probe_tensor is None:
            raise RuntimeConsumptionProofError("runtime bias probe did not exercise F.interpolate path")
        up = probe_tensor.reshape(1, 2, 3, *BIAS_RUNTIME_PROBE_CAMERA_SHAPE).detach().cpu()
        slot_indices = {
            "bias_r": (0, 0),
            "bias_b": (0, 2),
            "bias_g": (1, 1),
        }
        slot_proofs: list[dict[str, Any]] = []
        for param, (frame_idx, channel_idx) in slot_indices.items():
            expected_after = BIAS_RUNTIME_PROBE_BASE_VALUE + expected_biases[param]
            observed = up[:, frame_idx, channel_idx]
            max_abs_error = float((observed - expected_after).abs().max().item())
            if max_abs_error > 1e-6:
                raise RuntimeConsumptionProofError(
                    f"runtime bias probe did not consume {param}: max_abs_error={max_abs_error}"
                )
            slot_proofs.append(
                {
                    "param": param,
                    "slot": SUPPORTED_BIAS_SLOTS[param],
                    "expected_delta": expected_biases[param],
                    "before_value": BIAS_RUNTIME_PROBE_BASE_VALUE,
                    "observed_after_mean": float(observed.mean().item()),
                    "max_abs_error": max_abs_error,
                }
            )

        unmodified_errors: list[float] = []
        for frame_idx in range(2):
            for channel_idx in range(3):
                if (frame_idx, channel_idx) in set(slot_indices.values()):
                    continue
                observed = up[:, frame_idx, channel_idx]
                unmodified_errors.append(
                    float((observed - BIAS_RUNTIME_PROBE_BASE_VALUE).abs().max().item())
                )
        max_unmodified_abs_error = max(unmodified_errors) if unmodified_errors else 0.0
        if max_unmodified_abs_error > 1e-6:
            raise RuntimeConsumptionProofError(
                "runtime bias probe changed an unexpected frame/channel slot"
            )

        expected_output_bytes = 2 * BIAS_RUNTIME_PROBE_CAMERA_SHAPE[0] * BIAS_RUNTIME_PROBE_CAMERA_SHAPE[1] * 3
        if n_frames != 2 or output_bytes != expected_output_bytes:
            raise RuntimeConsumptionProofError(
                f"runtime bias probe output shape mismatch: n_frames={n_frames}, bytes={output_bytes}"
            )

        return {
            "proof_kind": "local_no_scorer_real_inflate_bias_runtime_probe_v1",
            "inflate_py_path": inflate_record["path"],
            "inflate_py_sha256": inflate_record["sha256"],
            "packet_inflate_function_executed": True,
            "supported_bias_params_consumed_by_runtime_logic": True,
            "supported_bias_slot_proofs": sorted(slot_proofs, key=lambda row: row["param"]),
            "unmodified_slots_max_abs_error": max_unmodified_abs_error,
            "probe_camera_shape": list(BIAS_RUNTIME_PROBE_CAMERA_SHAPE),
            "probe_output_bytes": output_bytes,
            "probe_n_frames": n_frames,
            "parse_archive_stubbed": True,
            "decoder_stubbed": True,
            "interpolate_stubbed_to_tiny_tensor": True,
            "runtime_stdout_sha256": _sha256_text(runtime_stdout.getvalue()),
            "runtime_stderr_sha256": _sha256_text(runtime_stderr.getvalue()),
            "scorer_import_block_markers": list(FORBIDDEN_RUNTIME_PROBE_IMPORT_MARKERS),
            "blocked_scorer_import_attempts": blocked_imports,
            "scorers_invoked": False,
            "gpu_required": False,
        }
    finally:
        if interpolate_owner is not None:
            if interpolate_was_present:
                interpolate_owner.interpolate = original_interpolate
            elif hasattr(interpolate_owner, "interpolate"):
                delattr(interpolate_owner, "interpolate")
        builtins.__import__ = real_import
        sys.path[:] = original_sys_path
        sys.modules.pop(module_name, None)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _verify_bias_only_candidate_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or not all(isinstance(row, str) for row in blockers):
        raise RuntimeConsumptionProofError("blockers must be a list of strings")
    removed_blockers_present = [blocker for blocker in REMOVED_UNSUPPORTED_BLOCKERS if blocker in blockers]
    if removed_blockers_present:
        raise RuntimeConsumptionProofError(
            "runtime packet blockers still advertise removed unsupported proxy params: "
            f"{removed_blockers_present}"
        )
    if "unsupported_params" in manifest:
        raise RuntimeConsumptionProofError("manifest must not advertise unsupported_params as candidate params")

    if manifest.get("candidate_param_schema") != CANDIDATE_PARAM_SCHEMA:
        raise RuntimeConsumptionProofError(
            f"candidate_param_schema must be {CANDIDATE_PARAM_SCHEMA!r}"
        )
    candidate_params = _require_mapping(manifest.get("candidate_params"), "candidate_params")
    consumed = _require_mapping(manifest.get("runtime_consumed_params"), "runtime_consumed_params")
    if set(candidate_params) != set(SUPPORTED_BIAS_SLOTS):
        raise RuntimeConsumptionProofError("candidate_params must contain only supported bias params")
    if set(consumed) != set(SUPPORTED_BIAS_SLOTS):
        raise RuntimeConsumptionProofError("runtime_consumed_params must contain only supported bias params")
    if dict(candidate_params) != dict(consumed):
        raise RuntimeConsumptionProofError("candidate_params must match runtime_consumed_params")

    ignored = _require_mapping(
        manifest.get("ignored_legacy_handoff_params"),
        "ignored_legacy_handoff_params",
    )
    ignored_rows: dict[str, Any] = {}
    for param, raw_row in ignored.items():
        if param not in UNSUPPORTED_PARAMS:
            raise RuntimeConsumptionProofError(f"unexpected ignored legacy handoff param: {param}")
        row = _require_mapping(raw_row, f"ignored_legacy_handoff_params.{param}")
        if row.get("candidate_param") is not False:
            raise RuntimeConsumptionProofError(
                f"ignored_legacy_handoff_params.{param}.candidate_param must be false"
            )
        if row.get("runtime_consumed") is not False:
            raise RuntimeConsumptionProofError(
                f"ignored_legacy_handoff_params.{param}.runtime_consumed must be false"
            )
        ignored_rows[param] = dict(row)
    return {
        "candidate_param_schema": CANDIDATE_PARAM_SCHEMA,
        "candidate_params": dict(candidate_params),
        "runtime_consumed_params": dict(consumed),
        "ignored_legacy_handoff_params": ignored_rows,
        "removed_unsupported_param_blockers_absent": list(REMOVED_UNSUPPORTED_BLOCKERS),
    }


def build_runtime_consumption_proof(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    proof_path: Path | None = None,
) -> dict[str, Any]:
    """Build and write a local proof for a PR101 Kaggle proxy runtime packet."""

    manifest_path = _repo_path(manifest_path)
    manifest = _require_mapping(read_json(manifest_path), "runtime_packet_manifest")
    if manifest.get("schema") != PACKET_SCHEMA:
        raise RuntimeConsumptionProofError(f"manifest schema must be {PACKET_SCHEMA!r}")
    _verify_manifest_self_hash(manifest)
    for field in (
        "score_claim",
        "score_claim_valid",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "exact_auth_eval_performed",
        "contest_cuda_auth_eval",
        "scorers_invoked",
    ):
        _require_false(manifest, field)

    packet_dir_value = manifest.get("packet_dir")
    if not isinstance(packet_dir_value, str) or not packet_dir_value:
        raise RuntimeConsumptionProofError("packet_dir must be a non-empty string")
    packet_dir = _repo_path(Path(packet_dir_value))
    if not packet_dir.is_dir():
        raise RuntimeConsumptionProofError(f"packet_dir missing: {packet_dir}")

    archive_proof = _verify_archive_unchanged(manifest, packet_dir)
    inflate_proof = _verify_inflate_file(manifest, packet_dir)
    candidate_contract_proof = _verify_bias_only_candidate_contract(manifest)
    wrapper_route_proof = _verify_inflate_wrapper_routes_to_packet_inflate_py(
        manifest,
        packet_dir,
        inflate_proof,
    )
    bias_runtime_proof = _verify_bias_params_execute_in_inflate_logic(
        manifest,
        packet_dir,
        inflate_proof,
    )

    proof_path = packet_dir / PROOF_NAME if proof_path is None else _repo_path(proof_path)

    proof: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "proof_kind": PROOF_KIND,
        "tool": TOOL_NAME,
        "candidate_id": manifest.get("candidate_id", ""),
        "manifest_path": _repo_rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_dir": _repo_rel(packet_dir),
        "proof_path": _repo_rel(proof_path),
        "archive_unchanged_proof": archive_proof,
        "inflate_static_bias_patch_proof": inflate_proof,
        "inflate_wrapper_route_proof": wrapper_route_proof,
        "inflate_runtime_bias_logic_proof": bias_runtime_proof,
        "candidate_contract_proof": candidate_contract_proof,
        "supported_bias_params_static_patch_proven": True,
        "inflate_sh_routes_to_packet_inflate_py": True,
        "runtime_consumption_proven_for_supported_bias_params": True,
        "runtime_consumption_boolean_rationale": (
            "This proof verifies static patched bias lines, local inflate.sh -> packet "
            "inflate.py routing, and real packet inflate() execution under tiny no-scorer "
            "stubs that observe the supported bias params mutating the runtime tensor. "
            "It is still not scorer-backed output validation or contest-CUDA auth eval."
        ),
        "scorers_invoked": False,
        "gpu_required": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_blockers": [
            "proxy_substrate_not_contest_exact_eval",
            "no_contest_cuda_auth_eval",
            "no_scorer_runtime_probe_not_contest_auth_eval",
            "active_level2_lane_dispatch_claim_required_before_exact_eval",
        ],
    }
    proof["proof_sha256_excluding_self"] = _canonical_json_sha256(proof)
    write_json(proof_path, proof)
    return proof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--proof-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proof = build_runtime_consumption_proof(
        manifest_path=args.manifest,
        proof_path=args.proof_path,
    )
    print(json_text({
        "schema": "pr101_kaggle_proxy_runtime_consumption_proof_stdout_v1",
        "proof": proof["proof_path"],
        "candidate_id": proof["candidate_id"],
        "proof_kind": proof["proof_kind"],
        **FALSE_AUTHORITY_FIELDS,
        "inflate_sh_routes_to_packet_inflate_py": proof["inflate_sh_routes_to_packet_inflate_py"],
        "runtime_consumption_proven_for_supported_bias_params": proof[
            "runtime_consumption_proven_for_supported_bias_params"
        ],
        "proof_sha256_excluding_self": proof["proof_sha256_excluding_self"],
        "dispatch_blockers": proof["dispatch_blockers"],
    }), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
