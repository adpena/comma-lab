from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from experiments import ddm_mt1_modal_multitoken_sign_gate as mt1

REPO = Path(__file__).resolve().parents[2]
LOCAL_MOUNTS = {"add_local_dir", "add_local_file", "add_local_python_source"}
BUILD_STEPS = {"apt_install", "pip_install", "run_commands", "run_function"}


def _worker_image_chain() -> list[tuple[str, ast.Call]]:
    path = REPO / "experiments/ddm_mt1_modal_multitoken_sign_gate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and getattr(node, "value", None) is not None
        and any(
            isinstance(target, ast.Name) and target.id == "worker_image"
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    )

    def flatten(node: ast.AST) -> list[tuple[str, ast.Call]]:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return []
        return [*flatten(node.func.value), (node.func.attr, node)]

    return flatten(assignment.value)


def _copies_into_image(call: ast.Call) -> bool:
    return any(
        keyword.arg == "copy"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def test_mt1_image_has_no_build_step_after_runtime_local_mount() -> None:
    runtime_mount_seen = False
    late_builds: list[str] = []
    methods: list[str] = []
    for method, call in _worker_image_chain():
        methods.append(method)
        if method in LOCAL_MOUNTS:
            if method == "add_local_python_source" or not _copies_into_image(call):
                runtime_mount_seen = True
        elif runtime_mount_seen and method in BUILD_STEPS:
            late_builds.append(method)

    assert late_builds == []
    assert methods.index("run_commands") < methods.index("add_local_file")
    assert methods[-1] == "add_local_python_source"


def test_mt1_sealed_loader_does_not_bind_dispatcher_source_sha(tmp_path: Path) -> None:
    fire_inputs = tmp_path / "fire_inputs"
    fire_inputs.mkdir()
    payload_path = fire_inputs / "payload.bin"
    payload_path.write_bytes(b"payload")
    request = {
        "payloads": {
            "payload.bin": {
                "path": str(payload_path),
                "bytes": 7,
                "sha256": hashlib.sha256(b"payload").hexdigest(),
            }
        },
        "sources": {
            "dispatcher": {
                "path": "historical dispatcher",
                "bytes": 1,
                "sha256": "0" * 64,
            }
        },
    }
    request_path = tmp_path / "SEALED_REQUEST.json"
    request_path.write_bytes(mt1.canonical_json(request))

    payloads, loaded = mt1.load_sealed(
        request_path,
        mt1.sha256_file(request_path),
        fire_inputs,
    )

    assert payloads == {"payload.bin": b"payload"}
    assert loaded == json.loads(request_path.read_text())
