#!/usr/bin/env python3
"""Build local PR98 fixed-channel postprocess ablation candidates.

This is a local-only preparation tool. It never dispatches GPU work and never
claims a score. Each candidate keeps the PR98 archive bytes unchanged and
creates a sanitized runtime variant with a deterministic patch to
``inflate.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.contest_auth_eval import _runtime_dependency_manifest  # noqa: E402

READINESS = REPO / "experiments/results/final_packet_readiness_pr98_pr99_20260504_codex"
SOURCE_RUNTIME = READINESS / "runtime_snapshots/pr98_runtime"
SOURCE_ARCHIVE = READINESS / "archives/pr98_archive.zip"
OUT = Path(__file__).resolve().parent
UPSTREAM = REPO / "upstream"

EXPECTED_ARCHIVE_SHA256 = "7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb"
EXPECTED_ARCHIVE_BYTES = 178392


@dataclass(frozen=True)
class Op:
    op_id: str
    description: str
    source: str


OPS = (
    Op("f0_red_minus1", "frame0 red -1", "            up[:, 0, 0].sub_(1.0)\n"),
    Op("f0_blue_minus1", "frame0 blue -1", "            up[:, 0, 2].sub_(1.0)\n"),
    Op("f1_green_minus1", "frame1 green -1", "            up[:, 1, 1].sub_(1.0)\n"),
)

VARIANTS = (
    ("baseline_all_ops", ()),
    ("drop_f0_red", ("f0_red_minus1",)),
    ("drop_f0_blue", ("f0_blue_minus1",)),
    ("drop_f1_green", ("f1_green_minus1",)),
    ("drop_f0_red_f0_blue", ("f0_red_minus1", "f0_blue_minus1")),
    ("drop_f0_red_f1_green", ("f0_red_minus1", "f1_green_minus1")),
    ("drop_f0_blue_f1_green", ("f0_blue_minus1", "f1_green_minus1")),
    ("drop_all_three", ("f0_red_minus1", "f0_blue_minus1", "f1_green_minus1")),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_file_manifest(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = item.relative_to(root).as_posix()
        if "__pycache__" in item.parts or item.name.startswith(".") or item.name.startswith("._"):
            raise RuntimeError(f"forbidden runtime artifact: {rel}")
        mode = stat.S_IMODE(item.stat().st_mode)
        digest = sha256_file(item)
        size = item.stat().st_size
        entries.append({"relative_path": rel, "bytes": size, "mode": oct(mode), "sha256": digest})
    return entries


def copy_runtime(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    for item in sorted(SOURCE_RUNTIME.rglob("*")):
        rel = item.relative_to(SOURCE_RUNTIME)
        target = dst / rel
        if item.is_dir():
            if "__pycache__" in item.parts or item.name.startswith("."):
                continue
            target.mkdir(parents=True, exist_ok=True)
            continue
        if "__pycache__" in item.parts or item.name.startswith(".") or item.name.startswith("._"):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def patch_inflate(runtime: Path, disabled_ops: tuple[str, ...]) -> dict[str, object]:
    inflate = runtime / "inflate.py"
    text = inflate.read_text()
    applied: list[dict[str, str]] = []
    disabled = set(disabled_ops)
    for op in OPS:
        occurrences = text.count(op.source)
        if occurrences != 1:
            raise RuntimeError(f"{op.op_id} expected once, found {occurrences}")
        if op.op_id in disabled:
            replacement = (
                f"            # ablated {op.description}; original: "
                f"{op.source.strip()}\n"
            )
            text = text.replace(op.source, replacement)
            applied.append({"op_id": op.op_id, "description": op.description, "status": "disabled"})
        else:
            applied.append({"op_id": op.op_id, "description": op.description, "status": "kept"})
    inflate.write_text(text)
    return {"ops": applied}


def compile_python(runtime: Path) -> list[str]:
    compiled: list[str] = []
    for py_file in sorted(runtime.rglob("*.py")):
        source = py_file.read_text()
        compile(source, str(py_file), "exec")
        compiled.append(py_file.relative_to(runtime).as_posix())
    return compiled


def run(cmd: list[str], *, cwd: Path = REPO) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    if not SOURCE_RUNTIME.is_dir():
        raise SystemExit(f"missing sanitized PR98 runtime: {SOURCE_RUNTIME}")
    if not SOURCE_ARCHIVE.is_file():
        raise SystemExit(f"missing PR98 archive: {SOURCE_ARCHIVE}")
    if sha256_file(SOURCE_ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise SystemExit("PR98 archive SHA mismatch")
    if SOURCE_ARCHIVE.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise SystemExit("PR98 archive byte-size mismatch")

    archive_dst = OUT / "archives/pr98_archive.zip"
    archive_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ARCHIVE, archive_dst)

    results = {
        "schema": "pr98_channel_ablation_candidates_v1",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_archive": str(SOURCE_ARCHIVE.relative_to(REPO)),
        "source_runtime": str(SOURCE_RUNTIME.relative_to(REPO)),
        "archive": {
            "path": str(archive_dst.relative_to(REPO)),
            "bytes": archive_dst.stat().st_size,
            "sha256": sha256_file(archive_dst),
            "variant_policy": "unchanged for all runtime ablations",
        },
        "postprocess_ops": [
            {"op_id": op.op_id, "description": op.description, "source": op.source.strip()}
            for op in OPS
        ],
        "candidates": [],
    }

    for name, disabled in VARIANTS:
        runtime = OUT / "runtime_variants" / name
        copy_runtime(runtime)
        patch_report = patch_inflate(runtime, disabled)
        compiled = compile_python(runtime)
        bash_check = run(["bash", "-n", str(runtime / "inflate.sh")])
        if bash_check["returncode"] != 0:
            raise RuntimeError(f"bash -n failed for {name}: {bash_check}")
        files = local_file_manifest(runtime)
        runtime_manifest = _runtime_dependency_manifest(runtime / "inflate.sh", UPSTREAM)
        runtime_sha = runtime_manifest["runtime_tree_sha256"]
        preflight_json = OUT / f"{name}.preflight.json"
        preflight = run([
            sys.executable,
            "experiments/preflight_public_replay_intake.py",
            "--archive",
            str(archive_dst),
            "--inflate-sh",
            str(runtime / "inflate.sh"),
            "--expected-archive-sha256",
            EXPECTED_ARCHIVE_SHA256,
            "--expected-archive-size-bytes",
            str(EXPECTED_ARCHIVE_BYTES),
            "--expected-runtime-tree-sha256",
            runtime_sha,
            "--json-out",
            str(preflight_json),
            "--fail-if-not-ready",
        ])
        if preflight["returncode"] != 0:
            raise RuntimeError(f"preflight failed for {name}: {preflight}")
        candidates_dir = OUT / "eval_command_plans"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        lane_id = f"pr98_channel_ablation_{name}_t4_replay"
        job_name = f"exact_eval_pr98_channel_ablation_{name}_t4_YYYYMMDDTHHMMSSZ"
        exact_eval_command = [
            "# first create a dispatch claim with the concrete timestamped job name",
            ".venv/bin/python tools/claim_lane_dispatch.py claim "
            f"--lane-id {lane_id} --platform lightning "
            f"--instance-job-id {job_name} --agent codex:gpt-5.5 "
            "--predicted-eta-utc YYYY-MM-DDTHH:MM:SSZ "
            f"--status active --notes \"PR98 channel ablation {name}; runtime tree {runtime_sha}; archive unchanged\"",
            ".venv/bin/python scripts/launch_lightning_batch_job.py exact-eval "
            "--state-path .omx/state/lightning_batch_jobs.json "
            f"--job-name {job_name} "
            f"--archive {archive_dst.relative_to(REPO)} "
            "--repo-dir /teamspace/studios/this_studio/pact "
            "--upstream-dir /teamspace/studios/this_studio/pact/upstream "
            f"--inflate-sh {runtime.relative_to(REPO) / 'inflate.sh'} "
            "--machine T4_SMALL "
            "--studio pact "
            "--remote-preflight-ssh-target lightning-pact "
            f"--dispatch-lane-id {lane_id} "
            f"--expected-archive-sha256 {EXPECTED_ARCHIVE_SHA256} "
            f"--expected-archive-size-bytes {EXPECTED_ARCHIVE_BYTES} "
            "--source-manifest PATH_TO_LIGHTNING_SOURCE_MANIFEST.json "
            "--adjudicate --max-sane-score 1.0 --component-trace",
        ]
        command_plan = {
            "candidate": name,
            "lane_id": lane_id,
            "job_name_template": job_name,
            "runtime_tree_sha256": runtime_sha,
            "commands": exact_eval_command,
        }
        command_plan_path = candidates_dir / f"{name}.exact_eval_plan.json"
        command_plan_path.write_text(json.dumps(command_plan, indent=2, sort_keys=True) + "\n")
        results["candidates"].append({
            "name": name,
            "disabled_ops": list(disabled),
            "kept_ops": [op.op_id for op in OPS if op.op_id not in disabled],
            "archive_path": str(archive_dst.relative_to(REPO)),
            "archive_bytes": archive_dst.stat().st_size,
            "archive_sha256": sha256_file(archive_dst),
            "runtime_root": str(runtime.relative_to(REPO)),
            "inflate_sh": str((runtime / "inflate.sh").relative_to(REPO)),
            "runtime_tree_sha256": runtime_sha,
            "runtime_file_count": len(files),
            "runtime_files": files,
            "runtime_manifest": runtime_manifest,
            "patch_report": patch_report,
            "python_compile_files": compiled,
            "preflight_json": str(preflight_json.relative_to(REPO)),
            "preflight_status": "passed",
            "exact_eval_plan": str(command_plan_path.relative_to(REPO)),
            "score_claim": "none",
            "evidence_grade": "local_static_preflight_only",
        })

    unzip_check = run(["unzip", "-t", str(archive_dst)])
    results["verification"] = {
        "unzip_test": unzip_check,
        "preflight": "passed for all candidates",
        "python_compile": "passed for all runtime variants without pycache writes",
        "bash_n": "passed for all inflate.sh variants",
    }
    (OUT / "candidates_manifest.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate_count": len(results["candidates"]),
        "archive_sha256": results["archive"]["sha256"],
        "archive_bytes": results["archive"]["bytes"],
        "manifest": str((OUT / "candidates_manifest.json").relative_to(REPO)),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
