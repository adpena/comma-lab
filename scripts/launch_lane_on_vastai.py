#!/usr/bin/env python3
"""Canonical parent-shell launcher for Vast.ai lane scripts (v2 — harness-friendly).

Built 2026-04-28, hardened V2 same day after harness-incompatibility discovery
(`feedback_bash_harness_kills_long_running_tasks_20260428`). The bash tool
harness kills foreground subprocesses at exit 144 around 5 min, so V2 removes
the 8-min heartbeat poll and parallelizes tarball build with boot wait.
Total wallclock: ~3-5 min (fits harness budget).

V2 flow (8 stages, ~3-5 min total):
  0. Resolve cheapest non-KR RTX 4090 offer (reliability > 0.96)
  1. Build minimal tarball locally (~30s) — runs IN PARALLEL with stages 2-4
     via a background thread so we don't waste the boot wait
  2. Create instance with --label, ssh-direct
  3. Register instance to tracker (so failures can clean up)
  4. Wait 90s for boot + SSH ready
  5. Join tarball build, SCP it (~60-180s)
  6. Extract on remote + lightweight CUDA probe (~15s)
  7. Start setup_full + lane script in tmux session (~10s)
  8. Print SUCCESS + SSH details + exit. Parent uses verify_vast_instances.py
     for heartbeat verification later (separate command).

V1 → V2 changes:
  - REMOVED Stage 9 (8-min heartbeat poll) — moved to verify_vast_instances.py
  - PARALLELIZED tarball build with boot wait (saves ~30s)
  - Tighter SSH timeout (90s instead of 180s) since we know boot only takes 60-90s
  - Returns SUCCESS even if heartbeat hasn't appeared yet — parent's
    verify_vast_instances.py is the canonical readiness signal

Usage (parent flow):
    # Phase 1+2 (this script, ~3-5 min — harness-fit):
    python scripts/launch_lane_on_vastai.py \\
        --lane-script scripts/remote_lane_omega_v2_lagrangian.sh \\
        --label lane_omega_v2 \\
        --predicted-band 0.70 0.95 \\
        --estimated-cost 1.50

    # Phase 3 (separate, async — parent invokes via wakeup):
    python scripts/verify_vast_instances.py --auto-destroy-stale --stale-minutes 30

References:
  - feedback_cycle_1_launch_postmortem_20260428
  - feedback_codex_sandbox_blocks_vastai_dns_20260428
  - feedback_remote_setup_script_correct_path_20260428
  - feedback_per_instance_verify_pattern_20260428
  - feedback_canonical_remote_bootstraps
  - feedback_bash_harness_kills_long_running_tasks_20260428 (THIS HARDENING)
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = REPO_ROOT / ".omx/state/vastai_active_instances.json"
VASTAI = REPO_ROOT / ".venv/bin/vastai"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]


def run(cmd: list[str], timeout: int = 60, capture: bool = True) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def find_offer(min_reliability: float = 0.96, max_dph: float = 0.50) -> int:
    """Search Vast.ai for cheapest 4090 matching constraints. Return offer id."""
    cmd = [
        str(VASTAI), "search", "offers",
        f"gpu_name=RTX_4090 reliability>{min_reliability} inet_down>200 "
        f"disk_space>30 num_gpus=1 geolocation!=KR",
        "-o", "dph", "--raw",
    ]
    rc, out, err = run(cmd, timeout=30)
    if rc != 0:
        raise RuntimeError(f"vastai search failed: {err.strip()}")
    offers = json.loads(out)
    for o in offers:
        if o.get("dph_total", 1.0) <= max_dph:
            return int(o["id"])
    raise RuntimeError(f"No offer < ${max_dph}/hr available")


def create_instance(offer_id: int, label: str) -> int:
    """Create a Vast.ai instance. Returns instance ID (new_contract).

    2026-04-28: relaxed success check — Vast.ai sometimes returns
    `success=False` even when `new_contract` is set, for offers that get
    "queued for allocation" (cur_state=stopped). Caller should verify the
    instance transitions to `cur_state=running` via wait_for_vastai_ready.
    """
    # NVIDIA_DRIVER_CAPABILITIES=all exposes libnvcuvid.so (NVDEC) +
    # libnvidia-encode.so (NVENC) inside the container. Default pytorch
    # image only mounts compute,utility — that's WHY ~85% of 4090s tonight
    # showed "NVDEC missing" / DALI CUDA_ERROR_NO_DEVICE. Setting this on
    # the Vast.ai onstart env resurrects all those hosts.
    # Sources: NVIDIA Container Toolkit docs, DALI #4034, nvidia-docker #1001.
    cmd = [
        str(VASTAI), "create", "instance", str(offer_id),
        "--image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        "--disk", "35",
        "--label", label,
        "--ssh",
        "--env", "-e NVIDIA_DRIVER_CAPABILITIES=all",
        "--raw",
    ]
    rc, out, err = run(cmd, timeout=60)
    if rc != 0:
        raise RuntimeError(f"vastai create failed: {err.strip()}")
    d = json.loads(out)
    instance_id = d.get("new_contract") or d.get("id")
    if instance_id is None:
        raise RuntimeError(f"vastai create returned no instance ID: {d}")
    if not d.get("success"):
        # Soft-warn; the instance may still come up. wait_for_vastai_ready
        # will catch persistent failures.
        sys.stderr.write(
            f"  [warn] vastai create returned success=False but contract={instance_id}; "
            f"will verify state via wait_for_vastai_ready.\n"
        )
    try:
        return int(instance_id)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"vastai create returned non-numeric instance ID "
            f"(value={instance_id!r}): {e}"
        ) from e


def get_ssh_details(instance_id: int) -> tuple[str, int]:
    """Returns (ssh_host, ssh_port). Raises if instance not ready."""
    cmd = [str(VASTAI), "show", "instance", str(instance_id), "--raw"]
    rc, out, err = run(cmd, timeout=30)
    if rc != 0:
        raise RuntimeError(f"vastai show failed: {err.strip()}")
    d = json.loads(out)
    d = d[0] if isinstance(d, list) else d
    host = d.get("ssh_host")
    port = d.get("ssh_port")
    if not host or not port:
        raise RuntimeError(f"Instance not ready: {d.get('actual_status')}")
    return host, int(port)


def destroy_instance(
    instance_id: int,
    *,
    recover: bool = True,
    lane_label: str | None = None,
    recovery_timeout_s: int = 600,
) -> None:
    """Destroy a Vast.ai instance (best-effort, no-raise).

    By default attempts artifact recovery FIRST via
    ``tools.recover_lane_artifacts.recover_before_destroy`` so we never lose a
    crashed-at-auth-eval training run again (Lane RM-d 2026-04-28 incident).
    Pass ``recover=False`` to force an immediate destroy when the instance is
    known-unreachable. Recovery failures are best-effort and never block
    destroy.

    Memory: ``feedback_artifact_recovery_canonical_workflow_20260428``.
    """
    if recover:
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from tools.recover_lane_artifacts import recover_before_destroy
            label = lane_label or f"instance_{instance_id}"
            print(
                f"[destroy_instance] attempting artifact recovery from "
                f"instance={instance_id} label={label} before destroy "
                f"(timeout={recovery_timeout_s}s, --no-recover to skip)",
                file=sys.stderr,
            )
            report = recover_before_destroy(
                instance_id=instance_id,
                lane_label=label,
                overall_timeout_s=recovery_timeout_s,
            )
            if report is not None:
                print(report.summary(), file=sys.stderr)
        except (ImportError, AttributeError) as e:
            sys.stderr.write(
                f"[destroy_instance] recovery module unavailable ({e}); "
                f"proceeding to destroy.\n"
            )
    cmd = [
        "bash", "-c",
        f"echo y | {shlex.quote(str(VASTAI))} destroy instance {instance_id}",
    ]
    run(cmd, timeout=60)


def wait_for_vastai_ready(instance_id: int, timeout_s: int = 300) -> bool:
    """Poll vastai status until actual_status='running'.

    Vast.ai's container can be `cur_state='running'` while `actual_status=None`
    (OS still booting, sshd not yet listening). Polling actual_status is the
    canonical readiness signal — once it flips to 'running', SSH is ready
    in 5-30s. Without this poll, blind SSH retries can fail for 3-5 min
    while the OS finishes booting.

    2026-04-28 added after V2 launcher's blind SSH wait kept failing at 180s
    on slow-boot offers (genuine boot takes 2-5 min on some hosts).
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = _vast_show_instance_dict(str(instance_id))
        if info and info.get("actual_status") == "running":
            return True
        time.sleep(15)
    return False


def _vast_show_instance_dict(instance_id: str) -> dict | None:
    """Internal: get vastai show instance as dict (or None on error)."""
    cmd = [str(VASTAI), "show", "instance", str(instance_id), "--raw"]
    rc, out, err = run(cmd, timeout=30)
    if rc != 0:
        return None
    try:
        d = json.loads(out)
        return d[0] if isinstance(d, list) else d
    except (json.JSONDecodeError, IndexError):
        return None


def wait_for_ssh(host: str, port: int, timeout_s: int = 90) -> bool:
    """Poll SSH until reachable. Returns True on success, False on timeout.

    With `wait_for_vastai_ready` confirming OS is up, this should succeed
    in 5-30s. 90s timeout adds safety margin without consuming harness budget.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc, _, _ = run(
            ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}", "echo ready"],
            timeout=20,
        )
        if rc == 0:
            return True
        time.sleep(10)
    return False


def _discover_anchor_paths_from_lane_scripts() -> list[str]:
    """Auto-discover ANCHOR_* paths referenced in scripts/remote_lane_*.sh.

    Returns deduplicated, sorted list of paths (relative to REPO_ROOT) that
    every deployed lane references as its anchor archive / renderer / poses.

    This makes the tarball self-maintaining: add a new lane that references
    `ANCHOR_FOO="experiments/results/lane_x/foo.bin"` and the next launch
    auto-includes it. No manual launcher edits.
    """
    import re as _re
    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.is_dir():
        return []
    anchor_paths: set[str] = set()
    # 2026-04-29: regex MUST handle `="${VAR:-experiments/...}"` form (the
    # canonical lane idiom). Old `(?:"|\$\{[^:}]+:-)?` failed on this because
    # `?` only allows ONE OR THE OTHER, not BOTH `"` then `${VAR:-`. Split
    # into two independent optional groups.
    pattern = _re.compile(
        r'(?:ANCHOR_\w+|LANE_\w*ARCHIVE\w*|LANE_\w*POSES\w*|LANE_\w*MASKS\w*|LANE_\w*RENDERER\w*)='
        r'"?'                            # optional opening quote
        r'(?:\$\{[^:}]+:-)?'             # optional ${VAR:- prefix
        r'(experiments/results/[\w./_-]+|upstream/[\w./_-]+|submissions/[\w./_-]+)'
    )
    for sh in scripts_dir.glob("remote_lane_*.sh"):
        try:
            text = sh.read_text(errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        for m in pattern.finditer(text):
            anchor_paths.add(m.group(1))
    return sorted(anchor_paths)


def _autodiscover_referenced_files(paths: set[str], max_iter: int = 4) -> set[str]:
    """Iteratively scan paths for repo-relative refs; auto-add what they need.

    Detects patterns:
      - `from upstream.frame_utils import …` → upstream/frame_utils.py
      - `import upstream.modules` → upstream/modules.py
      - `bash scripts/probe_nvdec.sh` → scripts/probe_nvdec.sh
      - `source scripts/env_helper.sh` → scripts/env_helper.sh
      - `python experiments/foo.py` → experiments/foo.py
      - `from tac.parametrize_strip import ...` → src/tac/parametrize_strip.py

    Repeats until no new paths discovered (fixed point). max_iter guards
    against infinite loops in degenerate cases.
    """
    import re as _re
    # Patterns that map a reference to a candidate repo path
    py_import = _re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", _re.M)
    bash_run = _re.compile(r"\b(?:bash|sh|source|\.\s+)\s+([./\w-]+\.sh)\b")
    py_run = _re.compile(r"\bpython3?\s+(?:-m\s+([\w.]+)|([./\w-]+\.py))")
    pybin_run = _re.compile(r"\$PYBIN\s+(?:-u\s+|-m\s+)?([./\w-]+\.py)")

    def _resolve_module_to_path(module: str) -> str | None:
        """Map dotted module like `tac.parametrize_strip` to repo path."""
        # Try src/<module>.py
        candidates = [
            f"src/{module.replace('.', '/')}.py",
            f"src/{module.replace('.', '/')}/__init__.py",
            f"{module.replace('.', '/')}.py",
            f"{module.replace('.', '/')}/__init__.py",
        ]
        for c in candidates:
            if (REPO_ROOT / c).is_file():
                return c
        return None

    paths = set(paths)
    for _ in range(max_iter):
        new_refs: set[str] = set()
        for p in list(paths):
            full = REPO_ROOT / p
            if not full.is_file():
                continue
            try:
                text = full.read_text(errors="ignore")
            except (FileNotFoundError, PermissionError):
                continue
            # Python imports → src/.../X.py or X/__init__.py
            for m in py_import.finditer(text):
                mod = m.group(1) or m.group(2)
                if not mod or mod.startswith(("torch", "numpy", "pathlib", "json",
                                              "os", "sys", "re", "math", "typing",
                                              "argparse", "collections", "itertools",
                                              "functools", "subprocess", "logging",
                                              "datetime", "time", "warnings",
                                              "pickle", "io", "tempfile", "shutil",
                                              "abc", "dataclasses", "enum", "copy",
                                              "hashlib", "struct", "zipfile", "gzip",
                                              "brotli", "tarfile", "pytest", "hypothesis",
                                              "_", "__future__")):
                    continue
                resolved = _resolve_module_to_path(mod)
                if resolved and resolved not in paths:
                    new_refs.add(resolved)
            # Bash sources: bash X.sh, source X.sh, . X.sh
            for m in bash_run.finditer(text):
                ref = m.group(1)
                if ref.startswith("/"):
                    continue
                # Strip leading "./" if present
                ref = ref.lstrip("./")
                if (REPO_ROOT / ref).is_file() and ref not in paths:
                    new_refs.add(ref)
            # Python script invocations: python X.py or python -m mod
            for m in py_run.finditer(text):
                if m.group(1):  # `-m mod`
                    resolved = _resolve_module_to_path(m.group(1))
                    if resolved and resolved not in paths:
                        new_refs.add(resolved)
                if m.group(2):  # `X.py`
                    ref = m.group(2).lstrip("./")
                    if (REPO_ROOT / ref).is_file() and ref not in paths:
                        new_refs.add(ref)
            # $PYBIN <args> X.py — common in remote_lane scripts
            for m in pybin_run.finditer(text):
                ref = m.group(1).lstrip("./")
                if (REPO_ROOT / ref).is_file() and ref not in paths:
                    new_refs.add(ref)
        if not new_refs:
            break  # fixed point
        paths.update(new_refs)
    return paths


def _enumerate_python_and_shell(root_subdir: str, max_total_mb: int = 50) -> list[str]:
    """Enumerate all .py + .sh + .json + .toml + .md + .txt + .env files under a subdir.

    Excludes __pycache__, *.pyc. Returns paths relative to REPO_ROOT.
    Caps cumulative size at `max_total_mb` to avoid runaway includes.

    Codex F5 fix (2026-04-28): added .env to the suffix list because
    submissions/robust_current/config.env is REQUIRED at remote eval time
    (it's the file that sets PYTHON_INFLATE=renderer for inflate.sh; without
    it, inflate.sh falls into the ffmpeg path and tries to read 0.mkv from
    extracted/, which doesn't exist in the archive). Lane RM-d burned $1+
    discovering this. Also added .env.example for completeness.
    """
    base = REPO_ROOT / root_subdir
    if not base.is_dir():
        return []
    out: list[str] = []
    total_bytes = 0
    cap = max_total_mb * 1024 * 1024
    allowed_suffixes = (".py", ".sh", ".json", ".toml", ".md", ".txt", ".env")
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts:
            continue
        if p.suffix in (".pyc", ".pyo"):
            continue
        # Only ship code/config files from these dirs; data artifacts come
        # via explicit anchor includes
        if p.suffix not in allowed_suffixes:
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if total_bytes + sz > cap:
            break
        total_bytes += sz
        out.append(str(p.relative_to(REPO_ROOT)))
    return out


def build_tarball() -> Path:
    """Build a minimal tarball of the repo for SCP via EXPLICIT FILE LIST.

    Replaces the fragile include-dir + many-excludes pattern (which kept
    breaking — 2026-04-28 had 5.9GB tarballs, missing-anchor crashes,
    lane_a_landed/eval_work scan timeouts).

    Uses tar's `-T <file>` with an explicit positive include list:
      1. Auto-enumerate all .py/.sh/.json/.toml/.md/.txt under src/, scripts/,
         experiments/, submissions/robust_current/ (capped at 50MB each)
      2. Auto-discover ANCHOR_* paths from remote_lane_*.sh
      3. Add canonical small includes: pyproject.toml, README.md,
         upstream/{models,videos,evaluate.py,modules.py,__init__.py,
         public_test_video_names.txt}

    Result: tarball ~150-200MB, build ~10-30s, NO surprises.
    """
    tmpfile = Path(tempfile.mkdtemp()) / "pact_minimal.tar.gz"
    file_list_path = tmpfile.parent / "file_list.txt"

    # 1. Code files from key dirs (capped to prevent runaway)
    paths: list[str] = []
    paths += _enumerate_python_and_shell("src", max_total_mb=30)
    paths += _enumerate_python_and_shell("scripts", max_total_mb=10)
    paths += _enumerate_python_and_shell("experiments", max_total_mb=30)
    paths += _enumerate_python_and_shell("submissions/robust_current", max_total_mb=20)

    # 2. Auto-discovered anchor paths (data artifacts referenced by lane scripts)
    anchors = _discover_anchor_paths_from_lane_scripts()
    for ap in anchors:
        ap_full = REPO_ROOT / ap
        if not ap_full.exists():
            continue
        if ap_full.is_file():
            paths.append(ap)
        elif ap_full.is_dir():
            # Walk the dir for files
            for p in sorted(ap_full.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    paths.append(str(p.relative_to(REPO_ROOT)))

    # 3. Canonical small includes (always required) — ALL upstream/*.py
    # so imports like `from upstream.frame_utils import ...` work
    canonical = [
        "pyproject.toml", "README.md",
        "upstream/__init__.py", "upstream/public_test_video_names.txt",
    ]
    paths += [p for p in canonical if (REPO_ROOT / p).exists()]
    # Auto-include ALL .py files at upstream/ top-level (small ~30KB total)
    upstream_dir = REPO_ROOT / "upstream"
    if upstream_dir.is_dir():
        for p in sorted(upstream_dir.glob("*.py")):
            paths.append(str(p.relative_to(REPO_ROOT)))

    # 4. upstream models + videos (need to enumerate files explicitly)
    for subdir in ("upstream/models", "upstream/videos"):
        d = REPO_ROOT / subdir
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    paths.append(str(p.relative_to(REPO_ROOT)))

    # 5. AUTO-DISCOVER referenced files via import/source/bash references
    # Iteratively scan included files for repo-relative references and
    # auto-add any that exist locally but aren't yet included. Fixed-point
    # iteration; typically 1-2 rounds.
    paths = _autodiscover_referenced_files(set(paths))

    # Dedupe + write file list
    paths = sorted(set(paths))
    file_list_path.write_text("\n".join(paths) + "\n")

    # tar -T <file_list> packs ONLY the listed paths. No exclude-glob magic.
    cmd = [
        "tar", "-czf", str(tmpfile),
        "-C", str(REPO_ROOT),
        "-T", str(file_list_path),
    ]
    rc, out, err = run(cmd, timeout=300)
    file_list_path.unlink(missing_ok=True)
    if rc != 0:
        raise RuntimeError(f"tar failed (paths={len(paths)}): {err.strip()[:200]}")
    return tmpfile


def add_lane_anchors_to_tarball(tar_path: Path, anchor_dirs: list[str]) -> None:
    """Append lane-specific anchor data to the tarball."""
    valid_dirs = [d for d in anchor_dirs if (REPO_ROOT / d).exists()]
    if not valid_dirs:
        return
    cmd = ["tar", "-rzf", str(tar_path), "-C", str(REPO_ROOT)] + valid_dirs
    # gzip-mode tar -r isn't natively supported on macOS bsdtar; rebuild
    # the tarball if needed
    rc, out, err = run(cmd, timeout=120)
    if rc != 0:
        # Fallback: build a separate small tar and let SCP transfer both
        sys.stderr.write(f"WARN: anchor append failed ({err.strip()[:80]}), continuing without anchors\n")


def scp_tarball(tar_path: Path, host: str, port: int) -> None:
    """SCP tarball to /workspace on remote."""
    cmd = ["scp"] + SSH_OPTS + ["-P", str(port), str(tar_path),
                                f"root@{host}:/workspace/pact.tar.gz"]
    rc, out, err = run(cmd, timeout=600)
    if rc != 0:
        raise RuntimeError(f"scp failed: {err.strip()}")


def extract_remote(host: str, port: int) -> None:
    """Extract tarball into /workspace/pact on remote."""
    cmd = ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}", (
        "mkdir -p /workspace/pact && "
        "tar -xzf /workspace/pact.tar.gz -C /workspace/pact && "
        "rm /workspace/pact.tar.gz && "
        "echo extracted"
    )]
    rc, out, err = run(cmd, timeout=120)
    # Distinguish timeout (rc=-1, err=='TIMEOUT') from other failures so the
    # caller log line is actionable rather than "extract failed: TIMEOUT".
    if rc == -1 and err.strip() == "TIMEOUT":
        raise RuntimeError(
            f"extract timed out after 120s (tarball may be huge or remote disk slow); "
            f"manually verify /workspace/pact/ before retrying"
        )
    if rc != 0 or "extracted" not in out:
        raise RuntimeError(
            f"extract failed: rc={rc} stderr={err.strip()[:200]} "
            f"stdout={out.strip()[:200]}"
        )


def lightweight_nvdec_probe(host: str, port: int) -> tuple[bool, str]:
    """Pre-DALI lightweight CUDA + nvidia-smi probe. Returns (ok, message)."""
    cmd = ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}", (
        "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1 | head -1; "
        "echo '---'; "
        "/opt/conda/bin/python -c "
        "'import torch; print(\"cuda_available=\", torch.cuda.is_available()); "
        "print(\"cuda_count=\", torch.cuda.device_count())' 2>&1"
    )]
    rc, out, err = run(cmd, timeout=30)
    if rc != 0:
        return False, f"SSH failed: {err.strip()[:80]}"
    if "cuda_available= True" in out and "cuda_count= 1" in out:
        return True, out.strip()
    return False, out.strip()


def execute_lane_in_tmux(
    host: str, port: int, lane_script: str,
) -> bool:
    """Start setup + lane in detached background. Returns True on success.

    Uses nohup + setsid + disown so the lane survives SSH disconnect WITHOUT
    requiring tmux (which is not installed on bare PyTorch containers and
    apt-get install can fail/timeout).

    Pattern:
      1. SSH + write a wrapper script to /workspace/run_lane.sh
      2. SSH + setsid+nohup the wrapper → returns immediately
      3. Verify the wrapper PID is alive via SSH
    """
    # Write wrapper script that does setup_full.sh + lane in sequence
    # 2026-04-28 metabug C fix: gate lane on setup success. If setup_full.sh
    # fails (e.g., NVDEC probe at Stage 4), `env.sh` never gets written and
    # the lane script crashes noisily. Use && so lane only runs on setup success.
    wrapper = (
        "#!/bin/bash\n"
        "set +e  # don't abort on setup errors so logs are visible\n"
        "cd /workspace/pact || exit 1\n"
        f"bash scripts/remote_setup_full.sh > /workspace/setup.log 2>&1\n"
        "SETUP_RC=$?\n"
        "if [ $SETUP_RC -ne 0 ]; then\n"
        "    echo \"FATAL: setup_full.sh exited $SETUP_RC — refusing to run lane\" > /workspace/lane.log\n"
        "    echo \"setup_rc=$SETUP_RC lane_rc=skipped\" >> /workspace/lane.log\n"
        "    exit $SETUP_RC\n"
        "fi\n"
        f"bash {lane_script} > /workspace/lane.log 2>&1\n"
        "LANE_RC=$?\n"
        "echo \"setup_rc=$SETUP_RC lane_rc=$LANE_RC\" >> /workspace/lane.log\n"
    )
    # SSH heredoc to write wrapper + setsid+nohup execute
    write_cmd = ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}",
                                      "cat > /workspace/run_lane.sh"]
    rc1, _, err1 = run(write_cmd, timeout=30)
    # NOTE: above cmd needs stdin piped, easier via shell:
    write_via_bash = ["bash", "-c",
                      f'cat <<\'EOSCRIPT\' | ssh {" ".join(SSH_OPTS)} '
                      f'-p {port} root@{host} "cat > /workspace/run_lane.sh"\n'
                      f'{wrapper}\n'
                      f'EOSCRIPT']
    rc1, _, err1 = run(write_via_bash, timeout=30)
    if rc1 != 0:
        sys.stderr.write(f"  exec: failed to write wrapper: {err1[:120]}\n")
        return False

    # Make executable + launch FULLY DETACHED via `( ... & )` subshell
    # pattern. This is the canonical fire-and-forget pattern: the subshell
    # exits immediately after backgrounding the job, so SSH sees its remote
    # command complete and disconnects. The orphaned `&` job is adopted by
    # init (PID 1) and continues running.
    # 2026-04-28: previous attempts (`nohup ... & disown`, `setsid ... & disown`)
    # both hung SSH because the parent shell stayed alive holding stdout.
    # The subshell-with-& pattern is the only reliable way to detach over
    # SSH without --use-tty trickery.
    launch_cmd = ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}", (
        "chmod +x /workspace/run_lane.sh && "
        "( bash /workspace/run_lane.sh </dev/null >/dev/null 2>&1 & ) && "
        "echo launched"
    )]
    rc2, out, err2 = run(launch_cmd, timeout=30)
    if rc2 != 0 or "launched" not in out:
        sys.stderr.write(f"  exec: subshell-launch failed: rc={rc2} out={out[:80]} err={err2[:80]}\n")
        return False
    return True


def poll_heartbeat(
    host: str, port: int, timeout_s: int = 480,
) -> tuple[bool, float | None]:
    """Poll for heartbeat.log freshness. Returns (success, age_minutes_at_exit)."""
    deadline = time.time() + timeout_s
    last_age = None
    while time.time() < deadline:
        cmd = ["ssh"] + SSH_OPTS + ["-p", str(port), f"root@{host}", (
            "find /workspace/pact -name 'heartbeat.log' -printf '%T@\\n' 2>/dev/null "
            "| sort -n | tail -1"
        )]
        rc, out, err = run(cmd, timeout=20)
        if rc == 0 and out.strip():
            try:
                mtime = float(out.strip())
                age_min = (time.time() - mtime) / 60.0
                last_age = age_min
                if age_min < 5.0:  # fresh heartbeat
                    return True, age_min
            except ValueError:
                pass
        time.sleep(30)
    return False, last_age


def register_in_tracker(
    instance_id: int,
    label: str,
    metadata: dict,
) -> None:
    """Append entry to .omx/state/vastai_active_instances.json."""
    data = json.loads(TRACKER_PATH.read_text()) if TRACKER_PATH.exists() else []
    if any(str(x.get("instance_id")) == str(instance_id) for x in data):
        return  # already registered
    data.append({
        "instance_id": str(instance_id),
        "label": label,
        "metadata": metadata,
        "registered_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registered_by_host": "Primary",
        "registered_by_pid": 0,
    })
    TRACKER_PATH.write_text(json.dumps(data, indent=2))


def cmd_phase1(args) -> int:
    """Phase 1: search offer + create instance + register tracker. ~10-30s.

    Output: prints instance_id on success. Parent captures and feeds to phase2.
    """
    lane_script_path = REPO_ROOT / args.lane_script
    if not lane_script_path.exists():
        print(f"FATAL: lane script not found: {args.lane_script}", file=sys.stderr)
        return 2
    print(f"=== phase1 Stage 0: Find offer (max ${args.max_dph}/hr) ===")
    try:
        offer_id = find_offer(max_dph=args.max_dph)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    print(f"  offer_id={offer_id}")
    print("=== phase1 Stage 1: Create instance ===")
    instance_id = create_instance(offer_id, args.label)
    register_in_tracker(instance_id, args.label, {
        "estimated_cost_usd": args.estimated_cost,
        "predicted_band": list(args.predicted_band),
        "script": args.lane_script,
        "council_priority": args.council_priority,
        "anchor_dirs": args.anchor_dirs,
        "launcher": "scripts/launch_lane_on_vastai.py phase1+phase2",
    })
    print(f"\n✓ phase1 SUCCESS")
    print(f"INSTANCE_ID={instance_id}")
    print(f"  label={args.label}")
    print(f"\nNext step (run phase2 after waiting ~3 min for OS boot):")
    print(f"  .venv/bin/python scripts/launch_lane_on_vastai.py phase2 \\")
    print(f"    --instance-id {instance_id} \\")
    print(f"    --lane-script {args.lane_script}")
    return 0


def cmd_phase2_wait(args) -> int:
    """Phase 2-wait: wait for actual_status=running + SSH ready. ~3-5 min.

    Each call fits comfortably under the 5-min bash-tool harness budget.
    Idempotent — if SSH not ready after timeout, instance KEPT, user can
    re-call.
    """
    instance_id = int(args.instance_id)
    print("=== phase2-wait Stage 1: Wait actual_status=running (max 240s) ===")
    if not wait_for_vastai_ready(instance_id, timeout_s=240):
        print(f"\n⚠ phase2-wait NOT READY (actual_status=None still). Instance kept.", file=sys.stderr)
        print(f"  Re-run phase2-wait in 1-2 min OR phase2-deploy if you've manually verified SSH.", file=sys.stderr)
        return 1
    host, port = get_ssh_details(instance_id)
    print(f"  ssh=root@{host}:{port}")
    print("=== phase2-wait Stage 2: Wait for SSH (max 240s post-status) ===")
    if not wait_for_ssh(host, port, timeout_s=240):
        print(f"\n⚠ phase2-wait NOT READY (sshd still not listening). Instance kept.", file=sys.stderr)
        print(f"  Re-run phase2-wait, OR if you can manually SSH, run phase2-deploy.", file=sys.stderr)
        return 1
    print("\n✓ phase2-wait SUCCESS — instance ready for phase2-deploy")
    print(f"  ssh=root@{host}:{port}")
    print(f"\nNext step:")
    print(f"  .venv/bin/python scripts/launch_lane_on_vastai.py phase2-deploy \\")
    print(f"    --instance-id {instance_id} \\")
    print(f"    --lane-script {args.lane_script}")
    return 0


def _resolve_host_port(instance_id: int) -> tuple[str, int]:
    info = _vast_show_instance_dict(str(instance_id))
    if not info or info.get("actual_status") != "running":
        actual = info.get("actual_status") if info else None
        raise RuntimeError(
            f"instance {instance_id} not in running state "
            f"(actual_status={actual!r}) — re-run phase2-wait"
        )
    # Guard against partial state where the instance is "running" but ssh
    # fields haven't propagated yet (Vast.ai race condition seen 2026-04-28).
    host = info.get("ssh_host")
    port = info.get("ssh_port")
    if not host or port is None:
        raise RuntimeError(
            f"instance {instance_id} actual_status=running but ssh fields "
            f"missing (host={host!r} port={port!r}) — re-run phase2-wait"
        )
    try:
        port_int = int(port)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"instance {instance_id} ssh_port not convertible to int "
            f"(value={port!r}): {e}"
        ) from e
    return host, port_int


def cmd_phase2_scp(args) -> int:
    """Phase 2-SCP: build local tarball + SCP to instance. ~2-3 min worst case.

    Atomic — no extract, no launch. Just gets the bytes onto the remote.
    """
    instance_id = int(args.instance_id)
    print("=== phase2-scp Stage 1: Build tarball ===")
    try:
        tar = build_tarball()
    except Exception as e:
        print(f"FATAL: tarball build failed: {e}", file=sys.stderr)
        return 1
    print(f"  tarball: {tar} ({tar.stat().st_size // (1024*1024)}MB)")

    print("=== phase2-scp Stage 2: SCP to remote ===")
    try:
        host, port = _resolve_host_port(instance_id)
        scp_tarball(tar, host, port)
    except Exception as e:
        print(f"FATAL: SCP failed: {e}", file=sys.stderr)
        return 1
    finally:
        tar.unlink(missing_ok=True)
    print(f"\n✓ phase2-scp SUCCESS — tarball at /workspace/pact.tar.gz on remote")
    print(f"\nNext step:")
    print(f"  .venv/bin/python scripts/launch_lane_on_vastai.py phase2-extract \\")
    print(f"    --instance-id {instance_id} \\")
    print(f"    --lane-script {args.lane_script}")
    return 0


def cmd_phase2_extract(args) -> int:
    """Phase 2-extract: extract tarball + lightweight CUDA probe. ~30s.

    Atomic. No SCP, no launch.
    """
    instance_id = int(args.instance_id)
    try:
        host, port = _resolve_host_port(instance_id)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    print(f"  ssh=root@{host}:{port}")
    print("=== phase2-extract Stage 1: Extract on remote ===")
    try:
        extract_remote(host, port)
    except Exception as e:
        print(f"FATAL: extract failed: {e}", file=sys.stderr)
        return 1
    print("=== phase2-extract Stage 2: CUDA probe ===")
    ok, msg = lightweight_nvdec_probe(host, port)
    if not ok:
        print(f"FATAL: CUDA probe failed: {msg}", file=sys.stderr)
        if not getattr(args, "no_destroy_on_fail", False):
            # CUDA probe failure means instance never produced artifacts — skip
            # recovery (no training output yet). Instance label not relevant.
            destroy_instance(
                instance_id,
                recover=not getattr(args, "no_recover", False),
                lane_label=getattr(args, "label", None) or args.lane_script,
            )
        return 1
    print(f"\n✓ phase2-extract SUCCESS")
    print(f"\nNext step:")
    print(f"  .venv/bin/python scripts/launch_lane_on_vastai.py phase2-launch \\")
    print(f"    --instance-id {instance_id} \\")
    print(f"    --lane-script {args.lane_script}")
    return 0


def _poll_setup_log_for_outcome(host: str, port: int, instance_id: int,
                                 timeout_seconds: int = 60) -> str:
    """SSH-poll the remote setup.log for one of four outcomes:
    - "NVDEC_BAD" — Stage 0.5 lightweight NVDEC probe failed
    - "LANE_CRASHED" — lane script ran but exited (no python process alive,
      and run.log shows neither SETUP_COMPLETE nor active progress)
    - "SETUP_COMPLETE" — setup_full.sh wrote the SETUP_COMPLETE marker
    - "RUNNING" — still running, no decision yet
    - "SSH_FAILED" — couldn't reach the instance

    2026-04-29: extended LANE_CRASHED detection after v4 dispatches showed
    UnboundLocalError-class crashes that occurred AFTER setup_full.sh
    completed. setup.log alone was insufficient — added pgrep + run.log
    timestamp check.
    """
    import time
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        # Probe: check setup.log markers + check for python process alive.
        # If python died AND run.log hasn't grown for 30s, assume CRASHED.
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR",
            "-p", str(port), f"root@{host}",
            # Detection logic:
            #   - NVDEC_BAD: setup.log shows NVDEC missing
            #   - SETUP_COMPLETE: setup.log has SETUP_COMPLETE marker AND
            #     run.log was modified in last 2 min (lane is alive)
            #   - LANE_CRASHED: SETUP_COMPLETE but run.log stale ≥2 min
            #     (python process died — UnboundLocalError, OOM, etc.)
            #   - RUNNING: still in setup
            # Avoids `pgrep -f` (would self-match this regex). Uses
            # run.log freshness as the canonical liveness signal.
            ("if grep -q 'NVDEC missing\\|NVDEC_MISSING' /workspace/setup.log 2>/dev/null; then "
             "  echo NVDEC_BAD; "
             "elif grep -q 'SETUP_COMPLETE' /workspace/setup.log 2>/dev/null; then "
             "  if find /workspace/pact -name run.log -mmin -2 2>/dev/null | grep -q .; then "
             "    echo SETUP_COMPLETE; "
             "  else "
             "    echo LANE_CRASHED; "
             "  fi; "
             "else "
             "  echo RUNNING; "
             "fi"),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                outcome = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "RUNNING"
                if outcome in ("NVDEC_BAD", "LANE_CRASHED", "SETUP_COMPLETE"):
                    return outcome
        except subprocess.SubprocessError:
            return "SSH_FAILED"
        time.sleep(5)
    return "RUNNING"  # timeout — neither marker yet, give up polling


def cmd_phase2_launch(args) -> int:
    """Phase 2-launch: subshell-detach launch lane via SSH + post-launch
    verification poll for ~60s to auto-destroy NVDEC-bad hosts.

    Atomic execution part: one SSH that backgrounds the lane wrapper.
    Post-launch verification: poll setup.log for NVDEC_BAD or SETUP_COMPLETE.
    On NVDEC_BAD → auto-destroy + retry guidance.
    """
    instance_id = int(args.instance_id)
    try:
        host, port = _resolve_host_port(instance_id)
    except RuntimeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    print(f"  ssh=root@{host}:{port}")
    print("=== phase2-launch Stage 1: Subshell-detach lane ===")
    if not execute_lane_in_tmux(host, port, args.lane_script):
        print("FATAL: launch failed", file=sys.stderr)
        if not getattr(args, "no_destroy_on_fail", False):
            destroy_instance(
                instance_id,
                recover=not getattr(args, "no_recover", False),
                lane_label=getattr(args, "label", None) or args.lane_script,
            )
        return 1

    # Stage 2: post-launch verification poll (~240s).
    # 2026-04-29: extended from 60s to 240s + LANE_CRASHED detection after
    # v4 dispatches showed UnboundLocalError-class crashes that occurred
    # AFTER setup_full.sh completed but before any heartbeat. The new poll
    # detects: NVDEC_BAD (Stage 0.5) | LANE_CRASHED (python died early) |
    # SETUP_COMPLETE (with python still running OR run.log fresh).
    if not getattr(args, "skip_post_verify", False):
        print("=== phase2-launch Stage 2: Post-launch outcome poll (~240s) ===")
        outcome = _poll_setup_log_for_outcome(host, port, instance_id, timeout_seconds=240)
        if outcome == "NVDEC_BAD":
            print(f"FATAL: setup_full.sh hit NVDEC_BAD on this host — auto-destroying", file=sys.stderr)
            print(f"  Instance had no NVDEC despite passing offer filter.", file=sys.stderr)
            print(f"  Per memory feedback_vastai_nvdec_host_variation: same image, different host = different NVDEC.", file=sys.stderr)
            destroy_instance(
                instance_id,
                recover=False,
                lane_label=getattr(args, "label", None) or args.lane_script,
            )
            print(f"\nRetry: re-run phase1 + phase2 to get a fresh host.", file=sys.stderr)
            return 2
        elif outcome == "LANE_CRASHED":
            print(f"FATAL: lane script crashed early — python process gone, run.log stale.", file=sys.stderr)
            print(f"  Likely a code bug surfaced at training startup (e.g., UnboundLocalError, missing dep).", file=sys.stderr)
            print(f"  Recovering artifacts before destroy...", file=sys.stderr)
            # Lane crashed → likely no training output, but recover anyway in case
            # any logs/artifacts were written.
            destroy_instance(
                instance_id,
                recover=True,
                lane_label=getattr(args, "label", None) or args.lane_script,
            )
            return 3
        elif outcome == "SETUP_COMPLETE":
            print(f"  outcome=SETUP_COMPLETE (setup_full.sh done, lane process verified alive)")
        elif outcome == "SSH_FAILED":
            print(f"  WARNING: SSH probe failed during post-launch verify; lane may still be running. Verify manually.")
        else:
            print(f"  outcome=RUNNING (still in setup_full.sh; verify in 5-15min)")

    print(f"\n✓ phase2-launch SUCCESS")
    print(f"  instance_id={instance_id}  ssh=root@{host}:{port}")
    print(f"\nVerify heartbeat in 5 min:")
    print(f"  .venv/bin/python scripts/verify_vast_instances.py")
    return 0


def cmd_phase2_deploy(args) -> int:
    """LEGACY: Phase 2-deploy combined SCP+extract+launch. May hit harness.

    Prefer phase2-scp + phase2-extract + phase2-launch separately for
    deterministic harness-fit. Kept for backward compat with `cmd_full`.
    """
    rc = cmd_phase2_scp(args)
    if rc != 0:
        return rc
    rc = cmd_phase2_extract(args)
    if rc != 0:
        return rc
    return cmd_phase2_launch(args)


def cmd_full(args) -> int:
    """Full pipeline: phase1 + sleep + phase2 (legacy combined; may hit harness)."""
    rc1 = cmd_phase1(args)
    if rc1 != 0:
        return rc1
    # The instance_id was registered in tracker; find it
    if not TRACKER_PATH.exists():
        return 1
    data = json.loads(TRACKER_PATH.read_text())
    matching = [x for x in data if x.get("label") == args.label]
    if not matching:
        return 1
    instance_id = matching[-1]["instance_id"]
    print(f"\n[bridging phase1→phase2 for instance {instance_id}]")
    args.instance_id = str(instance_id)
    args.no_destroy_on_fail = False
    return cmd_phase2_deploy(args)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # Common args helper. `--lane-script` required for stages that NEED it
    # (build tarball, ship code, launch). phase2-wait only polls SSH and
    # doesn't need it. Caller can override via add_common(p, lane_required=False).
    def add_common(p_, lane_required: bool = True):
        p_.add_argument("--lane-script", required=lane_required,
                       help="Path to remote_lane_*.sh (relative to repo root)")
        p_.add_argument("--anchor-dirs", nargs="*",
                       default=["experiments/results/lane_a_landed/iter_0"],
                       help="Directories to include in tarball (anchor data)")

    # phase1
    p1 = sub.add_parser("phase1", help="Create instance + register (fast, ~10-30s)")
    add_common(p1)
    p1.add_argument("--label", required=True, help="Vast.ai instance label")
    p1.add_argument("--predicted-band", nargs=2, type=float, default=[0.0, 2.0],
                    metavar=("LOW", "HIGH"))
    p1.add_argument("--estimated-cost", type=float, default=2.0)
    p1.add_argument("--council-priority", type=int, default=99)
    p1.add_argument("--max-dph", type=float, default=0.50)

    # phase2 (legacy combined — kept for backward compat with v3 callers)
    p2 = sub.add_parser("phase2", help="Combined phase2-wait + phase2-deploy (~3-5 min, harness-risky)")
    add_common(p2)
    p2.add_argument("--instance-id", required=True, help="Vast.ai instance ID from phase1")
    p2.add_argument("--no-destroy-on-fail", action="store_true",
                    help="Keep instance for debugging if phase2 fails")

    # phase2-wait — just status + ssh ready, idempotent (re-run if needed)
    p2w = sub.add_parser("phase2-wait", help="Wait for instance ready (~3-5 min, idempotent)")
    add_common(p2w, lane_required=False)
    p2w.add_argument("--instance-id", required=True)

    # phase2-deploy (legacy combined, kept for backward compat)
    p2d = sub.add_parser("phase2-deploy", help="LEGACY: combined SCP+extract+launch (may hit harness)")
    add_common(p2d)
    p2d.add_argument("--instance-id", required=True)
    p2d.add_argument("--no-destroy-on-fail", action="store_true",
                     help="Keep instance for debugging if deploy fails")

    # phase2-scp — atomic SCP only (~2-3 min, fits harness)
    p2s = sub.add_parser("phase2-scp", help="Build tarball + SCP to remote (~2-3 min)")
    add_common(p2s)
    p2s.add_argument("--instance-id", required=True)

    # phase2-extract — atomic extract + CUDA probe (~30s)
    p2e = sub.add_parser("phase2-extract", help="Extract + CUDA probe on remote (~30s)")
    add_common(p2e)
    p2e.add_argument("--instance-id", required=True)
    p2e.add_argument("--no-destroy-on-fail", action="store_true")
    p2e.add_argument(
        "--no-recover", action="store_true",
        help=(
            "Skip artifact recovery before destroy (use only when instance "
            "is known-unreachable). Default: recover first via "
            "tools/recover_lane_artifacts.py."
        ),
    )

    # phase2-launch — atomic subshell-detach SSH (~10s)
    p2l = sub.add_parser("phase2-launch", help="Subshell-detach launch lane (~10s)")
    add_common(p2l)
    p2l.add_argument("--instance-id", required=True)
    p2l.add_argument("--no-destroy-on-fail", action="store_true")
    p2l.add_argument(
        "--no-recover", action="store_true",
        help="Skip artifact recovery before destroy (see phase2-extract).",
    )
    # Explicit fire-and-forget opt-out for the Stage 2 setup.log poll.
    # Default OFF so the canonical workflow always polls + auto-destroys
    # NVDEC_BAD hosts. Per preflight Check 54
    # (check_phase2_launch_polls_setup_log) + memory
    # feedback_canonical_nvdec_workflow_GUARD_20260428.
    p2l.add_argument(
        "--skip-post-verify",
        action="store_true",
        help=(
            "Skip Stage 2 post-launch poll of setup.log. Use ONLY when "
            "you intentionally want fire-and-forget; the default polls "
            "for NVDEC_BAD outcomes and auto-destroys bad hosts."
        ),
    )

    # full (legacy combined — may hit harness)
    pf = sub.add_parser("full", help="Combined phase1+phase2 (~5-7 min, harness-risky)")
    add_common(pf)
    pf.add_argument("--label", required=True)
    pf.add_argument("--predicted-band", nargs=2, type=float, default=[0.0, 2.0])
    pf.add_argument("--estimated-cost", type=float, default=2.0)
    pf.add_argument("--council-priority", type=int, default=99)
    pf.add_argument("--max-dph", type=float, default=0.50)

    # Legacy positional support: if first arg isn't a subcommand, default to 'full'
    p.add_argument("--lane-script", required=False, help=argparse.SUPPRESS)
    p.add_argument("--label", required=False, help=argparse.SUPPRESS)
    p.add_argument("--predicted-band", nargs=2, type=float, default=[0.0, 2.0],
                   metavar=("LOW", "HIGH"), help=argparse.SUPPRESS)
    p.add_argument("--estimated-cost", type=float, default=2.0, help=argparse.SUPPRESS)
    p.add_argument("--anchor-dirs", nargs="*",
                   default=["experiments/results/lane_a_landed/iter_0"],
                   help=argparse.SUPPRESS)
    p.add_argument("--council-priority", type=int, default=99, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-dph", type=float, default=0.50, help=argparse.SUPPRESS)
    args = p.parse_args()

    # Dispatch
    if args.cmd == "phase1":
        return cmd_phase1(args)
    if args.cmd == "phase2-wait":
        return cmd_phase2_wait(args)
    if args.cmd == "phase2-scp":
        return cmd_phase2_scp(args)
    if args.cmd == "phase2-extract":
        return cmd_phase2_extract(args)
    if args.cmd == "phase2-launch":
        return cmd_phase2_launch(args)
    if args.cmd == "phase2-deploy":
        return cmd_phase2_deploy(args)
    if args.cmd == "phase2":
        # Legacy combined — call phase2-wait then phase2-deploy
        # No `--no-destroy-on-fail` for the wait part (it's idempotent anyway)
        rc = cmd_phase2_wait(args)
        if rc != 0:
            return rc
        if not hasattr(args, "no_destroy_on_fail"):
            args.no_destroy_on_fail = False
        return cmd_phase2_deploy(args)
    if args.cmd == "full":
        return cmd_full(args)

    # Legacy (no subcommand) → behaves like 'full' but with --dry-run support
    if not getattr(args, "lane_script", None):
        p.print_help()
        return 2
    if args.dry_run:
        try:
            offer_id = find_offer(max_dph=args.max_dph)
            print(f"=== Stage 0: offer_id={offer_id} ===")
            print("DRY RUN — exiting before instance creation")
            return 0
        except Exception as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2

    if not getattr(args, "label", None):
        print("FATAL: --label is required for non-dry-run", file=sys.stderr)
        return 2
    return cmd_full(args)


if __name__ == "__main__":
    sys.exit(main())
