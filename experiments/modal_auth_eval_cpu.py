"""Run canonical CPU contest auth eval on Modal Linux x86_64 container.

This is the CPU sister of ``experiments/modal_auth_eval.py``. It provides the
1:1 contest-compliant CPU substrate required by CLAUDE.md "Submission auth
eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".

The contest leaderboard ranks by `--device cpu` runs of `upstream/evaluate.py`
on the contest CI runner family (Ubuntu / x86_64). Running CPU eval locally
on Apple Silicon (M-series ARM) is not 1:1 (ARM SIMD vs x86_64 SSE/AVX
floating-point intrinsics differ in ways that can affect SegNet/PoseNet
output bytes). Modal's debian_slim base image is `linux/x86_64`, which
matches the contest CI runner family, so a Modal CPU container is the
canonical 1:1 CPU substrate.

The wrapper fails closed:

  - It refuses to run if the container is NOT Linux x86_64 (e.g. accidental
    aarch64 image).
  - It records `torch.cuda.is_available() == False` in the preflight, and
    refuses to proceed if a GPU is detected (mis-spec).
  - It tags results `[contest-CPU]` ONLY when `contest_auth_eval.py`'s own
    evidence-grade contract returns ``contest-CPU`` (which already requires
    Linux + x86_64 + 600 samples).
  - Results are NOT promotion-eligible (CPU axis) — adjudication still runs
    over the harvested ``contest_auth_eval.json``.

The score path remains the repository's canonical evaluator:

    archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu

Usage:
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run \
        experiments/modal_auth_eval_cpu.py \
        --archive experiments/results/public_pr_intake_full/.../archive.zip \
        --inflate-sh experiments/results/.../source/submissions/apogee/inflate.sh \
        --output-dir experiments/results/pr107_apogee_cpu_auth_eval_<ts>
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import modal

from tac.repo_io import json_text, read_json, sha256_file, write_json

APP_NAME = "comma-auth-eval-cpu"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
REMOTE_OUT = Path("/tmp/modal_auth_eval_cpu")
REQUIRED_SAMPLES = 600

app = modal.App(APP_NAME)


# Modal CPU container — debian_slim is linux/x86_64 by default. We pin a CPU
# torch wheel directly rather than the +cu124 wheel used by the CUDA path.
# nvidia-dali is intentionally OMITTED (it's CUDA-only and would either
# refuse to import or confuse the device probe).
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libglib2.0-0",
        "libgl1",
        "unzip",
        "xz-utils",
    )
    .pip_install(
        # CPU-only torch wheel from PyTorch's CPU index — avoids pulling cu124
        # libs we don't need and reduces image size + startup latency.
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "click",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        "brotli>=1.0",
        "constriction>=0.4,<0.5",
        "pyppmd>=1.3,<2.0",
        "cryptography>=41.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .run_commands(
        "curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg-master.tar.xz",
        "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-master",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
        "/usr/local/bin/ffmpeg-master -hide_banner -h filter=scale 2>&1 | grep -q in_primaries || (echo FATAL: ffmpeg-master lacks in_primaries; exit 1)",
        "/usr/local/bin/ffmpeg-master -encoders 2>&1 | grep -qi svtav1 || (echo FATAL: ffmpeg-master lacks libsvtav1; exit 1)",
        "rm /tmp/ffmpeg-master.tar.xz",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)


# Resource directories that must be present inside the Modal container for
# the canonical inflate.sh -> upstream/evaluate.py path to work. We mount
# them at the same relative path the local invocation would use.
eval_image = (
    base_image
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir("upstream", remote_path=str(REMOTE_REPO / "upstream"))
    .add_local_dir(
        "submissions/robust_current",
        remote_path=str(REMOTE_REPO / "submissions/robust_current"),
    )
    .add_local_dir(
        "experiments/public_runtime_adapters",
        remote_path=str(REMOTE_REPO / "experiments/public_runtime_adapters"),
    )
    # NB: ``experiments/results/public_pr_intake_full`` is ~17 GB total. Only
    # the per-PR source trees we exercise (PR107 apogee, PR102 hnerv_lc_v2)
    # are mounted; other intakes are intentionally excluded to keep image
    # builds fast. To run a different PR, add its intake source here.
    .add_local_dir(
        "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/source/submissions/apogee",
        remote_path=str(
            REMOTE_REPO
            / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/source/submissions/apogee"
        ),
    )
    .add_local_dir(
        "experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/source/submissions/hnerv_lc_v2_scale095_rplus1",
        remote_path=str(
            REMOTE_REPO
            / "experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto/source/submissions/hnerv_lc_v2_scale095_rplus1"
        ),
    )
    .add_local_file(
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_sha256_path = sha256_file


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json_text(payload).encode("utf-8")


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "archive"


def _probe_cpu_environment() -> dict[str, Any]:
    """Probe the Modal container for CPU-axis 1:1 contest compliance.

    Records platform, torch CPU/CUDA availability (we want CPU-only),
    cpu count, and per CLAUDE.md emphasises that this should NEVER show
    cuda_available=True (would mean we accidentally booked a GPU container).
    """
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import time

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_auth_eval_cpu.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "device_required": "cpu",
        "score_claim": False,
        "promotion_eligible": False,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }

    try:
        import torch

        preflight["torch_version"] = torch.__version__
        preflight["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        preflight["torch_cuda_available"] = bool(torch.cuda.is_available())
        preflight["torch_cuda_device_count"] = int(torch.cuda.device_count())
        # MPS lives on macOS only; verify it's not available (we're on Linux)
        try:
            preflight["torch_mps_available"] = bool(
                getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available()
            )
        except Exception:
            preflight["torch_mps_available"] = False
    except Exception as exc:  # pragma: no cover - remote diagnostic path
        preflight["torch_probe_error"] = repr(exc)
        preflight["torch_cuda_available"] = None

    # Confirm there's no GPU on this host (defensive — we book CPU-only above).
    nvidia_smi = shutil.which("nvidia-smi")
    preflight["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        try:
            preflight["nvidia_smi_query"] = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=15,
            ).strip()
        except Exception as exc:  # pragma: no cover - remote diagnostic path
            preflight["nvidia_smi_error"] = repr(exc)

    # Linux x86_64 attestation per CLAUDE.md "1:1 hardware-compliance" rule.
    is_linux = preflight["platform_system"] == "Linux"
    machine = str(preflight["platform_machine"] or "").lower()
    is_x86_64 = machine in {"x86_64", "amd64"}
    preflight["is_linux_x86_64"] = bool(is_linux and is_x86_64)

    return preflight


def _validate_contest_result(
    payload: dict[str, Any],
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
) -> list[str]:
    errors: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["contest_auth_eval.json missing provenance object"]

    if provenance.get("device") != "cpu":
        errors.append(f"provenance.device={provenance.get('device')!r}, expected 'cpu'")
    if provenance.get("cuda_available") is True:
        errors.append("provenance.cuda_available is True; this is a CPU-only run")
    if provenance.get("platform_system") != "Linux":
        errors.append(
            f"provenance.platform_system={provenance.get('platform_system')!r}, "
            "expected 'Linux' for 1:1 contest-CPU compliance"
        )
    machine = str(provenance.get("platform_machine") or "").lower()
    if machine not in {"x86_64", "amd64"}:
        errors.append(
            f"provenance.platform_machine={provenance.get('platform_machine')!r}, "
            "expected x86_64/amd64 for 1:1 contest-CPU compliance"
        )
    if provenance.get("archive_sha256") != expected_archive_sha256:
        errors.append(
            "provenance.archive_sha256 mismatch: "
            f"{provenance.get('archive_sha256')!r} != {expected_archive_sha256!r}"
        )
    if payload.get("archive_size_bytes") != expected_archive_size_bytes:
        errors.append(
            "archive_size_bytes mismatch: "
            f"{payload.get('archive_size_bytes')!r} != {expected_archive_size_bytes!r}"
        )
    if payload.get("n_samples") != REQUIRED_SAMPLES:
        errors.append(f"n_samples={payload.get('n_samples')!r}, expected {REQUIRED_SAMPLES}")

    score = payload.get("score_recomputed_from_components")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        errors.append("score_recomputed_from_components is missing or non-finite")

    # Affirm the evidence-grade emitted by contest_auth_eval.py itself
    # carries the contest-CPU axis (it already enforces Linux x86_64 + 600
    # samples — we re-check here for cross-validation).
    grade = payload.get("evidence_grade")
    if grade != "contest-CPU":
        errors.append(
            f"evidence_grade={grade!r}; expected 'contest-CPU' "
            "(contest_auth_eval.py determined this run is NOT 1:1 contest-CPU)"
        )
    if payload.get("score_axis") != "contest_cpu":
        errors.append(
            f"score_axis={payload.get('score_axis')!r}; expected 'contest_cpu'"
        )

    return errors


def _collect_artifacts(out_dir: Path, work_dir: Path) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for path in (
        out_dir / "modal_cpu_preflight.json",
        out_dir / "modal_cpu_auth_eval_validation.json",
        out_dir / "contest_auth_eval.stdout.log",
        out_dir / "contest_auth_eval.stderr.log",
        work_dir / "contest_auth_eval.json",
        work_dir / "provenance.json",
        work_dir / "report.txt",
    ):
        if path.is_file():
            artifacts[path.name] = path.read_bytes()
    # contest_auth_eval.py also writes contest_auth_eval.adjudicated.json
    # in some flows; capture if present for parity with CUDA wrapper.
    adjudicated = work_dir / "contest_auth_eval.adjudicated.json"
    if adjudicated.is_file():
        artifacts[adjudicated.name] = adjudicated.read_bytes()
    return artifacts


def _run_auth_eval_inner(
    *,
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    import time

    out_dir = REMOTE_OUT
    work_dir = out_dir / "eval_work"
    archive_path = out_dir / "archive.zip"
    uv_env = out_dir / "uv_project_env"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    preflight = _probe_cpu_environment()
    preflight.update(
        {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_sh_rel": inflate_sh_rel,
            "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu",
        }
    )
    write_json(out_dir / "modal_cpu_preflight.json", preflight)

    # Hard refuse if NOT Linux x86_64 — would corrupt the [contest-CPU] axis.
    if not preflight.get("is_linux_x86_64"):
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 3,
            "error": (
                "Modal container is not Linux x86_64 — refusing to claim "
                f"contest-CPU axis. system={preflight.get('platform_system')!r} "
                f"machine={preflight.get('platform_machine')!r}"
            ),
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}

    # Refuse to run if a GPU was somehow attached (mis-spec). The contest-CPU
    # axis must be exclusively CPU.
    if preflight.get("torch_cuda_available") is True:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 4,
            "error": (
                "torch.cuda.is_available() is True — this Modal container has a "
                "GPU attached. Refusing CPU eval on a GPU container."
            ),
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}

    archive_path.write_bytes(archive_bytes)
    observed_sha = _sha256_path(archive_path)
    observed_size = archive_path.stat().st_size
    if observed_sha != archive_sha256 or observed_size != archive_size_bytes:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 5,
            "error": "uploaded archive custody mismatch",
            "expected_archive_sha256": archive_sha256,
            "observed_archive_sha256": observed_sha,
            "expected_archive_size_bytes": archive_size_bytes,
            "observed_archive_size_bytes": observed_size,
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}

    inflate_sh_path = (REMOTE_REPO / inflate_sh_rel).resolve()
    remote_repo = REMOTE_REPO.resolve()
    if not str(inflate_sh_path).startswith(str(remote_repo) + "/"):
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 6,
            "error": f"inflate_sh path escapes remote repo: {inflate_sh_rel}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}
    if not inflate_sh_path.is_file():
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 7,
            "error": f"inflate_sh not found in Modal image: {inflate_sh_rel}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}

    cmd = [
        sys.executable,
        "-u",
        str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(inflate_sh_path),
        "--upstream-dir",
        str(REMOTE_REPO / "upstream"),
        "--video-names-file",
        str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
        "--device",
        "cpu",
        "--keep-work-dir",
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(int(inflate_timeout)),
        "--evaluate-timeout",
        str(int(evaluate_timeout)),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": str(uv_env),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        # Ensure inflate.sh's PYBIN check works — point at the modal image's
        # python. The PR102 adapter expects $REPO_ROOT/.venv/bin/python by
        # default; export PYTHON to the system python instead.
        "PYTHON": sys.executable,
        # Defensive: hide GPUs even if Modal accidentally exposes one.
        "CUDA_VISIBLE_DEVICES": "",
    }

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REMOTE_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(inflate_timeout) + int(evaluate_timeout) + 600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (out_dir / "contest_auth_eval.stdout.log").write_text(stdout)
        (out_dir / "contest_auth_eval.stderr.log").write_text(stderr)
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 124,
            "modal_elapsed_seconds": elapsed,
            "command": cmd,
            "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu",
            "error": "outer Modal contest_auth_eval timeout expired",
            "score_claim": False,
            "promotion_eligible": False,
            "adjudication_required": True,
            "allowed_use": ["debug", "no_score_claim", "no_promotion"],
        }
        (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}
    elapsed = time.monotonic() - started
    (out_dir / "contest_auth_eval.stdout.log").write_text(proc.stdout)
    (out_dir / "contest_auth_eval.stderr.log").write_text(proc.stderr)
    print(proc.stdout[-4096:] if proc.stdout else "")
    if proc.stderr:
        print(proc.stderr[-2048:], file=sys.stderr)

    result_json = work_dir / "contest_auth_eval.json"
    payload: dict[str, Any] | None = None
    validation_errors: list[str] = []
    if result_json.is_file():
        try:
            payload = read_json(result_json)
        except json.JSONDecodeError as exc:
            validation_errors.append(f"contest_auth_eval.json is not valid JSON: {exc}")
    else:
        validation_errors.append("contest_auth_eval.json was not produced")

    if payload is not None:
        validation_errors.extend(
            _validate_contest_result(
                payload,
                expected_archive_sha256=archive_sha256,
                expected_archive_size_bytes=archive_size_bytes,
            )
        )

    passed = proc.returncode == 0 and not validation_errors
    validation = {
        "schema_version": 1,
        "passed": passed,
        "returncode": proc.returncode if passed else (proc.returncode or 10),
        "modal_elapsed_seconds": elapsed,
        "command": cmd,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu",
        "expected_archive_sha256": archive_sha256,
        "expected_archive_size_bytes": archive_size_bytes,
        "inflate_sh_rel": inflate_sh_rel,
        "validation_errors": validation_errors,
        "score_claim": bool(passed),
        "promotion_eligible": False,  # CPU axis: not promotion-eligible
        "adjudication_required": True,
        "score_axis": "contest_cpu",
        "evidence_grade": "contest-CPU" if passed else None,
        "allowed_use": (
            ["public_leaderboard_reproduction", "cpu_cuda_drift_diagnosis"]
            if passed
            else ["debug", "no_score_claim", "no_promotion"]
        ),
    }
    if payload is not None:
        validation.update(
            {
                "final_score": payload.get("final_score"),
                "score_recomputed_from_components": payload.get(
                    "score_recomputed_from_components"
                ),
                "avg_posenet_dist": payload.get("avg_posenet_dist"),
                "avg_segnet_dist": payload.get("avg_segnet_dist"),
                "n_samples": payload.get("n_samples"),
                "archive_size_bytes": payload.get("archive_size_bytes"),
                "provenance_device": (payload.get("provenance") or {}).get("device")
                if isinstance(payload.get("provenance"), dict)
                else None,
                "provenance_platform_system": (payload.get("provenance") or {}).get(
                    "platform_system"
                )
                if isinstance(payload.get("provenance"), dict)
                else None,
                "provenance_platform_machine": (payload.get("provenance") or {}).get(
                    "platform_machine"
                )
                if isinstance(payload.get("provenance"), dict)
                else None,
                "evidence_grade_from_payload": payload.get("evidence_grade"),
                "score_axis_from_payload": payload.get("score_axis"),
            }
        )
    (out_dir / "modal_cpu_auth_eval_validation.json").write_bytes(_json_bytes(validation))

    return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}


# CPU-only Modal function. ``cpu`` parameter requests N logical cores; default
# 8 should be plenty for 600-sample CPU eval. Memory is generous (16 GiB) since
# upstream/evaluate.py loads SegNet (smp Unet B2) + PoseNet (FastViT-T12) + the
# 1200-frame uncompressed-raw and inflated-raw payloads. Timeout 9000s = 2.5h
# accounts for ~60-120 min eval + image start + safety margin.
@app.function(
    image=eval_image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=9000,
)
def run_auth_eval_cpu(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 5400,
) -> dict[str, Any]:
    """Run the canonical CPU auth eval on Modal Linux x86_64."""

    return _run_auth_eval_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
    )


@app.local_entrypoint()
def main(
    archive: str = "/tmp/modal_submission/archive.zip",
    output_dir: str = "",
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 5400,
) -> None:
    """Upload an archive and harvest Modal CPU auth-eval artifacts."""

    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise SystemExit(f"FATAL: archive not found: {archive_path}")

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    archive_size_bytes = len(archive_bytes)
    inflate_sh_path = Path(inflate_sh)
    if inflate_sh_path.is_absolute():
        try:
            inflate_sh_rel = str(inflate_sh_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError as exc:
            raise SystemExit(
                "FATAL: --inflate-sh must be relative to repo root or inside it: "
                f"{inflate_sh_path}"
            ) from exc
    else:
        inflate_sh_rel = str(inflate_sh_path)
    if ".." in Path(inflate_sh_rel).parts:
        raise SystemExit(
            f"FATAL: --inflate-sh must not contain parent traversal: {inflate_sh_rel}"
        )
    label = _safe_label(archive_path.stem)
    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path("experiments/results/modal_auth_eval_cpu") / f"{label}_{archive_sha256[:12]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    local_summary = {
        "schema_version": 1,
        "tool": "experiments/modal_auth_eval_cpu.py",
        "app": APP_NAME,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "inflate_sh": inflate_sh_rel,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu",
        "score_claim": False,
        "promotion_eligible": False,
        "adjudication_required": True,
        "score_axis": "contest_cpu",
    }
    write_json(out_dir / "modal_cpu_auth_eval_local_request.json", local_summary)

    print(
        f"Uploading {archive_size_bytes:,} bytes to Modal CPU container "
        f"(sha256={archive_sha256}) for [contest-CPU] auth eval..."
    )

    result = run_auth_eval_cpu.remote(
        archive_bytes,
        archive_sha256,
        archive_size_bytes,
        inflate_sh_rel,
        int(inflate_timeout),
        int(evaluate_timeout),
    )

    artifacts = result.pop("artifacts", {})
    for name, data in sorted(artifacts.items()):
        (out_dir / name).write_bytes(data)

    result["local_output_dir"] = str(out_dir)
    result["archive_path"] = str(archive_path)
    result["archive_sha256"] = archive_sha256
    result["archive_size_bytes"] = archive_size_bytes
    result["inflate_sh"] = inflate_sh_rel
    write_json(out_dir / "modal_cpu_auth_eval_result.json", result)

    print("=" * 60)
    if result.get("passed"):
        print("MODAL CPU AUTH EVAL PASSED PATH VALIDATION  [contest-CPU]")
        print(f"  Score recomputed: {result.get('score_recomputed_from_components')}")
        print(f"  PoseNet dist:     {result.get('avg_posenet_dist')}")
        print(f"  SegNet dist:      {result.get('avg_segnet_dist')}")
        print(f"  Archive bytes:    {result.get('archive_size_bytes')}")
        print(f"  Platform:         {result.get('provenance_platform_system')} "
              f"{result.get('provenance_platform_machine')}")
        print(f"  Evidence grade:   {result.get('evidence_grade_from_payload')}")
        print("  Promotion:        not eligible (CPU axis is leaderboard-only)")
    else:
        print("MODAL CPU AUTH EVAL FAILED CLOSED")
        print(f"  Error:            {result.get('error')}")
        print(f"  Validation:       {result.get('validation_errors')}")
    print(f"  Artifacts:        {out_dir}")
    print("=" * 60)

    if not result.get("passed"):
        raise SystemExit(int(result.get("returncode") or 1))
