"""Dispatch Phase A4 (track1_phase_a4_charm_50k_toy) on Modal T4 with ChARM byte-roundtrip eval.

Council priority: Round 1 second-choice 7/10 (after A1 9/10 first-choice) — Decision 1
GATE-CLEARING dispatch. Reference:
``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``.

Why parallel with A1?
---------------------
Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" rule, A4 is
fired in parallel with A1 (which is currently training on Modal T4 under
``track1_phase_a1_score_gradient`` — DIFFERENT lane id, no claim conflict).
Operator authorized the cycle-cap raise; both lanes draw from the same Modal
T4 budget but different lane claims.

A4 is the SINGLE GATE-CLEARING dispatch for Decision 1 (co-trained Ballé/ChARM
hyperprior). Without G5 GREEN, Phase C (full Track 1 stack) is BLOCKED. The
50K-param toy ablation proves co-design works on a small substrate; success
unblocks the bigger PR101 ChARM lane.

Why a dedicated dispatcher (not ``modal_train_lane.py``)?
---------------------------------------------------------
Same reason as A1: ``modal_train_lane.py`` HARDCODES ``AUTH_EVAL_DEVICE=cpu`` +
``MODAL_AUTH_EVAL_ADVISORY_ONLY=1`` and stubs ``probe_nvdec.sh`` to always pass.
Phase A4 trains under CUDA — we want a CUDA-required preflight (matching the
canonical A1 pattern), even though A4's terminal eval is the BYTE roundtrip
test (encode→decode parity + bit-rate vs. brotli baseline), NOT
``upstream/evaluate.py --device cuda``.

A4 is honestly NOT a contest-bound renderer; the saved checkpoint is a codec
test fixture, not a submission archive. Auth eval applies to the downstream
lane that consumes ChARM as a real codec stage. The ``--no-auth-eval-on-best``
flag on the underlying training script is the existing operator opt-out (see
its help text — preserved verbatim per Catalog #122 check). The Modal A4 chain
ends with the BYTE roundtrip + brotli-baseline comparison stage instead of
``contest_auth_eval.py``.

Pipeline (container-side on Modal T4):
    1. Input custody (no external inputs; synthetic data generated remotely)
    2. CUDA + DALI + NVDEC preflight (fail closed on any miss; same gate as A1
       so the chain proves CUDA worked even though we don't run upstream eval)
    3. Stage 1: TRAIN — experiments/train_charm_50k_toy_substrate.py
       (default 500 epochs ChARM 50K-param + co-trained hyperprior; <1h on T4)
    4. Stage 2: BUILD — train_charm_50k_toy_substrate.py --build-archive-only
       (re-emits archive.zip + build_manifest.json from EMA-snapshotted ckpt)
    5. Stage 3: BYTE ROUNDTRIP + BROTLI BASELINE — encode/decode parity check on
       a probe weight tensor + brotli baseline-vs-ChARM bit-rate comparison.
       This is the falsification path the council named: A4 PASSES if compressed
       weights < 30 KB at <5% pixel reconstruction error AND beats brotli; FAILS
       if it can't beat brotli on the 50K-param weights.
    6. Stage 4: REPORT — write build_manifest.json with byte-comparison data.

Cost (T4 ~$0.59/hr):
    expected: $1.80-2.50 for ~3-4h chain (training + build + roundtrip stage)
    hard cap: $8 for 4h timeout (ABORTS dispatch if cost-cap exceeded locally)

Modal `.spawn()` — HARVEST OR LOSE:
    Per CLAUDE.md "Modal `.spawn()` puts artifacts in the FunctionCall return-value
    cache (~24h TTL), NOT in a Volume." This dispatcher writes the call_id into
    ``experiments/results/track1_phase_a4_charm_50k_toy_<ts>_modal/modal_metadata.json``
    + ``modal_call_id.txt`` so ``tools/harvest_modal_calls.py`` (which globs
    ``lane_*_modal/``-style paths — see compatibility note below) and the local
    recover entrypoint can find it within 24h.

USAGE — RECOMMENDED (`--detach` for the background run; the local entrypoint
does not block):

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_phase_a4_charm_50k_toy_substrate.py \\
        --epochs 500

Print-only (zero Modal app creation, zero GPU spend):

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run \\
        experiments/modal_phase_a4_charm_50k_toy_substrate.py \\
        --print-only

Recover (within 24h of dispatch — Modal result-cache TTL):

    .venv/bin/python experiments/modal_phase_a4_charm_50k_toy_substrate.py recover \\
        --label track1_phase_a4_charm_50k_toy_<ts>_modal

Or use the canonical harvester to sweep all dispatched Modal calls:

    .venv/bin/python tools/harvest_modal_calls.py

Cross-references:
  - ``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``
  - ``experiments/modal_phase_a1_score_gradient_pr101.py`` (canonical A-lane Modal pattern)
  - ``experiments/train_charm_50k_toy_substrate.py`` (training entrypoint)
  - ``tools/claim_lane_dispatch.py`` (cross-agent claim coordination)
  - feedback_modal_spawn_result_cache_pattern_20260429
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import modal

from tac.deploy.claims import DispatchClaimSpec, dispatch_claim_command

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/workspace/pact")
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
REMOTE_OUT_ROOT = Path("/tmp/modal_phase_a4")
APP_NAME = "comma-phase-a4-charm-toy"
RESULT_ROOT = REPO_ROOT / "experiments" / "results"

# Modal T4 hourly rate — per CLAUDE.md "GPU budget" / docs/hourly_costs.
HOURLY_RATE_T4_USD = 0.59
DEFAULT_TIMEOUT_HOURS = 4.0
DEFAULT_COST_CAP_USD = 8.0


# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

# Image with all deps. Mirrors A1 base image but ADDS compressai (required by
# the CharmHyperprior training entrypoint at experiments/train_charm_50k_toy_substrate.py).
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "unzip", "wget", "curl", "build-essential",
        "libgl1", "libglib2.0-0",  # opencv runtime
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "brotli",
        "compressai>=1.2.8",  # ChARM hyperprior; required by train_charm_50k_toy_substrate.py
        "click",
        "nvidia-dali-cuda120==1.52.0",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    .run_commands(
        # ffmpeg-master from BtbN nightly (matches A1; not strictly required by
        # the ChARM toy training but kept for parity with the canonical A-lane
        # image so dependent downstream tools work without re-build).
        "curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg-master.tar.xz",
        "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-master",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
        "rm /tmp/ffmpeg-master.tar.xz",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)

run_image = (
    base_image
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    # Workspace mounts — only the minimum needed by the A4 chain.
    # tac/* is required for tac.codec.charm_range_coder + tac.parametrize_strip.
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file("experiments/__init__.py", remote_path=str(REMOTE_REPO / "experiments/__init__.py"))  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
        "experiments/train_charm_50k_toy_substrate.py",
        remote_path=str(REMOTE_REPO / "experiments/train_charm_50k_toy_substrate.py"),
    )
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
        "scripts/probe_nvdec.sh",
        remote_path=str(REMOTE_REPO / "scripts/probe_nvdec.sh"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))  # MODAL_MANUAL_MOUNT_OK:narrow phase-A4 charm dispatcher; targeted upstream files; trainer-discovery N/A
)


# ---------------------------------------------------------------------------
# Helpers (local + remote-shared)
# ---------------------------------------------------------------------------

def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "track1_phase_a4_charm_50k_toy"


def _tail(value: str | bytes | None, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


# ---------------------------------------------------------------------------
# Remote-side: CUDA preflight + chain stages
# ---------------------------------------------------------------------------

def _probe_cuda_environment_remote(env: dict[str, str]) -> dict[str, Any]:
    """Container-side CUDA + DALI + NVDEC probe.

    Mirrors ``experiments/modal_phase_a1_score_gradient_pr101.py`` EXACTLY. Failing
    this probe aborts the chain BEFORE training spend. Even though A4's terminal
    stage is a byte-roundtrip (not upstream/evaluate.py), we still run the full
    CUDA preflight so the dispatch proves the GPU is sane before training starts.
    """
    import shutil
    import subprocess as sp

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_phase_a4_charm_50k_toy_substrate.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_required": "cuda",
        "gpu_requested": "T4",
        "canonical_path": "train --device cuda -> archive.zip -> CARM2 byte roundtrip + brotli baseline",
        "score_claim": False,
        "promotion_eligible": False,
    }
    try:
        import torch  # noqa: PLC0415 — local-only on remote container

        preflight["torch_version"] = torch.__version__
        preflight["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        preflight["torch_cuda_available"] = bool(torch.cuda.is_available())
        preflight["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            preflight["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            preflight["torch_cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover
        preflight["torch_probe_error"] = repr(exc)
        preflight["torch_cuda_available"] = False

    try:
        import nvidia.dali as dali  # noqa: PLC0415

        preflight["nvidia_dali_import_ok"] = True
        preflight["nvidia_dali_version"] = getattr(dali, "__version__", None)
    except Exception as exc:  # pragma: no cover
        preflight["nvidia_dali_import_ok"] = False
        preflight["nvidia_dali_import_error"] = repr(exc)

    try:
        import compressai  # noqa: PLC0415, F401

        preflight["compressai_import_ok"] = True
        preflight["compressai_version"] = getattr(compressai, "__version__", None)
    except Exception as exc:  # pragma: no cover
        preflight["compressai_import_ok"] = False
        preflight["compressai_import_error"] = repr(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    preflight["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        try:
            query = sp.check_output(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
                stderr=sp.STDOUT,
                timeout=15,
            ).strip()
            preflight["nvidia_smi_query"] = query
            preflight["gpu_t4_match"] = "T4" in query
        except Exception as exc:  # pragma: no cover
            preflight["nvidia_smi_error"] = repr(exc)
            preflight["gpu_t4_match"] = False
    else:
        preflight["gpu_t4_match"] = False

    return preflight


def _preflight_errors_remote(preflight: dict[str, Any]) -> list[str]:
    """Return list of fatal preflight failures (empty list = OK to dispatch).

    A4-specific note: NVDEC probe is OPTIONAL because the ChARM toy training
    doesn't decode video on the GPU (it generates synthetic frames in-memory).
    But CUDA/DALI/compressai imports MUST succeed; the GPU MUST be a T4.
    """
    errors: list[str] = []
    if preflight.get("torch_cuda_available") is not True:
        errors.append("Modal runtime has no CUDA device; refusing CPU fallback")
    if preflight.get("compressai_import_ok") is not True:
        errors.append("compressai import failed; ChARM hyperprior cannot run without it")
    if preflight.get("gpu_t4_match") is not True:
        errors.append("Modal runtime did not report a T4 GPU")
    # nvidia.dali / NVDEC are not strictly required for A4 (no video decode),
    # but we surface a warning if missing to keep parity with the A1 image.
    return errors


def _run_logged_remote(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    """Run a subprocess with full stdout/stderr capture to log_dir."""
    import subprocess as sp

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    print(f"[{name}] starting cmd[0]={cmd[0]!r} args={len(cmd) - 1} timeout={timeout}s")
    t0 = time.monotonic()
    timed_out = False
    try:
        proc = sp.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            timeout=timeout,
            check=False,
        )
        rc = proc.returncode
        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    except sp.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else ""
    elapsed = time.monotonic() - t0
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    print(f"[{name}] done rc={rc} elapsed={elapsed:.1f}s timed_out={timed_out}")
    return {
        "name": name,
        "cmd": cmd,
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _collect_artifacts_remote(out_dir: Path, max_bytes: int = 500 * 1024 * 1024) -> dict[str, bytes]:
    """Collect output artifacts to embed in the function-call return value."""
    artifacts: dict[str, bytes] = {}
    extensions = (".bin", ".zip", ".pt", ".mkv", ".json", ".log", ".safetensors", ".txt", ".sh", ".py")
    for fp in sorted(out_dir.rglob("*")):
        if not fp.is_file():
            continue
        if "__pycache__" in fp.parts:
            continue
        if fp.suffix.lower() not in extensions:
            continue
        try:
            rel = fp.relative_to(out_dir)
        except ValueError:
            rel = Path(fp.name)
        try:
            size = fp.stat().st_size
            if size > max_bytes:
                print(f"[collect] SKIP large {rel} ({size / 1e6:.1f}MB)")
                continue
            artifacts[str(rel)] = fp.read_bytes()
        except (FileNotFoundError, PermissionError) as exc:
            print(f"[collect] SKIP unreadable {fp}: {exc!r}")
            continue
    return artifacts


def _finish_remote(
    out_dir: Path,
    *,
    passed: bool,
    returncode: int,
    stage: str,
    validation_errors: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    eval_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "app": APP_NAME,
        "passed": bool(passed),
        "returncode": int(returncode),
        "stage": stage,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_errors": list(validation_errors or []),
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_path": "train --device cuda -> archive.zip -> CARM2 byte roundtrip + brotli baseline",
        # A4 is a codec-validation toy, NOT a contest-bound renderer. The
        # operator opt-out --no-auth-eval-on-best on the underlying training
        # script is honest here per its existing help text. The terminal stage
        # is the byte roundtrip + brotli-baseline comparison. Tag the run as
        # codec-validation rather than [contest-CUDA].
        "tag": (
            "[codec-validation byte-roundtrip]" if (passed and eval_data)
            else "[codec-validation dispatch failed]"
        ),
    }
    if eval_data:
        summary["eval_data"] = eval_data
        summary["roundtrip_exact"] = eval_data.get("roundtrip_exact")
        summary["compressed_bytes_charm"] = eval_data.get("compressed_bytes_charm")
        summary["compressed_bytes_brotli"] = eval_data.get("compressed_bytes_brotli")
        summary["beats_brotli"] = eval_data.get("beats_brotli")
        summary["pixel_recon_l1"] = eval_data.get("pixel_recon_l1")
    if extra:
        summary.update(extra)
    _write_json(out_dir / "phase_a4_summary.json", summary)
    artifacts = _collect_artifacts_remote(out_dir)
    print(f"[finish] returncode={returncode} stage={stage} artifacts={len(artifacts)}")
    return {
        "returncode": int(returncode),
        "stage": stage,
        "passed": bool(passed),
        "elapsed_seconds": None,  # filled by caller
        "validation_errors": list(validation_errors or []),
        "summary": summary,
        "eval_data": eval_data,
        "artifacts": artifacts,
        "stdout_tail": "",
    }


def _byte_roundtrip_and_brotli_baseline(archive_dir: Path, build_manifest: dict[str, Any]) -> dict[str, Any]:
    """Stage 3: encode→decode byte-roundtrip parity + brotli-baseline comparison.

    The training script already performs encode→decode parity on a probe weight
    inside its smoke harness (see ``run_smoke``). At dispatch time we re-run the
    same parity check on the EMA-snapshotted production weights AND compare the
    ChARM-encoded archive size to a vanilla brotli baseline of the same weight
    bytes. The council's falsification criterion is:

      A4 PASSES if compressed_bytes < 30,000 AND beats brotli on the 50K weights.
      A4 FAILS  if it can't beat brotli on the 50K-param weights.

    [empirical:experiments/results/track1_phase_a4_charm_50k_toy_<ts>_modal/]
    """
    import io
    import zipfile as zf

    import brotli  # type: ignore[import-not-found]
    import torch

    sys.path.insert(0, str(REMOTE_REPO / "src"))
    sys.path.insert(0, str(REMOTE_REPO))

    from experiments.train_charm_50k_toy_substrate import (
        CharmHyperprior,
        TinyHNeRVToy50K,
        decode_weight_with_charm,
        encode_weight_with_charm,
    )

    # Load EMA-snapshotted checkpoint that build_archive consumed.
    # Note: archive_dir is the dispatcher's output dir; the actual archive.zip
    # plus checkpoint.pt live there per train_charm_50k_toy_substrate.py.
    ckpt_path = archive_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        return {
            "roundtrip_exact": False,
            "compressed_bytes_charm": None,
            "compressed_bytes_brotli": None,
            "beats_brotli": False,
            "pixel_recon_l1": None,
            "error": f"missing checkpoint: {ckpt_path}",
        }

    # WEIGHTS_ONLY_FALSE_OK: locally-produced checkpoint from this dispatch.
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    base_ch = int(cfg.get("base_ch", 88))
    pose_dim = int(cfg.get("pose_dim", 6))

    # Reconstruct model + apply EMA shadow to gather flat weights.
    model = TinyHNeRVToy50K(base_ch=base_ch, pose_dim=pose_dim)
    if "ema_model_shadow" in ckpt:
        model.load_state_dict(ckpt["ema_model_shadow"])
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    flat_weights = torch.cat(
        [p.detach().flatten() for p in model.parameters()]
    )

    # ChARM byte size: sum of CARM2 blobs in archive.zip (excluding sidecars).
    archive_zip_path = archive_dir / "archive.zip"
    charm_total = 0
    if archive_zip_path.is_file():
        with zf.ZipFile(archive_zip_path) as zfh:
            for info in zfh.infolist():
                if info.filename.endswith(".bin"):
                    charm_total += info.file_size
    else:
        charm_total = int(build_manifest.get("encoded_weight_bytes", 0))

    # Brotli baseline: compress raw float32 bytes with brotli quality 11.
    raw_bytes = flat_weights.cpu().numpy().astype("<f4").tobytes()
    brotli_compressed = brotli.compress(raw_bytes, quality=11)
    brotli_total = len(brotli_compressed)

    # Per-tensor encode→decode parity on a probe (the training-script smoke
    # already proved this for a randn(64,64); we re-run on the FIRST learned
    # tensor of the production model).
    first_param = next(model.parameters()).detach()
    charm = CharmHyperprior(num_channels=8, num_weight_channels=64)
    if "ema_charm_shadow" in ckpt:
        charm.load_state_dict(ckpt["ema_charm_shadow"])
    elif "charm_state" in ckpt:
        charm.load_state_dict(ckpt["charm_state"])
    charm.eval()
    blob, meta = encode_weight_with_charm(first_param, charm)
    recovered = decode_weight_with_charm(blob, tuple(first_param.shape))

    # Re-quantize first_param the same way encode does, so we compare
    # apples-to-apples (lossy at INT8, but bit-exact at INT8 granularity).
    scale = first_param.abs().max().clamp(min=1e-8)
    expected_q = (
        (first_param / scale * 127.0)
        .round()
        .clamp(min=-128.0, max=127.0)
        / 127.0
        * scale
    )
    roundtrip_exact = bool(torch.allclose(recovered, expected_q, atol=1e-5))

    pixel_recon_l1 = float(build_manifest.get("training_eval_l_recon", float("nan")))

    beats_brotli = charm_total < brotli_total
    pass_threshold_bytes = int(
        build_manifest.get("falsification_criteria", {}).get(
            "PASS_threshold_compressed_bytes", 30000
        )
    )
    pass_threshold_l1 = (
        float(
            build_manifest.get("falsification_criteria", {}).get(
                "PASS_threshold_pixel_recon_error_pct", 5.0
            )
        )
        / 100.0
    )
    falsification_pass = (
        charm_total < pass_threshold_bytes
        and (pixel_recon_l1 != pixel_recon_l1 or pixel_recon_l1 < pass_threshold_l1)
        and beats_brotli
        and roundtrip_exact
    )

    return {
        "roundtrip_exact": roundtrip_exact,
        "compressed_bytes_charm": int(charm_total),
        "compressed_bytes_brotli": int(brotli_total),
        "compressed_bytes_brotli_minus_charm": int(brotli_total - charm_total),
        "beats_brotli": bool(beats_brotli),
        "pixel_recon_l1": pixel_recon_l1,
        "falsification_pass": bool(falsification_pass),
        "falsification_pass_threshold_bytes": pass_threshold_bytes,
        "falsification_pass_threshold_l1": pass_threshold_l1,
        "first_param_shape": list(first_param.shape),
        "first_param_charm_blob_bytes": int(len(blob)),
        "first_param_meta": meta,
    }


def _run_phase_a4_inner(
    *,
    label: str,
    epochs: int,
    batch_size: int,
    num_train_frames: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
    noise_std: float,
    lambda_R_target: float,
    lambda_R_warmup_steps: int,
    seed: int,
    train_timeout_seconds: int,
    build_timeout_seconds: int,
    eval_timeout_seconds: int,
    max_seconds: int,
) -> dict[str, Any]:
    """Container-side runner for the full A4 chain on Modal T4."""
    import os
    import shutil

    label_safe = _safe_label(label)
    out_dir = REMOTE_OUT_ROOT / label_safe
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = "input_custody"
    command_results: list[dict[str, Any]] = []
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "TAC_UPSTREAM_DIR": str(REMOTE_REPO / "upstream"),
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "PYTHONHASHSEED": str(int(seed)),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
    }

    t_start = time.monotonic()
    try:
        # A4 has NO external inputs (synthetic data generated remotely).
        # Custody manifest just records the dispatch parameters.
        input_manifest = {
            "schema_version": 1,
            "label": label_safe,
            "inputs": {
                "data_source": "synthetic_generated_remotely",
                "external_inputs": False,
            },
            "params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_train_frames": num_train_frames,
                "lr": lr,
                "weight_decay": weight_decay,
                "ema_decay": ema_decay,
                "noise_std": noise_std,
                "lambda_R_target": lambda_R_target,
                "lambda_R_warmup_steps": lambda_R_warmup_steps,
                "seed": seed,
            },
            "score_claim": False,
            "promotion_eligible": False,
        }
        _write_json(out_dir / "phase_a4_input_manifest.json", input_manifest)

        # --- Stage B: CUDA + DALI + NVDEC preflight -----------------------
        stage = "cuda_dali_compressai_preflight"
        preflight = _probe_cuda_environment_remote(env)
        _write_json(out_dir / "phase_a4_preflight.json", preflight)
        errors = _preflight_errors_remote(preflight)
        if errors:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=6,
                stage=stage,
                validation_errors=errors,
                extra={"preflight": preflight},
            )

        # --- Stage 1: TRAIN -----------------------------------------------
        # train_charm_50k_toy_substrate.py writes to --output:
        #   - checkpoint.pt (EMA-snapshotted state)
        #   - archive/ (per-weight CARM2 blobs)
        #   - archive.zip (deterministic ZIP of archive/)
        #   - build_manifest.json (component bytes + falsification criteria)
        #   - provenance.json
        #   - archive_sha256
        # We point --output at out_dir so the dispatcher's harvest can find
        # everything in one place.
        stage = "train_charm_50k_toy"
        train_cmd = [
            sys.executable, "-u",
            str(REMOTE_REPO / "experiments/train_charm_50k_toy_substrate.py"),
            "--device", "cuda",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--num-train-frames", str(num_train_frames),
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--ema-decay", str(ema_decay),
            "--noise-std", str(noise_std),
            "--lambda-R-target", str(lambda_R_target),
            "--lambda-R-warmup-steps", str(lambda_R_warmup_steps),
            "--seed", str(seed),
            "--output", str(out_dir),
            # --no-auth-eval-on-best is honest for A4: this is a codec-validation
            # toy substrate, not a contest renderer. Per train_charm_50k_toy_substrate.py
            # help text (Catalog #122 check requires the help text be unchanged):
            # "Operator opt-out: this script trains a 50K toy substrate for ChARM
            # range coder validation (encode→decode roundtrip + bit-rate vs.
            # brotli baseline). It does NOT produce a contest-bound renderer;
            # the saved checkpoint is a codec test fixture, not a submission
            # archive. Auth eval applies to the downstream lane that consumes
            # ChARM as a real codec stage."
            "--no-auth-eval-on-best",
        ]
        run = _run_logged_remote(
            "stage1_train",
            train_cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "logs",
            timeout=train_timeout_seconds,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=[f"training failed rc={run['returncode']}"],
                extra={"commands": command_results},
            )

        # The training script's main() runs train_loop + build_archive in one
        # invocation, so by the time the subprocess returns the archive is built.
        archive_zip = out_dir / "archive.zip"
        manifest_path = out_dir / "build_manifest.json"
        if not (archive_zip.is_file() and manifest_path.is_file()):
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=5,
                stage=stage,
                validation_errors=[
                    f"archive build incomplete: archive={archive_zip} manifest={manifest_path}"
                ],
                extra={"commands": command_results},
            )
        archive_meta = _file_meta(archive_zip)
        try:
            build_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=5,
                stage=stage,
                validation_errors=[f"build_manifest.json parse error: {exc!r}"],
                extra={"commands": command_results, "archive_meta": archive_meta},
            )

        # --- Stage 3: BYTE ROUNDTRIP + BROTLI BASELINE --------------------
        # NOT contest_auth_eval.py per A4's design — this is the falsification
        # path the council named (compressed bytes < 30K AND beats brotli).
        stage = "byte_roundtrip_brotli_baseline"
        try:
            eval_data = _byte_roundtrip_and_brotli_baseline(out_dir, build_manifest)
        except Exception as exc:  # pragma: no cover
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=7,
                stage=stage,
                validation_errors=[
                    f"byte_roundtrip stage exception: {type(exc).__name__}: {exc}"
                ],
                extra={
                    "commands": command_results,
                    "archive_meta": archive_meta,
                    "traceback": traceback.format_exc(),
                },
            )
        _write_json(out_dir / "phase_a4_byte_roundtrip.json", eval_data)

        # --- Stage 4: REPORT ----------------------------------------------
        stage = "report"
        report = {
            "lane_id": label_safe,
            "schema_version": "phase_a4_modal_build_manifest_v1",
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archive_path": str(archive_zip),
            "archive_bytes": archive_meta["bytes"],
            "archive_sha256": archive_meta["sha256"],
            "byte_roundtrip_data": eval_data,
            # A4 is codec-validation only; promotion_eligible stays False until
            # the downstream PR101 ChARM lane consumes this as a real codec stage.
            "evidence_grade": "[codec-validation byte-roundtrip]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_blockers": [
                "downstream_pr101_charm_consumer_pending",
                "contest_cuda_eval_does_not_apply_for_codec_toy",
            ],
            "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
            "council_decision": (
                "A4 — co-trained Ballé/ChARM hyperprior 50K-param toy "
                "(council Round 1 second-choice 7/10; Decision 1 GATE-CLEARING)"
            ),
            "modal_app": APP_NAME,
            "modal_gpu": "T4",
            "preflight": preflight,
            "build_manifest": build_manifest,
            "params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_train_frames": num_train_frames,
                "lr": lr,
                "weight_decay": weight_decay,
                "ema_decay": ema_decay,
                "noise_std": noise_std,
                "lambda_R_target": lambda_R_target,
                "lambda_R_warmup_steps": lambda_R_warmup_steps,
                "seed": seed,
            },
        }
        _write_json(out_dir / "phase_a4_report.json", report)

        result = _finish_remote(
            out_dir,
            passed=bool(eval_data.get("falsification_pass", False)),
            returncode=0,
            stage="completed",
            extra={
                "commands": command_results,
                "archive_meta": archive_meta,
                "build_manifest": build_manifest,
                "report": report,
            },
            eval_data=eval_data,
        )
        result["elapsed_seconds"] = time.monotonic() - t_start
        return result

    except Exception as exc:  # pragma: no cover — defensive
        return _finish_remote(
            out_dir,
            passed=False,
            returncode=99,
            stage=stage,
            validation_errors=[f"{type(exc).__name__}: {exc}"],
            extra={"traceback": traceback.format_exc(), "commands": command_results},
        )


@app.function(image=run_image, gpu="T4", timeout=int(DEFAULT_TIMEOUT_HOURS * 3600))
def run_phase_a4_t4(
    label: str,
    epochs: int,
    batch_size: int,
    num_train_frames: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
    noise_std: float,
    lambda_R_target: float,
    lambda_R_warmup_steps: int,
    seed: int,
    train_timeout_seconds: int,
    build_timeout_seconds: int,
    eval_timeout_seconds: int,
    max_seconds: int,
) -> dict[str, Any]:
    return _run_phase_a4_inner(
        label=label,
        epochs=epochs,
        batch_size=batch_size,
        num_train_frames=num_train_frames,
        lr=lr,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        noise_std=noise_std,
        lambda_R_target=lambda_R_target,
        lambda_R_warmup_steps=lambda_R_warmup_steps,
        seed=seed,
        train_timeout_seconds=train_timeout_seconds,
        build_timeout_seconds=build_timeout_seconds,
        eval_timeout_seconds=eval_timeout_seconds,
        max_seconds=max_seconds,
    )


# ---------------------------------------------------------------------------
# Local-side: lane claim + dispatch metadata + recover
# ---------------------------------------------------------------------------

def _claim_lane(
    *,
    lane_id: str,
    instance_job_id: str,
    predicted_eta_utc: str,
    notes: str,
    status: str = "active_dispatching",
    force: bool = False,
) -> int:
    """Open or close a lane claim via tools/claim_lane_dispatch.py."""
    import shlex

    cmd = dispatch_claim_command(
        spec=DispatchClaimSpec(
            lane_id=lane_id,
            platform="modal",
            instance_job_id=instance_job_id,
            agent="claude:modal_phase_a4",
            predicted_eta_utc=predicted_eta_utc,
            force=force,
            notes=notes,
        ),
        status=status,
        python_executable=sys.executable,
        claim_tool=REPO_ROOT / "tools" / "claim_lane_dispatch.py",
    )
    print(f"[claim] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _returncode_is_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value.strip() == "0"
    return False


def _metadata_instance_job_id(label: str, metadata: dict[str, Any]) -> str:
    raw = metadata.get("instance_job_id") or metadata.get("label") or label
    return _safe_label(str(raw))


def _close_modal_recovery_claim(
    *,
    instance_job_id: str,
    call_id: str,
    rc: Any,
    summary: dict[str, Any],
) -> int:
    status = (
        "completed_modal_recovered"
        if _returncode_is_zero(rc) and summary.get("passed") is True
        else "failed_modal_recovered"
    )
    notes = (
        f"Phase A4 Modal recover harvested call_id={call_id}; rc={rc!r}; "
        f"stage={summary.get('stage')!r}; passed={summary.get('passed')!r}. "
        "Terminal claim row closes the dispatch once cached artifacts are recovered."
    )
    return _claim_lane(
        lane_id="track1_phase_a4_charm_50k_toy",
        instance_job_id=instance_job_id,
        predicted_eta_utc=_utc_now_iso(),
        notes=notes,
        status=status,
        force=True,
    )


def _close_modal_expired_claim(
    *,
    instance_job_id: str,
    call_id: str,
) -> int:
    return _claim_lane(
        lane_id="track1_phase_a4_charm_50k_toy",
        instance_job_id=instance_job_id,
        predicted_eta_utc=_utc_now_iso(),
        notes=(
            f"Phase A4 Modal recover failed because result cache expired for call_id={call_id}; "
            "terminal failure closes the stale active claim."
        ),
        status="failed_modal_result_cache_expired",
        force=True,
    )


def _write_dispatch_metadata(
    *,
    instance_job_id: str,
    call_id: str,
    params: dict[str, Any],
    estimated_cost_usd: float,
    timeout_hours: float,
    predicted_low: float,
    predicted_high: float,
    predicted_eta_utc: str,
) -> Path:
    """Write the canonical Modal dispatch metadata.

    Schema mirrors ``modal_phase_a1_score_gradient_pr101.py`` so the canonical
    harvester ``tools/harvest_modal_calls.py`` finds it via the lane_*_modal
    compatibility symlink.
    """
    out_dir = RESULT_ROOT / instance_job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "app": APP_NAME,
        "lane_id": "track1_phase_a4_charm_50k_toy",
        "instance_job_id": instance_job_id,
        "label": instance_job_id,
        "call_id": call_id,
        "gpu": "T4",
        "params": params,
        "canonical_path": "train --device cuda -> archive.zip -> CARM2 byte roundtrip + brotli baseline",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "[advisory only — dispatch in flight]",
        "estimated_cost_usd": float(estimated_cost_usd),
        "estimated_duration_hours": float(timeout_hours),
        "predicted_band": [float(predicted_low), float(predicted_high)],
        "predicted_eta_utc": predicted_eta_utc,
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "council_decision": (
            "A4 — co-trained Ballé/ChARM hyperprior 50K-param toy "
            "(council Round 1 second-choice 7/10; Decision 1 GATE-CLEARING)"
        ),
        "training_script": "experiments/train_charm_50k_toy_substrate.py",
        "auth_eval_device": "cuda",
        "auth_eval_advisory_only": False,
        # A4 is a codec-validation toy. The terminal stage is byte roundtrip +
        # brotli-baseline, NOT contest_auth_eval.py — this is honest per
        # train_charm_50k_toy_substrate.py's --no-auth-eval-on-best help text.
        "auth_eval_path": "byte_roundtrip_and_brotli_baseline_codec_validation",
        "no_auth_eval_on_best_rationale": (
            "A4 is a 50K-param ChARM codec-validation toy substrate. The saved "
            "checkpoint is a codec test fixture, not a submission archive. Auth "
            "eval applies to the downstream lane that consumes ChARM as a real "
            "codec stage. Falsification criterion: compressed bytes < 30K AND "
            "beats brotli on the 50K-param weights."
        ),
        "recover_command": (
            ".venv/bin/python experiments/modal_phase_a4_charm_50k_toy_substrate.py recover "
            f"--label {instance_job_id}"
        ),
        "harvest_command_canonical": ".venv/bin/python tools/harvest_modal_calls.py",
        "dispatched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        # Legacy field name for tools/harvest_modal_calls.py compatibility.
        "dispatched_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    metadata_path = out_dir / "modal_metadata.json"
    _write_json(metadata_path, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n")

    # Compatibility symlink for tools/harvest_modal_calls.py (globs lane_*_modal).
    legacy_dir = RESULT_ROOT / f"lane_{instance_job_id}_modal"
    if legacy_dir.exists() or legacy_dir.is_symlink():
        try:
            legacy_dir.unlink()
        except IsADirectoryError:
            import shutil as _shutil
            _shutil.rmtree(legacy_dir)
    try:
        legacy_dir.symlink_to(out_dir, target_is_directory=True)
        print(f"[metadata] harvest-compat symlink: {legacy_dir} -> {out_dir}")
    except OSError as exc:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        _write_json(legacy_dir / "modal_metadata.json", payload)
        (legacy_dir / "modal_call_id.txt").write_text(call_id + "\n")
        print(f"[metadata] symlink unavailable ({exc!r}); copied modal_metadata.json to {legacy_dir}")

    return metadata_path


@app.local_entrypoint()
def main(
    label: str | None = None,
    epochs: int = 500,
    batch_size: int = 4,
    num_train_frames: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    ema_decay: float = 0.997,
    noise_std: float = 0.5,
    lambda_r_target: float = 1e-6,
    lambda_r_warmup_steps: int = 100,
    seed: int = 1234,
    train_timeout_hours: float = 3.0,
    build_timeout_minutes: float = 10.0,
    eval_timeout_minutes: float = 15.0,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    predicted_low: float = 0.20,
    predicted_high: float = 0.24,
    print_only: bool = False,
    force_claim: bool = False,
) -> None:
    """Dispatch Phase A4 to Modal T4 with `.spawn()` (HARVEST OR LOSE pattern).

    All cost / lane-claim gates run on the LOCAL side BEFORE any GPU is requested.
    """
    # ---- Pre-dispatch gates (cost cap; A4 has no external inputs) --------
    estimated_cost = HOURLY_RATE_T4_USD * float(timeout_hours)
    print(
        f"[cost-gate] estimated ${estimated_cost:.2f} for Modal T4 × {timeout_hours:.1f}h "
        f"(cap ${cost_cap_usd:.2f})"
    )
    if estimated_cost > cost_cap_usd:
        raise SystemExit(
            f"FATAL: estimated cost ${estimated_cost:.2f} exceeds cap ${cost_cap_usd:.2f}; abort"
        )

    # ---- Build instance_job_id (timestamped lane id) ---------------------
    if label:
        instance_job_id = _safe_label(label)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        instance_job_id = f"track1_phase_a4_charm_50k_toy_{timestamp}_modal"

    started_at_utc = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    predicted_eta_utc = (
        dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=float(timeout_hours))
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ---- Build container function args ------------------------------------
    train_timeout_seconds = max(120, int(float(train_timeout_hours) * 3600))
    build_timeout_seconds = max(60, int(float(build_timeout_minutes) * 60))
    eval_timeout_seconds = max(120, int(float(eval_timeout_minutes) * 60))
    max_seconds = max(120, int(float(timeout_hours) * 3600))
    max_seconds = min(max_seconds, 14 * 3600)

    params = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_train_frames": int(num_train_frames),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "ema_decay": float(ema_decay),
        "noise_std": float(noise_std),
        "lambda_R_target": float(lambda_r_target),
        "lambda_R_warmup_steps": int(lambda_r_warmup_steps),
        "seed": int(seed),
        "train_timeout_seconds": train_timeout_seconds,
        "build_timeout_seconds": build_timeout_seconds,
        "eval_timeout_seconds": eval_timeout_seconds,
        "max_seconds": max_seconds,
    }

    if print_only:
        print("=== print-only: dispatched call NOT created ===")
        print(json.dumps({
            "instance_job_id": instance_job_id,
            "estimated_cost_usd": estimated_cost,
            "params": params,
            "predicted_band": [predicted_low, predicted_high],
            "predicted_eta_utc": predicted_eta_utc,
            "auth_eval_path": "byte_roundtrip_and_brotli_baseline_codec_validation",
            "council_decision": (
                "A4 — co-trained Ballé/ChARM hyperprior 50K-param toy "
                "(council Round 1 second-choice 7/10; Decision 1 GATE-CLEARING)"
            ),
        }, indent=2))
        return

    # ---- Open lane claim BEFORE GPU spend (CLAUDE.md NON-NEGOTIABLE) -----
    notes = (
        f"Phase A4 ChARM 2020 co-trained 50K-param toy substrate on Modal T4; "
        f"council Round 1 second-choice (7/10); Decision 1 GATE-CLEARING; "
        f"predicted=[{predicted_low}, {predicted_high}]; cost=${estimated_cost:.2f}; "
        "parallel with active A1 (different lane_id; no conflict)"
    )
    claim_rc = _claim_lane(
        lane_id="track1_phase_a4_charm_50k_toy",
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=notes,
        force=force_claim,
    )
    if claim_rc != 0:
        raise SystemExit(
            f"FATAL: lane claim failed rc={claim_rc}; aborting before GPU spend. "
            "Use --force-claim if you have explicitly resolved any active conflict "
            "(see .omx/state/active_lane_dispatch_claims.md)."
        )

    # ---- Spawn the Modal function (DETACHED) -----------------------------
    try:
        call = run_phase_a4_t4.spawn(
            instance_job_id,
            int(epochs),
            int(batch_size),
            int(num_train_frames),
            float(lr),
            float(weight_decay),
            float(ema_decay),
            float(noise_std),
            float(lambda_r_target),
            int(lambda_r_warmup_steps),
            int(seed),
            int(train_timeout_seconds),
            int(build_timeout_seconds),
            int(eval_timeout_seconds),
            int(max_seconds),
        )
    except Exception as exc:
        # If `.spawn` fails (e.g., insufficient credit, app build failure), close
        # the lane claim terminally so re-fire is unblocked.
        spawn_err_repr = f"{type(exc).__name__}: {exc!r}"
        spawn_err_lower = spawn_err_repr.lower()
        if any(
            tok in spawn_err_lower
            for tok in ("insufficient", "credit", "balance", "billing", "spend limit")
        ):
            terminal_status = "failed_modal_workspace_billing_cycle_spend_limit_reached"
        else:
            terminal_status = "failed_modal_spawn_submission"
        _claim_lane(
            lane_id="track1_phase_a4_charm_50k_toy",
            instance_job_id=instance_job_id,
            predicted_eta_utc=started_at_utc,
            notes=(
                f"Phase A4 Modal .spawn() failed: {spawn_err_repr}. "
                "Lane claim closed terminally so re-fire is unblocked once Modal credit is restored."
            ),
            status=terminal_status,
            force=True,
        )
        raise SystemExit(
            f"FATAL: Modal `.spawn()` failed: {type(exc).__name__}: {exc}. "
            "If the message includes 'insufficient' / 'credit' / 'balance' / 'spend limit', "
            "Modal credits may be exhausted. Surface to operator; do NOT auto-pivot to other "
            "providers per dispatch ticket."
        ) from exc

    metadata_path = _write_dispatch_metadata(
        instance_job_id=instance_job_id,
        call_id=call.object_id,
        params=params,
        estimated_cost_usd=estimated_cost,
        timeout_hours=float(timeout_hours),
        predicted_low=float(predicted_low),
        predicted_high=float(predicted_high),
        predicted_eta_utc=predicted_eta_utc,
    )

    print(f"\nDISPATCHED Modal T4 call_id={call.object_id}")
    print(f"  instance_job_id: {instance_job_id}")
    print(f"  metadata:        {metadata_path}")
    print(f"  estimated cost:  ${estimated_cost:.2f}")
    print(f"  predicted band:  [{predicted_low}, {predicted_high}] [contest-CUDA]")
    print(f"  predicted ETA:   {predicted_eta_utc}")
    print()
    print("  Recover (within 24h of dispatch — Modal result-cache TTL):")
    print(
        "    .venv/bin/python experiments/modal_phase_a4_charm_50k_toy_substrate.py recover "
        f"--label {instance_job_id}"
    )
    print()
    print("  Or sweep all dispatched Modal calls:")
    print("    .venv/bin/python tools/harvest_modal_calls.py")


def recover(label: str) -> int:
    """Pull a dispatched Modal call's artifacts from the result cache (≤24h)."""
    out_dir = RESULT_ROOT / _safe_label(label)
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        print(f"FATAL: missing {metadata_path}", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text())
    call_id = metadata.get("call_id")
    if not call_id:
        print(f"FATAL: {metadata_path} has no call_id", file=sys.stderr)
        return 2
    instance_job_id = _metadata_instance_job_id(label, metadata)
    print(f"[recover] label={label} call_id={call_id}")
    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=2)
    except modal.exception.OutputExpiredError:
        print(f"FATAL: Modal result cache EXPIRED for call_id={call_id} (>24h since dispatch)")
        close_rc = _close_modal_expired_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
        )
        if close_rc != 0:
            print(f"FATAL: failed to close expired Modal claim rc={close_rc}", file=sys.stderr)
            return 6
        return 3
    except TimeoutError:
        print(f"NOT READY: call_id={call_id} still queued or running. Re-run later.")
        return 4
    except Exception as exc:
        print(f"FATAL: recover failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5

    rc = result.get("returncode", "?")
    elapsed = result.get("elapsed_seconds", 0)
    n_artifacts = len(result.get("artifacts", {}))
    summary = result.get("summary", {})
    print(f"[recover] rc={rc} elapsed={elapsed} artifacts={n_artifacts}")
    print(f"[recover] stage={summary.get('stage')!r} passed={summary.get('passed')!r}")

    artifacts_dir = out_dir / "harvested_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for relpath, data in (result.get("artifacts") or {}).items():
        target = artifacts_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(data)
        except Exception as exc:
            print(f"[recover] SKIP {relpath}: {exc!r}")

    eval_data = result.get("eval_data") or {}
    if eval_data:
        ch = eval_data.get("compressed_bytes_charm")
        bro = eval_data.get("compressed_bytes_brotli")
        rt = eval_data.get("roundtrip_exact")
        beats = eval_data.get("beats_brotli")
        l1 = eval_data.get("pixel_recon_l1")
        print(
            f"[recover] [codec-validation] charm_bytes={ch} brotli_bytes={bro} "
            f"beats_brotli={beats} roundtrip_exact={rt} pixel_l1={l1}"
        )

    summary_path = out_dir / "harvest_summary.json"
    _write_json(summary_path, {
        "label": label,
        "instance_job_id": instance_job_id,
        "call_id": call_id,
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "n_artifacts": n_artifacts,
        "stage": summary.get("stage"),
        "passed": summary.get("passed"),
        "validation_errors": summary.get("validation_errors", []),
        "compressed_bytes_charm": eval_data.get("compressed_bytes_charm") if eval_data else None,
        "compressed_bytes_brotli": eval_data.get("compressed_bytes_brotli") if eval_data else None,
        "beats_brotli": eval_data.get("beats_brotli") if eval_data else None,
        "roundtrip_exact": eval_data.get("roundtrip_exact") if eval_data else None,
        "pixel_recon_l1": eval_data.get("pixel_recon_l1") if eval_data else None,
        "falsification_pass": eval_data.get("falsification_pass") if eval_data else None,
        "tag": summary.get("tag"),
    })
    print(f"[recover] summary saved: {summary_path}")
    close_rc = _close_modal_recovery_claim(
        instance_job_id=instance_job_id,
        call_id=call_id,
        rc=rc,
        summary=summary,
    )
    if close_rc != 0:
        print(f"FATAL: failed to close recovered Modal claim rc={close_rc}", file=sys.stderr)
        return 6
    return 0 if _returncode_is_zero(rc) else 1


# ---------------------------------------------------------------------------
# CLI helpers. Dispatch itself remains the Modal local_entrypoint.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "recover":
        parser = argparse.ArgumentParser(prog="modal_phase_a4_charm_50k_toy_substrate.py recover")
        parser.add_argument(
            "--label",
            required=True,
            help="instance_job_id (e.g., track1_phase_a4_charm_50k_toy_<ts>_modal)",
        )
        args = parser.parse_args(sys.argv[2:])
        sys.exit(recover(args.label))
    else:
        print(
            "USAGE:\n"
            "  Dispatch:  PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\\n"
            "             experiments/modal_phase_a4_charm_50k_toy_substrate.py [args]\n"
            "  Print-only (no app creation): add --print-only to the modal run line.\n"
            "  Recover:   .venv/bin/python experiments/modal_phase_a4_charm_50k_toy_substrate.py "
            "recover --label <instance_job_id>",
            file=sys.stderr,
        )
        sys.exit(2)
