"""Dispatch Phase A1 (track1_phase_a1_score_gradient) on Modal T4 with CUDA auth eval.

Council priority: 22/22 ENDORSE, UNANIMOUS HIGHEST PRIORITY (Quantizr/Carmack/Hinton).
Reference: ``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``.

Why Modal (not Lightning / Vast.ai)?
------------------------------------
2026-05-08: Lightning Studio dispatch chain failed twice — (1) Studio had no GPU
attached; (2) ``insufficient balance to start the cloud space``. Vast.ai API
returned ``400 'Your account lacks credit'``. Operator authorized Modal as the
credit-source for A1 ($2-3 expected on Modal T4).

Why a dedicated dispatcher (not ``modal_train_lane.py``)?
---------------------------------------------------------
``modal_train_lane.py`` HARDCODES ``AUTH_EVAL_DEVICE=cpu`` + ``MODAL_AUTH_EVAL_ADVISORY_ONLY=1``
and stubs ``probe_nvdec.sh`` to always pass — its lanes produce ``[advisory only]``
scores by design (it was designed for Vast.ai NVDEC roulette mitigation).

Phase A1 requires ``[contest-CUDA]`` evidence per CLAUDE.md "Auth eval EVERYWHERE"
+ "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA". So this
dispatcher mirrors the proven CUDA pattern in ``modal_alpha_geo0_pose_regen.py``:
strict ``_probe_cuda_environment`` preflight, fail-loud on CPU fallback, run
contest_auth_eval --device cuda on the rebuilt archive.

Pipeline (container-side on Modal T4):
    1. Input custody (sha256 + size verification)
    2. CUDA + DALI + NVDEC preflight (fail closed on any miss)
    3. Stage 1: TRAIN — experiments/train_score_gradient_pr101_finetune.py
       (200 epochs, ~2-3h on T4; eval_roundtrip + EMA(0.997) defaults)
    4. Stage 2: BUILD — tools/build_pr101_finetuned_archive.py
       (re-encode fine-tuned state_dict → submission_dir/ + archive.zip)
    5. Stage 3: EVAL — experiments/contest_auth_eval.py --device cuda
       (the actual ``[contest-CUDA]`` measurement)
    6. Stage 4: REPORT — write build_manifest.json with score components

Cost (T4 ~$0.59/hr):
    expected: $2-3 for ~3.5h chain (training dominates)
    hard cap: $8 for 4h timeout (ABORTS dispatch if cost-cap exceeded locally)

Modal `.spawn()` — HARVEST OR LOSE:
    Per CLAUDE.md "Modal `.spawn()` puts artifacts in the FunctionCall return-value
    cache (~24h TTL), NOT in a Volume." This dispatcher writes the call_id into
    ``experiments/results/track1_phase_a1_score_gradient_<ts>_modal/modal_metadata.json``
    + ``modal_call_id.txt`` so ``tools/harvest_modal_calls.py`` (which globs
    ``lane_*_modal/``-style paths — see compatibility note below) and the local
    recover entrypoint can find it within 24h.

USAGE — RECOMMENDED (`--detach` for the background run; the local entrypoint
does not block):

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_phase_a1_score_gradient_pr101.py \\
        --pr101-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\
        --pr101-source-dir experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src \\
        --video-path upstream/videos/0.mkv \\
        --epochs 200

Local plan only (no Modal app creation, no lane claim, no GPU spend):

    .venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py plan \\
        --pr101-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \\
        --pr101-source-dir experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src \\
        --video-path upstream/videos/0.mkv \\
        --epochs 200 \\
        --json-out experiments/results/track1_phase_a1_score_gradient_plan.json

Recover (within 24h of dispatch — Modal result-cache TTL):

    .venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \\
        --label track1_phase_a1_score_gradient_<ts>

Or use the canonical harvester to sweep all dispatched Modal calls:

    .venv/bin/python tools/harvest_modal_calls.py

Cross-references:
  - ``.omx/research/grand_council_extreme_rigor_track_1_20260508.md``
  - ``experiments/modal_alpha_geo0_pose_regen.py`` (CUDA-auth-eval pattern source)
  - ``scripts/remote_track1_phase_a1_score_gradient_pr101.sh`` (Lightning Studio-SSH variant)
  - ``tools/dispatch_phase_a1_score_gradient_pr101.py`` (Lightning batch-jobs dispatcher)
  - ``tools/claim_lane_dispatch.py`` (cross-agent claim coordination)
  - feedback_modal_spawn_result_cache_pattern_20260429
  - feedback_dual_cpu_cuda_auth_eval_mandatory_20260508
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
import zipfile
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/workspace/pact")
# Keep score-bearing eval work outside tempfile roots. contest_auth_eval.py
# refuses temp evidence unless explicitly downgraded to diagnostic scratch.
REMOTE_OUT_ROOT = REMOTE_REPO / "experiments/results/modal_phase_a1_remote"
APP_NAME = "comma-phase-a1-score-gradient"
RESULT_ROOT = REPO_ROOT / "experiments" / "results"

# Modal T4 hourly rate — per CLAUDE.md "GPU budget" / docs/hourly_costs.
HOURLY_RATE_T4_USD = 0.59
DEFAULT_TIMEOUT_HOURS = 6.0
DEFAULT_COST_CAP_USD = 8.0
MODAL_TIMEOUT_SAFETY_SECONDS = 10 * 60

# Default operator inputs (override via CLI). These are the operator-specified
# A1 paths from the dispatch ticket.
DEFAULT_PR101_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src"
)
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream/videos/0.mkv"
CHECKPOINT_SELECTIONS = {
    "final_ema": "checkpoint_ema.pt",
    "best_proxy": "checkpoint_best_proxy.pt",
}
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
MOUNTED_CODE_PATHS = (
    "src",
    "upstream/evaluate.py",
    "upstream/frame_utils.py",
    "upstream/modules.py",
    "upstream/public_test_video_names.txt",
    "upstream/pyproject.toml",
    "upstream/uv.lock",
    "experiments/train_score_gradient_pr101_finetune.py",
    "experiments/contest_auth_eval.py",
    "scripts/probe_nvdec.sh",
    "scripts/adjudicate_contest_auth_eval.py",
    "tools/build_pr101_finetuned_archive.py",
    "submissions/factorized_hnerv_v1",
    "pyproject.toml",
    "uv.lock",
)


def _ensure_repo_import_paths() -> None:
    """Make local and Modal-worker repo imports work without shell PYTHONPATH."""
    for path in (
        REPO_ROOT / "src",
        REPO_ROOT / "upstream",
        REPO_ROOT,
        REMOTE_REPO / "src",
        REMOTE_REPO / "upstream",
        REMOTE_REPO,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_repo_import_paths()

from tac.deploy.modal.runtime import (  # noqa: E402
    CONTEST_SCORER_IMPORT_PROBE_MODULES,
    DALI_DISABLE_NVML_VALUE,
    PYTORCH_CUDA_ALLOC_CONF_VALUE,
    build_contest_cuda_base_image,
)


# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

FFMPEG_MASTER_INSTALL_COMMANDS = (
    # ffmpeg-master from BtbN nightly (in_primaries + libsvtav1 — Vast.ai parity).
    "curl -sL "
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-linux64-gpl.tar.xz "
    "-o /tmp/ffmpeg-master.tar.xz",
    "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
    "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg "
    "/usr/local/bin/ffmpeg-master",
    "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
    "/usr/local/bin/ffmpeg-master -hide_banner -h filter=scale 2>&1 "
    "| grep -q in_primaries "
    "|| (echo FATAL: ffmpeg-master lacks in_primaries; exit 1)",
    "/usr/local/bin/ffmpeg-master -encoders 2>&1 "
    "| grep -qi svtav1 "
    "|| (echo FATAL: ffmpeg-master lacks libsvtav1; exit 1)",
    "rm /tmp/ffmpeg-master.tar.xz",
)


def _build_phase_a1_base_image(modal_module: Any):
    """Build the A1 image from the shared contest-CUDA scorer runtime."""
    return build_contest_cuda_base_image(
        modal_module,
        python_version="3.11",
    ).run_commands(*FFMPEG_MASTER_INSTALL_COMMANDS)


base_image = _build_phase_a1_base_image(modal)

run_image = (
    base_image
    .env(
        {
            # Modal containers can expose CUDA while denying NVML internals.
            # DALI's video reader works without NVML when this is set; without
            # it, the NVDEC probe can fail with nvml error (999) before eval.
            "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
            "PYTHONPATH": REMOTE_PYTHONPATH,
        }
    )
    # Workspace mounts — only the bare minimum needed by the A1 chain.
    # Full src/tac is needed because train_score_gradient_pr101_finetune.py +
    # build_pr101_finetuned_archive.py + contest_auth_eval.py all import from tac.
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir("upstream/models", remote_path=str(REMOTE_REPO / "upstream/models"))
    .add_local_dir("upstream/videos", remote_path=str(REMOTE_REPO / "upstream/videos"))
    .add_local_file("upstream/evaluate.py", remote_path=str(REMOTE_REPO / "upstream/evaluate.py"))
    .add_local_file("upstream/frame_utils.py", remote_path=str(REMOTE_REPO / "upstream/frame_utils.py"))
    .add_local_file("upstream/modules.py", remote_path=str(REMOTE_REPO / "upstream/modules.py"))
    .add_local_file(
        "upstream/public_test_video_names.txt",
        remote_path=str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
    )
    .add_local_file("upstream/pyproject.toml", remote_path=str(REMOTE_REPO / "upstream/pyproject.toml"))
    .add_local_file("upstream/uv.lock", remote_path=str(REMOTE_REPO / "upstream/uv.lock"))
    .add_local_file("experiments/__init__.py", remote_path=str(REMOTE_REPO / "experiments/__init__.py"))
    .add_local_file(
        "experiments/train_score_gradient_pr101_finetune.py",
        remote_path=str(REMOTE_REPO / "experiments/train_score_gradient_pr101_finetune.py"),
    )
    .add_local_file(
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file(
        "scripts/probe_nvdec.sh",
        remote_path=str(REMOTE_REPO / "scripts/probe_nvdec.sh"),
    )
    .add_local_file(
        "scripts/adjudicate_contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "scripts/adjudicate_contest_auth_eval.py"),
    )
    .add_local_file(
        "tools/build_pr101_finetuned_archive.py",
        remote_path=str(REMOTE_REPO / "tools/build_pr101_finetuned_archive.py"),
    )
    # factorized_hnerv_v1 supplies HNeRVDecoder for the trainer.
    .add_local_dir(
        "submissions/factorized_hnerv_v1",
        remote_path=str(REMOTE_REPO / "submissions/factorized_hnerv_v1"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))
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


def _run_git_text(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _mounted_code_snapshot(out_dir: Path) -> dict[str, Any]:
    """Record the exact local code state Modal mounted for score-bearing work."""
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mounted_paths = list(MOUNTED_CODE_PATHS)
    head_rc, head_stdout, head_stderr = _run_git_text(["rev-parse", "HEAD"])
    status_rc, status_stdout, status_stderr = _run_git_text(
        ["status", "--short", "--", *mounted_paths]
    )
    diff_rc, diff_stdout, diff_stderr = _run_git_text(["diff", "--binary", "--", *mounted_paths])
    cached_rc, cached_stdout, cached_stderr = _run_git_text(
        ["diff", "--cached", "--binary", "--", *mounted_paths]
    )
    status_path = snapshot_dir / "mounted_code_status.txt"
    diff_path = snapshot_dir / "mounted_code_worktree.patch"
    cached_diff_path = snapshot_dir / "mounted_code_index.patch"
    status_path.write_text(status_stdout)
    diff_path.write_text(diff_stdout)
    cached_diff_path.write_text(cached_stdout)
    return {
        "schema_version": "mounted_modal_code_snapshot_v1",
        "git_head": head_stdout.strip() if head_rc == 0 else None,
        "git_head_error": head_stderr.strip() if head_rc != 0 else None,
        "mounted_code_paths": mounted_paths,
        "dirty": bool(status_stdout.strip()),
        "status_short": status_stdout.splitlines(),
        "status_rc": status_rc,
        "status_error": status_stderr.strip() if status_rc != 0 else None,
        "worktree_diff_bytes": len(diff_stdout.encode("utf-8")),
        "worktree_diff_sha256": _sha256_bytes(diff_stdout.encode("utf-8")),
        "worktree_diff_path": str(diff_path),
        "worktree_diff_rc": diff_rc,
        "worktree_diff_error": diff_stderr.strip() if diff_rc != 0 else None,
        "index_diff_bytes": len(cached_stdout.encode("utf-8")),
        "index_diff_sha256": _sha256_bytes(cached_stdout.encode("utf-8")),
        "index_diff_path": str(cached_diff_path),
        "index_diff_rc": cached_rc,
        "index_diff_error": cached_stderr.strip() if cached_rc != 0 else None,
        "status_path": str(status_path),
        "score_promotion_note": (
            "If dirty=true, this dispatch remains reproducible only with the "
            "recorded patch artifacts or by rerunning from a clean commit."
        ),
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "track1_phase_a1_score_gradient"


def _tail(value: str | bytes | None, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _first_numeric(*values: Any) -> float | None:
    for value in values:
        parsed = _numeric_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _eval_metric_summary(eval_data: dict[str, Any] | None) -> dict[str, float | None]:
    """Normalize contest_auth_eval schemas used by Modal harvesters.

    Older wrappers returned ``score_components`` with ``pose``/``seg``/``rate``.
    The current canonical evaluator emits ``canonical_score``,
    ``score_recomputed_from_components``, ``avg_posenet_dist``,
    ``avg_segnet_dist``, and ``score_rate_contribution``. Do not let a harvest
    hide an exact score just because one schema is absent.
    """
    if not eval_data:
        return {
            "score": None,
            "pose_avg": None,
            "seg_avg": None,
            "rate": None,
            "rate_unscaled": None,
        }
    sc = eval_data.get("score_components") or {}
    if not isinstance(sc, dict):
        sc = {}
    return {
        "score": _first_numeric(
            eval_data.get("canonical_score"),
            eval_data.get("score_recomputed_from_components"),
            eval_data.get("score"),
            eval_data.get("total_score"),
            eval_data.get("final_score"),
        ),
        "pose_avg": _first_numeric(
            eval_data.get("avg_posenet_dist"),
            eval_data.get("pose_avg"),
            sc.get("pose"),
            sc.get("pose_avg"),
            sc.get("posenet"),
        ),
        "seg_avg": _first_numeric(
            eval_data.get("avg_segnet_dist"),
            eval_data.get("seg_avg"),
            sc.get("seg"),
            sc.get("seg_avg"),
            sc.get("segnet"),
        ),
        "rate": _first_numeric(
            eval_data.get("score_rate_contribution"),
            eval_data.get("rate"),
            sc.get("rate"),
            sc.get("rate_term"),
        ),
        "rate_unscaled": _first_numeric(
            eval_data.get("rate_unscaled"),
            sc.get("rate_unscaled"),
        ),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _recover_evidence_summary(
    eval_data: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return fail-closed local evidence status for harvested Modal results."""
    from tac.auth_eval_schema import (  # noqa: PLC0415
        eval_metric_summary,
        required_contest_cuda_evidence_blockers,
    )

    build_manifest = _as_dict(summary.get("build_manifest"))
    archive_meta = _as_dict(summary.get("archive_meta"))
    expected_archive_bytes = build_manifest.get("archive_bytes", archive_meta.get("bytes"))
    if not isinstance(expected_archive_bytes, int):
        expected_archive_bytes = None

    metrics = eval_metric_summary(eval_data)
    auth_eval_blockers = required_contest_cuda_evidence_blockers(
        eval_data,
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=600,
    )

    remote_claim_blockers: list[str] = []
    if summary.get("passed") is not True:
        remote_claim_blockers.append("remote_summary_passed_not_true")
    if not build_manifest:
        remote_claim_blockers.append("remote_build_manifest_missing")
    elif build_manifest.get("score_claim_valid") is not True:
        remote_claim_blockers.append("remote_build_manifest_score_claim_valid_not_true")
    if build_manifest and build_manifest.get("exact_cuda_eval_complete") is not True:
        remote_claim_blockers.append("remote_build_manifest_exact_cuda_eval_complete_not_true")
    if build_manifest and build_manifest.get("score_claim") is not True:
        remote_claim_blockers.append("remote_build_manifest_score_claim_not_true")

    claim_blockers = _unique_strings(auth_eval_blockers + remote_claim_blockers)
    score_claim = not claim_blockers
    dispatch_blockers = _unique_strings(build_manifest.get("dispatch_blockers"))
    return {
        "metrics": metrics,
        "score_claim": score_claim,
        "score_claim_valid": score_claim,
        "exact_cuda_eval_complete": score_claim,
        "promotion_eligible": build_manifest.get("promotion_eligible") is True,
        "rank_or_kill_eligible": build_manifest.get("rank_or_kill_eligible") is True,
        "evidence_grade": "[contest-CUDA]" if score_claim else "[exact-eval incomplete]",
        "auth_eval_blockers": auth_eval_blockers,
        "claim_blockers": claim_blockers,
        "dispatch_blockers": dispatch_blockers,
        "canonical_score_source": metrics["canonical_score_source"],
        "eval_archive_size_bytes": metrics["archive_size_bytes"],
        "n_samples": metrics["n_samples"],
        "score_axis": eval_data.get("score_axis") if isinstance(eval_data, dict) else None,
        "evidence_semantics": (
            eval_data.get("evidence_semantics") if isinstance(eval_data, dict) else None
        ),
        "lane_tag": eval_data.get("lane_tag") if isinstance(eval_data, dict) else None,
    }


def _selected_checkpoint_path(train_output: Path, checkpoint_selection: str) -> Path:
    checkpoint_name = CHECKPOINT_SELECTIONS.get(checkpoint_selection)
    if checkpoint_name is None:
        raise ValueError(
            f"unknown checkpoint_selection={checkpoint_selection!r}; "
            f"expected one of {sorted(CHECKPOINT_SELECTIONS)}"
        )
    return train_output / checkpoint_name


def _stage_timeout_budget_seconds(
    *,
    train_timeout_hours: float,
    build_timeout_minutes: float,
    eval_timeout_minutes: float,
) -> int:
    return (
        max(120, int(float(train_timeout_hours) * 3600))
        + max(60, int(float(build_timeout_minutes) * 60))
        + max(120, int(float(eval_timeout_minutes) * 60))
    )


def _modal_timeout_validation_errors(
    *,
    timeout_hours: float,
    train_timeout_hours: float,
    build_timeout_minutes: float,
    eval_timeout_minutes: float,
) -> list[str]:
    """Return blockers when user-facing timeout no longer matches Modal reality."""
    errors: list[str] = []
    if abs(float(timeout_hours) - DEFAULT_TIMEOUT_HOURS) > 1e-9:
        errors.append(
            "timeout_hours_must_match_modal_function_timeout:"
            f"requested={float(timeout_hours):.2f}:actual={DEFAULT_TIMEOUT_HOURS:.2f}"
        )
    stage_budget = _stage_timeout_budget_seconds(
        train_timeout_hours=train_timeout_hours,
        build_timeout_minutes=build_timeout_minutes,
        eval_timeout_minutes=eval_timeout_minutes,
    )
    usable_budget = int(DEFAULT_TIMEOUT_HOURS * 3600) - MODAL_TIMEOUT_SAFETY_SECONDS
    if stage_budget > usable_budget:
        errors.append(
            "stage_timeouts_exceed_modal_budget:"
            f"stage_budget_s={stage_budget}:usable_modal_budget_s={usable_budget}"
        )
    return errors


def _read_input(path: Path, label: str) -> tuple[bytes, str, int]:
    if not path.is_file():
        raise SystemExit(f"FATAL: {label} not found: {path}")
    data = path.read_bytes()
    return data, _sha256_bytes(data), len(data)


def _read_dir_as_tarball(directory: Path, label: str) -> tuple[bytes, str, int]:
    """Pack a directory as an in-memory zip of its files (no compression)."""
    import io

    if not directory.is_dir():
        raise SystemExit(f"FATAL: {label} not found or not a dir: {directory}")
    if label == "PR101 source dir":
        _validate_pr101_source_root(directory)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        # Sort for determinism.
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            # Skip __pycache__ and .pyc.
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            arcname = path.relative_to(directory).as_posix()
            zf.write(path, arcname=arcname)
    data = buf.getvalue()
    return data, _sha256_bytes(data), len(data)


def _validate_pr101_source_root(directory: Path) -> None:
    """Fail closed unless directory is the PR101 runtime source root.

    The build tool expects ``codec.py`` and ``model.py`` at the root. Passing
    the detached PR checkout's parent ``source/`` directory silently uploaded
    hundreds of MB and failed only after a full Modal training run.
    """
    missing = [name for name in ("codec.py", "model.py") if not (directory / name).is_file()]
    if not missing:
        return
    candidate = directory / "submissions" / "hnerv_ft_microcodec" / "src"
    hint = f"; did you mean {candidate}?" if (candidate / "codec.py").is_file() else ""
    raise SystemExit(
        f"FATAL: PR101 source dir must contain codec.py and model.py at its root; "
        f"missing {', '.join(missing)} under {directory}{hint}"
    )


# ---------------------------------------------------------------------------
# Remote-side: CUDA preflight + chain stages
# ---------------------------------------------------------------------------

def _probe_cuda_environment_remote(env: dict[str, str]) -> dict[str, Any]:
    """Container-side CUDA + DALI + NVDEC probe.

    Mirrors ``experiments/modal_alpha_geo0_pose_regen.py::_probe_cuda_environment``
    EXACTLY. Failing this probe aborts the chain BEFORE training spend.
    """
    import shutil
    import subprocess as sp

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_phase_a1_score_gradient_pr101.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_required": "cuda",
        "gpu_requested": "T4",
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
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

    probe = REMOTE_REPO / "scripts/probe_nvdec.sh"
    if probe.is_file():
        try:
            proc = sp.run(
                ["bash", str(probe)],
                cwd=str(REMOTE_REPO),
                env={**env, "PYBIN": sys.executable},
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            preflight["nvdec_probe_returncode"] = int(proc.returncode)
            preflight["nvdec_probe_passed"] = proc.returncode == 0
            preflight["nvdec_probe_stdout_tail"] = _tail(proc.stdout)
            preflight["nvdec_probe_stderr_tail"] = _tail(proc.stderr)
        except Exception as exc:  # pragma: no cover
            preflight["nvdec_probe_returncode"] = 125
            preflight["nvdec_probe_passed"] = False
            preflight["nvdec_probe_error"] = repr(exc)
    else:
        preflight["nvdec_probe_passed"] = False
        preflight["nvdec_probe_error"] = f"missing probe script: {probe}"

    return preflight


def _preflight_errors_remote(preflight: dict[str, Any]) -> list[str]:
    """Return list of fatal preflight failures (empty list = OK to dispatch)."""
    errors: list[str] = []
    if preflight.get("torch_cuda_available") is not True:
        errors.append("Modal runtime has no CUDA device; refusing CPU fallback")
    if preflight.get("nvidia_dali_import_ok") is not True:
        errors.append("nvidia.dali import failed; refusing non-DALI CUDA eval")
    if preflight.get("nvdec_probe_passed") is not True:
        errors.append("NVDEC/DALI video probe failed; upstream/evaluate.py CUDA path is unsafe")
    if preflight.get("gpu_t4_match") is not True:
        errors.append("Modal runtime did not report a T4 GPU")
    return errors


def _scorer_runtime_import_probe_cmd() -> list[str]:
    imports = "; ".join(
        f"import {module}" for module in CONTEST_SCORER_IMPORT_PROBE_MODULES
    )
    return [
        sys.executable,
        "-c",
        (
            f"{imports}; "
            "from tac.scorer import load_differentiable_scorers; "
            "print('phase a1 modal scorer import probe OK')"
        ),
    ]


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
    """Collect output artifacts to embed in the function-call return value.

    Mirrors the size-limited collection in modal_train_lane.py (which caps at
    500MB to leave room for return-value cache TTL). Skips __pycache__ + .pyc.
    """
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
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "tag": "[contest-CUDA]" if (passed and eval_data) else "[contest-CUDA dispatch failed]",
    }
    if eval_data:
        summary["eval_data"] = eval_data
        metrics = _eval_metric_summary(eval_data)
        summary["score"] = metrics["score"]
        summary["pose_avg"] = metrics["pose_avg"]
        summary["seg_avg"] = metrics["seg_avg"]
        summary["rate"] = metrics["rate"]
        summary["rate_unscaled"] = metrics["rate_unscaled"]
    if extra:
        summary.update(extra)
    _write_json(out_dir / "phase_a1_summary.json", summary)
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


def _run_phase_a1_inner(
    *,
    pr101_archive_bytes: bytes,
    pr101_archive_sha256: str,
    pr101_archive_size_bytes: int,
    pr101_source_zip_bytes: bytes,
    pr101_source_zip_sha256: str,
    pr101_source_zip_size_bytes: int,
    video_bytes: bytes,
    video_sha256: str,
    video_size_bytes: int,
    label: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    max_frames: int,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
    checkpoint_selection: str,
    continue_after_nvdec_failure: bool,
    train_timeout_seconds: int,
    build_timeout_seconds: int,
    eval_timeout_seconds: int,
    max_seconds: int,
) -> dict[str, Any]:
    """Container-side runner for the full A1 chain on Modal T4."""
    import os
    import shutil

    label_safe = _safe_label(label)
    out_dir = REMOTE_OUT_ROOT / label_safe
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage A: stage inputs -------------------------------------------
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    pr101_archive = inputs_dir / "archive.zip"
    pr101_source_zip = inputs_dir / "pr101_src.zip"
    pr101_source_dir = inputs_dir / "pr101_src"
    video_path = inputs_dir / "0.mkv"
    pr101_archive.write_bytes(pr101_archive_bytes)
    pr101_source_zip.write_bytes(pr101_source_zip_bytes)
    video_path.write_bytes(video_bytes)
    # Unpack pr101_source_zip → pr101_source_dir.
    # Use tac.submission_archive.safe_extract_zip — raw ZipFile.extractall is
    # forbidden by preflight check_no_raw_zip_extractall (zip-slip, duplicate
    # members, symlinks all hide until later in the pipeline).
    from tac.submission_archive import safe_extract_zip
    safe_extract_zip(pr101_source_zip, pr101_source_dir)

    stage = "input_custody"
    command_results: list[dict[str, Any]] = []
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "TAC_UPSTREAM_DIR": str(REMOTE_REPO / "upstream"),
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "PYTHONHASHSEED": "20",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
    }
    os.environ["DALI_DISABLE_NVML"] = env["DALI_DISABLE_NVML"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env["PYTORCH_CUDA_ALLOC_CONF"]

    t_start = time.monotonic()
    try:
        observed_inputs = {
            "pr101_archive": _file_meta(pr101_archive),
            "pr101_source_zip": _file_meta(pr101_source_zip),
            "video": _file_meta(video_path),
        }
        custody_errors: list[str] = []
        expected = {
            "pr101_archive": (pr101_archive_sha256, pr101_archive_size_bytes),
            "pr101_source_zip": (pr101_source_zip_sha256, pr101_source_zip_size_bytes),
            "video": (video_sha256, video_size_bytes),
        }
        for key, (sha, size) in expected.items():
            meta = observed_inputs[key]
            if meta["sha256"] != sha or meta["bytes"] != size:
                custody_errors.append(f"{key} custody mismatch")
        input_manifest = {
            "schema_version": 1,
            "label": label_safe,
            "inputs": observed_inputs,
            "params": {
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "lr": lr,
                "max_frames": max_frames,
                "aux_kl_weight": aux_kl_weight,
                "aux_pixel_l1_weight": aux_pixel_l1_weight,
                "checkpoint_selection": checkpoint_selection,
            },
            "score_claim": False,
            "promotion_eligible": False,
        }
        _write_json(out_dir / "phase_a1_input_manifest.json", input_manifest)
        if custody_errors:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=5,
                stage=stage,
                validation_errors=custody_errors,
            )

        # --- Stage B: CUDA + DALI + NVDEC preflight -----------------------
        stage = "cuda_dali_nvdec_preflight"
        preflight = _probe_cuda_environment_remote(env)
        _write_json(out_dir / "phase_a1_preflight.json", preflight)
        errors = _preflight_errors_remote(preflight)
        nvdec_errors = [
            err for err in errors
            if err.startswith("NVDEC/DALI video probe failed")
        ]
        fatal_preflight_errors = [err for err in errors if err not in nvdec_errors]
        skip_cuda_eval_reason: str | None = None
        if fatal_preflight_errors or (errors and not continue_after_nvdec_failure):
            # NVDEC probe failures on Modal are recoverable (train doesn't need
            # NVDEC; only contest_auth_eval --device cuda does). But we surface
            # a HARD STOP per CLAUDE.md "MPS auth eval is NOISE" — without
            # NVDEC the eval would silently fall back to AVVideoDataset on CPU
            # ("[advisory only]"), so dispatching is wasted spend.
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=6,
                stage=stage,
                validation_errors=errors,
                extra={"preflight": preflight},
            )
        if nvdec_errors and continue_after_nvdec_failure:
            skip_cuda_eval_reason = (
                "Modal T4 compute preflight passed but DALI/NVDEC exact-eval "
                "probe failed; continuing training/build only with score_claim=false."
            )
            preflight["nvdec_failure_training_continues_no_exact_eval"] = True
            preflight["auth_eval_skip_reason"] = skip_cuda_eval_reason

        # --- Stage C: scorer runtime dependency import probe --------------
        stage = "scorer_runtime_import_probe"
        import_probe = _run_logged_remote(
            "stage0_scorer_runtime_import_probe",
            _scorer_runtime_import_probe_cmd(),
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "logs",
            timeout=180,
        )
        command_results.append(import_probe)
        if import_probe["returncode"] != 0:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=import_probe["returncode"],
                stage="remote_import_probe_failed",
                validation_errors=[
                    f"remote_import_probe_rc={import_probe['returncode']}"
                ],
                extra={"commands": command_results, "preflight": preflight},
            )

        # --- Stage 1: TRAIN -----------------------------------------------
        stage = "train_score_gradient_pr101"
        train_output = out_dir / "train"
        train_output.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable, "-u",
            str(REMOTE_REPO / "experiments/train_score_gradient_pr101_finetune.py"),
            "--device", "cuda",
            "--epochs", str(epochs),
            "--steps-per-epoch", str(steps_per_epoch),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--pr101-archive", str(pr101_archive),
            "--pr101-source-dir", str(pr101_source_dir),
            "--video-path", str(video_path),
            "--max-frames", str(max_frames),
            "--aux-kl-weight", str(aux_kl_weight),
            "--aux-pixel-l1-weight", str(aux_pixel_l1_weight),
            "--output", str(train_output),
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
        checkpoint_path = _selected_checkpoint_path(train_output, checkpoint_selection)
        if not checkpoint_path.is_file():
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=4,
                stage=stage,
                validation_errors=[
                    f"selected checkpoint {checkpoint_selection!r} missing at {checkpoint_path}"
                ],
                extra={"commands": command_results},
            )

        # --- Stage 2: BUILD finetuned PR101 archive -----------------------
        stage = "build_pr101_finetuned_archive"
        archive_output = out_dir / "finetuned_archive"
        archive_output.mkdir(parents=True, exist_ok=True)
        build_cmd = [
            sys.executable, "-u",
            str(REMOTE_REPO / "tools/build_pr101_finetuned_archive.py"),
            "--state-dict", str(checkpoint_path),
            "--source-archive", str(pr101_archive),
            "--pr101-source-dir", str(pr101_source_dir),
            "--output-dir", str(archive_output),
            "--lane-id", label_safe,
        ]
        run = _run_logged_remote(
            "stage2_build",
            build_cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "logs",
            timeout=build_timeout_seconds,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=[f"archive build failed rc={run['returncode']}"],
                extra={"commands": command_results},
            )
        archive_zip = archive_output / "archive.zip"
        inflate_sh = archive_output / "submission_dir" / "inflate.sh"
        if not (archive_zip.is_file() and inflate_sh.is_file()):
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=5,
                stage=stage,
                validation_errors=[
                    f"archive build incomplete: archive={archive_zip} inflate={inflate_sh}"
                ],
                extra={"commands": command_results},
            )
        archive_meta = _file_meta(archive_zip)

        if skip_cuda_eval_reason:
            build_manifest = {
                "lane_id": label_safe,
                "schema_version": "phase_a1_modal_build_manifest_v1",
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "archive_path": str(archive_zip),
                "archive_bytes": archive_meta["bytes"],
                "archive_sha256": archive_meta["sha256"],
                "checkpoint_selection": checkpoint_selection,
                "checkpoint_path": str(checkpoint_path),
                "eval_work_dir": None,
                "eval_rc": None,
                "eval_data": None,
                "evidence_grade": "[cuda-training-build-only]",
                "score_claim": False,
                "score_claim_valid": False,
                "ready_for_exact_eval_dispatch": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "dispatch_blockers": [
                    "contest_cuda_eval_pending_due_modal_dali_nvdec_preflight_failure",
                    "contest_cpu_eval_pending",
                ],
                "auth_eval_skipped": True,
                "auth_eval_skip_reason": skip_cuda_eval_reason,
                "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
                "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
                "modal_app": APP_NAME,
                "modal_gpu": "T4",
                "preflight": preflight,
                "params": {
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_frames": max_frames,
                    "aux_kl_weight": aux_kl_weight,
                    "aux_pixel_l1_weight": aux_pixel_l1_weight,
                    "checkpoint_selection": checkpoint_selection,
                    "continue_after_nvdec_failure": continue_after_nvdec_failure,
                },
            }
            _write_json(out_dir / "build_manifest.json", build_manifest)
            result = _finish_remote(
                out_dir,
                passed=True,
                returncode=0,
                stage="completed_training_build_cuda_eval_skipped_nvdec_preflight",
                validation_errors=[],
                extra={
                    "commands": command_results,
                    "archive_meta": archive_meta,
                    "build_manifest": build_manifest,
                    "preflight": preflight,
                    "auth_eval_skipped": True,
                    "auth_eval_skip_reason": skip_cuda_eval_reason,
                },
                eval_data=None,
            )
            result["elapsed_seconds"] = time.monotonic() - t_start
            return result

        # --- Stage 3: contest_auth_eval --device cuda ---------------------
        stage = "contest_auth_eval_cuda"
        eval_work = out_dir / "eval_work"
        eval_work.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            sys.executable, "-u",
            str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
            "--archive", str(archive_zip),
            "--inflate-sh", str(inflate_sh),
            "--upstream-dir", str(REMOTE_REPO / "upstream"),
            "--device", "cuda",
            "--work-dir", str(eval_work),
            "--keep-work-dir",
        ]
        run = _run_logged_remote(
            "stage3_eval",
            eval_cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "logs",
            timeout=eval_timeout_seconds,
        )
        command_results.append(run)
        eval_rc = run["returncode"]
        eval_data: dict[str, Any] | None = None
        eval_json_path = eval_work / "contest_auth_eval.json"
        if eval_json_path.is_file():
            try:
                eval_data = json.loads(eval_json_path.read_text())
            except json.JSONDecodeError as exc:
                eval_data = {"error": f"contest_auth_eval.json parse error: {exc!r}"}
        if eval_rc != 0:
            return _finish_remote(
                out_dir,
                passed=False,
                returncode=eval_rc,
                stage=stage,
                validation_errors=[f"contest_auth_eval failed rc={eval_rc}"],
                extra={"commands": command_results, "archive_meta": archive_meta},
                eval_data=eval_data,
            )

        # --- Stage 4: REPORT ----------------------------------------------
        stage = "report"
        from tac.auth_eval_schema import (
            eval_metric_summary,
            required_contest_cuda_evidence_blockers,
        )

        metrics = eval_metric_summary(eval_data)
        metric_blockers = required_contest_cuda_evidence_blockers(
            eval_data,
            metrics,
            expected_archive_bytes=archive_meta["bytes"],
            expected_n_samples=600,
        )
        score_claim = eval_rc == 0 and not metric_blockers
        build_manifest = {
            "lane_id": label_safe,
            "schema_version": "phase_a1_modal_build_manifest_v1",
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archive_path": str(archive_zip),
            "archive_bytes": archive_meta["bytes"],
            "archive_sha256": archive_meta["sha256"],
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_path": str(checkpoint_path),
            "eval_work_dir": str(eval_work),
            "eval_rc": eval_rc,
            "eval_data": eval_data,
            "evidence_grade": "[contest-CUDA]" if score_claim else "[exact-eval incomplete]",
            "score_claim": score_claim,
            "score_claim_valid": score_claim,
            "ready_for_exact_eval_dispatch": False,
            "exact_cuda_eval_complete": score_claim,
            # Promotion still requires the CPU paired axis per CLAUDE.md
            # "Submission auth eval — BOTH CPU AND CUDA". The Modal CUDA result
            # IS the contest-CUDA truth; CPU eval comes via GHA later.
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "dispatch_blockers": (
                ["contest_cpu_eval_pending"] if score_claim else metric_blockers
            ),
            "canonical_score_source": metrics["canonical_score_source"],
            "eval_archive_size_bytes": metrics["archive_size_bytes"],
            "n_samples": metrics["n_samples"],
            "score_axis": eval_data.get("score_axis") if isinstance(eval_data, dict) else None,
            "evidence_semantics": (
                eval_data.get("evidence_semantics") if isinstance(eval_data, dict) else None
            ),
            "lane_tag": eval_data.get("lane_tag") if isinstance(eval_data, dict) else None,
            "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
            "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
            "modal_app": APP_NAME,
            "modal_gpu": "T4",
            "preflight": preflight,
            "params": {
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "lr": lr,
                "max_frames": max_frames,
                "aux_kl_weight": aux_kl_weight,
                "aux_pixel_l1_weight": aux_pixel_l1_weight,
                "checkpoint_selection": checkpoint_selection,
            },
        }
        build_manifest["score"] = metrics["score"]
        build_manifest["pose_avg"] = metrics["pose_avg"]
        build_manifest["seg_avg"] = metrics["seg_avg"]
        build_manifest["rate"] = metrics["rate"]
        build_manifest["rate_unscaled"] = metrics["rate_unscaled"]
        _write_json(out_dir / "build_manifest.json", build_manifest)

        if not score_claim:
            result = _finish_remote(
                out_dir,
                passed=False,
                returncode=6,
                stage="contest_cuda_evidence_refused",
                validation_errors=metric_blockers,
                extra={
                    "commands": command_results,
                    "archive_meta": archive_meta,
                    "build_manifest": build_manifest,
                },
                eval_data=eval_data,
            )
            result["elapsed_seconds"] = time.monotonic() - t_start
            return result

        result = _finish_remote(
            out_dir,
            passed=True,
            returncode=0,
            stage="completed",
            extra={
                "commands": command_results,
                "archive_meta": archive_meta,
                "build_manifest": build_manifest,
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
def run_phase_a1_t4(
    pr101_archive_bytes: bytes,
    pr101_archive_sha256: str,
    pr101_archive_size_bytes: int,
    pr101_source_zip_bytes: bytes,
    pr101_source_zip_sha256: str,
    pr101_source_zip_size_bytes: int,
    video_bytes: bytes,
    video_sha256: str,
    video_size_bytes: int,
    label: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    max_frames: int,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
    checkpoint_selection: str,
    continue_after_nvdec_failure: bool,
    train_timeout_seconds: int,
    build_timeout_seconds: int,
    eval_timeout_seconds: int,
    max_seconds: int,
) -> dict[str, Any]:
    return _run_phase_a1_inner(
        pr101_archive_bytes=pr101_archive_bytes,
        pr101_archive_sha256=pr101_archive_sha256,
        pr101_archive_size_bytes=pr101_archive_size_bytes,
        pr101_source_zip_bytes=pr101_source_zip_bytes,
        pr101_source_zip_sha256=pr101_source_zip_sha256,
        pr101_source_zip_size_bytes=pr101_source_zip_size_bytes,
        video_bytes=video_bytes,
        video_sha256=video_sha256,
        video_size_bytes=video_size_bytes,
        label=label,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        lr=lr,
        max_frames=max_frames,
        aux_kl_weight=aux_kl_weight,
        aux_pixel_l1_weight=aux_pixel_l1_weight,
        checkpoint_selection=checkpoint_selection,
        continue_after_nvdec_failure=continue_after_nvdec_failure,
        train_timeout_seconds=train_timeout_seconds,
        build_timeout_seconds=build_timeout_seconds,
        eval_timeout_seconds=eval_timeout_seconds,
        max_seconds=max_seconds,
    )


# ---------------------------------------------------------------------------
# Local-side: lane claim + dispatch metadata + recover
# ---------------------------------------------------------------------------

def _phase_a1_dispatch_command(
    *,
    pr101_archive: Path,
    pr101_source_dir: Path,
    video_path: Path,
    label: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    max_frames: int,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
    checkpoint_selection: str,
    continue_after_nvdec_failure: bool,
    train_timeout_hours: float,
    build_timeout_minutes: float,
    eval_timeout_minutes: float,
    timeout_hours: float,
    cost_cap_usd: float,
    predicted_low: float,
    predicted_high: float,
) -> list[str]:
    """Return the audited Modal command a human/operator can dispatch later."""
    command = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_phase_a1_score_gradient_pr101.py",
        "--pr101-archive", str(pr101_archive),
        "--pr101-source-dir", str(pr101_source_dir),
        "--video-path", str(video_path),
        "--label", label,
        "--epochs", str(int(epochs)),
        "--steps-per-epoch", str(int(steps_per_epoch)),
        "--batch-size", str(int(batch_size)),
        "--lr", str(float(lr)),
        "--max-frames", str(int(max_frames)),
        "--aux-kl-weight", str(float(aux_kl_weight)),
        "--aux-pixel-l1-weight", str(float(aux_pixel_l1_weight)),
        "--checkpoint-selection", checkpoint_selection,
        "--train-timeout-hours", str(float(train_timeout_hours)),
        "--build-timeout-minutes", str(float(build_timeout_minutes)),
        "--eval-timeout-minutes", str(float(eval_timeout_minutes)),
        "--timeout-hours", str(float(timeout_hours)),
        "--cost-cap-usd", str(float(cost_cap_usd)),
        "--predicted-low", str(float(predicted_low)),
        "--predicted-high", str(float(predicted_high)),
    ]
    if continue_after_nvdec_failure:
        command.append("--continue-after-nvdec-failure")
    return command


def build_local_plan(
    *,
    pr101_archive: str,
    pr101_source_dir: str,
    video_path: str,
    label: str | None,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    max_frames: int,
    aux_kl_weight: float,
    aux_pixel_l1_weight: float,
    checkpoint_selection: str,
    continue_after_nvdec_failure: bool,
    train_timeout_hours: float,
    build_timeout_minutes: float,
    eval_timeout_minutes: float,
    timeout_hours: float,
    cost_cap_usd: float,
    predicted_low: float,
    predicted_high: float,
) -> tuple[dict[str, Any], int]:
    """Build a local-only A1 dispatch plan without touching Modal or claims.

    ``modal run --print-only`` still asks Modal to create the app. This path is
    a zero-remote, zero-GPU custody artifact for operator review and later
    dispatch once provider billing/auth is healthy.
    """
    pr101_archive_path = Path(pr101_archive).resolve()
    pr101_source_dir_path = Path(pr101_source_dir).resolve()
    video_path_resolved = Path(video_path).resolve()

    if label:
        instance_job_id = _safe_label(label)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        instance_job_id = f"track1_phase_a1_score_gradient_{timestamp}_modal"

    now = dt.datetime.now(tz=dt.UTC)
    predicted_eta_utc = (
        now + dt.timedelta(hours=DEFAULT_TIMEOUT_HOURS)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    estimated_cost = HOURLY_RATE_T4_USD * DEFAULT_TIMEOUT_HOURS

    validation_errors: list[str] = []
    validation_errors.extend(
        _modal_timeout_validation_errors(
            timeout_hours=timeout_hours,
            train_timeout_hours=train_timeout_hours,
            build_timeout_minutes=build_timeout_minutes,
            eval_timeout_minutes=eval_timeout_minutes,
        )
    )
    if not pr101_archive_path.is_file():
        validation_errors.append(f"missing_pr101_archive:{pr101_archive_path}")
    if not pr101_source_dir_path.is_dir():
        validation_errors.append(f"missing_pr101_source_dir:{pr101_source_dir_path}")
    if not video_path_resolved.is_file():
        validation_errors.append(f"missing_video_path:{video_path_resolved}")
    if estimated_cost > float(cost_cap_usd):
        validation_errors.append(
            f"estimated_cost_exceeds_cap:{estimated_cost:.2f}>{float(cost_cap_usd):.2f}"
        )

    inputs: dict[str, Any] = {
        "pr101_archive": {"path": str(pr101_archive_path), "exists": pr101_archive_path.is_file()},
        "pr101_source_dir": {"path": str(pr101_source_dir_path), "exists": pr101_source_dir_path.is_dir()},
        "video": {"path": str(video_path_resolved), "exists": video_path_resolved.is_file()},
    }
    if not validation_errors:
        pr101_archive_data, pr101_archive_sha, pr101_archive_size = _read_input(
            pr101_archive_path, "PR101 archive"
        )
        del pr101_archive_data
        pr101_source_zip, pr101_source_zip_sha, pr101_source_zip_size = _read_dir_as_tarball(
            pr101_source_dir_path, "PR101 source dir"
        )
        del pr101_source_zip
        video_data, video_sha, video_size = _read_input(video_path_resolved, "video")
        del video_data
        inputs.update(
            {
                "pr101_archive": {
                    "path": str(pr101_archive_path),
                    "exists": True,
                    "bytes": pr101_archive_size,
                    "sha256": pr101_archive_sha,
                },
                "pr101_source_zip": {
                    "source_dir": str(pr101_source_dir_path),
                    "exists": True,
                    "bytes": pr101_source_zip_size,
                    "sha256": pr101_source_zip_sha,
                    "format": "zip_stored_directory_snapshot",
                },
                "video": {
                    "path": str(video_path_resolved),
                    "exists": True,
                    "bytes": video_size,
                    "sha256": video_sha,
                },
            }
        )

    params = {
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "max_frames": int(max_frames),
        "aux_kl_weight": float(aux_kl_weight),
        "aux_pixel_l1_weight": float(aux_pixel_l1_weight),
        "checkpoint_selection": checkpoint_selection,
        "continue_after_nvdec_failure": bool(continue_after_nvdec_failure),
        "train_timeout_hours": float(train_timeout_hours),
        "build_timeout_minutes": float(build_timeout_minutes),
        "eval_timeout_minutes": float(eval_timeout_minutes),
        "timeout_hours": float(timeout_hours),
        "cost_cap_usd": float(cost_cap_usd),
    }
    dispatch_command = _phase_a1_dispatch_command(
        pr101_archive=pr101_archive_path,
        pr101_source_dir=pr101_source_dir_path,
        video_path=video_path_resolved,
        label=instance_job_id,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        lr=lr,
        max_frames=max_frames,
        aux_kl_weight=aux_kl_weight,
        aux_pixel_l1_weight=aux_pixel_l1_weight,
        checkpoint_selection=checkpoint_selection,
        continue_after_nvdec_failure=continue_after_nvdec_failure,
        train_timeout_hours=train_timeout_hours,
        build_timeout_minutes=build_timeout_minutes,
        eval_timeout_minutes=eval_timeout_minutes,
        timeout_hours=timeout_hours,
        cost_cap_usd=cost_cap_usd,
        predicted_low=predicted_low,
        predicted_high=predicted_high,
    )
    plan = {
        "schema_version": "phase_a1_modal_local_plan_v1",
        "tool": "experiments/modal_phase_a1_score_gradient_pr101.py",
        "app": APP_NAME,
        "lane_id": "track1_phase_a1_score_gradient",
        "instance_job_id": instance_job_id,
        "created_at_utc": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "predicted_eta_utc": predicted_eta_utc,
        "estimated_cost_usd": estimated_cost,
        "modal_function_timeout_hours": DEFAULT_TIMEOUT_HOURS,
        "requested_timeout_hours": float(timeout_hours),
        "modal_timeout_safety_seconds": MODAL_TIMEOUT_SAFETY_SECONDS,
        "predicted_band": [float(predicted_low), float(predicted_high)],
        "inputs": inputs,
        "params": params,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "auth_eval_device": "cuda",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "remote_or_gpu_eval_started": False,
        "lane_claim_opened": False,
        "modal_app_creation_not_attempted": True,
        "modal_billing_safe": True,
        "ready_for_modal_dispatch_command": not validation_errors,
        "validation_errors": validation_errors,
        "dispatch_command": dispatch_command,
        "dispatch_command_shell": " ".join(dispatch_command),
        "recover_command_after_dispatch": (
            ".venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover "
            f"--label {instance_job_id}"
        ),
        "evidence_grade": "[plan-only]",
        "notes": [
            "Local plan path exists because `modal run --print-only` still creates a Modal app.",
            "No lane claim is required until an actual remote/GPU dispatch is attempted.",
            "If continue_after_nvdec_failure is true, the run may produce a training/build archive but must not claim a CUDA score.",
            "No score promotion is allowed from this plan artifact.",
        ],
    }
    return plan, 0 if not validation_errors else 2


def plan_cli(argv: list[str] | None = None) -> int:
    """Direct Python CLI for local-only A1 plans."""
    parser = argparse.ArgumentParser(
        prog="modal_phase_a1_score_gradient_pr101.py plan",
        description="Write a Modal A1 dispatch plan without app creation, lane claim, or GPU spend.",
    )
    parser.add_argument("--pr101-archive", default=str(DEFAULT_PR101_ARCHIVE))
    parser.add_argument("--pr101-source-dir", default=str(DEFAULT_PR101_SOURCE_DIR))
    parser.add_argument("--video-path", default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--label", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--aux-kl-weight", type=float, default=1.0)
    parser.add_argument("--aux-pixel-l1-weight", type=float, default=0.01)
    parser.add_argument(
        "--checkpoint-selection",
        choices=sorted(CHECKPOINT_SELECTIONS),
        default="final_ema",
        help=(
            "Which training checkpoint to build into the archive. final_ema "
            "preserves historical behavior; best_proxy uses the EMA snapshot "
            "from the lowest training weighted_proxy epoch."
        ),
    )
    parser.add_argument(
        "--continue-after-nvdec-failure",
        action="store_true",
        help=(
            "Plan a training/build-only fallback if Modal T4 compute is healthy "
            "but DALI/NVDEC exact-eval preflight fails. This never promotes a score."
        ),
    )
    parser.add_argument("--train-timeout-hours", type=float, default=3.5)
    parser.add_argument("--build-timeout-minutes", type=float, default=10.0)
    parser.add_argument("--eval-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--timeout-hours", type=float, default=DEFAULT_TIMEOUT_HOURS)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--predicted-low", type=float, default=0.150)
    parser.add_argument("--predicted-high", type=float, default=0.220)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    payload, rc = build_local_plan(
        pr101_archive=args.pr101_archive,
        pr101_source_dir=args.pr101_source_dir,
        video_path=args.video_path,
        label=args.label,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        max_frames=args.max_frames,
        aux_kl_weight=args.aux_kl_weight,
        aux_pixel_l1_weight=args.aux_pixel_l1_weight,
        checkpoint_selection=args.checkpoint_selection,
        continue_after_nvdec_failure=args.continue_after_nvdec_failure,
        train_timeout_hours=args.train_timeout_hours,
        build_timeout_minutes=args.build_timeout_minutes,
        eval_timeout_minutes=args.eval_timeout_minutes,
        timeout_hours=args.timeout_hours,
        cost_cap_usd=args.cost_cap_usd,
        predicted_low=args.predicted_low,
        predicted_high=args.predicted_high,
    )
    if args.json_out:
        _write_json(Path(args.json_out), payload)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rc


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

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id", lane_id,
        "--platform", "modal",
        "--instance-job-id", instance_job_id,
        "--agent", "claude:modal_phase_a1",
        "--predicted-eta-utc", predicted_eta_utc,
        "--status", status,
    ]
    if notes:
        cmd.extend(["--notes", notes])
    if force:
        cmd.append("--force")
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
    score_claim_valid: bool,
) -> int:
    if _returncode_is_zero(rc) and summary.get("passed") is True and score_claim_valid:
        status = "completed_modal_contest_cuda_recovered"
    elif _returncode_is_zero(rc) and summary.get("passed") is True:
        status = "failed_modal_recovered_no_score_claim"
    else:
        status = "failed_modal_recovered"
    notes = (
        f"Phase A1 Modal recover harvested call_id={call_id}; rc={rc!r}; "
        f"stage={summary.get('stage')!r}; passed={summary.get('passed')!r}; "
        f"score_claim_valid={score_claim_valid}. "
        "Terminal claim row closes the dispatch once cached artifacts are recovered."
    )
    return _claim_lane(
        lane_id="track1_phase_a1_score_gradient",
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
        lane_id="track1_phase_a1_score_gradient",
        instance_job_id=instance_job_id,
        predicted_eta_utc=_utc_now_iso(),
        notes=(
            f"Phase A1 Modal recover failed because result cache expired for call_id={call_id}; "
            "terminal failure closes the stale active claim."
        ),
        status="failed_modal_result_cache_expired",
        force=True,
    )


def _close_modal_exception_claim(
    *,
    instance_job_id: str,
    call_id: str,
    exc: BaseException,
) -> int:
    return _claim_lane(
        lane_id="track1_phase_a1_score_gradient",
        instance_job_id=instance_job_id,
        predicted_eta_utc=_utc_now_iso(),
        notes=(
            f"Phase A1 Modal recover failed before artifact harvest for call_id={call_id}; "
            f"exception={type(exc).__name__}: {exc}. Terminal failure closes the active claim."
        ),
        status="failed_modal_recover_exception",
        force=True,
    )


def _write_dispatch_metadata(
    *,
    instance_job_id: str,
    call_id: str,
    paths: dict[str, Path],
    metas: dict[str, dict[str, Any]],
    params: dict[str, Any],
    estimated_cost_usd: float,
    timeout_hours: float,
    predicted_low: float,
    predicted_high: float,
    predicted_eta_utc: str,
) -> Path:
    """Write the canonical Modal dispatch metadata.

    The schema MIRRORS ``modal_train_lane.py``'s ``modal_metadata.json`` so the
    canonical harvester ``tools/harvest_modal_calls.py`` (which globs
    ``lane_*_modal/``) can pick it up. We also write to a NEW directory shape
    ``track1_phase_a1_score_gradient_<ts>_modal/`` under experiments/results/.

    Compatibility note: ``tools/harvest_modal_calls.py`` uses the glob
    ``lane_*_modal``. Our directory shape ``track1_*_modal`` does NOT match.
    To remain harvestable, we ALSO write a sibling symlink directory
    ``lane_track1_phase_a1_score_gradient_<ts>_modal`` pointing at our canonical
    output dir; the harvester picks up the symlink, finds modal_metadata.json,
    and recovers correctly. The recover entrypoint below uses the canonical
    path directly.
    """
    out_dir = RESULT_ROOT / instance_job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    code_snapshot = _mounted_code_snapshot(out_dir)
    payload = {
        "schema_version": 1,
        "app": APP_NAME,
        "lane_id": "track1_phase_a1_score_gradient",
        "instance_job_id": instance_job_id,
        "label": instance_job_id,
        "call_id": call_id,
        "gpu": "T4",
        "paths": {key: str(path) for key, path in sorted(paths.items())},
        "inputs": metas,
        "mounted_code_snapshot": code_snapshot,
        "params": params,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "score_claim": False,  # Becomes True ONLY after harvest succeeds with eval_data.
        "promotion_eligible": False,
        "evidence_grade": "[advisory only — dispatch in flight]",
        "estimated_cost_usd": float(estimated_cost_usd),
        "estimated_duration_hours": float(timeout_hours),
        "predicted_band": [float(predicted_low), float(predicted_high)],
        "predicted_eta_utc": predicted_eta_utc,
        "council_memo_ref": ".omx/research/grand_council_extreme_rigor_track_1_20260508.md",
        "council_decision": "A1 — score-gradient supervision (UNANIMOUS HIGHEST PRIORITY)",
        "lane_script": "scripts/remote_track1_phase_a1_score_gradient_pr101.sh",
        "training_script": "experiments/train_score_gradient_pr101_finetune.py",
        "archive_build_tool": "tools/build_pr101_finetuned_archive.py",
        "auth_eval_device": "cuda",
        "auth_eval_advisory_only": False,
        "recover_command": (
            ".venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover "
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
        # Symlink creation can fail on some filesystems — write a sibling directory
        # with a copy of modal_metadata.json instead so the harvester still finds it.
        legacy_dir.mkdir(parents=True, exist_ok=True)
        _write_json(legacy_dir / "modal_metadata.json", payload)
        (legacy_dir / "modal_call_id.txt").write_text(call_id + "\n")
        print(f"[metadata] symlink unavailable ({exc!r}); copied modal_metadata.json to {legacy_dir}")

    return metadata_path


@app.local_entrypoint()
def main(
    pr101_archive: str = str(DEFAULT_PR101_ARCHIVE),
    pr101_source_dir: str = str(DEFAULT_PR101_SOURCE_DIR),
    video_path: str = str(DEFAULT_VIDEO_PATH),
    label: str | None = None,
    epochs: int = 200,
    steps_per_epoch: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_frames: int = 1200,
    aux_kl_weight: float = 1.0,
    aux_pixel_l1_weight: float = 0.01,
    checkpoint_selection: str = "final_ema",
    continue_after_nvdec_failure: bool = False,
    train_timeout_hours: float = 3.5,
    build_timeout_minutes: float = 10.0,
    eval_timeout_minutes: float = 30.0,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    predicted_low: float = 0.150,
    predicted_high: float = 0.220,
    print_only: bool = False,
    force_claim: bool = False,
) -> None:
    """Dispatch Phase A1 to Modal T4 with `.spawn()` (HARVEST OR LOSE pattern).

    All cost / lane-claim gates run on the LOCAL side BEFORE any GPU is requested.
    """
    pr101_archive_path = Path(pr101_archive).resolve()
    pr101_source_dir_path = Path(pr101_source_dir).resolve()
    video_path_resolved = Path(video_path).resolve()

    # ---- Pre-dispatch gates (cost cap + path checks) ---------------------
    if not pr101_archive_path.is_file():
        raise SystemExit(f"FATAL: --pr101-archive not found: {pr101_archive_path}")
    if not pr101_source_dir_path.is_dir():
        raise SystemExit(f"FATAL: --pr101-source-dir not found: {pr101_source_dir_path}")
    if not video_path_resolved.is_file():
        raise SystemExit(f"FATAL: --video-path not found: {video_path_resolved}")

    timeout_errors = _modal_timeout_validation_errors(
        timeout_hours=timeout_hours,
        train_timeout_hours=train_timeout_hours,
        build_timeout_minutes=build_timeout_minutes,
        eval_timeout_minutes=eval_timeout_minutes,
    )
    if timeout_errors:
        raise SystemExit(f"FATAL: Modal timeout validation failed: {timeout_errors}")

    estimated_cost = HOURLY_RATE_T4_USD * DEFAULT_TIMEOUT_HOURS
    print(
        f"[cost-gate] estimated ${estimated_cost:.2f} for Modal T4 × {DEFAULT_TIMEOUT_HOURS:.1f}h "
        f"(cap ${cost_cap_usd:.2f})"
    )
    if estimated_cost > cost_cap_usd:
        raise SystemExit(
            f"FATAL: estimated cost ${estimated_cost:.2f} exceeds cap ${cost_cap_usd:.2f}; abort"
        )
    if checkpoint_selection not in CHECKPOINT_SELECTIONS:
        raise SystemExit(
            f"FATAL: checkpoint_selection={checkpoint_selection!r}; "
            f"expected one of {sorted(CHECKPOINT_SELECTIONS)}"
        )

    # ---- Read inputs (verifies sha + size) -------------------------------
    pr101_archive_bytes, pr101_archive_sha, pr101_archive_size = _read_input(
        pr101_archive_path, "PR101 archive"
    )
    pr101_source_zip_bytes, pr101_source_zip_sha, pr101_source_zip_size = _read_dir_as_tarball(
        pr101_source_dir_path, "PR101 source dir"
    )
    video_bytes, video_sha, video_size = _read_input(video_path_resolved, "video")

    print(f"[inputs] pr101_archive: {pr101_archive_size} B sha={pr101_archive_sha[:16]}")
    print(f"[inputs] pr101_source_zip: {pr101_source_zip_size} B sha={pr101_source_zip_sha[:16]}")
    print(f"[inputs] video: {video_size} B sha={video_sha[:16]}")

    # ---- Build instance_job_id (timestamped lane id) ---------------------
    if label:
        instance_job_id = _safe_label(label)
    else:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        instance_job_id = f"track1_phase_a1_score_gradient_{timestamp}_modal"

    started_at_utc = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    predicted_eta_utc = (
        dt.datetime.now(tz=dt.UTC) + dt.timedelta(hours=DEFAULT_TIMEOUT_HOURS)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ---- Build container function args ------------------------------------
    train_timeout_seconds = max(120, int(float(train_timeout_hours) * 3600))
    build_timeout_seconds = max(60, int(float(build_timeout_minutes) * 60))
    eval_timeout_seconds = max(120, int(float(eval_timeout_minutes) * 60))
    max_seconds = max(120, int(DEFAULT_TIMEOUT_HOURS * 3600))

    params = {
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "max_frames": int(max_frames),
        "aux_kl_weight": float(aux_kl_weight),
        "aux_pixel_l1_weight": float(aux_pixel_l1_weight),
        "checkpoint_selection": checkpoint_selection,
        "continue_after_nvdec_failure": bool(continue_after_nvdec_failure),
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
            "score_claim": False,
            "remote_or_gpu_eval_started": False,
            "continue_after_nvdec_failure": bool(continue_after_nvdec_failure),
            "predicted_eta_utc": predicted_eta_utc,
            "inputs": {
                "pr101_archive": {"sha256": pr101_archive_sha, "bytes": pr101_archive_size},
                "pr101_source_zip": {"sha256": pr101_source_zip_sha, "bytes": pr101_source_zip_size},
                "video": {"sha256": video_sha, "bytes": video_size},
            },
        }, indent=2))
        return

    # ---- Open lane claim BEFORE GPU spend (CLAUDE.md NON-NEGOTIABLE) -----
    notes = (
        f"Phase A1 score-gradient supervision PR101 fine-tune on Modal T4; "
        f"council UNANIMOUS HIGHEST PRIORITY; predicted=[{predicted_low}, {predicted_high}]; "
        f"cost=${estimated_cost:.2f}; archive={pr101_archive_path.name}"
    )
    claim_rc = _claim_lane(
        lane_id="track1_phase_a1_score_gradient",
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
        call = run_phase_a1_t4.spawn(
            pr101_archive_bytes,
            pr101_archive_sha,
            pr101_archive_size,
            pr101_source_zip_bytes,
            pr101_source_zip_sha,
            pr101_source_zip_size,
            video_bytes,
            video_sha,
            video_size,
            instance_job_id,
            int(epochs),
            int(steps_per_epoch),
            int(batch_size),
            float(lr),
            int(max_frames),
            float(aux_kl_weight),
            float(aux_pixel_l1_weight),
            str(checkpoint_selection),
            bool(continue_after_nvdec_failure),
            int(train_timeout_seconds),
            int(build_timeout_seconds),
            int(eval_timeout_seconds),
            int(max_seconds),
        )
    except Exception as exc:
        # If `.spawn` fails (e.g., insufficient credit, app build failure), close
        # the lane claim terminally so re-fire is unblocked.
        _claim_lane(
            lane_id="track1_phase_a1_score_gradient",
            instance_job_id=instance_job_id,
            predicted_eta_utc=started_at_utc,
            notes=(
                f"Phase A1 Modal .spawn() failed: {type(exc).__name__}: {exc!r}. "
                "Lane claim closed terminally so re-fire is unblocked once Modal credit is restored."
            ),
            status="failed_modal_spawn_submission",
            force=True,
        )
        raise SystemExit(
            f"FATAL: Modal `.spawn()` failed: {type(exc).__name__}: {exc}. "
            "If the message includes 'insufficient' or 'credit' or 'balance', "
            "Modal credits may be exhausted (similar to the Lightning failure earlier today). "
            "Surface to operator; do NOT auto-pivot to other providers per dispatch ticket."
        ) from exc

    metadata_path = _write_dispatch_metadata(
        instance_job_id=instance_job_id,
        call_id=call.object_id,
        paths={
            "pr101_archive": pr101_archive_path,
            "pr101_source_dir": pr101_source_dir_path,
            "video": video_path_resolved,
        },
        metas={
            "pr101_archive": {"sha256": pr101_archive_sha, "bytes": pr101_archive_size},
            "pr101_source_zip": {"sha256": pr101_source_zip_sha, "bytes": pr101_source_zip_size},
            "video": {"sha256": video_sha, "bytes": video_size},
        },
        params=params,
        estimated_cost_usd=estimated_cost,
        timeout_hours=DEFAULT_TIMEOUT_HOURS,
        predicted_low=float(predicted_low),
        predicted_high=float(predicted_high),
        predicted_eta_utc=predicted_eta_utc,
    )

    print(f"\n✓ DISPATCHED Modal T4 call_id={call.object_id}")
    print(f"  instance_job_id: {instance_job_id}")
    print(f"  metadata:        {metadata_path}")
    print(f"  estimated cost:  ${estimated_cost:.2f}")
    print(
        f"  predicted band:  [{predicted_low}, {predicted_high}] "
        "[planning forecast; not evidence]"
    )
    print(f"  predicted ETA:   {predicted_eta_utc}")
    print()
    print("  Recover (within 24h of dispatch — Modal result-cache TTL):")
    print(
        "    .venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover "
        f"--label {instance_job_id}"
    )
    print()
    print("  Or sweep all dispatched Modal calls:")
    print("    .venv/bin/python tools/harvest_modal_calls.py")
    print()
    print(
        "  Stream remote logs:  .venv/bin/modal app logs <app-id>  "
        "(see `modal app list` for the comma-phase-a1-score-gradient app)"
    )


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
        close_rc = _close_modal_exception_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
            exc=exc,
        )
        if close_rc != 0:
            print(f"FATAL: failed to close exception Modal claim rc={close_rc}", file=sys.stderr)
            return 6
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

    eval_data = result.get("eval_data") if isinstance(result.get("eval_data"), dict) else None
    evidence = _recover_evidence_summary(eval_data, summary)
    metrics = evidence["metrics"]
    if eval_data:
        if evidence["score_claim_valid"]:
            print(
                f"[recover] [contest-CUDA] score={metrics['score']} "
                f"pose={metrics['pose_avg']} seg={metrics['seg_avg']} "
                f"rate={metrics['rate']}"
            )
        else:
            print(
                f"[recover] [exact-eval incomplete] score={metrics['score']} "
                f"pose={metrics['pose_avg']} seg={metrics['seg_avg']} "
                f"rate={metrics['rate']} blockers={evidence['claim_blockers']}"
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
        "score": metrics["score"],
        "pose_avg": metrics["pose_avg"],
        "seg_avg": metrics["seg_avg"],
        "rate": metrics["rate"],
        "rate_unscaled": metrics["rate_unscaled"],
        "evidence_grade": evidence["evidence_grade"],
        "score_claim": evidence["score_claim"],
        "score_claim_valid": evidence["score_claim_valid"],
        "exact_cuda_eval_complete": evidence["exact_cuda_eval_complete"],
        "promotion_eligible": evidence["promotion_eligible"],
        "rank_or_kill_eligible": evidence["rank_or_kill_eligible"],
        "auth_eval_blockers": evidence["auth_eval_blockers"],
        "claim_blockers": evidence["claim_blockers"],
        "dispatch_blockers": evidence["dispatch_blockers"],
        "canonical_score_source": evidence["canonical_score_source"],
        "eval_archive_size_bytes": evidence["eval_archive_size_bytes"],
        "n_samples": evidence["n_samples"],
        "score_axis": evidence["score_axis"],
        "evidence_semantics": evidence["evidence_semantics"],
        "lane_tag": evidence["lane_tag"],
        "tag": summary.get("tag"),
    })
    print(f"[recover] summary saved: {summary_path}")
    close_rc = _close_modal_recovery_claim(
        instance_job_id=instance_job_id,
        call_id=call_id,
        rc=rc,
        summary=summary,
        score_claim_valid=bool(evidence["score_claim_valid"]),
    )
    if close_rc != 0:
        print(f"FATAL: failed to close recovered Modal claim rc={close_rc}", file=sys.stderr)
        return 6
    return 0 if _returncode_is_zero(rc) and evidence["score_claim_valid"] else 1


# ---------------------------------------------------------------------------
# CLI helpers. Dispatch itself remains the Modal local_entrypoint.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When invoked as `python experiments/modal_phase_a1_score_gradient_pr101.py recover ...`
    # we handle the recover subcommand directly (without going through `modal run`).
    if len(sys.argv) >= 2 and sys.argv[1] == "recover":
        parser = argparse.ArgumentParser(prog="modal_phase_a1_score_gradient_pr101.py recover")
        parser.add_argument("--label", required=True,
                            help="instance_job_id (e.g., track1_phase_a1_score_gradient_<ts>_modal)")
        args = parser.parse_args(sys.argv[2:])
        sys.exit(recover(args.label))
    elif len(sys.argv) >= 2 and sys.argv[1] == "plan":
        sys.exit(plan_cli(sys.argv[2:]))
    else:
        print(
            "USAGE:\n"
            "  Plan only: .venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py "
            "plan [args]\n"
            "  Dispatch:  PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\\n"
            "             experiments/modal_phase_a1_score_gradient_pr101.py [args]\n"
            "  Recover:   .venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py "
            "recover --label <instance_job_id>",
            file=sys.stderr,
        )
        sys.exit(2)
