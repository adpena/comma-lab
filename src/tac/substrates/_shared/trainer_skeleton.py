"""Canonical substrate-trainer utilities.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + the
2026-05-13 dedup audit (`feedback_canon_dedup_1_LANDED_20260513.md`), the
14 substrate trainers under ``experiments/train_substrate_*.py`` share a
small set of truly substrate-agnostic helpers. The dominant variants of
each helper are reproduced here as the canonical implementation.

Scope:
    Substrate-agnostic byte-faithful utilities only. Substrate-specific
    archive grammar, runtime emission, and architecture stay in each
    trainer (per HNeRV parity discipline lessons 3 + 4 + 5).

Compatibility:
    All 14 substrate trainers under ``experiments/train_substrate_*.py``
    import from this module (migration completed 2026-05-13).

Exception-type policy:
    Helper invariants (missing upstream frame_utils.py, pyav unavailable,
    insufficient decoded frames) raise plain ``RuntimeError`` rather than a
    dedicated ``SubstrateError`` subclass. Per R6-1 (2026-05-13 recursive
    review): there is no architectural reason to invent a dedicated
    exception class for canonical-helper invariants — these signal
    operator-environment misconfiguration (substrate's call site cannot
    recover and should fail loud). ``FileNotFoundError`` is used for
    missing files where the standard exception type fits exactly.
    Sibling ``tac.preflight`` raises ``PreflightError`` /
    ``MetaBugViolation`` because those are CI-gating violations with
    structured suppression semantics; this module's failures are
    environment-level. Cross-module catch should use ``Exception``
    rather than ``PreflightError`` when wrapping substrate trainer calls.

Cross-refs:
    - CLAUDE.md "Beauty, simplicity, and developer experience"
    - Catalog #146 (Phase 1 trainer runtime contract — substrate-specific
      ``_write_runtime`` stays per-trainer)
    - Catalog #164 (scorer preprocess — substrate score-aware loss stays
      per-trainer; the canonical helper is ``score_aware_common.py``)
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]

EVAL_HW: tuple[int, int] = (384, 512)

# Canonical (axis, gpu_token) -> hardware_substrate map, aligned with
# `tac.continual_learning.TAG_HARDWARE_REQUIREMENT` accepted-substrate set.
# Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #127
# (`check_authoritative_tag_requires_custody_metadata`): the posterior write
# MUST record the actual GPU substrate the dispatch ran on, not a default
# placeholder. The 14 substrate trainers previously hardcoded
# `"linux_x86_64_t4"` regardless of the dispatched GPU — for A100/4090/H100
# dispatches that produces a silent custody mislabel.
_GPU_TOKEN_TO_SUBSTRATE: dict[str, str] = {
    "t4": "linux_x86_64_t4",
    "rtx_4090": "linux_x86_64_4090",
    "4090": "linux_x86_64_4090",
    "a100": "linux_x86_64_a100",
    "a100-40gb": "linux_x86_64_a100",
    "a100-80gb": "linux_x86_64_a100",
    "h100": "linux_x86_64_h100",
    "a10g": "linux_x86_64_a10g",
    "l40s": "linux_x86_64_l40s",
}


def pin_seeds(seed: int) -> None:
    """Deterministic seed pinning (torch + python + numpy if present)."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def torch_version_string() -> str:
    """Return torch version (or '<unknown>' on import failure)."""
    try:
        import torch

        return f"{torch.__version__}"
    except Exception:
        return "<unknown>"


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def git_head_sha(repo_root: Path | None = None) -> str:
    """Return git HEAD sha (or '<unknown>' on failure)."""
    root = repo_root if repo_root is not None else REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "<unknown>"


def utc_now_iso() -> str:
    """UTC ISO-8601 timestamp suitable for provenance/stage logs."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def vendor_shared_inflate_runtime(
    submission_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Copy the shared raw-output inflate helper into a submission runtime tree.

    Substrate archives vendor only the minimal inflate-time package. Any
    substrate runtime importing ``tac.substrates._shared.inflate_runtime`` must
    call this helper from its trainer ``_write_runtime`` path, otherwise local
    source-tree tests pass while contest auth eval fails with an import error.
    """

    root = repo_root if repo_root is not None else REPO_ROOT
    shared_src = root / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py"
    if not shared_src.is_file():
        raise FileNotFoundError(f"shared inflate runtime helper missing: {shared_src}")
    shared_dst = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dst.mkdir(parents=True, exist_ok=True)
    (shared_dst / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(shared_src, shared_dst / "inflate_runtime.py")


def detect_hardware_substrate(
    *,
    axis: str = "cuda",
    substrate_tag: str,
    provenance_path: Path | None = None,
    env_var_candidates: tuple[str, ...] = (),
) -> str:
    """Resolve the canonical ``hardware_substrate`` token for posterior writes.

    Per CLAUDE.md SIREN audit (2026-05-13) CRITICAL #1 + Catalog #190
    (``check_substrate_trainer_does_not_hardcode_hardware_substrate``): the
    14 substrate trainers used to hardcode ``"linux_x86_64_t4"`` regardless
    of the dispatched GPU. This helper resolves the substrate dynamically
    from (1) the remote driver's ``provenance.json``, (2) environment-var
    candidates (typically ``<SUBSTRATE>_GPU`` then ``MODAL_GPU``), or
    (3) live ``nvidia-smi`` query. Falls back to
    ``"linux_x86_64_unknown_cuda"`` with a stderr-warning if all sources
    are silent — never silently mislabels.

    Args:
        axis: ``"cuda"`` (default) or ``"cpu"``. Drives the lookup table.
        substrate_tag: Short label used in the warning banner (e.g. ``"siren"``).
        provenance_path: Optional path to a substrate-emitted
            ``provenance.json`` carrying a ``gpu_name`` field (typically from
            ``scripts/remote_lane_substrate_<id>.sh``).
        env_var_candidates: Ordered tuple of env var names to consult in
            priority order (e.g. ``("SIREN_GPU", "MODAL_GPU")``).

    Returns:
        Canonical substrate token from
        ``tac.continual_learning.TAG_HARDWARE_REQUIREMENT``: e.g.
        ``"linux_x86_64_a100"``, ``"linux_x86_64_t4"``, ``"linux_x86_64_4090"``,
        ``"linux_x86_64_unknown_cuda"`` (fallback). For ``axis="cpu"``,
        returns ``"linux_x86_64_modal_cpu"`` (Linux x86_64 non-GHA) or
        ``"unknown_cpu"``.
    """
    if axis == "cpu":
        return "linux_x86_64_modal_cpu"
    if axis != "cuda":
        return "unknown"

    gpu_token = ""

    # (1) Prefer provenance.json (remote driver writes the actual GPU name).
    if provenance_path is not None:
        try:
            if provenance_path.is_file():
                prov = json.loads(provenance_path.read_text())
                gpu_name = str(prov.get("gpu_name") or "").strip().lower()
                gpu_token = gpu_name
        except Exception:
            gpu_token = ""

    # (2) Environment-var ladder (operator wrapper / Modal env_overrides).
    if not gpu_token:
        for env_name in env_var_candidates:
            value = os.environ.get(env_name)
            if value:
                gpu_token = value.strip().lower()
                break

    # (3) Live nvidia-smi probe (CUDA-only).
    if not gpu_token:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                gpu_token = proc.stdout.strip().splitlines()[0].lower()
        except Exception:
            gpu_token = ""

    # Map the token to a canonical substrate.
    for key, substrate in _GPU_TOKEN_TO_SUBSTRATE.items():
        if key in gpu_token:
            return substrate

    # All sources silent or unrecognized GPU.
    import sys as _sys

    print(
        f"[{substrate_tag}] WARN: hardware_substrate detection found no GPU "
        f"token (provenance={provenance_path}, env_candidates={env_var_candidates}, "
        f"resolved={gpu_token!r}); falling back to 'linux_x86_64_unknown_cuda'. "
        "Posterior write will record this fallback explicitly per CLAUDE.md "
        "forbidden-empirical-claim-without-evidence-tag discipline.",
        file=_sys.stderr,
    )
    return "linux_x86_64_unknown_cuda"


class StageLog:
    """Append-only stage tracker for provenance ``stage_log`` blocks.

    Each substrate trainer's ``_full_main`` builds a stage_log dict-list
    consumed by the provenance.json writer. This helper canonicalizes the
    pattern.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def stage(self, name: str) -> None:
        """Append a stage marker keyed to the current UTC time."""
        self._entries.append({"stage": name, "at": utc_now_iso()})

    def entries(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the recorded stage entries."""
        return list(self._entries)


def device_or_die(name: str, *, smoke: bool, substrate_tag: str):
    """Resolve compute device or raise SystemExit.

    Args:
        name: One of {'cuda', 'cpu'}.
        smoke: True iff this is the smoke path (CPU permitted).
        substrate_tag: Short label used in error messages (e.g. 'cool_chic').

    Per CLAUDE.md "MPS auth eval is NOISE" + "EMA — non-negotiable":
    cuda is the default for full training, cpu is permitted only with
    --smoke, mps is FORBIDDEN.
    """
    import torch

    if name == "cpu":
        if not smoke:
            raise SystemExit(
                f"[{substrate_tag}] --device cpu is permitted only with "
                "--smoke per CLAUDE.md 'MPS auth eval is NOISE' + 'EMA — "
                "non-negotiable' + full-training-needs-CUDA convention. "
                "Use --device cuda for promotion-grade training. CPU smoke is "
                "allowed only when deterministic-bytes acceptable."
            )
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                f"[{substrate_tag}] --device cuda requested but cuda not available"
            )
        # Canonical substrate-trainer fast-math policy. Catalog #178 forbids
        # each trainer from rediscovering or silently omitting this Ampere/
        # Hopper speed path. TF32 affects CUDA matmul/convolution kernels only;
        # exact score authority still comes from archive/runtime auth eval.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    raise SystemExit(f"[{substrate_tag}] unknown --device {name!r}")


def load_upstream_yuv420_to_rgb(*, substrate_tag: str, repo_root: Path | None = None):
    """Load upstream/frame_utils.py's ``yuv420_to_rgb`` without patching upstream.

    Per CLAUDE.md "Non-Negotiable Upstream Rule": upstream is the source of
    truth; we re-use the canonical contest-faithful decode path (BT.601 /
    no in-place ops) without modifying upstream files.

    Args:
        substrate_tag: Short label used for the importlib spec name to
            avoid collisions when multiple substrates share a process.
        repo_root: Optional override; defaults to repo root.
    """
    import importlib.util

    root = repo_root if repo_root is not None else REPO_ROOT
    frame_utils_path = root / "upstream" / "frame_utils.py"
    if not frame_utils_path.is_file():
        raise FileNotFoundError(
            f"upstream/frame_utils.py not found at {frame_utils_path}; "
            "verify --upstream-dir is correct."
        )
    spec = importlib.util.spec_from_file_location(
        f"pact_{substrate_tag}_upstream_frame_utils", frame_utils_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"unable to load upstream frame_utils.py from {frame_utils_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    substrate_tag: str,
    max_pairs: int | None = None,
    repo_root: Path | None = None,
):
    """Decode real contest pairs (0,1), (2,3), ... at EVAL_HW (384, 512).

    Returns:
        torch.Tensor shape ``(N, 2, 3, 384, 512)`` float32 in ``[0, 255]``.

    Raises:
        FileNotFoundError: ``video_path`` is missing.
        RuntimeError: pyav not installed, or video yielded fewer frames
            than ``n_pairs * 2`` (with ``max_pairs`` accounted for).

    Per CLAUDE.md Catalog #114 + the HNeRV parity lesson L1 (score-aware
    substrate trains against real contest video, NOT synthetic data).
    """
    import torch
    import torch.nn.functional as F

    if not video_path.is_file():
        raise FileNotFoundError(
            f"real target video not found: {video_path}. Non-smoke "
            "training requires upstream/videos/0.mkv."
        )
    try:
        import av
    except Exception as exc:
        raise RuntimeError(
            f"pyav (`av`) is required for non-smoke {substrate_tag} "
            "training; run `uv pip install av`"
        ) from exc

    yuv420_to_rgb = load_upstream_yuv420_to_rgb(
        substrate_tag=substrate_tag, repo_root=repo_root
    )
    target_pairs = n_pairs if max_pairs is None else min(n_pairs, max_pairs)
    frames_needed = target_pairs * 2
    frames_chw: list = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb_hwc = yuv420_to_rgb(frame)
            rgb_chw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(
                rgb_chw, size=EVAL_HW, mode="bilinear", align_corners=False
            )
            frames_chw.append(resized.squeeze(0).contiguous())
            if len(frames_chw) >= frames_needed:
                break
    finally:
        container.close()
    if len(frames_chw) < frames_needed:
        raise RuntimeError(
            f"{video_path} yielded {len(frames_chw)} frame(s), "
            f"need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


__all__ = [
    "EVAL_HW",
    "REPO_ROOT",
    "StageLog",
    "decode_real_pairs",
    "detect_hardware_substrate",
    "device_or_die",
    "git_head_sha",
    "load_upstream_yuv420_to_rgb",
    "pin_seeds",
    "sha256_bytes",
    "torch_version_string",
    "utc_now_iso",
]
