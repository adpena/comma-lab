#!/usr/bin/env python3
"""PARADIGM-δεζ Track 1 — Ballé hyperprior + 128K decoder end-to-end trainer.

This is the **build-only Phase 1 trainer** for T1. It takes the frozen A1
latent table as input, trains:

  - a fresh 128K-parameter Quantizr-class FiLM decoder (
    :class:`tac.paradigm_delta_epsilon_zeta.Decoder128K`)
  - a Ballé 2018 ScaleHyperprior on the latent stream (
    :class:`tac.paradigm_delta_epsilon_zeta.BalleHyperpriorWrapper`)
  - jointly under a Boyd-style adaptive-ρ Lagrangian-ADMM coordinator (
    :class:`tac.paradigm_delta_epsilon_zeta.JointLagrangianADMM`)

against the Phase 1 pixel proxy. Phase 2 wires real scorer-domain objectives
and exact eval dispatch once the runtime is byte-closed.

Phase 1 NOTE: the emitted archive/runtime is a **build-verification scaffold**,
not a contest-submission format. ``research_only=true`` until Phase 2 replaces
pickle/state-dict payloads with a deterministic byte-optimised wire format.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT:
  archive_grammar: single ZIP member ``x`` with scaffold sections
  parser_section_manifest: section lengths are emitted in provenance only
  inflate_runtime_loc_budget: scaffold runtime exceeds packet budget
  runtime_dep_closure: PACT_RUNTIME_DEPENDENCY_ROOT declares repo tac dependency
  export_format: research_only_no_export
  score_aware_loss: Phase 1 real-pixel eval-roundtrip proxy; Phase 2 scorer loss
  bolt_on_loc_budget: substrate_engineering
  no_op_detector_planned: exact old/new archive SHA plus inflate consumption

CLAUDE.md non-negotiables — wired through this file
---------------------------------------------------

- **EMA at decay 0.997 with snapshot+restore at eval time** — every
  evaluator call uses the EMA shadow temporarily; live weights are restored
  before training continues.
- **eval_roundtrip = True** — the proxy loss simulates the inflate roundtrip
  (384→874→uint8→384) so the proxy/auth gap stays bounded.
- **noise_std = 0.5** — straight-through quantisation noise applied to the
  decoder output to approximate the int8 cast.
- **NEVER MPS authoritative** — the trainer raises on ``--device mps`` and
  refuses to write ``[contest-CUDA]`` tags from any non-CUDA forward pass.
- **Auth eval is fail-closed for Phase 1** — the emitted runtime is
  ``research_only_no_export`` until Phase 2 replaces the scaffold pickle
  format with a contest-hermetic inflate contract. ``--auth-eval`` is refused
  before training so no scaffold archive can be promoted accidentally.
- **Predicted scores tagged** — the run manifest carries ``score_band:
  "[predicted; Phase 1 scaffold; not yet empirical]"`` until a
  contest-CUDA result lands.

Smoke mode (Phase 1 build verification)
---------------------------------------

``--smoke`` runs ONE epoch on ONE pair, builds a deterministic
``smoke_archive.zip`` (NOT submission-quality), and exits with rc=0 if the
end-to-end pipeline closed without error. Used by the dispatcher's local
preflight before any GPU spend.

CLI surface (DO NOT INVENT FLAGS — verified by tests)
-----------------------------------------------------

  --output-dir         where to write artifacts (required)
  --device             cuda|cpu (mps refused)
  --epochs             3000 default for Q-FAITHFUL
  --batch-size         16 default
  --learning-rate      1e-4 default
  --aux-learning-rate  1e-3 default (EntropyBottleneck quantile-loss)
  --ema-decay          0.997 default (CLAUDE.md non-negotiable)
  --rate-target-bytes  80000 default
  --seg-target         7e-4 default
  --pose-target        1.7e-4 default
  --rho-init           1.0 default
  --eval-every-epochs  100 default
  --auth-eval          refused in Phase 1 until runtime export closes
  --video-path         real contest video for non-smoke target pixels
  --target-pixels-path optional pre-extracted real target tensor
  --max-target-pairs   optional real-target pair cap for local debug
  --smoke              build-verification mode (1 epoch, 1 pair)
  --seed               20 default (matches PYTHONHASHSEED canonical)
  --canonical-a1-relpath  override A1 canonical dir (default
                       experiments/results/A1_canonical)
  --enable-t13-sqrt-n-budget  opt-in Fridrich √n per-pair latent budget hook
                       (default False; backward-compat). When enabled, the
                       trainer queries `tac.joint_source_rd_bound.per_pair_sqrt_n_budget`
                       to compute the undetectable-bits-per-pair budget for
                       the latent stream and shrinks ``rate_target_bytes``
                       accordingly. Predicted impact tag:
                       ``[predicted; T13 Fridrich sqrt-n latent shrink]``.
                       See memory `feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`.
  --t13-alpha          Fridrich proportionality constant (default 1.0; see
                       Ker-Pevný-Fridrich 2008).
  --t13-current-bits-per-pair  caller-supplied estimate of the trainer's
                       current per-pair latent rate (bits/pair). Default 3.0
                       per A1-substrate empirical anchor (per memo §6).
                       Determines how much rate to reallocate.
  --enable-t19-adaptive-rho  opt-in Boyd §3.4.1 / He-Yang 2000 adaptive-ρ
                       update via the standalone
                       `tac.joint_admm_coordinator.adaptive_rho_step` helper
                       (default False; backward-compat per coherence council
                       recommendation). Predicted impact tag:
                       ``[predicted; T19 adaptive ρ 2-3× convergence speedup;
                       not direct score]``.
                       See memory `feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`.
  --t19-tau-grow       T19 ρ-growth factor (default 2.0; Boyd canonical).
  --t19-tau-shrink     T19 ρ-shrink factor (default 0.5; Boyd canonical).

Lane class
----------

This trainer is a SUBSTRATE-ENGINEERING lane (per CLAUDE.md HNeRV parity
discipline lesson #7). It builds the score-aware substrate that downstream
representation/codec lanes consume; it does not itself emit a representation
into the contest packet without further bolt-ons. ``lane_class=substrate_engineering``
is recorded in the run manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.paradigm_delta_epsilon_zeta import (  # noqa: E402
    PARADIGM_DELTA_EPS_ZETA_PROVENANCE,
    PARADIGM_DELTA_EPS_ZETA_VERSION,
    BalleHyperpriorConfig,
    BalleHyperpriorWrapper,
    Decoder128K,
    Decoder128KConfig,
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    build_balle_hyperprior,
    build_decoder_128k,
    load_frozen_a1_encoder,
)
from tac.training import EMA  # canonical EMA class, decay default 0.997


CONTEST_AUTH_EVAL_RELATIVE = "experiments/contest_auth_eval.py"
DISPATCH_CLAIMS_RELATIVE = ".omx/state/active_lane_dispatch_claims.md"
INFLATE_ROUNDTRIP_CAMERA_HW = (874, 1164)
EVAL_HW = (384, 512)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def eval_roundtrip_pixel_l1(
    decoded: torch.Tensor, target_pixels: torch.Tensor, *, noise_std: float
) -> torch.Tensor:
    """Proxy distortion that simulates the inflate roundtrip + uint8 cast.

    decoded: (B, 2, 3, H_eval, W_eval) in [0, 255] (sigmoid * 255)
    target_pixels: same shape, [0, 255]

    Per CLAUDE.md "eval_roundtrip — non-negotiable", the proxy MUST simulate
    the contest's inflate→evaluate path. We:
      1. add small noise to mimic the int8 quantisation residual,
      2. interpolate to camera resolution and back to eval resolution
         (bicubic up + bicubic down) to mimic the inflate pipeline.
    """
    B, P, C, H, W = decoded.shape
    flat = decoded.reshape(B * P, C, H, W)
    if noise_std > 0:
        flat = flat + noise_std * torch.randn_like(flat)
    up = F.interpolate(
        flat,
        size=INFLATE_ROUNDTRIP_CAMERA_HW,
        mode="bicubic",
        align_corners=False,
    )
    # Contest evaluate.py downsamples GT to (384, 512) for SegNet/PoseNet.
    down = F.interpolate(up, size=EVAL_HW, mode="bicubic", align_corners=False)
    down = down.clamp(0.0, 255.0).reshape(B, P, C, H, W)
    return F.l1_loss(down, target_pixels)


# ---------------------------------------------------------------------------
# Smoke target generator (used in --smoke mode without real video data)
# ---------------------------------------------------------------------------


def make_smoke_target(n_pairs: int, latent_dim: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic latent + target frame pair for build-verification only.

    PER CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", this MUST
    be reachable only behind ``--smoke`` and any score derived from it MUST
    be tagged ``[smoke synthetic; not measurable]``.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    latents = torch.randn((n_pairs, latent_dim), generator=g)
    targets = torch.randint(0, 256, (n_pairs, 2, 3, *EVAL_HW), generator=g).float()
    return latents, targets


def _load_upstream_yuv420_to_rgb():
    """Load upstream's PyAV RGB conversion helper without patching upstream."""
    import importlib.util

    frame_utils_path = REPO_ROOT / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location(
        "pact_t1_upstream_frame_utils", frame_utils_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load upstream frame_utils.py from {frame_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def load_real_target_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """Decode real contest frame pairs for non-smoke training.

    This mirrors the upstream AVVideoDataset pair order: non-overlapping
    `(0,1), (2,3), ...` pairs, resized to the trainer's `(384, 512)` proxy
    resolution and returned as `(N, 2, 3, H, W)` float32 in `[0, 255]`.
    Synthetic targets are forbidden outside `--smoke`.
    """
    if not video_path.is_file():
        raise FileNotFoundError(
            f"real target video not found: {video_path}. Non-smoke T1 training "
            "requires upstream/videos/0.mkv or --target-pixels-path."
        )
    try:
        import av  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyav (`av`) is required for non-smoke T1 training") from exc

    yuv420_to_rgb = _load_upstream_yuv420_to_rgb()
    target_pairs = n_pairs if max_pairs is None else min(n_pairs, max_pairs)
    frames_needed = target_pairs * 2
    frames_chw: list[torch.Tensor] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb_hwc = yuv420_to_rgb(frame)
            rgb_chw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(
                rgb_chw,
                size=EVAL_HW,
                mode="bilinear",
                align_corners=False,
            )
            frames_chw.append(resized.squeeze(0).contiguous())
            if len(frames_chw) >= frames_needed:
                break
    finally:
        container.close()
    if len(frames_chw) < frames_needed:
        raise RuntimeError(
            f"{video_path} yielded {len(frames_chw)} frame(s), need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


def load_target_pixels_from_path(path: Path) -> torch.Tensor:
    """Load pre-extracted real target pixels from a torch payload."""
    if not path.is_file():
        raise FileNotFoundError(f"--target-pixels-path not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("target_pixels", "targets", "frame_pairs", "pairs"):
            value = payload.get(key)
            if torch.is_tensor(value):
                payload = value
                break
    if not torch.is_tensor(payload):
        raise ValueError(
            f"{path} did not contain a target tensor or one of "
            "target_pixels/targets/frame_pairs/pairs"
        )
    tensor = payload.float()
    if tensor.ndim != 5:
        raise ValueError(f"target pixels must have 5 dims; got shape {tuple(tensor.shape)}")
    # Accept either (N, 2, H, W, 3) or (N, 2, 3, H, W).
    if tensor.shape[2] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
    if tensor.shape[1:] != (2, 3, *EVAL_HW):
        raise ValueError(
            f"target pixels must have shape (N, 2, 3, {EVAL_HW[0]}, {EVAL_HW[1]}); "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "mps":
        raise SystemExit(
            "[t1] --device mps refused per CLAUDE.md MPS-NOISE rule. "
            "Use cuda for authoritative work or cpu for smoke."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[t1] --device cuda requested but cuda not available")
    return torch.device(name)


def _canonical_dir_name_from_relpath(value: str) -> str:
    """Return the canonical A1 directory name from a repo-relative path.

    ``load_frozen_a1_encoder`` intentionally accepts a directory name under
    ``experiments/results``. The trainer CLI exposes the more operator-friendly
    relpath, so normalize here and fail closed on ambiguous paths.
    """
    path = Path(value)
    parts = path.parts
    if parts == (path.name,):
        return path.name
    expected_prefix = ("experiments", "results")
    if len(parts) == 3 and parts[:2] == expected_prefix:
        return parts[2]
    raise SystemExit(
        "--canonical-a1-relpath must be either A1_canonical or "
        "experiments/results/<canonical_name>; got "
        f"{value!r}"
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)


def _active_claim_rows(*, lane_id: str, claims_path: Path) -> list[dict]:
    """Return active dispatch-claim rows for ``lane_id`` via the canonical helper."""
    helper = REPO_ROOT / "tools" / "claim_lane_dispatch.py"
    if not helper.exists():
        raise SystemExit(f"[t1] dispatch-claim helper not found: {helper}")
    if not claims_path.exists():
        raise SystemExit(f"[t1] dispatch-claims ledger not found: {claims_path}")
    cmd = [
        sys.executable,
        str(helper),
        "summary",
        "--claims-path", str(claims_path),
        "--format", "json",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            "[t1] dispatch-claim summary failed "
            f"(rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[t1] dispatch-claim summary emitted invalid JSON: {exc}") from exc
    return [row for row in payload.get("active", []) if row.get("lane_id") == lane_id]


def require_active_dispatch_claim(*, lane_id: str | None, claims_path: Path) -> None:
    """Fail closed unless an active same-lane claim exists for auth eval."""
    if not lane_id:
        raise SystemExit(
            "[t1] --auth-eval requires --dispatch-lane-id. Claim the lane "
            "with tools/claim_lane_dispatch.py before running eval."
        )
    rows = _active_claim_rows(lane_id=lane_id, claims_path=claims_path)
    if not rows:
        raise SystemExit(
            f"[t1] --auth-eval refused: no active dispatch claim for lane_id={lane_id!r} "
            f"in {claims_path}. Claim the lane before eval."
        )


def _restore_state_dict_with_loose_buffers(module: torch.nn.Module, state: dict) -> None:
    """``load_state_dict`` variant tolerant of buffer-shape changes.

    The CompressAI ``EntropyBottleneck`` mutates buffer shapes inside
    ``update()``. The default ``load_state_dict`` refuses to load tensors of
    different shape, breaking snapshot+restore. We therefore reassign buffers
    individually (replacing the parameter object) so any shape works.
    """
    own_state = module.state_dict()
    for name, value in state.items():
        if name not in own_state:
            continue
        # Try direct copy_ when shapes match; otherwise replace the buffer.
        cur = own_state[name]
        if cur.shape == value.shape and cur.dtype == value.dtype:
            cur.detach().copy_(value)
        else:
            # Walk the module tree to find the parent and replace the buffer.
            *parents, leaf = name.split(".")
            mod = module
            for p in parents:
                mod = getattr(mod, p)
            if leaf in getattr(mod, "_buffers", {}):
                mod._buffers[leaf] = value.detach().clone()
            elif leaf in getattr(mod, "_parameters", {}):
                # Wrap in nn.Parameter to preserve grad behaviour.
                import torch.nn as nn
                mod._parameters[leaf] = nn.Parameter(
                    value.detach().clone(),
                    requires_grad=mod._parameters[leaf].requires_grad,
                )


def _apply_ema_with_loose_buffers(ema: EMA, module: torch.nn.Module) -> None:
    """``ema.apply()`` variant tolerant of buffer-shape changes.

    Mirrors :func:`_restore_state_dict_with_loose_buffers` but for the EMA
    shadow state (which may also have been recorded BEFORE ``update()``
    mutated buffer shapes).
    """
    _restore_state_dict_with_loose_buffers(module, ema.shadow)


def _eval_ema_proxy(
    *,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    latents: torch.Tensor,
    target_pixels: torch.Tensor,
    noise_std: float,
) -> dict[str, float]:
    """Snapshot+restore eval (CLAUDE.md non-negotiable)."""
    decoder_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    balle_state = {k: v.detach().clone() for k, v in balle.state_dict().items()}
    ema_decoder.apply(decoder)
    _apply_ema_with_loose_buffers(ema_balle, balle)
    decoder.eval()
    balle.eval()
    try:
        with torch.no_grad():
            balle_out = balle(latents)
            decoded = decoder(balle_out["y_hat"])
            pixel_l1 = float(
                eval_roundtrip_pixel_l1(decoded, target_pixels, noise_std=noise_std)
            )
            rate_bits = float(balle_out["rate_total_bits"])
        return {
            "ema_proxy_pixel_l1": pixel_l1,
            "ema_proxy_rate_bits": rate_bits,
        }
    finally:
        decoder.load_state_dict(decoder_state)
        _restore_state_dict_with_loose_buffers(balle, balle_state)
        decoder.train()
        balle.train()


def _maybe_save_ema_checkpoint(
    *,
    output_dir: Path,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    coord: JointLagrangianADMM,
    proxy_score: float,
    epoch: int,
    best_proxy_score: float,
) -> float:
    """Save EMA shadow as best-proxy checkpoint when it improves."""
    if proxy_score >= best_proxy_score:
        return best_proxy_score
    ckpt = {
        "schema_version": 1,
        "epoch": epoch,
        "proxy_score": proxy_score,
        "decoder_ema_state_dict": ema_decoder.state_dict(),
        "balle_ema_state_dict": ema_balle.state_dict(),
        "coord_state_dict": coord.state_dict(),
        "tag": "[predicted; Phase 1 scaffold; not yet empirical]",
        "scaffold_version": PARADIGM_DELTA_EPS_ZETA_VERSION,
    }
    torch.save(ckpt, output_dir / "checkpoint_best_proxy_ema.pt")
    return proxy_score


def build_archive_from_ema(
    *,
    output_dir: Path,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    latents: torch.Tensor,
    decoder_config: Decoder128KConfig,
    balle_config: BalleHyperpriorConfig,
) -> Path:
    """Materialise an archive.zip + submission_dir for contest_auth_eval.

    Phase 1 NOTE: this is the **scaffold-quality** archive layout. The
    runtime side (inflate.py + codec.py) is generated procedurally from the
    trained module state dicts. Phase 2 will replace this with a fully
    optimised wire format.

    The archive layout (single ZIP member 'x'):
        uint32 LE: balle_strings_section_total_bytes (D)
        bytes (D - 4): pickled balle.compress(latents) output
        bytes : pickled decoder EMA state dict (torch.save format)
        bytes : pickled balle EMA state dict
    """
    submission_dir = output_dir / "submission_dir"
    submission_dir.mkdir(exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(exist_ok=True)

    # Snapshot+restore for EMA application.
    decoder_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    balle_state = {k: v.detach().clone() for k, v in balle.state_dict().items()}
    ema_decoder.apply(decoder)
    _apply_ema_with_loose_buffers(ema_balle, balle)
    decoder.eval()
    balle.eval()
    try:
        balle.update(force=True)
        # Move latents to balle's device for compression.
        device = next(balle.parameters()).device
        latents_dev = latents.to(device)
        with torch.no_grad():
            strings = balle.compress(latents_dev)

        # Build the wire-format inner blob (Phase 1 scaffold layout).
        import io
        import pickle

        strings_blob = pickle.dumps(strings)
        decoder_blob = pickle.dumps(ema_decoder.state_dict())
        balle_blob = pickle.dumps(ema_balle.state_dict())

        section_total_bytes = 4 + len(strings_blob)
        body = (
            struct.pack("<I", section_total_bytes)
            + strings_blob
            + struct.pack("<I", len(decoder_blob))
            + decoder_blob
            + struct.pack("<I", len(balle_blob))
            + balle_blob
        )

        archive_path = output_dir / "archive.zip"
        import zipfile

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
            info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, body)

        # Write inflate.py / inflate.sh / codec.py / model.py.
        _write_runtime(
            submission_dir=submission_dir,
            decoder_config=decoder_config,
            balle_config=balle_config,
        )
    finally:
        decoder.load_state_dict(decoder_state)
        _restore_state_dict_with_loose_buffers(balle, balle_state)
        decoder.train()
        balle.train()
    return archive_path


def _write_runtime(
    *,
    submission_dir: Path,
    decoder_config: Decoder128KConfig,
    balle_config: BalleHyperpriorConfig,
) -> None:
    """Emit a Phase 1 scaffold inflate.sh + inflate.py + src/{codec,model}.py.

    Phase 1 NOTE: this runtime is **build-verification only**; the wire
    format (pickle blobs) exceeds Phase 2's byte budget. Phase 2 will
    replace pickle with the trained Ballé wire format + a deterministic
    decoder serialiser.
    """
    src = submission_dir / "src"
    src.mkdir(exist_ok=True)

    inflate_sh = (
        "#!/bin/bash\n"
        "# PACT_RUNTIME_DEPENDENCY_ROOT = src/tac\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "exec uv run --with torch==2.5.1+cu124 --extra-index-url "
        "https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match "
        "--with compressai==1.2.8 \"$HERE/inflate.py\" \"$@\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh)
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"Phase 1 scaffold inflate (NOT byte-optimised).\"\"\"\n"
        "import pickle, struct, sys\n"
        "from pathlib import Path\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from model import Decoder128KRuntime, BalleRuntime\n"
        "CAMERA_H, CAMERA_W = 874, 1164\n"
        "EVAL_H, EVAL_W = 384, 512\n"
        "N_PAIRS = 600\n"
        "\n"
        "def inflate(src_bin: str, dst_raw: str) -> int:\n"
        "    body = Path(src_bin).read_bytes()\n"
        "    section_total = struct.unpack_from('<I', body, 0)[0]\n"
        "    offset = 4\n"
        "    strings = pickle.loads(body[offset:section_total])\n"
        "    offset = section_total\n"
        "    dec_len = struct.unpack_from('<I', body, offset)[0]\n"
        "    offset += 4\n"
        "    decoder_sd = pickle.loads(body[offset:offset + dec_len])\n"
        "    offset += dec_len\n"
        "    balle_len = struct.unpack_from('<I', body, offset)[0]\n"
        "    offset += 4\n"
        "    balle_sd = pickle.loads(body[offset:offset + balle_len])\n"
        "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "    balle = BalleRuntime().to(device)\n"
        "    balle.load_state_dict(balle_sd)\n"
        "    balle.eval()\n"
        "    balle.update(force=True)\n"
        "    decoder = Decoder128KRuntime().to(device)\n"
        "    decoder.load_state_dict(decoder_sd)\n"
        "    decoder.eval()\n"
        "    with torch.no_grad():\n"
        "        latents = balle.decompress(strings).to(device)\n"
        "    n = 0\n"
        "    with torch.no_grad(), open(dst_raw, 'wb') as fout:\n"
        "        for i in range(0, N_PAIRS, 16):\n"
        "            j = min(i + 16, N_PAIRS)\n"
        "            decoded = decoder(latents[i:j])\n"
        "            up = F.interpolate(\n"
        "                decoded.reshape(-1, 3, EVAL_H, EVAL_W),\n"
        "                size=(CAMERA_H, CAMERA_W),\n"
        "                mode='bicubic', align_corners=False,\n"
        "            ).clamp(0, 255).round().to(torch.uint8).cpu().numpy()\n"
        "            fout.write(up.tobytes())\n"
        "            n += up.shape[0]\n"
        "    print(f'saved {n} frames')\n"
        "    return n\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    if len(sys.argv) != 3:\n"
        "        sys.exit('Usage: inflate.py <src.bin> <dst.raw>')\n"
        "    inflate(sys.argv[1], sys.argv[2])\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py)

    codec_stub = (
        "# Phase 1 scaffold codec stub. Phase 2 replaces pickle with a\n"
        "# byte-optimised wire format derived from the trained Balle CDFs.\n"
    )
    (src / "codec.py").write_text(codec_stub)

    # The runtime model file embeds the same Decoder128K + Balle configs the
    # trainer used so the runtime can re-instantiate identically.
    model_py = (
        "import sys\n"
        "from pathlib import Path\n"
        "def _find_repo_src():\n"
        "    here = Path(__file__).resolve()\n"
        "    for parent in here.parents:\n"
        "        candidate = parent / 'src' / 'tac'\n"
        "        if candidate.exists():\n"
        "            return parent / 'src'\n"
        "    raise RuntimeError('repo-local tac runtime dependency not found')\n"
        "REPO_SRC = _find_repo_src()\n"
        "if str(REPO_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(REPO_SRC))\n"
        "from tac.paradigm_delta_epsilon_zeta.decoder_128k import Decoder128K, Decoder128KConfig\n"
        "from tac.paradigm_delta_epsilon_zeta.balle_hyperprior import BalleHyperpriorWrapper, BalleHyperpriorConfig\n"
        f"_DEC_CFG = Decoder128KConfig(latent_dim={decoder_config.latent_dim}, base_channels={decoder_config.base_channels})\n"
        f"_BALLE_CFG = BalleHyperpriorConfig(y_channels={balle_config.y_channels}, z_channels={balle_config.z_channels}, hyper_hidden={balle_config.hyper_hidden})\n"
        "class Decoder128KRuntime(Decoder128K):\n"
        "    def __init__(self):\n"
        "        super().__init__(_DEC_CFG)\n"
        "class BalleRuntime(BalleHyperpriorWrapper):\n"
        "    def __init__(self):\n"
        "        super().__init__(_BALLE_CFG)\n"
    )
    (src / "model.py").write_text(model_py)


def write_provenance(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    encoder_provenance: dict | None,
    n_decoder_params: int,
    n_balle_params: int,
    started_at_utc: str,
    completed_at_utc: str | None,
    t13_bit_reallocation: dict | None = None,
    t19_adaptive_rho: dict | None = None,
) -> Path:
    p = {
        "schema_version": 1,
        "tool": "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
        "scaffold_version": PARADIGM_DELTA_EPS_ZETA_VERSION,
        "scaffold_provenance": PARADIGM_DELTA_EPS_ZETA_PROVENANCE,
        # Per CLAUDE.md HNeRV parity discipline lesson #7: this trainer is
        # a substrate-engineering lane, not a representation lane. Tagged
        # explicitly so downstream lane-registry / preflight gates know to
        # apply substrate-engineering review (not bolt-on review).
        "lane_class": "substrate_engineering",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "args": vars(args),
        "encoder_provenance": encoder_provenance,
        "decoder_param_count": n_decoder_params,
        "balle_param_count": n_balle_params,
        "score_band": "[predicted; Phase 1 scaffold; not yet empirical]",
        # T13 / T19 wire-in surfaces (memo
        # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509). Both
        # default to None when the corresponding --enable flag is OFF
        # (backward-compat preserved).
        "t13_bit_reallocation": t13_bit_reallocation,
        "t19_adaptive_rho": t19_adaptive_rho,
        "platform": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(p, indent=2, default=str))
    return path


def maybe_run_auth_eval(
    *,
    archive_path: Path,
    submission_dir: Path,
    output_dir: Path,
    enabled: bool,
    dispatch_lane_id: str | None,
    dispatch_claims_path: Path,
) -> dict | None:
    if not enabled:
        return None
    raise SystemExit(
        "[t1] --auth-eval refused: Phase 1 emits a research_only_no_export "
        "runtime scaffold whose inflate.py signature is not the contest "
        "archive_dir -> inflated_dir -> video_names contract. Land the Phase "
        "2 deterministic wire format + hermetic inflate runtime before exact "
        "CUDA/CPU eval."
    )
    require_active_dispatch_claim(
        lane_id=dispatch_lane_id,
        claims_path=dispatch_claims_path,
    )
    auth_script = REPO_ROOT / CONTEST_AUTH_EVAL_RELATIVE
    if not auth_script.exists():
        print(f"[t1] auth-eval skipped: {auth_script} not found")
        return None
    work_dir = output_dir / "auth_eval_work"
    work_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        str(auth_script),
        "--archive", str(archive_path),
        "--inflate-sh", str(submission_dir / "inflate.sh"),
        "--upstream-dir", str(REPO_ROOT / "upstream"),
        "--device", "cuda",
        "--work-dir", str(work_dir),
        "--json-out", str(output_dir / "contest_auth_eval.json"),
        "--keep-work-dir",
    ]
    print(f"[t1] dispatching auth eval: {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    auth_json = work_dir / "contest_auth_eval.json"
    if rc != 0:
        raise SystemExit(f"[t1] auth eval failed rc={rc}; see {work_dir}")
    if not auth_json.exists():
        raise SystemExit(f"[t1] auth eval reported success but did not write {auth_json}")
    return {"returncode": rc, "auth_json_path": str(auth_json)}


def apply_t13_sqrt_n_budget(
    *,
    n_pairs: int,
    n_symbols_per_pair: int,
    current_bits_per_pair: float,
    rate_target_bytes: float,
    alpha: float = 1.0,
) -> dict:
    """Compute the T13 Fridrich √n latent reallocation envelope.

    Returns a dict suitable for emission into the run manifest. The dict
    carries:

    * ``bit_reallocation_t13_applied`` — always True (caller only invokes
      this when ``--enable-t13-sqrt-n-budget`` is set).
    * ``per_pair_undetectable_bits`` — closed-form Fridrich bound at α.
    * ``per_pair_current_bits`` — caller-supplied current rate.
    * ``per_pair_headroom_bits`` — undetectable - current. May be negative
      (current spends MORE than undetectable; reallocation is forbidden in
      that direction since the latent is already detectable; we clip to 0).
    * ``latent_bits_reduced`` / ``pose_bits_added`` — how the headroom was
      reallocated. Headroom is reallocated 1:1 from latent to pose stream
      per memo §6 lesson 6 ("re-allocate per-pair latent → per-pair pose
      budget per Fridrich √n bound").
    * ``rate_target_bytes_before`` / ``rate_target_bytes_after`` — only
      the LATENT portion of the rate target shrinks; pose is consumed by
      the same coordinator constraint set so we DO NOT add bytes back.
      The trainer's rate target is the ARCHIVE-level target; T13 shrinks
      the per-pair latent contribution and the saved bytes are consumed
      elsewhere downstream.
    * ``predicted_pose_distortion_decrease`` — qualitative tag (we cannot
      predict a numeric Δ without empirical anchor; per CLAUDE.md
      Forbidden Score Claims).
    * ``alpha`` / ``n_pairs`` / ``n_symbols_per_pair`` / ``notes`` —
      provenance.

    Per CLAUDE.md ``forbidden_dead_flag_wiring_pattern`` and the memo's
    integration log, this helper is callable standalone (covered by tests)
    and the trainer invokes it once at startup. The returned dict is
    written into ``provenance.json`` as ``t13_bit_reallocation``.

    Tagging: every numeric impact is ``[predicted; T13 Fridrich sqrt-n
    latent shrink]`` per CLAUDE.md Forbidden Score Claims.
    """
    from tac.joint_source_rd_bound import per_pair_sqrt_n_budget  # noqa: WPS433

    if n_symbols_per_pair <= 0:
        raise SystemExit(
            f"[t1] T13 requires positive n_symbols_per_pair; got "
            f"{n_symbols_per_pair!r}"
        )
    if current_bits_per_pair < 0:
        raise SystemExit(
            f"[t1] T13 requires non-negative current_bits_per_pair; got "
            f"{current_bits_per_pair!r}"
        )
    if rate_target_bytes <= 0:
        raise SystemExit(
            f"[t1] T13 requires positive rate_target_bytes; got "
            f"{rate_target_bytes!r}"
        )
    report = per_pair_sqrt_n_budget(
        n_pairs=n_pairs,
        n_symbols_per_pair=n_symbols_per_pair,
        alpha=alpha,
    )
    headroom_bits = report.undetectable_bits_per_pair - current_bits_per_pair
    # Reallocation is bounded: only positive headroom is realloc-able.
    # If the current per-pair rate already exceeds the undetectable budget,
    # we surface the negative gap as a warning but do not over-shrink the
    # rate target.
    realloc_per_pair_bits = max(0.0, headroom_bits)
    realloc_total_bits = realloc_per_pair_bits * n_pairs
    realloc_total_bytes = realloc_total_bits / 8.0
    rate_target_after = max(1.0, rate_target_bytes - realloc_total_bytes)
    return {
        "bit_reallocation_t13_applied": True,
        "per_pair_undetectable_bits": float(report.undetectable_bits_per_pair),
        "per_pair_current_bits": float(current_bits_per_pair),
        "per_pair_headroom_bits": float(headroom_bits),
        "latent_bits_reduced": float(realloc_total_bits),
        "pose_bits_added": float(realloc_total_bits),
        "rate_target_bytes_before": float(rate_target_bytes),
        "rate_target_bytes_after": float(rate_target_after),
        "predicted_pose_distortion_decrease": (
            "[predicted; T13 Fridrich sqrt-n latent shrink; "
            "qualitative direction only — empirical anchor required]"
        ),
        "alpha": float(alpha),
        "n_pairs": int(n_pairs),
        "n_symbols_per_pair": int(n_symbols_per_pair),
        "notes": list(report.notes),
        "tag": "[predicted; T13 Fridrich sqrt-n latent shrink]",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--rate-target-bytes", type=float, default=80_000.0)
    parser.add_argument("--seg-target", type=float, default=7e-4)
    parser.add_argument("--pose-target", type=float, default=1.7e-4)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--eval-every-epochs", type=int, default=100)
    parser.add_argument("--auth-eval", action="store_true", default=False)
    parser.add_argument("--no-auth-eval", dest="auth_eval", action="store_false")
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 1 pair build verification")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Real contest video used for non-smoke target pixels.",
    )
    parser.add_argument(
        "--target-pixels-path",
        type=Path,
        default=None,
        help=(
            "Optional torch tensor containing real target pixels with shape "
            "(N, 2, 3, 384, 512) or (N, 2, 384, 512, 3). Overrides --video-path."
        ),
    )
    parser.add_argument(
        "--max-target-pairs",
        type=int,
        default=None,
        help="Optional cap on real frame pairs for local non-smoke debugging.",
    )
    parser.add_argument(
        "--canonical-a1-relpath",
        default="experiments/results/A1_canonical",
        help="Path under repo root to the canonical A1 symlink",
    )
    parser.add_argument(
        "--allow-missing-canonical-a1",
        action="store_true",
        help="Skip frozen-A1 load (smoke mode only)",
    )
    parser.add_argument(
        "--dispatch-lane-id",
        default=None,
        help="Required with --auth-eval; must have an active dispatch claim.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=REPO_ROOT / DISPATCH_CLAIMS_RELATIVE,
        help="Dispatch-claim ledger checked before --auth-eval.",
    )
    # T13 — Fridrich √n per-pair latent budget (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t13-sqrt-n-budget",
        action="store_true",
        default=False,
        help="Opt-in T13 Fridrich sqrt(n) latent-budget hook. Default OFF for "
             "backward-compat. When ON, the trainer queries "
             "tac.joint_source_rd_bound.per_pair_sqrt_n_budget and shrinks "
             "rate_target_bytes by the reallocation envelope.",
    )
    parser.add_argument(
        "--t13-alpha",
        type=float,
        default=1.0,
        help="Fridrich proportionality constant (Ker-Pevný-Fridrich 2008). "
             "Default 1.0; values in [0.5, 2.0] cover the literature.",
    )
    parser.add_argument(
        "--t13-current-bits-per-pair",
        type=float,
        default=3.0,
        help="Caller-supplied estimate of the trainer's current per-pair "
             "latent rate (bits/pair). Default 3.0 per A1 substrate empirical "
             "anchor (memo §6).",
    )
    # T19 — adaptive ρ ADMM (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t19-adaptive-rho",
        action="store_true",
        default=False,
        help="Opt-in T19 Boyd §3.4.1 / He-Yang 2000 adaptive-ρ update via "
             "tac.joint_admm_coordinator.adaptive_rho_step. Default OFF for "
             "backward-compat per coherence council recommendation.",
    )
    parser.add_argument(
        "--t19-tau-grow",
        type=float,
        default=2.0,
        help="T19 ρ-growth factor (Boyd canonical 2.0). Must be > 1.",
    )
    parser.add_argument(
        "--t19-tau-shrink",
        type=float,
        default=0.5,
        help="T19 ρ-shrink factor (Boyd canonical 0.5). Must be in (0, 1).",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.auth_eval:
        raise SystemExit(
            "[t1] --auth-eval refused before training: Phase 1 archive/runtime "
            "is research_only_no_export. This avoids false [contest-CUDA] "
            "promotion from a non-hermetic scaffold."
        )
    if args.allow_missing_canonical_a1 and not args.smoke:
        raise SystemExit(
            "[t1] --allow-missing-canonical-a1 is smoke-only. Non-smoke "
            "training requires the real frozen A1 latent table."
        )
    if args.max_target_pairs is not None and args.max_target_pairs <= 0:
        raise SystemExit("[t1] --max-target-pairs must be positive when set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    print(f"[t1] scaffold version: {PARADIGM_DELTA_EPS_ZETA_VERSION}")
    print(f"[t1] device: {device}; smoke: {args.smoke}; epochs: {args.epochs}")

    # Load frozen A1 encoder (or smoke-synthesise).
    encoder_provenance = None
    if args.smoke and args.allow_missing_canonical_a1:
        print("[t1] smoke + allow-missing-canonical-a1 — synthesising latents")
        latents, target_pixels = make_smoke_target(
            n_pairs=1, latent_dim=28, seed=args.seed,
        )
    else:
        encoder = load_frozen_a1_encoder(
            repo_root=REPO_ROOT,
            canonical_dir_name=_canonical_dir_name_from_relpath(args.canonical_a1_relpath),
        )
        encoder_provenance = encoder.provenance
        latents = encoder.latents
        # In smoke we keep just one pair; in full mode we use all 600.
        if args.smoke:
            latents = latents[:1].clone()
        latents.requires_grad_(False)
        if args.smoke:
            # Build-verification only; non-smoke must use real frame targets.
            # The make_smoke_target n_pairs MUST match latents shape so the
            # broadcast shapes are aligned at loss time.
            _, target_pixels = make_smoke_target(
                n_pairs=int(latents.shape[0]),
                latent_dim=int(latents.shape[1]),
                seed=args.seed,
            )
        else:
            if args.target_pixels_path is not None:
                target_pixels = load_target_pixels_from_path(args.target_pixels_path)
                if args.max_target_pairs is not None:
                    target_pixels = target_pixels[: args.max_target_pairs].clone()
                    latents = latents[: target_pixels.shape[0]].clone()
            else:
                target_pixels = load_real_target_pairs(
                    args.video_path,
                    n_pairs=int(latents.shape[0]),
                    max_pairs=args.max_target_pairs,
                )
                latents = latents[: target_pixels.shape[0]].clone()
            if int(target_pixels.shape[0]) != int(latents.shape[0]):
                raise SystemExit(
                    "[t1] real target pair count mismatch: "
                    f"targets={target_pixels.shape[0]} latents={latents.shape[0]}. "
                    "Use --max-target-pairs to trim both explicitly."
                )

    latents = latents.to(device)
    target_pixels = target_pixels.to(device)

    # Build modules.
    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    n_decoder_params = sum(p.numel() for p in decoder.parameters())
    n_balle_params = sum(p.numel() for p in balle.parameters())
    print(f"[t1] decoder params: {n_decoder_params:,}; balle params: {n_balle_params:,}")

    # Optimisers (CompressAI requires SEPARATE aux optimiser).
    main_params = [p for n, p in list(decoder.named_parameters()) + list(balle.named_parameters())
                   if "entropy_bottleneck" not in n or "quantiles" not in n]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(
        [p for p in main_params if p.requires_grad], lr=args.learning_rate
    )
    optim_aux = torch.optim.Adam(
        [p for p in aux_params if p.requires_grad], lr=args.aux_learning_rate
    ) if aux_params else None

    # EMA shadows (decay 0.997 per CLAUDE.md non-negotiable).
    ema_decoder = EMA(decoder, decay=args.ema_decay)
    ema_balle = EMA(balle, decay=args.ema_decay)

    # T13 — Fridrich √n per-pair latent budget hook (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509). When OFF,
    # the trainer behaves identically to its pre-T13 form (backward-compat).
    rate_target_bytes_effective = float(args.rate_target_bytes)
    t13_report: dict | None = None
    if args.enable_t13_sqrt_n_budget:
        t13_report = apply_t13_sqrt_n_budget(
            n_pairs=int(latents.shape[0]),
            n_symbols_per_pair=int(latents.shape[1]),
            current_bits_per_pair=float(args.t13_current_bits_per_pair),
            rate_target_bytes=float(args.rate_target_bytes),
            alpha=float(args.t13_alpha),
        )
        rate_target_bytes_effective = t13_report["rate_target_bytes_after"]
        print(
            f"[t1] T13 enabled: per-pair undetectable bits = "
            f"{t13_report['per_pair_undetectable_bits']:.3f}; "
            f"per-pair current = {t13_report['per_pair_current_bits']:.3f}; "
            f"headroom = {t13_report['per_pair_headroom_bits']:.3f}; "
            f"rate_target_bytes "
            f"{t13_report['rate_target_bytes_before']:.0f} -> "
            f"{t13_report['rate_target_bytes_after']:.0f} "
            f"[predicted; T13 Fridrich sqrt-n latent shrink]"
        )

    # Joint Lagrangian-ADMM coordinator. T19 (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509) routes the
    # adaptive-ρ rule through tac.joint_admm_coordinator.adaptive_rho_step
    # when --enable-t19-adaptive-rho is set; otherwise the legacy windowed-
    # average grow/shrink rule is preserved (backward-compat per coherence
    # council recommendation).
    coord_kwargs: dict = {
        "rate_target_bytes": rate_target_bytes_effective,
        "seg_target": args.seg_target,
        "pose_target": args.pose_target,
        "rho_init": args.rho_init,
        "use_t19_adaptive_rho": bool(args.enable_t19_adaptive_rho),
        "t19_tau_grow": float(args.t19_tau_grow),
        "t19_tau_shrink": float(args.t19_tau_shrink),
    }
    if args.enable_t19_adaptive_rho:
        # The brief specifies rho_min=1e-3, rho_max=1e3 for T19 per memo
        # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.
        coord_kwargs["rho_min"] = 1e-3
        coord_kwargs["rho_max"] = 1e3
    coord_cfg = JointLagrangianADMMConfig(**coord_kwargs)
    coord = JointLagrangianADMM(coord_cfg)
    if args.enable_t19_adaptive_rho:
        print(
            f"[t1] T19 enabled: adaptive-ρ via "
            f"tac.joint_admm_coordinator.adaptive_rho_step "
            f"(tau_grow={args.t19_tau_grow}, tau_shrink={args.t19_tau_shrink}, "
            f"rho_min={coord_cfg.rho_min}, rho_max={coord_cfg.rho_max}) "
            f"[predicted; T19 adaptive ρ 2-3× convergence speedup; "
            f"not direct score]"
        )

    n_pairs = int(latents.shape[0])
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else min(args.batch_size, n_pairs)

    best_proxy = float("inf")
    history = []

    for epoch in range(epochs):
        decoder.train()
        balle.train()
        # Shuffle pair indices each epoch.
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]

            balle_out = balle(y)
            decoded = decoder(balle_out["y_hat"])
            distortion = eval_roundtrip_pixel_l1(decoded, tgt, noise_std=args.noise_std)
            # Per-pair seg/pose targets approximated as constant (Phase 2 will
            # plug real SegNet/PoseNet forwards under no-grad so the
            # Lagrangian sees real signal).
            seg_loss = torch.tensor(args.seg_target, device=device)
            pose_loss = torch.tensor(args.pose_target, device=device)
            res = coord.step(
                distortion=distortion,
                rate_bits=balle_out["rate_total_bits"],
                seg_loss=seg_loss,
                pose_loss=pose_loss,
            )
            optim_main.zero_grad()
            res.augmented_lagrangian.backward()
            optim_main.step()

            if optim_aux is not None:
                optim_aux.zero_grad()
                aux = balle.aux_loss()
                aux.backward()
                optim_aux.step()

            ema_decoder.update(decoder)
            ema_balle.update(balle)

            epoch_loss += float(res.augmented_lagrangian.detach())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            metrics = _eval_ema_proxy(
                decoder=decoder,
                balle=balle,
                ema_decoder=ema_decoder,
                ema_balle=ema_balle,
                latents=latents,
                target_pixels=target_pixels,
                noise_std=args.noise_std,
            )
            print(
                f"[t1] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
                f"ema_pixel_l1={metrics['ema_proxy_pixel_l1']:.4f} "
                f"ema_rate_bits={metrics['ema_proxy_rate_bits']:.0f} "
                f"rho={coord.rho:.3f}"
            )
            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                **metrics,
                "rho": coord.rho,
                "lambdas": dict(coord.lambdas),
            })
            best_proxy = _maybe_save_ema_checkpoint(
                output_dir=args.output_dir,
                decoder=decoder,
                balle=balle,
                ema_decoder=ema_decoder,
                ema_balle=ema_balle,
                coord=coord,
                proxy_score=metrics["ema_proxy_pixel_l1"],
                epoch=epoch + 1,
                best_proxy_score=best_proxy,
            )

    # Build the EMA archive + maybe-run auth eval.
    archive_path = build_archive_from_ema(
        output_dir=args.output_dir,
        decoder=decoder,
        balle=balle,
        ema_decoder=ema_decoder,
        ema_balle=ema_balle,
        latents=latents,
        decoder_config=decoder_config,
        balle_config=balle_config,
    )
    print(f"[t1] wrote archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    auth_eval_result = None
    if not args.smoke:  # never auth-eval the smoke-synthetic archive
        auth_eval_result = maybe_run_auth_eval(
            archive_path=archive_path,
            submission_dir=args.output_dir / "submission_dir",
            output_dir=args.output_dir,
            enabled=args.auth_eval,
            dispatch_lane_id=args.dispatch_lane_id,
            dispatch_claims_path=args.dispatch_claims_path,
        )

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # T19 — emit per-iteration ρ trajectory side-log (always, when the
    # coordinator has any entries; the helper appends only on actual ρ
    # changes). When --enable-t19-adaptive-rho is OFF the trajectory is
    # populated only when the legacy backend triggers a ρ update.
    t19_summary: dict | None = None
    if args.enable_t19_adaptive_rho or coord.rho_trajectory:
        rho_log_path = args.output_dir / "rho_trajectory.json"
        rho_log_path.write_text(
            json.dumps(coord.rho_trajectory, indent=2, default=str)
        )
        t19_summary = {
            "enabled": bool(args.enable_t19_adaptive_rho),
            "tau_grow": float(args.t19_tau_grow),
            "tau_shrink": float(args.t19_tau_shrink),
            "rho_min": float(coord.config.rho_min),
            "rho_max": float(coord.config.rho_max),
            "n_rho_updates": len(coord.rho_trajectory),
            "rho_final": float(coord.rho),
            "rho_trajectory_path": str(rho_log_path),
            "tag": (
                "[predicted; T19 adaptive ρ 2-3× convergence speedup; "
                "not direct score]"
                if args.enable_t19_adaptive_rho
                else "[legacy adaptive-ρ backend]"
            ),
        }
    write_provenance(
        output_dir=args.output_dir,
        args=args,
        encoder_provenance=encoder_provenance,
        n_decoder_params=n_decoder_params,
        n_balle_params=n_balle_params,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        t13_bit_reallocation=t13_report,
        t19_adaptive_rho=t19_summary,
    )
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "auth_eval_summary.json").write_text(
        json.dumps(auth_eval_result or {"skipped": True}, indent=2)
    )

    print("[t1] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
