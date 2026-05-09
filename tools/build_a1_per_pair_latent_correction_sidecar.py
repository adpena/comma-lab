r"""Re-search the per-pair latent correction sidecar against A1's substrate.

Background (from forensics dossier
`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`):

  PR99 introduced the per-pair latent correction sidecar mechanism: a
  single-dim perturbation per latent pair, grid-searched at compression time
  against joint SegNet+PoseNet distortion. PR99-103 all inherit this. The
  fixed delta vocabulary is ``[-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4,
  5, 6, 8, 10] × 0.01``. Each pair gets one of 1+28*16=449 choices: no-op or
  one of 16 deltas applied to one of 28 latent dims.

  A1's existing 607-byte sidecar is INHERITED from PR101 (its source archive).
  PR101's sidecar was searched against PR98's substrate (which is the same
  decoder weights). A1 then fine-tuned the latents/decoder away from PR98's,
  so the optimal per-pair sidecar for A1 may differ.

This tool re-runs the per-pair greedy search against A1's frozen decoder and
A1's frozen latents, OPTIONALLY using:
  - proxy MSE between A1's decoded frames and ground-truth uint8 frames as
    the search signal (fast; ~CPU-minutes on M-series ARM)
  - SegNet+PoseNet joint distortion as the search signal (slow; ~CPU-hours
    even on M-series ARM)

The search is greedy, single-pair-at-a-time, single-dim only (matching PR101's
mechanism). For each pair, evaluates 1+28*16=449 candidate perturbations and
picks the one that minimizes the chosen objective.

Output:
  experiments/results/a1_per_pair_latent_sidecar_resampled_<timestamp>/
    submission_dir/                         (variant submission_dir)
      archive.zip                           (NEW archive; same decoder/latents as A1
                                             but new sidecar bytes — score-affecting)
      inflate.py                            (A1's existing inflate.py with PR101 bias)
      inflate.sh                            (A1's existing)
      src/{codec,model}.py                  (A1's existing)
    sidecar_search.log                      (per-pair best deltas, search timings)
    sidecar_manifest.json                   (search config, search signal,
                                             old/new archive SHA, expected delta)

Per CLAUDE.md:
  - All claims tagged ``[predicted; per-pair latent sidecar resampled on A1
    substrate via <signal>]`` until GHA result returns
    ``[contest-CPU GHA Linux x86_64]``.
  - Per HNeRV-parity discipline lesson 11: no-op detector — sidecar bytes
    DO change archive bytes (~600 bytes); new score MUST be re-measured.
  - Per lesson 13: any "didn't beat baseline" finding is
    DEFERRED-pending-research with reactivation criteria.

NOTE on operational scope:
  The PROXY-MSE search is CPU-feasible (~10-30 min on M-series ARM for 600
  pairs × 449 candidates). The SegNet+PoseNet search is too expensive without
  GPU (~30+ hours on CPU for full 600×449 evaluation). This tool defaults to
  ``--search-signal proxy_mse``; ``--search-signal joint_seg_pose`` is gated
  behind ``--accept-cpu-budget`` and a budget ceiling.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

A1_SUBMISSION_DIR = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir"
)
A1_ARCHIVE_PATH = A1_SUBMISSION_DIR / "archive.zip"
A1_EXPECTED_ARCHIVE_SHA = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_EXPECTED_ARCHIVE_BYTES = 178262

# Sidecar fixed delta vocabulary (PR99-PR103 lineage)
SIDECAR_DELTAS_X100 = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10], dtype=np.int8
)
SIDECAR_BASE = 1 + 28 * len(SIDECAR_DELTAS_X100)  # 449

N_PAIRS = 600
LATENT_DIM = 28
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a repo-relative path when possible, else an absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_upstream_yuv420_to_rgb():
    """Load the upstream CPU-eval RGB conversion helper without patching it."""
    import importlib.util

    frame_utils_path = REPO_ROOT / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location(
        "pact_sidecar_upstream_frame_utils",
        frame_utils_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load upstream frame_utils.py from {frame_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def load_a1_archive_components(archive_path: Path) -> dict[str, Any]:
    """Parse A1's archive into decoder_sd, raw latents (pre-sidecar), and
    sidecar_blob — using A1's own codec module."""
    sys.path.insert(0, str(A1_SUBMISSION_DIR / "src"))
    # Import A1's codec module
    import importlib
    import zipfile

    if "codec" in sys.modules:
        importlib.reload(sys.modules["codec"])
    if "model" in sys.modules:
        importlib.reload(sys.modules["model"])
    import codec  # type: ignore

    with zipfile.ZipFile(archive_path, "r") as zf:
        member = zf.namelist()[0]
        archive_bytes = zf.read(member)
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total : section_total + codec.LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + codec.LATENT_BLOB_LEN :]
    decoder_sd = codec.decode_decoder_compact(decoder_blob)
    latents_pre_sidecar = codec.decode_latents_compact(latent_blob)
    latents_with_sidecar = codec.apply_latent_sidecar(latents_pre_sidecar, sidecar_blob)
    return {
        "archive_bytes": archive_bytes,
        "section_total": section_total,
        "decoder_blob": decoder_blob,
        "latent_blob": latent_blob,
        "sidecar_blob_old": sidecar_blob,
        "decoder_sd": decoder_sd,
        "latents_pre_sidecar": latents_pre_sidecar,
        "latents_with_sidecar_old": latents_with_sidecar,
        "n_pairs": codec.N_PAIRS,
        "latent_dim": codec.LATENT_DIM,
        "base_channels": codec.BASE_CHANNELS,
        "eval_size": codec.EVAL_SIZE,
        "codec_module": codec,
    }


def infer_sidecar_choices_from_latents(components: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Infer ``(dim, delta_idx)`` sidecar choices from old decoded latents.

    This preserves the inherited PR101 sidecar for any pair a partial/smoke
    search does not revisit. The codec supports several compact sidecar wire
    layouts, so inferring from ``latents_with_sidecar_old - latents_pre_sidecar``
    is simpler and harder to desynchronize from the runtime decoder.
    """
    diff = (
        components["latents_with_sidecar_old"]
        - components["latents_pre_sidecar"]
    ).detach().cpu().numpy()
    dims = np.full(N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(N_PAIRS, -1, dtype=np.int64)
    deltas = SIDECAR_DELTAS_X100.astype(np.float32) / 100.0
    for k in range(N_PAIRS):
        nz = np.flatnonzero(np.abs(diff[k]) > 1e-6)
        if nz.size == 0:
            continue
        if nz.size != 1:
            raise ValueError(
                f"old sidecar inference expected <=1 changed latent dim per pair; "
                f"pair={k} changed_dims={nz.tolist()}"
            )
        dim = int(nz[0])
        delta = float(diff[k, dim])
        matches = np.flatnonzero(np.isclose(deltas, delta, rtol=0.0, atol=1e-4))
        if matches.size != 1:
            raise ValueError(
                f"old sidecar delta {delta:.6f} for pair={k} dim={dim} "
                "does not match fixed PR99-PR103 delta vocabulary"
            )
        dims[k] = dim
        delta_idx[k] = int(matches[0])
    return dims, delta_idx


def encode_sidecar_huff_enum(dims: np.ndarray, delta_idx: np.ndarray) -> bytes:
    """Encode sidecar in PR101's HUFF_ENUM format (607 bytes for typical mix).

    For simplicity in this re-search tool we emit the SIDECAR_PACKED_LEN (661 B)
    layout instead of HUFF_ENUM (607 B) — encoding HUFF_ENUM requires the full
    Huffman+combinatorial machinery. PACKED_LEN is universally decodable by
    A1's codec.py (line 428-440) and only adds ~54 bytes vs HUFF_ENUM. This
    keeps the search results actionable while accepting a small archive bloat.

    A more aggressive future version could use the canonical Huffman path.
    """
    # Build choices array: 0=no-op, 1+i*16+d=dim i with delta d
    choices = np.zeros(N_PAIRS, dtype=np.int64)
    for k in range(N_PAIRS):
        if dims[k] != 255 and delta_idx[k] >= 0:
            choices[k] = 1 + dims[k] * len(SIDECAR_DELTAS_X100) + delta_idx[k]
    # Pack as base-449 mixed-radix integer little-endian (matches codec.py:428)
    value = 0
    for k in reversed(range(N_PAIRS)):
        value = value * SIDECAR_BASE + int(choices[k])
    n_bytes = (value.bit_length() + 7) // 8
    raw = value.to_bytes(n_bytes, "little")
    # Pad to SIDECAR_PACKED_LEN (661) — actually codec.py expects EXACTLY
    # SIDECAR_PACKED_LEN=661. Pad with zeros if smaller.
    SIDECAR_PACKED_LEN = 661
    if len(raw) < SIDECAR_PACKED_LEN:
        raw = raw + b"\x00" * (SIDECAR_PACKED_LEN - len(raw))
    elif len(raw) > SIDECAR_PACKED_LEN:
        raise ValueError(
            f"encoded packed sidecar overflow: {len(raw)} > {SIDECAR_PACKED_LEN}"
        )
    return raw


def encode_sidecar_n_pairs(dims: np.ndarray, delta_idx: np.ndarray) -> bytes:
    """Encode sidecar in the simplest 600-byte uint8-per-pair format
    (codec.py:441 path). Each pair gets one byte = choices[k]. Total=600 B."""
    choices = np.zeros(N_PAIRS, dtype=np.uint8)
    for k in range(N_PAIRS):
        if dims[k] != 255 and delta_idx[k] >= 0:
            value = 1 + dims[k] * len(SIDECAR_DELTAS_X100) + delta_idx[k]
            if value > 255:
                # 1+27*16+15 = 1+432+15 = 448 — cannot fit in uint8 (max 255).
                # The N_PAIRS layout (600 B) has limited reach.
                raise ValueError(
                    f"sidecar value {value} > 255; N_PAIRS layout cannot encode "
                    f"dim={dims[k]} delta_idx={delta_idx[k]}; switch to PACKED_LEN"
                )
            choices[k] = value
    return choices.tobytes()


def search_per_pair_proxy_mse(
    components: dict[str, Any],
    ground_truth_frames: np.ndarray,
    *,
    pair_indices: list[int] | None = None,
    log_path: Path | None = None,
    candidate_batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Greedy per-pair single-dim search using pixel MSE (fast proxy)."""
    decoder_sd = components["decoder_sd"]
    latents_base = components["latents_pre_sidecar"].clone()
    eval_h, eval_w = components["eval_size"]
    sys.path.insert(0, str(A1_SUBMISSION_DIR / "src"))
    if "model" not in sys.modules:
        import model as model_mod  # type: ignore
    else:
        model_mod = sys.modules["model"]
    decoder = model_mod.HNeRVDecoder(
        latent_dim=components["latent_dim"],
        base_channels=components["base_channels"],
        eval_size=(eval_h, eval_w),
    )
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    n_pairs = components["n_pairs"]
    pair_indices = pair_indices if pair_indices is not None else list(range(n_pairs))

    dims_out, delta_idx_out = infer_sidecar_choices_from_latents(components)
    if dims_out.shape[0] != n_pairs or delta_idx_out.shape[0] != n_pairs:
        raise ValueError(
            f"old sidecar inference shape mismatch: dims={dims_out.shape} "
            f"delta_idx={delta_idx_out.shape} expected={n_pairs}"
        )
    deltas = SIDECAR_DELTAS_X100.astype(np.float32) / 100.0

    log_lines = []
    t0 = time.time()
    progress_every = max(1, min(25, len(pair_indices)))
    for n_done, k in enumerate(pair_indices, start=1):
        # Ground truth uint8 frames for this pair
        gt = ground_truth_frames[k * 2 : k * 2 + 2]  # (2, CAMERA_H, CAMERA_W, 3)
        gt_eval = _resize_uint8_to_eval(gt)  # (2, EVAL_H, EVAL_W, 3) float32
        base_lat = latents_base[k : k + 1].clone()
        base_mse, best_mse, best_dim, best_didx = _best_pair_proxy_mse_candidate(
            decoder=decoder,
            base_lat=base_lat,
            gt_eval=gt_eval,
            eval_h=eval_h,
            eval_w=eval_w,
            deltas=deltas,
            candidate_batch_size=candidate_batch_size,
        )
        dims_out[k] = best_dim
        delta_idx_out[k] = best_didx
        if n_done % progress_every == 0 or n_done == len(pair_indices):
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (len(pair_indices) - n_done) / rate
            line = (
                f"[{n_done}/{len(pair_indices)}] pair={k} best_dim={best_dim} "
                f"best_didx={best_didx} mse_red={base_mse - best_mse:.4e} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}min"
            )
            print(line, flush=True)
            log_lines.append(line)
    elapsed = time.time() - t0
    searched_mask = np.zeros(n_pairs, dtype=bool)
    searched_mask[pair_indices] = True
    n_perturbed = int((dims_out != 255).sum())
    n_perturbed_searched = int(((dims_out != 255) & searched_mask).sum())
    print(
        f"[done] proxy-mse search: {len(pair_indices)} pairs in {elapsed:.1f}s; "
        f"{n_perturbed_searched} searched-pair perturbations; "
        f"{n_perturbed} total perturbations after preserving old sidecar",
        flush=True,
    )
    if log_path:
        log_path.write_text("\n".join(log_lines) + "\n")
    return dims_out, delta_idx_out, {
        "search_signal": "proxy_mse",
        "n_pairs_searched": len(pair_indices),
        "n_pairs_perturbed": n_perturbed,
        "n_pairs_perturbed_searched": n_perturbed_searched,
        "unsearched_pairs_preserve_old_sidecar": True,
        "candidate_batch_size": candidate_batch_size,
        "elapsed_seconds": elapsed,
    }


def _best_pair_proxy_mse_candidate(
    *,
    decoder: Any,
    base_lat: Any,
    gt_eval: np.ndarray,
    eval_h: int,
    eval_w: int,
    deltas: np.ndarray,
    candidate_batch_size: int,
) -> tuple[float, float, int, int]:
    """Return the best single-dim sidecar choice for one pair.

    The old implementation ran one decoder forward per candidate. This keeps
    identical greedy semantics while optionally evaluating perturbations in
    chunks. Large CPU chunks regressed the A1 smoke benchmark on 2026-05-09, so
    callers must opt into batching explicitly.
    """

    import torch

    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be >= 1")

    with torch.inference_mode():
        base_dec = decoder(base_lat).reshape(2, 3, eval_h, eval_w).detach().cpu().numpy()
    base_mse = _mse_uint8_after_clamp(base_dec, gt_eval)
    best_mse = base_mse
    best_dim = 255
    best_didx = -1

    if candidate_batch_size == 1:
        latent_dim = int(base_lat.shape[1])
        for d in range(latent_dim):
            for di, delta in enumerate(deltas):
                cand = base_lat.clone()
                cand[0, d] += float(delta)
                with torch.inference_mode():
                    cand_dec = decoder(cand).reshape(2, 3, eval_h, eval_w).detach().cpu().numpy()
                mse = _mse_uint8_after_clamp(cand_dec, gt_eval)
                if mse < best_mse:
                    best_mse = mse
                    best_dim = d
                    best_didx = di
        return base_mse, best_mse, best_dim, best_didx

    chunk: list[Any] = []
    metas: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal best_mse, best_dim, best_didx
        if not chunk:
            return
        batch = torch.cat(chunk, dim=0)
        with torch.inference_mode():
            decoded = decoder(batch)
        decoded_np = (
            decoded.reshape(len(chunk), 2, 3, eval_h, eval_w)
            .detach()
            .cpu()
            .numpy()
        )
        mses = _mse_uint8_batch_after_clamp(decoded_np, gt_eval)
        for mse, (dim, didx) in zip(mses, metas, strict=True):
            if float(mse) < best_mse:
                best_mse = float(mse)
                best_dim = dim
                best_didx = didx
        chunk.clear()
        metas.clear()

    latent_dim = int(base_lat.shape[1])
    for d in range(latent_dim):
        for di, delta in enumerate(deltas):
            cand = base_lat.clone()
            cand[0, d] += float(delta)
            chunk.append(cand)
            metas.append((d, di))
            if len(chunk) >= candidate_batch_size:
                flush()
    flush()
    return base_mse, best_mse, best_dim, best_didx


def _resize_uint8_to_eval(frames_uint8: np.ndarray) -> np.ndarray:
    """Resize uint8 (B, H, W, 3) ground-truth frames to (B, EVAL_H, EVAL_W, 3)
    float32 using bilinear, matching the *inverse* of the inflate-time bicubic
    upscale. Returns float32 in 0..255."""
    import torch
    import torch.nn.functional as F

    arr = (
        torch.from_numpy(frames_uint8.astype(np.float32))
        .permute(0, 3, 1, 2)  # B,C,H,W
    )
    arr = F.interpolate(arr, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
    return arr.permute(0, 2, 3, 1).numpy()  # B,H,W,C


def _mse_uint8_after_clamp(decoded: np.ndarray, gt: np.ndarray) -> float:
    """decoded is (2, 3, EVAL_H, EVAL_W); gt is (2, EVAL_H, EVAL_W, 3) float32 0..255."""
    dec = decoded.transpose(0, 2, 3, 1).clip(0, 255)
    return float(np.mean((dec - gt) ** 2))


def _mse_uint8_batch_after_clamp(decoded: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Vectorized MSE for decoded shape (B, 2, 3, H, W)."""

    dec = decoded.transpose(0, 1, 3, 4, 2).clip(0, 255)
    return np.mean((dec - gt[None, ...]) ** 2, axis=(1, 2, 3, 4))


def ground_truth_pairs_needed(pair_indices: list[int], n_pairs: int) -> int:
    """Return how many leading pairs must be decoded for ``pair_indices``."""

    if not pair_indices:
        return 0
    max_pair = max(pair_indices)
    if max_pair < 0:
        raise ValueError("pair indices must be nonnegative")
    if max_pair >= n_pairs:
        raise ValueError(f"pair index {max_pair} outside n_pairs={n_pairs}")
    return max_pair + 1


def load_ground_truth_pairs(video_path: Path, n_pairs: int = 600) -> np.ndarray:
    """Decode video into pair frames (n_pairs * 2, CAMERA_H, CAMERA_W, 3) uint8.

    Uses pyav (already a project dependency). Each "pair" is two consecutive
    frames at the seq_len=2 non-overlapping batching.
    """
    import av  # type: ignore

    yuv420_to_rgb = load_upstream_yuv420_to_rgb()
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for f in container.decode(stream):
        # Use the exact upstream CPU-eval conversion helper. Raw PyAV rgb24
        # takes a different colorspace path and would optimize the sidecar
        # against the wrong byte substrate.
        img = yuv420_to_rgb(f)  # torch (H, W, 3) uint8
        frames.append(img.cpu().numpy())
        if len(frames) >= n_pairs * 2:
            break
    container.close()
    arr = np.stack(frames[: n_pairs * 2], axis=0)
    assert arr.shape[1:] == (CAMERA_H, CAMERA_W, 3), f"unexpected shape {arr.shape}"
    return arr


def write_resampled_archive(
    components: dict[str, Any],
    new_sidecar_blob: bytes,
    out_archive_path: Path,
) -> None:
    """Build the new archive.zip = decoder_section || latent_blob || new_sidecar."""
    import zipfile

    archive_bytes = components["archive_bytes"]
    section_total = components["section_total"]
    new_inner = (
        archive_bytes[:section_total]
        + components["latent_blob"]
        + new_sidecar_blob
    )
    out_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zinfo = zipfile.ZipInfo(filename="x", date_time=(2024, 1, 1, 0, 0, 0))
        zinfo.external_attr = 0o644 << 16
        zf.writestr(zinfo, new_inner)


def enforce_manifest_dispatch_readiness(
    manifest: dict[str, Any],
    *,
    archive_path: Path,
) -> dict[str, Any]:
    """Fail-closed on exact-eval readiness unless custody invariants hold."""

    blockers = list(manifest.get("dispatch_blockers") or [])

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if not archive_path.is_file():
        block(f"materialized_archive_missing:{manifest_path(archive_path)}")
    else:
        actual_sha = sha256_of(archive_path)
        actual_bytes = archive_path.stat().st_size
        if manifest.get("new_archive_sha256") != actual_sha:
            block("materialized_archive_sha256_mismatch")
        if manifest.get("new_archive_bytes") != actual_bytes:
            block("materialized_archive_size_mismatch")

    if manifest.get("smoke_only") is True:
        block("smoke_only_not_exact_eval_ready")
    if manifest.get("runtime_smoke_checked") is not True:
        block("runtime_smoke_not_checked")
    no_op = manifest.get("no_op_detector")
    if not isinstance(no_op, dict) or no_op.get("sidecar_changed") is not True:
        block("sidecar_no_op_detector_not_changed")

    manifest["dispatch_blockers"] = blockers
    if blockers:
        manifest["ready_for_exact_eval_dispatch"] = False
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory for the resampled submission_dir + manifest",
    )
    p.add_argument(
        "--search-signal",
        choices=["proxy_mse", "joint_seg_pose"],
        default="proxy_mse",
        help="search objective (proxy_mse is fast; joint_seg_pose requires --accept-cpu-budget)",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream/videos/0.mkv",
        help="ground-truth video for proxy MSE",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=N_PAIRS,
        help="how many pairs to search (smoke=10; full=600)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run a 10-pair smoke search (verifies plumbing in ~30 sec)",
    )
    p.add_argument(
        "--accept-cpu-budget",
        action="store_true",
        help="acknowledge the search will take CPU-hours (required for joint_seg_pose)",
    )
    p.add_argument(
        "--encode-format",
        choices=["packed_661", "n_pairs_600"],
        default="packed_661",
        help="sidecar wire format (PACKED is more general; N_PAIRS limited to dim<16)",
    )
    p.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help=(
            "candidate latent perturbations per decoder forward chunk; default 1 "
            "preserves fastest measured scalar CPU path, larger values are experimental"
        ),
    )
    args = p.parse_args()

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / f"experiments/results/a1_per_pair_latent_sidecar_resampled_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.search_signal == "joint_seg_pose" and not args.accept_cpu_budget:
        sys.stderr.write(
            "[fatal] joint_seg_pose search requires --accept-cpu-budget acknowledgement "
            "(estimated ~30+ CPU hours on M-series ARM for full 600 pairs)\n"
        )
        return 2

    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        sys.stderr.write(
            f"[fatal] A1 archive SHA mismatch: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={actual_sha}\n"
        )
        return 2

    print(f"[ok] loading A1 archive components from {A1_ARCHIVE_PATH}", flush=True)
    components = load_a1_archive_components(A1_ARCHIVE_PATH)
    print(
        f"[ok] decoder tensors={len(components['decoder_sd'])} "
        f"latents shape={tuple(components['latents_pre_sidecar'].shape)} "
        f"old_sidecar_bytes={len(components['sidecar_blob_old'])}",
        flush=True,
    )

    n_pairs_to_search = 10 if args.smoke else args.n_pairs
    pair_indices = list(range(min(n_pairs_to_search, components["n_pairs"])))

    if args.search_signal == "proxy_mse":
        print(f"[start] proxy-mse search over {len(pair_indices)} pairs", flush=True)
        print(f"[ok] decoding ground-truth video {args.video_path}", flush=True)
        gt_pair_count = ground_truth_pairs_needed(pair_indices, components["n_pairs"])
        gt_frames = load_ground_truth_pairs(args.video_path, n_pairs=gt_pair_count)
        print(f"[ok] gt_frames shape={gt_frames.shape}", flush=True)
        log_path = out_dir / "sidecar_search.log"
        dims, delta_idx, search_meta = search_per_pair_proxy_mse(
            components,
            gt_frames,
            pair_indices=pair_indices,
            log_path=log_path,
            candidate_batch_size=args.candidate_batch_size,
        )
    else:
        sys.stderr.write(
            "[fatal] joint_seg_pose path not yet implemented in this tool — "
            "use proxy_mse and let GHA confirm. (Reactivation criterion: GPU "
            "available + budget approved.)\n"
        )
        return 2

    # Encode the new sidecar
    if args.encode_format == "n_pairs_600":
        new_sidecar = encode_sidecar_n_pairs(dims, delta_idx)
    else:
        new_sidecar = encode_sidecar_huff_enum(dims, delta_idx)
    print(
        f"[ok] new sidecar encoded: {len(new_sidecar)} bytes "
        f"(old was {len(components['sidecar_blob_old'])})",
        flush=True,
    )

    # Build the new submission_dir
    sub_dir = out_dir / "submission_dir"
    sub_dir.mkdir(exist_ok=True)
    new_archive_path = sub_dir / "archive.zip"
    write_resampled_archive(components, new_sidecar, new_archive_path)
    new_archive_sha = sha256_of(new_archive_path)
    new_archive_bytes = new_archive_path.stat().st_size
    # Copy A1's existing inflate.py, inflate.sh, src/* (the bias correction is
    # PRESERVED — we're only re-searching the latent sidecar, not the bias)
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.py", sub_dir / "inflate.py")
    inflate_sh_path = sub_dir / "inflate.sh"
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.sh", inflate_sh_path)
    inflate_sh_path.chmod(0o755)
    src_target = sub_dir / "src"
    src_target.mkdir(exist_ok=True)
    for fname in ("model.py", "codec.py"):
        shutil.copy2(A1_SUBMISSION_DIR / "src" / fname, src_target / fname)

    manifest = {
        "lane_id": "lane_a1_per_pair_latent_sidecar_resampled",
        "schema_version": "a1_per_pair_latent_sidecar_resampled_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "submission_name": f"a1_sidecar_resampled_{args.search_signal}_{timestamp}",
        "search_signal": args.search_signal,
        "search_meta": search_meta,
        "encode_format": args.encode_format,
        "old_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "old_archive_bytes": A1_EXPECTED_ARCHIVE_BYTES,
        "old_sidecar_bytes": len(components["sidecar_blob_old"]),
        "new_archive_path": manifest_path(new_archive_path),
        "new_archive_sha256": new_archive_sha,
        "new_archive_bytes": new_archive_bytes,
        "new_sidecar_bytes": len(new_sidecar),
        "delta_archive_bytes": new_archive_bytes - A1_EXPECTED_ARCHIVE_BYTES,
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": (
            f"[predicted; per-pair latent sidecar resampled on A1 substrate via "
            f"{args.search_signal}; pre-GHA-dispatch]"
        ),
        "dispatch_blockers": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
            "record runtime tree SHA and terminal dispatch claim row",
        ],
        "tag_discipline": {
            "before_eval": (
                f"[predicted; per-pair latent sidecar resampled via {args.search_signal}]"
            ),
            "after_eval": "[contest-CPU GHA Linux x86_64] iff GHA dispatch succeeds",
        },
        "smoke_only": args.smoke,
        "runtime_smoke_checked": False,
        "n_pairs_searched": len(pair_indices),
        "n_pairs_perturbed": int((dims != 255).sum()),
        "a1_canonical_baseline": {
            "score": 0.19284757743677347,
            "tag": "[contest-CPU GHA Linux x86_64]",
            "evidence_path": "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json",
        },
        "no_op_detector": {
            "old_inner_sidecar_sha256": hashlib.sha256(
                components["sidecar_blob_old"]
            ).hexdigest(),
            "new_inner_sidecar_sha256": hashlib.sha256(new_sidecar).hexdigest(),
            "sidecar_changed": components["sidecar_blob_old"] != new_sidecar,
        },
    }
    manifest = enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=new_archive_path,
    )
    (out_dir / "sidecar_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\n[done] sidecar_manifest written to "
        f"{manifest_path(out_dir / 'sidecar_manifest.json')}",
        flush=True,
    )
    print(
        f"  new_archive_bytes = {new_archive_bytes} "
        f"(Δ = {new_archive_bytes - A1_EXPECTED_ARCHIVE_BYTES:+d} from A1 baseline)",
        flush=True,
    )
    print(f"  new_archive_sha256 = {new_archive_sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
