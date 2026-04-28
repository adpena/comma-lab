#!/usr/bin/env python3
"""Lane LM-A: build a zero-archive-cost pose archive from a Lane A archive.

This is a STANDALONE tool that takes the verified Lane A archive (renderer +
masks + optimized_poses) and produces a NEW archive that OMITS
``optimized_poses.pt`` entirely, writing the ``zero_cost_poses_v1`` sentinel
in its place. The inflate side computes per-pair 6-DOF poses from lane-mark
mask displacement (``tac.lane_mark_pose.compute_zero_cost_poses_from_masks``)
when ``INFLATE_ZERO_COST_POSES=1`` is set, so the on-disk pose tensor is
unnecessary.

Strategic context
-----------------
* Memory ``project_lane_marking_speed_estimation``: lane-mark centroids
  encode ego-motion radial displacement at zero archive cost.
* Memory ``project_posenet_rank1_discovery``: PoseNet's effective Jacobian
  rank is 1.008 — only dim 0 (scalar radial zoom from FoE) carries signal.
* Lane A 1.15 [contest-CUDA] currently bundles ``optimized_poses.pt`` at
  ~15.3 KB. Removing it saves -0.010 score contribution at the rate term;
  the distortion impact is bounded by Yousfi's geometric analysis (≤0.18
  if dims 1-5 are predicted as zeros — the Fridrich strategy).

Predicted score band: [1.05, 1.15] [contest-CUDA].
* Floor (1.05): rate savings of ~0.010 land cleanly + lane-mark-derived
  dim 0 matches Lane A's optimized dim 0 closely enough that PoseNet
  distortion does not regress.
* Ceiling (1.15): the analytical estimate is too noisy and PoseNet
  distortion regresses by exactly the rate savings — net flat vs Lane A.

Why this is contest-compliant
-----------------------------
Per CLAUDE.md non-negotiable strict-scorer-rule: NO scorers loaded at
inflate time. The pose computation is pure geometric centroid arithmetic
on the mask tensor (already in the archive as masks.mkv) — it does NOT
load PoseNet or SegNet. See ``src/tac/lane_mark_pose.py``.

CLI
---
    python experiments/build_zero_cost_pose_archive.py \\
        --lane-a-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --output         experiments/results/lane_lm_a/archive_lane_lm_a.zip

Outputs
-------
* ``--output`` : the new archive (renderer + masks + sentinel).
* ``<output>.provenance.json`` : SHA256s + correlation report against the
  source Lane A poses (so we can sanity-check the lane-mark estimate
  vs the optimized values BEFORE burning a Vast.ai eval).

Calibration sanity check
------------------------
The script ALWAYS extracts the source archive's ``optimized_poses.pt``
(when present) and computes a Pearson correlation between Lane A's pose
dim 0 and the lane-mark-derived dim 0. A correlation < 0.30 raises a
loud warning (the analytical estimate is decoupled from the optimized
target — proceeding will likely regress PoseNet). A correlation > 0.50
is the green-light condition for the Lane LM-A predicted band.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

from tac.lane_mark_pose import (  # noqa: E402
    POSENET_DIM0_MEAN,
    POSENET_DIM0_PER_LOGZOOM,
    ZERO_COST_POSES_SENTINEL,
    compute_zero_cost_poses_from_masks,
)

# Reuse build_baseline_archive's deterministic-zip helper so the output
# bytes are reproducible across reruns (codex R5-r6 #5).
_DET_ZIP_DATE_TIME = (2026, 4, 27, 0, 0, 0)
_DET_ZIP_EXTERNAL_ATTR = (0o644 & 0xFFFF) << 16
_DET_ZIP_CREATE_SYSTEM = 3


def _det_zip_write(
    z: zipfile.ZipFile,
    arcname: str,
    src: Path,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 9,
) -> None:
    """Write ``src`` into ``z`` with FIXED metadata (deterministic-zip).

    Vanilla ``z.write`` embeds OS perm bits + source mtime which makes
    archives non-reproducible across reruns. This wrapper forces a
    constant date_time + Unix perms.
    """
    info = zipfile.ZipInfo(filename=arcname, date_time=_DET_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _DET_ZIP_EXTERNAL_ATTR
    info.create_system = _DET_ZIP_CREATE_SYSTEM
    with open(src, "rb") as f:
        data = f.read()
    z.writestr(info, data, compress_type=compress_type, compresslevel=compresslevel)


def _det_zip_writestr(
    z: zipfile.ZipFile,
    arcname: str,
    data: bytes,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int = 9,
) -> None:
    info = zipfile.ZipInfo(filename=arcname, date_time=_DET_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = _DET_ZIP_EXTERNAL_ATTR
    info.create_system = _DET_ZIP_CREATE_SYSTEM
    z.writestr(info, data, compress_type=compress_type, compresslevel=compresslevel)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _decode_masks_from_mkv(masks_mkv: Path) -> torch.Tensor:
    """Decode ``masks.mkv`` to a ``(N, H, W)`` long-tensor of class indices.

    Mirrors the gray-scale decode used by the inflate-side mask reader and
    by ``test_lane_mark_pose.test_zero_cost_pose_smoke_on_real_masks``:
    pixel value -> round(value / (255 // 4)) -> clamp [0, 4].
    """
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover — defensive
        raise SystemExit(
            f"PyAV is required to decode {masks_mkv}: {e!r}. "
            "Install with `uv pip install av` and retry."
        )
    container = av.open(str(masks_mkv))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="gray")
        frames.append(torch.from_numpy(arr))
    container.close()
    if not frames:
        raise SystemExit(f"FATAL: {masks_mkv} decoded zero frames")
    masks_uint8 = torch.stack(frames)
    scale_factor = 255 // 4
    return (masks_uint8.float() / scale_factor).round().long().clamp(0, 4)


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D float tensors.

    Uses population std (unbiased=False) so that corr(x, x) == 1 exactly
    regardless of N (Bessel correction would give 1 - 1/N).
    """
    a_c = a - a.mean()
    b_c = b - b.mean()
    a_std = a_c.std(unbiased=False).clamp(min=1e-10)
    b_std = b_c.std(unbiased=False).clamp(min=1e-10)
    return float(((a_c * b_c).mean() / (a_std * b_std)).item())


def _calibration_report(
    masks: torch.Tensor,
    lane_a_poses: torch.Tensor | None,
) -> dict:
    """Compute lane-mark-derived poses + sanity-check vs Lane A baseline.

    Returns a dict suitable for embedding in ``provenance.json``. When
    ``lane_a_poses`` is supplied, includes a correlation report between
    lane-mark dim 0 and Lane A's optimized dim 0; otherwise reports only
    the derived statistics.
    """
    derived = compute_zero_cost_poses_from_masks(masks)
    derived_dim0 = derived[:, 0]
    report: dict = {
        "num_pairs": int(derived.shape[0]),
        "pose_dim": int(derived.shape[1]),
        "derived_dim0_mean": float(derived_dim0.mean().item()),
        "derived_dim0_std": float(derived_dim0.std().item()),
        "derived_dim0_min": float(derived_dim0.min().item()),
        "derived_dim0_max": float(derived_dim0.max().item()),
        "posenet_dim0_mean_constant": POSENET_DIM0_MEAN,
        "posenet_dim0_per_logzoom_constant": POSENET_DIM0_PER_LOGZOOM,
    }
    if lane_a_poses is not None:
        if lane_a_poses.ndim != 2 or lane_a_poses.shape[1] < 1:
            raise SystemExit(
                f"FATAL: Lane A poses must be (N, ≥1), got "
                f"shape {tuple(lane_a_poses.shape)}"
            )
        if lane_a_poses.shape[0] != derived.shape[0]:
            raise SystemExit(
                f"FATAL: Lane A poses have {lane_a_poses.shape[0]} pairs, "
                f"but masks decode to {derived.shape[0]} pairs. The masks "
                f"and poses came from different runs — do NOT package this "
                f"archive (the renderer would see mismatched conditioning)."
            )
        lane_a_dim0 = lane_a_poses[:, 0].float()
        corr = _correlation(derived_dim0, lane_a_dim0)
        rmse = float(((derived_dim0 - lane_a_dim0) ** 2).mean().sqrt().item())
        max_abs_err = float((derived_dim0 - lane_a_dim0).abs().max().item())
        report.update({
            "lane_a_dim0_mean": float(lane_a_dim0.mean().item()),
            "lane_a_dim0_std": float(lane_a_dim0.std().item()),
            "lane_a_dim0_min": float(lane_a_dim0.min().item()),
            "lane_a_dim0_max": float(lane_a_dim0.max().item()),
            "correlation_dim0": corr,
            "rmse_dim0": rmse,
            "max_abs_err_dim0": max_abs_err,
        })
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lane-a-archive", type=Path, required=True,
        help="Source Lane A archive (.zip) to derive Lane LM-A from. "
             "Must contain renderer.bin, masks.mkv, and (for sanity check) "
             "optimized_poses.pt.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output archive path. Will contain renderer.bin + masks.mkv + "
             f"{ZERO_COST_POSES_SENTINEL} sentinel (no optimized_poses.pt).",
    )
    p.add_argument(
        "--min-correlation", type=float, default=0.30,
        help="Minimum acceptable Pearson correlation between lane-mark dim 0 "
             "and Lane A's optimized dim 0. Below this the script aborts to "
             "save a Vast.ai eval. Set to 0.0 to bypass the gate (NOT "
             "recommended — a low correlation strongly predicts a PoseNet "
             "regression).",
    )
    p.add_argument(
        "--allow-low-correlation", action="store_true", default=False,
        help="Bypass the --min-correlation gate. Use only if the operator "
             "explicitly accepts the predicted PoseNet regression (e.g. for "
             "calibration sweeps where the rate-side win is the focus).",
    )
    args = p.parse_args()

    if not args.lane_a_archive.exists():
        raise SystemExit(f"--lane-a-archive does not exist: {args.lane_a_archive}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # ---- Stage 1: extract source archive ----
        with zipfile.ZipFile(args.lane_a_archive) as z:
            members = set(z.namelist())
            required = {"renderer.bin", "masks.mkv"}
            missing = required - members
            if missing:
                raise SystemExit(
                    f"FATAL: --lane-a-archive missing required members: "
                    f"{sorted(missing)}. Found: {sorted(members)}"
                )
            for name in ("renderer.bin", "masks.mkv"):
                z.extract(name, td_path)
            lane_a_poses_path: Path | None = None
            if "optimized_poses.pt" in members:
                z.extract("optimized_poses.pt", td_path)
                lane_a_poses_path = td_path / "optimized_poses.pt"
            elif "poses.pt" in members:
                z.extract("poses.pt", td_path)
                lane_a_poses_path = td_path / "poses.pt"
            else:
                print(
                    f"[build-zcp] WARNING: source archive has no "
                    f"optimized_poses.pt / poses.pt. Skipping the "
                    f"calibration sanity check.",
                    file=sys.stderr,
                )

        renderer_path = td_path / "renderer.bin"
        masks_path = td_path / "masks.mkv"

        # ---- Stage 2: decode masks + compute calibration report ----
        print(f"[build-zcp] decoding masks from {masks_path} ...", file=sys.stderr)
        masks = _decode_masks_from_mkv(masks_path)
        print(
            f"[build-zcp] decoded {masks.shape[0]} frames at "
            f"{masks.shape[1]}x{masks.shape[2]}",
            file=sys.stderr,
        )

        lane_a_poses_tensor: torch.Tensor | None = None
        if lane_a_poses_path is not None:
            lane_a_poses_tensor = torch.load(
                str(lane_a_poses_path), map_location="cpu", weights_only=True,
            )
            print(
                f"[build-zcp] loaded Lane A poses for sanity check: "
                f"shape {tuple(lane_a_poses_tensor.shape)}",
                file=sys.stderr,
            )

        report = _calibration_report(masks, lane_a_poses_tensor)
        corr = report.get("correlation_dim0")
        if corr is not None:
            print(
                f"[build-zcp] CALIBRATION: lane-mark vs Lane A dim 0 "
                f"correlation={corr:.4f} rmse={report['rmse_dim0']:.4f} "
                f"max_abs_err={report['max_abs_err_dim0']:.4f}",
                file=sys.stderr,
            )
            if corr < args.min_correlation and not args.allow_low_correlation:
                raise SystemExit(
                    f"FATAL: lane-mark dim 0 correlation with Lane A "
                    f"dim 0 is {corr:.4f}, below --min-correlation "
                    f"{args.min_correlation}. The analytical estimate is "
                    f"too decoupled from the optimized values to predict "
                    f"a non-regressing PoseNet score. Re-investigate the "
                    f"calibration constants in tac.lane_mark_pose, OR pass "
                    f"--allow-low-correlation if you accept the predicted "
                    f"PoseNet regression."
                )

        # ---- Stage 3: assemble output archive ----
        sentinel_path = td_path / ZERO_COST_POSES_SENTINEL
        sentinel_path.write_bytes(b"")  # 0-byte marker

        det_entries: list[tuple[str, Path]] = [
            ("renderer.bin", renderer_path),
            ("masks.mkv", masks_path),
            (ZERO_COST_POSES_SENTINEL, sentinel_path),
        ]
        det_entries.sort(key=lambda kv: kv[0])
        with zipfile.ZipFile(
            args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9,
        ) as z:
            for arcname, src in det_entries:
                _det_zip_write(z, arcname, src)

        archive_size = args.output.stat().st_size
        rate_unscaled = archive_size / 37545489
        rate_contribution = 25 * rate_unscaled
        source_size = args.lane_a_archive.stat().st_size
        bytes_saved = source_size - archive_size
        score_savings = (bytes_saved / 37545489) * 25
        print(
            f"\n=== Lane LM-A archive built ===\n"
            f"  Source (Lane A): {args.lane_a_archive} ({source_size:,} bytes)\n"
            f"  Output:          {args.output} ({archive_size:,} bytes)\n"
            f"  Bytes saved:     {bytes_saved:,} (-{score_savings:.4f} score)\n"
            f"  Rate (unscaled): {rate_unscaled:.6f}\n"
            f"  Rate (score):    {rate_contribution:.4f}",
            file=sys.stderr,
        )

        # ---- Stage 4: provenance ----
        prov = {
            "schema_version": 1,
            "tool": "experiments/build_zero_cost_pose_archive.py",
            "lane": "lane_lm_a",
            "predicted_band": [1.05, 1.15],
            "built_at_utc": started_at,
            "source_archive_path": str(args.lane_a_archive),
            "source_archive_size_bytes": source_size,
            "source_archive_sha256": _sha256(args.lane_a_archive),
            "output_archive_path": str(args.output),
            "output_archive_size_bytes": archive_size,
            "output_archive_sha256": _sha256(args.output),
            "bytes_saved_vs_source": bytes_saved,
            "score_savings_vs_source": score_savings,
            "rate_unscaled": rate_unscaled,
            "rate_score_contribution": rate_contribution,
            "min_correlation_gate": args.min_correlation,
            "allow_low_correlation": bool(args.allow_low_correlation),
            "calibration_report": report,
            "components": {
                "renderer.bin": {
                    "source": "extracted from --lane-a-archive",
                    "size_bytes": renderer_path.stat().st_size,
                    "sha256": _sha256(renderer_path),
                },
                "masks.mkv": {
                    "source": "extracted from --lane-a-archive",
                    "size_bytes": masks_path.stat().st_size,
                    "sha256": _sha256(masks_path),
                },
                ZERO_COST_POSES_SENTINEL: {
                    "source": "0-byte sentinel; lane-mark mask displacement at inflate",
                    "size_bytes": 0,
                    "sha256": (
                        # SHA256 of empty bytes
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934"
                        "ca495991b7852b855"
                    ),
                    "kind": "lane_m_plus_zero_cost_pose_sentinel",
                    "note": (
                        "Inflate computes per-pair 6-DOF poses via "
                        "tac.lane_mark_pose.compute_zero_cost_poses_from_masks. "
                        "Requires env INFLATE_ZERO_COST_POSES=1."
                    ),
                },
            },
        }
        prov_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
        prov_path.write_text(json.dumps(prov, indent=2))
        print(f"[build-zcp] provenance: {prov_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
