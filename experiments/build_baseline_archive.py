#!/usr/bin/env python3
"""Build a baseline submission archive from canonical components.

Council 2026-04-27 strategic pivot: the saved "baseline_dilated_h64_0_90"
archive uses 48x64 masks (scoring 53.60), NOT the full-res 384x512 masks
that produced the historical 0.9001 record. This script regenerates the
correct full-res masks via SegNet on GT video and packages them with the
existing renderer + poses, producing an archive we can RE-VERIFY on CUDA.

Inputs (defaults match submissions/baseline_dilated_h64_0_90/):
  --renderer  : renderer.bin (ASYM)
  --poses     : optimized_poses.pt (or .bin)
  --gt-video  : upstream/videos/0.mkv
  --crf       : AV1 CRF for masks.mkv (default 50, matches "CRF=50" record)
  --output    : output archive.zip path

Output:
  archive.zip with renderer.bin + masks.mkv + optimized_poses.pt
  Plus a sidecar provenance.json with SHA256s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renderer", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90" / "renderer.bin")
    p.add_argument("--poses", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90" / "optimized_poses.pt")
    p.add_argument("--gt-video", type=Path,
                   default=REPO / "upstream" / "videos" / "0.mkv")
    p.add_argument("--crf", type=int, default=50,
                   help="AV1 CRF for masks (default 50, matches '0.9001 + CRF=50' record)")
    p.add_argument("--output", type=Path,
                   default=REPO / "submissions" / "baseline_dilated_h64_0_90"
                   / "archive_rebuilt_full_res.zip")
    p.add_argument("--device", type=str, default=None,
                   help="cpu/mps/cuda for SegNet. DEFAULTS TO CUDA-REQUIRED. "
                        "MPS produces different SegNet outputs per CLAUDE.md "
                        "→ different mask bytes → different score. CUDA is "
                        "the only fully reproducible choice.")
    p.add_argument("--seed", type=int, default=1234,
                   help="Random seed (matches upstream/evaluate.py default).")
    p.add_argument("--half-frame", action="store_true", default=False,
                   help="Quantizr trick: encode only 600 ODD-frame masks "
                        "(frames 1, 3, 5, ..., 1199) instead of 1200. The "
                        "inflate path detects this and warps to recover even "
                        "frames. Roughly halves the mask byte count. Required "
                        "to reach the historical 0.9001 archive size (~338KB).")
    p.add_argument("--with-uniward-delta", type=Path, default=None,
                   help="Path to a Lane C delta.bin produced by "
                        "experiments/optimize_uniward_delta.py. If set, the "
                        "file is bundled into the archive at the canonical "
                        "name 'delta.bin'. The inflate path detects it and "
                        "applies a sparse, L∞-bounded perturbation to the "
                        "rendered frames BEFORE upscale (no scorer at "
                        "inflate — pure additive lookup table). Council "
                        "Lane C target: ≤5KB blob, +0.003 rate cost, "
                        "predicted -0.05 to -0.20 distortion when stacked "
                        "with Lane A pose-TTO. NOTE: by default, a δ marked "
                        "compliance_status=pending_ruling will be REFUSED — "
                        "see --allow-pending-compliance for the explicit "
                        "override.")
    # Codex R5 HIGH — silent contest-noncompliance gate. Lane C δ.bin is a
    # scorer-derived artifact; without an explicit council ruling, bundling
    # it into a submission archive risks shipping a non-compliant entry.
    # We make that risk an EXPLICIT operator decision instead of a silent
    # default. The flag must be passed every time, and is recorded into
    # provenance so audit logs show it.
    p.add_argument("--allow-pending-compliance", action="store_true",
                   default=False,
                   help="Override the pending-ruling refusal for Lane C δ. "
                        "Required to bundle a δ.bin whose header carries "
                        "compliance_status=pending_ruling (the default for "
                        "any newly-built δ). Without this flag the script "
                        "exits non-zero. The override is recorded in the "
                        "provenance JSON so any downstream auditor can see "
                        "that the operator explicitly opted in. ONLY pass "
                        "this for diagnostic / paper-figure builds — never "
                        "for a contest submission until the Yousfi PR #35 "
                        "strict-scorer-rule ruling is in.")
    args = p.parse_args()

    for label, path in (("renderer", args.renderer), ("poses", args.poses),
                        ("gt-video", args.gt_video)):
        if not path.exists():
            raise SystemExit(f"--{label} does not exist: {path}")
    # Validate Lane C δ if requested.
    uniward_compliance_status: str | None = None
    if args.with_uniward_delta is not None:
        if not args.with_uniward_delta.exists():
            raise SystemExit(
                f"--with-uniward-delta does not exist: {args.with_uniward_delta}"
            )
        # Codex R5 HIGH fix — silent contest-noncompliance gate. Read the
        # δ.bin header BEFORE we open the SegNet / write the archive so the
        # operator wastes no work on a build that would be refused anyway.
        from tac.uniward_delta import (
            unpack_sparse_delta as _uwd_unpack_for_check,
            COMPLIANCE_PENDING as _UWD_PENDING,
            COMPLIANCE_APPROVED as _UWD_APPROVED,
            COMPLIANCE_REJECTED as _UWD_REJECTED,
        )
        try:
            _uwd_blob = args.with_uniward_delta.read_bytes()
            _uwd_spec_check = _uwd_unpack_for_check(_uwd_blob, device="cpu")
        except Exception as e:  # pragma: no cover — corrupt input
            raise SystemExit(
                f"FATAL: --with-uniward-delta could not be parsed: {e!r}. "
                f"Refusing to bundle an unverifiable δ into the archive."
            )
        uniward_compliance_status = _uwd_spec_check.compliance_status
        if uniward_compliance_status == _UWD_REJECTED:
            raise SystemExit(
                f"FATAL: --with-uniward-delta is marked "
                f"compliance_status={_UWD_REJECTED!r}. This δ has been "
                f"explicitly flagged as non-compliant by the council and "
                f"may NEVER be bundled. Build aborted."
            )
        if uniward_compliance_status == _UWD_PENDING and not args.allow_pending_compliance:
            raise SystemExit(
                "FATAL: --with-uniward-delta is marked "
                f"compliance_status={_UWD_PENDING!r}. Lane C δ.bin is a "
                "SCORER-DERIVED artifact; Yousfi PR #35 strict-scorer-rule "
                "may class this as non-compliant. Refusing to bundle.\n"
                "  To proceed for a diagnostic / paper-figure build only, "
                "pass --allow-pending-compliance — the override is recorded "
                "in the archive provenance JSON for audit. DO NOT use this "
                "for a contest submission until the council ruling lands in "
                ".omx/research/findings.md."
            )
        if uniward_compliance_status == _UWD_APPROVED:
            print(
                f"[build] Lane C δ compliance_status={_UWD_APPROVED!r} — "
                f"bundling without override.",
            )
        elif args.allow_pending_compliance:
            print(
                f"\n{'=' * 78}\n"
                f"[build] WARNING: bundling δ with compliance_status="
                f"{uniward_compliance_status!r} via "
                f"--allow-pending-compliance override.\n"
                f"  This archive is NOT contest-compliant until council "
                f"ruling on Yousfi PR #35.\n"
                f"  Tag any score [lane-c-pending-ruling] in the run-log.\n"
                f"{'=' * 78}\n",
            )
    elif args.allow_pending_compliance:
        # Catch the operator-error case: --allow-pending-compliance set but
        # no δ to compliance-gate. Surface it loudly rather than silently no-op.
        print(
            "[build] NOTE: --allow-pending-compliance was passed but "
            "--with-uniward-delta is None — flag has no effect.",
        )

    import os
    import torch
    import av  # noqa: F401  # ensure pyav available
    from tac.scorer import extract_gt_masks, load_scorers
    from tac.mask_codec import encode_masks

    # Determinism non-negotiable per CLAUDE.md. Set BEFORE any cuBLAS call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Note: torch.use_deterministic_algorithms(True) would force errors on
    # any nondet op. SegNet inference is no_grad pure-forward so it's safe
    # to skip; documenting here so future readers know this is intentional.

    if args.device is None:
        # Default to CUDA. MPS produces DIFFERENT SegNet outputs than CUDA
        # (per memory feedback_mps_cuda_drift_critical: 2x SegNet drift) —
        # generating masks on MPS would produce a DIFFERENT byte-level
        # archive than one generated on CUDA, breaking deterministic
        # reproducibility against contest CUDA eval.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise SystemExit(
                "FATAL: CUDA not available. Mask generation MUST run on CUDA "
                "for 100% deterministic reproducibility against contest eval. "
                "MPS produces different SegNet outputs (2x drift) → different "
                "masks.mkv bytes → different score. Run this script on a "
                "Vast.ai 4090 OR pass --device cpu and accept that the "
                "rebuilt archive will NOT byte-match a CUDA-built one."
            )
    else:
        device = torch.device(args.device)
        if str(device) == "mps":
            print("[build] WARNING: MPS produces different SegNet outputs "
                  "than CUDA (per feedback_mps_cuda_drift_critical). The "
                  "rebuilt archive bytes will NOT match a CUDA-built one. "
                  "This is acceptable for development smoke-testing only.",
                  file=sys.stderr)
    print(f"[build] device={device}")

    # Stage 1: decode GT video to (H, W, 3) uint8 frames
    print(f"[build] decoding GT video {args.gt_video}")
    t0 = time.monotonic()
    import av as _av
    container = _av.open(str(args.gt_video))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    container.close()
    print(f"[build] decoded {len(frames)} frames in {time.monotonic()-t0:.1f}s")

    # Stage 2: SegNet → masks at 384x512
    print(f"[build] loading SegNet from upstream/models/")
    _, segnet = load_scorers(
        posenet_path=REPO / "upstream" / "models" / "posenet.safetensors",
        segnet_path=REPO / "upstream" / "models" / "segnet.safetensors",
        device=device,
        upstream_dir=REPO / "upstream",
    )
    segnet.eval()
    print(f"[build] extracting masks at 384x512 via SegNet")
    t0 = time.monotonic()
    masks = extract_gt_masks(frames, segnet, device, batch_size=8)
    print(f"[build] masks shape={tuple(masks.shape)} in {time.monotonic()-t0:.1f}s")

    # Half-frame mode: keep only ODD-indexed frames (1, 3, 5, ..., 1199).
    # The inflate side detects 600-mask input and reconstructs even frames
    # via zoom-flow warp (see submissions/robust_current/inflate_renderer.py
    # _expand_half_frame_masks). This is the Quantizr trick — roughly halves
    # the mask byte count without quality loss because the renderer's motion
    # predictor handles the t -> t+1 warp anyway.
    if args.half_frame:
        masks = masks[1::2]  # frames 1, 3, 5, ..., 1199 → 600 frames
        print(f"[build] HALF-FRAME mode: kept odd-indexed only, shape={tuple(masks.shape)}")

    # Stage 3: encode masks as AV1 at requested CRF
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        masks_path = td_path / "masks.mkv"
        print(f"[build] encoding masks at CRF={args.crf}")
        t0 = time.monotonic()
        size = encode_masks(masks, masks_path, crf=args.crf, fps=20)
        print(f"[build] masks.mkv = {size:,} bytes in {time.monotonic()-t0:.1f}s")

        # Stage 4: build archive.zip
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(args.renderer, arcname="renderer.bin")
            z.write(masks_path, arcname="masks.mkv")
            z.write(args.poses, arcname="optimized_poses.pt")
            if args.with_uniward_delta is not None:
                z.write(args.with_uniward_delta, arcname="delta.bin")
                print(f"[build] bundled UNIWARD δ from {args.with_uniward_delta} "
                      f"({args.with_uniward_delta.stat().st_size:,} bytes)")
        archive_size = args.output.stat().st_size
        rate_unscaled = archive_size / 37545489
        rate_contribution = 25 * rate_unscaled
        print(f"\n=== Archive built ===")
        print(f"  Path: {args.output}")
        print(f"  Bytes: {archive_size:,}")
        print(f"  Rate (unscaled): {rate_unscaled:.6f}")
        print(f"  Rate (score contribution): {rate_contribution:.4f}")

        # Provenance — every input that affects the output bytes is
        # SHA-pinned so a future re-run can detect any drift.
        segnet_path = REPO / "upstream" / "models" / "segnet.safetensors"
        gpu_model = None
        if torch.cuda.is_available():
            try:
                gpu_model = torch.cuda.get_device_name(0)
            except Exception:
                pass
        prov = {
            "schema_version": 1,
            "tool": "experiments/build_baseline_archive.py",
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "gpu_model": gpu_model,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "crf": args.crf,
            "seed": args.seed,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "archive_path": str(args.output),
            "archive_size_bytes": archive_size,
            "archive_sha256": _sha256(args.output),
            "rate_unscaled": rate_unscaled,
            "rate_score_contribution": rate_contribution,
            "segnet_weights_sha256": _sha256(segnet_path) if segnet_path.exists() else None,
            "components": {
                "renderer.bin": {
                    "source": str(args.renderer),
                    "size_bytes": args.renderer.stat().st_size,
                    "sha256": _sha256(args.renderer),
                },
                "masks.mkv": {
                    "source": "rebuilt from GT via SegNet at 384x512 CRF=50",
                    "size_bytes": size,
                    "sha256": _sha256(masks_path),
                },
                "optimized_poses.pt": {
                    "source": str(args.poses),
                    "size_bytes": args.poses.stat().st_size,
                    "sha256": _sha256(args.poses),
                },
                **({"delta.bin": {
                    "source": str(args.with_uniward_delta),
                    "size_bytes": args.with_uniward_delta.stat().st_size,
                    "sha256": _sha256(args.with_uniward_delta),
                    "kind": "lane_c_uniward_sparse_delta",
                    # Codex R5 HIGH fix — record the compliance gate state
                    # so any future auditor reading provenance.json can see
                    # whether this build went through the pending-ruling
                    # override path. The flag MUST appear in the audit
                    # trail; CLAUDE.md "Strategic Secrecy" rule means we
                    # never ship a non-compliant archive without showing
                    # exactly when and how the override was approved.
                    "compliance_status": uniward_compliance_status,
                    "allow_pending_compliance_override": bool(
                        args.allow_pending_compliance
                    ),
                }} if args.with_uniward_delta is not None else {}),
            },
        }
        prov_path = args.output.with_suffix(".provenance.json")
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2)
        print(f"  Provenance: {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
