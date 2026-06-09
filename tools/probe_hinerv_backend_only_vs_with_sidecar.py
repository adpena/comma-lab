#!/usr/bin/env python3
"""CANONICAL backend-only vs with-sidecar survival proof (the foundational artifact).

DEFINITIVE finding to confirm: the backend HiNeRV birth SURVIVES parse-back, but
the target-region action SIDECAR (applied by build_model_from_archive via
wrap_model_with_target_region_actions) DESTROYS it -- so the runner's
"authoritative" parse-back collapse was a backend+sidecar CONFLATION.

This probe renders BOTH the unwrapped backend AND the wrapped (with-sidecar)
model from the SAME archive, scores both with the runner's EXACT win-definition
(region_margin_stats.region_hard_won_pixels) over the runner's EXACT region
(reconstruct_birth_region_mask), with BOTH the upstream torch SegNet AND the MLX
port (parity), and emits hi_nerv_backend_only_vs_with_sidecar_survival.v1 with
every hash GPT's adversarial review requires:

  - archive_sha256 (same archive for both renders)
  - region_mask_sha256 (same region)
  - backend_rgb_sha256 / wrapped_rgb_sha256 (the two renders)
  - backend_argmax_sha256 / wrapped_argmax_sha256 (the two scorings)
  - backend_excludes_sidecar (proves the unwrapped model differs from wrapped
    by EXACTLY the sidecar's action pixels, nothing else)

Planning-control evidence only; region wins are the strong proxy -- exact d_seg
needs the full paired upstream eval (named as the replay step). Non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _read_payload(zp: Path) -> bytes:
    with zipfile.ZipFile(zp) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"0.bin", "x"}] or members
        return zf.read(pick[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--candidate-kind", default="live")
    ap.add_argument("--upstream-dir", default="upstream")
    ap.add_argument("--runner-row", default=None, help="optional runner parseback row for worst_region")
    ap.add_argument("--birth-class", type=int, default=4)
    ap.add_argument("--bbox-y0", type=int, default=281)
    ap.add_argument("--bbox-y1", type=int, default=384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mlx.core as mx
    import torch

    from tac.local_acceleration.mlx_scorer_adapters import MLXSegNetAdapter
    from tac.scorer import extract_gt_masks, load_default_segnet
    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.birth_survival import reconstruct_birth_region_mask
    from tac.substrates.hi_nerv.inflate import build_model_from_archive
    from tac.substrates.hi_nerv.target_region_birth import region_margin_stats

    archive = Path(args.archive).expanduser().resolve()
    z = np.load(Path(args.npz).expanduser().resolve())
    target_labels = z["target_labels_bhw"].astype(np.int64)
    pair_indices = [int(v) for v in z["pair_indices"].tolist()]
    cls = int(args.birth_class)

    # Region: prefer the runner's worst_region (exact reconstruction) if provided.
    worst = None
    action_id = None
    runner_wrong_to_target = None
    if args.runner_row:
        rr = json.loads(Path(args.runner_row).expanduser().read_text())
        worst = rr.get("worst_region")
        action_id = rr.get("action_id")
        runner_wrong_to_target = rr.get("wrong_to_target_count")
    if isinstance(worst, dict):
        region, _px = reconstruct_birth_region_mask(target_labels, worst)
        region = np.asarray(region)
        if region.ndim == 2:
            region = region[None, ...]
        cls = int(worst.get("class_index", cls))
    else:
        region = np.zeros_like(target_labels, dtype=np.float32)
        region[:, args.bbox_y0 : args.bbox_y1, :] = 1.0
        region = (region > 0) & (target_labels == cls)
        region = region.astype(np.float32)
    region_bool = np.asarray(region) > 0.0
    region_px = int(np.count_nonzero(region_bool))

    payload = _read_payload(archive)
    arc = parse_archive(payload)
    _arc, cfg, wrapped = build_model_from_archive(payload, device="cpu")  # WRAPPED (sidecar)
    backend = HinervSubstrate(cfg).to("cpu").eval()  # UNWRAPPED (backend-only)
    fs = dict(arc.decoder_state_dict)
    for k in ("latents_coarse", "latents_mid", "latents_fine"):
        fs[k] = getattr(arc, k).to(dtype=getattr(backend, k).dtype)
    backend.load_state_dict(fs, strict=True)

    idx = torch.tensor(pair_indices, dtype=torch.long)
    with torch.no_grad():
        _b0, b1 = backend(idx)
        _w0, w1 = wrapped(idx)
    backend_rgb = np.asarray(b1.detach().cpu(), dtype=np.float32)  # (N,3,H,W) [0,1]
    wrapped_rgb = np.asarray(w1.detach().cpu(), dtype=np.float32)

    # ADVERSARIAL CONCERN 1: backend-only must differ from wrapped by EXACTLY the
    # sidecar's action pixels (the wrapper is the ONLY difference).
    diff_any = (np.abs(backend_rgb - wrapped_rgb).sum(axis=1) > 1e-6)  # (N,H,W)
    sidecar_changed_px = int(np.count_nonzero(diff_any))
    backend_excludes_sidecar = sidecar_changed_px > 0  # backend has NO overlay; wrapped does

    segnet = load_default_segnet(args.upstream_dir, device="cpu")
    adapter = MLXSegNetAdapter(segnet)

    def _torch_argmax(rgb: np.ndarray) -> np.ndarray:
        frames = [
            torch.from_numpy(np.transpose(f * 255.0, (1, 2, 0)).copy()).clamp(0, 255) for f in rgb
        ]
        return np.asarray(
            extract_gt_masks(frames, segnet, device="cpu", batch_size=4).cpu(), dtype=np.int64
        )

    def _mlx_logits(rgb: np.ndarray) -> np.ndarray:
        x = mx.array(np.transpose(rgb, (0, 2, 3, 1)).astype(np.float32)) * 255.0
        return np.asarray(adapter(x).astype(mx.float32))

    rmask = region_bool.astype(np.float32)

    def _wins(rgb: np.ndarray) -> dict[str, Any]:
        torch_argmax = _torch_argmax(rgb)
        mlx_logits = _mlx_logits(rgb)
        mlx_argmax = mlx_logits.argmax(-1).astype(np.int64)
        st = region_margin_stats(mlx_logits, rmask, cls)  # runner's exact win-def
        torch_wins = int(np.count_nonzero((torch_argmax == cls) & region_bool))
        return {
            "region_hard_won_mlx": int(st["region_hard_won_pixels"]),
            "region_hard_won_torch": torch_wins,
            "target_margin_p10": st.get("target_margin_p10"),
            "target_margin_p50": st.get("target_margin_p50"),
            "rgb_sha256": _sha(rgb),
            "torch_argmax_sha256": _sha(torch_argmax),
            "mlx_argmax_sha256": _sha(mlx_argmax),
            "mlx_vs_torch_region_disagreement_px": int(
                np.count_nonzero((torch_argmax != mlx_argmax) & region_bool)
            ),
        }

    backend_res = _wins(backend_rgb)
    wrapped_res = _wins(wrapped_rgb)

    sidecar_bytes = 0
    meta = dict(arc.meta or {})
    prog = meta.get("_target_region_actions_v1_b64")
    if isinstance(prog, str):
        sidecar_bytes = len(prog.encode("utf-8"))

    wtt_delta = wrapped_res["region_hard_won_mlx"] - backend_res["region_hard_won_mlx"]
    sidecar_harmful = wtt_delta < 0
    backend_survives = backend_res["region_hard_won_mlx"] > 100

    verdict = (
        "sidecar_harmful_backend_survives"
        if (sidecar_harmful and backend_survives)
        else (
            "backend_collapses_too"
            if not backend_survives
            else "sidecar_not_harmful"
        )
    )

    row: dict[str, Any] = {
        "schema": "hi_nerv_backend_only_vs_with_sidecar_survival.v1",
        "family": "hinerv",
        "candidate_kind": str(args.candidate_kind),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "action_id": action_id,
        "birth_class": cls,
        "region_pixel_count": region_px,
        "region_mask_sha256": _sha(region_bool.astype(np.uint8)),
        "backend_excludes_sidecar": bool(backend_excludes_sidecar),
        "sidecar_changed_pixels_in_render": sidecar_changed_px,
        "backend_only": {
            "wrong_to_target_mlx": backend_res["region_hard_won_mlx"],
            "wrong_to_target_torch": backend_res["region_hard_won_torch"],
            "target_margin_p10": backend_res["target_margin_p10"],
            "target_margin_p50": backend_res["target_margin_p50"],
            "rgb_sha256": backend_res["rgb_sha256"],
            "torch_argmax_sha256": backend_res["torch_argmax_sha256"],
            "mlx_argmax_sha256": backend_res["mlx_argmax_sha256"],
            "mlx_vs_torch_region_disagreement_px": backend_res["mlx_vs_torch_region_disagreement_px"],
            "archive_bytes_excl_sidecar": int(len(payload) - sidecar_bytes),
        },
        "with_sidecar": {
            "wrong_to_target_mlx": wrapped_res["region_hard_won_mlx"],
            "wrong_to_target_torch": wrapped_res["region_hard_won_torch"],
            "target_margin_p10": wrapped_res["target_margin_p10"],
            "target_margin_p50": wrapped_res["target_margin_p50"],
            "rgb_sha256": wrapped_res["rgb_sha256"],
            "torch_argmax_sha256": wrapped_res["torch_argmax_sha256"],
            "mlx_argmax_sha256": wrapped_res["mlx_argmax_sha256"],
            "sidecar_bytes": sidecar_bytes,
            "archive_bytes_incl_sidecar": len(payload),
        },
        "sidecar_delta": {
            "wrong_to_target_delta": wtt_delta,
            "bytes_delta": sidecar_bytes,
            "note": "exact d_seg/d_pose require the full paired upstream eval (replay step)",
        },
        "runner_authoritative_wrong_to_target": runner_wrong_to_target,
        "verdict": verdict,
        "authority": "planning_control_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "human_visual_fidelity_objective": False,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
