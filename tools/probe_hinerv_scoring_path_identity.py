#!/usr/bin/env python3
"""Localize the scoring-path axis that maps ~11297 backend wins -> the runner's 2.

Two INDEPENDENT re-measurements of the same v9 archive say ~11k backend region
wins (codec grid uint8-MLX = 10781; sidecar probe float-torch = 11297), while the
runner's authoritative parse-back says 2. This probe holds the RENDER FIXED (the
torch archive backend) and varies ONLY the scorer axes, so the first axis whose
count drops ~11k -> ~2 is the divergence (per GPT's scoring-path identity ladder):

  Axis A (input path):  float [0,255]  vs  uint8 camera-res roundtrip (the eval bottleneck)
  Axis D (win def):      absolute argmax==target  vs  wrong->target transition (vs npz pre-birth)

Same SegNet (tac.scorer.extract_gt_masks), same region (GT==target in bbox), same
RGB render -> any count change is attributable to the varied axis alone. Emits
hi_nerv_scoring_path_identity.v1. Planning-control evidence only; non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_NON_AUTHORITY = {
    "authority": "planning_control_false_authority",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "human_visual_fidelity_objective": False,
}


def _read_payload(zp: Path) -> bytes:
    with zipfile.ZipFile(zp) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"0.bin", "x"}] or members
        return zf.read(pick[0])


def _segnet_argmax(frames_chw_0_255: np.ndarray, segnet: Any) -> np.ndarray:
    import torch

    from tac.scorer import extract_gt_masks

    frames = [
        torch.from_numpy(np.transpose(f, (1, 2, 0)).copy()).clamp(0, 255)
        for f in frames_chw_0_255
    ]
    return np.asarray(
        extract_gt_masks(frames, segnet, device="cpu", batch_size=4).cpu(), dtype=np.int64
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--birth-class", type=int, default=4)
    ap.add_argument("--bbox-y0", type=int, default=281)
    ap.add_argument("--bbox-y1", type=int, default=384)
    ap.add_argument("--upstream-dir", default="upstream")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    from tac.scorer import load_default_segnet
    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.archive import parse_archive

    archive = Path(args.archive).expanduser().resolve()
    z = np.load(Path(args.npz).expanduser().resolve())
    target_labels = z["target_labels_bhw"].astype(np.int64)
    candidate_argmax = z["candidate_argmax_bhw"].astype(np.int64)  # pre-birth miner input
    pair_indices = [int(v) for v in z["pair_indices"].tolist()]
    cls = int(args.birth_class)

    payload = _read_payload(archive)
    arc = parse_archive(payload)
    cfg = None
    from tac.substrates.hi_nerv.inflate import build_model_from_archive

    _arc, cfg, _recv = build_model_from_archive(payload, device="cpu")
    backend = HinervSubstrate(cfg).to("cpu").eval()
    fs = dict(arc.decoder_state_dict)
    fs["latents_coarse"] = arc.latents_coarse.to(dtype=backend.latents_coarse.dtype)
    fs["latents_mid"] = arc.latents_mid.to(dtype=backend.latents_mid.dtype)
    fs["latents_fine"] = arc.latents_fine.to(dtype=backend.latents_fine.dtype)
    backend.load_state_dict(fs, strict=True)

    idx = torch.tensor(pair_indices, dtype=torch.long)
    with torch.no_grad():
        _r0, r1 = backend(idx)  # [0,1] CHW
    r1_np = np.asarray(r1.detach().cpu(), dtype=np.float32)  # (N,3,H,W) in [0,1]

    # Axis A: float [0,255] vs uint8 camera-res roundtrip (the eval bottleneck).
    float_255 = r1_np * 255.0
    uint8_255 = np.round(np.clip(float_255, 0.0, 255.0)).astype(np.uint8).astype(np.float32)

    segnet = load_default_segnet(args.upstream_dir, device="cpu")
    argmax_float = _segnet_argmax(float_255, segnet)
    argmax_uint8 = _segnet_argmax(uint8_255, segnet)

    region = np.zeros_like(target_labels, dtype=bool)
    region[:, args.bbox_y0 : args.bbox_y1, :] = True
    region &= target_labels == cls
    pre_wrong = candidate_argmax != cls  # pre-birth (miner input): all region wrong

    def _counts(argmax: np.ndarray) -> dict[str, int]:
        won = (argmax == cls) & region
        return {
            "absolute_argmax_eq_target": int(np.count_nonzero(won)),
            "wrong_to_target_transition": int(np.count_nonzero(won & pre_wrong)),
        }

    cells = []
    for src_name, argmax in (("float_255", argmax_float), ("uint8_roundtrip_255", argmax_uint8)):
        c = _counts(argmax)
        for wd, n in c.items():
            cells.append(
                {
                    "rgb_source": src_name,
                    "scorer": "extract_gt_masks",
                    "region": "gt_class_in_bbox",
                    "win_definition": wd,
                    "count": n,
                }
            )

    float_abs = next(c["count"] for c in cells if c["rgb_source"] == "float_255" and c["win_definition"] == "absolute_argmax_eq_target")
    uint8_abs = next(c["count"] for c in cells if c["rgb_source"] == "uint8_roundtrip_255" and c["win_definition"] == "absolute_argmax_eq_target")
    float_trans = next(c["count"] for c in cells if c["rgb_source"] == "float_255" and c["win_definition"] == "wrong_to_target_transition")

    # First-divergence localization (toward the runner's ~2).
    uint8_drop = float_abs - uint8_abs
    transition_drop = float_abs - float_trans
    if uint8_abs <= 5 and uint8_drop > 0.5 * max(1, float_abs):
        first_axis = "input_path_uint8_eval_roundtrip"
        verdict = "uint8_eval_roundtrip_collapses_birth"
    elif float_trans <= 5 and transition_drop > 0.5 * max(1, float_abs):
        first_axis = "win_definition_transition"
        verdict = "win_definition_explains_runner_2"
    else:
        first_axis = "neither_uint8_nor_transition_in_this_probe"
        verdict = "divergence_is_scorer_or_region_or_render_axis"

    row: dict[str, Any] = {
        "schema": "hi_nerv_scoring_path_identity.v1",
        "family": "hinerv",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "target_class": cls,
        "region_pixel_count": int(np.count_nonzero(region)),
        "rgb_sources": [
            {"name": "float_255", "range": "[0,255] float (no uint8 quant)",
             "sha256": hashlib.sha256(float_255.tobytes()).hexdigest()},
            {"name": "uint8_roundtrip_255", "range": "[0,255] uint8-quantized (eval bottleneck)",
             "sha256": hashlib.sha256(uint8_255.tobytes()).hexdigest()},
        ],
        "counts": cells,
        "float_absolute_wins": float_abs,
        "uint8_absolute_wins": uint8_abs,
        "float_transition_wins": float_trans,
        "uint8_eval_roundtrip_win_drop": uint8_drop,
        "transition_win_drop": transition_drop,
        "runner_authoritative_backend_wins": 2,
        "first_divergence_axis": first_axis,
        "verdict": verdict,
        "note": (
            "render held FIXED (torch archive backend); only the input-path (float vs "
            "uint8) and win-definition vary, so any count change is that axis alone. "
            "If neither collapses to ~2 here, the divergence is the scorer path "
            "(_candidate_logits_np vs extract_gt_masks) or the region mask — the next "
            "matrix axis to add."
        ),
        **_NON_AUTHORITY,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
