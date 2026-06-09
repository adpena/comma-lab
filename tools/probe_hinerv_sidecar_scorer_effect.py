#!/usr/bin/env python3
"""Measure the SHIPPED sidecar's SegNet scorer effect over L (Rule #8).

The runner's authoritative parse-back proved the BACKEND HiNeRV birth is dead
(2 region wins, both ema+live) and the target-region action sidecar overwrites
2286 pixels (program survives) but its SCORER effect is UNMEASURED. This probe
closes that gap on ALREADY-PERSISTED artifacts (no GPU smoke):

  1. render the live archive frame-1 BACKEND (unwrapped HinervSubstrate),
  2. render the SAME archive frame-1 WITH the sidecar (wrap_model_with_target_region_actions),
  3. score both with the authoritative torch SegNet (upstream convention),
  4. count region wins (argmax==birth_class) over the birth region, and over
     L = W_live \\ W_backend, split by inside/outside the sidecar's 2286-px support.

Emits hi_nerv_sidecar_scorer_effect.v1. Planning-control evidence only:
[macOS-CPU advisory] / non-promotable; the authority is exact upstream eval.

Usage:
  .venv/bin/python tools/probe_hinerv_sidecar_scorer_effect.py \
      --archive <live_or_ema archive.zip> \
      --npz <hi_nerv_hard_region_miner_inputs.npz> \
      --birth-class 4 --bbox-y0 281 --bbox-y1 384 \
      --out <dir>/hi_nerv_sidecar_scorer_effect.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

CONTEST_ARCHIVE_RATE_DENOM = 37_545_489
CONTEST_NUM_EVAL_SAMPLES = 600
_NON_AUTHORITY = {
    "authority": "planning_control_false_authority",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "human_visual_fidelity_objective": False,
}


def _read_hiv1_payload(archive_zip: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_zip) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"0.bin", "x"}] or members
        if len(pick) != 1:
            raise SystemExit(f"expected one payload member; got {members}")
        return pick[0], zf.read(pick[0])


def _render_frame1_chw(model: Any, idx_list: list[int]) -> np.ndarray:
    import torch

    idx = torch.tensor(idx_list, dtype=torch.long)
    with torch.no_grad():
        _rgb0, rgb1 = model(idx)
    # The HiNeRV receiver emits frame RGB in [0, 1] CHW (verified: render range
    # 0..1).  extract_gt_masks expects uint8-RANGE [0, 255] HWC, so scale by 255.
    return np.asarray(rgb1.detach().cpu(), dtype=np.float32) * 255.0


def _segnet_argmax_bhw(frame1_chw_0_255: np.ndarray, segnet: Any) -> np.ndarray:
    import torch

    from tac.scorer import extract_gt_masks

    # extract_gt_masks takes a list of (H, W, 3) uint8-range tensors.
    frames = [
        torch.from_numpy(np.transpose(f, (1, 2, 0)).copy()).clamp(0, 255)
        for f in frame1_chw_0_255
    ]
    masks = extract_gt_masks(frames, segnet, device="cpu", batch_size=4)
    return np.asarray(masks.cpu(), dtype=np.int64)  # (N, segH, segW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--birth-class", type=int, default=4)
    ap.add_argument("--bbox-y0", type=int, default=281)
    ap.add_argument("--bbox-y1", type=int, default=384)
    ap.add_argument("--candidate-kind", default="live")
    ap.add_argument("--upstream-dir", default="upstream")
    ap.add_argument(
        "--expected-backend-wins",
        type=int,
        default=None,
        help=(
            "the runner's authoritative backend wrong_to_target for this archive. "
            "If the probe's independently-scored backend does not match, the row is "
            "marked UNVALIDATED and no sidecar verdict is trusted (fail closed)."
        ),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    archive = Path(args.archive).expanduser().resolve()
    z = np.load(Path(args.npz).expanduser().resolve())
    target_labels = z["target_labels_bhw"].astype(np.int64)  # (1, segH, segW) GT argmax
    candidate_argmax = z["candidate_argmax_bhw"].astype(np.int64)  # live candidate argmax
    pair_indices = [int(v) for v in z["pair_indices"].tolist()]
    cls = int(args.birth_class)

    from tac.scorer import load_default_segnet
    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.inflate import build_model_from_archive

    member, payload = _read_hiv1_payload(archive)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    arc = parse_archive(payload)
    meta = dict(arc.meta or {})
    prog_b64 = meta.get("_target_region_actions_v1_b64") or ""
    encoded_program_sha = hashlib.sha256(str(prog_b64).encode("utf-8")).hexdigest()

    # WITH-sidecar receiver (the shipped program): build_model_from_archive wraps it.
    _arc, cfg, with_sidecar = build_model_from_archive(payload, device="cpu")
    # BACKEND (unwrapped): rebuild the inner HinervSubstrate WITHOUT the action wrap.
    from tac.substrates.hi_nerv.architecture import HinervSubstrate

    backend = HinervSubstrate(cfg).to("cpu").eval()
    full_state = dict(arc.decoder_state_dict)
    full_state["latents_coarse"] = arc.latents_coarse.to(dtype=backend.latents_coarse.dtype)
    full_state["latents_mid"] = arc.latents_mid.to(dtype=backend.latents_mid.dtype)
    full_state["latents_fine"] = arc.latents_fine.to(dtype=backend.latents_fine.dtype)
    backend.load_state_dict(full_state, strict=True)

    segnet = load_default_segnet(args.upstream_dir, device="cpu")

    backend_argmax = _segnet_argmax_bhw(_render_frame1_chw(backend, pair_indices), segnet)
    sidecar_argmax = _segnet_argmax_bhw(_render_frame1_chw(with_sidecar, pair_indices), segnet)

    segH, segW = target_labels.shape[1], target_labels.shape[2]
    if backend_argmax.shape[1:] != (segH, segW):
        raise SystemExit(
            f"segnet argmax {backend_argmax.shape[1:]} != labels {(segH, segW)}"
        )

    # Region = GT target_labels==cls within the worst-region bbox (matches the
    # runner's region_pixel_count). originally_wrong = live candidate != cls.
    region = np.zeros_like(target_labels, dtype=bool)
    region[:, args.bbox_y0 : args.bbox_y1, :] = True
    region &= target_labels == cls
    # NOTE: the npz candidate_argmax is the PRE-birth miner input (all region
    # pixels are originally-wrong), so it cannot define W_live (the post-birth
    # live render is not persisted).  The decisive Rule #8 set is therefore the
    # debt set itself: L = region pixels the BACKEND parse-back failed to win.
    orig_wrong = candidate_argmax != cls
    base = region & orig_wrong  # the birth's debt set (== region here)
    w_backend = (backend_argmax == cls) & region  # backend-won region pixels
    w_sidecar = (sidecar_argmax == cls) & region  # sidecar-applied-won region pixels
    l_backend = base & ~w_backend  # debt the backend parse-back lost
    # Sidecar support at SegNet res = pixels whose argmax the sidecar flipped vs
    # the backend (the 2286 RGB overwrite may flip more/fewer argmax pixels).
    sidecar_support = (sidecar_argmax != backend_argmax) & region

    def _cnt(m: np.ndarray) -> int:
        return int(np.count_nonzero(m))

    sidecar_win_on_L = _cnt(w_sidecar & l_backend)
    backend_win_on_L = _cnt(w_backend & l_backend)  # == 0 by construction
    sidecar_win_on_support = _cnt(w_sidecar & sidecar_support & l_backend)
    sidecar_win_outside_support = _cnt(w_sidecar & ~sidecar_support & l_backend)

    # Margin over L for the sidecar render is not available without logits; we
    # report argmax-win counts (scorer authority) here.
    sidecar_archive_bytes = len(str(prog_b64).encode("utf-8"))
    delta_pixels = sidecar_win_on_L - backend_win_on_L
    # Estimated SegNet score gain (advisory; per-pair extrapolated to eval samples).
    est_seg_gain = 100.0 * float(delta_pixels) / (
        float(CONTEST_NUM_EVAL_SAMPLES) * float(segH) * float(segW)
    )
    rate_cost = 25.0 * float(sidecar_archive_bytes) / float(CONTEST_ARCHIVE_RATE_DENOM)
    est_delta_score_total = rate_cost - est_seg_gain

    # NO-FAKE validation gate: this probe's INDEPENDENTLY-scored backend must
    # reproduce the runner's AUTHORITATIVE backend wrong_to_target.  If it does
    # not, our scoring path (render convention / uint8 eval-roundtrip / win
    # definition / region) diverges from the authoritative one, so NO win count
    # from this probe — including the sidecar verdict — can be trusted.
    measured_backend = _cnt(w_backend)
    backend_validation: dict[str, Any] = {
        "expected_backend_wins_authoritative": args.expected_backend_wins,
        "measured_backend_wins_this_probe": measured_backend,
        "backend_scoring_path_matches_authoritative": (
            args.expected_backend_wins is not None
            and abs(measured_backend - int(args.expected_backend_wins)) <= 1
        ),
    }
    validated = backend_validation["backend_scoring_path_matches_authoritative"]
    survived = (
        validated and sidecar_win_on_L > backend_win_on_L and delta_pixels > 0
    )
    row: dict[str, Any] = {
        "schema": "hi_nerv_sidecar_scorer_effect.v1",
        "family": "hinerv",
        "candidate_kind": str(args.candidate_kind),
        "archive_path": archive.as_posix(),
        "archive_sha256": archive_sha,
        "archive_member": member,
        "encoded_program_sha256": encoded_program_sha,
        "birth_class": cls,
        "region_pixel_count": _cnt(region),
        "base_debt_pixel_count": _cnt(base),
        "base_backend_wrong_to_target": _cnt(w_backend),
        "l_backend_size": _cnt(l_backend),
        "sidecar_support_pixels_segres": _cnt(sidecar_support),
        "sidecar_won_count": _cnt(w_sidecar),
        "sidecar_win_on_L": sidecar_win_on_L,
        "sidecar_win_on_support_in_L": sidecar_win_on_support,
        "sidecar_win_outside_support_in_L": sidecar_win_outside_support,
        "sidecar_archive_bytes": sidecar_archive_bytes,
        "delta_pixels_sidecar_vs_backend_on_L": delta_pixels,
        "estimated_seg_score_gain_units": est_seg_gain,
        "archive_rate_cost_units": rate_cost,
        "estimated_delta_score_total_units": est_delta_score_total,
        "estimated_economics_is_advisory": True,
        "parseback_payload_survived": True,
        "parseback_program_survived": True,
        "parseback_scorer_effect_survived": (None if not validated else bool(survived)),
        "scorer_effect_survival_measured": bool(validated),
        "backend_validation": backend_validation,
        "verdict": (
            "UNVALIDATED_backend_scoring_path_mismatch"
            if not validated
            else ("sidecar_rescues_L" if survived else "sidecar_does_not_rescue_L_rule8_fail")
        ),
        **_NON_AUTHORITY,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
