#!/usr/bin/env python3
"""Build the BACKEND-ONLY HiNeRV archive (strip the harmful target-region sidecar).

DEFINITIVE 2026-06-08 finding: the HiNeRV backend birth SURVIVES parse-back
(~11306 region wins) but the bundled target-region action sidecar DESTROYS it
(-> 3 wins) while adding ~8 KB.  Stripping the sidecar is a DOUBLE WIN: it lowers
d_seg (restores ~11303 wins) AND lowers the rate term (-~8 KB).

This tool losslessly strips the sidecar: it parses the archive and RE-PACKS the
SAME decoder/latent tensors with the SAME codecs but NO action program.  The
re-quant is IDEMPOTENT (already-quantized values round-trip exactly), so the
backend render is unchanged (verified by region-win re-measurement when a scorer
is available).  Emits hi_nerv_backend_only_exact_replay.v1 with the EXACT zip
byte delta + the region-win proxy + the exact upstream eval command for d_seg/
d_pose.  Region wins are the strong proxy; promotion requires the paired eval.

Usage:
  .venv/bin/python tools/build_hi_nerv_backend_only_archive.py \
      --archive <in>/archive.zip --out-archive <out>/archive_backend_only.zip \
      --out-row <out>/hi_nerv_backend_only_exact_replay.json \
      [--npz <hard_region_inputs.npz> --runner-row <parseback_row.json> --upstream-dir upstream]
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


def _read_member(zp: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(zp) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        pick = [n for n in members if n in {"0.bin", "x"}] or members
        if len(pick) != 1:
            raise SystemExit(f"expected one payload member; got {members}")
        return pick[0], zf.read(pick[0])


def _write_zip(out_zip: Path, member_name: str, payload: bytes, extra: dict[str, bytes]) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, payload)
        for name, data in extra.items():
            zf.writestr(name, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out-archive", required=True)
    ap.add_argument("--out-row", required=True)
    ap.add_argument("--decoder-codec", default="int8_mixed")
    ap.add_argument("--latent-codec", default="int8_brotli_q11")
    ap.add_argument("--npz", default=None, help="optional hard-region npz for region-win re-measure")
    ap.add_argument("--runner-row", default=None, help="optional parseback row for worst_region")
    ap.add_argument("--upstream-dir", default="upstream")
    args = ap.parse_args()

    import torch

    from tac.substrates.hi_nerv.archive import parse_archive
    from tac.substrates.hi_nerv.archive_candidate import pack_archive_from_exported_state_dict
    from tac.substrates.hi_nerv.inflate import build_model_from_archive
    from tac.substrates.hi_nerv.target_region_actions import TARGET_REGION_ACTION_META_KEY

    in_zip = Path(args.archive).expanduser().resolve()
    member, payload = _read_member(in_zip)
    arc = parse_archive(payload)
    had_action = TARGET_REGION_ACTION_META_KEY in dict(arc.meta or {})
    _arc, cfg, _recv = build_model_from_archive(payload, device="cpu")

    # Lossless strip: re-pack the SAME tensors + codecs with NO action program.
    exported: dict[str, np.ndarray] = {
        k: np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, dtype=np.float32)
        for k, v in arc.decoder_state_dict.items()
    }
    exported["latents_coarse"] = np.asarray(arc.latents_coarse.detach().cpu(), np.float32)
    exported["latents_mid"] = np.asarray(arc.latents_mid.detach().cpu(), np.float32)
    exported["latents_fine"] = np.asarray(arc.latents_fine.detach().cpu(), np.float32)
    blob_bo = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
        decoder_codec=str(args.decoder_codec),
        latent_codec=str(args.latent_codec),
        target_region_action_program_base64=None,
    )
    if isinstance(blob_bo, tuple):
        blob_bo = blob_bo[0]
    arc_bo = parse_archive(blob_bo)
    bo_has_action = TARGET_REGION_ACTION_META_KEY in dict(arc_bo.meta or {})
    if bo_has_action:
        raise SystemExit("backend-only re-pack still has the action meta — strip failed")

    # Carry any sibling runtime members (inflate.py/.sh/README) verbatim.
    extra: dict[str, bytes] = {}
    with zipfile.ZipFile(in_zip) as zf:
        for n in zf.namelist():
            if n.endswith("/") or n == member:
                continue
            extra[n] = zf.read(n)
    out_zip = Path(args.out_archive).expanduser().resolve()
    _write_zip(out_zip, member, blob_bo, extra)

    orig_zip_bytes = int(in_zip.stat().st_size)
    bo_zip_bytes = int(out_zip.stat().st_size)
    payload_delta = len(blob_bo) - len(payload)
    zip_delta = bo_zip_bytes - orig_zip_bytes
    rate_delta_score = 25.0 * float(zip_delta) / float(CONTEST_ARCHIVE_RATE_DENOM)

    # Optional region-win re-measure (the d_seg proxy) when npz + scorer present.
    region_win_orig = None
    region_win_bo = None
    est_seg_delta_score = None
    if args.npz and args.runner_row:
        import mlx.core as mx

        from tac.local_acceleration.mlx_scorer_adapters import MLXSegNetAdapter
        from tac.scorer import load_default_segnet
        from tac.substrates.hi_nerv.birth_survival import reconstruct_birth_region_mask
        from tac.substrates.hi_nerv.target_region_birth import region_margin_stats

        z = np.load(Path(args.npz).expanduser().resolve())
        tl = z["target_labels_bhw"].astype(np.int64)
        rr = json.loads(Path(args.runner_row).expanduser().read_text())
        worst = rr["worst_region"]
        cls = int(worst["class_index"])
        region, _ = reconstruct_birth_region_mask(tl, worst)
        region = np.asarray(region)
        rmask = (region[None, ...] if region.ndim == 2 else region).astype(np.float32)
        segnet = load_default_segnet(args.upstream_dir, device="cpu")
        adapter = MLXSegNetAdapter(segnet)

        def _wins(blob: bytes) -> int:
            _a, _c, recv = build_model_from_archive(blob, device="cpu")
            with torch.no_grad():
                _r0, r1 = recv(torch.tensor([0]))
            r1np = np.asarray(r1.detach().cpu(), np.float32)
            x = mx.array(np.transpose(r1np, (0, 2, 3, 1)).astype(np.float32)) * 255.0
            logits = np.asarray(adapter(x).astype(mx.float32))
            return int(region_margin_stats(logits, rmask, cls)["region_hard_won_pixels"])

        region_win_orig = _wins(payload)
        region_win_bo = _wins(blob_bo)
        # ADVISORY d_seg proxy: per-pair region wins extrapolated to eval samples.
        h, w = int(tl.shape[1]), int(tl.shape[2])
        est_seg_delta_score = -100.0 * float(region_win_bo - region_win_orig) / (
            float(CONTEST_NUM_EVAL_SAMPLES) * float(h) * float(w)
        )

    est_delta_score_total = rate_delta_score + (est_seg_delta_score or 0.0)
    row: dict[str, Any] = {
        "schema": "hi_nerv_backend_only_exact_replay.v1",
        "family": "hinerv",
        "input_archive": in_zip.as_posix(),
        "input_archive_sha256": hashlib.sha256(in_zip.read_bytes()).hexdigest(),
        "input_had_target_region_action": had_action,
        "backend_only_archive": out_zip.as_posix(),
        "backend_only_archive_sha256": hashlib.sha256(out_zip.read_bytes()).hexdigest(),
        "payload_bytes_original": len(payload),
        "payload_bytes_backend_only": len(blob_bo),
        "payload_bytes_delta": payload_delta,
        "zip_bytes_original": orig_zip_bytes,
        "zip_bytes_backend_only": bo_zip_bytes,
        "zip_bytes_delta": zip_delta,
        "exact_rate_delta_score": rate_delta_score,
        "region_win_original": region_win_orig,
        "region_win_backend_only": region_win_bo,
        "region_win_delta": (
            None if region_win_bo is None else region_win_bo - region_win_orig
        ),
        "estimated_seg_delta_score_advisory": est_seg_delta_score,
        "estimated_delta_score_total_advisory": est_delta_score_total,
        "exact_eval_command": (
            f"inflate.sh {out_zip.as_posix()} <outdir> <file_list> && "
            "upstream/evaluate.py --device cpu  (and --device cuda) for exact d_seg/d_pose"
        ),
        "verdict": (
            "backend_only_double_win_strip_sidecar"
            if zip_delta < 0 and (region_win_bo is None or region_win_bo >= (region_win_orig or 0))
            else "needs_exact_eval"
        ),
        "note": (
            "exact_rate_delta_score is EXACT (measured zip bytes). seg delta is an "
            "ADVISORY per-pair proxy; promotion requires paired upstream d_seg/d_pose."
        ),
        "authority": "planning_control_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": True,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "human_visual_fidelity_objective": False,
    }
    out_row = Path(args.out_row).expanduser().resolve()
    out_row.parent.mkdir(parents=True, exist_ok=True)
    out_row.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
