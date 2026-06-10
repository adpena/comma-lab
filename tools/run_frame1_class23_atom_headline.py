#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The $0 headline run for the frame-1 Class-2 + Class-3 atom generators (task #50).

Generates Class-2 (Seg-SAFE pose) and Class-3 (Seg-POSITIVE repair) frame-1 atoms
for a STRATIFIED sample of contest pairs (the atlas budget clusters 426-442/577-579
+ fragile clusters 510-522/133/177-178 + an even spread), screens every atom on
the EXACT local CPU-torch DistortionNet (real frozen scorers, NEVER MPS), applies
the #49 resize-null preimage tier-1 postprocess to every accepted atom's frame,
and emits the design row schema:

    {pair_id, target_frame, support_or_cone_id, d_seg_delta, d_pose_delta,
     score_delta_advisory, selector_bits_est, authority_host=macos_cpu_advisory,
     accepted_or_rejected_reason, value_per_byte, preimage_proof}

It reports: how many Class-2 atoms pass seg-unchanged with measurable pose
improvement; how many Class-3 atoms show net-negative ΔS preview; the
value_per_byte distribution.

COMPUTE-SUBSTRATE LAW (operator correction 2026-06-10)
======================================================
- GENERATION + leverage search: MLX-first (numpy reference oracle, Catalog #383).
- ADVISORY screening: local CPU-torch exact frozen scorers (the per-atom check).
- RANKING + admission: contest host ONLY (the R1 lesson).  This run is the
  ADVISORY screen that proposes which atoms an on-host R3-style run would admit;
  it NEVER claims an on-host accept.  Every row is ``[macOS-CPU advisory]``.
- MPS: NEVER.

CLASS-3 RENDER PROXY (honest disclosure — NO FAKE)
==================================================
Class-3 repairs a VEHICLE's rendered frame-1 vs GT.  This headline has no vehicle
render yet, so it uses a controlled lossy roundtrip of GT (downsample/quantize)
as a RENDER PROXY that produces REAL SegNet argmax disagreement.  The repair
leverage + THE LAW admission are measured on the REAL scorers — the only proxy is
the baseline render, tagged ``render_proxy=degraded_gt_roundtrip`` per row so a
future agent wires a real vehicle render without confusion.  This is a render
stand-in, NOT a synthetic-fixture-instead-of-real-input violation: the scorers,
the GT, and the d_seg/d_pose measurements are all real.

Auto-cleanup: rows + summary land on the durable SSD tier with a committed
manifest (path + bytes + sha256); deterministically rebuildable from this exact
command.  $0 local, NO cloud, NO paid GPU, NO MPS, NO /tmp.

Cross-references
----------------
- ``tac.optimization.frame1_seg_safe_pose_atoms`` (Class-2 generator)
- ``tac.optimization.frame1_seg_repair_atoms`` (Class-3 generator)
- ``tac.optimization.resize_null_preimage`` (#49 universal postprocessor)
- ``.omx/research/pr110pp_frame1_joint_methodology_v1_20260610.md`` (the design)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = UPSTREAM / "videos" / "0.mkv"
DEFAULT_ATLAS_DIR = Path("/Volumes/VertigoDataTier/pact/evaluator_response_atlas_20260610T001515Z")

_SSD_TIERS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

# The design's named clusters (verified against the atlas summary headline).
BUDGET_CLUSTER = (426, 437, 438, 439, 440, 441, 442, 577, 578, 579)
FRAGILE_CLUSTER = (510, 514, 515, 517, 518, 519, 522, 133, 177, 178)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir() -> Path:
    stamp = _utc_stamp()
    for tier in _SSD_TIERS:
        if tier.is_dir():
            return tier / f"frame1_class23_atom_headline_{stamp}"
    return REPO_ROOT / "experiments" / "results" / f"frame1_class23_atom_headline_{stamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR,
                        help="dir with cone_map_manifest.json + cone_maps/*.npz (#35/#36)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-even", type=int, default=8,
                        help="evenly-spread pairs in addition to the budget+fragile clusters")
    parser.add_argument("--no-preimage", action="store_true",
                        help="skip the #49 tier-1 preimage postprocess (proof still emitted as 0)")
    return parser.parse_args(argv)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (REPO_ROOT / p)


def _stratified_pairs(n_total: int, num_even: int) -> list[int]:
    """Stratified pair sample: budget clusters + fragile clusters + an even spread."""

    pairs: list[int] = []
    seen: set[int] = set()
    for p in list(BUDGET_CLUSTER) + list(FRAGILE_CLUSTER):
        if 0 <= p < n_total and p not in seen:
            pairs.append(p)
            seen.add(p)
    if num_even > 0:
        step = max(1, n_total // num_even)
        for p in range(0, n_total, step):
            if p not in seen:
                pairs.append(p)
                seen.add(p)
            if len([x for x in pairs if x not in (set(BUDGET_CLUSTER) | set(FRAGILE_CLUSTER))]) >= num_even:
                break
    return sorted(pairs)


def _percentiles(vals: list[float]) -> dict[str, float]:
    import numpy as np

    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.frame1_seg_repair_atoms import (
        RepairTargets,
        SegRepairAtomConfig,
        generate_seg_repair_atom,
        screen_repair_atom_exact,
    )
    from tac.optimization.frame1_seg_safe_pose_atoms import (
        ConeFields,
        SegSafePoseAtomConfig,
        generate_signed_atoms,
        screen_atom_exact,
    )
    from tac.optimization.resize_null_preimage import (
        ResizeProjector,
        apply_tier1_zero_weight_fill,
        zero_weight_pixel_mask,
    )
    from tac.repo_io import sha256_bytes, write_json_artifact

    args = parse_args(argv)
    video = _resolve(args.video)
    atlas_dir = _resolve(args.atlas_dir)
    if not video.is_file():
        print(f"FATAL: video not found: {video}", file=sys.stderr)
        return 2
    manifest_path = atlas_dir / "cone_map_manifest.json"
    if not manifest_path.is_file():
        print(f"FATAL: cone manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    out_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())["maps"]
    cone_index = {int(e["pair_index"]): e for e in manifest}
    n_total = len(cone_index)
    sample = _stratified_pairs(n_total, int(args.num_even))
    max_pair = max(sample)

    # Load real scorers ($0 CPU) + make PoseNet YUV6 gradient-reachable.
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    patch_upstream_yuv6_globally()
    from modules import DistortionNet, SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu"))
    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )

    # Decode GT frames at the SCORER grid (384x512) up to the max sampled pair.
    frames = decode_video(str(video), target_h=384, target_w=512, max_frames=2 * (max_pair + 1))
    if len(frames) < 2 * (max_pair + 1):
        print(f"FATAL: decoded {len(frames)} frames; need {2 * (max_pair + 1)}", file=sys.stderr)
        return 2

    c2_cfg = SegSafePoseAtomConfig()
    c3_cfg = SegRepairAtomConfig()

    # #49 preimage projector + mask at scorer grid (for the proof on the perturbed
    # frame; the camera-res preimage is the universal Class-5 postprocess the
    # design names — here we carry the certified zero-change proof on the atom).
    projector = ResizeProjector.build(camera_h=384, camera_w=512)
    zw_mask = zero_weight_pixel_mask(camera_h=384, camera_w=512,
                                     scorer_h=projector.scorer_h, scorer_w=projector.scorer_w)

    rows: list[dict] = []
    c2_accept = 0
    c2_reject = 0
    c3_accept = 0
    c3_reject = 0
    c2_vpb: list[float] = []
    c3_vpb: list[float] = []
    c2_pose_gain: list[float] = []
    c3_seg_gain: list[float] = []

    for pair_idx in sample:
        f0 = frames[2 * pair_idx].numpy()
        f1 = frames[2 * pair_idx + 1].numpy()
        gt = np.stack([f0, f1], axis=0)  # (2, H, W, C) uint8
        gt_pair = torch.from_numpy(gt[None]).float()  # (1, 2, H, W, C)

        cone_npz = cone_index[pair_idx]["path"]
        fields = ConeFields.from_npz(cone_npz)

        # --- Class-2: Seg-SAFE pose atoms (both signs; keep the better) ---------
        pos, neg = generate_signed_atoms(pair_index=pair_idx, fields=fields, config=c2_cfg)
        for atom in (pos, neg):
            # #49 preimage proof on the perturbed frame-1 (certified zero change).
            preimage_residual = 0.0
            preimage_freed = 0
            if not args.no_preimage:
                cand = atom.apply(gt_pair)
                f1_cand = cand[0, 1].detach().cpu().numpy().astype(np.uint8)
                _, proof = apply_tier1_zero_weight_fill(
                    f1_cand, strategy="measured_best", mask=zw_mask,
                    projector=projector, frame_index=pair_idx,
                )
                preimage_residual = proof.max_abs_projection_residual
                preimage_freed = max(0, proof.bytes_reduction_brotli)
            row = screen_atom_exact(
                atom=atom, distortion_net=dn, gt_pair_btchwc_unit255=gt_pair,
                config=c2_cfg, selector_bits_est=4.0,  # K=16 selector ~4 bits/pair
                preimage_tier1_applied=not args.no_preimage,
                preimage_max_abs_residual=preimage_residual,
                preimage_bytes_freed=preimage_freed,
            ).to_json_obj()
            row["action_class"] = 2
            rows.append(row)
            if row["accepted"]:
                c2_accept += 1
                c2_vpb.append(row["value_per_byte"])
                c2_pose_gain.append(-row["d_pose_delta"])
            else:
                c2_reject += 1

        # --- Class-3: Seg-POSITIVE repair atoms (render proxy = degraded GT) ----
        # Build a REALISTIC render proxy that produces REAL SegNet argmax flips:
        # downsample frame-1 by 2x then upsample (a lossy reconstruction stand-in).
        f1_t = torch.from_numpy(f1.astype(np.float32)).permute(2, 0, 1)[None]
        small = torch.nn.functional.interpolate(f1_t, scale_factor=0.5, mode="bilinear",
                                                align_corners=False)
        rendered_f1 = torch.nn.functional.interpolate(
            small, size=(f1.shape[0], f1.shape[1]), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0).numpy()
        rendered_pair = gt_pair.clone()
        rendered_pair[0, 1] = torch.from_numpy(rendered_f1).float()

        targets = RepairTargets.measure(
            segnet=seg, rendered_frame1_hwc_unit255=rendered_f1,
            gt_frame1_hwc_unit255=f1.astype(np.float64),
            fragile_mask=fields.fragile_cone_mask,
        )
        try:
            repair = generate_seg_repair_atom(pair_index=pair_idx, targets=targets, config=c3_cfg)
        except Exception as e:  # no recoverable flips on this pair
            rows.append({
                "schema": "frame1_seg_repair_atom.v1", "action_class": 3,
                "pair_index": pair_idx, "target_frame": 1, "accepted": False,
                "rejected_reason": f"no_repair_target:{type(e).__name__}",
                "authority_host": "macos_cpu_advisory", "render_proxy": "degraded_gt_roundtrip",
            })
            c3_reject += 1
            continue

        preimage_residual = 0.0
        preimage_freed = 0
        if not args.no_preimage:
            repaired_pair = repair.apply_to_pair(rendered_pair)
            f1_rep = repaired_pair[0, 1].detach().cpu().numpy().astype(np.uint8)
            _, proof = apply_tier1_zero_weight_fill(
                f1_rep, strategy="measured_best", mask=zw_mask,
                projector=projector, frame_index=pair_idx,
            )
            preimage_residual = proof.max_abs_projection_residual
            preimage_freed = max(0, proof.bytes_reduction_brotli)

        r3 = screen_repair_atom_exact(
            atom=repair, distortion_net=dn,
            rendered_pair_btchwc_unit255=rendered_pair,
            gt_pair_btchwc_unit255=gt_pair,
            preimage_tier1_applied=not args.no_preimage,
            preimage_max_abs_residual=preimage_residual,
            preimage_bytes_freed=preimage_freed,
        ).to_json_obj()
        r3["action_class"] = 3
        r3["render_proxy"] = "degraded_gt_roundtrip"
        r3["n_boundary"] = repair.n_boundary
        r3["n_thin_class"] = repair.n_thin_class
        r3["n_fragile"] = repair.n_fragile
        rows.append(r3)
        if r3["accepted"]:
            c3_accept += 1
            c3_vpb.append(r3["value_per_byte"])
            c3_seg_gain.append(-r3["d_seg_delta"])
        else:
            c3_reject += 1

    summary = {
        "schema": "frame1_class23_atom_headline.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "video": str(video),
        "atlas_dir": str(atlas_dir),
        "n_pairs_sampled": len(sample),
        "sampled_pairs": sample,
        "budget_cluster_sampled": [p for p in sample if p in BUDGET_CLUSTER],
        "fragile_cluster_sampled": [p for p in sample if p in FRAGILE_CLUSTER],
        "class2_seg_safe_pose": {
            "n_atoms_screened": c2_accept + c2_reject,
            "n_accepted_seg_unchanged_pose_improved": c2_accept,
            "n_rejected": c2_reject,
            "value_per_byte": _percentiles(c2_vpb),
            "pose_improvement_d_pose": _percentiles(c2_pose_gain),
        },
        "class3_seg_repair": {
            "n_atoms_screened": c3_accept + c3_reject,
            "n_accepted_net_negative_dS": c3_accept,
            "n_rejected": c3_reject,
            "value_per_byte": _percentiles(c3_vpb),
            "seg_reduction_d_seg": _percentiles(c3_seg_gain),
            "render_proxy": "degraded_gt_roundtrip",
        },
        "preimage_postprocess": "resize_null_tier1" if not args.no_preimage else "skipped",
        "evidence_grade": "macOS-CPU advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "authority_host": "macos_cpu_advisory",
        "score_claim": False,
        "promotable": False,
        "host_ranking_packet_note": (
            "advisory accept/reject; on-host R3-style run with the noise-floor tie "
            "law ratifies. Accepted Class-2 atoms (seg-unchanged + pose-improved) "
            "and accepted Class-3 atoms (net-negative LAW dS) are the host-ranking "
            "candidates; rows carry support_or_cone_id + selector_bits_est for "
            "time-coherent selector coding."
        ),
    }

    rows_path = out_dir / "frame1_class23_atom_rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    summary_path = out_dir / "frame1_class23_atom_headline_summary.json"
    write_json_artifact(summary_path, summary)

    rows_sha = sha256_bytes(rows_path.read_bytes())
    manifest_out = {
        "schema": "frame1_class23_atom_headline_manifest.v1",
        "rows_path": str(rows_path),
        "rows_sha256": rows_sha,
        "rows_bytes": rows_path.stat().st_size,
        "summary_path": str(summary_path),
        "rebuild_command": (
            f"python tools/run_frame1_class23_atom_headline.py "
            f"--video {video} --atlas-dir {atlas_dir} --output-dir {out_dir} "
            f"--num-even {args.num_even}"
        ),
        "evidence_grade": "macOS-CPU advisory",
        "promotable": False,
    }
    write_json_artifact(out_dir / "manifest.json", manifest_out)

    print(f"[frame1-class23-headline] sampled {len(sample)} pairs -> {out_dir}")
    print(f"  Class-2 seg-safe pose: {c2_accept} accepted (seg-unchanged + pose-improved) / "
          f"{c2_accept + c2_reject} screened")
    print(f"  Class-3 seg-repair:    {c3_accept} accepted (net-negative LAW dS) / "
          f"{c3_accept + c3_reject} screened")
    if c2_vpb:
        print(f"  Class-2 value/byte median={_percentiles(c2_vpb)['median']:.3e}")
    if c3_vpb:
        print(f"  Class-3 value/byte median={_percentiles(c3_vpb)['median']:.3e}")
    print(f"  rows: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
