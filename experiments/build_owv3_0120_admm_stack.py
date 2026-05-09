#!/usr/bin/env python3
# ADMM_WAIVED:B4-reviewed historical/planning naming; docstrings or delegated coordinator code clarify whether this is Lagrangian, bridge, or actual iterative ADMM.
"""Lane Joint-ADMM cross-stream coordinator over the owv3_0120 deploy archive.

Champion archive (1.0024 [contest-CUDA RTX 4090]):
    archive_lane_g_v3_owv3_0120_LANDED.zip = {
        renderer.bin       (OWV3, bbr=0.66, protect=0.002, 211_903 B),
        masks.mkv          (AV1 CRF=50, 421_483 B),
        optimized_poses.pt (fp32 (600,6), 15_620 B),
    }
    total = 617_410 B; pose 0.003564 / seg 0.004025 / rate 0.01644

Joint-ADMM analyses three streams as separate proximal codecs and the coordinator
finds the cross-stream byte allocation that minimises the joint score under a
total byte budget. The coordinator does NOT load scorers — every per-stream
``score_at_bytes`` callback is a CACHED frontier built before the ADMM loop.

Per memory ``project_codec_stacking_composition_canonical_orders_20260429``:
    canonical order is representation -> prediction -> quantization -> hyperprior
    -> arithmetic -> pack; ADMM is the QUANTIZATION-stage cross-stream coordinator.

CACHED FRONTIERS used (all empirical [contest-CUDA RTX 4090]):

    Renderer (OWV3, 6 R(D) samples + 1 cliff sample):
        bbr=0.700, p=0.002 -> bytes=624_996, score=1.0088
        bbr=0.685, p=0.002 -> bytes=622_407, score=1.0061
        bbr=0.680, p=0.002 -> bytes=621_914, score=1.0100
        bbr=0.660, p=0.0018 -> bytes=618_443, score=1.0027
        bbr=0.660, p=0.002 -> bytes=617_410, score=1.0024  [CHAMPION]
        bbr=0.600, p=0.05  -> bytes=590_176, score=1.1120  [CLIFF]
        bbr=0.550, p=0.03  -> bytes=578_273, score=1.0593

    Pose (locked at fp32 (600,6) = 15_620 B):
        pose_delta_codec at delta_bits=8 fails roundtrip (max-abs 0.543
        > tol 5e-2) on Lane G v3 trajectories -- the codec REJECTS the
        encoding via RuntimeError. The pose stream therefore has a
        single-point frontier at 15_620 B.

    Masks (AV1 grayscale; sweep over CRF):
        Re-encode the decoded label tensor at CRF in {50, 53, 56, 59, 62}
        and report bytes. Score-cost surface is [prediction] anchored on
        the empirical CRF=50 score (0.004025 SegNet) + memory's
        ``masks dominate'' tag (60KB savings -> 0.04 score regression).

ADMM convergence target: dScore/dByte equilibration across UNSATURATED active
streams (KKT waterline; see ``tac.joint_admm_coordinator.kkt_waterline_residual``).

CLAUDE.md compliance
--------------------
* COMPRESS-time only. No scorer load anywhere.
* No silent defaults: every per-stream config is explicit; every score-cost
  surface is a CACHED measurement or a [prediction]-tagged surface from
  memory.
* Strict-scorer-rule compliant: ADMM never loads SegNet/PoseNet.
* Tagged claims: ``predicted_score_after_admm`` is [prediction] until
  contest-CUDA eval lands.
* Deterministic ZIP: timestamp pinned to (1980, 1, 1, 0, 0, 0); double-build
  SHA verified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CHAMPION_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_owv3_wave3_LANDED_20260501"
    / "archive_lane_g_v3_owv3_0120_LANDED.zip"
)
# [empirical: experiments/results/lane_g_v3_owv3_wave3_LANDED_20260501/ contest-CUDA RTX 4090 dispatch 35958897]
CHAMPION_SCORE = 1.0023898347421667
CHAMPION_BYTES = 617_410  # [empirical: source same as CHAMPION_SCORE]
CHAMPION_POSE = 0.0035645  # [empirical: source same as CHAMPION_SCORE]
CHAMPION_SEG = 0.00402483  # [empirical: source same as CHAMPION_SCORE]
CHAMPION_RATE_UNSCALED = 0.01644432  # [empirical: source same as CHAMPION_SCORE]

LANE_G_V3_DIR = REPO_ROOT / "experiments" / "results" / "lane_g_v3_landed"
LANE_G_V3_RENDERER = LANE_G_V3_DIR / "iter_0" / "renderer.bin"
LANE_G_V3_POSES = LANE_G_V3_DIR / "iter_0" / "optimized_poses.pt"
LANE_G_V3_MASKS = LANE_G_V3_DIR / "iter_0" / "masks.mkv"

SENSITIVITY_MAP = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_owv3_fisher_beta_20260501_LANDED"
    / "lane_g_v3_owv3_fisher_stack_results"
    / "owv3_sensitivity_map.pt"
)

# Renderer R(D) frontier from wave3 + wave4 [contest-CUDA RTX 4090].
# Each entry: (bbr, protect, archive_bytes, score, pose, seg).
# These are WHOLE-ARCHIVE bytes -- the per-stream renderer byte share is
# derived as archive_bytes - 421_483 (masks) - 15_620 (poses) - 350 (zip
# overhead). Sorted ascending by archive_bytes.
RENDERER_RD_FRONTIER: list[tuple[float, float, int, float, float, float]] = [
    # CLIFF below bbr=0.66 -- reported for diagnostic only; ADMM should
    # avoid these (very high marginal pose cost).
    (0.55, 0.05, 577_849, 1.0689, 0.007290, 0.004142),
    (0.55, 0.03, 578_273, 1.0593, 0.006943, 0.004108),
    (0.60, 0.05, 590_176, 1.1120, 0.010034, 0.004023),
    # Champion neighbourhood (the safe operating band).
    (0.66, 0.002, 617_410, 1.0024, 0.003564, 0.004025),  # CHAMPION
    (0.66, 0.0018, 618_443, 1.0027, 0.003576, 0.004018),
    (0.68, 0.002, 621_914, 1.0100, 0.003788, 0.004013),
    (0.685, 0.002, 622_407, 1.0061, 0.003595, 0.004021),
    (0.70, 0.002, 624_996, 1.0088, 0.003637, 0.004020),
]

# Per-stream COMPRESSED byte ledger from champion zip inspection:
#   renderer.bin -> 190_730 compressed (raw 211_903)
#   masks.mkv    -> 412_169 compressed (raw 421_483)
#   optimized_poses.pt -> 14_183 compressed (raw 15_620)
#   total compressed = 617_082; archive total = 617_410 (incl ~328B headers)
CHAMPION_ARCHIVE_BREAKDOWN = {
    "renderer_bin_raw": 211_903,
    "renderer_bin_compressed": 190_730,
    "masks_mkv_raw": 421_483,
    "masks_mkv_compressed": 412_169,
    "poses_pt_raw": 15_620,
    "poses_pt_compressed": 14_183,
    "zip_overhead": 328,
    "archive_total": 617_410,
}

# Mask R(D) frontier: bytes sampled empirically by re-encoding existing labels
# at CRF in {50, 53, 56, 59, 62}; score-cost predicted from canonical-order
# memory (60KB savings -> 0.04 score regression at SegNet boundary).
# The re-encode pass runs locally on CPU in <1 minute total.
MASK_CRF_GRID = (50, 53, 56, 59, 62)


def _sha256(data: bytes | Path) -> str:
    h = hashlib.sha256()
    if isinstance(data, Path):
        with data.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
    else:
        h.update(data)
    return h.hexdigest()


def _zinfo(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o644 & 0xFFFF) << 16
    return info


def _archive_bytes(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in members:
            z.writestr(_zinfo(name), data)
    return buf.getvalue()


def _extract_member(archive_path: Path, name: str) -> bytes:
    with zipfile.ZipFile(archive_path, "r") as z:
        return z.read(name)


# ─────────────────────────────────────────────────────────────────────────────
# Per-stream cached frontiers
# ─────────────────────────────────────────────────────────────────────────────


def build_renderer_frontier() -> list[tuple[int, float]]:
    """Return [(renderer_compressed_bytes, score_cost)] sorted by bytes ascending.

    score_cost is the renderer-specific contribution to the joint score,
    isolated by subtracting the (constant across wave3) mask + pose terms
    from the empirical whole-archive score:

        score = pose_term + seg_term + rate_term
        renderer_score_cost = score(s) - pose_term_champion - seg_term_champion

    The byte axis is the COMPRESSED renderer.bin size in the deterministic
    zip, derived as `archive_bytes - mask_compressed - pose_compressed -
    zip_overhead`. This is the byte axis ADMM should equilibrate against
    the other streams' compressed byte axes.
    """
    pose_term_champion = 0.18879883
    seg_term_champion = 0.40248300
    other_streams_compressed = (
        CHAMPION_ARCHIVE_BREAKDOWN["masks_mkv_compressed"]
        + CHAMPION_ARCHIVE_BREAKDOWN["poses_pt_compressed"]
        + CHAMPION_ARCHIVE_BREAKDOWN["zip_overhead"]
    )
    samples = []
    for bbr, protect, archive_bytes, score, pose, seg in RENDERER_RD_FRONTIER:
        # Per-stream renderer compressed bytes = archive total minus the
        # constant mask + pose + zip-overhead contribution. Wave3 only varied
        # the renderer; mask + pose + overhead are bit-identical across rows.
        renderer_compressed = int(archive_bytes) - other_streams_compressed
        renderer_cost = float(score) - pose_term_champion - seg_term_champion
        samples.append((renderer_compressed, float(renderer_cost)))
    samples.sort(key=lambda s: s[0])
    return samples


def build_pose_frontier() -> list[tuple[int, float]]:
    """Pose stream: locked at 14_183 B compressed (raw 15_620, fp32 (600,6)).

    pose_delta_codec at delta_bits=8 fails roundtrip on Lane G v3 trajectories
    (max-abs error 0.543 > tol 5e-2). PD-V2 (varint per-channel) on this
    trajectory similarly cannot meet the codec's strict roundtrip gate.
    Single-point frontier at the champion compressed size.
    """
    # Use COMPRESSED bytes (matches the other streams' axis).
    pose_compressed = CHAMPION_ARCHIVE_BREAKDOWN["poses_pt_compressed"]
    # Score-cost = pose-distortion contribution at the current bytes.
    # √(10 * pose_dist) where pose_dist = 0.003564.
    pose_term_at_locked = float(np.sqrt(10.0 * CHAMPION_POSE))
    return [(pose_compressed, pose_term_at_locked)]


def sample_mask_frontier(
    crf_grid: tuple[int, ...],
    out_dir: Path,
    seg_score_per_kb_savings: float = 4e-4,  # [prediction] from memory
) -> list[tuple[int, float, int, str]]:
    """Re-encode existing masks.mkv labels at each CRF; return frontier.

    Returns list of (bytes_used, predicted_score_cost, crf, sha256) sorted by bytes.

    Score-cost is PREDICTED from the canonical-order memory tag:
        memory project_codec_stacking_composition_canonical_orders_20260429:
        "MASKS dominate (45x pose headroom; 60KB=0.04, pose cap ~5KB=-0.003)"
        => coefficient seg_score_per_kb_savings ~ 0.04/60 ~ 6.7e-4 score per KB
        Conservative: use 4e-4 (60% of memory midpoint to avoid over-promising).

    The CHAMPION mask at CRF=50 anchors the frontier with EMPIRICAL cost
    (seg_term = 100 * 0.00402483 = 0.40248); other CRF entries are tagged
    [prediction] until contest-CUDA verifies.
    """
    from tac.mask_codec import decode_masks, encode_masks

    out_dir.mkdir(parents=True, exist_ok=True)

    masks = decode_masks(str(LANE_G_V3_MASKS))
    print(f"[mask-frontier] decoded {tuple(masks.shape)} from masks.mkv")

    seg_term_anchor = 100.0 * CHAMPION_SEG  # 0.40248 -- CRF=50 empirical

    # The CHAMPION mask point uses the EXISTING (Lane G v3) masks.mkv: 421_483
    # raw / 412_169 compressed. Re-encoding even at CRF=50 introduces 0.14% pixel
    # disagreement (verified empirically) which WILL alter SegNet output. So
    # the champion sample uses the existing file's bytes + empirical seg cost;
    # any re-encoded sample (including CRF=50 re-encode) gets a [prediction]
    # tag because the relabel drift cost is not measured until contest-CUDA.
    samples: list[tuple[int, float, int, str]] = [
        (
            CHAMPION_ARCHIVE_BREAKDOWN["masks_mkv_compressed"],
            seg_term_anchor,
            50,  # original CRF
            _sha256(LANE_G_V3_MASKS) + "::champion",
        )
    ]

    baseline_size = LANE_G_V3_MASKS.stat().st_size
    for crf in crf_grid:
        out_path = out_dir / f"masks_crf{crf}.mkv"
        size_raw = encode_masks(masks, str(out_path), crf=crf, fps=20)
        # Estimate compressed-in-zip size: AV1 streams don't compress further
        # under deflate (they're already entropy-coded). We model
        # compressed = raw * 412169/421483 = raw * 0.97791 (the empirical
        # AV1-in-zip compression ratio on the champion mask).
        compressed = int(round(size_raw * 412169 / 421483))
        sha = _sha256(out_path)
        # Score cost: anchor at CRF=50, regress as CRF rises (saves bytes).
        delta_kb = (baseline_size - size_raw) / 1024.0
        if delta_kb >= 0:
            predicted_cost = seg_term_anchor + delta_kb * seg_score_per_kb_savings
        else:
            predicted_cost = seg_term_anchor  # we never spend MORE bytes
        # Add a small "relabel-drift" tax to the CRF=50 re-encode point too,
        # since the empirical 0.14% pixel disagreement WILL alter SegNet
        # output in some unknown direction. Conservative: 0.001 score penalty
        # for the crf=50 relabel; CRF>50 already absorbed by the per-kb model.
        if crf == 50:
            predicted_cost += 0.001  # [prediction] relabel-drift tax
        samples.append((compressed, float(predicted_cost), int(crf), sha))
        tag = "[empirical]" if crf == 50 else "[prediction]"
        print(
            f"[mask-frontier] crf={crf} -> raw={size_raw:,}B compressed={compressed:,}B "
            f"(delta={baseline_size-size_raw:+,}B) predicted_cost={predicted_cost:.5f} "
            f"{tag} (re-encoded -- relabel-drift cost not measured)"
        )
    samples.sort(key=lambda s: s[0])
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Stream wrappers (StreamProximalCodec)
# ─────────────────────────────────────────────────────────────────────────────


class CachedFrontierStream:
    """StreamProximalCodec wrapper around a precomputed (bytes, score) list.

    The proximal step picks the largest sample with bytes <= target_bytes
    (the standard discrete proximal selection). Marginal is finite-differenced
    against the next-larger sample.
    """

    def __init__(
        self, name: str, frontier: list[tuple[int, float]]
    ) -> None:
        if not frontier:
            raise ValueError(f"{name}: empty frontier not allowed")
        self._name = name
        self._frontier = sorted(frontier, key=lambda s: s[0])

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(self, target_bytes: float, dual: float):
        from tac.joint_admm_coordinator import ProximalStepResult

        # Pick largest bytes <= target_bytes (or smallest if target undershoots).
        selected_idx = 0
        for i, (b, _) in enumerate(self._frontier):
            if b <= target_bytes:
                selected_idx = i
            else:
                break
        b_sel, s_sel = self._frontier[selected_idx]
        if selected_idx + 1 < len(self._frontier):
            b_nxt, s_nxt = self._frontier[selected_idx + 1]
            db = float(b_nxt - b_sel)
            ds = float(s_sel - s_nxt)  # +ve when more bytes lowers score
            marginal = (ds / db) if db > 0 else 0.0
        else:
            marginal = 0.0
        return ProximalStepResult(
            encoded_bytes=int(b_sel),
            score_delta=float(s_sel),
            marginal=float(marginal),
            state=(selected_idx, b_sel, s_sel),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Composer
# ─────────────────────────────────────────────────────────────────────────────


def compose_archive(
    renderer_bytes: bytes,
    mask_path: Path,
    poses_bytes: bytes,
    output: Path,
) -> tuple[int, str, bool]:
    members = [
        ("renderer.bin", renderer_bytes),
        ("masks.mkv", mask_path.read_bytes()),
        ("optimized_poses.pt", poses_bytes),
    ]
    archive_data = _archive_bytes(members)
    archive_rebuild = _archive_bytes(members)
    deterministic = archive_data == archive_rebuild
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(archive_data)
    return output.stat().st_size, _sha256(output), deterministic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "results" / "lane_owv3_0120_admm_stack_20260501",
    )
    parser.add_argument(
        "--byte-budget",
        type=int,
        default=CHAMPION_BYTES,
        help=(
            "Joint byte budget for ADMM. Default = champion bytes. Lower "
            "values force the coordinator to pick a sub-champion allocation."
        ),
    )
    parser.add_argument(
        "--seg-score-per-kb-savings",
        type=float,
        default=4e-4,
        help=(
            "Predicted SegNet score regression per KB of mask byte savings. "
            "Default 4e-4 is conservative (60%% of memory midpoint 6.7e-4)."
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="ADMM iteration cap (default Boyd §3.4 guidance).",
    )
    parser.add_argument(
        "--rho-init",
        type=float,
        default=1.0,
        help="ADMM initial penalty (Boyd default 1.0; Q4B refines on iter 1).",
    )
    parser.add_argument(
        "--skip-mask-resample",
        action="store_true",
        help="Skip mask CRF re-encode pass (use existing masks.mkv at CRF 50 only).",
    )
    parser.add_argument(
        "--no-build-archive",
        action="store_true",
        help="Run ADMM only; skip composing the candidate archive.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[admm-stack {time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        log_lines.append(line)

    log("=== Joint-ADMM cross-stream coordinator over owv3_0120 ===")
    log(f"champion archive: {CHAMPION_ARCHIVE}")
    log(f"champion score:   {CHAMPION_SCORE:.6f}")
    log(f"champion bytes:   {CHAMPION_BYTES:,}")
    log(f"byte budget:      {args.byte_budget:,}")

    # Stage 1: anchor verification
    log("Stage 1: verify champion archive + extract members")
    if not CHAMPION_ARCHIVE.exists():
        raise FileNotFoundError(CHAMPION_ARCHIVE)
    champ_renderer = _extract_member(CHAMPION_ARCHIVE, "renderer.bin")
    champ_masks = _extract_member(CHAMPION_ARCHIVE, "masks.mkv")
    champ_poses = _extract_member(CHAMPION_ARCHIVE, "optimized_poses.pt")
    log(f"  renderer.bin: {len(champ_renderer):,}B sha={_sha256(champ_renderer)[:12]}")
    log(f"  masks.mkv:    {len(champ_masks):,}B sha={_sha256(champ_masks)[:12]}")
    log(f"  optimized_poses.pt: {len(champ_poses):,}B sha={_sha256(champ_poses)[:12]}")

    # Stage 2: build cached frontiers
    log("Stage 2: build cached per-stream R(D) frontiers")
    renderer_frontier = build_renderer_frontier()
    log(f"  renderer frontier: {len(renderer_frontier)} samples")
    for b, s in renderer_frontier:
        log(f"    bytes={b:,} renderer_cost={s:.5f}")

    pose_frontier = build_pose_frontier()
    log(f"  pose frontier:     {len(pose_frontier)} samples (locked)")
    for b, s in pose_frontier:
        log(f"    bytes={b:,} pose_term={s:.5f}")

    if args.skip_mask_resample:
        # Use only the empirical CRF=50 anchor.
        mask_frontier_full = [
            (
                LANE_G_V3_MASKS.stat().st_size,
                100.0 * CHAMPION_SEG,
                50,
                _sha256(LANE_G_V3_MASKS),
            )
        ]
        log("  mask frontier:     skipped re-encode (--skip-mask-resample)")
    else:
        mask_frontier_full = sample_mask_frontier(
            MASK_CRF_GRID,
            args.output_dir / "mask_resample",
            seg_score_per_kb_savings=args.seg_score_per_kb_savings,
        )
    mask_frontier = [(b, c) for (b, c, _crf, _sha) in mask_frontier_full]
    mask_meta = {
        b: {"crf": crf, "sha256": sha, "predicted_cost": c}
        for (b, c, crf, sha) in mask_frontier_full
    }
    log(f"  mask frontier:     {len(mask_frontier)} samples")
    for b, c in mask_frontier:
        log(f"    bytes={b:,} predicted_seg_cost={c:.5f}")

    # Stage 3: build streams + run ADMM
    log("Stage 3: instantiate streams + run ADMM coordinator")
    from tac.joint_admm_coordinator import (
        JointADMMConfig,
        run_admm,
    )

    streams = [
        CachedFrontierStream("renderer", renderer_frontier),
        CachedFrontierStream("masks", mask_frontier),
        CachedFrontierStream("poses", pose_frontier),
    ]

    cfg = JointADMMConfig(
        byte_budget=float(args.byte_budget),
        max_iters=int(args.max_iters),
        rho_init=float(args.rho_init),
        verbose=False,
    )
    t0 = time.monotonic()
    result = run_admm(streams, cfg)
    admm_elapsed = time.monotonic() - t0
    log(
        f"  ADMM done in {admm_elapsed:.2f}s "
        f"(iters={result.iters}, restarts={result.restarts}, "
        f"converged={result.converged})"
    )
    log(f"  KKT waterline residual: {result.waterline_kkt_residual:.6e}")
    log(f"  KKT waterline satisfied: {result.waterline_satisfied}")
    log("  final per-stream allocation:")
    stream_names = [s.name for s in streams]
    for name, b, sc, m in zip(
        stream_names,
        result.final_bytes_per_stream,
        result.final_score_per_stream,
        result.final_marginal_per_stream,
    ):
        log(f"    {name:>10}: bytes={b:>9,.0f} score_cost={sc:.5f} marginal={m:.4e}")

    final_alloc = {
        name: int(round(b))
        for name, b in zip(stream_names, result.final_bytes_per_stream)
    }
    final_total_bytes = sum(final_alloc.values())
    log(f"  total bytes allocated: {final_total_bytes:,}")

    # Stage 4: derive predicted score for the ADMM allocation
    # score = pose_term + seg_term + rate_term
    # pose_term ~ sqrt(10 * pose_dist); we hold pose at champion since the
    # codec is locked.
    # seg_term comes from the mask frontier predicted cost.
    # The renderer frontier carries (score - pose_term_champ - seg_term_champ),
    # so the ADMM-allocated renderer score_cost is the *renderer* contribution
    # at its operating point; we recombine:
    pose_term_champ = float(np.sqrt(10.0 * CHAMPION_POSE))
    seg_term_champ = 100.0 * CHAMPION_SEG

    renderer_score_cost = next(
        sc
        for n, sc in zip(stream_names, result.final_score_per_stream)
        if n == "renderer"
    )
    mask_score_cost = next(
        sc
        for n, sc in zip(stream_names, result.final_score_per_stream)
        if n == "masks"
    )
    # Re-add the pose+seg champion terms that were subtracted at frontier
    # construction; the renderer R(D) was empirically measured WITH champion
    # masks+poses, so renderer_cost + mask_term + pose_term reconstructs.
    # Use ADMM-picked mask cost (predicted) for seg_term; champion pose_term.
    predicted_score = renderer_score_cost + mask_score_cost + pose_term_champ
    log(
        f"  predicted joint score = renderer({renderer_score_cost:.5f}) + "
        f"masks({mask_score_cost:.5f}) + pose({pose_term_champ:.5f}) "
        f"= {predicted_score:.5f} [prediction]"
    )
    log(
        f"  vs champion {CHAMPION_SCORE:.6f}: "
        f"delta={predicted_score - CHAMPION_SCORE:+.5f}"
    )

    # Stage 5: assemble candidate archive (only if ADMM picked a non-champion
    # allocation; if the renderer + mask pick equals the champion config, the
    # archive is BIT-IDENTICAL to the champion and we skip the build).
    renderer_bytes_picked = final_alloc.get("renderer")
    mask_bytes_picked = final_alloc.get("masks")
    archive_path = args.output_dir / "archive.zip"

    champion_renderer_compressed = CHAMPION_ARCHIVE_BREAKDOWN["renderer_bin_compressed"]
    champion_mask_compressed = CHAMPION_ARCHIVE_BREAKDOWN["masks_mkv_compressed"]
    if (
        renderer_bytes_picked == champion_renderer_compressed
        and mask_bytes_picked == champion_mask_compressed
    ):
        log("  ADMM converged to CHAMPION allocation -- archive bit-identical")
        archive_size = CHAMPION_ARCHIVE.stat().st_size
        archive_sha = _sha256(CHAMPION_ARCHIVE)
        deterministic = True
        archive_path = CHAMPION_ARCHIVE  # point at the champion
    elif args.no_build_archive:
        log("  --no-build-archive -- skipping archive composition")
        archive_size = 0
        archive_sha = "skipped"
        deterministic = False
    else:
        log("Stage 5: assemble candidate archive")
        # The renderer R(D) frontier byte axis is COMPRESSED bytes; map back
        # to (bbr, protect) by inverting the per-stream subtraction.
        other_streams_compressed = (
            champion_mask_compressed
            + CHAMPION_ARCHIVE_BREAKDOWN["poses_pt_compressed"]
            + CHAMPION_ARCHIVE_BREAKDOWN["zip_overhead"]
        )
        target_archive_bytes = renderer_bytes_picked + other_streams_compressed
        renderer_cfg = None
        for bbr, protect, archive_b, sc, p, sg in RENDERER_RD_FRONTIER:
            if abs(int(archive_b) - target_archive_bytes) <= 5:
                renderer_cfg = (bbr, protect, archive_b)
                break
        if renderer_cfg is None:
            log(
                f"  ADMM picked renderer_compressed={renderer_bytes_picked}B "
                f"(would imply archive {target_archive_bytes}B); not in frontier; "
                f"falling back to champion renderer.bin ({champion_renderer_compressed}B compressed)"
            )
            renderer_bytes_blob = champ_renderer
            renderer_cfg = (0.66, 0.002, CHAMPION_BYTES)
        elif renderer_cfg[2] == CHAMPION_BYTES:
            renderer_bytes_blob = champ_renderer
        else:
            # We don't have a pre-built blob for this point; build it.
            log(
                f"  building renderer at bbr={renderer_cfg[0]}, "
                f"protect={renderer_cfg[1]} (target archive ~{renderer_cfg[2]:,}B)"
            )
            renderer_bytes_blob = _build_renderer_at(
                bbr=renderer_cfg[0], protect=renderer_cfg[1]
            )

        # Mask: pick the right CRF re-encode.
        mask_blob_path = None
        for b, c, crf, sha in mask_frontier_full:
            if int(b) == mask_bytes_picked:
                if "::champion" in sha:
                    mask_blob_path = LANE_G_V3_MASKS
                else:
                    mask_blob_path = args.output_dir / "mask_resample" / f"masks_crf{crf}.mkv"
                break
        if mask_blob_path is None or not mask_blob_path.exists():
            log(
                f"  ADMM picked mask_bytes={mask_bytes_picked} not in resample "
                f"directory; using champion masks at CRF=50"
            )
            mask_blob_path = LANE_G_V3_MASKS

        # Poses are LOCKED.
        archive_size, archive_sha, deterministic = compose_archive(
            renderer_bytes=renderer_bytes_blob,
            mask_path=mask_blob_path,
            poses_bytes=champ_poses,
            output=archive_path,
        )
        log(
            f"  wrote {archive_path} ({archive_size:,}B sha={archive_sha[:12]} "
            f"deterministic={deterministic})"
        )

    # Stage 6: save provenance
    log("Stage 6: write provenance.json")
    provenance = {
        "format": "lane_joint_admm_owv3_0120_v1",
        "lane_id": "lane_joint_admm_owv3_0120",
        "build_started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - admm_elapsed)),
        "build_completed_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion": {
            "archive": str(CHAMPION_ARCHIVE),
            "score": CHAMPION_SCORE,
            "bytes": CHAMPION_BYTES,
        },
        "byte_budget": args.byte_budget,
        "admm_config": {
            "rho_init": args.rho_init,
            "max_iters": args.max_iters,
            "primal_tol": cfg.primal_tol,
            "dual_tol": cfg.dual_tol,
            "kkt_waterline_tol": cfg.kkt_waterline_tol,
        },
        "admm_result": {
            "converged": result.converged,
            "iters": result.iters,
            "restarts": result.restarts,
            "elapsed_seconds": admm_elapsed,
            "kkt_waterline_residual": result.waterline_kkt_residual,
            "kkt_waterline_satisfied": result.waterline_satisfied,
            "final_alloc": final_alloc,
            "final_total_bytes": final_total_bytes,
            "final_score_per_stream": dict(zip(stream_names, result.final_score_per_stream)),
            "final_marginal_per_stream": dict(zip(stream_names, result.final_marginal_per_stream)),
        },
        "frontiers": {
            "renderer": [
                {
                    "bbr": bbr,
                    "protect": protect,
                    "bytes": b,
                    "score": s,
                    "pose": p,
                    "seg": sg,
                }
                for (bbr, protect, b, s, p, sg) in RENDERER_RD_FRONTIER
            ],
            "pose": [{"bytes": b, "pose_term": s} for (b, s) in pose_frontier],
            "masks": mask_meta,
        },
        "predicted_score": predicted_score,
        "predicted_score_tag": "[prediction]",
        "score_validation_required": "contest-CUDA via experiments/contest_auth_eval.py",
        "candidate_archive": {
            "path": str(archive_path) if archive_path is not None else None,
            "size_bytes": archive_size,
            "sha256": archive_sha,
            "deterministic_rebuild": deterministic,
        },
        "strict_scorer_rule_compliant": True,
        "design_memo": "experiments/build_owv3_0120_admm_stack.py",
    }
    prov_path = args.output_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    log(f"  wrote provenance: {prov_path}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nADMM coordinator complete. Output dir: {args.output_dir}")
    print(f"Predicted score: {predicted_score:.5f} [prediction]")
    print(f"Champion score:  {CHAMPION_SCORE:.6f}")
    print(f"Delta:           {predicted_score - CHAMPION_SCORE:+.5f}")
    if predicted_score < CHAMPION_SCORE - 1e-4:
        print("Status: PROMISING -- run contest-CUDA eval to verify.")
    elif abs(predicted_score - CHAMPION_SCORE) < 1e-4:
        print("Status: CONVERGED-TO-CHAMPION -- ADMM picked the existing optimum.")
    else:
        print("Status: REGRESSION -- ADMM allocation predicts higher score; do NOT eval.")
    return 0


# Renderer bytes from CHAMPION inspection (the renderer.bin extracted).
CHAMPION_BYTES_OF_RENDERER = 211_903


def _build_renderer_at(bbr: float, protect: float) -> bytes:
    """Re-build the OWV3 renderer at given knobs. Returns the encoded bytes.

    Uses tac.owv3_sensitivity_weighted.encode_owv3_archive against the
    Lane G v3 ASYM renderer + Fisher sensitivity map.
    """
    from tac.renderer_export import load_renderer_checkpoint
    from tac.sensitivity_map import (
        load_sensitivity_map,
        require_authoritative_device,
        validate_sensitivity_map_for_model,
    )
    from tac.owv3_sensitivity_weighted import encode_owv3_archive

    if not SENSITIVITY_MAP.exists():
        raise FileNotFoundError(SENSITIVITY_MAP)
    model = load_renderer_checkpoint(str(LANE_G_V3_RENDERER))
    model.eval()
    sensitivities, sens_meta = load_sensitivity_map(SENSITIVITY_MAP)
    sens_dev = sens_meta.get("source_device") or sens_meta.get("device")
    require_authoritative_device(sens_dev)
    validate_sensitivity_map_for_model(sensitivities, model, require_all_conv=True)
    blob = encode_owv3_archive(
        model=model,
        sensitivities=sensitivities,
        bit_budget_ratio=float(bbr),
        protect_threshold=float(protect),
        aggressive_threshold=1e-5,
        require_all_conv_sensitivity=True,
        fallback_action="keep_asym",
    )
    return blob


if __name__ == "__main__":
    raise SystemExit(main())
