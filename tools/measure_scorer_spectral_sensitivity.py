#!/usr/bin/env python3
"""Scorer spectral-sensitivity analyzer — the empirical scorer TRANSFER FUNCTION.

THE ARBITRARINESS CURE (operator 2026-06-09): the carrier hand-sets a single global
SIREN frequency w=30. That is one symptom of the arbitrariness class — a magic
constant by convention, not derived/measured/learned. This tool MEASURES which
spatial frequencies the FROZEN contest scorers (SegNet argmax on frame1, PoseNet on
the two-frame YUV6 motion) actually react to, so the carrier's frequency budget can
be DERIVED from the scorer instead of guessed.

METHOD (linear-response transfer function, $0 frozen-scorer measurement):
  1. decode source -> camera-res uint8 .raw (GT).
  2. for each radial frequency band k in [0,1] (DC -> Nyquist), B bands:
       build a band-limited perturbation delta_k (2D-FFT radial mask of white noise),
       normalized to a FIXED relative energy eps of each frame's std (equal energy
       per band -> H(k) is pure spectral sensitivity, not band area/energy);
       perturb ONLY frame1 of every pair (frame1 drives SegNet AND shifts the
       inter-frame motion PoseNet reads), write perturbed .raw;
       score perturbed-vs-source through the EXACT DistortionNet (ladder.score_pairs).
  3. H_seg(k) = mean d_seg(k); H_pose(k) = mean d_pose(k). The band(s) with the
     largest H are where the scorer is most sensitive -> where the carrier should
     place its frequency content (the DERIVED initializer for a learnable omega /
     Fourier-feature bandwidth / water-fill levels).

Authority: [macOS-CPU advisory] / exact_pair_scorer (uses the EXACT DistortionNet) ->
mechanism_update_eligible ONLY. It measures the scorer's sensitivity (a mechanism
fact that directs the frequency-basis design); it is NOT a candidate score. NOT
promotable; does NOT update the score roadmap. Cross-ref
`.omx/research/principled_frequency_basis_synthesis_20260609.md`.

Two subcommands:
  * ``v1`` — the original isotropic-radial argmax-d_seg curve (one amplitude, no
    coordinate conversion). PRESERVED for backward compatibility but risks
    manufacturing a misleading single "peak band".
  * ``v2`` — the hardened transfer-function ATLAS (Deliverable 1, operator
    2026-06-09): amplitude sweep, three response levels (logit-margin /
    argmax / exact), frame-incidence, RGB+YUV channel basis, orientation +
    random-phase CI, energy audit, and full coordinate conversion (camera
    cyc/px, scorer cyc/px, normalized omega, SIREN-w-equivalent). The reusable
    physics lives in ``tac.analysis.scorer_spectral_sensitivity_v2``.

Run v1 (fast; a few min for N~24 x B~8 on the frozen scorer):
  .venv/bin/python tools/measure_scorer_spectral_sensitivity.py v1 \\
    --n-pairs 24 --n-bands 8 --rel-energy 0.05 \\
    --work-dir /Volumes/VertigoDataTier/pact/scorer_spectral_sensitivity_<utc> \\
    --out <work>/scorer_spectral_sensitivity.v1.json

Run v2 atlas (the full grid is large; configure the grid to bound wall-clock —
default grid is 5760 cells x 6 pairs x 2 phases = ~69k scorer-forward groups on
CPU. Trim orientations/channels/amplitudes for a fast pass):
  .venv/bin/python tools/measure_scorer_spectral_sensitivity.py v2 \\
    --n-pairs 6 --n-bands 8 --band-spacing log \\
    --amplitudes-lsb 0.5,2,8 --orientations isotropic,horizontal,vertical \\
    --channel-bases rgb,yuv --rgb-channels all --yuv-channels y,all \\
    --frame-incidences frame1_only,both_opposite --n-phase-samples 2 \\
    --work-dir /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_<utc> \\
    --out <work>/scorer_spectral_sensitivity.v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _utc() -> str:
    import subprocess

    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip()


def _tier_names() -> tuple[str, ...]:
    """Canonical tier names from the runner's preset dict (single source)."""
    from tac.analysis.scorer_spectral_atlas_runner import TIER_PRESETS

    return tuple(TIER_PRESETS)


def _band_limited_perturbation(
    shape: tuple[int, int, int], r_lo: float, r_hi: float, rng: Any
) -> Any:
    """Real band-limited field on the radial-frequency annulus [r_lo, r_hi] (r in
    [0,1], 0=DC, 1=Nyquist corner), unit std. One independent draw per channel."""
    import numpy as np

    h, w, c = shape
    fy = np.fft.fftfreq(h)[:, None]  # [-0.5, 0.5)
    fx = np.fft.fftfreq(w)[None, :]
    # normalize radius so 1.0 == the Nyquist corner (max |f| = 0.5 on each axis).
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2) / np.sqrt(2.0)
    band = ((radius >= r_lo) & (radius < r_hi)).astype(np.float64)
    if band.sum() == 0:
        return np.zeros(shape, dtype=np.float64)
    out = np.empty(shape, dtype=np.float64)
    for ch in range(c):
        noise = rng.standard_normal((h, w))
        spec = np.fft.fft2(noise) * band
        field = np.real(np.fft.ifft2(spec))
        s = field.std()
        out[..., ch] = field / s if s > 0 else field
    return out


def _run_v1(args: argparse.Namespace) -> int:
    import hi_nerv_renderer_sanity_ladder as ladder
    import numpy as np

    work = args.work_dir.resolve()
    if "/tmp" in str(work):
        raise SystemExit("work-dir must not be under /tmp (durable SSD only)")
    work.mkdir(parents=True, exist_ok=True)
    H, W, C = ladder._frame_dims()
    N = int(args.n_pairs)
    B = int(args.n_bands)
    eps = float(args.rel_energy)

    # 1. GT camera-res .raw (first 2N frames).
    src_raw = work / "source.raw"
    ladder.decode_source_to_raw(_REPO / "upstream" / "videos" / "0.mkv", src_raw, max_frames=2 * N)
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    navail = frames.shape[0] // 2
    N = min(N, navail)
    indices = list(range(N))

    # sanity: source-vs-source must be ~0 distortion (the measurement baseline).
    base = ladder.score_pairs(src_raw, src_raw, indices, device=args.device)
    base_rows = base["pairs"]
    base_seg = float(np.mean([r["d_seg"] for r in base_rows]))
    base_pose = float(np.mean([r["d_pose"] for r in base_rows]))

    rng = np.random.default_rng(args.seed)
    bands: list[dict[str, Any]] = []
    pert_raw = work / "perturbed.raw"
    for k in range(B):
        r_lo, r_hi = k / B, (k + 1) / B
        # Build perturbed frames: copy, perturb ONLY frame1 (odd index) of each pair.
        pert = np.array(frames[: 2 * N], dtype=np.float64)
        for i in range(N):
            f1 = pert[2 * i + 1]
            field = _band_limited_perturbation((H, W, C), r_lo, r_hi, rng)
            delta = eps * float(f1.std()) * field  # equal relative energy per band
            pert[2 * i + 1] = np.clip(f1 + delta, 0.0, 255.0)
        pert.astype(np.uint8).tofile(pert_raw)
        res = ladder.score_pairs(pert_raw, src_raw, indices, device=args.device)
        rows = res["pairs"]
        d_seg = float(np.mean([r["d_seg"] for r in rows]))
        d_pose = float(np.mean([r["d_pose"] for r in rows]))
        bands.append(
            {
                "band_index": k,
                "r_lo": r_lo,
                "r_hi": r_hi,
                "H_seg": d_seg - base_seg,  # transfer function (response above baseline)
                "H_pose": d_pose - base_pose,
                "d_seg_raw": d_seg,
                "d_pose_raw": d_pose,
            }
        )
        print(
            f"[spectral] band {k} r[{r_lo:.3f},{r_hi:.3f}] "
            f"H_seg={d_seg - base_seg:+.5f} H_pose={d_pose - base_pose:+.4f}"
        )

    seg_peak = max(bands, key=lambda b: b["H_seg"])
    pose_peak = max(bands, key=lambda b: b["H_pose"])
    artifact: dict[str, Any] = {
        "schema": "scorer_spectral_sensitivity.v1",
        "utc": _utc(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "score_roadmap_update_eligible": False,
        "mechanism_update_eligible": True,
        "promotable": False,
        "n_pairs": N,
        "n_bands": B,
        "rel_energy": eps,
        "baseline_d_seg": base_seg,
        "baseline_d_pose": base_pose,
        "bands": bands,
        "seg_peak_band": seg_peak,
        "pose_peak_band": pose_peak,
        "derived_frequency_target": {
            "note": (
                "H_seg / H_pose are the scorer's per-band spectral sensitivity (Δdistortion "
                "per equal-energy perturbation). The band(s) with peak H are where the carrier "
                "should place frequency content — the DERIVED (not hand-tuned) target for a "
                "learnable omega / Fourier-feature bandwidth / per-band water-fill level. "
                "r is normalized radial frequency (0=DC, 1=Nyquist corner)."
            ),
            "seg_peak_band_index": seg_peak["band_index"],
            "seg_peak_r_center": 0.5 * (seg_peak["r_lo"] + seg_peak["r_hi"]),
            "pose_peak_band_index": pose_peak["band_index"],
            "pose_peak_r_center": 0.5 * (pose_peak["r_lo"] + pose_peak["r_hi"]),
        },
        "arbitrariness_note": (
            "This measurement converts the arbitrary SIREN w from a convention into a "
            "scorer-derived quantity (non-arbitrariness principle). Sister design: "
            ".omx/research/principled_frequency_basis_synthesis_20260609.md"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print("=== scorer spectral-sensitivity transfer function ===")
    print(f"  baseline d_seg={base_seg:.5f} d_pose={base_pose:.4f} (source-vs-source)")
    print(f"  SEG  peak band {seg_peak['band_index']} r~{0.5*(seg_peak['r_lo']+seg_peak['r_hi']):.3f}  H_seg={seg_peak['H_seg']:+.5f}")
    print(f"  POSE peak band {pose_peak['band_index']} r~{0.5*(pose_peak['r_lo']+pose_peak['r_hi']):.3f}  H_pose={pose_peak['H_pose']:+.4f}")
    print(f"  wrote {args.out}")
    return 0


def _csv_floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _csv_strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _run_v2(args: argparse.Namespace) -> int:
    """The hardened transfer-function ATLAS (Deliverable 1).

    Builds the configurable v2 grid {pair, band, orientation, amplitude, basis,
    channel, frame_incidence} and measures three response levels through the
    EXACT frozen scorer, with full coordinate conversion + energy audit + CI.
    """
    import hi_nerv_renderer_sanity_ladder as ladder
    import numpy as np

    from tac.analysis import scorer_spectral_sensitivity_v2 as v2

    work = args.work_dir.resolve()
    if "/tmp" in str(work):
        raise SystemExit("work-dir must not be under /tmp (durable SSD only)")
    work.mkdir(parents=True, exist_ok=True)
    H, W, C = ladder._frame_dims()

    # Decode the source pairs (camera-res GT) once.
    src_raw = work / "source.raw"
    ladder.decode_source_to_raw(
        _REPO / "upstream" / "videos" / "0.mkv",
        src_raw,
        max_frames=2 * int(args.n_pairs),
    )
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    n_avail = frames.shape[0] // 2
    n_pairs = min(int(args.n_pairs), n_avail)
    pairs = np.array(frames[: 2 * n_pairs]).reshape(n_pairs, 2, H, W, C)

    grid = v2.AtlasGrid(
        n_pairs=n_pairs,
        n_bands=int(args.n_bands),
        band_spacing=args.band_spacing,
        amplitudes_lsb=tuple(args.amplitudes_lsb),
        orientations=tuple(args.orientations),
        frame_incidences=tuple(args.frame_incidences),
        channel_bases=tuple(args.channel_bases),
        rgb_channels=tuple(args.rgb_channels),
        yuv_channels=tuple(args.yuv_channels),
        n_phase_samples=int(args.n_phase_samples),
        seed=int(args.seed),
    )
    print(
        f"[spectral.v2] grid: {grid.total_cells()} cells x {n_pairs} pairs x "
        f"{grid.n_phase_samples} phases = {grid.total_scorer_forwards()} scorer-forward "
        f"groups (each = source + perturbed). device={args.device}",
        flush=True,
    )

    artifact = v2.measure_atlas(pairs, grid, device=args.device, progress=True)
    artifact["utc"] = _utc()
    artifact["source_raw"] = {
        "path": str(src_raw),
        "n_pairs": n_pairs,
        "camera_hw": [H, W],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    hl = artifact["headline"]
    print("=== scorer spectral-sensitivity ATLAS (v2) ===")
    print(
        f"  baseline d_seg={artifact['baseline']['d_seg']:.5f} "
        f"d_pose={artifact['baseline']['d_pose']:.4f} (source-vs-source)"
    )
    for name, key in (("SEG", "seg_peak"), ("POSE", "pose_peak"), ("MARGIN", "logit_margin_peak")):
        p = hl[key]
        if not p:
            continue
        print(
            f"  {name:6s} peak: H={p['H']:+.5f} @ band{p['band_index']} {p['orientation']} "
            f"a={p['amplitude_lsb']}LSB {p['channel_basis']}:{p['channel']} {p['frame_incidence']} "
            f"| w_equiv={p['siren_w_equivalent']:.1f} "
            f"scorer_cyc/px={p['scorer_cycles_per_pixel']:.3f} alias={p['aliases_at_scorer']}"
        )
    print(f"  W-VERDICT: {hl['w_verdict_note']}")
    print(f"  wrote {args.out}")
    return 0


def _run_v2_resume(args: argparse.Namespace) -> int:
    """The RESUMABLE, multi-tier atlas (operator standing directive 2026-06-11).

    Streams each cell to a durable JSONL as it is computed, SKIPS cells already
    present on startup (idempotent resume), and re-aggregates the final atlas
    from the full JSONL. ``--tier {quick,medium,exhaustive}`` selects a preset
    grid; explicit grid flags override the preset. Writes a DONE.marker + a
    progress sidecar so an external check sees %-complete without parsing the
    JSONL.
    """
    import hi_nerv_renderer_sanity_ladder as ladder
    import numpy as np

    from tac.analysis import scorer_spectral_atlas_parallel as parallel
    from tac.analysis import scorer_spectral_atlas_runner as runner

    work = args.work_dir.resolve()
    if "/tmp" in str(work):
        raise SystemExit("work-dir must not be under /tmp (durable SSD only)")
    work.mkdir(parents=True, exist_ok=True)

    paths = runner.AtlasRunPaths.under(work, atlas_out=args.out)

    # Tier preset -> base grid; explicit flags override the preset values.
    grid = runner.grid_for_tier(args.tier, seed=int(args.seed))
    overrides: dict[str, object] = {}
    if args.n_pairs is not None:
        overrides["n_pairs"] = int(args.n_pairs)
    if args.n_bands is not None:
        overrides["n_bands"] = int(args.n_bands)
    if args.band_spacing is not None:
        overrides["band_spacing"] = args.band_spacing
    if args.amplitudes_lsb is not None:
        overrides["amplitudes_lsb"] = tuple(args.amplitudes_lsb)
    if args.orientations is not None:
        overrides["orientations"] = tuple(args.orientations)
    if args.frame_incidences is not None:
        overrides["frame_incidences"] = tuple(args.frame_incidences)
    if args.channel_bases is not None:
        overrides["channel_bases"] = tuple(args.channel_bases)
    if args.rgb_channels is not None:
        overrides["rgb_channels"] = tuple(args.rgb_channels)
    if args.yuv_channels is not None:
        overrides["yuv_channels"] = tuple(args.yuv_channels)
    if args.n_phase_samples is not None:
        overrides["n_phase_samples"] = int(args.n_phase_samples)
    if overrides:
        from dataclasses import replace

        grid = replace(grid, **overrides)

    H, W, C = ladder._frame_dims()
    # Decode the source pairs (camera-res GT) once.
    src_raw = work / "source.raw"
    ladder.decode_source_to_raw(
        _REPO / "upstream" / "videos" / "0.mkv",
        src_raw,
        max_frames=2 * int(grid.n_pairs),
    )
    frames = np.memmap(src_raw, dtype=np.uint8, mode="r").reshape(-1, H, W, C)
    n_avail = frames.shape[0] // 2
    n_pairs = min(int(grid.n_pairs), n_avail)
    pairs = np.array(frames[: 2 * n_pairs]).reshape(n_pairs, 2, H, W, C)

    total = grid.total_cells()

    # Resolve the worker count: --workers auto -> auto_worker_count() (leaves
    # headroom for the coexisting capstone daemon); an explicit int is used as-is;
    # 1 is the serial path (the bit-identity reference).
    workers = (
        parallel.auto_worker_count()
        if args.workers == "auto"
        else max(1, int(args.workers))
    )

    # rough ETA from the killed-run anchor (~2.3 min/cell at n_pairs=12, phases=3
    # on the CPU scorer); scaled by this run's pairs*phases vs that anchor, then
    # DIVIDED by the worker count (cells are independent — near-linear speedup).
    anchor_sec_per_cell = 138.0  # 2.3 min/cell @ 12 pairs * 3 phases (the kill anchor)
    anchor_work = 12 * 3
    this_work = max(1, n_pairs * int(grid.n_phase_samples))
    est_sec_per_cell = anchor_sec_per_cell * (this_work / anchor_work)
    completed_keys, _ = runner.load_completed_cells(paths.cells_jsonl)
    remaining = total - len(completed_keys)
    est_remaining_h = est_sec_per_cell * remaining / 3600.0 / max(1, workers)
    est_total_h = est_sec_per_cell * total / 3600.0 / max(1, workers)
    print(
        f"[spectral.v2-resume] tier={args.tier} cells={total} "
        f"(already_done={len(completed_keys)}) n_pairs={n_pairs} "
        f"phases={grid.n_phase_samples} workers={workers} "
        f"rough_ETA~{est_total_h:.2f}h "
        f"(~{est_remaining_h:.2f}h remaining at {workers} workers) "
        f"device={args.device}\n"
        f"  jsonl={paths.cells_jsonl}\n  progress={paths.progress_json}\n"
        f"  done_marker={paths.done_marker}\n  atlas={paths.atlas_json}",
        flush=True,
    )

    exit_code = 0
    try:
        atlas = parallel.run_resumable_atlas_parallel(
            pairs,
            grid,
            paths,
            tier=args.tier,
            workers=workers,
            raw_path=src_raw,
            device=args.device,
            torch_threads_per_worker=int(args.torch_threads_per_worker),
            progress_every=int(args.progress_every),
        )
        hl = atlas.get("headline", {})
        print("=== scorer spectral-sensitivity ATLAS (v2-resume) ===")
        print(f"  cells_measured={atlas.get('cells_measured')}/{total}")
        for name, key in (("SEG", "seg_peak"), ("POSE", "pose_peak"), ("MARGIN", "logit_margin_peak")):
            p = hl.get(key) or {}
            if not p:
                continue
            print(
                f"  {name:6s} peak: H={p['H']:+.5f} @ band{p['band_index']} {p['orientation']} "
                f"a={p['amplitude_lsb']}LSB {p['channel_basis']}:{p['channel']} {p['frame_incidence']} "
                f"| w_equiv={p['siren_w_equivalent']:.1f} aliases={p['aliases_at_scorer']}"
            )
        lo = atlas.get("lowering_opportunities", {})
        shed = lo.get("shed_here_low_sensitivity_cells", [])
        print(
            f"  SHED-BYTES candidates (scorer-blind cells): {lo.get('n_shed_candidate_cells', 0)} "
            f"(top {len(shed)} reported); consumer={lo.get('consumer')}"
        )
        print(f"  wrote {paths.atlas_json}")
    except KeyboardInterrupt:
        exit_code = 130
        print("[spectral.v2-resume] interrupted; progress preserved in JSONL", flush=True)
    except Exception as exc:
        exit_code = 1
        print(f"[spectral.v2-resume] FAILED: {exc!r}; progress preserved in JSONL", flush=True)
        raise
    finally:
        completed_now, _ = runner.load_completed_cells(paths.cells_jsonl)
        runner.write_done_marker(
            paths.done_marker,
            exit_code=exit_code,
            completed=len(completed_now),
            total=total,
        )
    return exit_code


def _run_v2_aggregate(args: argparse.Namespace) -> int:
    """Re-aggregate the atlas (+ lowering analysis) from a partial JSONL.

    Lets an operator inspect the atlas + the score-lowering opportunities of a
    STILL-RUNNING (or killed) sweep WITHOUT touching the daemon — purely reads
    the durable JSONL, rebuilds the headline + lowering analysis. No scorer load.
    """
    from tac.analysis import scorer_spectral_atlas_runner as runner

    jsonl = args.cells_jsonl.resolve()
    if not jsonl.exists():
        raise SystemExit(f"cells JSONL not found: {jsonl}")
    _keys, cells = runner.load_completed_cells(jsonl)
    if not cells:
        raise SystemExit(f"no cells in {jsonl}")

    lo = runner.analyze_lowering_opportunities(cells)
    out = {
        "schema": "scorer_spectral_atlas_aggregate.v1",
        "utc": runner._utc_iso(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "promotable": False,
        "mechanism_update_eligible": True,
        "cells_jsonl": str(jsonl),
        "cells_measured": len(cells),
        "lowering_opportunities": lo,
    }
    # headline peaks via the same selection used in the live aggregation
    seg = max(cells, key=lambda c: c.get("H_seg", 0.0))
    pose = max(cells, key=lambda c: c.get("H_pose", 0.0))
    out["seg_peak_cell"] = seg
    out["pose_peak_cell"] = pose
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[spectral.v2-aggregate] {len(cells)} cells -> {args.out}")
    print(
        f"  SEG peak H_seg={seg.get('H_seg'):+.5f} @ band{seg.get('band_index')} "
        f"{seg.get('orientation')} {seg.get('channel_basis')}:{seg.get('channel')}"
    )
    print(
        f"  POSE peak H_pose={pose.get('H_pose'):+.4f} @ band{pose.get('band_index')} "
        f"{pose.get('orientation')} {pose.get('channel_basis')}:{pose.get('channel')}"
    )
    print(
        f"  SHED-BYTES candidates: {lo.get('n_shed_candidate_cells', 0)}; "
        f"consumer={lo.get('consumer')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    # v1 (the original isotropic-radial argmax-d_seg transfer function; preserved).
    v1 = sub.add_parser("v1", help="original v1 isotropic-radial argmax-d_seg curve")
    v1.add_argument("--n-pairs", type=int, default=24, help="pairs to average per band")
    v1.add_argument("--n-bands", type=int, default=8, help="radial frequency bands DC->Nyquist")
    v1.add_argument(
        "--rel-energy",
        type=float,
        default=0.05,
        help="perturbation std as a fraction of each frame's std (equal per band)",
    )
    v1.add_argument("--device", default="cpu")
    v1.add_argument("--seed", type=int, default=0)
    v1.add_argument("--work-dir", required=True, type=Path)
    v1.add_argument("--out", required=True, type=Path)
    v1.set_defaults(_fn=_run_v1)

    # v2 (the hardened atlas; Deliverable 1).
    v2p = sub.add_parser("v2", help="hardened transfer-function ATLAS (3 levels, amp/orient/channel sweep)")
    v2p.add_argument("--n-pairs", type=int, default=6, help="source pairs to average per cell")
    v2p.add_argument("--n-bands", type=int, default=6, help="radial frequency bands")
    v2p.add_argument(
        "--band-spacing",
        default="linear",
        choices=("linear", "log"),
        help="linear=uniform spectrum; log=denser low-freq (resolves the w=1..30 regime)",
    )
    v2p.add_argument(
        "--amplitudes-lsb",
        type=_csv_floats,
        default=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        help="comma-separated perturbation amplitudes in LSB (1 LSB = 1/255)",
    )
    v2p.add_argument(
        "--orientations",
        type=_csv_strs,
        default=("isotropic", "horizontal", "vertical", "diag_plus", "diag_minus"),
        help="comma-separated subset of isotropic,horizontal,vertical,diag_plus,diag_minus",
    )
    v2p.add_argument(
        "--frame-incidences",
        type=_csv_strs,
        default=("frame0_only", "frame1_only", "both_same", "both_opposite"),
        help="comma-separated subset of frame0_only,frame1_only,both_same,both_opposite",
    )
    v2p.add_argument(
        "--channel-bases",
        type=_csv_strs,
        default=("rgb", "yuv"),
        help="comma-separated subset of rgb,yuv",
    )
    v2p.add_argument(
        "--rgb-channels",
        type=_csv_strs,
        default=("all", "r", "g", "b"),
        help="comma-separated subset of all,r,g,b (perturbed when channel_basis=rgb)",
    )
    v2p.add_argument(
        "--yuv-channels",
        type=_csv_strs,
        default=("all", "y", "u", "v"),
        help="comma-separated subset of all,y,u,v (perturbed when channel_basis=yuv)",
    )
    v2p.add_argument("--n-phase-samples", type=int, default=2, help="random-phase draws per cell (CI)")
    v2p.add_argument("--device", default="cpu")
    v2p.add_argument("--seed", type=int, default=0)
    v2p.add_argument("--work-dir", required=True, type=Path)
    v2p.add_argument("--out", required=True, type=Path)
    v2p.set_defaults(_fn=_run_v2)

    # v2-resume (the RESUMABLE, multi-tier atlas — Deliverable 2026-06-11).
    v2r = sub.add_parser(
        "v2-resume",
        help="RESUMABLE multi-tier atlas: per-cell JSONL, skip-completed resume, tiers, DONE.marker",
    )
    v2r.add_argument(
        "--tier",
        default="medium",
        choices=tuple(_tier_names()),
        help="resolution preset: quick (~32 cells, min) | medium (~192 cells, ~1-2h) | exhaustive (~6400, ~days)",
    )
    # Explicit grid overrides (None = use the tier preset's value).
    v2r.add_argument("--n-pairs", type=int, default=None, help="override tier n_pairs")
    v2r.add_argument("--n-bands", type=int, default=None, help="override tier n_bands")
    v2r.add_argument(
        "--band-spacing", default=None, choices=("linear", "log"), help="override tier band spacing"
    )
    v2r.add_argument("--amplitudes-lsb", type=_csv_floats, default=None, help="override tier amplitudes")
    v2r.add_argument("--orientations", type=_csv_strs, default=None, help="override tier orientations")
    v2r.add_argument("--frame-incidences", type=_csv_strs, default=None, help="override tier frame incidences")
    v2r.add_argument("--channel-bases", type=_csv_strs, default=None, help="override tier channel bases")
    v2r.add_argument("--rgb-channels", type=_csv_strs, default=None, help="override tier rgb channels")
    v2r.add_argument("--yuv-channels", type=_csv_strs, default=None, help="override tier yuv channels")
    v2r.add_argument("--n-phase-samples", type=int, default=None, help="override tier phase samples")
    v2r.add_argument("--progress-every", type=int, default=1, help="refresh progress sidecar every N cells")
    v2r.add_argument(
        "--workers",
        default="auto",
        help=(
            "cell-level parallelism: 'auto' (=min(12, physical_cores-4), leaves "
            "headroom for the capstone daemon) | an int | 1 (serial, the "
            "bit-identity reference). Cells are independent => near-linear speedup."
        ),
    )
    v2r.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=1,
        help="torch.set_num_threads + OMP/MKL per worker (workers*threads <= ~cores)",
    )
    v2r.add_argument("--device", default="cpu")
    v2r.add_argument("--seed", type=int, default=0)
    v2r.add_argument("--work-dir", required=True, type=Path)
    v2r.add_argument(
        "--out",
        type=Path,
        default=None,
        help="final atlas json (default: <work-dir>/atlas.json)",
    )
    v2r.set_defaults(_fn=_run_v2_resume)

    # v2-aggregate (re-aggregate atlas + lowering analysis from a PARTIAL JSONL).
    v2a = sub.add_parser(
        "v2-aggregate",
        help="re-aggregate atlas + score-lowering analysis from a partial/complete cells JSONL (no scorer load)",
    )
    v2a.add_argument("--cells-jsonl", required=True, type=Path, help="path to atlas_cells.jsonl")
    v2a.add_argument("--out", required=True, type=Path, help="aggregate json output")
    v2a.set_defaults(_fn=_run_v2_aggregate)

    args = ap.parse_args(argv)
    # v2-resume: --out defaults to <work-dir>/atlas.json (computed post-parse).
    if getattr(args, "cmd", None) == "v2-resume" and args.out is None:
        args.out = args.work_dir.resolve() / "atlas.json"
    if not getattr(args, "_fn", None):
        ap.print_help()
        return 2
    return int(args._fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
