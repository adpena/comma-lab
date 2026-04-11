#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run *cmd* via subprocess.  When *capture* is True, stdout/stderr are captured
    as bytes (text=False); otherwise they pass through as text (text=True)."""
    return subprocess.run(
        cmd,
        check=True,
        text=not capture,
        capture_output=capture,
    )


def _ffprobe(path: Path, ffprobe_bin: str) -> dict:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    cp = _run(cmd, capture=True)
    return json.loads(cp.stdout)


def _fps_from_rate(rate: str) -> float:
    frac = Fraction(rate)
    return frac.numerator / frac.denominator if frac.denominator else 0.0


def _video_meta(path: Path, ffprobe_bin: str) -> dict:
    data = _ffprobe(path, ffprobe_bin)
    stream = data["streams"][0]
    fps = _fps_from_rate(stream.get("avg_frame_rate", "0/1"))
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    total_frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    if total_frames_raw in (None, "N/A"):
        total_frames = int(round(duration * fps))
    else:
        total_frames = int(total_frames_raw)
    return {
        "source_width": int(stream["width"]),
        "source_height": int(stream["height"]),
        "fps": fps,
        "duration_sec": duration,
        "total_frames": total_frames,
    }


def _extract_sampled_gray(
    video: Path,
    scale_w: int,
    scale_h: int,
    sample_step: int,
    ffmpeg_bin: str,
    source_color_range: str,
    source_color_matrix: str,
    source_color_primaries: str,
    source_color_trc: str,
) -> tuple[list[bytes], list[int]]:
    # The backslash-escaped comma in the select expression is required by ffmpeg's
    # filter_complex parser.  Since we use subprocess (no shell), no further quoting
    # is needed.
    vf = (
        f"select='not(mod(n\\,{sample_step}))',"
        f"scale={scale_w}:{scale_h}:flags=bilinear:"
        f"in_range={source_color_range}:out_range={source_color_range}:"
        f"in_color_matrix={source_color_matrix}:out_color_matrix={source_color_matrix}:"
        f"in_primaries={source_color_primaries}:out_primaries={source_color_primaries}:"
        f"in_transfer={source_color_trc}:out_transfer={source_color_trc},"
        f"format=gray"
    )
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-",
    ]
    cp = _run(cmd, capture=True)
    frame_size = scale_w * scale_h
    raw: bytes = cp.stdout  # type: ignore[assignment]
    if len(raw) % frame_size != 0:
        raise ValueError(f"Unexpected raw buffer size {len(raw)} for frame size {frame_size}")
    frames = [raw[i : i + frame_size] for i in range(0, len(raw), frame_size)]
    frame_indices = [i * sample_step for i in range(len(frames))]
    return frames, frame_indices


@dataclass
class RoiBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class WindowRoi:
    index: int
    frame_start: int
    frame_end: int
    main_roi: RoiBox
    aux_roi: RoiBox | None
    stats: dict[str, float]


@dataclass
class RoiMetadata:
    video: str
    source_width: int
    source_height: int
    scale_width: int
    scale_height: int
    fps: float
    total_frames: int
    sample_step: int
    window_frames: int
    tile_cols: int
    tile_rows: int
    windows: list[WindowRoi]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _even(value: int, minimum: int = 2) -> int:
    if value < minimum:
        value = minimum
    if value % 2:
        value -= 1
    return max(value, minimum)


def _tile_bounds(scale_w: int, scale_h: int, cols: int, rows: int, cx: int, cy: int) -> tuple[int, int, int, int]:
    x0 = (scale_w * cx) // cols
    x1 = (scale_w * (cx + 1)) // cols
    y0 = (scale_h * cy) // rows
    y1 = (scale_h * (cy + 1)) // rows
    if (x1 - x0) % 2:
        x1 -= 1
    if (y1 - y0) % 2:
        y1 -= 1
    return x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)


def _window_sample_indices(frame_indices: list[int], start: int, end: int) -> list[int]:
    idxs = [i for i, src_idx in enumerate(frame_indices) if start <= src_idx < end]
    if len(idxs) >= 2:
        return idxs
    if not frame_indices:
        return []
    nearest = sorted(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - start))
    return sorted(nearest[: min(2, len(nearest))])


def _tile_metrics(
    frames: list[bytes],
    frame_indices: list[int],
    *,
    scale_w: int,
    scale_h: int,
    cols: int,
    rows: int,
    window_start: int,
    window_end: int,
) -> tuple[list[list[float]], list[list[float]]]:
    import numpy as _np

    idxs = _window_sample_indices(frame_indices, window_start, window_end)
    if not idxs:
        raise ValueError("No sampled frames available for window")
    pairs = list(zip(idxs, idxs[1:]))
    diff_scores = [[0.0 for _ in range(cols)] for _ in range(rows)]
    edge_scores = [[0.0 for _ in range(cols)] for _ in range(rows)]

    # Convert frames to numpy arrays for vectorized operations
    np_frames = {i: _np.frombuffer(frames[i], dtype=_np.uint8).reshape(scale_h, scale_w) for i in set(sum(pairs, (idxs[-1],)))}
    ref_arr = np_frames[idxs[-1]]

    for ty in range(rows):
        for tx in range(cols):
            x0, y0, x1, y1 = _tile_bounds(scale_w, scale_h, cols, rows, tx, ty)
            pixels = (x1 - x0) * (y1 - y0)

            # Vectorized temporal difference
            diff_total = 0.0
            for left_idx, right_idx in pairs:
                left_tile = np_frames[left_idx][y0:y1, x0:x1].astype(_np.int16)
                right_tile = np_frames[right_idx][y0:y1, x0:x1].astype(_np.int16)
                diff_total += float(_np.abs(right_tile - left_tile).sum())
            pair_count = max(1, len(pairs))
            diff_scores[ty][tx] = diff_total / (pixels * pair_count)

            # Vectorized edge detection (subsampled 2x)
            tile = ref_arr[y0:y1, x0:x1].astype(_np.int16)
            if tile.shape[0] > 1 and tile.shape[1] > 1:
                horiz = _np.abs(tile[1::2, 1::2] - tile[1::2, 0:-1:2])
                vert = _np.abs(tile[1::2, 1::2] - tile[0:-1:2, 1::2])
                edge_count = horiz.size + vert.size
                edge_scores[ty][tx] = float(horiz.sum() + vert.sum()) / max(1, edge_count)
    return diff_scores, edge_scores


def _normalize_grid(grid: list[list[float]]) -> list[list[float]]:
    flat = [value for row in grid for value in row]
    lo = min(flat)
    hi = max(flat)
    if hi - lo < 1e-9:
        return [[0.0 for _ in row] for row in grid]
    return [[(value - lo) / (hi - lo) for value in row] for row in grid]


def _derive_rois(
    diff_scores: list[list[float]],
    edge_scores: list[list[float]],
    *,
    scale_w: int,
    scale_h: int,
    cols: int,
    rows: int,
) -> tuple[RoiBox, RoiBox | None, dict[str, float]]:
    diff_norm = _normalize_grid(diff_scores)
    edge_norm = _normalize_grid(edge_scores)
    combined = [[0.0 for _ in range(cols)] for _ in range(rows)]
    total_mass = 0.0
    cx_weight = 0.0
    cy_weight = 0.0

    for ty in range(rows):
        for tx in range(cols):
            cx = (tx + 0.5) / cols
            cy = (ty + 0.5) / rows
            center_prior = math.exp(-(((cx - 0.50) ** 2) / 0.050 + ((cy - 0.58) ** 2) / 0.035))
            road_prior = 1.0 if 0.30 <= cy <= 0.92 else 0.35
            value = 0.62 * diff_norm[ty][tx] + 0.18 * edge_norm[ty][tx] + 0.15 * center_prior + 0.05 * road_prior
            combined[ty][tx] = value
            mass = value * value
            total_mass += mass
            cx_weight += mass * cx
            cy_weight += mass * cy

    centroid_x = cx_weight / max(total_mass, 1e-9)
    centroid_y = cy_weight / max(total_mass, 1e-9)
    centroid_x = 0.55 * centroid_x + 0.45 * 0.50
    centroid_y = 0.70 * centroid_y + 0.30 * 0.58

    spread_x = 0.0
    spread_y = 0.0
    for ty in range(rows):
        for tx in range(cols):
            cx = (tx + 0.5) / cols
            cy = (ty + 0.5) / rows
            mass = combined[ty][tx] * combined[ty][tx]
            spread_x += mass * (cx - centroid_x) ** 2
            spread_y += mass * (cy - centroid_y) ** 2
    spread_x = math.sqrt(spread_x / max(total_mass, 1e-9))
    spread_y = math.sqrt(spread_y / max(total_mass, 1e-9))

    main_w = _even(int(round(scale_w * _clamp(0.54 + 2.0 * spread_x, 0.50, 0.74))))
    main_h = _even(int(round(scale_h * _clamp(0.42 + 2.6 * spread_y, 0.40, 0.68))))
    main_x = _even(int(round(scale_w * centroid_x - main_w / 2)), minimum=0)
    main_y = _even(int(round(scale_h * centroid_y - main_h / 2)), minimum=0)
    main_x = min(main_x, scale_w - main_w)
    main_y = min(main_y, scale_h - main_h)
    main = RoiBox(x=main_x, y=main_y, w=main_w, h=main_h)

    flat_scores = [value for row in combined for value in row]
    score_mean = sum(flat_scores) / len(flat_scores)
    score_var = sum((v - score_mean) ** 2 for v in flat_scores) / len(flat_scores)
    score_std = math.sqrt(score_var)
    threshold = max(score_mean + 0.85 * score_std, max(flat_scores) * 0.74)

    aux_tiles: list[tuple[int, int, float]] = []
    for ty in range(rows):
        for tx in range(cols):
            x0, y0, x1, y1 = _tile_bounds(scale_w, scale_h, cols, rows, tx, ty)
            center_x_px = (x0 + x1) / 2
            center_y_px = (y0 + y1) / 2
            inside_main = (
                main.x <= center_x_px <= main.x + main.w
                and main.y <= center_y_px <= main.y + main.h
            )
            if not inside_main and combined[ty][tx] >= threshold:
                aux_tiles.append((tx, ty, combined[ty][tx]))

    aux: RoiBox | None = None
    aux_mass = sum(v for _, _, v in aux_tiles)
    if aux_tiles and aux_mass / max(sum(flat_scores), 1e-9) > 0.16:
        min_tx = min(tx for tx, _, _ in aux_tiles)
        max_tx = max(tx for tx, _, _ in aux_tiles)
        min_ty = min(ty for _, ty, _ in aux_tiles)
        max_ty = max(ty for _, ty, _ in aux_tiles)
        x0, y0, _, _ = _tile_bounds(scale_w, scale_h, cols, rows, max(min_tx - 1, 0), max(min_ty - 1, 0))
        _, _, x1, y1 = _tile_bounds(scale_w, scale_h, cols, rows, min(max_tx + 1, cols - 1), min(max_ty + 1, rows - 1))
        aux_w = _even(int(_clamp(x1 - x0, scale_w * 0.12, scale_w * 0.28)))
        aux_h = _even(int(_clamp(y1 - y0, scale_h * 0.18, scale_h * 0.56)))
        aux_x = min(_even(x0, minimum=0), scale_w - aux_w)
        aux_y = min(_even(y0, minimum=0), scale_h - aux_h)
        aux_center_x = aux_x + aux_w / 2
        main_center_x = main.x + main.w / 2
        if abs(aux_center_x - main_center_x) >= scale_w * 0.16:
            aux = RoiBox(x=aux_x, y=aux_y, w=aux_w, h=aux_h)

    stats = {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "score_mean": score_mean,
        "score_std": score_std,
        "aux_mass_fraction": aux_mass / max(sum(flat_scores), 1e-9),
    }
    return main, aux, stats


def analyze(args: argparse.Namespace) -> int:
    video = Path(args.video)
    meta = _video_meta(video, args.ffprobe_bin)
    frames, frame_indices = _extract_sampled_gray(
        video,
        args.scale_w,
        args.scale_h,
        args.sample_step,
        args.ffmpeg_bin,
        args.source_color_range,
        args.source_color_matrix,
        args.source_color_primaries,
        args.source_color_trc,
    )

    windows: list[WindowRoi] = []
    frame_start = 0
    index = 0
    while frame_start < meta["total_frames"]:
        frame_end = min(meta["total_frames"], frame_start + args.window_frames)
        diff_scores, edge_scores = _tile_metrics(
            frames,
            frame_indices,
            scale_w=args.scale_w,
            scale_h=args.scale_h,
            cols=args.tile_cols,
            rows=args.tile_rows,
            window_start=frame_start,
            window_end=frame_end,
        )
        main, aux, stats = _derive_rois(
            diff_scores,
            edge_scores,
            scale_w=args.scale_w,
            scale_h=args.scale_h,
            cols=args.tile_cols,
            rows=args.tile_rows,
        )
        windows.append(
            WindowRoi(
                index=index,
                frame_start=frame_start,
                frame_end=frame_end,
                main_roi=main,
                aux_roi=aux,
                stats=stats,
            )
        )
        frame_start = frame_end
        index += 1

    metadata = RoiMetadata(
        video=video.name,
        source_width=meta["source_width"],
        source_height=meta["source_height"],
        scale_width=args.scale_w,
        scale_height=args.scale_h,
        fps=meta["fps"],
        total_frames=meta["total_frames"],
        sample_step=args.sample_step,
        window_frames=args.window_frames,
        tile_cols=args.tile_cols,
        tile_rows=args.tile_rows,
        windows=windows,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True))
    return 0


def emit_windows(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.metadata).read_text())
    for window in data["windows"]:
        aux = window.get("aux_roi") or {}
        fields = [
            f"{window['index']:03d}",
            str(window["frame_start"]),
            str(window["frame_end"]),
            str(window["main_roi"]["x"]),
            str(window["main_roi"]["y"]),
            str(window["main_roi"]["w"]),
            str(window["main_roi"]["h"]),
            str(aux.get("x", "")),
            str(aux.get("y", "")),
            str(aux.get("w", "")),
            str(aux.get("h", "")),
        ]
        print("|".join(fields))
    return 0


def emit_globals(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.metadata).read_text())
    print(
        "\t".join(
            [
                str(data["source_width"]),
                str(data["source_height"]),
                str(data["scale_width"]),
                str(data["scale_height"]),
            ]
        )
    )
    return 0


def _load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def _subprocess_run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _overlay_output_suffix(postfilter: str) -> str:
    return f",format=rgb24,{postfilter}" if postfilter else ",format=rgb24"


def _downscale_filter(
    scale_w: int,
    scale_h: int,
    downscale_flags: str,
    source_color_range: str,
    source_color_matrix: str,
    source_color_primaries: str,
    source_color_trc: str,
) -> str:
    return (
        f"scale={scale_w}:{scale_h}:flags={downscale_flags}:"
        f"in_range={source_color_range}:out_range={source_color_range}:"
        f"in_color_matrix={source_color_matrix}:out_color_matrix={source_color_matrix}:"
        f"in_primaries={source_color_primaries}:out_primaries={source_color_primaries}:"
        f"in_transfer={source_color_trc}:out_transfer={source_color_trc},"
        f"format=yuv420p"
    )


def _upscale_rgb_filter(
    width: int,
    height: int,
    upscale_flags: str,
    source_color_range: str,
    source_color_matrix: str,
    source_color_primaries: str,
    source_color_trc: str,
    rgb_output_range: str,
) -> str:
    return (
        f"scale={width}:{height}:flags={upscale_flags}:"
        f"in_range={source_color_range}:out_range={rgb_output_range}:"
        f"in_color_matrix={source_color_matrix}:"
        f"in_primaries={source_color_primaries}:"
        f"in_transfer={source_color_trc},"
        f"format=rgb24"
    )


def _codec_encode_args(args: argparse.Namespace, crf: int) -> list:
    """Return ffmpeg codec args for the given CRF, branching on args.codec."""
    if args.codec == 'libsvtav1':
        result = ['-c:v', 'libsvtav1', '-preset', str(args.svtav1_preset), '-crf', str(crf)]
        if args.svtav1_params:
            result += ['-svtav1-params', args.svtav1_params]
        return result
    # default: libx265
    result = ['-c:v', 'libx265', '-preset', args.preset, '-crf', str(crf)]
    if args.x265_params:
        result += ['-x265-params', args.x265_params]
    return result


def encode_metadata(args: argparse.Namespace) -> int:
    data = _load_metadata(Path(args.metadata))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    downscale_filter = _downscale_filter(
        args.scale_w,
        args.scale_h,
        args.downscale_flags,
        args.source_color_range,
        args.source_color_matrix,
        args.source_color_primaries,
        args.source_color_trc,
    )
    for window in data["windows"]:
        idx = f"{int(window['index']):03d}"
        main = window["main_roi"]
        aux = window.get("aux_roi")
        base_path = out_dir / f"base_{idx}.mkv"
        roi_path = out_dir / f"roi_{idx}.mkv"
        roi2_path = out_dir / f"roi2_{idx}.mkv"
        if aux and args.roi2_enable:
            print(f"Encoding dynamic ROI window {idx} {args.video} -> {base_path} + {roi_path} + {roi2_path}", flush=True)
            filter_complex = (
                f"[0:v]trim=start_frame={window['frame_start']}:end_frame={window['frame_end']},"
                f"setpts=PTS-STARTPTS,{downscale_filter},"
                f"split=3[base][crop1][crop2];"
                f"[crop1]crop={main['w']}:{main['h']}:{main['x']}:{main['y']}[roi1];"
                f"[crop2]crop={aux['w']}:{aux['h']}:{aux['x']}:{aux['y']}[roi2]"
            )
            cmd = [
                args.ffmpeg_bin, '-y', '-i', args.video,
                '-filter_complex', filter_complex,
                '-map', '[base]', '-an', *_codec_encode_args(args, args.base_crf), '-color_range', args.source_color_range, '-colorspace', args.source_color_matrix, '-color_primaries', args.source_color_primaries, '-color_trc', args.source_color_trc, str(base_path),
                '-map', '[roi1]', '-an', *_codec_encode_args(args, args.roi_crf), '-color_range', args.source_color_range, '-colorspace', args.source_color_matrix, '-color_primaries', args.source_color_primaries, '-color_trc', args.source_color_trc, str(roi_path),
                '-map', '[roi2]', '-an', *_codec_encode_args(args, args.roi2_crf), '-color_range', args.source_color_range, '-colorspace', args.source_color_matrix, '-color_primaries', args.source_color_primaries, '-color_trc', args.source_color_trc, str(roi2_path),
            ]
        else:
            print(f"Encoding dynamic ROI window {idx} {args.video} -> {base_path} + {roi_path}", flush=True)
            filter_complex = (
                f"[0:v]trim=start_frame={window['frame_start']}:end_frame={window['frame_end']},"
                f"setpts=PTS-STARTPTS,{downscale_filter},"
                f"split=2[base][crop];[crop]crop={main['w']}:{main['h']}:{main['x']}:{main['y']}[roi]"
            )
            cmd = [
                args.ffmpeg_bin, '-y', '-i', args.video,
                '-filter_complex', filter_complex,
                '-map', '[base]', '-an', *_codec_encode_args(args, args.base_crf), '-color_range', args.source_color_range, '-colorspace', args.source_color_matrix, '-color_primaries', args.source_color_primaries, '-color_trc', args.source_color_trc, str(base_path),
                '-map', '[roi]', '-an', *_codec_encode_args(args, args.roi_crf), '-color_range', args.source_color_range, '-colorspace', args.source_color_matrix, '-color_primaries', args.source_color_primaries, '-color_trc', args.source_color_trc, str(roi_path),
            ]
        _subprocess_run(cmd)
    return 0


def inflate_metadata(args: argparse.Namespace) -> int:
    data = _load_metadata(Path(args.metadata))
    archive_dir = Path(args.archive_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = archive_dir / 'inflate_tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_w = int(data['source_width'])
    source_h = int(data['source_height'])
    scale_w = int(data['scale_width'])
    scale_h = int(data['scale_height'])
    overlay_suffix = _overlay_output_suffix(args.postfilter)

    def scale_coord(coord: int, src: int, scaled: int, is_size: bool) -> int:
        value = round(coord * src / scaled)
        if is_size and value < 2:
            value = 2
        if value % 2:
            value -= 1
        if not is_size and value < 0:
            value = 0
        return value

    with out_path.open('wb') as out_f:
        for window in data['windows']:
            idx = f"{int(window['index']):03d}"
            main = window['main_roi']
            aux = window.get('aux_roi')
            base_path = archive_dir / f'base_{idx}.mkv'
            roi_path = archive_dir / f'roi_{idx}.mkv'
            roi2_path = archive_dir / f'roi2_{idx}.mkv'
            part_path = tmp_dir / f'part_{idx}.raw'
            main_x = scale_coord(int(main['x']), source_w, scale_w, False)
            main_y = scale_coord(int(main['y']), source_h, scale_h, False)
            main_w = scale_coord(int(main['w']), source_w, scale_w, True)
            main_h = scale_coord(int(main['h']), source_h, scale_h, True)
            if aux and roi2_path.exists() and args.roi2_enable:
                aux_x = scale_coord(int(aux['x']), source_w, scale_w, False)
                aux_y = scale_coord(int(aux['y']), source_h, scale_h, False)
                aux_w = scale_coord(int(aux['w']), source_w, scale_w, True)
                aux_h = scale_coord(int(aux['h']), source_h, scale_h, True)
                print(f"Inflating dynamic ROI window {idx} {base_path} + {roi_path} + {roi2_path} -> {part_path}", flush=True)
                filter_complex = (
                    f"[0:v]{_upscale_rgb_filter(source_w, source_h, args.upscale_flags, args.source_color_range, args.source_color_matrix, args.source_color_primaries, args.source_color_trc, args.rgb_output_range)}[base];"
                    f"[1:v]{_upscale_rgb_filter(main_w, main_h, args.upscale_flags, args.source_color_range, args.source_color_matrix, args.source_color_primaries, args.source_color_trc, args.rgb_output_range)}[roi1];"
                    f"[2:v]{_upscale_rgb_filter(aux_w, aux_h, args.upscale_flags, args.source_color_range, args.source_color_matrix, args.source_color_primaries, args.source_color_trc, args.rgb_output_range)}[roi2];"
                    f"[base][roi1]overlay={main_x}:{main_y}[tmp];"
                    f"[tmp][roi2]overlay={aux_x}:{aux_y}{overlay_suffix}[out]"
                )
                cmd = [args.ffmpeg_bin, '-y', '-i', str(base_path), '-i', str(roi_path), '-i', str(roi2_path), '-filter_complex', filter_complex, '-map', '[out]', '-an', '-sn', '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(part_path)]
            else:
                print(f"Inflating dynamic ROI window {idx} {base_path} + {roi_path} -> {part_path}", flush=True)
                filter_complex = (
                    f"[0:v]{_upscale_rgb_filter(source_w, source_h, args.upscale_flags, args.source_color_range, args.source_color_matrix, args.source_color_primaries, args.source_color_trc, args.rgb_output_range)}[base];"
                    f"[1:v]{_upscale_rgb_filter(main_w, main_h, args.upscale_flags, args.source_color_range, args.source_color_matrix, args.source_color_primaries, args.source_color_trc, args.rgb_output_range)}[roi];"
                    f"[base][roi]overlay={main_x}:{main_y}{overlay_suffix}[out]"
                )
                cmd = [args.ffmpeg_bin, '-y', '-i', str(base_path), '-i', str(roi_path), '-filter_complex', filter_complex, '-map', '[out]', '-an', '-sn', '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(part_path)]
            _subprocess_run(cmd)
            with part_path.open('rb') as part_f:
                out_f.write(part_f.read())
            part_path.unlink(missing_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze dashcam video for dynamic main ROI metadata")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze")
    p.add_argument("--video", required=True)
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("--source-color-range", default="tv")
    p.add_argument("--source-color-matrix", default="bt709")
    p.add_argument("--source-color-primaries", default="bt709")
    p.add_argument("--source-color-trc", default="bt709")
    p.add_argument("--scale-w", type=int, required=True)
    p.add_argument("--scale-h", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window-frames", type=int, default=200)
    p.add_argument("--sample-step", type=int, default=10)
    p.add_argument("--tile-cols", type=int, default=12)
    p.add_argument("--tile-rows", type=int, default=9)
    p.set_defaults(func=analyze)

    p = sub.add_parser("emit-windows")
    p.add_argument("--metadata", required=True)
    p.set_defaults(func=emit_windows)

    p = sub.add_parser("emit-globals")
    p.add_argument("--metadata", required=True)
    p.set_defaults(func=emit_globals)

    p = sub.add_parser("encode-metadata")
    p.add_argument('--video', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ffmpeg-bin', default='ffmpeg')
    p.add_argument('--scale-w', type=int, required=True)
    p.add_argument('--scale-h', type=int, required=True)
    p.add_argument('--downscale-flags', required=True)
    p.add_argument('--source-color-range', default='tv')
    p.add_argument('--source-color-matrix', default='bt709')
    p.add_argument('--source-color-primaries', default='bt709')
    p.add_argument('--source-color-trc', default='bt709')
    p.add_argument('--codec', required=True, choices=['libx265', 'libsvtav1'])
    p.add_argument('--preset', required=True)
    p.add_argument('--base-crf', type=int, required=True)
    p.add_argument('--roi-crf', type=int, required=True)
    p.add_argument('--roi2-crf', type=int, default=0)
    p.add_argument('--roi2-enable', action='store_true')
    p.add_argument('--x265-params', default=None)
    p.add_argument('--svtav1-preset', default=None)
    p.add_argument('--svtav1-crf', type=int, default=None)
    p.add_argument('--svtav1-params', default=None)
    p.set_defaults(func=encode_metadata)

    p = sub.add_parser("inflate-metadata")
    p.add_argument('--archive-dir', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ffmpeg-bin', default='ffmpeg')
    p.add_argument('--upscale-flags', required=True)
    p.add_argument('--source-color-range', default='tv')
    p.add_argument('--source-color-matrix', default='bt709')
    p.add_argument('--source-color-primaries', default='bt709')
    p.add_argument('--source-color-trc', default='bt709')
    p.add_argument('--rgb-output-range', default='pc')
    p.add_argument('--postfilter', default='')
    p.add_argument('--roi2-enable', action='store_true')
    p.set_defaults(func=inflate_metadata)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
