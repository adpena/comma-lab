#!/usr/bin/env python3
"""Stitch the level-set witness scenes into ONE continuous film.

Designed to keep working as the scenes are updated + polished + added to:

  * DISCOVERS scenes automatically — any ``sceneNN_*.py`` with a ``Scene`` /
    ``ThreeDScene`` subclass is picked up and ordered by its ``NN`` prefix, so a
    new ``scene04_*.py`` joins the film with no edit here.
  * RENDERS each through the SAME isolated manim venv (``.venv/bin/manim``) with
    the LaTeX PATH, at a chosen quality — so a scene with no clip at that quality
    is generated on the spot. Manim's own frame cache makes unchanged scenes
    fast, so the film always reflects the latest code by construction. Pass
    ``--no-render`` to skip manim and stitch only what is already rendered.
  * CONCATENATES robustly — every clip is normalized to the target
    resolution / fps / pixel-format first, so a scene left at a different quality
    can never corrupt the join. Hard cuts by default (each scene already fades
    through black at its boundaries); optional cross-fade via ``--transition``.

Everything is stdlib-only (subprocess / pathlib / re / argparse); it shells out
to the venv ``manim`` and to ``ffmpeg`` / ``ffprobe``.

Usage
-----
    # full film at 1080p60 (renders any missing/stale scenes, then stitches):
    ./stitch_film.py

    # fast preview at 720p30:
    ./stitch_film.py -q m

    # just re-stitch already-rendered clips (no manim), with 0.4s cross-fades:
    ./stitch_film.py --no-render --transition 0.4

    # a subset, force a clean re-render:
    ./stitch_film.py --scenes 00,01 --force

Output: ``media/levelset_witness_film[_<quality>].mp4`` (+ a sibling ``.json``
manifest of what was stitched, in order, with per-scene durations).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENV_MANIM = _HERE / ".venv" / "bin" / "manim"
_PREP = _HERE / "scenes" / "_prep_hardest_frame.py"
_MAIN_VENV_PY = _HERE.parents[1] / ".venv" / "bin" / "python"   # main repo venv (for prep)
_TEXBIN = "/Library/TeX/texbin"

# manim quality flag -> (output-subdir, width, height, fps)
_QUALITY = {
    "l": ("480p15", 854, 480, 15),
    "m": ("720p30", 1280, 720, 30),
    "h": ("1080p60", 1920, 1080, 60),
    "k": ("2160p60", 3840, 2160, 60),
}

# scene files look like scene00_witness.py, scene12_foo_bar.py, ...
_SCENE_RE = re.compile(r"^scene(\d\d)_.*\.py$")
# a renderable class: `class Name(Base1, Base2):` where some base ends in "Scene"
_CLASS_RE = re.compile(r"^class\s+(\w+)\s*\(([^)]*)\)\s*:", re.MULTILINE)


@dataclass
class SceneEntry:
    order: str            # the two-digit prefix, e.g. "00"
    path: Path            # scenes/sceneNN_*.py
    stem: str             # module stem, e.g. "scene00_witness"
    cls: str              # the Scene subclass name, e.g. "Witness"
    clip: Path | None = None       # rendered mp4 (filled in during render/locate)
    duration: float = 0.0
    note: str = field(default="")


def _tool(name: str) -> str:
    """Locate an external tool (ffmpeg/ffprobe), honoring common Homebrew paths."""
    found = shutil.which(name)
    if found:
        return found
    for cand in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", f"/usr/bin/{name}"):
        if Path(cand).exists():
            return cand
    raise SystemExit(f"error: '{name}' not found on PATH — install ffmpeg (brew install ffmpeg)")


def _env_with_tex() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{_TEXBIN}:{env.get('PATH', '')}"
    return env


def discover_scenes(select: set[str] | None) -> list[SceneEntry]:
    """Find every sceneNN_*.py with a Scene subclass, ordered by NN prefix."""
    entries: list[SceneEntry] = []
    for path in sorted((_HERE / "scenes").glob("scene*.py")):
        m = _SCENE_RE.match(path.name)
        if not m:
            continue
        order = m.group(1)
        if select is not None and order not in select:
            continue
        cls = _scene_class(path)
        if cls is None:
            print(f"  ! {path.name}: no Scene subclass found — skipping", file=sys.stderr)
            continue
        entries.append(SceneEntry(order=order, path=path, stem=path.stem, cls=cls))
    entries.sort(key=lambda e: e.order)
    return entries


def _scene_class(path: Path) -> str | None:
    """Return the name of the (first) class whose bases include a *Scene type."""
    text = path.read_text()
    for name, bases in _CLASS_RE.findall(text):
        if any(b.strip().endswith("Scene") for b in bases.split(",")):
            return name
    return None


def _probe_duration(clip: Path, ffprobe: str) -> float:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def ensure_assets() -> None:
    """Scenes read from assets/ (npy/png stacks). Regenerate if missing."""
    assets = _HERE / "assets"
    needed = ["ego_clip.npy", "ego_sep_stack.npy", "meta.json", "hardest_argmax.png"]
    if assets.is_dir() and all((assets / n).exists() for n in needed):
        return
    if not _PREP.exists():
        print("  ! assets missing and no prep script — scenes may fail", file=sys.stderr)
        return
    py = str(_MAIN_VENV_PY) if _MAIN_VENV_PY.exists() else sys.executable
    print("  · assets missing — running prep (reads the gt cache once)…")
    subprocess.run([py, str(_PREP)], cwd=str(_HERE), check=True)


def render_scene(e: SceneEntry, quality: str) -> None:
    if not _VENV_MANIM.exists():
        raise SystemExit(f"error: manim venv not found at {_VENV_MANIM} — see README setup")
    print(f"  · render {e.order} {e.cls}  ({e.stem}) …")
    subprocess.run(
        [str(_VENV_MANIM), f"-q{quality}", str(e.path.relative_to(_HERE)), e.cls],
        cwd=str(_HERE), env=_env_with_tex(), check=True)


def locate_clip(e: SceneEntry, quality: str) -> tuple[Path | None, str | None]:
    """Find the rendered clip for this scene.

    Prefer the requested quality; if this scene was only rendered at another
    quality (e.g. --no-render mode, or a scene you rendered at 720p while the
    film is 1080p), fall back to the best available so nothing is silently
    dropped. Returns (clip_path, quality_key) or (None, None).
    """
    exact = _HERE / "media" / "videos" / e.stem / _QUALITY[quality][0] / f"{e.cls}.mp4"
    if exact.exists():
        return exact, quality
    for q in ("k", "h", "m", "l"):                     # highest-res available
        if q == quality:
            continue
        cand = _HERE / "media" / "videos" / e.stem / _QUALITY[q][0] / f"{e.cls}.mp4"
        if cand.exists():
            return cand, q
    return None, None


def normalize(clip: Path, dst: Path, quality: str, ffmpeg: str) -> None:
    """Re-encode to the exact target geometry so concat/xfade can never mismatch."""
    _, w, h, fps = _QUALITY[quality]
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"fps={fps},format=yuv420p,setsar=1")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip), "-vf", vf,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an", str(dst)],
        check=True)


def concat_hardcut(norm_clips: list[Path], out: Path, ffmpeg: str, tmp: Path) -> None:
    listing = tmp / "concat_list.txt"
    listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in norm_clips))
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)], check=True)


def concat_xfade(norm_clips: list[Path], durations: list[float], out: Path,
                 quality: str, transition: float, ffmpeg: str) -> None:
    """Chain xfade filters with running offsets (each cut overlaps by `transition`)."""
    fps = _QUALITY[quality][3]
    inputs: list[str] = []
    for c in norm_clips:
        inputs += ["-i", str(c)]
    parts: list[str] = []
    prev = "[0:v]"
    offset = durations[0] - transition
    for i in range(1, len(norm_clips)):
        label = f"[vx{i}]"
        parts.append(
            f"{prev}[{i}:v]xfade=transition=fade:duration={transition}:"
            f"offset={max(offset, 0):.3f}{label}")
        prev = label
        offset += durations[i] - transition
    filtergraph = ";".join(parts)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", *inputs,
         "-filter_complex", filtergraph, "-map", prev,
         "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--quality", choices=list(_QUALITY), default="h",
                    help="manim quality: l=480p15 m=720p30 h=1080p60 k=2160p60 (default h)")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated NN prefixes to include (default: all discovered)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip manim; stitch already-rendered clips only")
    ap.add_argument("--force", action="store_true",
                    help="disable manim's frame cache for a clean re-render")
    ap.add_argument("--transition", type=float, default=0.0,
                    help="cross-fade seconds between scenes (0 = hard cut, the default)")
    ap.add_argument("--prep", action="store_true",
                    help="force-regenerate assets/ before rendering")
    ap.add_argument("-o", "--output", default=None,
                    help="output path (default media/levelset_witness_film[_q].mp4)")
    args = ap.parse_args()

    ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
    select = {s.strip().zfill(2) for s in args.scenes.split(",")} if args.scenes else None

    scenes = discover_scenes(select)
    if not scenes:
        print("no scenes discovered (looked for sceneNN_*.py with a Scene subclass)",
              file=sys.stderr)
        return 1
    print(f"discovered {len(scenes)} scene(s): "
          + ", ".join(f"{e.order}:{e.cls}" for e in scenes))

    if args.prep:
        (_HERE / "assets").mkdir(exist_ok=True)
        subprocess.run([str(_MAIN_VENV_PY) if _MAIN_VENV_PY.exists() else sys.executable,
                        str(_PREP)], cwd=str(_HERE), check=True)
    elif not args.no_render:
        ensure_assets()

    # 1) render (generates any missing scene at the requested quality) then locate
    if args.force:
        os.environ["MANIM_DISABLE_CACHING"] = "1"     # manim honors this env
    for e in scenes:
        if not args.no_render:
            render_scene(e, args.quality)
        e.clip, found_q = locate_clip(e, args.quality)
        if e.clip is None:
            e.note = "no render found at any quality — excluded from film"
            print(f"  ! {e.order} {e.cls}: {e.note}", file=sys.stderr)
        else:
            e.duration = _probe_duration(e.clip, ffprobe)
            if found_q != args.quality:
                e.note = f"using {_QUALITY[found_q][0]} (no {_QUALITY[args.quality][0]} render found)"
                print(f"  · {e.order} {e.cls}: {e.note}")

    usable = [e for e in scenes if e.clip is not None]
    if not usable:
        print("no rendered clips found to stitch", file=sys.stderr)
        return 1

    # 2) normalize every clip to identical geometry (concat/xfade safety)
    tmp = _HERE / "media" / ".stitch_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    norm: list[Path] = []
    for e in usable:
        dst = tmp / f"norm_{e.order}.mp4"
        print(f"  · normalize {e.order} {e.cls}  ({e.duration:.1f}s)")
        normalize(e.clip, dst, args.quality, ffmpeg)
        norm.append(dst)

    # 3) concatenate
    out = Path(args.output) if args.output else (
        _HERE / "media" / (f"levelset_witness_film"
                           + ("" if args.quality == "h" else f"_{args.quality}") + ".mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.transition and args.transition > 0 and len(norm) > 1:
        print(f"  · concat with {args.transition}s cross-fades")
        concat_xfade(norm, [e.duration for e in usable], out, args.quality,
                     args.transition, ffmpeg)
    else:
        print("  · concat (hard cuts)")
        concat_hardcut(norm, out, ffmpeg, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    total = _probe_duration(out, ffprobe)
    manifest = {
        "output": str(out),
        "quality": args.quality,
        "transition_s": args.transition,
        "total_duration_s": round(total, 2),
        "scenes": [{"order": e.order, "class": e.cls, "file": e.path.name,
                    "duration_s": round(e.duration, 2),
                    "included": e.clip is not None, "note": e.note} for e in scenes],
    }
    (out.with_suffix(".json")).write_text(json.dumps(manifest, indent=2))

    print(f"\n✓ film: {out}  ({total:.1f}s, {len(usable)} scenes, {args.quality})")
    if any(e.clip is None for e in scenes):
        print("  (note: some scenes were missing — see the .json manifest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
