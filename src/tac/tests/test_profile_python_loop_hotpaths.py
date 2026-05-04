from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILER_PATH = REPO_ROOT / "experiments" / "profile_python_loop_hotpaths.py"


def _load_profiler():
    spec = importlib.util.spec_from_file_location("python_loop_hotpath_profiler_test", PROFILER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loop_hotpath_profile_is_non_promotable_and_ranks_nested_mask_loops(tmp_path: Path) -> None:
    profiler = _load_profiler()
    source = tmp_path / "experiments" / "planner.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            [
                "def slow(mask, height, width):",
                "    out = []",
                "    for frame in range(mask.shape[0]):",
                "        for y in range(height):",
                "            for x in range(width):",
                "                out.append(mask[frame, y, x])",
                "    return out",
                "",
            ]
        )
    )

    payload = profiler.build_profile(
        roots=[source.parent],
        output_json=tmp_path / "profile.json",
        limit=5,
    )

    assert json.loads((tmp_path / "profile.json").read_text()) == payload
    assert payload["schema"] == "python_loop_hotpath_profile_v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["top_hotpaths"][0]["static_hotpath_score"] > 0
    assert "mask" in payload["top_hotpaths"][0]["hot_keywords"]
    assert "vectorization" in payload["top_hotpaths"][0]["vectorization_recommendation"]


def test_loop_hotpath_profile_excludes_generated_results_and_site_packages(tmp_path: Path) -> None:
    profiler = _load_profiler()
    real_source = tmp_path / "experiments" / "real.py"
    generated = tmp_path / "experiments" / "results" / "run" / "site-packages" / "vendored.py"
    real_source.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    real_source.write_text("for frame in range(3):\n    pass\n")
    generated.write_text("for pixel in range(1000):\n    pass\n")

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        payload = profiler.build_profile(
            roots=[tmp_path / "experiments"],
            output_json=tmp_path / "profile.json",
            limit=20,
        )
    finally:
        os.chdir(old_cwd)

    paths = {Path(row["path"]).name for row in payload["top_hotpaths"]}
    assert "real.py" in paths
    assert "vendored.py" not in paths


def test_loop_hotpath_profile_prioritizes_active_builders_over_preflight_guards(tmp_path: Path) -> None:
    profiler = _load_profiler()
    active_builder = tmp_path / "experiments" / "plan_mask_atoms.py"
    preflight_guard = tmp_path / "src" / "tac" / "preflight.py"
    active_builder.parent.mkdir(parents=True)
    preflight_guard.parent.mkdir(parents=True)
    active_builder.write_text(
        "\n".join(
            [
                "def active(mask, height, width):",
                "    total = 0",
                "    for frame in range(mask.shape[0]):",
                "        for y in range(height):",
                "            for x in range(width):",
                "                total += int(mask[frame, y, x])",
                "    return total",
                "",
            ]
        )
    )
    preflight_guard.write_text(
        "\n".join(
            [
                "def guard(scan_dirs):",
                "    hits = []",
                "    for d in scan_dirs:",
                "        for sh in d.rglob('*.sh'):",
                "            for line in sh.read_text(errors='ignore').splitlines():",
                "                if 'mask frame pixel width height' in line:",
                "                    hits.append(line)",
                "    return hits",
                "",
            ]
        )
    )

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        payload = profiler.build_profile(
            roots=[tmp_path / "experiments", tmp_path / "src" / "tac"],
            output_json=tmp_path / "profile.json",
            limit=10,
        )
    finally:
        os.chdir(old_cwd)

    assert Path(payload["top_hotpaths"][0]["path"]).name == "plan_mask_atoms.py"
    guard_rows = [
        row
        for row in payload["top_hotpaths"]
        if Path(str(row["path"])).name == "preflight.py"
    ]
    assert guard_rows
    assert "guard/training infrastructure" in guard_rows[0]["vectorization_recommendation"]
