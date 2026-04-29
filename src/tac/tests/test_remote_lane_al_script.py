"""Lane AL — structural tests for scripts/remote_lane_al_analog_latent.sh.

Mirrors the Lane EC test pattern (Check 43 — every CLI flag emitted by
the lane script must exist in the target's argparse). Specifically:

  * Stage 0 NVDEC probe (Check 33).
  * Stage 1 anchor parity checks (Check 43).
  * Stage 2 invokes experiments/optimize_grayscale_canvas.py with REAL
    flags only.
  * Stage 3a invokes experiments/build_lane_al_archive.py with REAL flags.
  * Stage 3b runs contest_auth_eval [contest-CUDA] with the
    PYTHON_INFLATE=renderer_grayscale dispatch (Lane MM arm).
  * provenance.json + heartbeat written.
  * No git pull / git reset --hard (tarball-only parity per
    feedback_canonical_lane_lifecycle_DECISION_TREE_20260428).
"""
from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_al_analog_latent.sh"
OPTIMIZE = REPO / "experiments" / "optimize_grayscale_canvas.py"
BUILD = REPO / "experiments" / "build_lane_al_archive.py"
CONTEST_EVAL = REPO / "experiments" / "contest_auth_eval.py"
INFLATE_GRAYSCALE = (
    REPO / "submissions" / "robust_current" / "inflate_renderer_grayscale.py"
)


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def script_src() -> str:
    return SCRIPT.read_text()


def _argparse_flags(script_path: Path) -> set[str]:
    """Statically extract every `parser.add_argument('--flag', ...)` flag."""
    text = script_path.read_text()
    return set(re.findall(r"add_argument\(\s*['\"](--[\w-]+)['\"]", text))


def _strip_shell_comments(text: str) -> str:
    """Strip bash comment lines (so the regex can't false-match the
    documentation block at the top of the lane script)."""
    out = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#"):
            out.append(" " * (len(line) - 1) + ("\n" if line.endswith("\n") else ""))
        else:
            out.append(line)
    return "".join(out)


def _extract_invocation_flags(script_text: str, target_basename: str) -> set[str]:
    """Pull every `--flag` token from the lane script's invocation
    of a given target script."""
    body = _strip_shell_comments(script_text)
    invocations: list[str] = []
    pat = re.compile(
        rf"\S*{re.escape(target_basename)}(?P<args>(?:[^\n]+\\\n)*[^\n]*)",
        re.MULTILINE,
    )
    for m in pat.finditer(body):
        block = m.group("args")
        block = block.replace("\\\n", " ")
        invocations.append(block)
    flags: set[str] = set()
    for inv in invocations:
        flags.update(re.findall(r"(--[\w-]+)", inv))
    return flags


# ── basic structure ─────────────────────────────────────────────────────


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"Lane AL script must exist at {SCRIPT}"


def test_script_is_executable() -> None:
    """Check 30: remote scripts must be chmod +x."""
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "Lane AL script must be executable"


def test_script_has_set_e(script_src: str) -> None:
    """zip_dep_bootstrap_trap memory: must use `set -euo pipefail`."""
    assert "set -euo pipefail" in script_src, (
        "Lane AL script must use `set -euo pipefail` (CLAUDE.md "
        "non-negotiable; missing -e cascades silent failures)."
    )


def test_script_writes_provenance(script_src: str) -> None:
    """canonical_remote_bootstraps memory: provenance.json required."""
    assert "PROVENANCE=" in script_src
    assert "provenance.json" in script_src


def test_script_writes_heartbeat(script_src: str) -> None:
    assert "HEARTBEAT=" in script_src
    assert "heartbeat.log" in script_src


def test_script_uses_container_python(script_src: str) -> None:
    """Avoid bootstrap.sh / venv on remote (fragile vs container python)."""
    assert "/opt/conda/bin/python" in script_src


# ── tarball-only parity (Check 66) ──────────────────────────────────────


def test_script_does_not_git_pull(script_src: str) -> None:
    """Lane AL must NOT git-pull / reset (tarball is the parity)."""
    body = _strip_shell_comments(script_src)
    assert "git pull" not in body, (
        "git pull on remote nukes locally-anchored artifacts (memory: "
        "feedback_git_reset_nukes_anchors_20260429). Tarball deploy only."
    )
    assert "git reset --hard" not in body


# ── NVDEC probe (Check 33) ──────────────────────────────────────────────


def test_script_runs_nvdec_probe(script_src: str) -> None:
    """Check 33: every Vast.ai lane must probe NVDEC at Stage 0."""
    assert "scripts/probe_nvdec.sh" in script_src
    assert "--ensure-dali" in script_src


# ── anchor parity (Check 43) ────────────────────────────────────────────


def test_script_anchors_lane_a(script_src: str) -> None:
    """Lane AL anchors on Lane A's verified-1.15 archive."""
    assert "experiments/results/lane_a_landed/archive_lane_a.zip" in script_src


def test_script_checks_anchor_files_exist(script_src: str) -> None:
    """Stage 1 must verify required anchor files (fail-loud, not silent
    skip per CLAUDE.md)."""
    body = _strip_shell_comments(script_src)
    assert "[ -f " in body
    # masks/poses are inside archive — but the GT video + scorer weights
    # are referenced directly.
    assert "upstream/videos/0.mkv" in body
    assert "upstream/models/segnet.safetensors" in body
    assert "upstream/models/posenet.safetensors" in body


# ── flag wiring (NEVER invent CLI flags) ────────────────────────────────


def test_optimize_grayscale_flags_exist_in_target(script_src: str) -> None:
    """Every --flag passed to optimize_grayscale_canvas.py must exist
    in its argparse (CLAUDE.md NEVER invent CLI flags; memory:
    feedback_dead_flag_wiring_pattern)."""
    target_flags = _argparse_flags(OPTIMIZE)
    used_flags = _extract_invocation_flags(
        script_src, "optimize_grayscale_canvas.py",
    )
    invented = used_flags - target_flags
    assert not invented, (
        f"Lane AL invokes flags not in optimize_grayscale_canvas.py argparse: "
        f"{invented}. Target flags: {target_flags}"
    )


def test_build_lane_al_flags_exist_in_target(script_src: str) -> None:
    target_flags = _argparse_flags(BUILD)
    used_flags = _extract_invocation_flags(
        script_src, "build_lane_al_archive.py",
    )
    invented = used_flags - target_flags
    assert not invented, (
        f"Lane AL invokes flags not in build_lane_al_archive.py argparse: "
        f"{invented}. Target flags: {target_flags}"
    )


def test_contest_auth_eval_flags_exist_in_target(script_src: str) -> None:
    target_flags = _argparse_flags(CONTEST_EVAL)
    used_flags = _extract_invocation_flags(
        script_src, "contest_auth_eval.py",
    )
    invented = used_flags - target_flags
    assert not invented, (
        f"Lane AL invokes flags not in contest_auth_eval.py argparse: "
        f"{invented}. Target flags: {target_flags}"
    )


# ── inflate dispatch (Lane MM arm) ──────────────────────────────────────


def test_script_uses_renderer_grayscale_inflate_arm(script_src: str) -> None:
    """Lane AL reuses Lane MM's inflate path (PYTHON_INFLATE=renderer_grayscale)."""
    body = _strip_shell_comments(script_src)
    assert "PYTHON_INFLATE=renderer_grayscale" in body, (
        "Lane AL must dispatch through the grayscale-LUT inflate arm "
        "(Lane MM path). Without this the inflate.sh router falls back "
        "to the Lane A masks.mkv decoder, which expects class*63 encoding "
        "and will mis-decode the analog-latent grayscale.mkv."
    )


def test_script_runs_contest_auth_eval(script_src: str) -> None:
    """Auth eval EVERYWHERE — CLAUDE.md non-negotiable."""
    assert "experiments/contest_auth_eval.py" in script_src
    assert "RESULT_JSON" in script_src


# ── score lane tag ──────────────────────────────────────────────────────


def test_script_tags_score_as_contest_cuda(script_src: str) -> None:
    """Per check_scores_have_lane_tag — every score line must carry the
    [contest-CUDA] tag (or one of the canonical alternatives)."""
    body = _strip_shell_comments(script_src)
    assert "[contest-CUDA]" in body, (
        "Lane AL completion line must tag the score `[contest-CUDA]`."
    )


# ── predicted band ──────────────────────────────────────────────────────


def test_provenance_records_predicted_band(script_src: str) -> None:
    """Predicted band documented for council review (per memory:
    project_lane_g_v3_stacking_skunkworks_20260428 wedge attribution)."""
    assert "predicted_band" in script_src
    # 0.65 - 0.85 is the Lane AL predicted band (down -0.05 to -0.15
    # vs Lane MM 0.78 baseline).
    assert "0.65" in script_src
    assert "0.85" in script_src


# ── deterministic archive ───────────────────────────────────────────────


def test_build_lane_al_uses_deterministic_zip() -> None:
    """codex R5-r6 #5: archive builders must use ZipInfo with fixed
    timestamp so byte hash is reproducible."""
    src = BUILD.read_text()
    assert "ZipInfo" in src
    assert "1980, 1, 1" in src


# ── strict-scorer-rule compliance ───────────────────────────────────────


def test_inflate_path_does_not_load_scorers() -> None:
    """Lane AL is a COMPRESS-TIME tool; the inflate path
    (renderer_grayscale arm) must NOT load PoseNet/SegNet."""
    src = INFLATE_GRAYSCALE.read_text()
    forbidden = ["load_scorers", "load_differentiable_scorers", "PoseNet(", "SegNet("]
    for term in forbidden:
        assert term not in src, (
            f"inflate_renderer_grayscale.py contains forbidden term `{term}` "
            "(strict-scorer-rule: no scorer at inflate time)."
        )
