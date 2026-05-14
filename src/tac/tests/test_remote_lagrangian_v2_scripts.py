# SPDX-License-Identifier: MIT
"""Structural regression tests for the three Lane V2 Lagrangian-control scripts.

Pinned scripts (each replaces a hand-derived constant with a Lagrangian-controlled
parameter — see ``project_arbitrariness_audit_full_catalog_20260427`` Group A):

* ``scripts/remote_lane_g_v3_v2_lagrangian_snr.sh`` — replaces
  ``--kl-distill-weight 0.002`` with ``--kl-distill-snr-target 0.10``.
* ``scripts/remote_lane_omega_v3_rate_frontier.sh`` — replaces single
  ``target_bits=600,000`` with a 5-budget ε-constraint frontier sweep.
* ``scripts/remote_lane_s_v2_auto_warmup.sh`` — replaces hard-coded
  ``--self-compress-lambda-ramp-start-frac=0.3`` with the SAGA-style
  scorer-loss convergence detector.

Every script must satisfy the canonical CLAUDE.md non-negotiable structural
guards: ``set -euo pipefail``, NVDEC probe, anchor verification, dead-flag
scan against the target's argparse, ``--device cuda`` only, provenance JSON,
heartbeat log, archive built via ``zipfile.ZipFile`` (NOT shell zip),
``RESULT_JSON`` validation on the auth-eval log, ``[contest-CUDA]`` lane tag.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO / "scripts"

LANE_G_SCRIPT = SCRIPTS_DIR / "remote_lane_g_v3_v2_lagrangian_snr.sh"
LANE_OMEGA_V3_SCRIPT = SCRIPTS_DIR / "remote_lane_omega_v3_rate_frontier.sh"
LANE_S_V2_SCRIPT = SCRIPTS_DIR / "remote_lane_s_v2_auto_warmup.sh"

OPTIMIZE_POSES_PY = REPO / "experiments" / "optimize_poses.py"
SWEEP_OMEGA_PY = REPO / "experiments" / "sweep_omega_rate_frontier.py"
QAT_OMEGA_PY = REPO / "experiments" / "qat_omega_lagrangian.py"
TRAIN_RENDERER_PY = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


def _argparse_flags(py_path: Path) -> set[str]:
    src = py_path.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


def _used_flags_in(text: str, py_invocation: str) -> set[str]:
    """Return the set of '--flag' tokens in the script section after the
    given Python script invocation (handles both ``python <script>`` and
    ``$PYBIN -u <script>`` forms).

    The body is everything from the invocation until the next ``log "===``
    line, the next ``# Stage`` comment, the next ``[ -f`` validator
    (Lane G's pattern), or EOF. Multiple terminators because each script
    has a slightly different post-invocation structure.
    """
    # Use the LAST occurrence of the invocation in case the script
    # references it in a comment first (e.g., the dead-flag scan
    # mentions the path inside a Python -c block).
    occurrences = list(re.finditer(re.escape(py_invocation), text))
    if not occurrences:
        return set()
    # Pick the occurrence whose immediate context is a shell command
    # invocation (preceded by `$PYBIN -u ` or `python `) — not a comment
    # nor a string. Prefer occurrences that come AFTER a `-u ` token in
    # the same line.
    chosen = None
    for occ in occurrences:
        line_start = text.rfind("\n", 0, occ.start()) + 1
        line_prefix = text[line_start:occ.start()]
        if "-u " in line_prefix or "python " in line_prefix.lower():
            chosen = occ
    if chosen is None:
        chosen = occurrences[-1]
    body = text[chosen.end():]
    # Stop at the first terminator.
    for term_re in (
        r"\n\s*\[\s*-f",
        r"\nlog \"===",
        r"\n# Stage",
        r"\n\s*BEST_FP32=",
        r"\Z",
    ):
        m = re.search(term_re, body)
        if m:
            body = body[:m.start()]
            break
    return set(re.findall(r"\B--([a-z][a-z0-9-]+)", body))


# ────────────────────────────────────────────────────────────────────────
# Shared structural guards — parametrised across all 3 scripts
# ────────────────────────────────────────────────────────────────────────

ALL_SCRIPTS = pytest.mark.parametrize(
    "script_path",
    [LANE_G_SCRIPT, LANE_OMEGA_V3_SCRIPT, LANE_S_V2_SCRIPT],
    ids=["lane_g_v3_v2", "lane_omega_v3", "lane_s_v2"],
)


@ALL_SCRIPTS
def test_script_exists(script_path: Path):
    assert script_path.exists(), f"missing script: {script_path}"


@ALL_SCRIPTS
def test_script_is_executable(script_path: Path):
    assert os.access(script_path, os.X_OK), f"{script_path} not chmod +x"


@ALL_SCRIPTS
def test_set_euo_pipefail(script_path: Path):
    """`set -euo pipefail` is mandatory per CLAUDE.md (LANE-B 6.5h trap)."""
    assert "set -euo pipefail" in script_path.read_text(), (
        f"{script_path}: missing `set -euo pipefail` (FORBIDDEN PATTERNS)"
    )


@ALL_SCRIPTS
def test_nvdec_probe_present(script_path: Path):
    """Stage 0 NVDEC probe required (memory: feedback_vastai_nvdec_host_variation)."""
    txt = script_path.read_text()
    assert "probe_nvdec.sh" in txt, (
        f"{script_path}: must invoke probe_nvdec.sh in Stage 0"
    )


@ALL_SCRIPTS
def test_nvdec_probe_aborts_on_failure(script_path: Path):
    """NVDEC probe must abort with `exit N` on failure."""
    section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_path.read_text(), re.DOTALL,
    )
    assert section is not None, (
        f"{script_path}: NVDEC probe must abort with exit N on failure"
    )


@ALL_SCRIPTS
def test_device_cuda_required_no_mps(script_path: Path):
    """Every Python invocation must pass --device cuda. No MPS / no CPU
    fallback (CLAUDE.md FORBIDDEN PATTERNS — MPS-CUDA drift = 23x)."""
    txt = script_path.read_text()
    code = "\n".join(
        ln for ln in txt.splitlines() if not ln.strip().startswith("#")
    )
    assert "--device cuda" in code, f"{script_path}: missing --device cuda"
    assert not re.search(r"--device\s+mps\b", code), (
        f"{script_path}: --device mps is FORBIDDEN"
    )
    assert not re.search(r"--device\s+cpu\b", code), (
        f"{script_path}: --device cpu is FORBIDDEN"
    )
    bad_patterns = [
        r"device\s*=\s*[\"']mps[\"']",
        r"\.to\(\s*[\"']mps[\"']",
    ]
    for pat in bad_patterns:
        assert not re.search(pat, code), (
            f"{script_path}: forbidden MPS pattern {pat!r}"
        )


@ALL_SCRIPTS
def test_writes_provenance_json(script_path: Path):
    txt = script_path.read_text()
    assert "provenance.json" in txt or "PROVENANCE=" in txt, (
        f"{script_path}: missing provenance.json (canonical pipeline)"
    )


@ALL_SCRIPTS
def test_writes_heartbeat_log(script_path: Path):
    txt = script_path.read_text()
    assert "heartbeat.log" in txt or "HEARTBEAT=" in txt, (
        f"{script_path}: missing heartbeat.log"
    )


@ALL_SCRIPTS
def test_provenance_records_predicted_band(script_path: Path):
    txt = script_path.read_text()
    assert "predicted_band" in txt, (
        f"{script_path}: provenance must record predicted_band"
    )


@ALL_SCRIPTS
def test_provenance_records_lagrangian_target(script_path: Path):
    """V2 provenance MUST record the lagrangian_target (the controlled
    quantity) so the dual-ascent semantics are auditable."""
    txt = script_path.read_text()
    assert "lagrangian_target" in txt, (
        f"{script_path}: provenance must record lagrangian_target field"
    )


@ALL_SCRIPTS
def test_no_shell_zip_binary(script_path: Path):
    """PyTorch container has no `zip` shell binary
    (memory: feedback_zip_dep_bootstrap_trap)."""
    code = "\n".join(
        ln for ln in script_path.read_text().splitlines()
        if not ln.strip().startswith("#")
    )
    bad = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad, (
        f"{script_path}: shell `zip` binary is FORBIDDEN; "
        f"match: {bad.group(0) if bad else None!r}"
    )


@ALL_SCRIPTS
def test_uses_python_zipfile_for_archive(script_path: Path):
    """Archives must be built via `zipfile.ZipFile`, not shell zip."""
    txt = script_path.read_text()
    # Lane Ω-V3 builds many archives in a loop; Lane G & S build one each.
    assert "zipfile.ZipFile" in txt, (
        f"{script_path}: must build archives via zipfile.ZipFile"
    )


@ALL_SCRIPTS
def test_runs_contest_auth_eval(script_path: Path):
    txt = script_path.read_text()
    assert "contest_auth_eval.py" in txt, (
        f"{script_path}: must run contest_auth_eval.py"
    )


@ALL_SCRIPTS
def test_auth_eval_result_validated(script_path: Path):
    """RESULT_JSON guard catches silent auth_eval crashes
    (LANE-B 2026-04-26 cascade pattern)."""
    txt = script_path.read_text()
    assert "RESULT_JSON" in txt, (
        f"{script_path}: must guard auth_eval log with RESULT_JSON grep"
    )


@ALL_SCRIPTS
def test_lane_tag_contest_cuda(script_path: Path):
    """Every score-emitting completion log must carry [contest-CUDA] tag
    (CLAUDE.md non-negotiable: every score must have a lane tag)."""
    assert "[contest-CUDA]" in script_path.read_text(), (
        f"{script_path}: completion log must tag [contest-CUDA]"
    )


@ALL_SCRIPTS
def test_anchors_on_lane_a(script_path: Path):
    """Every V2 script anchors on Lane A's verified 1.15 [contest-CUDA]."""
    assert "experiments/results/lane_a_landed" in script_path.read_text()


@ALL_SCRIPTS
def test_dead_flag_scan_present(script_path: Path):
    """Each script must run a runtime preflight that grep's argparse."""
    txt = script_path.read_text()
    assert "INVENTED FLAGS" in txt, (
        f"{script_path}: must include a runtime dead-flag preflight"
    )


# ────────────────────────────────────────────────────────────────────────
# Per-script assertions — Lane G V3-V2
# ────────────────────────────────────────────────────────────────────────


def test_lane_g_v3_v2_passes_snr_target_flag():
    txt = LANE_G_SCRIPT.read_text()
    assert "--kl-distill-snr-target" in txt, (
        "Lane G V3-V2 must pass --kl-distill-snr-target to optimize_poses"
    )
    assert "--kl-distill-snr-eta" in txt, (
        "Lane G V3-V2 must pass --kl-distill-snr-eta to optimize_poses"
    )
    assert "0.10" in txt, (
        "Lane G V3-V2 SNR target should be 0.10 (Hinton 2015 auxiliary regime)"
    )


def test_lane_g_v3_v2_flags_real_in_optimize_poses_argparse():
    real = _argparse_flags(OPTIMIZE_POSES_PY)
    used = _used_flags_in(LANE_G_SCRIPT.read_text(), "experiments/optimize_poses.py")
    invented = used - real
    assert not invented, (
        f"Lane G V3-V2 invents flags not in optimize_poses argparse: "
        f"{sorted(invented)}"
    )
    # Confirm critical flags present.
    for f in ("kl-distill-snr-target", "kl-distill-snr-eta",
              "kl-distill-weight", "eval-roundtrip", "device"):
        assert f in used, (
            f"Lane G V3-V2 must pass --{f} to optimize_poses; "
            f"used={sorted(used)}"
        )


def test_lane_g_v3_v2_provenance_records_snr_target():
    txt = LANE_G_SCRIPT.read_text()
    assert "kl_distill_snr_target" in txt
    # SNR target should be 0.10 in provenance
    assert re.search(r"kl_distill_snr_target['\"]?\s*:\s*0\.10", txt), (
        "Lane G V3-V2 provenance must record kl_distill_snr_target=0.10"
    )


# ────────────────────────────────────────────────────────────────────────
# Per-script assertions — Lane Ω-V3
# ────────────────────────────────────────────────────────────────────────


def test_lane_omega_v3_uses_sweep_orchestrator():
    txt = LANE_OMEGA_V3_SCRIPT.read_text()
    assert "sweep_omega_rate_frontier.py" in txt, (
        "Lane Ω-V3 must invoke experiments/sweep_omega_rate_frontier.py"
    )


def test_lane_omega_v3_sweeps_multiple_budgets():
    txt = LANE_OMEGA_V3_SCRIPT.read_text()
    # Must pass --target-bits-per-weight with comma-separated list
    assert "--target-bits-per-weight" in txt, (
        "Lane Ω-V3 must pass --target-bits-per-weight to the sweep tool"
    )
    # The sweep should cover at least 4 budgets.
    m = re.search(r"--target-bits-per-weight\s+\"?([0-9.,]+)\"?", txt)
    assert m, "could not parse --target-bits-per-weight value"
    budgets = [float(t) for t in m.group(1).split(",") if t.strip()]
    assert len(budgets) >= 4, (
        f"Lane Ω-V3 frontier must sweep ≥ 4 budgets (got {budgets})"
    )


def test_lane_omega_v3_flags_real_in_sweep_argparse():
    real = _argparse_flags(SWEEP_OMEGA_PY)
    used = _used_flags_in(
        LANE_OMEGA_V3_SCRIPT.read_text(),
        "experiments/sweep_omega_rate_frontier.py",
    )
    invented = used - real
    assert not invented, (
        f"Lane Ω-V3 invents sweep flags: {sorted(invented)}"
    )


def test_lane_omega_v3_provenance_records_budgets():
    txt = LANE_OMEGA_V3_SCRIPT.read_text()
    assert "budgets_bits_per_weight" in txt, (
        "Lane Ω-V3 provenance must record the budget list"
    )


def test_lane_omega_v3_writes_scores_csv():
    """The sweep loop must aggregate per-budget auth scores into a CSV."""
    txt = LANE_OMEGA_V3_SCRIPT.read_text()
    assert "scores.csv" in txt or "frontier.csv" in txt, (
        "Lane Ω-V3 must write a per-budget scores aggregate CSV"
    )


# ────────────────────────────────────────────────────────────────────────
# Per-script assertions — Lane S V2
# ────────────────────────────────────────────────────────────────────────


def test_lane_s_v2_passes_auto_warmup_flag():
    txt = LANE_S_V2_SCRIPT.read_text()
    assert "--auto-warmup-lambda" in txt, (
        "Lane S V2 must pass --auto-warmup-lambda to train_renderer"
    )


def test_lane_s_v2_passes_auto_warmup_tuning_flags():
    txt = LANE_S_V2_SCRIPT.read_text()
    for f in ("--auto-warmup-window", "--auto-warmup-slope-tol",
              "--auto-warmup-min-epochs"):
        assert f in txt, (
            f"Lane S V2 must pass {f} to train_renderer (the detector "
            f"hyperparameters MUST be auditable)"
        )


def test_lane_s_v2_flags_real_in_train_renderer_argparse():
    real = _argparse_flags(TRAIN_RENDERER_PY)
    used = _used_flags_in(
        LANE_S_V2_SCRIPT.read_text(), "src/tac/experiments/train_renderer.py",
    )
    invented = used - real
    assert not invented, (
        f"Lane S V2 invents train_renderer flags: {sorted(invented)}"
    )
    # The detector flags MUST be in real argparse.
    for f in ("auto-warmup-lambda", "auto-warmup-window",
              "auto-warmup-slope-tol", "auto-warmup-min-epochs"):
        assert f in real, (
            f"--{f} not declared in train_renderer.py argparse — "
            f"add the flag before deploying Lane S V2"
        )


def test_lane_s_v2_provenance_records_detector_config():
    txt = LANE_S_V2_SCRIPT.read_text()
    for k in ("auto_warmup_window", "auto_warmup_slope_tol",
              "auto_warmup_min_epochs"):
        assert k in txt, (
            f"Lane S V2 provenance must record detector key {k}"
        )


def test_lane_s_v2_keeps_static_fallback_for_safety():
    """Even in V2, the static --self-compress-lambda-ramp-start-frac is
    passed as a fallback (the detector falls back to it if it never
    fires before total_epochs)."""
    txt = LANE_S_V2_SCRIPT.read_text()
    assert "--self-compress-lambda-ramp-start-frac" in txt, (
        "Lane S V2 must keep --self-compress-lambda-ramp-start-frac as "
        "a fallback when the convergence detector never fires"
    )
