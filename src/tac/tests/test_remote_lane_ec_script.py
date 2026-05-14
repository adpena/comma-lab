# SPDX-License-Identifier: MIT
"""Lane EC: structural tests for scripts/remote_lane_ec_engineered_corrections.sh.

Lane-EC-specific properties:
  - Stage 0 NVDEC probe (check 33).
  - Stage 1 stages Lane A baseline artifacts (renderer + masks + poses).
  - Stage 2 invokes engineered_quant_noise.py with REAL flags only
    (NEVER invent CLI flags — verified against the target's argparse).
  - Stage 3 builds the new archive containing gradient_corrections.bin
    via Python `zipfile.ZipFile` (NOT shell `zip`), with deterministic
    timestamps (codex R5-r6 #5).
  - Stage 4 runs contest_auth_eval [contest-CUDA].
  - Predicted band [0.85, 1.15] documented in provenance.
  - Strict-scorer-rule preserved: corrections.bin is data, not model;
    no scorer is loaded at inflate time.

Plus all standard CLAUDE.md non-negotiable checks (set -euo pipefail,
provenance, heartbeat, --device cuda, no MPS/CPU fallback, container
Python).
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_ec_engineered_corrections.sh"
ENGINEERED = REPO / "experiments" / "engineered_quant_noise.py"
CONTEST_EVAL = REPO / "experiments" / "contest_auth_eval.py"
INFLATE_RENDERER = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def script_src() -> str:
    return SCRIPT.read_text()


def _argparse_flags(script_path: Path) -> set[str]:
    """Statically extract every `parser.add_argument('--flag', ...)` flag
    name from a target script. Avoids importing the target (which would
    drag in CUDA + scorer deps). Mirrors the Lane H / Lane SI test
    pattern that prevents the dead-flag-wiring bug class.
    """
    text = script_path.read_text()
    return set(re.findall(r"add_argument\(\s*['\"](--[\w-]+)['\"]", text))


def _strip_shell_comments(text: str) -> str:
    """Replace bash `#`-comment lines with same-length whitespace so the
    invocation-extraction regex can't false-match against the header
    docstring (which mentions every CLI flag for documentation).
    """
    out = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#"):
            # preserve newline + indices
            out.append(" " * (len(line) - 1) + ("\n" if line.endswith("\n") else ""))
        else:
            out.append(line)
    return "".join(out)


# ── basic structural ──────────────────────────────────────────────────────


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"Lane EC script must exist at {SCRIPT}"


def test_script_is_executable() -> None:
    """Check 30: remote scripts must be chmod +x."""
    mode = os.stat(SCRIPT).st_mode
    assert mode & stat.S_IXUSR, "Lane EC script must be chmod +x"


def test_set_euo_pipefail_present(script_src: str) -> None:
    """Memory feedback_zip_dep_bootstrap_trap: `set -uo pipefail` (no -e)
    silently cascades empty captures. Lane EC must use the full triple.
    """
    assert "set -euo pipefail" in script_src, (
        "Lane EC must declare `set -euo pipefail` per CLAUDE.md "
        "FORBIDDEN PATTERN (zip_dep_bootstrap_trap)."
    )
    bad_lines = [ln for ln in script_src.splitlines() if re.search(r"^\s*set\s+-uo\s", ln)]
    assert not bad_lines, f"Forbidden `set -uo` (missing -e): {bad_lines}"


# ── Stage 0: NVDEC probe (check 33) ───────────────────────────────────────


def test_nvdec_probe_stage_0(script_src: str) -> None:
    """check_remote_scripts_probe_nvdec_early: the probe MUST be Stage 0.
    7/12 4090 hosts had compute-CUDA but missing NVDEC; probe in 5s
    saves ~$3-4 per bad host.
    """
    assert "probe_nvdec.sh" in script_src
    # Verify it's actually called (not just commented), with --ensure-dali
    # so a fresh container that hasn't run setup_full still gets DALI.
    assert "--ensure-dali" in script_src, (
        "Lane EC must call probe_nvdec.sh with --ensure-dali so a fresh "
        "container without DALI auto-installs it (codex R5-3-round-4)."
    )


def test_nvdec_probe_appears_before_engineered_quant_noise(script_src: str) -> None:
    """The probe must run BEFORE the heavy compute step (engineered
    correction search). Otherwise we waste 5-30 min on a host that will
    fail eval at the end. Strip comment lines so the docstring header's
    mention of `engineered_quant_noise.py` doesn't false-positive.
    """
    code_only_lines = []
    for line in script_src.splitlines():
        if line.lstrip().startswith("#"):
            code_only_lines.append(" " * len(line))  # preserve indices
        else:
            code_only_lines.append(line)
    code = "\n".join(code_only_lines)
    probe_idx = code.index("probe_nvdec.sh")
    eng_idx = code.index("engineered_quant_noise.py")
    assert probe_idx < eng_idx, (
        "NVDEC probe must precede engineered_quant_noise — check 33."
    )


# ── provenance + heartbeat (canonical_remote_bootstraps) ──────────────────


def test_provenance_json_written(script_src: str) -> None:
    assert "provenance.json" in script_src
    assert "git_hash" in script_src or "GIT_HASH" in script_src
    assert "gpu_name" in script_src.lower() or "GPU_NAME" in script_src


def test_heartbeat_log_written(script_src: str) -> None:
    assert "heartbeat" in script_src.lower(), (
        "Lane EC must emit a heartbeat (memory feedback_canonical_remote_bootstraps)."
    )


def test_provenance_records_predicted_band(script_src: str) -> None:
    """check_remote_scripts_record_predicted_band (check 31).

    Lane EC band: [0.85, 1.15].
      Floor 0.85: best-case SegNet reduction 0.0046 → 0.001 (~0.34 score
                  improvement) offset by +0.001 rate cost from 50KB
                  corrections.bin.
      Ceiling 1.15: zero net change (corrections.bin too large to fit
                    budget OR renderer already at SegNet floor for the
                    ±max-delta=2 perturbations the search can find).
    """
    assert "predicted_band" in script_src, (
        "provenance must record predicted_band — preflight check 31."
    )
    assert "[0.85, 1.15]" in script_src, (
        "Lane EC predicted band must be [0.85, 1.15] (Lane A 1.15 baseline "
        "minus expected SegNet improvement plus corrections.bin rate cost)."
    )


def test_completion_tagged_contest_cuda(script_src: str) -> None:
    """check_remote_scripts_tag_contest_cuda_at_completion (check 32):
    LANE_X_DONE log line must include `[contest-CUDA]` so the score is
    self-tagging per CLAUDE.md score-tag rule.
    """
    assert "[contest-CUDA]" in script_src, (
        "Lane EC completion line must carry [contest-CUDA] tag."
    )
    assert "LANE_EC_DONE" in script_src, (
        "Completion marker must be LANE_EC_DONE for grep-tail consumers."
    )


# ── platform discipline ──────────────────────────────────────────────────


def test_python_zipfile_not_shell_zip(script_src: str) -> None:
    """check_no_shell_zip_binary: PyTorch container has no `zip` binary
    (memory feedback_zip_dep_bootstrap_trap)."""
    bad = re.findall(r"^\s*zip\s+", script_src, re.MULTILINE)
    assert not bad, f"Forbidden shell `zip` invocation: {bad}"
    assert "zipfile.ZipFile" in script_src, (
        "Lane EC must build the archive with python zipfile.ZipFile."
    )


def test_archive_is_deterministic(script_src: str) -> None:
    """Codex R5-r6 #5 archive-builder rule: ZipInfo with fixed timestamp
    so archive bytes hash is reproducible across reruns.
    """
    # Either a fixed (1980, 1, 1, ...) date_time tuple OR an explicit
    # ZipInfo construction with a literal date.
    has_fixed_date = "date_time=(1980" in script_src or "date_time = (1980" in script_src
    has_zipinfo = "ZipInfo(" in script_src
    assert has_zipinfo and has_fixed_date, (
        "Lane EC archive build must use ZipInfo with a fixed date_time "
        "tuple (codex R5-r6 #5 deterministic-zip rule)."
    )


def test_no_cpu_or_mps_device(script_src: str) -> None:
    """CLAUDE.md FORBIDDEN PATTERN: no MPS/CPU fallback (mps_cuda_drift_critical).
    Lane EC must run --device cuda on every subprocess call.
    """
    assert "--device cpu" not in script_src, (
        "CLAUDE.md non-negotiable: --device cpu forbidden in lane scripts."
    )
    assert "--device mps" not in script_src, (
        "CLAUDE.md non-negotiable: --device mps forbidden (23x score drift)."
    )
    assert "--device cuda" in script_src


def test_uses_container_python(script_src: str) -> None:
    """memory feedback_canonical_remote_bootstraps: use container Python
    /opt/conda/bin/python NOT a venv (the container has the right CUDA
    + DALI install)."""
    assert "/opt/conda/bin/python" in script_src
    assert ".venv/bin/python" not in script_src


def test_workspace_path_is_canonical(script_src: str) -> None:
    assert "WORKSPACE=/workspace/pact" in script_src


def test_env_sh_sourced(script_src: str) -> None:
    """sourcing env.sh declares dependence on setup_full's macOS resource
    fork purge (check 37)."""
    assert 'source "$WORKSPACE/env.sh"' in script_src


# ── Stage 1: required artifact preflight ─────────────────────────────────


def test_required_artifacts_preflight(script_src: str) -> None:
    """Lane EC seeds from Lane A's verified 1.15 baseline. All three
    artifacts (renderer + masks + poses) plus the GT video and scorer
    weights must exist before any work begins."""
    required = [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
        "upstream/videos/0.mkv",
        "segnet.safetensors",
        "posenet.safetensors",
    ]
    for f in required:
        assert f in script_src, f"missing artifact preflight for {f!r}"
    assert "submissions/baseline_dilated_h64_0_90" in script_src, (
        "Lane EC must seed from the Lane A baseline directory "
        "(verified 1.15 [contest-CUDA] on 2026-04-27)."
    )


# ── Stage 2: engineered_quant_noise CLI flags (NEVER invent flags) ───────


def test_invokes_engineered_quant_noise_with_real_flags(script_src: str) -> None:
    """CLAUDE.md non-negotiable (feedback_dead_flag_wiring_pattern): every
    flag passed to a subprocess must exist in the target's argparse.

    This test introspects experiments/engineered_quant_noise.py's
    add_argument calls and asserts our invocation set is a SUBSET. It
    catches the DEN-V2 / R1-R2 dead-flag bug class at commit time.
    """
    real_flags = _argparse_flags(ENGINEERED)
    # Sanity: the target itself must declare these — guards against the
    # target script being renamed / argparse refactored without us noticing.
    expected_target_flags = {
        "--checkpoint", "--device", "--n-frames", "--batch-size",
        "--max-delta", "--output-dir", "--video", "--smoke",
        "--gt-poses-path", "--quantize-bits", "--max-artifact-bytes",
    }
    missing_in_target = expected_target_flags - real_flags
    assert not missing_in_target, (
        f"engineered_quant_noise.py missing flags this test depends on: "
        f"{missing_in_target}. Either the target was refactored or this "
        f"test is stale."
    )

    # Extract the flag set from our invocation block. Anchor on the actual
    # `$PYBIN ... engineered_quant_noise.py` shell invocation (NOT comments
    # that mention the filename), then collect that line + all backslash-
    # continued lines until a non-continued line.
    lines = script_src.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        # Match real shell invocation: must contain $PYBIN AND the filename,
        # AND not be a comment line. This skips top-of-file documentation
        # comments that mention the script.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "$PYBIN" in line and "engineered_quant_noise.py" in line:
            start_idx = i
            break
    assert start_idx is not None, (
        "couldn't locate $PYBIN engineered_quant_noise.py invocation in "
        "the deploy script — Lane EC must invoke the tool via the canonical "
        "$PYBIN -u pattern (matches our remote_lane_*.sh template)."
    )
    # Collect this line + continuation lines (line preceding ends in `\`).
    block_lines = [lines[start_idx]]
    j = start_idx
    while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):
        j += 1
        block_lines.append(lines[j])
    block = "\n".join(block_lines)

    # Tiny shim so the rest of this test reads naturally.
    class _M:
        def __init__(self, s: str) -> None:
            self._s = s
        def group(self, _i: int) -> str:
            return self._s
    m = _M(block)
    invoked = set(re.findall(r"\s(--[\w-]+)", m.group(0)))
    invented = invoked - real_flags
    assert not invented, (
        f"Lane EC INVENTED CLI flags for engineered_quant_noise.py: {invented}. "
        f"feedback_dead_flag_wiring_pattern: every flag must come from the "
        f"target's add_argument list. Real flags are: {sorted(real_flags)}"
    )

    # The two load-bearing flags MUST be present:
    assert "--max-artifact-bytes" in invoked, (
        "Lane EC must cap rate cost via --max-artifact-bytes (council "
        "Quantizr 2026-04-26: without the cap a bad renderer can ship a "
        "rate-busting corrections.bin)."
    )
    assert "--max-delta" in invoked, (
        "Lane EC must specify --max-delta (perturbation magnitude)."
    )


def test_max_artifact_bytes_bounded(script_src: str) -> None:
    """The byte cap controls the rate-cost ceiling. 51200 bytes = 0.0014
    rate cost → bounded downside even if SegNet improvement is zero."""
    assert "--max-artifact-bytes 51200" in script_src, (
        "Lane EC must cap corrections.bin at 51200 bytes (0.0014 rate "
        "cost) per council Quantizr; larger artifacts wipe out gain."
    )


def test_corrections_size_validated_before_archive(script_src: str) -> None:
    """The script must FAIL LOUD if corrections.bin is empty or larger
    than the cap (not WARN-and-skip). CLAUDE.md no-wasted-resources rule.
    """
    assert "$CORRECTIONS_BYTES" in script_src
    # Look for the explicit bounds check.
    assert "-le 0" in script_src or "<= 0" in script_src
    assert "51200" in script_src


# ── Stage 3: archive structure includes gradient_corrections.bin ─────────


def test_archive_contains_gradient_corrections_bin(script_src: str) -> None:
    """Inflate-side dispatcher hardcodes `gradient_corrections.bin` —
    without that exact name in the archive the deltas are silently
    skipped (test_archive_includes_corrections_when_provided locks this in).
    """
    assert "gradient_corrections.bin" in script_src
    # Verify the archive build literally lists it among entries.
    assert "'gradient_corrections.bin'" in script_src or \
           '"gradient_corrections.bin"' in script_src, (
        "The Stage 3 archive builder must list gradient_corrections.bin "
        "explicitly so the inflate dispatch finds it."
    )


def test_archive_contains_lane_a_artifacts(script_src: str) -> None:
    """Lane EC archive = Lane A artifacts + corrections.bin. All four
    files must be in the archive structure block."""
    archive_section = re.search(r"Stage 3:.*?Stage 4:", script_src, re.DOTALL)
    assert archive_section is not None
    block = archive_section.group(0)
    for name in ("renderer.bin", "masks.mkv",
                 "optimized_poses.pt", "gradient_corrections.bin"):
        assert name in block, (
            f"archive build block missing {name!r} — inflate would silently "
            f"skip the missing artifact."
        )


def test_archive_validated_after_build(script_src: str) -> None:
    """The script must verify the archive structurally contains all 4
    required entries before calling auth_eval. Otherwise a silent-skip
    in the build step manifests only as a flat (~1.15) score and we
    waste 30 min of GPU on a mis-built archive.
    """
    assert "z.namelist()" in script_src, (
        "Lane EC must validate the archive contents (zipfile.namelist) "
        "before calling auth_eval — no-wasted-resources rule."
    )


# ── Stage 4: contest_auth_eval CLI flags (NEVER invent flags) ────────────


def test_stage_4_runs_contest_auth_eval(script_src: str) -> None:
    assert "contest_auth_eval.py" in script_src


def test_invokes_contest_auth_eval_with_real_flags(script_src: str) -> None:
    """Mirror of test_invokes_engineered_quant_noise_with_real_flags
    against the auth-eval driver. Same dead-flag-wiring memory."""
    real_flags = _argparse_flags(CONTEST_EVAL)
    expected_in_target = {
        "--archive", "--inflate-sh", "--upstream-dir",
        "--device", "--keep-work-dir", "--work-dir",
    }
    missing_in_target = expected_in_target - real_flags
    assert not missing_in_target, (
        f"contest_auth_eval.py missing flags this test depends on: "
        f"{missing_in_target}."
    )
    pat = r"contest_auth_eval\.py.+?(?=\n\S|\Z)"
    m = re.search(pat, script_src, re.DOTALL)
    assert m is not None
    invoked = set(re.findall(r"\s(--[\w-]+)", m.group(0)))
    invented = invoked - real_flags
    assert not invented, (
        f"Lane EC INVENTED CLI flags for contest_auth_eval.py: {invented}. "
        f"Real flags are: {sorted(real_flags)}"
    )


def test_auth_eval_result_json_validated(script_src: str) -> None:
    """no-wasted-resources rule: if RESULT_JSON is missing the run is
    invalid and we must abort loudly, not silently log success.
    """
    assert "grep -q '^RESULT_JSON'" in script_src
    assert "FATAL" in script_src


# ── strict-scorer-rule: no scorer load at INFLATE ─────────────────────────


def test_inflate_renderer_loads_corrections_without_scorer(script_src: str) -> None:
    """Strict-scorer-rule (CLAUDE.md non-negotiable): corrections.bin is
    DATA, not MODEL. The inflate dispatcher must already load it via
    sparse-decoder + apply_corrections — no scorer (PoseNet/SegNet)
    forward pass at inflate.

    This is a structural cross-check on the inflate-side code path,
    independent of the lane script. Lane EC's value depends entirely on
    this dispatch already existing.
    """
    inflate_text = INFLATE_RENDERER.read_text()
    # The dispatch path lives in inflate_renderer.py (already verified
    # 2026-04-27) — we just confirm it hasn't regressed.
    assert "gradient_corrections.bin" in inflate_text, (
        "Inflate dispatcher must load gradient_corrections.bin — the "
        "wire-in is what makes Lane EC's archive non-trivial."
    )
    assert "_unpack_sparse_corrections" in inflate_text
    assert "_apply_gradient_corrections" in inflate_text
    # The application path must NOT load a scorer (strict-scorer-rule).
    # We check that the corrections-application code path doesn't sit
    # next to a `load_scorers` / `load_differentiable_scorers` call.
    # Lane EC explicitly: corrections.bin is data, not model.
    # The full scorer-at-inflate ban is enforced by check_no_scorer_load_at_inflate;
    # here we just assert the corrections code path itself doesn't
    # introduce one as a side effect.
    corr_block_match = re.search(
        r"Load gradient corrections.*?Load Lane C UNIWARD",
        inflate_text, re.DOTALL,
    )
    assert corr_block_match is not None, (
        "Couldn't locate corrections-load block in inflate_renderer.py "
        "(comment markers may have moved — refactor of the dispatch needs "
        "this test updated)."
    )
    block = corr_block_match.group(0)
    assert "load_scorers" not in block, (
        "Strict-scorer-rule violation: corrections-load path imports a "
        "scorer. corrections.bin must be DATA only."
    )
    assert "load_differentiable_scorers" not in block, (
        "Strict-scorer-rule violation: corrections-load path imports a "
        "differentiable scorer. corrections.bin must be DATA only."
    )


# ── self-documenting predicted band reasoning ─────────────────────────────


def test_script_documents_predicted_band_reasoning(script_src: str) -> None:
    """Header docstring must explain WHY the band is [0.85, 1.15].
    Per memory project_baseline_poses_load_bearing: explicit reasoning in
    the script header lets a fresh agent understand the experiment without
    re-running it.
    """
    assert "PREDICTED BAND" in script_src or "predicted band" in script_src.lower()
    assert "1.15" in script_src and "0.85" in script_src
    # The reasoning must mention BOTH directions: SegNet improvement AND
    # rate cost. A band that only mentions one side is overconfident.
    has_segnet = "SegNet" in script_src or "segnet" in script_src
    has_rate = "rate" in script_src.lower() or "byte" in script_src.lower()
    assert has_segnet and has_rate, (
        "Predicted band reasoning must cover both SegNet improvement "
        "AND rate cost (council Contrarian: one-sided bands are "
        "wishful thinking)."
    )


# ── strict-scorer-rule + no-MPS regression at the script level ───────────


def test_script_does_not_load_scorers_at_inflate(script_src: str) -> None:
    """Lane EC ONLY loads scorers at COMPRESS TIME (Stage 2 — engineered
    correction search). It must not invoke any inflate-time scorer-load.
    """
    # The only scorer references should be in Stage 2 (engineered_quant_noise
    # is a compress-time tool that loads SegNet/PoseNet to compute the
    # gradient deltas). Nothing in Stage 3 (build) or Stage 4 (eval) should
    # re-load scorers from the lane script side.
    stage_3_4 = re.search(r"Stage 3:.*", script_src, re.DOTALL)
    assert stage_3_4 is not None
    block = stage_3_4.group(0)
    forbidden = ["load_scorers", "load_differentiable_scorers", "PoseNet(", "SegNet("]
    for tok in forbidden:
        assert tok not in block, (
            f"Strict-scorer-rule violation: token {tok!r} appears in "
            f"Stage 3+4 of the lane script. Scorers may only be loaded "
            f"at compress time (Stage 2 — inside engineered_quant_noise.py)."
        )
