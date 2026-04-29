"""Lane M + Lane N regression: verify optimize_poses.py pose-mode + L∞ wiring.

Lane M (radial-zoom-only pose mode): per memory
`project_posenet_rank1_discovery`, PoseNet's Jacobian is rank ≈ 1.008 with
99.8% variance in the canonical "z forward" radial-zoom dim. Optimizing 6
DOF when only 1 carries scoring signal wastes optimizer steps and adds
noise. The flag exposes a `(N, 1)` parameter that is projected to `(N, 6)`
before the renderer call (the renderer was trained on 6-DOF input).

Lane N (Fridrich L∞ pose penalty): per memory
`project_fridrich_inverse_steganalysis` Principle 3, an L∞ budget on
`(current_pose - baseline_pose)` keeps perturbations uniformly small —
PoseNet detects concentrated changes, so a soft L∞-ball constraint biases
the optimizer toward spreading information across dims. The helper is
`tac.fridrich.linf_pose_penalty` (per `feedback_existing_fridrich_code`,
the canonical Fridrich code already exists; do not rebuild).

What the wiring must guarantee (Lane M):
  M.A. `--pose-mode {full-6dof, radial-zoom}` flag is visible in --help
       and defaults to `full-6dof` (preserve baseline call path).
  M.B. argparse `choices=` validates the flag value (rejects garbage).
  M.C. When `pose_mode='radial-zoom'`, `optimize_poses_batch` initialises
       the optimizable as `(B, 1)` (NOT `(B, 6)`).
  M.D. When `pose_mode='radial-zoom'`, the optimizable is projected to
       `(B, 6)` before the renderer call (zero-pad the auxiliary 5 dims).
  M.E. When `pose_mode='radial-zoom'`, the saved `optimized_poses.pt` is
       `(N, 1)` and `optimized_poses.meta` records `pose_mode`.

What the wiring must guarantee (Lane N):
  N.A. `--linf-pose-weight` flag is visible in --help and defaults to 0.0.
  N.B. `--linf-pose-budget` flag is visible in --help and defaults to 0.05.
  N.C. The body imports the canonical helper
       `from tac.fridrich import linf_pose_penalty` — NOT a re-implementation.
  N.D. Per-step log appends `linf=...` when `linf_pose_weight > 0`.
  N.E. With weight=0.0, the L∞ block does NOT run (gated path so the
       default is byte-identical to baseline).

Methodology mirrors `test_optimize_poses_kl_distill_wiring.py` — argparse
introspection via subprocess, source-grep for call-site threading, and
balanced-paren walker for the `optimize_poses_batch(...)` call site.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "optimize_poses.py"


def _extract_calls(text: str, fn_name: str) -> list[str]:
    """Walk balanced parens to extract every `fn_name(...)` call body.

    Used over regex because nested parens (e.g. `torch.arange(...)`) inside
    the call site break a naive pattern. Mirrors the helper inside
    test_optimize_poses_kl_distill_wiring.py.
    """
    out: list[str] = []
    idx = 0
    token = fn_name + "("
    while True:
        i = text.find(token, idx)
        if i < 0:
            break
        start = i + len(token)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[start:j - 1])
        idx = j
    return out


# ════════════════════════════════════════════════════════════════════════
#  Lane M tests
# ════════════════════════════════════════════════════════════════════════


def test_pose_mode_default_is_full_6dof():
    """--pose-mode must default to 'full-6dof' so existing callers see
    byte-identical behaviour. Source-grep the argparse default to catch
    silent default flips (the historical "default override antipattern"
    per CLAUDE.md)."""
    src = SCRIPT.read_text()
    m = re.search(
        r'add_argument\(\s*"--pose-mode"[^)]*default\s*=\s*"([^"]+)"',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "--pose-mode argparse declaration not found in optimize_poses.py"
    )
    assert m.group(1) == "full-6dof", (
        f"--pose-mode default must be 'full-6dof' (preserve baseline), got "
        f"{m.group(1)!r} — Lane M would silently activate for ALL existing "
        f"pose-TTO callers."
    )


def test_pose_mode_argparse_choice_validates():
    """--pose-mode must use argparse choices=[...] so an invalid value
    fails fast at CLI-parse time instead of silently producing garbage
    inside the optimization loop."""
    src = SCRIPT.read_text()
    m = re.search(
        r'add_argument\(\s*"--pose-mode"[^)]*choices\s*=\s*\[([^\]]+)\]',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "--pose-mode must use choices=[...] so unknown values fail at "
        "CLI-parse time. Without it a typo silently selects 'full-6dof'."
    )
    choices_text = m.group(1)
    assert '"full-6dof"' in choices_text, (
        f"--pose-mode choices must include 'full-6dof', got: {choices_text}"
    )
    assert '"radial-zoom"' in choices_text, (
        f"--pose-mode choices must include 'radial-zoom', got: {choices_text}"
    )


def test_pose_mode_visible_in_help():
    """Subprocess --help must surface --pose-mode (real argparse — what
    the operator sees). This is the canonical dead-flag check per
    CLAUDE.md "NEVER invent CLI flags": don't trust source comments,
    ask argparse itself."""
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0, (
        f"optimize_poses.py --help failed: stderr={result.stderr[:500]}"
    )
    assert "--pose-mode" in result.stdout, (
        "Lane M wiring missing: --pose-mode not in --help. Operators cannot "
        "select radial-zoom mode."
    )


def test_pose_mode_radial_zoom_initializes_1dof_param():
    """When pose_mode='radial-zoom', the optimizable conditioning tensor
    must be (B, 1) (the canonical "z forward" component) — NOT (B, 6).
    Source-grep the gated init logic inside optimize_poses_batch."""
    src = SCRIPT.read_text()

    # The signature must accept pose_mode.
    sig_re = re.compile(
        r"def optimize_poses_batch\((?P<sig>.*?)\)\s*->\s*tuple",
        re.DOTALL,
    )
    sig_match = sig_re.search(src)
    assert sig_match is not None, "optimize_poses_batch signature not found"
    sig = sig_match.group("sig")
    assert 'pose_mode: str = "full-6dof"' in sig, (
        "optimize_poses_batch must accept `pose_mode: str = \"full-6dof\"` "
        "with a safe default so existing callers don't need to pass it."
    )

    # The body must have a branch that sets pose_dim_internal = 1 when
    # pose_mode == 'radial-zoom'. Without it the optimizable would still
    # be (B, 6) and the flag would be dead.
    assert re.search(
        r'if\s+pose_mode\s*==\s*"radial-zoom"\s*:\s*\n\s*pose_dim_internal\s*=\s*1',
        src,
    ) is not None, (
        "Lane M body missing: must set `pose_dim_internal = 1` when "
        "`pose_mode == \"radial-zoom\"`. Without this branch the radial-zoom "
        "flag is dead — the optimizable stays (B, 6) and the rank-1 thesis "
        "(project_posenet_rank1_discovery) is never tested."
    )


def test_pose_mode_radial_zoom_projects_to_6dof_before_render():
    """When pose_mode='radial-zoom', the (B, 1) optimizable must be
    projected to (B, 6) (zero-padded) before being passed to the
    renderer's FiLM layer (which was trained on 6-DOF input). Without
    this projection the renderer.forward(pose=...) would shape-mismatch
    immediately. Verify the projection helper exists in source."""
    src = SCRIPT.read_text()

    # The script must define the projection helper. Look for a function
    # or block that lifts (B, pose_dim_internal) → (B, pose_dim).
    has_helper = "_project_to_renderer_pose" in src
    assert has_helper, (
        "Lane M body missing: must define `_project_to_renderer_pose(...)` "
        "(or equivalent) that lifts the (B, 1) optimizable to (B, 6) before "
        "the renderer.forward() call. Without it the renderer's FiLM layer "
        "shape-mismatches and crashes."
    )

    # The renderer call must consume the projected value, not the raw
    # 1-DOF slice. Either the closure idiom (V2:
    # `_project_to_renderer_pose(conditioning)`) or the V3-clean idiom
    # (`_project_pose_for_render(conditioning)`, a thin partial-application
    # of the module-level helper) is acceptable.
    has_v2_closure = "_project_to_renderer_pose(conditioning)" in src
    has_v3_partial = "_project_pose_for_render(conditioning)" in src
    assert has_v2_closure or has_v3_partial, (
        "Lane M body missing: the renderer forward must consume the "
        "projected pose. Accepts either V2 closure "
        "`_project_to_renderer_pose(conditioning)` or V3-clean partial "
        "`_project_pose_for_render(conditioning)`. Without one of these, "
        "pose_part is (B, 1) and the renderer crashes immediately."
    )


def test_pose_mode_radial_zoom_saves_6dof_with_frozen_baseline_padding():
    """Lane M-V2 (2026-04-27): when pose_mode='radial-zoom', the saved
    optimized_poses.pt must be (N, 6) where dim 0 is the optimized scalar
    and dims 1-5 are the FROZEN baseline values from `--gt-poses-path`.

    The original Lane M-V1 saved (N, 1) and relied on the inflate side
    zero-padding dims 1-5. That was the V1 bug: PoseNet's auxiliary 5
    dims encode the rank-1 information the renderer was trained on, so
    zero-padding them destroyed the per-pair signal (V1 score 2.35 vs
    Lane A 1.15).

    V2 fixes this by composing the (N, 6) tensor at SAVE time so the
    file is consumable by inflate without any pose_mode-aware adapter.
    """
    src = SCRIPT.read_text()

    # The save block must source the FROZEN baseline aux dims from
    # init_poses. Lane M-V2 used `init_poses[:n_pairs, 1:6]` inline;
    # Lane M-V3-clean passes the full `init_poses[:n_pairs, :6]` tensor
    # through the shared `_project_to_renderer_pose` helper. Either is
    # acceptable — both compose the same (N, 6) bytes.
    has_v2_slice = "init_poses[:n_pairs, 1:6]" in src
    has_v3_full = "init_poses[:n_pairs, :6]" in src
    assert has_v2_slice or has_v3_full, (
        "save block must source the FROZEN baseline aux dims from "
        "init_poses (V2: [:n_pairs, 1:6], V3-clean: [:n_pairs, :6] passed "
        "to _project_to_renderer_pose). Without one of these the saved "
        "file regresses to (N, 1) and reproduces the V1 zero-pad bug."
    )
    has_v2_cat = 'torch.cat([pose_part[:, :1].cpu(), baseline_aux]' in src
    has_v3_helper = bool(re.search(
        r'optimized_poses\s*=\s*_project_to_renderer_pose\(\s*pose_part\.cpu\(\)',
        src,
    ))
    assert has_v2_cat or has_v3_helper, (
        "save block must compose (N, 6) — either V2 inline "
        "`torch.cat([pose_part[:, :1].cpu(), baseline_aux], dim=-1)` or "
        "V3-clean `_project_to_renderer_pose(pose_part.cpu(), ...)`. "
        "Without this composition the saved file is (N, 1) and reproduces "
        "the V1 zero-pad bug at inflate."
    )

    # The meta sidecar must STILL include `pose_mode` for observability
    # + paper provenance, even though the saved file is now self-
    # describing as (N, 6).
    assert '"pose_mode": args.pose_mode' in src, (
        "Lane M-V2 save: optimized_poses.meta must include "
        "`\"pose_mode\": args.pose_mode` for observability + paper "
        "provenance (the saved file is (N, 6) but the OPTIMIZATION mode "
        "is still meaningful metadata)."
    )


def test_pose_mode_threaded_through_call_site():
    """main() must thread pose_mode (+ Lane N flags) to optimize_poses_batch.
    Walk balanced parens to extract the call body and assert the keyword
    arguments are present. Mirrors the Lane G test methodology."""
    src = SCRIPT.read_text()
    call_blocks = _extract_calls(src, "optimize_poses_batch")
    assert call_blocks, "optimize_poses_batch call site not found"
    # Drop the def-site (annotated with type hints).
    call_sites = [
        block for block in call_blocks
        if "torch.Tensor | None" not in block and "torch.nn.Module" not in block
    ]
    assert call_sites, (
        "Could not find a non-def-site optimize_poses_batch( call. "
        "Did the call-site signature change?"
    )
    call_block = call_sites[-1]
    assert "pose_mode=args.pose_mode" in call_block, (
        "main() must pass `pose_mode=args.pose_mode` to optimize_poses_batch "
        "— without this the CLI flag is dead (per CLAUDE.md "
        "'NEVER invent CLI flags')."
    )
    assert "linf_pose_weight=args.linf_pose_weight" in call_block, (
        "main() must pass `linf_pose_weight=args.linf_pose_weight` — "
        "without this the Lane N CLI flag is dead."
    )
    assert "linf_pose_budget=args.linf_pose_budget" in call_block, (
        "main() must pass `linf_pose_budget=args.linf_pose_budget` — "
        "without this the Lane N budget knob is dead."
    )


# ════════════════════════════════════════════════════════════════════════
#  Lane N tests
# ════════════════════════════════════════════════════════════════════════


def test_linf_pose_default_off_matches_baseline():
    """--linf-pose-weight must default to 0.0 so the L∞ block is gated
    off and existing experiments produce the identical loss surface they
    did pre-Lane-N. Source-grep the argparse block to catch silent
    default flips (the "default override antipattern" per CLAUDE.md)."""
    src = SCRIPT.read_text()
    m = re.search(
        r'add_argument\(\s*"--linf-pose-weight"[^)]*default\s*=\s*([0-9.]+)',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "--linf-pose-weight argparse declaration not found"
    )
    assert float(m.group(1)) == 0.0, (
        f"--linf-pose-weight default must be 0.0 (off), got {m.group(1)} — "
        f"this would silently enable Lane N for ALL existing pose-TTO callers."
    )


def test_linf_pose_weight_argparse_visible():
    """Subprocess --help must surface --linf-pose-weight (real argparse
    introspection — what the operator sees)."""
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert result.returncode == 0, (
        f"optimize_poses.py --help failed: stderr={result.stderr[:500]}"
    )
    assert "--linf-pose-weight" in result.stdout, (
        "Lane N wiring missing: --linf-pose-weight not in --help. "
        "Operators cannot enable the Fridrich L∞ pose penalty."
    )


def test_linf_pose_budget_argparse_default_0_05():
    """--linf-pose-budget must default to 0.05 — small enough not to
    perturb the rank-1 dominant dim (per project_posenet_rank1_discovery),
    large enough to allow exploration on aux dims. Verify the default
    is the council-approved value."""
    src = SCRIPT.read_text()
    m = re.search(
        r'add_argument\(\s*"--linf-pose-budget"[^)]*default\s*=\s*([0-9.]+)',
        src,
        re.DOTALL,
    )
    assert m is not None, (
        "--linf-pose-budget argparse declaration not found"
    )
    assert float(m.group(1)) == 0.05, (
        f"--linf-pose-budget default must be 0.05 (Lane N council-approved "
        f"radius), got {m.group(1)}. Larger values risk perturbing the "
        f"rank-1 dominant Jacobian dim; smaller values are degenerate."
    )

    # --help should also surface the budget flag.
    env_pythonpath = f"{REPO / 'src'}:{REPO / 'upstream'}"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        timeout=30,
    )
    assert "--linf-pose-budget" in result.stdout, (
        "Lane N: --linf-pose-budget not in --help."
    )


def test_linf_pose_helper_imported_correctly():
    """The wiring MUST import `linf_pose_penalty` from tac.fridrich (the
    canonical helper). Per CLAUDE.md "Fridrich Code Already Exists" memory
    + `feedback_existing_fridrich_code`: don't rebuild. Source-grep the
    import statement; reject inline implementations that bypass the
    canonical helper."""
    src = SCRIPT.read_text()

    # The import is conditional (inside the `if linf_pose_weight > 0:`
    # block to avoid pulling fridrich.py + numpy at module load time
    # for the default-off path). So check for the import string in the
    # file regardless of indentation.
    assert "from tac.fridrich import linf_pose_penalty" in src, (
        "optimize_poses.py must import `linf_pose_penalty` from tac.fridrich. "
        "Per CLAUDE.md 'Fridrich Code Already Exists' — the canonical helper "
        "is in src/tac/fridrich.py; do not inline a re-implementation."
    )

    # Verify the helper actually exists in tac.fridrich (catches the
    # "imported a function that doesn't exist" failure mode).
    fridrich_src = (REPO / "src" / "tac" / "fridrich.py").read_text()
    assert "def linf_pose_penalty(" in fridrich_src, (
        "tac.fridrich must export `linf_pose_penalty(...)` — the optimize_poses "
        "wiring depends on it."
    )
    assert '"linf_pose_penalty"' in fridrich_src, (
        "tac.fridrich.__all__ must include 'linf_pose_penalty' so the canonical "
        "import surface is explicit."
    )


def test_linf_pose_log_line_includes_linf_when_active():
    """When linf_pose_weight > 0, the per-step log line must surface
    the L∞ violation value so operators can monitor it during long
    pose-TTO runs. Silent L∞ = wasted GPU per CLAUDE.md "no wasted
    resources" rule."""
    src = SCRIPT.read_text()
    assert (
        'linf_str = f" linf={linf_loss_val:.6f}" if linf_pose_weight > 0 else ""'
    ) in src, (
        "Per-step log must include `linf={value:.6f}` when "
        "linf_pose_weight > 0 so the operator can monitor the Fridrich "
        "L∞ penalty in real time. CLAUDE.md no-wasted-resources rule."
    )


def test_linf_pose_block_gated_on_weight():
    """The L∞ penalty block must be gated on `if linf_pose_weight > 0`
    so the default callers (weight=0.0) hit the zero-overhead path —
    NO import, NO computation. This guarantees byte-identical baseline
    behaviour pre-Lane-N."""
    src = SCRIPT.read_text()
    assert re.search(
        r"if\s+linf_pose_weight\s*>\s*0\s*:",
        src,
    ) is not None, (
        "Lane N block must be gated on `if linf_pose_weight > 0:` — "
        "without the gate, default callers (weight=0.0) would still pay "
        "the import cost (and worse, the L∞ violation would be added to "
        "total_loss with weight 0, which is a wasted gradient computation)."
    )

    # The function signature must accept linf_pose_weight + linf_pose_budget
    # with safe defaults so existing callers don't need to pass them.
    sig_re = re.compile(
        r"def optimize_poses_batch\((?P<sig>.*?)\)\s*->\s*tuple",
        re.DOTALL,
    )
    sig_match = sig_re.search(src)
    assert sig_match is not None
    sig = sig_match.group("sig")
    assert "linf_pose_weight: float = 0.0" in sig, (
        "optimize_poses_batch must accept `linf_pose_weight: float = 0.0` "
        "so existing callers default to disabled."
    )
    assert "linf_pose_budget: float = 0.05" in sig, (
        "optimize_poses_batch must accept `linf_pose_budget: float = 0.05` "
        "(council-approved default). Existing callers default to a "
        "scientifically meaningful budget."
    )


# ════════════════════════════════════════════════════════════════════════
#  Cross-lane: linf_pose_penalty helper unit test
# ════════════════════════════════════════════════════════════════════════


def test_linf_pose_penalty_helper_semantics():
    """Direct unit test of `tac.fridrich.linf_pose_penalty`. The penalty
    must be:
      - Zero when the per-element |delta| is within budget.
      - Equal to sum(max(0, |delta| - budget)) outside the budget.
      - Differentiable w.r.t. current_pose (gradient flows back).
      - Shape-agnostic (works for both (B, 1) and (B, 6) tensors).
    """
    import torch

    from tac.fridrich import linf_pose_penalty

    # Inside-budget: zero penalty.
    baseline = torch.zeros(4, 6)
    current = torch.full((4, 6), 0.04, requires_grad=True)
    pen = linf_pose_penalty(current, baseline, budget=0.05)
    assert pen.item() == pytest.approx(0.0), (
        f"Inside-budget penalty must be 0.0, got {pen.item()}"
    )

    # Outside-budget: sum(max(0, |delta| - budget)).
    current2 = torch.full((4, 6), 0.10, requires_grad=True)
    pen2 = linf_pose_penalty(current2, baseline, budget=0.05)
    expected = 4 * 6 * (0.10 - 0.05)
    assert pen2.item() == pytest.approx(expected, rel=1e-5), (
        f"Outside-budget penalty must be {expected}, got {pen2.item()}"
    )

    # Gradient flows back.
    pen2.backward()
    assert current2.grad is not None
    assert current2.grad.abs().max() > 0, (
        "Gradient of L∞ penalty w.r.t. current_pose must be nonzero outside "
        "the budget — without it the penalty does nothing during optimization."
    )

    # Shape-agnostic: also works for (B, 1) (Lane M radial-zoom mode).
    base_1d = torch.zeros(4, 1)
    cur_1d = torch.full((4, 1), 0.10, requires_grad=True)
    pen_1d = linf_pose_penalty(cur_1d, base_1d, budget=0.05)
    expected_1d = 4 * 1 * (0.10 - 0.05)
    assert pen_1d.item() == pytest.approx(expected_1d, rel=1e-5), (
        f"(B, 1) penalty must be {expected_1d}, got {pen_1d.item()}"
    )
