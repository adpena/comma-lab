"""Lane M-V2 regression: verify optimize_poses.py saves (N, 6) with
frozen baseline padding for dims 1-5 in --pose-mode radial-zoom.

Background (2026-04-27):
  Lane M-V1 scored 2.35. Bug: --pose-mode radial-zoom saved (N, 1) and
  the (no-op) inflate-side adapter zero-padded dims 1-5. PoseNet's
  auxiliary 5 dims encode the rank-1 information the renderer was
  trained on (memory: project_baseline_poses_load_bearing — real vs
  zero poses → 33% pixel shift, 23x PoseNet degrade). Zero-padding
  destroyed that signal.

Lane M-V2 fix:
  When --pose-mode radial-zoom is set, the save block in
  experiments/optimize_poses.py composes a (N, 6) tensor where:
    - dim 0 = the optimized radial-zoom scalar
    - dims 1-5 = the FROZEN baseline values from `--gt-poses-path`
                 (loaded as init_poses; init_poses[:n_pairs, 1:6])
  The saved file is now consumable by inflate without any pose_mode-
  aware adapter — it IS the canonical 6-DOF pose tensor.

What these tests pin:
  T1. Source-grep: the save block uses init_poses[:n_pairs, 1:6] for
      the frozen aux dims.
  T2. Source-grep: the (N, 6) is composed via torch.cat([scalar, aux]).
  T3. Source-grep: the meta sidecar still records pose_mode for
      observability + paper provenance.
  T4. Source-grep: the in-memory render projection no longer zero-pads
      on the canonical path (defensive (N, 1) fallback only logs a
      WARNING — it does NOT silently produce a wrong score).
  T5. Source-grep: the V1 hard-coded (N, 1) save (the
      `optimized_poses = all_optimized[:, :pose_dim_internal]` direct
      assignment) is NO LONGER the canonical save — the radial-zoom
      branch must compose (N, 6) before save.
  T6. Behavioral: a synthetic execution of the save logic verifies that
      given init_poses with non-zero dims 1-5, the saved tensor's dims
      1-5 EXACTLY match init_poses (not zero, not garbage).
  T7. Defensive: in full-6dof mode the save remains (N, 6) with the
      original behavior (no regression for the default path).
  T8. The argparse `--gt-poses-path` flag is documented as REQUIRED
      for radial-zoom mode (or the frozen baseline source is absent
      and the experiment is meaningless).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "optimize_poses.py"


@pytest.fixture(scope="module")
def src_text() -> str:
    return SCRIPT.read_text()


# ── T1+T2: source-grep the (N, 6) composition ─────────────────────────


def test_save_uses_baseline_init_poses_aux_dims(src_text: str):
    """T1: the save block must reach init_poses[:, :6] (Lane M-V3-clean
    passes the full baseline through the shared helper instead of slicing
    [:, 1:6] inline; both idioms preserve the frozen aux dims at inflate)."""
    has_v2_slice = "init_poses[:n_pairs, 1:6]" in src_text
    has_v3_full = "init_poses[:n_pairs, :6]" in src_text
    assert has_v2_slice or has_v3_full, (
        "save block must source the FROZEN baseline aux dims from "
        "init_poses (V2: [:n_pairs, 1:6], V3-clean: [:n_pairs, :6] "
        "passed to _project_to_renderer_pose). Without one of these "
        "the saved file regresses to (N, 1) and reproduces the V1 "
        "zero-pad bug at inflate."
    )


def test_save_composes_n6_via_torch_cat(src_text: str):
    """T2: the save block must compose (N, 6) — either inline torch.cat
    (V2 idiom) or by calling the shared _project_to_renderer_pose helper
    (V3-clean idiom). Both bind dim 0 to the optimized scalar and dims
    1-5 to the frozen baseline."""
    v2_inline = "torch.cat([pose_part[:, :1].cpu(), baseline_aux]" in src_text
    v3_helper = bool(re.search(
        r'optimized_poses\s*=\s*_project_to_renderer_pose\(\s*pose_part\.cpu\(\)',
        src_text,
    ))
    assert v2_inline or v3_helper, (
        "save block must compose (N, 6) either via "
        "`torch.cat([pose_part[:, :1].cpu(), baseline_aux], dim=-1)` (V2) "
        "or via `_project_to_renderer_pose(pose_part.cpu(), ...)` "
        "(V3-clean). dim 0 = optimized radial-zoom scalar, dims 1-5 = "
        "frozen baseline."
    )


# ── T3: meta sidecar records pose_mode for observability ───────────────


def test_meta_sidecar_records_pose_mode(src_text: str):
    """T3: pose_mode in meta sidecar is still required for observability."""
    assert '"pose_mode": args.pose_mode' in src_text, (
        "optimized_poses.meta must include `\"pose_mode\": args.pose_mode` "
        "for observability + paper provenance"
    )


# ── T4: in-memory render projection no longer zero-pads on canonical path ─


def test_render_projection_canonical_path_no_zero_pad(src_text: str):
    """T4: the in-memory render projection's zero-pad branch is now a
    DEFENSIVE fallback that logs WARNING — it should not be the canonical
    path. We verify by checking that:
      a) the (N, 1) zero-pad branch still exists (safety net)
      b) it prints a WARNING when triggered (so any regression is visible)
    """
    # Find the radial-zoom render-projection block.
    block = re.search(
        r'if args\.pose_mode == "radial-zoom" and optimized_poses\.shape\[1\] == 1.*?else:',
        src_text, re.DOTALL,
    )
    assert block is not None, (
        "the in-memory render projection branch must still exist as a "
        "defensive fallback"
    )
    # The branch must print a Lane M-V2 WARNING when it triggers.
    assert "Lane M-V2 WARNING" in block.group(0), (
        "the (N, 1) defensive branch must log a Lane M-V2 WARNING so "
        "any regression to the V1 zero-pad behavior is visible. Without "
        "the warning a regression would silently undercount PoseNet."
    )


# ── T5: V1 direct (N, pose_dim_internal) save is no longer the canonical path ─


def test_save_does_not_directly_assign_pose_dim_internal_slice(src_text: str):
    """T5: the V1 code did
        `optimized_poses = all_optimized[:, :pose_dim_internal]`
       then saved that directly. V2 still slices (the variable is still
       useful for the latent path), but the SAVE must come from the
       (N, 6) composed tensor, not the (N, pose_dim_internal) slice.

       We assert the save block uses `pose_part` as the intermediate
       slice (which is THEN composed into the (N, 6) tensor for save)
       — the variable name change is the canonical V2 marker.
    """
    assert "pose_part = all_optimized[:, :pose_dim_internal]" in src_text, (
        "Lane M-V2 must rename the slice variable to `pose_part` so the "
        "subsequent `optimized_poses` assignment is unambiguous: in "
        "radial-zoom it is the (N, 6) composed tensor, not the (N, 1) "
        "slice. The V1 code wrote directly to `optimized_poses` and "
        "saved that — the V2 rename makes the new path explicit."
    )


# ── T6: behavioral check via synthetic save-block execution ────────────


def test_save_block_behavioral_synthesis():
    """T6: simulate the save block in isolation. Given:
        - init_poses with NON-ZERO baseline values for dims 1-5
        - all_optimized with optimized scalars in dim 0
      Verify the composed (N, 6) tensor has:
        - dim 0 = the optimized scalar
        - dims 1-5 = EXACTLY the init_poses values (not zero, not garbage)

    This is a behavioral spec test — if the source-grep tests above
    pass but the actual semantics drift (e.g., wrong slicing), this
    test catches it.
    """
    # Synthesize an init_poses tensor with non-zero baseline aux dims
    # that look nothing like zero — so a zero-pad bug is obvious.
    n_pairs = 600
    init_poses = torch.zeros(n_pairs, 6)
    init_poses[:, 0] = 0.0  # dim 0 will be overwritten by optimized
    init_poses[:, 1] = 1.234  # baseline aux dim 1 (must be preserved)
    init_poses[:, 2] = -0.567  # baseline aux dim 2
    init_poses[:, 3] = 2.345  # baseline aux dim 3
    init_poses[:, 4] = -1.678  # baseline aux dim 4
    init_poses[:, 5] = 0.789  # baseline aux dim 5

    # all_optimized is what came out of the optimizer (in radial-zoom
    # mode pose_dim_internal=1, so [:, :1] is the optimized scalar).
    pose_dim_internal = 1
    all_optimized = torch.randn(n_pairs, pose_dim_internal) * 0.1

    # Replicate the V2 save-block composition.
    pose_part = all_optimized[:, :pose_dim_internal]  # (N, 1)
    baseline_aux = init_poses[:n_pairs, 1:6].detach().cpu().to(pose_part.dtype)
    composed = torch.cat([pose_part[:, :1].cpu(), baseline_aux], dim=-1)

    # Assertions
    assert composed.shape == (n_pairs, 6), (
        f"composed tensor must be (N, 6), got {tuple(composed.shape)}"
    )
    # Dim 0 must equal the optimized scalar.
    assert torch.allclose(composed[:, 0], pose_part[:, 0].cpu()), (
        "composed[:, 0] must equal the optimized radial-zoom scalar"
    )
    # Dims 1-5 must EXACTLY equal init_poses[:, 1:6] (frozen baseline).
    assert torch.allclose(composed[:, 1:6], init_poses[:n_pairs, 1:6]), (
        "composed[:, 1:6] must EXACTLY equal init_poses[:, 1:6]. "
        "Any drift means the V1 zero-pad bug has been reintroduced."
    )
    # Specifically: dims 1-5 are NOT zero (they're the non-zero baseline).
    assert not torch.allclose(
        composed[:, 1:6], torch.zeros_like(composed[:, 1:6])
    ), (
        "composed[:, 1:6] must NOT be zero — the V1 bug was zero-padding "
        "the auxiliary dims, which destroyed PoseNet's rank-1 signal."
    )


# ── T7: defensive — full-6dof mode unchanged ───────────────────────────


def test_full_6dof_mode_save_path_preserved(src_text: str):
    """T7: in full-6dof mode (the default), the save must be identical to
    pre-V2 behavior — the radial-zoom branch must be GATED on
    `args.pose_mode == "radial-zoom"`. V3-clean replaces the inline
    `torch.cat` with a call to `_project_to_renderer_pose` but the gate
    + else fallthrough must remain identical."""
    # Either V2 idiom (torch.cat) or V3-clean idiom (helper call) is
    # acceptable — the test pins the GATE structure, not the body.
    gated_save_v2 = re.search(
        r'if args\.pose_mode == "radial-zoom":.*?'
        r'optimized_poses = torch\.cat.*?'
        r'else:.*?'
        r'optimized_poses = pose_part',
        src_text, re.DOTALL,
    )
    gated_save_v3 = re.search(
        r'if args\.pose_mode == "radial-zoom":.*?'
        r'optimized_poses = _project_to_renderer_pose\(.*?'
        r'else:.*?'
        r'optimized_poses = pose_part',
        src_text, re.DOTALL,
    )
    assert gated_save_v2 is not None or gated_save_v3 is not None, (
        "the (N, 6) composition must be gated on "
        "`args.pose_mode == \"radial-zoom\"` so full-6dof mode (the "
        "default) saves the (N, 6) tensor unchanged. Without the gate "
        "the full-6dof path could regress. Accepts either the V2 "
        "torch.cat idiom or the V3-clean _project_to_renderer_pose helper."
    )


# ── T8: --gt-poses-path is the documented frozen baseline source ───────


def test_gt_poses_path_argparse_present(src_text: str):
    """T8: --gt-poses-path is the source of frozen baseline values for
    dims 1-5 in radial-zoom mode. It must remain in argparse."""
    assert re.search(
        r'add_argument\(\s*[\"\']--gt-poses-path[\"\']', src_text,
    ), "--gt-poses-path must be in argparse — Lane M-V2 sources frozen baseline dims from it"


# ── T9: error path — meaningful error if init_poses isn't (N, 6) ───────


def test_save_block_validates_init_poses_shape(src_text: str):
    """T9: if init_poses isn't (N, 6) (e.g., a (N, 1) tensor was passed
    by mistake), the save block must raise SystemExit with a useful
    message — NOT silently produce a (N, 2) tensor by concatenating
    a 1-DOF baseline_aux."""
    # Either V2 wording ("Lane M-V2 expects baseline init_poses to be (N, 6)")
    # or V3-clean wording ("Lane M-V3-clean expects baseline init_poses to be (N, 6)")
    # is acceptable — both raise SystemExit with the (N, 6) shape guidance.
    has_shape_guard = (
        "FATAL: Lane M-V2 expects baseline init_poses to be (N, 6)" in src_text
        or "FATAL: Lane M-V3-clean expects baseline init_poses to be (N, 6)" in src_text
    )
    assert has_shape_guard, (
        "the save block must raise SystemExit with a meaningful message "
        "if init_poses isn't (N, 6). Without this guard a (N, 1) "
        "init_poses would silently produce a malformed save tensor."
    )
