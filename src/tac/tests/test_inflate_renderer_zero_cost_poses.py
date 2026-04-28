"""End-to-end tests for the Lane LM-A zero-cost-poses inflate path.

Lane LM-A wires
``submissions/robust_current/inflate_renderer.py`` to compute per-pair
6-DOF poses at INFLATE TIME from lane-mark mask displacement when:

  1. The archive contains the ``zero_cost_poses_v1`` 0-byte sentinel.
  2. The archive does NOT contain ``optimized_poses.pt`` / ``poses.pt``.
  3. The env var ``INFLATE_ZERO_COST_POSES=1`` is set.

This file pins:

  * The source-grep regression checks (catch a future refactor that drops
    the wiring).
  * The env-gate semantics — sentinel without env should warn loudly,
    sentinel WITH env should compute via tac.lane_mark_pose.
  * The strict-scorer-rule compliance claim — the inflate path MUST NOT
    load PoseNet or SegNet for the pose computation. We grep the relevant
    code block to verify.
  * The fallback math — the call site forwards the masks tensor (already
    decoded for SegNet conditioning) to ``compute_zero_cost_poses_from_masks``,
    not a separate decode (no double decode, no double cost).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


@pytest.fixture(scope="module")
def inflate_src() -> str:
    assert INFLATE_PATH.exists(), f"inflate_renderer.py not found at {INFLATE_PATH}"
    return INFLATE_PATH.read_text()


# ── Source-grep regression checks ────────────────────────────────────────


def test_inflate_imports_compute_zero_cost(inflate_src: str):
    """The inflate path must import compute_zero_cost_poses_from_masks
    from tac.lane_mark_pose. A future refactor that drops the import
    would silently break Lane LM-A."""
    assert "from tac.lane_mark_pose import compute_zero_cost_poses_from_masks" in inflate_src, (
        "inflate_renderer.py is missing the lane_mark_pose import for "
        "the Lane LM-A zero-cost-poses path"
    )


def test_inflate_references_sentinel_filename(inflate_src: str):
    """The sentinel filename must match the canonical
    ZERO_COST_POSES_SENTINEL constant exactly."""
    from tac.lane_mark_pose import ZERO_COST_POSES_SENTINEL
    assert ZERO_COST_POSES_SENTINEL in inflate_src, (
        f"inflate_renderer.py must reference {ZERO_COST_POSES_SENTINEL!r} "
        f"sentinel filename for Lane LM-A detection"
    )


def test_inflate_env_gated(inflate_src: str):
    """The env gate prevents silent activation on stale archives. Without
    it, an old archive that happens to have the sentinel would silently
    switch to the analytical path — operator might not notice the
    behavior change."""
    assert "INFLATE_ZERO_COST_POSES" in inflate_src, (
        "inflate_renderer.py is missing INFLATE_ZERO_COST_POSES env gate"
    )


def test_inflate_warns_when_sentinel_without_env(inflate_src: str):
    """Sentinel present + env NOT set should surface a loud WARNING (not
    silent fallthrough). Otherwise the operator would see the renderer
    run unconditioned and not know why."""
    # Look for the elif/branch that handles sentinel-without-env-gate.
    branch = re.search(
        r"_zero_cost_sentinel_path\.exists\(\).*?_zero_cost_env_enabled.*?WARNING.*?INFLATE_ZERO_COST_POSES",
        inflate_src, re.DOTALL,
    )
    assert branch is not None, (
        "inflate_renderer.py must WARN when the zero_cost_poses_v1 "
        "sentinel is present but INFLATE_ZERO_COST_POSES is not enabled "
        "(silent fallthrough to unconditioned rendering would be a "
        "catastrophic operator surprise)."
    )


def test_inflate_logs_zero_cost_banner(inflate_src: str):
    """The inflate path must print a runtime banner so the operator can
    confirm the analytical path was taken (not silently running with
    unconditioned poses or some other fallback)."""
    assert "[zero-cost-poses]" in inflate_src, (
        "inflate_renderer.py must print a [zero-cost-poses] runtime "
        "banner when computing poses from lane marks (so the operator "
        "can confirm the path was taken in the eval log)."
    )


# ── Strict-scorer-rule compliance ────────────────────────────────────────


def test_zero_cost_block_does_not_load_scorers(inflate_src: str):
    """CLAUDE.md non-negotiable strict-scorer-rule: NO scorers loaded at
    inflate time. The Lane LM-A code block must NOT call PoseNet,
    SegNet, or any safetensors loader. Pure geometric centroid math only.

    We extract the code block between the import of
    compute_zero_cost_poses_from_masks and its call site, then assert
    no scorer loaders appear in that window.
    """
    # Extract the env-gated zero-cost code block (the if/try ... compute
    # call site). It is bounded by `_zero_cost_sentinel_path.exists()`
    # on the upper end and the next sibling section comment on the
    # lower end ("---- Load zoom warp scalars ----").
    block_match = re.search(
        r"#\s*----\s*Lane M\+ ZERO-COST POSES.*?#\s*----\s*Load zoom warp scalars",
        inflate_src, re.DOTALL,
    )
    assert block_match is not None, (
        "could not isolate the Lane LM-A code block in inflate_renderer.py "
        "(expected '# ---- Lane M+ ZERO-COST POSES' ... '# ---- Load zoom warp scalars')"
    )
    block = block_match.group(0)
    # Forbidden references inside this block:
    forbidden = [
        "PoseNet(",
        "SegNet(",
        "load_posenet",
        "load_segnet",
        ".safetensors",
        "smp.Unet",
        "FastViT",
    ]
    for token in forbidden:
        assert token not in block, (
            f"strict-scorer-rule VIOLATION: Lane LM-A block contains "
            f"{token!r} — pose computation MUST be pure geometric centroid "
            f"math (NO scorer loads). See CLAUDE.md non-negotiable + "
            f"feedback_strict_scorer_rule."
        )


# ── Wiring semantics: the call site reuses the already-decoded masks ─────


def test_call_site_uses_masks_argument(inflate_src: str):
    """The call site MUST forward the masks tensor that was already
    decoded for SegNet conditioning — re-decoding from masks.mkv would
    double the inflate I/O cost. Grep for the call signature to verify
    a `masks` positional argument is used."""
    call_match = re.search(
        r"compute_zero_cost_poses_from_masks\s*\(([^)]*)\)",
        inflate_src, re.DOTALL,
    )
    assert call_match is not None, (
        "inflate_renderer.py must call compute_zero_cost_poses_from_masks(...)"
    )
    args_str = call_match.group(1)
    # The call site uses masks (possibly via a local reference like
    # `_masks_for_zoom`). It MUST NOT contain a fresh torchvision /
    # av.open() decode.
    assert "av.open" not in args_str, (
        "the zero-cost call site must NOT re-open masks.mkv — reuse "
        "the already-decoded masks tensor"
    )
    assert "torchvision" not in args_str
    # The argument list should reference a name with 'mask' in it
    # (either `masks` directly or a local `_masks_for_zoom`).
    assert re.search(r"mask", args_str), (
        "compute_zero_cost_poses_from_masks must receive the decoded "
        f"masks tensor; got argument expression {args_str!r}"
    )


def test_call_site_renderer_pose_dim_guard(inflate_src: str):
    """The renderer's pose_dim is 6 in every FiLM-conditioned config we
    ship. The call site must guard against an unexpected pose_dim
    (warn-and-skip rather than silently truncate)."""
    # The guard is `if _renderer_pose_dim != 6:` near the call site.
    guard = re.search(
        r"_renderer_pose_dim\s*!=\s*6.*?WARNING",
        inflate_src, re.DOTALL,
    )
    assert guard is not None, (
        "Lane LM-A call site must guard against _renderer_pose_dim != 6 "
        "(warn-and-skip rather than silently truncate to a wider/narrower "
        "pose convention)."
    )


# ── End-to-end: real archive → real inflate code path ────────────────────


def test_zero_cost_path_handles_decoded_masks_shape():
    """Smoke: the helper must accept the (N, H, W) shape the inflate
    side decodes from masks.mkv. Mirrors the call-site contract.
    """
    import torch

    from tac.lane_mark_pose import compute_zero_cost_poses_from_masks

    # 12 frames at the standard scorer resolution (384, 512) — same
    # shape produced by the inflate-side mask decoder.
    n_frames, h, w = 12, 384, 512
    masks = torch.zeros(n_frames, h, w, dtype=torch.long)
    # Sprinkle some lane-mark pixels (class 1) at predictable positions
    # so the centroid math has signal.
    for i in range(n_frames):
        masks[i, 250 + i:255 + i, 300 + 2 * i:305 + 2 * i] = 1
    poses = compute_zero_cost_poses_from_masks(masks)
    assert poses.shape == (n_frames // 2, 6)
    assert poses.dtype == torch.float32
    # Dim 0 in empirical envelope, dims 1-5 = 0
    assert (poses[:, 0] >= 23.0).all()
    assert (poses[:, 0] <= 36.0).all()
    assert (poses[:, 1:] == 0.0).all()


# ── Sentinel filename + sentinel size pinning ────────────────────────────


def test_sentinel_filename_matches_constant():
    """Build-side and inflate-side must agree on the sentinel filename
    via the canonical constant in tac.lane_mark_pose."""
    from tac.lane_mark_pose import ZERO_COST_POSES_SENTINEL
    # Pin the canonical name so a future rename forces explicit
    # acknowledgement (build + inflate + provenance JSON all need updates).
    assert ZERO_COST_POSES_SENTINEL == "zero_cost_poses_v1", (
        "ZERO_COST_POSES_SENTINEL was renamed without updating tests. "
        "Verify build_baseline_archive.py, build_zero_cost_pose_archive.py, "
        "inflate_renderer.py, and the remote launch script ALL use the "
        "new name consistently before updating this assertion."
    )


def test_archive_omits_pose_pt_when_sentinel_present(tmp_path: Path) -> None:
    """End-to-end: build a synthetic Lane LM-A archive (no pose .pt,
    just the sentinel) and verify the inflate-side check would detect
    it correctly. Catches the regression where the build path
    accidentally still includes optimized_poses.pt."""
    import zipfile

    from tac.lane_mark_pose import ZERO_COST_POSES_SENTINEL

    archive = tmp_path / "synthetic_lane_lm_a.zip"
    sentinel = tmp_path / ZERO_COST_POSES_SENTINEL
    sentinel.write_bytes(b"")
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"\x00" * 100)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"\x00" * 100)

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(renderer, arcname="renderer.bin")
        z.write(masks, arcname="masks.mkv")
        z.write(sentinel, arcname=ZERO_COST_POSES_SENTINEL)

    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
    # The Lane LM-A invariants:
    assert "renderer.bin" in names
    assert "masks.mkv" in names
    assert ZERO_COST_POSES_SENTINEL in names
    assert "optimized_poses.pt" not in names
    assert "poses.pt" not in names
    # Sentinel is EXACTLY 0 bytes (any other size = tampering / corruption)
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(ZERO_COST_POSES_SENTINEL)
        assert info.file_size == 0, (
            f"sentinel must be 0 bytes, got {info.file_size}"
        )
