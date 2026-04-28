"""End-to-end tests for the Lane M-V3 (Path A) pose-from-embedding
inflate dispatch.

Lane M-V3 wires ``submissions/robust_current/inflate_renderer.py`` to
predict per-pair 6-DOF poses at INFLATE TIME from a distilled MLP when:

  1. The archive contains the ``pose_from_embedding_v1`` 0-byte sentinel.
  2. The archive contains the companion ``pose_from_embedding_v1.pt`` MLP.
  3. The archive does NOT contain ``optimized_poses.pt`` / ``poses.pt``.

This file pins:

  * The source-grep regression checks (catch a future refactor that drops
    the wiring or introduces a stale env-gate).
  * The strict-scorer-rule compliance — the inflate path MUST NOT load
    PoseNet or SegNet for the prediction. We grep the relevant code
    block to verify (no .safetensors load, no PoseNet(), etc.).
  * The HARD-FAIL semantics: sentinel without companion weights should
    raise (NOT silently fall through to unconditioned rendering).
  * The MLP runs on mask features alone — zero embedding input matches
    the dropout-trained inflate regime.
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


def test_inflate_imports_pose_from_embedding(inflate_src: str):
    """The inflate path must import load_mlp + POSENET_EMBEDDING_DIM
    from tac.pose_from_embedding. A future refactor that drops the
    import would silently break Lane M-V3."""
    assert "from tac.pose_from_embedding import" in inflate_src, (
        "inflate_renderer.py is missing the tac.pose_from_embedding import "
        "for the Lane M-V3 dispatch"
    )
    assert "load_mlp as _load_pose_emb_mlp" in inflate_src, (
        "inflate_renderer.py must alias load_mlp to _load_pose_emb_mlp "
        "(naming convention used by the dispatch block)"
    )


def test_inflate_references_sentinel_filename(inflate_src: str):
    """The sentinel filename must match the canonical
    POSE_FROM_EMBEDDING_SENTINEL constant exactly."""
    from tac.pose_from_embedding import POSE_FROM_EMBEDDING_SENTINEL
    assert POSE_FROM_EMBEDDING_SENTINEL in inflate_src, (
        f"inflate_renderer.py must reference {POSE_FROM_EMBEDDING_SENTINEL!r} "
        f"sentinel filename for Lane M-V3 detection"
    )


def test_inflate_references_weights_filename(inflate_src: str):
    """The companion weights filename must be referenced literally so a
    future filename rename forces an explicit edit (catches silent drift)."""
    from tac.pose_from_embedding import POSE_FROM_EMBEDDING_WEIGHTS_FILENAME
    assert POSE_FROM_EMBEDDING_WEIGHTS_FILENAME in inflate_src, (
        f"inflate_renderer.py must reference "
        f"{POSE_FROM_EMBEDDING_WEIGHTS_FILENAME!r} weights filename"
    )


def test_inflate_logs_pose_from_embedding_banner(inflate_src: str):
    """The inflate path must print a runtime banner so the operator can
    confirm the MLP path was taken (not silently running with
    unconditioned poses or some other fallback)."""
    assert "[pose-from-embedding]" in inflate_src, (
        "inflate_renderer.py must print a [pose-from-embedding] runtime "
        "banner when predicting poses with the MLP (so the operator "
        "can confirm the path was taken in the eval log)."
    )


# ── Strict-scorer-rule compliance ────────────────────────────────────────


def test_pose_from_embedding_block_does_not_load_scorers(inflate_src: str):
    """CLAUDE.md non-negotiable strict-scorer-rule: NO scorers loaded at
    inflate time. The Lane M-V3 code block must NOT call PoseNet,
    SegNet, or any safetensors loader. Pure MLP forward only.

    We extract the code block between the section comment and the next
    sibling section, then assert no scorer loaders appear in that window.
    """
    # Run-inflate variant block boundaries.
    block_match = re.search(
        r"#\s*----\s*Lane M-V3 POSE-FROM-EMBEDDING.*?(?=#\s*----\s*Load zoom warp scalars|#\s*----\s*Generate renderer frames)",
        inflate_src, re.DOTALL,
    )
    assert block_match is not None, (
        "could not isolate the Lane M-V3 code block in inflate_renderer.py "
        "(expected '# ---- Lane M-V3 POSE-FROM-EMBEDDING' section)"
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
        "load_default_scorers",
        "load_differentiable_scorers",
    ]
    for token in forbidden:
        assert token not in block, (
            f"strict-scorer-rule VIOLATION: Lane M-V3 block contains "
            f"{token!r} — pose prediction MUST be a pure MLP forward "
            f"(NO scorer loads). See CLAUDE.md non-negotiable + "
            f"feedback_strict_scorer_rule."
        )


def test_pose_from_embedding_block_uses_zero_embedding(inflate_src: str):
    """The MLP was distilled with embedding-dropout so the inflate path
    MUST use a zero embedding input (the dropout-trained regime). A
    non-zero embedding at inflate would be out-of-distribution."""
    # The dispatch block must construct torch.zeros for the embedding.
    block_match = re.search(
        r"#\s*----\s*Lane M-V3 POSE-FROM-EMBEDDING.*?(?=#\s*----\s*Load zoom warp scalars|#\s*----\s*Generate renderer frames)",
        inflate_src, re.DOTALL,
    )
    assert block_match is not None
    block = block_match.group(0)
    assert "torch.zeros" in block, (
        "Lane M-V3 inflate path must construct a zero embedding tensor "
        "(the dropout-trained inflate regime). Without this the MLP "
        "would run on garbage state and produce out-of-distribution poses."
    )
    assert "_PNET_EMB_DIM" in block, (
        "Lane M-V3 inflate path must import POSENET_EMBEDDING_DIM "
        "(aliased _PNET_EMB_DIM) to size the zero embedding correctly."
    )


# ── HARD-FAIL semantics ─────────────────────────────────────────────────


def test_sentinel_without_weights_hard_fails(inflate_src: str):
    """A sentinel without the companion weights file is a corrupt archive.
    The dispatch must raise instead of silently running unconditioned."""
    block_match = re.search(
        r"_pose_emb_sentinel_path\.exists\(\).*?(?:RuntimeError|raise\s+RuntimeError)",
        inflate_src, re.DOTALL,
    )
    assert block_match is not None, (
        "Lane M-V3 inflate path must `raise RuntimeError(...)` when the "
        "sentinel is present but pose_from_embedding_v1.pt is missing. "
        "Silent fallthrough to unconditioned rendering is forbidden."
    )


def test_sentinel_without_masks_hard_fails(inflate_src: str):
    """The MLP feature extractor needs the mask tensor. Without it the
    dispatch must raise (not silently fall through)."""
    # Look for an `elif` that catches sentinel + masks is None
    elif_block = re.search(
        r"_pose_emb_sentinel_path\.exists\(\)\s*\n\s*and\s+masks\s+is\s+None.*?raise\s+RuntimeError",
        inflate_src, re.DOTALL,
    )
    assert elif_block is not None, (
        "Lane M-V3 inflate path must `raise RuntimeError` when the "
        "sentinel is present but masks tensor is None. Silent fallthrough "
        "to unconditioned rendering is forbidden."
    )


# ── Renderer pose_dim guard ─────────────────────────────────────────────


def test_call_site_renderer_pose_dim_guard(inflate_src: str):
    """The renderer's pose_dim is 6 in every FiLM-conditioned config we
    ship. The call site must guard against an unexpected pose_dim
    (warn-and-skip rather than silently truncate)."""
    # The guard `_renderer_pose_dim != 6` appears in the new block too.
    # Find a region that contains both the new section comment AND the guard.
    section = re.search(
        r"#\s*----\s*Lane M-V3 POSE-FROM-EMBEDDING.*?_renderer_pose_dim\s*!=\s*6.*?WARNING",
        inflate_src, re.DOTALL,
    )
    assert section is not None, (
        "Lane M-V3 call site must guard against _renderer_pose_dim != 6 "
        "(warn-and-skip rather than silently truncate to a wider/narrower "
        "pose convention)."
    )


# ── Mini-TTO inflate variant has the same wiring ─────────────────────────


def test_minitto_variant_also_dispatches(inflate_src: str):
    """The mini-TTO inflate path is a separate function that mirrors
    run_inflate(). It must ALSO have the Lane M-V3 dispatch (otherwise
    a mini-TTO eval would silently skip the MLP and run unconditioned)."""
    # Count how many times the section header appears — there should be
    # at least 2 (run_inflate variant + mini-TTO variant).
    n = len(re.findall(r"#\s*----\s*Lane M-V3 POSE-FROM-EMBEDDING", inflate_src))
    assert n >= 2, (
        f"Lane M-V3 dispatch only appears in {n} place(s); expected at "
        f"least 2 (run_inflate variant + mini-TTO inflate variant)."
    )


# ── Sentinel filename + sentinel size pinning ────────────────────────────


def test_sentinel_filename_matches_constant():
    from tac.pose_from_embedding import POSE_FROM_EMBEDDING_SENTINEL
    assert POSE_FROM_EMBEDDING_SENTINEL == "pose_from_embedding_v1", (
        "POSE_FROM_EMBEDDING_SENTINEL was renamed without updating tests. "
        "Verify distill_pose_from_embedding.py, inflate_renderer.py, and "
        "the remote launch script ALL use the new name consistently."
    )


def test_archive_omits_pose_pt_when_sentinel_present(tmp_path: Path) -> None:
    """End-to-end: build a synthetic Lane M-V3 archive (no pose .pt,
    just sentinel + MLP weights) and verify the inflate-side check
    would detect it correctly."""
    import zipfile

    from tac.pose_from_embedding import (
        POSE_FROM_EMBEDDING_SENTINEL,
        POSE_FROM_EMBEDDING_WEIGHTS_FILENAME,
        PoseFromEmbeddingMLP,
        save_mlp,
    )

    archive = tmp_path / "synthetic_lane_m_v3.zip"
    sentinel = tmp_path / POSE_FROM_EMBEDDING_SENTINEL
    sentinel.write_bytes(b"")
    weights = tmp_path / POSE_FROM_EMBEDDING_WEIGHTS_FILENAME
    save_mlp(PoseFromEmbeddingMLP(), weights, fp16=True)
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"\x00" * 100)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"\x00" * 100)

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(renderer, arcname="renderer.bin")
        z.write(masks, arcname="masks.mkv")
        z.write(weights, arcname=POSE_FROM_EMBEDDING_WEIGHTS_FILENAME)
        z.write(sentinel, arcname=POSE_FROM_EMBEDDING_SENTINEL)

    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
    # Lane M-V3 invariants:
    assert "renderer.bin" in names
    assert "masks.mkv" in names
    assert POSE_FROM_EMBEDDING_SENTINEL in names
    assert POSE_FROM_EMBEDDING_WEIGHTS_FILENAME in names
    assert "optimized_poses.pt" not in names
    assert "poses.pt" not in names
    # Sentinel is EXACTLY 0 bytes
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(POSE_FROM_EMBEDDING_SENTINEL)
        assert info.file_size == 0
