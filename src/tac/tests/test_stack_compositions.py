"""Lane J-NWCS × Lane EC composition regression tests.

Per Round 26/28 magnitude-anchor convention, every test asserts a SIGN/VALUE
behavior — never just "ran without error".

Round-28 review-rotation coverage:
  * #1 compositions don't introduce hidden state →
        test_compose_does_not_mutate_inputs
        test_jnwcs_ec_composition_is_deterministic
  * #2 bit-budget split correctness →
        test_bit_budget_split_search_produces_pareto_curve
        test_bit_budget_split_returns_grid_cross_product_size
  * #3 strict-scorer-rule preservation →
        test_inflate_path_loads_both_artifacts_without_scorer
        test_validate_rejects_renderer_with_scorer_loading_magic
  * #4 anchor reuse honesty →
        test_compose_does_not_silently_drop_either_artifact
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tac.stack_compositions import (
    REQUIRED_ARCHIVE_MEMBERS,
    BitBudgetSplit,
    bit_budget_split_search,
    compose_jnwcs_with_ec,
    validate_jnwcs_ec_composition,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _write_dummy_renderer_bin(path: Path, magic: bytes = b"ASYM",
                              payload_size: int = 200_000) -> None:
    """Write a small renderer.bin-like file with a chosen magic and
    enough random-ish payload to exercise the zip compresor.
    """
    body = magic + bytes(range(256)) * (payload_size // 256)
    path.write_bytes(body[:payload_size])


def _write_dummy_corrections_bin(path: Path, n_bytes: int = 30_000) -> None:
    """Write a Lane-EC-shaped sidecar payload (zlib-compressed sparse
    int8 deltas). For test purposes we just need a non-empty buffer of
    realistic size.
    """
    # Use a deterministic byte pattern so two compose calls produce the
    # same output (test_jnwcs_ec_composition_is_deterministic).
    seed = 0xEC
    buf = bytes([(seed * (i + 1)) & 0xFF for i in range(n_bytes)])
    path.write_bytes(buf)


def _write_dummy_masks(path: Path, n_bytes: int = 80_000) -> None:
    path.write_bytes(b"\x1aMKV" + b"\x00" * (n_bytes - 4))


def _write_dummy_poses(path: Path, n_bytes: int = 8_000) -> None:
    path.write_bytes(b"PT\x00\x00" + b"\x42" * (n_bytes - 4))


@pytest.fixture
def stack_inputs(tmp_path: Path) -> dict[str, Path]:
    """Standard set of input artifacts for the J-NWCS × EC stack."""
    inputs = {
        "renderer": tmp_path / "renderer_nwcs.bin",
        "corrections": tmp_path / "gradient_corrections.bin",
        "masks": tmp_path / "masks.mkv",
        "poses": tmp_path / "optimized_poses.pt",
    }
    _write_dummy_renderer_bin(inputs["renderer"], magic=b"ASYM")
    _write_dummy_corrections_bin(inputs["corrections"], n_bytes=30_000)
    _write_dummy_masks(inputs["masks"], n_bytes=80_000)
    _write_dummy_poses(inputs["poses"], n_bytes=8_000)
    return inputs


# ── 1. compose produces a loadable archive ───────────────────────────────


def test_compose_jnwcs_with_ec_produces_loadable_archive(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """The composition must yield a zip the standard inflate path can
    read: renderer.bin (NWCS-magic ASYM) AND gradient_corrections.bin
    must both be present and non-empty.
    """
    out_archive = tmp_path / "archive.zip"
    result = compose_jnwcs_with_ec(
        renderer_path=stack_inputs["renderer"],
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=out_archive,
    )
    assert result == out_archive
    assert out_archive.exists()
    assert out_archive.stat().st_size > 0

    with zipfile.ZipFile(out_archive) as zf:
        names = set(zf.namelist())
        # Both load-bearing artifacts must be present
        assert "renderer.bin" in names
        assert "gradient_corrections.bin" in names
        # And every required member
        assert set(REQUIRED_ARCHIVE_MEMBERS) <= names

        # Renderer.bin must carry the NWCS-compatible magic ASYM (or any
        # other scorer-free magic). Read first 4 bytes.
        with zf.open("renderer.bin") as fh:
            magic = fh.read(4)
        assert magic == b"ASYM", (
            f"renderer.bin magic should be ASYM (Lane G v3 anchor), got {magic!r}"
        )
        # Sidecar must be > 0 bytes (anti silent-drop)
        ec_info = zf.getinfo("gradient_corrections.bin")
        assert ec_info.file_size > 0


# ── 2. archive size fits 300KB-class budget on a synthetic anchor ────────


def test_compose_jnwcs_with_ec_archive_size_under_300kb(
    tmp_path: Path,
):
    """With a small synthetic Lane G v3 anchor + 30KB EC + tiny J-NWCS
    renderer, the composed archive must fit a 300KB budget.

    NON-ZERO anchor: archive_bytes > 0 (catches silent-empty bug).
    """
    renderer = tmp_path / "renderer_nwcs.bin"
    corrections = tmp_path / "gradient_corrections.bin"
    masks = tmp_path / "masks.mkv"
    poses = tmp_path / "optimized_poses.pt"
    # tiny J-NWCS renderer (~80KB) + 30KB EC + tiny masks.mkv (~70KB)
    # + 8KB poses + headers should fit comfortably under 300KB.
    _write_dummy_renderer_bin(renderer, magic=b"ASYM", payload_size=80_000)
    _write_dummy_corrections_bin(corrections, n_bytes=30_000)
    _write_dummy_masks(masks, n_bytes=70_000)
    _write_dummy_poses(poses, n_bytes=8_000)

    archive = tmp_path / "archive.zip"
    compose_jnwcs_with_ec(
        renderer_path=renderer,
        ec_corrections_path=corrections,
        masks_path=masks,
        poses_path=poses,
        output_archive_path=archive,
    )
    archive_bytes = archive.stat().st_size
    assert archive_bytes > 0  # NON-ZERO anchor
    # ZIP_DEFLATED + zeros-tail in masks/poses compresses heavily; assert
    # the COMPRESSED size is comfortably below 300KB.
    assert archive_bytes < 300_000, (
        f"archive {archive_bytes} bytes exceeded 300KB budget"
    )


# ── 3. strict-scorer-rule preservation (validate side) ───────────────────


def test_inflate_path_loads_both_artifacts_without_scorer(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """Validate succeeds for ASYM renderer + EC sidecar — neither
    artifact requires scorer load at inflate.
    """
    archive = tmp_path / "archive.zip"
    compose_jnwcs_with_ec(
        renderer_path=stack_inputs["renderer"],
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=archive,
    )
    summary = validate_jnwcs_ec_composition(archive)
    assert summary["compose_strategy"] == "jnwcs_with_ec"
    assert summary["renderer_magic"] == b"ASYM"
    assert summary["archive_bytes"] > 0
    # Both load-bearing artifacts non-empty
    assert summary["members"]["renderer.bin"] > 0
    assert summary["members"]["gradient_corrections.bin"] > 0


def test_validate_rejects_renderer_with_scorer_loading_magic(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """If renderer.bin starts with an unknown magic (proxy for a
    scorer-loading variant we don't allow), validation MUST raise.
    """
    bad_renderer = tmp_path / "renderer_bad.bin"
    # SCNR magic is intentionally not in _SCORER_FREE_RENDERER_MAGICS;
    # it stands in for a hypothetical "scorer-required" format.
    _write_dummy_renderer_bin(bad_renderer, magic=b"SCNR")

    archive = tmp_path / "archive.zip"
    compose_jnwcs_with_ec(
        renderer_path=bad_renderer,
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=archive,
    )
    with pytest.raises(ValueError, match="scorer-free allowlist"):
        validate_jnwcs_ec_composition(archive)


# ── 4. Pareto-curve correctness (Round-28 #2) ─────────────────────────────


def test_bit_budget_split_search_produces_pareto_curve():
    """Sweep produces multiple grid points, archive sizes monotone in
    weight bits at fixed ec_cap, monotone in ec_cap at fixed weight bits.
    """
    splits = bit_budget_split_search(target_archive_size=600_000, n_grid=5)
    assert isinstance(splits, list)
    # 5 weight-bit ladder × 5 ec-rate ladder = 25 grid points.
    assert len(splits) == 25, f"expected 25 grid points, got {len(splits)}"
    assert all(isinstance(s, BitBudgetSplit) for s in splits)

    # Sorted ascending (lower predicted score = better).
    predicted_scores = [s.predicted_score for s in splits]
    assert predicted_scores == sorted(predicted_scores)

    # Group by ec_rate_cap_bytes: archive_bytes monotone increasing in
    # weight_avg_bits.
    by_ec_cap: dict[int, list[BitBudgetSplit]] = {}
    for s in splits:
        by_ec_cap.setdefault(s.ec_rate_cap_bytes, []).append(s)
    for ec_cap, grp in by_ec_cap.items():
        grp_sorted = sorted(grp, key=lambda s: s.weight_avg_bits)
        sizes = [s.archive_bytes for s in grp_sorted]
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], (
                f"at ec_cap={ec_cap}, archive_bytes not monotone in "
                f"weight_avg_bits: {sizes}"
            )

    # Group by weight_avg_bits: archive_bytes monotone increasing in
    # ec_rate_cap_bytes.
    by_w_bits: dict[float, list[BitBudgetSplit]] = {}
    for s in splits:
        by_w_bits.setdefault(s.weight_avg_bits, []).append(s)
    for w, grp in by_w_bits.items():
        grp_sorted = sorted(grp, key=lambda s: s.ec_rate_cap_bytes)
        sizes = [s.archive_bytes for s in grp_sorted]
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1], (
                f"at weight_bits={w}, archive_bytes not monotone in "
                f"ec_rate_cap: {sizes}"
            )

    # Predicted_score at FIXED ec_cap should be monotone DECREASING in
    # weight_avg_bits up to the rate-knee. Because the heuristic linear
    # model has weight_distortion = 0 above 4.0, monotonicity holds at
    # weight_bits ≤ 4.0 if rate_term penalty is small at that grid range.
    # Test the lowest 3 weight-bit settings (2, 3, 4) at ec_cap = 30K.
    grp = by_ec_cap[30_000]
    grp_sorted = sorted(grp, key=lambda s: s.weight_avg_bits)
    low_bits_scores = [
        s.predicted_score for s in grp_sorted if s.weight_avg_bits <= 4.0
    ]
    assert len(low_bits_scores) >= 2
    for i in range(1, len(low_bits_scores)):
        # NON-ZERO anchor + monotone-decreasing
        assert low_bits_scores[i] < low_bits_scores[i - 1], (
            f"predicted_score not monotone-decreasing at low weight "
            f"bits with fixed ec_cap=30K: {low_bits_scores}"
        )


def test_bit_budget_split_returns_grid_cross_product_size():
    """Custom grids of different sizes return |W| × |E| splits."""
    weight_grid = [2.0, 4.0, 6.0, 8.0]
    ec_grid = [5_000, 25_000, 45_000]
    splits = bit_budget_split_search(
        target_archive_size=900_000,
        n_grid=4,
        weight_bits_grid=weight_grid,
        ec_rate_cap_grid=ec_grid,
    )
    assert len(splits) == len(weight_grid) * len(ec_grid)
    pairs = {(s.weight_avg_bits, s.ec_rate_cap_bytes) for s in splits}
    assert pairs == {(w, e) for w in weight_grid for e in ec_grid}


# ── 5. silent-drop guard (Round-28 #4 + Round-28 #1) ─────────────────────


def test_compose_does_not_silently_drop_either_artifact(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """If either critical artifact is missing or empty, the compose
    function MUST raise — not produce a half-archive.
    """
    archive = tmp_path / "archive.zip"

    # Missing renderer
    missing_path = tmp_path / "missing_renderer.bin"
    with pytest.raises(FileNotFoundError, match="renderer.bin"):
        compose_jnwcs_with_ec(
            renderer_path=missing_path,
            ec_corrections_path=stack_inputs["corrections"],
            masks_path=stack_inputs["masks"],
            poses_path=stack_inputs["poses"],
            output_archive_path=archive,
        )

    # Missing EC sidecar
    missing_ec = tmp_path / "missing_corrections.bin"
    with pytest.raises(FileNotFoundError, match="gradient_corrections.bin"):
        compose_jnwcs_with_ec(
            renderer_path=stack_inputs["renderer"],
            ec_corrections_path=missing_ec,
            masks_path=stack_inputs["masks"],
            poses_path=stack_inputs["poses"],
            output_archive_path=archive,
        )

    # Empty input file
    empty_renderer = tmp_path / "empty_renderer.bin"
    empty_renderer.write_bytes(b"")
    with pytest.raises(ValueError, match="empty input"):
        compose_jnwcs_with_ec(
            renderer_path=empty_renderer,
            ec_corrections_path=stack_inputs["corrections"],
            masks_path=stack_inputs["masks"],
            poses_path=stack_inputs["poses"],
            output_archive_path=archive,
        )


# ── 6. Determinism (Round-28 #1) ─────────────────────────────────────────


def test_jnwcs_ec_composition_is_deterministic(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """Same inputs → byte-identical archives across two compose calls.

    This protects against ZipInfo-timestamp drift (codex R5-r6 #5) and
    is the canonical "compose function doesn't mutate inputs" anchor.
    """
    archive_a = tmp_path / "archive_a.zip"
    archive_b = tmp_path / "archive_b.zip"
    compose_jnwcs_with_ec(
        renderer_path=stack_inputs["renderer"],
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=archive_a,
    )
    compose_jnwcs_with_ec(
        renderer_path=stack_inputs["renderer"],
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=archive_b,
    )
    digest_a = hashlib.sha256(archive_a.read_bytes()).hexdigest()
    digest_b = hashlib.sha256(archive_b.read_bytes()).hexdigest()
    assert digest_a == digest_b, (
        f"compose_jnwcs_with_ec is non-deterministic: "
        f"{digest_a!r} != {digest_b!r}"
    )


# ── 7. Round-28 #1: compose does not mutate input files ─────────────────


def test_compose_does_not_mutate_inputs(
    tmp_path: Path, stack_inputs: dict[str, Path]
):
    """The compose function reads inputs as-is; their on-disk SHA must
    not change after composition.
    """
    pre_sha = {
        k: hashlib.sha256(p.read_bytes()).hexdigest()
        for k, p in stack_inputs.items()
    }
    archive = tmp_path / "archive.zip"
    compose_jnwcs_with_ec(
        renderer_path=stack_inputs["renderer"],
        ec_corrections_path=stack_inputs["corrections"],
        masks_path=stack_inputs["masks"],
        poses_path=stack_inputs["poses"],
        output_archive_path=archive,
    )
    post_sha = {
        k: hashlib.sha256(p.read_bytes()).hexdigest()
        for k, p in stack_inputs.items()
    }
    assert pre_sha == post_sha, (
        f"compose_jnwcs_with_ec mutated inputs: "
        f"pre={pre_sha} post={post_sha}"
    )
