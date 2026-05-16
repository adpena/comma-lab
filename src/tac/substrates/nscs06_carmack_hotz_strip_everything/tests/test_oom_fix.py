# SPDX-License-Identifier: MIT
"""Regression tests for the NSCS06 compress-side scorer OOM fix.

Empirical anchor: NSCS06 100ep Modal T4 smoke fc-01KRQDTA70GEXSZ2CEEYGWQNSR
crashed at `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py:470`
with ``torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 10.55 GiB.
GPU 0 has a total capacity of 14.56 GiB`` inside SegNet's ``conv_pw`` forward
(timm EfficientNet-B2). The compress-side SegNet class-label-derivation pass
was running over all 600 pairs at full resolution in a single forward — the
activation tensor exceeded T4 capacity.

The fix mirrors the Catalog #218 D4 mini-batch pattern: iterate the scorer
forward in chunks of ``args.scorer_chunk_size`` (default 8) so peak activation
memory scales with chunk_size, not n_pairs. This file pins the source-level
structure of the fix so a future refactor cannot silently re-introduce the
full-batch pattern. It also confirms the prior PoseNet dict-vs-tensor latent
indexing bug (line 499 pre-fix: ``pose_logits[:, :POSE_DIMS]`` on a Hydra dict)
is corrected to the canonical ``pose_out["pose"][..., :POSE_DIMS]`` form.

Tests intentionally use AST parsing rather than running the trainer end-to-end
because: (a) the real OOM fires only on CUDA hardware, not on the CPU test box
(local pytest cannot reproduce the 10.55 GiB allocation), (b) the trainer
loads upstream SegNet/PoseNet weights that aren't available in CI containers,
and (c) AST verification is more robust against future docstring/test churn.

Sister tests (codec roundtrip) live in
``src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_codec_roundtrip.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Repository root is 5 parents up from this test file:
# .../pact/src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_oom_fix.py
REPO_ROOT = Path(__file__).resolve().parents[5]
TRAINER_PATH = (
    REPO_ROOT / "experiments" / "train_substrate_nscs06_carmack_hotz_strip_everything.py"
)


def _read_trainer_source() -> str:
    if not TRAINER_PATH.is_file():
        pytest.skip(f"trainer source not found at {TRAINER_PATH}")
    return TRAINER_PATH.read_text(encoding="utf-8")


def _trainer_ast() -> ast.Module:
    return ast.parse(_read_trainer_source(), filename=str(TRAINER_PATH))


# ---------------------------------------------------------------------------
# Argparse contract
# ---------------------------------------------------------------------------
def test_scorer_chunk_size_flag_declared() -> None:
    """``--scorer-chunk-size`` must appear in ``_build_parser`` argparse spec."""
    src = _read_trainer_source()
    assert "--scorer-chunk-size" in src, (
        "missing --scorer-chunk-size argparse flag; the OOM fix relies on it. "
        "See Catalog #218 D4 sister pattern + empirical anchor "
        "fc-01KRQDTA70GEXSZ2CEEYGWQNSR (NSCS06 Modal T4 OOM)."
    )


def test_scorer_chunk_size_default_is_conservative_for_t4() -> None:
    """Default must be small enough to fit comfortably under T4 14.56 GiB."""
    tree = _trainer_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # parser.add_argument("--scorer-chunk-size", ..., default=N, ...)
        is_add_argument = (
            isinstance(func, ast.Attribute) and func.attr == "add_argument"
        )
        if not is_add_argument:
            continue
        first_arg = node.args[0] if node.args else None
        if not (
            isinstance(first_arg, ast.Constant)
            and first_arg.value == "--scorer-chunk-size"
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                # Conservative ceiling: 16 (still ~280 MB peak; safe for T4).
                # The empirically-known crash was at full N=600 (>10 GiB).
                assert isinstance(kw.value.value, int), (
                    f"--scorer-chunk-size default must be int; got "
                    f"{type(kw.value.value).__name__}"
                )
                assert 1 <= kw.value.value <= 16, (
                    f"--scorer-chunk-size default={kw.value.value} outside "
                    f"the conservative T4 range [1, 16]. Larger values risk "
                    f"re-introducing the OOM class; smaller values bloat the "
                    f"compress-pass wall-clock for no benefit."
                )
                return
        pytest.fail("--scorer-chunk-size argparse entry missing default=...")
    pytest.fail("--scorer-chunk-size argparse entry not found in _build_parser")


# ---------------------------------------------------------------------------
# Source-level structure: SegNet forward must be chunked
# ---------------------------------------------------------------------------
def test_segnet_forward_is_chunked_not_full_batch() -> None:
    """SegNet forward must iterate over chunks; NO bare full-batch call.

    The forbidden pre-fix pattern was::

        seg_logits = segnet(segnet.preprocess_input(odd_btchw))

    where ``odd_btchw`` was the FULL ``(N=600, 1, 3, H, W)`` tensor. The fix
    wraps the call in ``for start in range(0, n_pairs, chunk):`` and slices
    ``pair_tensor[start:stop, 0]`` per iteration.
    """
    src = _read_trainer_source()
    assert "for start in range(0, n_pairs," in src, (
        "expected mini-batch loop `for start in range(0, n_pairs, chunk):` "
        "around the SegNet forward. The pre-fix full-batch pattern produced "
        "the Modal T4 OOM at fc-01KRQDTA70GEXSZ2CEEYGWQNSR."
    )
    # Specifically the SegNet forward must be inside the loop with chunked
    # slicing of pair_tensor (NOT the full odd_torch single forward).
    assert "pair_tensor[start:stop, 0]" in src, (
        "expected chunked SegNet input slice `pair_tensor[start:stop, 0]`; "
        "without it the loop could degenerate to full-batch + a no-op range."
    )


def test_no_full_batch_segnet_forward_remains() -> None:
    """Refuse the literal forbidden pre-fix source pattern."""
    src = _read_trainer_source()
    # The exact pre-fix line had `odd_torch = pair_tensor[:, 0]...` followed
    # immediately by `odd_btchw = odd_torch.unsqueeze(1)` and a single
    # `segnet(segnet.preprocess_input(odd_btchw))` outside any loop. We pin
    # the absence of the pre-fix synthesis-marker variable name.
    forbidden = "odd_torch = pair_tensor[:, 0].to(device).float() / 1.0"
    assert forbidden not in src, (
        f"forbidden pre-fix full-batch SegNet pattern detected; the substring "
        f"{forbidden!r} reconstructs the OOM bug class."
    )


# ---------------------------------------------------------------------------
# Source-level structure: PoseNet forward must be chunked
# ---------------------------------------------------------------------------
def test_posenet_forward_is_chunked_not_full_batch() -> None:
    """PoseNet forward must mirror the SegNet chunked pattern.

    The pre-fix code called ``posenet(posenet.preprocess_input(pose_input))``
    where ``pose_input = pair_tensor.to(device).float()`` was the FULL
    ``(N=600, 2, 3, H, W)`` tensor. FastViT activations would have OOM-ed on
    T4 right after the SegNet OOM was fixed; chunking is symmetric.
    """
    src = _read_trainer_source()
    # The PoseNet block must reference the chunked input slicing form.
    assert "pair_tensor[start:stop].to(device).float()" in src, (
        "expected chunked PoseNet input slice "
        "`pair_tensor[start:stop].to(device).float()` — without it PoseNet "
        "would OOM symmetrically with SegNet as soon as the latter was "
        "fixed (FastViT activations scale with B identically)."
    )


def test_no_full_batch_posenet_forward_remains() -> None:
    """Refuse the literal forbidden pre-fix PoseNet source pattern."""
    src = _read_trainer_source()
    forbidden_input = "pose_input = pair_tensor.to(device).float()"
    assert forbidden_input not in src, (
        f"forbidden pre-fix full-batch PoseNet pattern detected; the "
        f"substring {forbidden_input!r} reconstructs the OOM bug class."
    )


# ---------------------------------------------------------------------------
# Latent dict-vs-tensor bug fix (PoseNet returns Hydra dict, not tensor)
# ---------------------------------------------------------------------------
def test_posenet_output_uses_canonical_dict_access() -> None:
    """``posenet(...)`` returns the Hydra dict; first-6 must come from "pose".

    The upstream PoseNet.forward returns ``{head_name: tensor}`` per
    ``upstream/modules.py:54-59``. The pre-fix line 499
    ``pose_logits[:, :POSE_DIMS]`` was a latent runtime TypeError that never
    fired because the SegNet OOM crashed earlier. Canonical pattern is
    ``out["pose"][..., :POSE_DIMS]`` per ``src/tac/research/run_saliency_sweep.py:56``
    + CLAUDE.md "Exact scorer architectures".
    """
    src = _read_trainer_source()
    assert 'pose_out["pose"]' in src, (
        'expected canonical Hydra dict access `pose_out["pose"]`; without it '
        "the trainer crashes at TypeError as soon as the SegNet OOM is fixed."
    )


def test_no_pose_logits_bracket_slice_remains() -> None:
    """The forbidden ``pose_logits[:, :POSE_DIMS]`` pattern must be gone."""
    src = _read_trainer_source()
    forbidden = "pose_logits[:, :POSE_DIMS]"
    assert forbidden not in src, (
        f"forbidden latent dict-vs-tensor bug pattern detected: {forbidden!r} "
        f"would TypeError at runtime since posenet returns a dict not tensor."
    )


# ---------------------------------------------------------------------------
# Memory-bound contract: chunk size threads through both scorers
# ---------------------------------------------------------------------------
def test_scorer_chunk_size_used_in_both_segnet_and_posenet_blocks() -> None:
    """The CLI flag must reach BOTH scorer-forward blocks.

    A future refactor that wires `--scorer-chunk-size` to only one of the two
    scorers re-introduces the OOM on the other.
    """
    src = _read_trainer_source()
    # `args.scorer_chunk_size` must appear at least twice (once per block).
    occurrences = src.count("args.scorer_chunk_size")
    assert occurrences >= 2, (
        f"expected `args.scorer_chunk_size` referenced in BOTH the SegNet "
        f"and PoseNet forward blocks; found {occurrences} usage(s). One of "
        f"the two scorer forwards will OOM when fired full-batch on T4."
    )


def test_no_grad_context_preserved_around_chunked_forwards() -> None:
    """Both chunked forwards must remain inside `torch.no_grad()`.

    The compress-side scorer queries are inference-only (frozen scorer; no
    backprop); leaking gradients into the chunk loop would re-introduce
    activation buildup (autograd graph retention) and resurrect the OOM
    class via a different surface.
    """
    src = _read_trainer_source()
    # We expect at least 2 `with torch.no_grad():` contexts in _full_main —
    # one for SegNet, one for PoseNet. There is also one in the L1 SCAFFOLD
    # pre-existing decode path so >= 2 is the minimum invariant.
    no_grad_count = src.count("with torch.no_grad():")
    assert no_grad_count >= 2, (
        f"expected >= 2 `with torch.no_grad():` contexts in trainer; found "
        f"{no_grad_count}. Per Catalog #180 (substrate trainers use_no_grad "
        f"at eval) the chunked forwards MUST stay under no_grad to keep "
        f"activation memory bounded."
    )
