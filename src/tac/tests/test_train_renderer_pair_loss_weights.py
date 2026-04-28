# LOSS_CONVERGENCE_NOT_REQUIRED: this is a CLI-wiring test for
# --pair-loss-weights argparse plumbing (per feedback_dead_flag_wiring_pattern),
# NOT a loss-function mathematical test. Convergence checks live in the
# train_renderer integration tests, not here.
"""Lane W: tests for train_renderer.py --pair-loss-weights wiring.

Per memory feedback_dead_flag_wiring_pattern: every CLI flag must be
introspected against the target's argparse BEFORE wiring it into a
subprocess call. This test suite locks in:

  1. The flag actually exists in train_renderer's parser.
  2. parse_args plumbs args.pair_loss_weights through.
  3. The training loop file READS it via getattr() / Path() (not a typo).
  4. The per-step loss accumulation actually multiplies by the weight
     for the current pair_idx_int (not pair_idx itself, not start).
  5. Validation: shape mismatch / non-finite / negative weights raise
     loud (not warn-and-skip — that was the R3-3 silent-skip class).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TRAIN_PATH = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


@pytest.fixture(scope="module")
def train_src() -> str:
    return TRAIN_PATH.read_text()


def test_train_renderer_has_pair_loss_weights_flag(train_src):
    """The flag must exist in argparse with --pair-loss-weights spelling."""
    assert re.search(
        r'p\.add_argument\(\s*"--pair-loss-weights"',
        train_src,
    ), (
        "train_renderer.py argparse must declare --pair-loss-weights so "
        "remote_lane_w_hard_pair_self_compress.sh can pass it without "
        "argparse silently rejecting it (memory feedback_dead_flag_wiring_pattern)."
    )


def test_parse_args_exposes_pair_loss_weights():
    """parse_args(['--tag', 'smoke']) yields args.pair_loss_weights == None."""
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402

    a = parse_args(["--tag", "smoke"])
    assert hasattr(a, "pair_loss_weights"), (
        "args.pair_loss_weights missing — argparse hyphen-to-underscore "
        "translation broke."
    )
    assert a.pair_loss_weights is None  # default

    a2 = parse_args(["--tag", "smoke", "--pair-loss-weights", "/tmp/w.pt"])
    assert a2.pair_loss_weights == "/tmp/w.pt"


def test_loader_validates_shape(train_src):
    """The loader block must explicitly validate shape == (n_total,) and raise."""
    # We expect a `if _plw.ndim != 1 or _plw.shape[0] != n_total:` followed
    # by a raise — not a warn-and-default.
    assert re.search(
        r"_plw\.ndim\s*!=\s*1\s*or\s*_plw\.shape\[0\]\s*!=\s*n_total",
        train_src,
    ), "shape validation missing — silent-skip class regression."
    # Same loader block must `raise ValueError` not just `print`.
    loader_block = re.search(
        r"# Lane W: optional per-pair loss weights.*?print\(\s*f?\"\[train\] Lane W:.*?\)",
        train_src,
        re.DOTALL,
    )
    assert loader_block is not None, "Lane W loader block not found"
    block = loader_block.group(0)
    assert "raise ValueError" in block or "raise FileNotFoundError" in block, (
        "Lane W loader must HARD-FAIL on bad input. Warn-and-skip is the "
        "exact silent-skip class CLAUDE.md forbids."
    )


def test_loader_validates_finiteness(train_src):
    """NaN/Inf in the weight tensor would silently corrupt training."""
    assert re.search(r"torch\.isfinite\(_plw\)", train_src), (
        "missing finiteness check on pair_loss_weights — NaN/Inf would "
        "silently destroy training."
    )


def test_loader_rejects_negatives(train_src):
    """Negative weights would invert gradient direction — must reject."""
    assert "(_plw < 0).any()" in train_src, (
        "missing negative-weight check — flipping gradient sign on a hard "
        "pair would WORSEN that pair every step."
    )


def test_per_step_scaling_uses_pair_idx_int(train_src):
    """Per-step scaling must index by pair_idx_int (the canonical 0..n_total-1
    pair id), NOT by pair_idx (the perm tensor element) and NOT by start
    (the frame index)."""
    # Find the 'if pair_loss_weights is not None:' block and verify it
    # indexes by pair_idx_int specifically.
    m = re.search(
        r"if pair_loss_weights is not None:.*?loss = loss \* _w",
        train_src,
        re.DOTALL,
    )
    assert m is not None, "Lane W per-step scaling block not found"
    block = m.group(0)
    assert "pair_loss_weights[pair_idx_int]" in block, (
        "per-step scaling must index by pair_idx_int, not pair_idx (the "
        "perm tensor) and not start (the frame index). Mis-indexing would "
        "silently apply the wrong weight to each pair."
    )


def test_scaling_applied_after_rate_penalty(train_src):
    """Per-pair scaling must multiply the FINAL loss (after the SC rate
    penalty has been added). Otherwise the SC bit-allocation gradient
    won't see the per-pair weight — defeating the entire premise."""
    # The order in train_renderer.py must be:
    #   loss = loss + _rate_pen        # (SC rate penalty)
    #   if pair_loss_weights is not None: loss = loss * _w  # (Lane W)
    #   scaled_loss = loss / accum
    rate_idx = train_src.find("loss = loss + _rate_pen")
    lane_w_idx = train_src.find("if pair_loss_weights is not None:")
    accum_idx = train_src.find("scaled_loss = loss / accum")
    assert rate_idx > 0, "rate penalty block not found"
    assert lane_w_idx > 0, "Lane W per-pair scaling block not found"
    assert accum_idx > 0, "scaled_loss assignment not found"
    assert rate_idx < lane_w_idx < accum_idx, (
        "Order must be: rate_pen → pair_loss_weights → scaled_loss. "
        "If Lane W comes BEFORE the rate penalty, the SC bit-allocation "
        "gradient ignores the per-pair weight (defeats the whole premise). "
        "If it comes AFTER scaled_loss, accum is wrong."
    )


def test_no_iteration_order_change(train_src):
    """Lane W must not change the deterministic per-pair iteration order.

    The per-pair training loop iterates `for step, pair_idx in enumerate(perm):`
    and the perm is sampled once per epoch. Lane W must NOT alter this
    sampling — otherwise reproducibility breaks.
    """
    # Make sure we did NOT touch the existing perm sampling block.
    assert "perm = torch.randperm(n_total)[:train_size]" in train_src, (
        "Legacy uniform sampling must remain intact for non-Phase-5 paths."
    )
    assert "perm = torch.multinomial(probs, train_size, replacement=False)" in train_src, (
        "Phase 5 hard-pair sampling must remain intact (Lane W is *additive*: "
        "it weights the LOSS, it does not change which pairs are sampled)."
    )


def test_pair_loss_weights_doc_mentions_full_score_formula(train_src):
    """Per CLAUDE.md feedback_curriculum_must_use_full_score: any hard-pair
    feature MUST document that it uses the FULL score formula (100*seg +
    sqrt(10*pose)), not PoseNet alone."""
    # Look for the help text string
    m = re.search(
        r'add_argument\(\s*"--pair-loss-weights".*?help=\(?\s*"([^"]+(?:"\s*"[^"]+)*)"',
        train_src,
        re.DOTALL,
    )
    assert m is not None, "could not locate --pair-loss-weights help text"
    # The DOCSTRING/comment block above the loader is where the formula lives;
    # check the comment too.
    assert "100*seg + sqrt(10*pose)" in train_src or "100*seg_i + sqrt(10*pose_i)" in train_src, (
        "Lane W must document that the contribution metric uses the FULL "
        "score formula (memory: feedback_curriculum_must_use_full_score). "
        "PoseNet-only ranking would miss the SegNet 77x weight."
    )


def test_no_silent_warn_and_continue(train_src):
    """The Lane W loader must not WARN-and-continue on bad input. Same
    silent-skip class as R3-3 (auth-eval-on-best skip-if-missing-flag)."""
    # No 'continue' or 'pass' immediately after the loader's 'except ValueError'
    # for our shape check.
    bad = re.search(
        r"shape.*?does not match.*?\n[^\n]*print\([^\n]*WARN[^\n]*\n[^\n]*continue",
        train_src,
        re.IGNORECASE,
    )
    assert bad is None, (
        "Lane W loader must hard-fail on shape mismatch, not warn-and-continue."
    )


def test_argparse_help_text_documents_consumer(train_src):
    """The CLI help must point to the canonical producer
    (profile_pair_sensitivity.py) so future operators don't have to
    spelunk for it."""
    m = re.search(
        r'add_argument\(\s*"--pair-loss-weights".*?help=(.*?)\)\s*\n',
        train_src,
        re.DOTALL,
    )
    assert m is not None
    help_blob = m.group(1)
    assert "profile_pair_sensitivity" in help_blob, (
        "help text must reference experiments/profile_pair_sensitivity.py "
        "(the canonical producer) so the link is discoverable from --help."
    )
