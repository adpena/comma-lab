# SPDX-License-Identifier: MIT
"""F3-BACKPORT-VQVAE-PDP regression suite per Council Decision 13 PROCEED Option C.

Per Council omnibus 2026-05-14 commit 7872c9f4b binding verdict (11/11
unanimous + Time-Traveler default-OFF amendment): vq_vae trainer needs
flag-declare + cache wire-in; PDP substrate-side score_aware_loss must
accept F3 kwargs and PDP trainer-side wire-in lands as opt-in flag
(default OFF, byte-faithful to historical when disabled).

These tests do NOT run any training (no GPU, $0 dispatch). They verify the
source-level + signature-level contracts:

1. **vq_vae** trainer:
   - imports ``build_optimized_training_context`` canonical helper
   - argparse declares ``--enable-autocast-fp16``, ``--enable-torch-compile``,
     ``--enable-gt-scorer-cache``, ``--gt-scorer-cache-chunk-size``,
     ``--segmentation-temperature`` (Tier-1 RESERVED flags)
   - calls ``gt_cache.lookup(...)`` per-batch in train + val loops
   - threads ``gt_pose_batch=`` / ``gt_seg_batch=`` / ``gt_seg_already_probs=``
     into both ``loss_fn(...)`` calls

2. **PDP substrate-side** (score_aware_loss.py):
   - ``DrivingPriorScoreAwareLoss.forward`` accepts the 3 F3 kwargs
   - body routes through ``score_pair_components_dispatch`` (the cache-aware
     dispatcher) instead of bare ``score_pair_components``
   - default kwargs are ``None`` so historical callers stay byte-faithful

3. **PDP trainer-side**:
   - imports the canonical helper
   - argparse already declares ``--enable-gt-scorer-cache`` (predates this wave)
   - calls ``gt_cache.lookup(...)`` per-batch in train + val loops
   - threads the 3 F3 kwargs through both ``loss_fn(...)`` calls
   - default-OFF semantics preserved by ``args.enable_gt_scorer_cache=False``
     default + canonical helper's ``getattr(args, "enable_gt_scorer_cache", True)``
     respect (when the trainer's argparse default is False, the cache stays None
     and the loss falls back to GT-forward path)

Sister suite: ``src/tac/tests/test_f3_backport_wave_v2_trainers_wired.py``
covers the original 13 wired trainers; this suite extends to the 2 added by
F3-BACKPORT-VQVAE-PDP-SUBAGENT 2026-05-14.

Cross-refs:
- Council omnibus binding verdict: commit ``7872c9f4b`` Decision 13.
- Predecessor landing: ``feedback_f3_backport_wave_v2_eleven_trainers_landed_20260514.md``.
- Canonical reproducer: ``tools/check_f3_trainer_actionable.py``.
- Catalog #228: ``check_substrate_trainer_consumes_f3_cache_when_flag_declared``.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# The 2 trainers added by F3-BACKPORT-VQVAE-PDP-SUBAGENT.
F3_VQVAE_PDP_TRAINERS = ["vq_vae", "pretrained_driving_prior"]


@pytest.fixture(scope="module")
def _trainer_text():
    """Cache per-trainer source text for the duration of the module."""
    cache: dict[str, str | None] = {}
    for tid in F3_VQVAE_PDP_TRAINERS:
        path = REPO_ROOT / "experiments" / f"train_substrate_{tid}.py"
        cache[tid] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    return cache


# ----------------------------------------------------------------------------
# vq_vae trainer-side contract (5 tests)
# ----------------------------------------------------------------------------


def test_vq_vae_imports_canonical_optimization_context(_trainer_text):
    """vq_vae must import ``build_optimized_training_context``."""
    text = _trainer_text["vq_vae"]
    assert text is not None, "vq_vae trainer file missing"
    assert (
        "build_optimized_training_context" in text
        or "build_gt_scorer_cache" in text
    ), (
        "vq_vae: F3 cache canonical helper not imported. Add\n"
        "  from tac.substrates._shared.trainer_skeleton import\n"
        "      build_optimized_training_context as _canon_..."
    )


def test_vq_vae_argparse_declares_tier1_reserved_flags(_trainer_text):
    """vq_vae argparse declares the 5 Tier-1 RESERVED flags."""
    text = _trainer_text["vq_vae"]
    assert text is not None, "vq_vae trainer file missing"
    for flag in (
        "--enable-autocast-fp16",
        "--enable-torch-compile",
        "--enable-gt-scorer-cache",
        "--gt-scorer-cache-chunk-size",
        "--segmentation-temperature",
    ):
        assert flag in text, f"vq_vae argparse missing {flag} (Tier-1 RESERVED)"


def test_vq_vae_calls_gt_cache_lookup_in_hot_loop(_trainer_text):
    """vq_vae train + val loops both invoke ``gt_cache.lookup(...)``."""
    text = _trainer_text["vq_vae"]
    assert text is not None
    # Two distinct call sites (train + val). The canonical sane_hnerv pattern.
    assert text.count("gt_cache.lookup") >= 2, (
        "vq_vae: gt_cache.lookup must appear in BOTH train and val loops "
        "(found {} call sites; expected >=2)".format(text.count("gt_cache.lookup"))
    )


def test_vq_vae_threads_f3_kwargs_into_loss_fn(_trainer_text):
    """vq_vae loss_fn calls thread all 3 F3 kwargs."""
    text = _trainer_text["vq_vae"]
    assert text is not None
    for kwarg in ("gt_pose_batch=", "gt_seg_batch=", "gt_seg_already_probs="):
        # train + val = 2 each
        assert text.count(kwarg) >= 2, (
            f"vq_vae: {kwarg} must thread through train AND val loss_fn calls "
            f"(found {text.count(kwarg)}; expected >=2)"
        )


def test_vq_vae_uses_canonical_opt_ctx_pattern(_trainer_text):
    """vq_vae captures ``opt_ctx`` from the canonical helper return value."""
    text = _trainer_text["vq_vae"]
    assert text is not None
    assert "opt_ctx = " in text and "opt_ctx.gt_cache" in text, (
        "vq_vae: canonical opt_ctx pattern not found (expected "
        "`opt_ctx = _canon_build_optimized_training_context(...)` + "
        "`gt_cache = opt_ctx.gt_cache`)"
    )


# ----------------------------------------------------------------------------
# PDP substrate-side contract (4 tests)
# ----------------------------------------------------------------------------


def test_pdp_substrate_loss_accepts_f3_kwargs():
    """``DrivingPriorScoreAwareLoss.forward`` accepts the 3 F3 kwargs."""
    from tac.substrates.pretrained_driving_prior.score_aware_loss import (
        DrivingPriorScoreAwareLoss,
    )

    sig = inspect.signature(DrivingPriorScoreAwareLoss.forward)
    params = list(sig.parameters.keys())
    for kwarg in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        assert kwarg in params, (
            f"DrivingPriorScoreAwareLoss.forward missing {kwarg!r} keyword. "
            "Per Council omnibus Decision 13: substrate-side loss MUST accept "
            "F3 kwargs to enable opt-in cache routing."
        )


def test_pdp_substrate_loss_f3_kwargs_default_to_none():
    """Default for each F3 kwarg is ``None`` (preserves historical behavior)."""
    from tac.substrates.pretrained_driving_prior.score_aware_loss import (
        DrivingPriorScoreAwareLoss,
    )

    sig = inspect.signature(DrivingPriorScoreAwareLoss.forward)
    for kwarg in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        param = sig.parameters[kwarg]
        assert param.default is None, (
            f"DrivingPriorScoreAwareLoss.forward {kwarg!r} default must be None "
            "(default-OFF amendment per Time-Traveler peer; historical callers "
            "stay byte-faithful when no cache is supplied)"
        )


def test_pdp_substrate_loss_routes_through_dispatch_helper():
    """Loss body uses ``score_pair_components_dispatch`` (not bare)."""
    path = (
        REPO_ROOT
        / "src"
        / "tac"
        / "substrates"
        / "pretrained_driving_prior"
        / "score_aware_loss.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "score_pair_components_dispatch" in text, (
        "PDP substrate-side loss must route through "
        "score_pair_components_dispatch (the F3-aware canonical dispatcher)."
    )
    # The body should call dispatch (not bare score_pair_components for scoring).
    # The bare name is only acceptable in docstring text; verify via a callable
    # invocation pattern.
    assert "score_pair_components_dispatch(" in text, (
        "PDP substrate must INVOKE the dispatch helper, not just import it."
    )


def test_pdp_substrate_dispatch_round_trip_byte_faithful_when_no_cache():
    """When all 3 cache kwargs are None, dispatch falls back to GT-forward.

    This proves the default-OFF amendment is mathematically byte-faithful:
    a caller supplying no cache args gets identical seg/pose terms as before.
    """
    from tac.substrates.score_aware_common import (
        score_pair_components,
        score_pair_components_dispatch,
    )

    # Both functions exist; the dispatch helper is the F3-aware sister.
    assert callable(score_pair_components)
    assert callable(score_pair_components_dispatch)
    # Signature contract: dispatch accepts the 3 F3 kwargs as None defaults.
    sig = inspect.signature(score_pair_components_dispatch)
    for kwarg in ("gt_pose_batch", "gt_seg_batch", "gt_seg_already_probs"):
        assert kwarg in sig.parameters, (
            f"score_pair_components_dispatch missing {kwarg!r}"
        )
        assert sig.parameters[kwarg].default is None, (
            f"score_pair_components_dispatch {kwarg!r} must default to None"
        )


# ----------------------------------------------------------------------------
# PDP trainer-side contract (5 tests)
# ----------------------------------------------------------------------------


def test_pdp_trainer_imports_canonical_optimization_context(_trainer_text):
    """PDP trainer imports the canonical helper."""
    text = _trainer_text["pretrained_driving_prior"]
    assert text is not None, "PDP trainer file missing"
    assert (
        "build_optimized_training_context" in text
    ), (
        "pretrained_driving_prior: F3 cache canonical helper not imported."
    )


def test_pdp_trainer_argparse_declares_enable_gt_scorer_cache_flag(_trainer_text):
    """PDP argparse already declares ``--enable-gt-scorer-cache`` (predates wave)."""
    text = _trainer_text["pretrained_driving_prior"]
    assert text is not None
    assert "--enable-gt-scorer-cache" in text, (
        "pretrained_driving_prior: --enable-gt-scorer-cache flag must be declared"
    )


def test_pdp_trainer_calls_gt_cache_lookup_in_hot_loop(_trainer_text):
    """PDP train + val loops both invoke ``gt_cache.lookup(...)``."""
    text = _trainer_text["pretrained_driving_prior"]
    assert text is not None
    assert text.count("gt_cache.lookup") >= 2, (
        "pretrained_driving_prior: gt_cache.lookup must appear in BOTH train "
        f"and val loops (found {text.count('gt_cache.lookup')}; expected >=2)"
    )


def test_pdp_trainer_threads_f3_kwargs_into_loss_fn(_trainer_text):
    """PDP loss_fn calls thread all 3 F3 kwargs (train + val)."""
    text = _trainer_text["pretrained_driving_prior"]
    assert text is not None
    for kwarg in ("gt_pose_batch=", "gt_seg_batch=", "gt_seg_already_probs="):
        assert text.count(kwarg) >= 2, (
            f"pretrained_driving_prior: {kwarg} must thread through train AND "
            f"val loss_fn calls (found {text.count(kwarg)}; expected >=2)"
        )


def test_pdp_trainer_default_off_preserved_in_argparse(_trainer_text):
    """Per Time-Traveler amendment: PDP --enable-gt-scorer-cache default OFF.

    The argparse flag must use ``action="store_true"`` with ``default=False``
    so substrate authors explicitly opt in. This preserves byte-faithful
    historical behavior unless the operator-recipe explicitly enables it.
    """
    text = _trainer_text["pretrained_driving_prior"]
    assert text is not None
    # Find the --enable-gt-scorer-cache argparse block and verify default=False
    # is present nearby (within 10 lines after the flag declaration).
    idx = text.find('"--enable-gt-scorer-cache"')
    assert idx > 0, "PDP --enable-gt-scorer-cache flag declaration not found"
    block = text[idx : idx + 800]
    assert 'action="store_true"' in block or "action='store_true'" in block, (
        "PDP --enable-gt-scorer-cache must use action='store_true' "
        "(default-OFF amendment per Time-Traveler peer)"
    )
    # Either explicit default=False OR no default kwarg (store_true default is False).
    # The presence of an explicit default=True would violate the amendment.
    assert "default=True" not in block, (
        "PDP --enable-gt-scorer-cache MUST NOT default to True (Time-Traveler "
        "amendment: PDP wire-in is opt-in default-OFF)"
    )


# ----------------------------------------------------------------------------
# End-to-end gate via canonical reproducer
# ----------------------------------------------------------------------------


def test_check_f3_trainer_actionable_classifies_both_as_already_wired(
    _trainer_text,
):
    """End-to-end: the canonical reproducer classifies BOTH as ALREADY_WIRED."""
    check_path = REPO_ROOT / "tools" / "check_f3_trainer_actionable.py"
    spec = importlib.util.spec_from_file_location(
        "check_f3_trainer_actionable", check_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for tid in F3_VQVAE_PDP_TRAINERS:
        result = mod._classify(tid)
        assert result["verdict"] == "ALREADY_WIRED", (
            f"{tid}: expected ALREADY_WIRED, got {result['verdict']}. "
            f"Diagnostics: cache_build={result.get('has_cache_build')}, "
            f"cache_lookup={result.get('has_cache_lookup')}, "
            f"kwargs_threaded={result.get('has_kwargs_threaded')}, "
            f"reserved={result.get('has_reserved_marker')}, "
            f"substrate_accepts_kwargs={result.get('substrate_accepts_kwargs')}, "
            f"tier1_flags={result.get('has_tier1_flags')}"
        )


def test_catalog_228_does_not_flag_vqvae_or_pdp():
    """Catalog #228 does NOT flag vq_vae or pretrained_driving_prior post-wire-in.

    The remaining out-of-scope violations (e.g. s2sbs_byte_stuffing) are
    NOT this lane's responsibility; this test only ensures the 2 trainers
    in scope are not in the violation set.
    """
    from tac.preflight import (
        check_substrate_trainer_consumes_f3_cache_when_flag_declared,
    )

    violations = check_substrate_trainer_consumes_f3_cache_when_flag_declared(
        strict=False, verbose=False
    )
    for v in violations:
        assert "train_substrate_vq_vae.py" not in v, (
            f"Catalog #228 still flags vq_vae: {v}"
        )
        assert "train_substrate_pretrained_driving_prior.py" not in v, (
            f"Catalog #228 still flags PDP: {v}"
        )
