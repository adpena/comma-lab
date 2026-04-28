"""Regression tests for tac.parametrize_strip — the canonical strip helper.

Caused by `project_lane_i_crashed_parametrize_strip_20260428` (Lane I lost
~$1.60 because Stage 3 export had its own inline strip that didn't match).
"""
from __future__ import annotations

import torch

from tac.parametrize_strip import strip_parametrize_hooks, has_parametrize_keys


def test_no_strip_needed_returns_copy() -> None:
    """Plain state passes through unchanged."""
    state = {"layer.weight": torch.zeros(3), "layer.bias": torch.ones(3)}
    out = strip_parametrize_hooks(state)
    assert sorted(out.keys()) == ["layer.bias", "layer.weight"]
    # Verify it's a copy (modifying out doesn't change input)
    out["new"] = torch.zeros(1)
    assert "new" not in state


def test_original_key_renamed_to_plain_weight() -> None:
    """The canonical mapping: parametrizations.weight.original → weight."""
    val = torch.tensor([1.5, 2.5])
    state = {"layer.parametrizations.weight.original": val}
    out = strip_parametrize_hooks(state)
    assert "layer.weight" in out
    assert "layer.parametrizations.weight.original" not in out
    assert torch.equal(out["layer.weight"], val)


def test_codebook_dropped_by_default() -> None:
    """Internal parametrize keys (codebook, etc.) are dropped."""
    state = {
        "layer.parametrizations.weight.original": torch.zeros(1),
        "layer.parametrizations.weight.0.codebook": torch.full((1,), 7.0),
        "layer.parametrizations.weight.0._buffers": torch.zeros(2),
    }
    out = strip_parametrize_hooks(state)
    assert sorted(out.keys()) == ["layer.weight"]


def test_drop_internal_false_preserves_all() -> None:
    """drop_internal=False keeps codebook/_buffers in output (unusual case)."""
    state = {
        "layer.parametrizations.weight.original": torch.zeros(1),
        "layer.parametrizations.weight.0.codebook": torch.full((1,), 7.0),
    }
    out = strip_parametrize_hooks(state, drop_internal=False)
    assert "layer.weight" in out
    assert "layer.parametrizations.weight.0.codebook" in out


def test_mixed_state_partially_stripped() -> None:
    """Plain keys + parametrize keys — plain pass through, parametrize stripped."""
    state = {
        "renderer.head.weight": torch.zeros(1),
        "renderer.head.bias": torch.zeros(1),
        "renderer.conv1.parametrizations.weight.original": torch.ones(1),
        "renderer.conv1.parametrizations.weight.0.codebook": torch.full((1,), 7.0),
    }
    out = strip_parametrize_hooks(state)
    assert sorted(out.keys()) == [
        "renderer.conv1.weight", "renderer.head.bias", "renderer.head.weight",
    ]
    assert out["renderer.conv1.weight"].item() == 1.0


def test_has_parametrize_keys_detects() -> None:
    """has_parametrize_keys returns True iff any .parametrizations. key present."""
    plain = {"a.weight": torch.zeros(1)}
    assert has_parametrize_keys(plain) is False
    parametrized = {"a.parametrizations.weight.original": torch.zeros(1)}
    assert has_parametrize_keys(parametrized) is True
    mixed = {"a.weight": torch.zeros(1), "b.parametrizations.weight.original": torch.zeros(1)}
    assert has_parametrize_keys(mixed) is True


def test_empty_state() -> None:
    """Empty input → empty output, no error."""
    assert strip_parametrize_hooks({}) == {}
    assert has_parametrize_keys({}) is False


def test_lane_i_crash_exact_reproduction() -> None:
    """Reproduce the exact key set from the Lane I Stage 3 crash log.
    Reference: project_lane_i_crashed_parametrize_strip_20260428."""
    # Per the crash log, these are the exact unexpected keys observed:
    state = {
        "renderer.class_embed.parametrizations.weight.original": torch.zeros(5, 6),
        "renderer.class_embed.parametrizations.weight.0.codebook": torch.zeros(16),
        "renderer.decoder.0.parametrizations.weight.original": torch.zeros(8, 32, 3, 3),
        "renderer.decoder.0.parametrizations.weight.0.codebook": torch.zeros(16),
        "renderer.decoder.2.parametrizations.weight.original": torch.zeros(8, 16, 3, 3),
        "renderer.decoder.2.parametrizations.weight.0.codebook": torch.zeros(16),
        "renderer.decoder.4.parametrizations.weight.original": torch.zeros(3, 8, 3, 3),
        "renderer.decoder.4.parametrizations.weight.0.codebook": torch.zeros(16),
        "motion.embedding.parametrizations.weight.original": torch.zeros(5, 6),
        "motion.embedding.parametrizations.weight.0.codebook": torch.zeros(16),
        "motion.stem.0.parametrizations.weight.original": torch.zeros(16, 12, 3, 3),
        "motion.stem.0.parametrizations.weight.0.codebook": torch.zeros(16),
    }
    out = strip_parametrize_hooks(state)
    # All 6 missing keys from the crash log should now be present:
    expected = {
        "renderer.class_embed.weight", "renderer.decoder.0.weight",
        "renderer.decoder.2.weight", "renderer.decoder.4.weight",
        "motion.embedding.weight", "motion.stem.0.weight",
    }
    assert expected.issubset(out.keys())
    # No parametrize internals leak through
    assert not any(".parametrizations." in k for k in out.keys())


# ─────────────────────────────────────────────────────────────────────────
# Round 11 codex-review edge cases (2026-04-28)
# ─────────────────────────────────────────────────────────────────────────


def test_root_level_parametrize_key_renamed() -> None:
    """ROOT-level `parametrizations.weight.original` → `weight`.

    PyTorch emits keys like `parametrizations.weight.original` (NO leading
    path) when `register_parametrization` is applied directly on the root
    nn.Module. The substring-based detection from the original implementation
    missed this — Codex Round 11 finding.
    """
    state = {"parametrizations.weight.original": torch.tensor([1.5])}
    out = strip_parametrize_hooks(state)
    assert "weight" in out
    assert "parametrizations.weight.original" not in out
    assert out["weight"].item() == 1.5


def test_mixed_root_and_nested() -> None:
    """ROOT and nested parametrize keys coexist correctly."""
    state = {
        "parametrizations.bias.original": torch.tensor([0.5]),
        "renderer.conv.parametrizations.weight.original": torch.ones(1),
        "plain.bias": torch.zeros(1),
    }
    out = strip_parametrize_hooks(state)
    assert sorted(out.keys()) == ["bias", "plain.bias", "renderer.conv.weight"]
    assert out["bias"].item() == 0.5


def test_multi_original_dropped_with_warning(recwarn) -> None:
    """weight_norm-style `original0`/`original1` dropped with UserWarning.

    PyTorch's parametrize.remove_parametrizations would COMPUTE the combined
    weight at strip time. We can't do that statically without instantiating
    the parametrize class — so we drop and warn.
    """
    from tac.parametrize_strip import reset_warning_cache
    reset_warning_cache()
    state = {
        "linear.parametrizations.weight.original0": torch.ones(3),  # weight_norm magnitude
        "linear.parametrizations.weight.original1": torch.zeros(3, 4),  # weight_norm direction
        "linear.bias": torch.zeros(3),
    }
    out = strip_parametrize_hooks(state)
    # linear.weight should NOT be present (caller must reconstruct via
    # parametrize.remove_parametrizations on a live module)
    assert "linear.weight" not in out
    assert "linear.parametrizations.weight.original0" not in out
    assert "linear.parametrizations.weight.original1" not in out
    # linear.bias passes through
    assert "linear.bias" in out
    # Warning was emitted (once per unique (path, name) pair)
    assert any("multi-original" in str(w.message) for w in recwarn.list)


def test_multi_original_warning_disabled() -> None:
    """warn_multi_original=False suppresses the warning."""
    import warnings as _w
    from tac.parametrize_strip import reset_warning_cache
    reset_warning_cache()
    state = {"linear.parametrizations.weight.original0": torch.ones(3)}
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = strip_parametrize_hooks(state, warn_multi_original=False)
        assert not any("multi-original" in str(w.message) for w in caught)


def test_nested_parametrize_chain() -> None:
    """`parametrizations.weight.0.parametrizations.weight.original` uses outermost.

    Nested parametrize chains can occur if a parametrize wraps an already-
    parametrized layer. The OUTERMOST `parametrizations` marker is the one
    that produces the unwrapped weight; the inner chain is dropped.
    """
    state = {
        # Outer parametrize wraps an inner parametrized weight; outermost
        # `original` would reconstruct the original weight after unwrap.
        # Per our policy, the outer `original` rename is canonical.
        "layer.parametrizations.weight.original": torch.ones(1),
        # Inner parametrize internals — dropped as parametrize internals
        "layer.parametrizations.weight.0.parametrizations.weight.original": torch.zeros(1),
        "layer.parametrizations.weight.0.parametrizations.weight.0.codebook": torch.zeros(16),
    }
    out = strip_parametrize_hooks(state)
    assert "layer.weight" in out
    assert out["layer.weight"].item() == 1.0
    # Inner chain dropped
    assert all("parametrizations" not in k for k in out.keys())


def test_has_parametrize_keys_root_level() -> None:
    """has_parametrize_keys() detects ROOT-level keys (no leading dot)."""
    assert has_parametrize_keys({"parametrizations.weight.original": torch.zeros(1)}) is True
    assert has_parametrize_keys({"plain.weight": torch.zeros(1)}) is False
    # Edge: a key that contains 'parametrizations' as a substring of a name
    # but NOT as a path component
    assert has_parametrize_keys({"my_parametrizations_layer.weight": torch.zeros(1)}) is False


def test_path_with_numeric_components() -> None:
    """Paths like `decoder.0.parametrizations.weight.original` work correctly."""
    state = {"decoder.0.parametrizations.weight.original": torch.ones(1)}
    out = strip_parametrize_hooks(state)
    assert "decoder.0.weight" in out


def test_short_parametrize_key_not_a_param() -> None:
    """A key like `parametrizations.weight` (only 2 components) is malformed
    — no suffix → not a valid parametrize key — should pass through."""
    state = {"parametrizations.weight": torch.zeros(1)}
    out = strip_parametrize_hooks(state)
    # Either passes through OR dropped as malformed; either is acceptable
    # as long as no spurious key is created
    assert "weight" not in out


def test_pytorch_register_parametrization_real_layout() -> None:
    """Verify against a REAL torch.nn parametrize layout.

    Round 11 finding: codex flagged that we should test with actual PyTorch
    parametrize, not just synthetic strings. This validates the strip pattern
    matches what PyTorch actually emits.
    """
    import torch.nn as nn
    import torch.nn.utils.parametrize as P

    class Identity(nn.Module):
        def forward(self, x):
            return x

    m = nn.Linear(3, 4, bias=True)
    P.register_parametrization(m, "weight", Identity())
    state = m.state_dict()
    # Sanity: PyTorch generated parametrize keys
    assert any("parametrizations.weight.original" in k for k in state)
    out = strip_parametrize_hooks(state)
    # After strip: plain `weight` + `bias`
    assert "weight" in out
    assert "bias" in out
    assert not any("parametrizations" in k for k in out)
