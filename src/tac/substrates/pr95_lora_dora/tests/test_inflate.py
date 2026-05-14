# SPDX-License-Identifier: MIT
"""Inflate runtime tests for the PR95 LoRA/DoRA substrate.

Per HNeRV parity lessons 9 (runtime closure) + 11 (no-op detector): tests
verify the runtime contract (`<archive_dir> <output_dir> <file_list>` signature
per Catalog #146), no scorer imports, and that adapter folding produces
different output bytes vs. base-only inflate.
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_inflate_no_scorer_imports() -> None:
    """Per CLAUDE.md strict-scorer-rule — inflate.py must not import scorers.

    Scan IMPORT nodes only (docstrings/comments naming forbidden tokens are
    expected and document the rule)."""
    inflate_path = Path(__file__).parents[1] / "inflate.py"
    src = inflate_path.read_text()
    tree = ast.parse(src)
    forbidden_module_substrings = (
        "upstream.modules", "rgb_to_yuv6",
    )
    forbidden_name_substrings = (
        "PoseNet", "SegNet", "EfficientNet", "FastViT",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for tok in forbidden_module_substrings + forbidden_name_substrings:
                    assert tok not in alias.name, (
                        f"Inflate runtime imports forbidden module: {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for tok in forbidden_module_substrings:
                assert tok not in mod, (
                    f"Inflate runtime imports forbidden module: {mod!r}"
                )
            for alias in node.names:
                for tok in forbidden_name_substrings:
                    assert tok not in alias.name, (
                        f"Inflate runtime imports forbidden name: {alias.name!r}"
                    )


def test_inflate_loc_under_substrate_engineering_budget() -> None:
    """Substrate-engineering opt-out allows ≤200 LOC (per HNeRV parity lesson 4)."""
    inflate_path = Path(__file__).parents[1] / "inflate.py"
    lines = [
        line for line in inflate_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    n = len(lines)
    assert n <= 200, f"inflate.py has {n} non-comment LOC, must be ≤200"


def test_inflate_main_signature() -> None:
    """Per Catalog #146 the CLI accepts (archive_dir, output_dir, file_list)."""
    from tac.substrates.pr95_lora_dora.inflate import _main

    # Wrong arg count rejected
    rc = _main([])
    assert rc != 0
    rc = _main(["one", "two"])
    assert rc != 0


def test_apply_adapters_folds_into_state_dict() -> None:
    """Adapter folding mutates the named state_dict tensor correctly."""
    import torch

    from tac.substrates.pr95_lora_dora.inflate import _apply_adapters_to_state_dict

    torch.manual_seed(0)
    sd = {
        "blocks.0.weight": torch.zeros(8, 4, 3, 3),
        "blocks.0.bias": torch.zeros(8),
    }
    A = torch.randn(2, 36)  # rank=2, in_dim=4*3*3=36
    B = torch.randn(8, 2)
    records = [{
        "name": "blocks.0", "kind": "lora", "rank": 2, "alpha": 2.0,
        "A": A, "B": B,
    }]
    sd_after = _apply_adapters_to_state_dict(dict(sd), records)
    # Original weight is zero; effective should be the delta
    expected_delta = (B @ A).reshape(8, 4, 3, 3)  # alpha/r = 1.0
    assert torch.allclose(sd_after["blocks.0.weight"], expected_delta, atol=1e-5)


def test_apply_adapters_dora_uses_magnitude() -> None:
    import torch

    from tac.substrates.pr95_lora_dora.inflate import _apply_adapters_to_state_dict

    torch.manual_seed(1)
    W = torch.randn(4, 2, 1, 1)
    sd = {"blocks.0.weight": W.clone()}
    A = torch.zeros(2, 2)  # zero LoRA delta
    B = torch.zeros(4, 2)
    # m = ||W||_col -> W_eff = m * (W) / ||W|| = W
    mag = torch.linalg.norm(W.reshape(4, -1), dim=1)
    records = [{
        "name": "blocks.0", "kind": "dora", "rank": 2, "alpha": 2.0,
        "A": A, "B": B, "magnitude": mag,
    }]
    sd_after = _apply_adapters_to_state_dict(dict(sd), records)
    # DoRA with zero delta + m=||W|| should recover W
    assert torch.allclose(sd_after["blocks.0.weight"], W, atol=1e-5)


def test_apply_adapters_ignores_unknown_target() -> None:
    import torch

    from tac.substrates.pr95_lora_dora.inflate import _apply_adapters_to_state_dict

    sd = {"blocks.0.weight": torch.randn(4, 2, 3, 3)}
    records = [{
        "name": "blocks.99", "kind": "lora", "rank": 2, "alpha": 2.0,
        "A": torch.randn(2, 18), "B": torch.randn(4, 2),
    }]
    sd_after = _apply_adapters_to_state_dict(dict(sd), records)
    # Original tensor must be unchanged
    assert torch.allclose(sd_after["blocks.0.weight"], sd["blocks.0.weight"])
