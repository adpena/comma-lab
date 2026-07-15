# SPDX-License-Identifier: MIT
"""#509 batch 3: ComputeDtype DSL Lever custody tests.

The lever composes ``--compute-dtype {bf16,fp16}`` (+ optional
``--compute-dtype-quality-check N``) — the mixed-precision compute seam (fp32 masters;
low-precision witness fwd/bwd only). Trainer default is fp32 (seam never constructed,
byte-identical); the lever exists so the flags are DSL-held (never hand-typed) and the
OFF arm is compose-nothing."""
from __future__ import annotations

import pytest

from tac.witness_dsl.curriculum_dsl import ComputeDtype


def test_lever_name_and_flag():
    lv = ComputeDtype("bf16")
    assert lv.name == "compute_dtype_seam"
    assert lv.overrides == {"--compute-dtype": "bf16"}


def test_default_is_bf16_no_qc():
    lv = ComputeDtype()
    assert lv.overrides == {"--compute-dtype": "bf16"}
    assert "--compute-dtype-quality-check" not in lv.overrides


def test_fp16_arm():
    assert ComputeDtype("fp16").overrides["--compute-dtype"] == "fp16"


def test_quality_check_composes_when_positive():
    lv = ComputeDtype("bf16", quality_check=50)
    assert lv.overrides["--compute-dtype-quality-check"] == 50


def test_quality_check_zero_omitted():
    assert "--compute-dtype-quality-check" not in ComputeDtype("bf16", 0).overrides


@pytest.mark.parametrize("d", ["fp32", "float32", "int8", ""])
def test_noop_or_unknown_dtype_refused(d):
    # off-is-orphan: fp32 is the incumbent — compose nothing instead of an inert lever;
    # unknown dtypes are never-invent-values.
    with pytest.raises(ValueError, match="bf16"):
        ComputeDtype(d)


def test_negative_quality_check_refused():
    with pytest.raises(ValueError, match=">= 0"):
        ComputeDtype("bf16", quality_check=-1)


def test_trainer_flags_exist():
    """Never-invent-flags: both emitted flags must exist in the levelset trainer argparse."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    src = (repo / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert re.search(r"add_argument\(\s*\"--compute-dtype\"", src)
    assert re.search(r"add_argument\(\s*\"--compute-dtype-quality-check\"", src)
