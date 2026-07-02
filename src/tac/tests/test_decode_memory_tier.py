# SPDX-License-Identifier: MIT
"""#224 Wave D — tests for the DECODE (inflate) memory-tier surface.

Verify: the 3-value knob resolves; contest tiers carry the bit-exact contract; production_edge is
NON-contest + REFUSED by the contest path; the default tier's inflate env is 1-thread BLAS (== the
inflate.py's own setdefault => byte-identical) with no worker override; and the tier drives the
downstream eval device (explicit wins).
"""

from __future__ import annotations

import pytest

from tac import decode_memory_tier as dmt


def test_default_tier_is_decode_cpu_16gb():
    assert dmt.DEFAULT_TIER_NAME == "decode_cpu_16gb"
    t = dmt.resolve_tier(None)
    assert t.name == "decode_cpu_16gb"
    assert t.contest is True and t.bit_exact_contract is True
    assert t.contest_eval_device == "cpu"


def test_three_tiers_present_and_distinct():
    names = set(dmt.DECODE_MEMORY_TIERS)
    assert names == {"decode_cpu_16gb", "decode_t4_16gb", "production_edge"}
    # the two contest tiers are DISTINCT by downstream eval device (not padded)
    assert dmt.resolve_tier("decode_cpu_16gb").contest_eval_device == "cpu"
    assert dmt.resolve_tier("decode_t4_16gb").contest_eval_device == "cuda"


def test_unknown_tier_raises():
    with pytest.raises(dmt.DecodeTierError):
        dmt.resolve_tier("decode_gpu_80gb")


def test_contest_tiers_are_bit_exact():
    for name in ("decode_cpu_16gb", "decode_t4_16gb"):
        t = dmt.require_contest_tier(dmt.resolve_tier(name))  # must not raise
        assert t.contest and t.bit_exact_contract


def test_production_edge_is_non_contest_and_not_bit_exact():
    t = dmt.resolve_tier("production_edge")
    assert t.contest is False and t.bit_exact_contract is False
    assert t.contest_eval_device is None


def test_production_edge_refused_by_contest_path():
    with pytest.raises(dmt.DecodeTierError) as ei:
        dmt.require_contest_tier(dmt.resolve_tier("production_edge"))
    assert "NON-contest" in str(ei.value) or "REFUSE" in str(ei.value)


def test_assert_contest_bit_exact_catches_contract_violation():
    bad = dmt.DecodeMemoryTier(
        name="rogue", contest=True, bit_exact_contract=False, contest_eval_device="cuda",
        blas_threads=0, worker_cap=None, note="fp32/CUDA rogue")
    with pytest.raises(dmt.DecodeTierError):
        dmt.assert_contest_bit_exact(bad)


def test_default_tier_inflate_env_is_1thread_blas_no_worker_override():
    # the DEFAULT tier (all cores, no cap) => only 1-thread BLAS keys == inflate.py setdefault(1)
    # => byte-identical to the current path, and no INFLATE_WORKERS override (fast locally).
    env = dmt.tier_inflate_env(dmt.resolve_tier("decode_cpu_16gb"))
    for k in dmt._BLAS_ENV_KEYS:
        assert env[k] == "1"
    assert "INFLATE_WORKERS" not in env


def test_worker_cap_env_when_tier_caps():
    capped = dmt.DecodeMemoryTier(
        name="capped", contest=True, bit_exact_contract=True, contest_eval_device="cpu",
        blas_threads=1, worker_cap=4, note="4-core cap")
    env = dmt.tier_inflate_env(capped, cpu_count=16)
    assert env["INFLATE_WORKERS"] == "4"      # min(16, 4)
    env2 = dmt.tier_inflate_env(capped, cpu_count=2)
    assert env2["INFLATE_WORKERS"] == "2"     # min(2, 4)


def test_non_contest_tier_emits_no_env():
    assert dmt.tier_inflate_env(dmt.resolve_tier("production_edge")) == {}


def test_resolve_eval_device_explicit_wins_else_tier():
    cpu = dmt.resolve_tier("decode_cpu_16gb")
    t4 = dmt.resolve_tier("decode_t4_16gb")
    assert dmt.resolve_eval_device(cpu, None) == "cpu"
    assert dmt.resolve_eval_device(t4, None) == "cuda"
    assert dmt.resolve_eval_device(t4, "cpu") == "cpu"   # explicit override wins
    assert dmt.resolve_eval_device(cpu, "cuda") == "cuda"
