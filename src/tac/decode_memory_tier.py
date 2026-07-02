# SPDX-License-Identifier: MIT
"""#224 Wave D — DECODE (inflate) memory-tier surface for the level-set byte-close/eval path.

The contest scores an ``archive.zip`` whose ``inflate.sh`` output must be produced within the
30-min ``upstream/evaluate.py`` budget on the target box. Two CONTEST targets exist (per the
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable): the 4-core/16GB CPU runner and the
T4-16GB CUDA runner. This surface declares + validates the DECODE contract per target and presets
the inflate's memory/thread hygiene so decode fits the budget.

THE BIT-EXACT CONTRACT (contest tiers, NON-NEGOTIABLE — a tier that would flip a uint8 boundary is
FORBIDDEN on contest tiers):
  * render-res: the archive's stored render_h/render_w (384x512) — NEVER downscaled at decode.
  * forward: float64 numpy activation (the shipped inflate op order).
  * R operator: CPU-torch (bicubic up render->camera, round, clamp -> uint8) — NOT CUDA.
  * BLAS: 1 thread per worker (multiprocess over INDEPENDENT pairs is bit-identical to serial —
    proven max_abs_uint8_diff=0; worker COUNT never changes the output bytes, only speed/RAM).
Both contest tiers share this EXACT inflate BY DESIGN (the CPU-vs-CUDA difference is DOWNSTREAM in
``evaluate.py``, NOT in the inflate) — the tier's real jobs are (a) asserting the contract and
(b) selecting the contest eval device. This is documented-shared-contract, the explicit opposite of
hidden enum-padding: ``decode_cpu_16gb`` and ``decode_t4_16gb`` are DISTINCT by their downstream
``contest_eval_device`` (cpu vs cuda) + target resource profile, not by a forked inflate.

``production_edge`` is the ONLY tier allowed to RELAX the contract (fp32 / CUDA-R / multithread =>
can flip uint8 boundaries). It is NON-contest; the byte-close/eval tool REFUSES it (fail-closed).
The relaxed edge NUMERIC path is DEFERRED (task #228; the configurable-TRAINING tiers are NOT built
this wave) — production_edge here is a DECLARED, validated, refused-for-contest tier only.

Authority: advisory; this surface never produces a score. Only ``upstream/evaluate.py`` on the exact
archive bytes does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_BLAS_ENV_KEYS = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class DecodeMemoryTier:
    """A decode (inflate) resource/contract profile.

    ``bit_exact_contract`` True == the contest numeric contract (fp64 fwd + CPU-torch R + 1-thread
    BLAS + stored render-res). ``contest`` True == eligible for a contest score (requires the
    bit-exact contract). ``worker_cap`` None => use all host cores (bit-identical output regardless);
    an int caps INFLATE_WORKERS. ``contest_eval_device`` selects the downstream evaluate.py device.
    """

    name: str
    contest: bool
    bit_exact_contract: bool
    contest_eval_device: Optional[str]   # "cpu" | "cuda" | None (edge/non-contest)
    blas_threads: int                    # 1 for the contest contract; 0 => leave inflate default
    worker_cap: Optional[int]            # None => all cores; int => min(cpu_count, cap)
    note: str


DECODE_MEMORY_TIERS: dict[str, DecodeMemoryTier] = {
    # DEFAULT: the current proven contest path — fp64 fwd + CPU-torch R + 1-thread BLAS, all host
    # cores (bit-identical output; on the 4-core contest CPU runner that IS 4). ~10-15min n600 g384.
    "decode_cpu_16gb": DecodeMemoryTier(
        name="decode_cpu_16gb", contest=True, bit_exact_contract=True, contest_eval_device="cpu",
        blas_threads=1, worker_cap=None,
        note="contest 4-core/16GB CPU eval runner; bit-exact (fp64/CPU-torch-R/1-thread BLAS); "
        "all host cores (bit-identical regardless of count); ~10-15min n600 g384."),
    # SAME bit-exact inflate BY DESIGN; distinct downstream eval device (cuda) + target profile
    # (T4 host typically has more vCPUs => faster CPU-side inflate). R stays CPU-torch (the T4 GPU
    # is the SCORER's during evaluate.py, NOT the inflate's).
    "decode_t4_16gb": DecodeMemoryTier(
        name="decode_t4_16gb", contest=True, bit_exact_contract=True, contest_eval_device="cuda",
        blas_threads=1, worker_cap=None,
        note="contest T4-16GB CUDA eval host; SAME bit-exact inflate as decode_cpu_16gb (CPU-numpy "
        "fp64 + CPU-torch R); downstream evaluate.py runs --device cuda; all host cores."),
    # NON-contest edge target: MAY relax the contract (fp32/CUDA-R/multithread) => can flip uint8
    # boundaries. The relaxed numeric path is DEFERRED (#228); refused by the contest byte-close tool.
    "production_edge": DecodeMemoryTier(
        name="production_edge", contest=False, bit_exact_contract=False, contest_eval_device=None,
        blas_threads=0, worker_cap=None,
        note="edge runtime ONLY — may relax fp32/CUDA-R/multithread (flips uint8 boundaries); "
        "NON-contest, NON-bit-exact; relaxed numeric path DEFERRED (#228); REFUSED for contest "
        "byte-close/eval."),
}

DEFAULT_TIER_NAME = "decode_cpu_16gb"


class DecodeTierError(ValueError):
    """A tier request violates the contest bit-exact contract or is contest-ineligible."""


def resolve_tier(name: Optional[str]) -> DecodeMemoryTier:
    key = (name or DEFAULT_TIER_NAME)
    if key not in DECODE_MEMORY_TIERS:
        raise DecodeTierError(
            f"unknown --memory-tier {key!r}; choices: {sorted(DECODE_MEMORY_TIERS)}")
    return DECODE_MEMORY_TIERS[key]


def assert_contest_bit_exact(tier: DecodeMemoryTier) -> DecodeMemoryTier:
    """Invariant guard: a CONTEST tier MUST carry the bit-exact numeric contract; a tier WITHOUT
    the bit-exact contract MUST NOT be contest-eligible. Raises :class:`DecodeTierError` on a
    contract violation (structural protection against a fp32/CUDA/multithread tier being scored)."""
    if tier.contest and not tier.bit_exact_contract:
        raise DecodeTierError(
            f"tier {tier.name!r} is contest=True but bit_exact_contract=False — a fp32/CUDA/"
            f"multithread decode can flip uint8 boundaries and is FORBIDDEN on contest tiers")
    if (not tier.contest) and tier.bit_exact_contract:
        raise DecodeTierError(
            f"tier {tier.name!r} is contest=False but bit_exact_contract=True — inconsistent "
            f"(a bit-exact tier should be contest-eligible)")
    return tier


def require_contest_tier(tier: DecodeMemoryTier) -> DecodeMemoryTier:
    """Fail-closed: the contest byte-close/eval path may ONLY use a contest (bit-exact) tier.
    production_edge is refused here (it belongs to the edge runtime, not the contest score path)."""
    assert_contest_bit_exact(tier)
    if not tier.contest:
        raise DecodeTierError(
            f"tier {tier.name!r} is NON-contest ({tier.note}); the contest byte-close/eval tool "
            f"REFUSES it. Use {sorted(t for t, v in DECODE_MEMORY_TIERS.items() if v.contest)}.")
    return tier


def tier_inflate_env(tier: DecodeMemoryTier, *, cpu_count: Optional[int] = None) -> dict[str, str]:
    """Env vars to set for the inflate subprocess so decode fits the tier's target.

    Contest tiers force 1-thread BLAS (the contract) and optionally cap INFLATE_WORKERS. The default
    tier (all cores, no cap) => only the BLAS keys, which equal the inflate.py's own ``setdefault(1)``
    => BYTE-IDENTICAL to the current path (and fast locally). A NON-contest tier returns ``{}`` (its
    relaxed numeric env is deferred, not emitted here)."""
    if not tier.bit_exact_contract:
        return {}
    env: dict[str, str] = {}
    if tier.blas_threads and tier.blas_threads >= 1:
        for k in _BLAS_ENV_KEYS:
            env[k] = str(int(tier.blas_threads))
    if tier.worker_cap is not None:
        cc = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
        env["INFLATE_WORKERS"] = str(max(1, min(cc, int(tier.worker_cap))))
    return env


def resolve_eval_device(tier: DecodeMemoryTier, explicit: Optional[str]) -> str:
    """The downstream evaluate.py device: an explicit --eval-device wins; else the tier's
    contest_eval_device; else 'cpu'. (MPS is never a choice — CPU/CUDA only.)"""
    if explicit:
        return explicit
    return tier.contest_eval_device or "cpu"
