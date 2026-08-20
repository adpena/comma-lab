# SPDX-License-Identifier: MIT
"""The contest wall-clock budget predicate: does a measured decode fit the 30-minute CI job?

THE BUG THIS EXTINCTS
---------------------
``experiments/contest_auth_eval.py`` defaults ``--inflate-timeout`` to 1800 s AND
``--evaluate-timeout`` to 1800 s.  Together they permit **3,600 s -- twice the entire CI job
wall** -- and no code path ever summed setup + inflate + evaluate against that wall.  So a
candidate could pass our Modal gate and still time out in the real CI.

That is not theoretical.  ``ddm_lc2`` MEASURED PR130 at **1,958 s -> rc=1 timeout** on a
*faster* 8-core box than the contest's 4-vCPU runner.  ``ddm_wc2`` then measured the shipping
T4 row (``br1``) at **inflate 1,246.928 s + evaluate 43.181 s**, which sits inside the CUDA
residual window only at its best end -- it fits *only if the runner's uv cache is warm*.

WHAT THE 30 MINUTES ACTUALLY BOUNDS
-----------------------------------
``upstream/.github/workflows/eval.yml:30`` sets ``timeout-minutes: 30`` as a sibling of
``runs-on:``/``steps:`` under ``jobs.test`` -- it bounds the **whole job**, not the decode.
The budget identity (``ddm_wc2`` §1.1) is::

    1800 s >= T_ci_setup + T_unzip + T_inflate(OURS) + T_assert + T_evaluate(UPSTREAM) + T_upload

    =>  T_inflate_ceiling = 1800 s - T_ci_setup - T_evaluate - (small fixed terms)

CLAUDE.md:302 and CLAUDE.md:929 say "the only constraint is the 30-min decode budget".  That
is CONFLATED and this module does not follow it; ``upstream/README.md:114`` and CLAUDE.md:294
are the correct reading.

WHY THE WINDOW IS NOT A BARE CONSTANT
-------------------------------------
The inherited claim "contest-CPU decode 831.5 s, 2.17x headroom" was wrong not because anyone
mis-multiplied, but because the denominator had been baked into a quotable scalar whose inputs
were no longer visible.  Nobody could see that it divided by the **full 1800 s** (pricing
``T_ci_setup`` and ``T_evaluate`` at zero) or that the numerator came off an **8-vCPU** box
when the contest runner has **4**.  A hardcoded ``T_RESIDUAL_CUDA = (822, 1302)`` is that same
defect one generation later.

So there is deliberately **no module-level window tuple**.  The window is the evaluated output
of a recorded derivation (:func:`residual_window`), and the returned
:class:`ResidualWindow` cannot be constructed without a grade and non-empty provenance.  Its
``to_dict()`` always carries both.  Any consumer quoting these seconds as MEASURED is making a
false-authority claim: only the **payload sizes** and the job wall were measured; every
per-step *second* is ua2's ESTIMATE and has never been timed on a real runner.

WHY THE VERDICT IS THREE-VALUED AND NEVER A BOOL
------------------------------------------------
The window has two ends *because* the answer depends on the uv cache state -- a ``uv sync
--group cu128`` miss puts 3.19 GB inside the wall and is the single largest mover.  Collapsing
that to ``fits: true/false`` would erase the dependency the predicate exists to surface (m52: a
bool flag is a UI over a continuum).  Hence :data:`PASS` / :data:`WARN` / :data:`REFUSE`.

WHAT THE GAUGE WOULD READ IF THE CURE WERE APPLIED AND NOTHING ELSE CHANGED
--------------------------------------------------------------------------
A NAIVE gauge -- "fraction of receipts carrying a budget verdict" -- is REJECTED: wiring this
module drives it to 100% by construction whether or not a single candidate fits.  The gauge
this module exposes is the **verdict distribution over real receipts**, which moves only when
a decode gets faster or the CI overhead changes.  Applied to the receipts wc2 read on
2026-08-20 it reports br1 = WARN and MC36-on-4-vCPU = REFUSE, which is the correct reading,
because neither has been repaired -- only made visible.

Axis: this module never produces a score.  It grades wall clock, and wall clock is a secondary
objective that must never be traded against the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AXES",
    "CONTEST_CPU",
    "CONTEST_CUDA",
    "DECODE_PATH_NATIVE_DISPATCHED",
    "DECODE_PATH_OTHER",
    "DECODE_PATH_PYTHON_FALLBACK",
    "DECODE_PATH_UNREPORTED",
    "GRADES",
    "GRADE_DERIVED",
    "GRADE_ESTIMATED",
    "GRADE_MEASURED",
    "GRADE_PROJECTION",
    "JOB_WALL_SECONDS",
    "JOB_WALL_SOURCE",
    "PASS",
    "REFUSE",
    "WARN",
    "BudgetInput",
    "BudgetVerdict",
    "CiStep",
    "ContestBudgetError",
    "ResidualWindow",
    "UnknownBudgetAxis",
    "axis_from_lane_tag",
    "budget_verdict_for_receipt",
    "classify_decode_path",
    "evaluate_budget",
    "normalize_axis",
    "residual_window",
]

# --- the one thing here that is MEASURED AT SOURCE -----------------------------------------
#: ``timeout-minutes: 30`` under ``jobs.test``, a sibling of ``runs-on:``/``steps:`` -- it
#: bounds the WHOLE job, not the decode.
JOB_WALL_SECONDS = 1800
JOB_WALL_SOURCE = "upstream/.github/workflows/eval.yml:30 (timeout-minutes: 30, job-scoped)"

# --- evidence ladder -----------------------------------------------------------------------
GRADE_MEASURED = "MEASURED"
GRADE_ESTIMATED = "ESTIMATED"
GRADE_DERIVED = "DERIVED"
GRADE_PROJECTION = "PROJECTION"
GRADES = frozenset({GRADE_MEASURED, GRADE_ESTIMATED, GRADE_DERIVED, GRADE_PROJECTION})

# --- axes --------------------------------------------------------------------------------
CONTEST_CPU = "contest-CPU"
CONTEST_CUDA = "contest-CUDA"
AXES = (CONTEST_CPU, CONTEST_CUDA)

#: Axis spellings seen in receipts and firers.  Two families of spelling are deliberately
#: ABSENT:
#:
#: * ADVISORY labels (``[macOS-CPU advisory]``, ``diagnostic_cpu``, ``mps``) -- mapping one onto
#:   ``contest-CPU`` would let an advisory decode inherit a contest-axis verdict.
#: * BARE DEVICE names (``cpu``, ``cuda``) -- ``args.device == "cpu"`` is the SAME string for a
#:   Linux x86_64 contest-CPU row and a macOS advisory row, so accepting it would launder the
#:   very distinction the axis label exists to carry.  Callers must pass the graded axis
#:   (``lane_tag`` / ``score_axis``), which the evidence contract already computed for them.
_AXIS_ALIASES: dict[str, str] = {
    "contest-cpu": CONTEST_CPU,
    "contest_cpu": CONTEST_CPU,
    "[contest-cpu]": CONTEST_CPU,
    "contest-cuda": CONTEST_CUDA,
    "contest_cuda": CONTEST_CUDA,
    "[contest-cuda]": CONTEST_CUDA,
}

# --- verdicts ------------------------------------------------------------------------------
PASS = "PASS"
WARN = "WARN"
REFUSE = "REFUSE"

# --- decode-path classes -------------------------------------------------------------------
DECODE_PATH_PYTHON_FALLBACK = "python_fallback"
DECODE_PATH_NATIVE_DISPATCHED = "native_dispatched"
DECODE_PATH_UNREPORTED = "unreported"
DECODE_PATH_OTHER = "other"

_PYTHON_FALLBACK_TOKENS = ("python", "scalar-python", "pure-python", "fallback")


class ContestBudgetError(RuntimeError):
    """Base class for every fail-closed refusal in this module."""


class UnknownBudgetAxis(ContestBudgetError):
    """The axis is not a contest axis, so the residual windows do not apply to it.

    Raised rather than defaulted.  Defaulting an unknown axis to ``contest-CPU`` would grade a
    macOS advisory decode against a contest window and call it compliance.
    """


def _require_grade(grade: str) -> str:
    if grade not in GRADES:
        raise ValueError(f"grade must be one of {sorted(GRADES)}, got {grade!r}")
    return grade


def _require_finite_seconds(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return v


@dataclass(frozen=True)
class BudgetInput:
    """One input to the residual derivation, carrying its own evidence grade.

    A value without a grade is how "PROJECTION" becomes "MEASURED" between two memos, so the
    grade is a constructor requirement rather than an optional annotation.
    """

    name: str
    value: float | int | str
    unit: str
    grade: str
    source: str

    def __post_init__(self) -> None:
        _require_grade(self.grade)
        if not self.source:
            raise ValueError(f"BudgetInput {self.name!r} requires a source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "grade": self.grade,
            "source": self.source,
        }


@dataclass(frozen=True)
class CiStep:
    """One CI step's typical/worst seconds, as tabulated by ``ddm_ua2`` §3.

    ``typical`` is the warm-cache corner and ``worst`` the cold-cache corner; the spread between
    them is dominated by the ``uv sync`` install payload.
    """

    step: str
    typical_seconds: float
    worst_seconds: float
    grade: str
    source: str

    def __post_init__(self) -> None:
        _require_grade(self.grade)
        _require_finite_seconds(f"{self.step}.typical_seconds", self.typical_seconds)
        _require_finite_seconds(f"{self.step}.worst_seconds", self.worst_seconds)
        if self.worst_seconds < self.typical_seconds:
            raise ValueError(
                f"CI step {self.step!r}: worst ({self.worst_seconds}) < typical "
                f"({self.typical_seconds}); the worst corner cannot be faster than the typical one"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "typical_seconds": self.typical_seconds,
            "worst_seconds": self.worst_seconds,
            "grade": self.grade,
            "source": self.source,
        }


@dataclass(frozen=True)
class ResidualWindow:
    """The per-axis ceiling on OUR decode, as the evaluated output of a recorded derivation.

    ``narrow_end_seconds`` is the ceiling under a COLD uv cache (the small, binding one);
    ``wide_end_seconds`` is the ceiling under a WARM cache.  The two ends exist because the
    answer genuinely depends on the cache state -- that dependency is the finding, not noise to
    average away.

    The object refuses to exist without a grade and a non-empty provenance tuple.  There is no
    ``__iter__``, so ``lo, hi = window`` fails: extracting a naked float pair has to be
    deliberate, and ``to_dict()`` always carries the grade with the numbers.
    """

    axis: str
    narrow_end_seconds: int
    wide_end_seconds: int
    grade: str
    provenance: tuple[str, ...]
    inputs: tuple[BudgetInput, ...] = ()
    ci_steps: tuple[CiStep, ...] = ()
    #: (worst, typical) minutes as ua2 published them, before the seconds rounding.
    published_residual_minutes: tuple[float, ...] = ()
    #: (typical, worst) seconds summed fresh from :attr:`ci_steps`, for the cross-check.
    step_table_sum_seconds: tuple[float, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_grade(self.grade)
        if not self.provenance:
            raise ValueError(
                f"ResidualWindow({self.axis!r}) requires non-empty provenance. A window without "
                "provenance is the quotable-scalar defect this module exists to prevent."
            )
        if self.narrow_end_seconds <= 0 or self.wide_end_seconds <= 0:
            raise ValueError(
                f"ResidualWindow({self.axis!r}) ends must be positive, got "
                f"({self.narrow_end_seconds}, {self.wide_end_seconds})"
            )
        if self.narrow_end_seconds > self.wide_end_seconds:
            raise ValueError(
                f"ResidualWindow({self.axis!r}): narrow end {self.narrow_end_seconds} exceeds "
                f"wide end {self.wide_end_seconds}; the cold-cache ceiling cannot be the larger one"
            )

    # --- cross-check between the published level set and the step table --------------------
    @property
    def step_table_narrow_end_seconds(self) -> float | None:
        """The cold-cache ceiling re-derived by summing the step table, for cross-check."""
        if not self.step_table_sum_seconds:
            return None
        return JOB_WALL_SECONDS - self.step_table_sum_seconds[1]

    @property
    def step_table_wide_end_seconds(self) -> float | None:
        """The warm-cache ceiling re-derived by summing the step table, for cross-check."""
        if not self.step_table_sum_seconds:
            return None
        return JOB_WALL_SECONDS - self.step_table_sum_seconds[0]

    @property
    def reconciliation_delta_seconds(self) -> tuple[float, float] | None:
        """(published - step-table) at each end.  Negative == the published end is stricter.

        Recorded rather than reconciled away: ua2 published its residual in ROUNDED MINUTES,
        and rounding a level set is exactly the kind of small silent step that later gets
        quoted as an exact measurement.
        """
        narrow = self.step_table_narrow_end_seconds
        wide = self.step_table_wide_end_seconds
        if narrow is None or wide is None:
            return None
        return (self.narrow_end_seconds - narrow, self.wide_end_seconds - wide)

    def __str__(self) -> str:  # keeps the grade attached even under naive f-string use
        return (
            f"ResidualWindow[{self.axis}] cold<= {self.narrow_end_seconds}s "
            f"warm<= {self.wide_end_seconds}s (grade={self.grade})"
        )

    def to_dict(self) -> dict[str, Any]:
        recon = self.reconciliation_delta_seconds
        return {
            "schema": "contest_budget_residual_window_v1",
            "axis": self.axis,
            "job_wall_seconds": JOB_WALL_SECONDS,
            "job_wall_source": JOB_WALL_SOURCE,
            "narrow_end_seconds": self.narrow_end_seconds,
            "narrow_end_assumes": "COLD uv cache (the binding corner)",
            "wide_end_seconds": self.wide_end_seconds,
            "wide_end_assumes": "WARM uv cache",
            "grade": self.grade,
            "provenance": list(self.provenance),
            "published_residual_minutes": list(self.published_residual_minutes),
            "ci_step_sum_seconds": {
                "typical": self.step_table_sum_seconds[0] if self.step_table_sum_seconds else None,
                "worst": self.step_table_sum_seconds[1] if self.step_table_sum_seconds else None,
            },
            "step_table_narrow_end_seconds": self.step_table_narrow_end_seconds,
            "step_table_wide_end_seconds": self.step_table_wide_end_seconds,
            "reconciliation_delta_seconds": list(recon) if recon is not None else None,
            "inputs": [i.to_dict() for i in self.inputs],
            "ci_steps": [s.to_dict() for s in self.ci_steps],
            "notes": list(self.notes),
            "false_authority_warning": (
                "PROJECTION. Only the job wall and the install/checkout PAYLOAD SIZES are "
                "MEASURED; every per-step SECOND is ua2's estimate and has never been timed on "
                "a real contest runner. Quoting this window as MEASURED is a false-authority "
                "claim."
            ),
        }


# --- the recorded derivation ---------------------------------------------------------------
# Payload sizes below are MEASURED (ua2 read them out of uv.lock / the workflow); the seconds
# are ESTIMATED. Keeping the two grades side by side is the whole point of the table.

_SHARED_INPUTS: tuple[BudgetInput, ...] = (
    BudgetInput("job_wall_seconds", JOB_WALL_SECONDS, "s", GRADE_MEASURED, JOB_WALL_SOURCE),
    BudgetInput(
        "checkout_payload_bytes", 31_600_000, "B", GRADE_MEASURED,
        ".omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md (31.6 MB)",
    ),
    BudgetInput(
        "git_lfs_pull_payload_bytes", 132_856_531, "B", GRADE_MEASURED,
        ".omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md",
    ),
)

_CPU_INPUTS: tuple[BudgetInput, ...] = (
    *_SHARED_INPUTS,
    BudgetInput(
        "uv_sync_group_cpu_payload_bytes", 78_000_000, "B", GRADE_MEASURED,
        "ddm_ua2 §3 (~78 MB sized in uv.lock)",
    ),
    BudgetInput(
        "evaluate_py_600_pairs_8_vcpu_seconds", 176.3, "s", GRADE_MEASURED,
        "clickpolish_pr110_phase2_modal_runbook_20260710.md; corroborated 174.9 s "
        "(pr106_latent_sidecar_dual_axis)",
    ),
    BudgetInput(
        "vcpu_8_to_4_projection_factor", "1.7-2.3x", "ratio", GRADE_DERIVED,
        "ddm_ua2 §3 -- the scorer forward is thread-parallel but not linear; NEVER measured",
    ),
)

_CUDA_INPUTS: tuple[BudgetInput, ...] = (
    *_SHARED_INPUTS,
    BudgetInput(
        "uv_sync_group_cu128_payload_bytes", 3_190_398_780, "B", GRADE_MEASURED,
        "ddm_ua2 §3, read out of uv.lock size fields (3.19 GB; ~40x the cpu group)",
    ),
    BudgetInput(
        "evaluate_py_600_pairs_t4_seconds", "120-180", "s", GRADE_ESTIMATED,
        "ddm_ua2 §3 -- explicitly labelled UNMEASURED BY US",
    ),
)

_CPU_STEPS: tuple[CiStep, ...] = (
    CiStep("checkout", 10, 15, GRADE_ESTIMATED, "eval.yml:36 (from measured 31.6 MB payload)"),
    CiStep("git_fetch_depth_1", 4, 5, GRADE_ESTIMATED, "eval.yml:44"),
    CiStep("curl_archive_zip", 2, 3, GRADE_ESTIMATED, "eval.yml:60 (OURS, bounded by the rate term)"),
    CiStep("apt_update_git_lfs", 32, 45, GRADE_ESTIMATED, "eval.yml:64-66"),
    CiStep("git_lfs_pull", 12, 20, GRADE_ESTIMATED, "eval.yml:69-71 (from measured 132,856,531 B)"),
    CiStep("setup_uv_cache_restore", 12, 20, GRADE_ESTIMATED, "eval.yml:73-77"),
    CiStep("uv_sync_group_cpu", 20, 120, GRADE_ESTIMATED, "eval.yml:79-81 (20 s hit / 120 s miss)"),
    CiStep("apt_update_ffmpeg", 65, 90, GRADE_ESTIMATED, "eval.yml:83-85"),
    CiStep("unzip", 1, 1, GRADE_ESTIMATED, "evaluate.sh:44"),
    CiStep("raw_existence_assert", 1, 1, GRADE_ESTIMATED, "evaluate.sh:50-62"),
    CiStep(
        "evaluate_py_600_pairs", 300, 400, GRADE_DERIVED,
        "evaluate.sh:69 -- DERIVED from 176.3 s MEASURED @ 8 vCPU via ua2's 1.7-2.3x 8->4 band",
    ),
    CiStep("upload_artifact", 10, 15, GRADE_ESTIMATED, "eval.yml:91-98"),
)

_CUDA_STEPS: tuple[CiStep, ...] = (
    CiStep(
        "checkout_fetch_curl_apt_lfs_nvidia_smi", 60, 90, GRADE_ESTIMATED,
        "eval.yml:36-54 (CPU steps 1-5 plus nvidia-smi at :54)",
    ),
    CiStep("setup_uv_cache_restore_multi_gb", 120, 180, GRADE_ESTIMATED, "eval.yml:73-77"),
    CiStep(
        "uv_sync_group_cu128", 120, 420, GRADE_ESTIMATED,
        "eval.yml:79-81 -- 3.19 GB MEASURED payload; 120 s hit / 420 s miss. THE single largest "
        "mover between the two window ends",
    ),
    CiStep("apt_update_ffmpeg", 65, 90, GRADE_ESTIMATED, "eval.yml:83-85"),
    CiStep(
        "unzip_existence_upload", 12, 17, GRADE_ESTIMATED,
        "evaluate.sh:44/50-62 + eval.yml:91-98 (ua2 groups these three)",
    ),
    CiStep(
        "evaluate_py_600_pairs_t4", 120, 180, GRADE_ESTIMATED,
        "evaluate.sh:69 -- DALI decode + GPU forward; ua2 labels this UNMEASURED BY US",
    ),
)

#: ua2's published level set (``ddm_ua2:189``), in MINUTES, as (worst, typical).  The window
#: seconds below are ``round(minutes * 60)`` -- ua2's own rounding, preserved rather than
#: silently re-derived, so the emitted numbers match the memo that adjudicated them.
_PUBLISHED_RESIDUAL_MINUTES: dict[str, tuple[float, float]] = {
    CONTEST_CPU: (17.4, 22.2),
    CONTEST_CUDA: (13.7, 21.7),
}

_AXIS_STEPS: dict[str, tuple[CiStep, ...]] = {CONTEST_CPU: _CPU_STEPS, CONTEST_CUDA: _CUDA_STEPS}
_AXIS_INPUTS: dict[str, tuple[BudgetInput, ...]] = {CONTEST_CPU: _CPU_INPUTS, CONTEST_CUDA: _CUDA_INPUTS}

_PROVENANCE: tuple[str, ...] = (
    ".omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md:189 (the level set)",
    ".omx/research/ddm_wc2_wall_clock_pass_20260820.md §1.1 (budget identity) / §5 (per-axis verdict) / §7.1 (this spec)",
    "/Volumes/APDataStore/pact/ddm_wc2/receipts/br1_t4_shipping_stage_split.json "
    "(sha256 2eac38af90b0a67304ee400f5842273e2852f8062f934ba83d1f9a4bef5e643a)",
    "upstream/.github/workflows/eval.yml:30 + upstream/README.md:114",
)

_WINDOW_CACHE: dict[str, ResidualWindow] = {}


def normalize_axis(axis: str) -> str:
    """Map a receipt/firer axis spelling onto a contest axis.  REFUSES anything else.

    Advisory spellings (``[macOS-CPU advisory]``, ``mps``, ``diagnostic_*``) refuse on purpose:
    the residual windows describe the contest runners, and letting an advisory decode borrow a
    contest window would manufacture compliance out of a label.
    """
    key = str(axis or "").strip().lower()
    resolved = _AXIS_ALIASES.get(key)
    if resolved is None:
        raise UnknownBudgetAxis(
            f"axis {axis!r} is not a contest axis. Known: {sorted(set(_AXIS_ALIASES.values()))}. "
            "Advisory axes are refused deliberately -- the residual windows are properties of the "
            "contest runners, not of our local box."
        )
    return resolved


def axis_from_lane_tag(lane_tag: str | None) -> str | None:
    """Contest axis for a harness ``lane_tag``, or ``None`` when the axis is not a contest one.

    Non-raising counterpart to :func:`normalize_axis`, for the harness path where an advisory
    run must still emit a receipt (with a NOT_APPLICABLE budget row) rather than crash.
    """
    if not lane_tag:
        return None
    try:
        return normalize_axis(lane_tag)
    except UnknownBudgetAxis:
        return None


def residual_window(axis: str) -> ResidualWindow:
    """Evaluate (and cache) the residual window for a contest axis.

    This is a FUNCTION and not a module constant on purpose: a bare
    ``T_RESIDUAL_CUDA = (822, 1302)`` is quotable without its inputs, which is precisely how the
    "2.17x headroom" defect happened.  The returned object carries its grade, its provenance,
    every graded input, and the full CI step table.
    """
    resolved = normalize_axis(axis)
    cached = _WINDOW_CACHE.get(resolved)
    if cached is not None:
        return cached

    steps = _AXIS_STEPS[resolved]
    typical_sum = float(sum(s.typical_seconds for s in steps))
    worst_sum = float(sum(s.worst_seconds for s in steps))
    worst_minutes, typical_minutes = _PUBLISHED_RESIDUAL_MINUTES[resolved]

    window = ResidualWindow(
        axis=resolved,
        narrow_end_seconds=round(worst_minutes * 60),
        wide_end_seconds=round(typical_minutes * 60),
        grade=GRADE_PROJECTION,
        provenance=_PROVENANCE,
        inputs=_AXIS_INPUTS[resolved],
        ci_steps=steps,
        published_residual_minutes=(worst_minutes, typical_minutes),
        step_table_sum_seconds=(typical_sum, worst_sum),
        notes=(
            "The two ends are the WARM and COLD uv-cache corners; the spread is dominated by "
            "the uv sync install payload (3.19 GB on cu128 vs ~78 MB on cpu, a ~40x asymmetry "
            "mitigated only by enable-cache: true at eval.yml:77).",
            "The window is a ceiling on OUR decode: the residual has ALREADY netted out an "
            "ESTIMATED T_evaluate, so charging a measured evaluate against it again is "
            "deliberately conservative -- see BudgetVerdict.charged_seconds.",
            "ua2 published this level set in rounded minutes; reconciliation_delta_seconds "
            "reports the gap against a fresh sum of the step table rather than hiding it.",
        ),
    )
    _WINDOW_CACHE[resolved] = window
    return window


def classify_decode_path(decode_path: str | None) -> str:
    """Bucket a decode-path label.  Never guesses: an unrecognised label is ``other``.

    The fail-closed dispatch ladder (AVX-512 -> AVX2 -> scalar-C -> NEON -> Python) is *silent
    by design*, so which path actually ran is not observable from the seconds alone.  It has to
    travel with the receipt or it is lost.
    """
    if decode_path is None:
        return DECODE_PATH_UNREPORTED
    label = str(decode_path).strip().lower()
    if not label or label in {"unknown", "unreported", "none"}:
        return DECODE_PATH_UNREPORTED
    # Native is tested FIRST on purpose. A mixed label ("native-hpac with python glue") read as
    # a fallback would DROP the unverified-fast-path warning, which is the unsafe direction;
    # read as native it keeps the warning. Ambiguity must cost caution, not lose it.
    if label.startswith("native") or "hpac" in label or "avx" in label or "neon" in label:
        return DECODE_PATH_NATIVE_DISPATCHED
    if any(tok in label for tok in _PYTHON_FALLBACK_TOKENS):
        return DECODE_PATH_PYTHON_FALLBACK
    return DECODE_PATH_OTHER


@dataclass(frozen=True)
class BudgetVerdict:
    """A three-valued wall-clock verdict on one measured decode.  Never a score."""

    verdict: str
    axis: str
    window: ResidualWindow
    inflate_seconds: float
    evaluate_seconds: float | None
    charged_seconds: float
    margin_vs_narrow_end_seconds: float
    margin_vs_wide_end_seconds: float
    decode_path: str | None
    decode_path_class: str
    margin_depends_on_unverified_fast_path: bool
    charge_is_lower_bound: bool
    rationale: str
    notes: tuple[str, ...] = field(default=())
    grade: str = GRADE_PROJECTION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "contest_budget_verdict_v1",
            "verdict": self.verdict,
            "axis": self.axis,
            "axis_label": f"[{self.axis}]",
            "grade": self.grade,
            "inflate_seconds": self.inflate_seconds,
            "evaluate_seconds": self.evaluate_seconds,
            "charged_seconds": self.charged_seconds,
            "charge_is_lower_bound": self.charge_is_lower_bound,
            "margin_vs_narrow_end_seconds": self.margin_vs_narrow_end_seconds,
            "margin_vs_wide_end_seconds": self.margin_vs_wide_end_seconds,
            "decode_path": self.decode_path,
            "decode_path_class": self.decode_path_class,
            "margin_depends_on_unverified_fast_path": self.margin_depends_on_unverified_fast_path,
            "rationale": self.rationale,
            "notes": list(self.notes),
            "residual_window": self.window.to_dict(),
            "is_score_claim": False,
        }


def evaluate_budget(
    axis: str,
    inflate_elapsed_seconds: float,
    evaluate_elapsed_seconds: float | None = None,
    *,
    decode_path: str | None = None,
) -> BudgetVerdict:
    """Grade a measured decode against the contest job wall.  Adds a verdict, never a measurement.

    Args:
        axis: ``contest-CPU`` or ``contest-CUDA`` (aliases in :data:`_AXIS_ALIASES` accepted).
            Advisory axes REFUSE -- see :func:`normalize_axis`.
        inflate_elapsed_seconds: measured ``inflate.sh`` wall clock, already in the receipt.
        evaluate_elapsed_seconds: measured ``upstream/evaluate.py`` wall clock, already in the
            receipt.  ``None`` is allowed and flagged: the charge then UNDERSTATES, so the
            verdict is optimistic and says so.
        decode_path: which decode path actually ran, when the runtime reports it.  It never
            overrides the seconds -- a label must not outrank a measurement -- but it decides
            which way the remaining risk points, and that lands in the rationale.

    Verdict ladder (three-valued on purpose; a bool would erase the cache-state dependency):
        ``PASS``   charge fits inside the COLD-cache (narrow) ceiling -> fits either way.
        ``WARN``   fits the WARM-cache ceiling only -> fits only if the runner's uv cache is warm.
        ``REFUSE`` outside both -> projected timeout.
    """
    window = residual_window(axis)
    t_inflate = _require_finite_seconds("inflate_elapsed_seconds", inflate_elapsed_seconds)
    t_evaluate = (
        None if evaluate_elapsed_seconds is None
        else _require_finite_seconds("evaluate_elapsed_seconds", evaluate_elapsed_seconds)
    )
    charged = t_inflate + (t_evaluate or 0.0)

    if charged <= window.narrow_end_seconds:
        verdict = PASS
    elif charged <= window.wide_end_seconds:
        verdict = WARN
    else:
        verdict = REFUSE

    path_class = classify_decode_path(decode_path)
    notes: list[str] = []

    if verdict == PASS:
        rationale = (
            f"{charged:.3f} s charged <= {window.narrow_end_seconds} s cold-cache ceiling on "
            f"{window.axis}: fits even when the runner's uv cache misses."
        )
    elif verdict == WARN:
        rationale = (
            f"{charged:.3f} s charged is above the {window.narrow_end_seconds} s cold-cache "
            f"ceiling and at or below the {window.wide_end_seconds} s warm-cache ceiling on "
            f"{window.axis}: it fits ONLY if the runner's uv cache is warm. A cu128 cache miss "
            "puts 3.19 GB inside the same 30-minute wall."
        )
    else:
        rationale = (
            f"{charged:.3f} s charged exceeds the {window.wide_end_seconds} s warm-cache ceiling "
            f"on {window.axis}: PROJECTED TIMEOUT in every corner. Measured precedent: lc2/PR130 "
            "hit 1,958 s and returned rc=1 on a faster 8-core box than the contest's 4 vCPU."
        )

    if t_evaluate is None:
        notes.append(
            "evaluate_elapsed_seconds was not reported, so the charge is a LOWER BOUND and this "
            "verdict is optimistic by the missing evaluate term."
        )

    # The residual already netted out an ESTIMATED evaluate, so charging the MEASURED evaluate
    # against it is algebraically `S_other + t_inflate + t_evaluate <= 1800 - E_est`: the true
    # budget condition plus a safety margin of exactly E_est. That margin is a feature -- and it
    # is also SELF-CORRECTING, because a measured evaluate that overruns ua2's estimate tightens
    # the verdict by exactly the overrun instead of being invisible.
    evaluate_estimate = next(
        (s for s in window.ci_steps if s.step.startswith("evaluate_py_600_pairs")), None
    )
    margin_note = (
        f" The margin it buys is ua2's estimated evaluate term, {evaluate_estimate.typical_seconds:.0f}"
        f"-{evaluate_estimate.worst_seconds:.0f} s on this axis."
        if evaluate_estimate is not None else ""
    )
    notes.append(
        "The residual already netted out an ESTIMATED evaluate term, so charging the measured "
        "evaluate against it double-counts by that estimate. That is deliberate: it errs toward "
        f"WARN/REFUSE and never toward a false PASS.{margin_note} inflate alone is "
        f"{t_inflate:.3f} s."
    )

    depends_on_fast_path = False
    if path_class == DECODE_PATH_PYTHON_FALLBACK:
        notes.append(
            "decode_path is the PYTHON FALLBACK -- the slowest rung of the dispatch ladder. The "
            "measured seconds are therefore an upper bound on this candidate's decode; a "
            "reachable native path would only improve the margin."
        )
    elif path_class == DECODE_PATH_NATIVE_DISPATCHED:
        depends_on_fast_path = verdict in (PASS, WARN)
        notes.append(
            "decode_path is a DISPATCHED NATIVE path. Its availability on the contest runner is "
            "UNVERIFIED (cpuid probe, compile-at-decode, and the sm_75 box are all unmeasured by "
            "us). The fallback is fail-closed and therefore SILENT: if it fires on the contest "
            "box the decode is slower than measured and this margin does not hold."
        )
    elif path_class == DECODE_PATH_UNREPORTED:
        notes.append(
            "decode_path was not reported by this runtime generation, so which rung of the "
            "dispatch ladder produced these seconds is unknown. A python-fallback decode is "
            "exactly the WARN/REFUSE case this predicate exists to surface, and it is invisible "
            "here."
        )
    else:
        notes.append(f"decode_path {decode_path!r} did not match a known dispatch rung.")

    return BudgetVerdict(
        verdict=verdict,
        axis=window.axis,
        window=window,
        inflate_seconds=t_inflate,
        evaluate_seconds=t_evaluate,
        charged_seconds=charged,
        margin_vs_narrow_end_seconds=window.narrow_end_seconds - charged,
        margin_vs_wide_end_seconds=window.wide_end_seconds - charged,
        decode_path=decode_path,
        decode_path_class=path_class,
        margin_depends_on_unverified_fast_path=depends_on_fast_path,
        charge_is_lower_bound=t_evaluate is None,
        rationale=rationale,
        notes=tuple(notes),
    )


def budget_verdict_for_receipt(
    result: dict[str, Any],
    *,
    decode_path: str | None = None,
) -> dict[str, Any]:
    """Adapter: grade a ``contest_auth_eval`` receipt dict, and never raise on it.

    A budget verdict is a GUARD on the gating instrument, not part of the score. So every
    failure mode here degrades to a structured ``NOT_APPLICABLE``/``ERROR`` row rather than
    taking down a run whose score is already computed and valid.
    """
    lane_tag = result.get("lane_tag")
    axis = axis_from_lane_tag(lane_tag)
    inflate_s = result.get("inflate_elapsed_seconds")
    evaluate_s = result.get("evaluate_elapsed_seconds")

    if axis is None:
        return {
            "schema": "contest_budget_verdict_v1",
            "verdict": "NOT_APPLICABLE",
            "axis": None,
            "axis_label": lane_tag,
            "reason": (
                "the residual windows describe the contest runners; this run is not on a contest "
                "axis, so its wall clock is advisory and ungraded here"
            ),
            "inflate_seconds": inflate_s,
            "evaluate_seconds": evaluate_s,
            "decode_path": decode_path,
            "decode_path_class": classify_decode_path(decode_path),
            "is_score_claim": False,
        }
    if inflate_s is None:
        return {
            "schema": "contest_budget_verdict_v1",
            "verdict": "NOT_APPLICABLE",
            "axis": axis,
            "axis_label": f"[{axis}]",
            "reason": "inflate_elapsed_seconds absent from the receipt; nothing to grade",
            "inflate_seconds": None,
            "evaluate_seconds": evaluate_s,
            "decode_path": decode_path,
            "decode_path_class": classify_decode_path(decode_path),
            "is_score_claim": False,
        }
    try:
        return evaluate_budget(
            axis, float(inflate_s), None if evaluate_s is None else float(evaluate_s),
            decode_path=decode_path,
        ).to_dict()
    except (ContestBudgetError, TypeError, ValueError) as exc:
        return {
            "schema": "contest_budget_verdict_v1",
            "verdict": "ERROR",
            "axis": axis,
            "axis_label": f"[{axis}]",
            "reason": f"{type(exc).__name__}: {exc}",
            "inflate_seconds": inflate_s,
            "evaluate_seconds": evaluate_s,
            "decode_path": decode_path,
            "decode_path_class": classify_decode_path(decode_path),
            "is_score_claim": False,
        }
