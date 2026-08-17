"""ddm_jr1 -- repair the judge that adjudicated the band objective (R6).

$0 re-analysis of ALREADY-RETAINED payloads. No training run, no new gradient,
no Modal spend. Full n600 population; no scope reduction of any kind.

axis: [macOS-CPU advisory] read-back of retained payloads -- NEVER a score.

Three legs, all from disk:

LEG A -- the out-of-sample residual judge.
    ``rg1b`` fit ``peak_flips = A * ||dw||_100 ** b`` on FOUR stock arms
    (C0, A1, A2, A3; n=4, dof=2) and scored the band arm against it at
    ``-0.871 sigma``.  The band arm is genuinely held out (the receipt key is
    literally ``law_from_the_four_stock_arms``), so the charter's "fifth point
    the law is fit on" premise is false.  The real defect is the DENOMINATOR:
    the residual was divided by ``sigma_log`` -- the in-sample residual RMS of a
    4-point 2-parameter fit -- rather than by the prediction standard error
    ``SE_pred(d) = sigma*sqrt(1 + 1/n + (ln d - xbar)^2 / Sxx)`` that ``rg1b``'s
    OWN bar section uses, and it was reported in "sigma" as if referred to a
    normal when ``dof=2`` demands ``t_2`` (95% two-sided = 4.3027).

    This leg also supplies the null control the judge never had: **W1**, the
    fifth ``ddm_lr1`` arm (float warmup, stock objective), which was excluded
    from the fit and is therefore a held-out STOCK point.

LEG B -- the realized step, not merely the gradient.
    B1  cos(dw_band(0->t), dw_A2(0->t)) at matched steps: the realized
        trajectory rotation, integrated.  Both arms share init, seed, data
        order, lr and curriculum; the sole difference is the objective.
    B2  the realized AdamW next-step direction ``u = mhat/(sqrt(vhat)+eps)``
        reconstructed from the retained optimizer moments.
    B3  the init identity: after one step from zero moments the AdamW update
        collapses to ``sign(g) * |g|/(|g|+eps)``, so ``rg1b``'s ``cos(sign g)``
        column IS the realized first-step cosine.  The precondition
        ``|g| >> eps`` is checked against retained ``exp_avg_sq``.

LEG C support -- the measured displacement-vs-step curve per arm, which is what
    a matched-``||dw||`` ticket must derive its step budget from.

Instrument control runs FIRST and is fail-closed: the tool must reproduce every
retained ``dw_100``/``dw_600``/``peak_dpx``/``end_dpx`` before any new number is
read.

PAYLOAD: every displacement vector and every AdamW direction vector is persisted
to ``--out-dir`` alongside the JSON receipt, with sha256 and byte count, per the
ALWAYS-KEEP-THE-PAYLOAD rule.  Nothing measured here is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOTAL_PX = 117_964_800
INIT_SEG = 0.00028616163465711804

DEFAULT_INIT = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts"
    "/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)
DEFAULT_LR1 = Path("/Volumes/APDataStore/pact/ddm_lr1")
DEFAULT_BAND = Path("/Volumes/APDataStore/pact/ddm_rg1/band_a1")

_STEP_RE = re.compile(r"step(\d+)")

# Retained receipts the instrument control checks itself against.
CONTROL_REFIT = Path(".omx/research/ddm_rg1b_lr1_refit_and_bar_20260816.json")
CONTROL_BAND = Path("/Volumes/APDataStore/pact/ddm_rg1/grad_cosine/RG1B_BAND_ARM_ON_THE_LAW.json")

# Reproduction tolerance.  The retained values are float64 round-trips of the
# same computation, so this is a byte-level agreement check, not a fudge.
CONTROL_RTOL = 1e-9


@dataclass
class Arm:
    """One retained training arm."""

    name: str
    directory: Path
    lr: float
    float_warmup_steps: int
    objective: str  # "stock" | "band"
    in_law_fit: bool  # was this arm one of the four the law was fit on?
    result_name: str = "result.json"

    # filled by load()
    steps: list[int] = field(default_factory=list)
    dw: dict[int, float] = field(default_factory=dict)  # retained convention
    dw_float64: dict[int, float] = field(default_factory=dict)  # full float64
    history: list[tuple[int, float]] = field(default_factory=list)


# The five ddm_lr1 arms plus the band arm.  W1 is present on disk and was
# EXCLUDED from rg1b's fit -- it is the held-out stock null control.
ARMS: tuple[Arm, ...] = (
    Arm("C0", DEFAULT_LR1 / "C0", 2e-7, 0, "stock", True),
    Arm("A1", DEFAULT_LR1 / "A1", 2e-6, 0, "stock", True),
    Arm("A2", DEFAULT_LR1 / "A2", 2e-5, 0, "stock", True, result_name="result.recovered.json"),
    Arm("A3", DEFAULT_LR1 / "A3", 2e-4, 0, "stock", True),
    Arm("W1", DEFAULT_LR1 / "W1", 2e-5, 100, "stock", False),
    Arm("band_a1", DEFAULT_BAND, 2e-5, 0, "band", False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_weights(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return the LIVE training weights.

    Pinned empirically: the retained ``dw`` values reproduce against
    ``training_state.model_state_dict`` (live), NOT the top-level
    ``state_dict`` (which carries the deployed/EMA weights and gives
    0.0013798 where the receipt says 0.0014435 for C0@100).
    """
    training_state = payload.get("training_state")
    if isinstance(training_state, dict) and isinstance(
        training_state.get("model_state_dict"), dict
    ):
        return training_state["model_state_dict"]
    raise KeyError("checkpoint has no training_state.model_state_dict (live weights)")


def _float_keys(reference: dict[str, torch.Tensor]) -> list[str]:
    return [
        key
        for key, value in reference.items()
        if torch.is_tensor(value) and value.is_floating_point()
    ]


def _flat(state: dict[str, torch.Tensor], keys: list[str]) -> np.ndarray:
    return np.concatenate([state[key].detach().to(torch.float64).reshape(-1).numpy() for key in keys])


def _dw_retained_convention(
    live: dict[str, torch.Tensor], init_state: dict[str, torch.Tensor], keys: list[str]
) -> float:
    """||dw|| in the convention the retained receipts used.

    Pinned empirically: the retained values accumulate the sum of squares
    PER TENSOR in float32 (torch's default dtype for these weights) and only
    then widen to float64 to add across tensors.  Computing the whole thing in
    float64 disagrees in the 9th significant figure -- harmless for every
    conclusion here, but it would break the byte-level reproduction that makes
    the instrument control meaningful, so the convention is reproduced exactly
    and the float64 value is reported alongside rather than substituted.
    """
    total = 0.0
    for key in keys:
        delta = live[key].detach().float() - init_state[key].detach().float()
        total += float((delta**2).sum().item())
    return math.sqrt(total)


def _checkpoints(directory: Path) -> dict[int, Path]:
    """Map step -> checkpoint path, from whatever prefix the arm used.

    C0/A1/A3/W1/band use ``ckpt.*``; A2 uses ``checkpoints.*``.  Stage names
    differ per arm (W1's float warmup shifts every boundary), so the step is
    parsed from the filename rather than assumed from the stage.
    """
    found: dict[int, Path] = {}
    for path in sorted(directory.glob("*.full_state.pt")):
        # The SSD tiers are ExFAT, so macOS writes an AppleDouble "._<name>"
        # sidecar next to every real file.  Those match the glob and are not
        # checkpoints; reading one as a checkpoint would silently corrupt the
        # measurement, so they are excluded by name rather than by try/except.
        if path.name.startswith("._"):
            continue
        match = _STEP_RE.search(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in found:
            raise RuntimeError(f"duplicate checkpoint for step {step} in {directory}")
        found[step] = path
    if not found:
        raise FileNotFoundError(f"no *.full_state.pt checkpoints under {directory}")
    return found


def _history(arm: Arm) -> list[tuple[int, float]]:
    payload = json.loads((arm.directory / arm.result_name).read_text())
    # A2's receipt is a recovery wrapper: the run record sits under "result".
    record = payload.get("result", payload)
    rows = record["history"]
    return [(int(r["step"]), float(r["quantized_exact_seg"])) for r in rows]


def load_arm(arm: Arm, init_state: dict[str, torch.Tensor], keys: list[str]) -> np.ndarray:
    """Populate ``arm`` and return the stacked dw vectors (n_steps, n_params)."""
    init_flat = _flat(init_state, keys)
    checkpoints = _checkpoints(arm.directory)
    arm.steps = sorted(checkpoints)
    vectors = []
    for step in arm.steps:
        payload = torch.load(checkpoints[step], map_location="cpu", weights_only=False)
        live = _live_weights(payload)
        delta = _flat(live, keys) - init_flat
        arm.dw[step] = _dw_retained_convention(live, init_state, keys)
        arm.dw_float64[step] = float(np.linalg.norm(delta))
        vectors.append(delta)
    arm.history = _history(arm)
    return np.stack(vectors)


def peak_end_dpx(history: list[tuple[int, float]]) -> tuple[float, float]:
    """Peak and end flip excursion above the init, in pixels.

    Pinned empirically against C0: peak 5879.000000000001, end 2164.0.
    """
    deltas = [(seg - INIT_SEG) * TOTAL_PX for _, seg in history]
    return max(deltas), deltas[-1]


def peak_step(history: list[tuple[int, float]]) -> int:
    """Which step the peak excursion occurs at.

    Matters because the law regresses the PEAK against the displacement at
    step 100: if every arm peaks at 100 the pairing is meaningful, and if one
    does not, that arm is not on the same curve for a mechanical reason.
    """
    deltas = [((seg - INIT_SEG) * TOTAL_PX, step) for step, seg in history]
    return max(deltas)[1]


def minimum_detectable_effect(fit: PowerLawFit, dw: float, level: float = 0.95) -> dict[str, float]:
    """Smallest log-effect this judge could resolve, and its asymptotic floor.

    The floor matters more than the value at n=4.  ``SE_pred`` decomposes as
    ``sigma * sqrt(1 + 1/n + leverage)``: adding arms shrinks ``1/n`` and the
    leverage term, but never the leading ``1`` -- that is the NEW observation's
    own scatter.  So the judge's resolution is bounded below by
    ``1.96 * sigma_log`` no matter how many arms are run.
    """
    return {
        "MDE_log_at_this_n": t_critical(level, fit.dof) * fit.se_pred(dw),
        "MDE_as_flip_ratio_at_this_n": math.exp(t_critical(level, fit.dof) * fit.se_pred(dw)),
        "asymptotic_floor_log_n_to_infinity": 1.959964 * fit.sigma_log,
        "asymptotic_floor_as_flip_ratio": math.exp(1.959964 * fit.sigma_log),
        "note": "floor = 1.96*sigma_log; unreachable by adding arms because the "
        "new observation's own scatter is irreducible",
    }


# --------------------------------------------------------------------------
# Statistics.  Closed forms only -- no scipy dependency, and for dof 1 and 2
# the Student-t CDF is exact and elementary, which keeps the judge auditable.
# --------------------------------------------------------------------------


def t_cdf(t: float, dof: int) -> float:
    """Exact Student-t CDF for dof in {1, 2}; those are the only dofs here."""
    if dof == 1:
        return 0.5 + math.atan(t) / math.pi
    if dof == 2:
        return 0.5 + t / (2.0 * math.sqrt(2.0 + t * t))
    raise ValueError(f"closed form only implemented for dof 1 or 2, got {dof}")


def t_two_sided_p(t: float, dof: int) -> float:
    return 2.0 * (1.0 - t_cdf(abs(t), dof))


def t_critical(level: float, dof: int) -> float:
    """Two-sided critical value by inverting the closed-form CDF."""
    target = 0.5 + level / 2.0
    if dof == 1:
        return math.tan(math.pi * (target - 0.5))
    if dof == 2:
        c = 2.0 * (target - 0.5)
        return math.sqrt(2.0 * c * c / (1.0 - c * c))
    raise ValueError(f"closed form only implemented for dof 1 or 2, got {dof}")


@dataclass
class PowerLawFit:
    """log-log least squares of peak_flips against ||dw||_100."""

    amplitude: float
    exponent: float
    r2: float
    sigma_log: float
    dof: int
    n: int
    xbar: float
    sxx: float
    residuals_log: list[float]
    arms: list[str]

    def predict(self, dw: float) -> float:
        return self.amplitude * dw**self.exponent

    def se_pred(self, dw: float) -> float:
        """Prediction standard error for a NEW observation at ``dw``.

        This is the denominator rg1b's own BREAK bar uses and its sigma-report
        does not.  It adds parameter uncertainty (1/n) and leverage
        ((x-xbar)^2/Sxx) to the residual scatter, then the new point's own
        scatter (the leading 1).
        """
        x = math.log(dw)
        return self.sigma_log * math.sqrt(1.0 + 1.0 / self.n + (x - self.xbar) ** 2 / self.sxx)

    def studentize(self, dw: float, peak: float) -> dict[str, float]:
        predicted = self.predict(dw)
        residual = math.log(peak) - math.log(predicted)
        se = self.se_pred(dw)
        t = residual / se
        return {
            "dw_100": dw,
            "measured_peak_dpx": peak,
            "predicted_peak_dpx": predicted,
            "log_residual": residual,
            "residual_over_sigma_log_RG1B_STYLE": residual / self.sigma_log,
            "SE_pred": se,
            "t_studentized": t,
            "dof": self.dof,
            "p_two_sided": t_two_sided_p(t, self.dof),
        }


def fit_power_law(x_values: list[float], y_values: list[float], names: list[str]) -> PowerLawFit:
    """Least squares of ``ln y`` on ``ln x``.

    Used for peak-vs-displacement, end-vs-displacement, and (with the roles
    swapped) lr-vs-displacement, so the arguments are named generically rather
    than for the first caller.
    """
    x = np.log(np.asarray(x_values, dtype=np.float64))
    y = np.log(np.asarray(y_values, dtype=np.float64))
    n = x.size
    dof = n - 2
    if dof < 1:
        raise ValueError(f"need at least 3 points to estimate scatter, got {n}")
    xbar = float(x.mean())
    sxx = float(((x - xbar) ** 2).sum())
    slope = float(((x - xbar) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * xbar)
    residuals = y - (intercept + slope * x)
    ss_res = float((residuals**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return PowerLawFit(
        amplitude=math.exp(intercept),
        exponent=slope,
        r2=1.0 - ss_res / ss_tot,
        sigma_log=math.sqrt(ss_res / dof),
        dof=dof,
        n=n,
        xbar=xbar,
        sxx=sxx,
        residuals_log=[float(v) for v in residuals],
        arms=list(names),
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def adamw_direction(
    directory: Path, step: int, keys: list[str]
) -> tuple[np.ndarray, dict[str, float]]:
    """Realized AdamW next-step direction from the RETAINED moment state.

    ``u = mhat / (sqrt(vhat) + eps)`` with bias correction read from the
    optimizer's own per-parameter ``step`` counter and the retained
    ``param_groups`` betas/eps.  Nothing is assumed.
    """
    path = _checkpoints(directory)[step]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload["training_state"]["optimizer"]
    group = optimizer["param_groups"][0]
    beta1, beta2 = (float(b) for b in group["betas"])
    eps = float(group["eps"])
    live = _live_weights(payload)
    # Optimizer state is indexed by parameter position, in the same order the
    # model registered its parameters -- which is the state_dict order.
    order = [k for k in live if torch.is_tensor(live[k]) and live[k].is_floating_point()]
    if order != keys:
        raise RuntimeError("parameter ordering drifted between init and checkpoint")
    state = optimizer["state"]
    # The optimizer indexes its state by parameter POSITION.  That maps onto
    # ``keys`` only if every parameter is a float tensor and none were filtered
    # out; a filtered non-float tensor would shift every later index and
    # silently pair the wrong moments with the wrong weights.  The per-tensor
    # shape check below catches most permutations, but equal-numel tensors
    # could swap undetected, so the count is asserted up front.
    if len(state) != len(keys):
        raise RuntimeError(
            f"optimizer state has {len(state)} entries but {len(keys)} float parameters; "
            "positional index into state is not safe"
        )
    chunks: list[np.ndarray] = []
    sqrt_vhat_min = math.inf
    below_eps = 0
    total = 0
    dead = 0
    for index, key in enumerate(keys):
        entry = state[index]
        t = float(entry["step"])
        m = entry["exp_avg"].detach().to(torch.float64).reshape(-1).numpy()
        v = entry["exp_avg_sq"].detach().to(torch.float64).reshape(-1).numpy()
        if m.size != live[key].numel():
            raise RuntimeError(f"moment/parameter shape mismatch at {key}")
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        root = np.sqrt(vhat)
        chunks.append(mhat / (root + eps))
        # The sign-limit identity needs |g| >> eps, but a coordinate that never
        # received gradient has v == 0 exactly and contributes a zero step
        # either way.  Those are excluded rather than counted as violations:
        # the meaningful question is whether eps distorts the LIVE coordinates.
        live_mask = root > 0.0
        if live_mask.any():
            sqrt_vhat_min = min(sqrt_vhat_min, float(root[live_mask].min()))
        below_eps += int((live_mask & (root < 100.0 * eps)).sum())
        total += int(live_mask.sum())
        dead += int((~live_mask).sum())
    return np.concatenate(chunks), {
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "optimizer_step": t,
        "min_sqrt_vhat_over_LIVE_coords": sqrt_vhat_min,
        "live_coords_within_100x_eps": below_eps,
        "live_coords": total,
        "dead_coords_v_exactly_zero": dead,
        "fraction_of_live_within_100x_eps": below_eps / total if total else float("nan"),
    }


def instrument_control(arms: dict[str, Arm]) -> dict[str, Any]:
    """Fail-closed reproduction of every retained number, before anything new."""
    refit = json.loads(CONTROL_REFIT.read_text())
    band_receipt = json.loads(CONTROL_BAND.read_text())
    checks: list[dict[str, Any]] = []

    def _check(label: str, got: float, want: float) -> None:
        ok = math.isclose(got, want, rel_tol=CONTROL_RTOL, abs_tol=0.0)
        checks.append({"quantity": label, "recomputed": got, "retained": want, "match": ok})

    for row in refit["arms"]:
        arm = arms[row["arm"]]
        peak, end = peak_end_dpx(arm.history)
        _check(f"{arm.name}.dw_100", arm.dw[100], row["dw_100"])
        _check(f"{arm.name}.dw_600", arm.dw[600], row["dw_600"])
        _check(f"{arm.name}.peak_dpx", peak, row["peak_dpx"])
        _check(f"{arm.name}.end_dpx", end, row["end_dpx"])

    band = arms["band_a1"]
    band_peak, band_end = peak_end_dpx(band.history)
    reference = band_receipt["band_arm"]
    _check("band_a1.dw_100", band.dw[100], reference["dw_100"])
    _check("band_a1.dw_600", band.dw[600], reference["dw_600"])
    _check("band_a1.peak_dpx", band_peak, reference["peak_dpx"])
    _check("band_a1.end_dpx", band_end, reference["end_dpx"])

    failures = [c for c in checks if not c["match"]]
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "PASS": not failures,
        "weights_used": "training_state.model_state_dict (LIVE, not the deployed/EMA state_dict)",
        "control_refit_sha256": _sha256(CONTROL_REFIT),
        "control_band_sha256": _sha256(CONTROL_BAND),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/Volumes/APDataStore/pact/ddm_jr1")
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    init_payload = torch.load(args.init, map_location="cpu", weights_only=False)
    init_state = init_payload["state_dict"]
    keys = _float_keys(init_state)

    arms = {arm.name: arm for arm in ARMS}
    dw_vectors: dict[str, np.ndarray] = {}
    for arm in ARMS:
        dw_vectors[arm.name] = load_arm(arm, init_state, keys)
        print(f"[load] {arm.name}: steps={arm.steps} dw_100={arm.dw.get(100)!r}")

    control = instrument_control(arms)
    print(f"[control] {control['n_checks']} checks, {control['n_failures']} failures")
    if not control["PASS"]:
        for row in control["checks"]:
            if not row["match"]:
                print(f"  MISMATCH {row['quantity']}: {row['recomputed']} != {row['retained']}")
        raise SystemExit("instrument control FAILED -- unit voided, no numbers reported")

    # ---------------- LEG A ----------------
    stock_fit_arms = [a for a in ARMS if a.in_law_fit]
    peaks = {a.name: peak_end_dpx(a.history)[0] for a in ARMS}
    law = fit_power_law(
        [a.dw[100] for a in stock_fit_arms],
        [peaks[a.name] for a in stock_fit_arms],
        [a.name for a in stock_fit_arms],
    )

    held_out = {
        name: law.studentize(arms[name].dw[100], peaks[name]) for name in ("band_a1", "W1")
    }

    # Leave-one-out over the four fitted stock arms: refit on three (dof=1),
    # predict the held-out one.  This is the honest predictive scatter of the
    # law and it is what the band arm must be judged against.
    loo: list[dict[str, Any]] = []
    for held in stock_fit_arms:
        others = [a for a in stock_fit_arms if a.name != held.name]
        sub = fit_power_law(
            [a.dw[100] for a in others],
            [peaks[a.name] for a in others],
            [a.name for a in others],
        )
        row = sub.studentize(held.dw[100], peaks[held.name])
        row["arm"] = held.name
        row["fit_on"] = sub.arms
        loo.append(row)

    loo_abs_t = [abs(r["t_studentized"]) for r in loo]
    loo_abs_log = [abs(r["log_residual"]) for r in loo]

    # ---------------- LEG B ----------------
    a2, band = arms["A2"], arms["band_a1"]
    shared_steps = [s for s in a2.steps if s in band.dw and s in a2.dw]
    a2_index = {s: i for i, s in enumerate(a2.steps)}
    band_index = {s: i for i, s in enumerate(band.steps)}

    b1 = []
    for step in shared_steps:
        va = dw_vectors["A2"][a2_index[step]]
        vb = dw_vectors["band_a1"][band_index[step]]
        b1.append(
            {
                "step": step,
                "cos_trajectory_displacement": cosine(va, vb),
                "norm_A2": float(np.linalg.norm(va)),
                "norm_band": float(np.linalg.norm(vb)),
                "rotated_below_0p95": cosine(va, vb) < 0.95,
            }
        )

    # Per-window increments: does the rotation persist, or is it a transient?
    b1_windows = []
    for previous, step in itertools.pairwise(shared_steps):
        da = dw_vectors["A2"][a2_index[step]] - dw_vectors["A2"][a2_index[previous]]
        db = dw_vectors["band_a1"][band_index[step]] - dw_vectors["band_a1"][band_index[previous]]
        b1_windows.append(
            {"window": f"{previous}->{step}", "cos_increment": cosine(da, db)}
        )

    b2 = []
    adam_meta: dict[str, Any] = {}
    for step in shared_steps:
        ua, meta_a = adamw_direction(a2.directory, step, keys)
        ub, meta_b = adamw_direction(band.directory, step, keys)
        adam_meta[f"A2@{step}"] = meta_a
        adam_meta[f"band@{step}"] = meta_b
        b2.append(
            {
                "step": step,
                "cos_realized_adamw_direction": cosine(ua, ub),
                "cos_sign_of_direction": cosine(np.sign(ua), np.sign(ub)),
                "sign_agreement": float(np.mean(np.sign(ua) == np.sign(ub))),
                "note": "the two arms sit at DIFFERENT weights by this step; "
                "this is 'do the two runs step the same way', not a same-point counterfactual",
            }
        )

    # B3: the init identity's precondition, measured on retained moments.
    worst = min(m["min_sqrt_vhat_over_LIVE_coords"] for m in adam_meta.values())
    near_eps = sum(m["live_coords_within_100x_eps"] for m in adam_meta.values())
    live_total = sum(m["live_coords"] for m in adam_meta.values())
    b3 = {
        "identity": "after one AdamW step from zero moments: mhat=g, vhat=g^2, "
        "so u = g/(|g|+eps) = sign(g) * |g|/(|g|+eps)",
        "consequence": "at the init the realized AdamW step IS the sign limit, so rg1b's "
        "cos(sign g) column (0.2087 / 0.5235 / 0.6185 across the three phases) is the "
        "realized FIRST-STEP cosine, not a proxy for it -- the gap rg1b's Sec 6.5 "
        "declared open is closed at the init by identity, not by measurement",
        "precondition": "|g| >> eps on coordinates that actually receive gradient",
        "min_sqrt_vhat_over_LIVE_coords": worst,
        "eps": next(iter(adam_meta.values()))["eps"],
        "live_coords_within_100x_eps": near_eps,
        "live_coords_total": live_total,
        "fraction_of_live_distorted_by_eps": near_eps / live_total if live_total else float("nan"),
        "precondition_holds": near_eps == 0,
        "caveat": "coordinates with v == 0 exactly never received gradient and take a zero "
        "step under either objective; they are excluded, not counted as violations",
    }

    # ---- the END law, which rg1b fit but never scored the band arm against ----
    ends = {a.name: peak_end_dpx(a.history)[1] for a in ARMS}
    end_law = fit_power_law(
        [a.dw[600] for a in stock_fit_arms],
        [ends[a.name] for a in stock_fit_arms],
        [a.name for a in stock_fit_arms],
    )
    end_held_out = {
        name: end_law.studentize(arms[name].dw[600], ends[name]) for name in ("band_a1", "W1")
    }

    # ---- direct, law-free comparison of the matched pair ----
    a2_hist = dict(a2.history)
    band_hist = dict(band.history)
    direct = [
        {
            "step": step,
            "A2_quantized_exact_seg": a2_hist.get(step),
            "band_quantized_exact_seg": band_hist.get(step),
            "A2_dpx": (a2_hist[step] - INIT_SEG) * TOTAL_PX if step in a2_hist else None,
            "band_dpx": (band_hist[step] - INIT_SEG) * TOTAL_PX if step in band_hist else None,
            "A2_dw": a2.dw.get(step),
            "band_dw": band.dw.get(step),
        }
        for step in sorted(set(a2_hist) | set(band_hist))
    ]

    # ---- Leg C: the lr that puts the band arm at A2's displacement ----
    # ||dw||_100 ~ lr^p.  p is measured on the STOCK arms; the band arm's own p
    # is UNMEASURED, so this is a declared extrapolation from a single band
    # point, not a fitted law.  Stated as such in the ticket.
    lr_fit = fit_power_law(
        [a.dw[100] for a in stock_fit_arms],
        [a.lr for a in stock_fit_arms],
        [a.name for a in stock_fit_arms],
    )
    p_dw_vs_lr = 1.0 / lr_fit.exponent  # invert: we fit lr vs dw, want dw vs lr
    target = a2.dw[100]
    ratio = target / band.dw[100]
    leg_c = {
        "matching_target_dw100": target,
        "band_measured_dw100_at_lr_2e-5": band.dw[100],
        "band_over_stock_displacement_ratio_at_same_lr": band.dw[100] / target,
        "exponent_p_in_dw100_propto_lr_p_STOCK_ARMS": p_dw_vs_lr,
        "implied_band_lr_for_matched_displacement": band.lr * ratio ** (1.0 / p_dw_vs_lr),
        "assumption": "the band arm shares the stock arms' dw-vs-lr exponent; UNMEASURED, "
        "one band point only -- the ticket must bracket rather than trust this",
        "minimum_detectable_effect_of_the_residual_judge": minimum_detectable_effect(
            law, target
        ),
        "band_measured_log_effect": abs(held_out["band_a1"]["log_residual"]),
    }

    # ---------------- LEG C support ----------------
    displacement_curves = {
        arm.name: {
            "lr": arm.lr,
            "float_warmup_steps": arm.float_warmup_steps,
            "objective": arm.objective,
            "dw_by_step": {str(s): arm.dw[s] for s in arm.steps},
            "dw_by_step_float64": {str(s): arm.dw_float64[s] for s in arm.steps},
        }
        for arm in ARMS
    }

    # ---------------- PAYLOAD ----------------
    npz_path = args.out_dir / "JR1_VECTORS.npz"
    to_save: dict[str, np.ndarray] = {
        f"dw__{name}": vectors.astype(np.float32) for name, vectors in dw_vectors.items()
    }
    for name in ("A2", "band_a1"):
        for step in shared_steps:
            direction, _ = adamw_direction(arms[name].directory, step, keys)
            to_save[f"adamw__{name}__{step}"] = direction.astype(np.float32)
    # Unicode, not object: an object array would force allow_pickle on every
    # future read of this payload.
    to_save["param_key_order"] = np.array(keys, dtype="U")
    np.savez_compressed(npz_path, **to_save)

    receipt = {
        "schema": "ddm_jr1_judge_repair.v1",
        "score_claim": False,
        "axis": "[macOS-CPU advisory] read-back of retained payloads -- NEVER a score",
        "instrument_control": control,
        "provenance": {
            "init": str(args.init),
            "init_sha256": _sha256(args.init),
            "n_parameters": int(sum(init_state[k].numel() for k in keys)),
            "arms": {
                a.name: {
                    "directory": str(a.directory),
                    "lr": a.lr,
                    "float_warmup_steps": a.float_warmup_steps,
                    "objective": a.objective,
                    "in_law_fit": a.in_law_fit,
                }
                for a in ARMS
            },
        },
        "leg_a": {
            "law_fit_on": law.arms,
            "fit": {
                "amplitude": law.amplitude,
                "exponent": law.exponent,
                "r2": law.r2,
                "sigma_log": law.sigma_log,
                "n": law.n,
                "dof": law.dof,
                "residuals_log": law.residuals_log,
            },
            "critical_values": {
                "t95_dof2": t_critical(0.95, 2),
                "t90_dof2": t_critical(0.90, 2),
                "t99_dof2": t_critical(0.99, 2),
                "t95_dof1": t_critical(0.95, 1),
            },
            "peak_step_by_arm": {a.name: peak_step(a.history) for a in ARMS},
            "held_out": held_out,
            "end_law": {
                "fit_on": end_law.arms,
                "amplitude": end_law.amplitude,
                "exponent": end_law.exponent,
                "r2": end_law.r2,
                "sigma_log": end_law.sigma_log,
                "held_out": end_held_out,
                "note": "rg1b fit this law but never scored the band arm on it",
            },
            "leave_one_out": loo,
            "loo_summary": {
                "max_abs_t": max(loo_abs_t),
                "mean_abs_t": sum(loo_abs_t) / len(loo_abs_t),
                "max_abs_log_residual": max(loo_abs_log),
                "rms_log_residual": math.sqrt(sum(v * v for v in loo_abs_log) / len(loo_abs_log)),
            },
        },
        "leg_b": {
            "b1_trajectory_displacement_cosine": b1,
            "b1_window_increments": b1_windows,
            "b2_realized_adamw_direction_cosine": b2,
            "b2_moment_metadata": adam_meta,
            "b3_init_identity": b3,
        },
        "leg_c_support": {
            "displacement_curves": displacement_curves,
            "matched_displacement_derivation": leg_c,
            "direct_law_free_comparison_A2_vs_band": direct,
        },
        "payload": {
            "vectors_npz": str(npz_path),
            "sha256": _sha256(npz_path),
            "bytes": npz_path.stat().st_size,
            "contents": sorted(to_save),
        },
    }

    out_json = args.out_dir / "JR1_JUDGE_REPAIR.json"
    out_json.write_text(json.dumps(receipt, indent=2))
    print(f"[write] {out_json} sha256={_sha256(out_json)} bytes={out_json.stat().st_size}")
    print(f"[write] {npz_path} sha256={receipt['payload']['sha256']} bytes={receipt['payload']['bytes']}")

    print("\n=== LEG A ===")
    print(f"law on {law.arms}: A={law.amplitude:.6f} b={law.exponent:.9f} "
          f"r2={law.r2:.9f} sigma={law.sigma_log:.9f} dof={law.dof}")
    for name, row in held_out.items():
        print(
            f"  {name:8s} dw={row['dw_100']:.6f} peak={row['measured_peak_dpx']:.0f} "
            f"pred={row['predicted_peak_dpx']:.0f} r={row['log_residual']:+.5f} "
            f"rg1b-style={row['residual_over_sigma_log_RG1B_STYLE']:+.3f}sigma "
            f"HONEST t={row['t_studentized']:+.3f} p={row['p_two_sided']:.3f}"
        )
    print("  LOO:", [f"{r['arm']}:{r['t_studentized']:+.2f}" for r in loo])
    print(f"  peak step by arm: {receipt['leg_a']['peak_step_by_arm']}")
    print(f"  END law (r2={end_law.r2:.4f} sigma={end_law.sigma_log:.4f}): "
          + " ".join(f"{k}:t={v['t_studentized']:+.2f}" for k, v in end_held_out.items()))
    mde = leg_c["minimum_detectable_effect_of_the_residual_judge"]
    print(f"  MDE at n=4: {mde['MDE_log_at_this_n']:.4f} log "
          f"({mde['MDE_as_flip_ratio_at_this_n']:.3f}x) | asymptotic floor "
          f"{mde['asymptotic_floor_log_n_to_infinity']:.4f} log "
          f"({mde['asymptotic_floor_as_flip_ratio']:.3f}x) | band effect "
          f"{leg_c['band_measured_log_effect']:.4f} log")

    print("\n=== LEG C support ===")
    print(f"  matched-dw target {leg_c['matching_target_dw100']:.6f}; band at lr 2e-5 gives "
          f"{leg_c['band_measured_dw100_at_lr_2e-5']:.6f} "
          f"({leg_c['band_over_stock_displacement_ratio_at_same_lr']:.3f}x)")
    print(f"  implied matched band lr = {leg_c['implied_band_lr_for_matched_displacement']:.4e} "
          f"(assumes stock dw-vs-lr exponent p={leg_c['exponent_p_in_dw100_propto_lr_p_STOCK_ARMS']:.4f})")

    print("\n=== LEG B ===")
    for row in b1:
        print(f"  step {row['step']:4d}  cos(dw_band, dw_A2) = {row['cos_trajectory_displacement']:+.4f}")
    for row in b2:
        print(f"  step {row['step']:4d}  cos(AdamW dir)      = {row['cos_realized_adamw_direction']:+.4f}"
              f"  sign-agree {row['sign_agreement']:.3f}")
    print(f"  B3 precondition |g|>>eps holds: {b3['precondition_holds']} "
          f"(min sqrt(vhat) over LIVE coords={b3['min_sqrt_vhat_over_LIVE_coords']:.3e} "
          f"vs eps={b3['eps']:.0e}; {b3['live_coords_within_100x_eps']} of "
          f"{b3['live_coords_total']} live coords within 100x eps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
