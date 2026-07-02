"""Witness curriculum/behavior DSL — compiles to the proven trainer CLI.

See package docstring. This module is Layer-0 of the here->theta* bridge: a
declarative front-end whose programs compile to validated launch commands for
``experiments/train_levelset_witness_realized_through_R_mlx.py``.

Design (recursion+math+enforced-behaviors, operator riff 2026-06-28):
  * The contest energy S = 100*INT d_seg + sqrt(10*INT d_pose) + 25*bytes is the
    ROOT; every lever is a term/relaxation of S, so composition is principled.
  * The curriculum is a homotopy of relaxations (CE -> tau -> l7), expressed as
    ``Stage`` tuples; the temperature anneal is a ``Schedule``.
  * Desired behaviors (preserve / contain / authority) are ENFORCED clauses, not
    advisory prose — ``validate()`` refuses a program that violates them, and
    ``compile_*`` bakes them into the emitted commands.
  * never-invent-flags is STRUCTURAL: ``validate()`` checks every emitted flag
    against the trainer's real argparse flag set (``real_trainer_flags``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_REL = "experiments/train_levelset_witness_realized_through_R_mlx.py"
TRAINER_PATH = _REPO_ROOT / TRAINER_REL


def real_trainer_flags(trainer_path: Path | None = None) -> frozenset[str]:
    """Parse the trainer's argparse and return the SET of real ``--flag`` names.

    This is the structural never-invent-flags guard: a program that emits a flag
    not in this set fails ``validate()`` before any launch.
    """
    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    text = path.read_text()
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


def real_store_true_flags(trainer_path: Path | None = None) -> frozenset[str]:
    """Flags whose action is ``store_true`` — these have NO ``--no-<flag>`` form,
    so emitting them as False (which ``compile`` renders as ``--no-X``) would crash
    argparse at launch. (DSL adversarial-review C2, 2026-06-28.)"""
    path = Path(trainer_path) if trainer_path is not None else TRAINER_PATH
    text = path.read_text()
    return frozenset(re.findall(
        r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*action\s*=\s*["\']store_true["\']', text))


# sentinel so with_lever() can explicitly CLEAR resume_from (fresh run) vs inherit it
_INHERIT = object()


# ---------------------------------------------------------------------------
# Schedule primitives (the homotopy / anneal math)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Anneal:
    """A cosine-annealed schedule start->end (e.g. softmax temperature tau)."""

    start: float
    end: float

    def flags(self, start_flag: str, end_flag: str) -> dict:
        return {start_flag: self.start, end_flag: self.end}


def Freeze(value: float) -> Anneal:  # noqa: N802 (DSL keyword)
    """Freeze a schedule at a constant value (Anneal with start==end)."""
    return Anneal(value, value)


# ---------------------------------------------------------------------------
# Curriculum stage (a relaxation of S) + regularizers (live PDE constraints)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """A curriculum relaxation. ``start_epoch`` maps to the trainer's stage gate."""

    name: str
    start_epoch_flag: str | None  # e.g. "--tau-softplus-start-epoch"; None for the CE base
    start_epoch: int | None = None

    def flags(self) -> dict:
        if self.start_epoch_flag is None or self.start_epoch is None:
            return {}
        return {self.start_epoch_flag: self.start_epoch}


@dataclass(frozen=True)
class Regularizer:
    """A live derivative/integral regularizer (eikonal |grad phi|=1, length INT ds)."""

    flag: str
    weight: float

    def flags(self) -> dict:
        return {self.flag: self.weight}


# ---------------------------------------------------------------------------
# Lever (an A/B toggle = a flag override set + optional epoch extension)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lever:
    """A named A/B lever: a set of flag overrides + optional extra epochs.

    Levers COMPOSE by merging their override dicts (later levers win on conflict),
    which is exactly theta* composition (binding the winning fragments).
    """

    name: str
    overrides: dict = field(default_factory=dict)
    epochs_delta: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Enforced-behavior clauses (preserve / contain / authority)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Preserve:
    """PRESERVE: per-stage boundary ckpts + intra-stage cadence (<=25, binding)."""

    stage_boundaries: bool = True
    ckpt_every: int = 25

    def flags(self) -> dict:
        f = {"--ckpt-every": self.ckpt_every}
        # --stage-checkpoints is BooleanOptionalAction default True; emit explicitly.
        f["--stage-checkpoints"] = bool(self.stage_boundaries)
        return f


@dataclass(frozen=True)
class Contain:
    """CONTAIN: daemon-level blast-radius bounds (>=10GB floor, RSS cap)."""

    min_free_gb: float = 10.0
    projected_gb: float = 40.0
    rss_cap_mb: int = 90000
    walltime_cap_s: int = 288000


@dataclass(frozen=True)
class Authority:
    """AUTHORITY: the verdict contract. macOS-MLX/CPU is ADVISORY; only a
    byte-closed contest-CPU/CUDA exact row is a score. Recorded, asserted."""

    realized_through_R: bool = True
    numpy_fp32_reference: bool = True
    advisory_until_byte_closed: bool = True


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WitnessProgram:
    out_dir: str
    gt_cache: str
    epochs: int
    num_pairs: int
    temp: Anneal
    stages: tuple[Stage, ...]
    regularizers: tuple[Regularizer, ...]
    preserve: Preserve
    contain: Contain
    authority: Authority
    base: dict = field(default_factory=dict)  # substrate flags (arch, basis, chroma, ...)
    levers: tuple[Lever, ...] = ()
    resume_from: str | None = None
    mlx_device: str = "gpu"
    # the fixed gauge (a tac.witness_dsl.gauge.GaugeChoice); annotated lazily (object) so
    # curriculum_dsl never imports the gauge module at load time (no import cycle). Does NOT
    # affect flag_dict() — it is the chart-selection meta-layer ABOVE the trainer flags.
    gauge: object | None = None

    # --- composition ---------------------------------------------------------
    def with_lever(self, *levers: Lever, resume_from=_INHERIT,
                   out_dir: str | None = None) -> "WitnessProgram":
        """Return a new program with levers appended (theta* composition step).

        Epochs are extended by the sum of the levers' ``epochs_delta`` (e.g. a
        Muon finisher adds its window on top of the warm-start epoch).

        ``resume_from`` defaults to INHERIT (keep the base's); pass ``None`` to
        explicitly CLEAR it for a fresh run, or a path to override (DSL review M2)."""
        new_epochs = self.epochs + sum(lv.epochs_delta for lv in levers)
        new_resume = self.resume_from if resume_from is _INHERIT else resume_from
        return replace(
            self,
            levers=self.levers + tuple(levers),
            epochs=new_epochs,
            resume_from=new_resume,
            out_dir=out_dir if out_dir is not None else self.out_dir,
        )

    def with_gauge(self, gauge_choice=None, *, table=None,
                   warp=None, carrier=None, residual=None, pose=None,
                   movables=None, generation=None) -> "WitnessProgram":
        """Fix the gauge for this program (the gauge-FIXING step of the 4-layer stack,
        FEED-ji). Returns a NEW program with a validated ``GaugeChoice`` attached; this one
        is UNMUTATED (pure composition, parallel to ``with_lever``). Composes:
        ``BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF, ...).with_lever(...)``.

        Pass a full ``GaugeChoice`` as the positional, OR per-component keyword overrides
        (unspecified components inherit this program's current gauge, else the canonical
        gauge). ``validate`` raises ``GaugeViolation`` if any selected chart is
        non-compliant / non-deterministic per the cost ``table`` (BY CONSTRUCTION).

        Imported lazily so ``curriculum_dsl`` never imports ``gauge`` at module-load time
        (the gauge module imports THIS module — lazy import breaks the cycle).
        """
        from tac.witness_dsl.gauge import GaugeChoice, CANONICAL_GAUGE
        if gauge_choice is None:
            base_gauge = self.gauge if isinstance(self.gauge, GaugeChoice) else CANONICAL_GAUGE
            overrides = {k: v for k, v in dict(
                warp=warp, carrier=carrier, residual=residual,
                pose=pose, movables=movables, generation=generation).items() if v is not None}
            gauge_choice = replace(base_gauge, **overrides)
        elif not isinstance(gauge_choice, GaugeChoice):
            raise TypeError("with_gauge expects a GaugeChoice (or per-component keyword charts)")
        gauge_choice.validate(table)
        return replace(self, gauge=gauge_choice)

    # --- flag assembly -------------------------------------------------------
    def flag_dict(self) -> dict:
        f: dict = {}
        f.update(self.base)
        f["--num-pairs"] = self.num_pairs
        f["--epochs"] = self.epochs
        f["--gt-cache"] = self.gt_cache
        f["--out-dir"] = self.out_dir
        f["--mlx-device"] = self.mlx_device
        f.update(self.temp.flags("--softmax-temp-start", "--softmax-temp-end"))
        for st in self.stages:
            f.update(st.flags())
        for rg in self.regularizers:
            f.update(rg.flags())
        f.update(self.preserve.flags())
        if self.resume_from is not None:
            f["--resume-from"] = self.resume_from
        # levers LAST so they override (the A/B toggle wins)
        for lv in self.levers:
            f.update(lv.overrides)
        return f

    # --- validation (structural never-invent-flags + behavior clauses) -------
    def validate(self, trainer_path: Path | None = None) -> list[str]:
        """Return a list of violations (empty == valid)."""
        problems: list[str] = []
        real = real_trainer_flags(trainer_path)
        fd = self.flag_dict()
        for flag in fd:
            if flag not in real:
                problems.append(f"INVENTED FLAG (not in trainer argparse): {flag}")
        # C2 (review): a False on a store_true flag compiles to --no-X → argparse crash
        store_true = real_store_true_flags(trainer_path)
        for flag, val in fd.items():
            if val is False and flag in store_true:
                problems.append(
                    f"INVALID --no-{flag[2:]}: {flag} is store_true (no --no- form); "
                    "False would crash argparse at launch")
        # C1 (review): DEAD ARM — resuming from a ckpt at/after epochs == zero gradient steps
        if self.resume_from is not None:
            try:
                import numpy as _np
                _p = Path(self.resume_from)
                if _p.exists():
                    _z = _np.load(_p, allow_pickle=True)
                    _ep = None
                    for _k in ("epoch", "__epoch", "__resume_epoch"):
                        if _k in _z.files:
                            _ep = int(_z[_k]); break
                    if _ep is not None and self.epochs <= _ep:
                        problems.append(
                            f"DEAD ARM: epochs={self.epochs} <= resume epoch {_ep} → "
                            "range(start,epochs) empty → ZERO gradient steps (give the "
                            "lever an epochs_delta window)")
            except Exception:
                pass  # validation must not hard-fail on a missing/odd ckpt
        # CURRICULUM ORDERING — surface the trainer's runtime assert at DSL-validate time so a
        # doomed config (tau/l7 stages that silently never run) is refused BEFORE any launch.
        _curr = fd.get("--curriculum")
        if _curr is True or _curr == 1:
            _tau_s = fd.get("--tau-softplus-start-epoch")
            _l7_s = fd.get("--l7-start-epoch")
            _ep = fd.get("--epochs")
            if None not in (_tau_s, _l7_s, _ep) and not (0 < _tau_s < _l7_s <= _ep):
                problems.append(
                    f"CURRICULUM ORDERING: need 0 < tau_start ({_tau_s}) < l7_start ({_l7_s}) "
                    f"<= epochs ({_ep}) (trainer asserts this; else tau/l7 stages never run)")
        # PRESERVE: ckpt cadence binding (<=25)
        if self.preserve.ckpt_every <= 0 or self.preserve.ckpt_every > 25:
            problems.append(
                f"PRESERVE violation: --ckpt-every={self.preserve.ckpt_every} (must be 1..25)")
        if not self.preserve.stage_boundaries:
            problems.append("PRESERVE violation: stage-boundary ckpts disabled")
        # CONTAIN: >=10GB floor binding
        if self.contain.min_free_gb < 10.0:
            problems.append(
                f"CONTAIN violation: min_free_gb={self.contain.min_free_gb} (<10GB floor)")
        # AUTHORITY: realized-through-R required for a trustworthy verdict
        if not self.authority.realized_through_R:
            problems.append("AUTHORITY violation: realized_through_R must be True")
        return problems

    # --- compilation ---------------------------------------------------------
    def compile_trainer_argv(self, python: str = ".venv/bin/python") -> list[str]:
        argv = [python, TRAINER_REL]
        for flag, val in self.flag_dict().items():
            if val is True:
                argv.append(flag)
            elif val is False:
                # BooleanOptionalAction: emit --no-<name>
                argv.append(flag.replace("--", "--no-", 1))
            else:
                argv.extend([flag, str(val)])
        return argv

    def compile_daemon_argv(self, label: str, log: str,
                            python: str = ".venv/bin/python") -> list[str]:
        """Wrap the trainer in the canonical durable daemon + containment caps."""
        argv = [
            python, "tools/spawn_durable_daemon.py",
            "--label", label, "--log", log,
            "--projected-gb", str(self.contain.projected_gb),
            "--min-free-gb", str(self.contain.min_free_gb),
            "--rss-cap-mb", str(self.contain.rss_cap_mb),
            "--walltime-cap-s", str(self.contain.walltime_cap_s),
            "--",
        ]
        argv.extend(self.compile_trainer_argv(python=python))
        return argv


# ---------------------------------------------------------------------------
# BASELINE — the exact completed CE->tau->l7 run, expressed as a program.
# (round-trip target: BASELINE.flag_dict() reproduces the launched config.)
# ---------------------------------------------------------------------------
_CE_CKPT = ("experiments/results/levelset_amort_decoder_n200_20260627T143830Z/"
            "levelset_resume_stageCE_ep299.npz")
_L7_CKPT = ("experiments/results/levelset_l7_preserved_snapshots/"
            "levelset_resume_stageL7_ep1500.npz")

BASELINE = WitnessProgram(
    out_dir="experiments/results/levelset_amort_deconf_n200_taualone_20260627T194432Z",
    gt_cache="experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz",
    epochs=1500,
    num_pairs=200,
    temp=Anneal(1.0, 0.05),
    stages=(
        Stage("CE", None, None),
        Stage("tau_softplus", "--tau-softplus-start-epoch", 300),
        Stage("l7_softplus", "--l7-start-epoch", 900),
    ),
    regularizers=(
        Regularizer("--eikonal-weight", 0.01),
        Regularizer("--length-weight", 0.001),
    ),
    preserve=Preserve(stage_boundaries=True, ckpt_every=25),
    contain=Contain(min_free_gb=10.0, projected_gb=40.0, rss_cap_mb=90000),
    authority=Authority(),
    resume_from=_CE_CKPT,
    base={
        "--render-h": 384, "--render-w": 512,
        "--hidden-dim": 96, "--mod-dim": 32,
        "--activation": "hosc", "--siren-init": True,
        "--curriculum": True,
        "--palette-anchor": True, "--self-orient": True, "--reorient-every": 50,
        "--freq-across": 32, "--n-dir-freqs": 2, "--freq-along": 4, "--max-bank-freq": 64,
        "--chroma": True,
        "--lane-edge-weight": 0, "--lane-edge-class": 1, "--lane-margin-target": 0.5,
        "--lane-edge-start-epoch": 300,
        "--w-seg": 100, "--w-pose": 1.0,
        "--ema-decay": 0.997, "--accum-pairs": 8, "--grad-clip": 1.0,
        "--verdict-pairs": 96, "--eval-every": 25,
        "--async-verdict": True,
    },
)


# ---------------------------------------------------------------------------
# Lever library (the A/B campaign, as composable DSL fragments)
# ---------------------------------------------------------------------------
def PoseDecouple(window: int = 100) -> Lever:  # noqa: N802 (DSL keyword) — A5
    """A5: drop pose from the loss (w-pose=0) to free decoder capacity for d_seg —
    a TRADE (d_pose worsens; pose is carried in-frame, NOT sidecar-able, per the
    byte-close finding). Carries a warm-start window (else dead-arm, review C1)."""
    return Lever("A5_pose_decouple", overrides={"--w-pose": 0.0}, epochs_delta=window,
                 notes="drop pose-loss to free d_seg capacity (trades d_pose up)")


def Muon(start_epoch: int, window: int = 100) -> Lever:  # noqa: N802 — A4
    """A4: Muon finisher from ``start_epoch`` for ``window`` epochs, with moments
    reset at the optimizer-stage transition and tau FROZEN at 0.05 (the run's
    final hard temperature) for apples-to-apples. muon-lr auto-derives 0.1*lr."""
    return Lever(
        "A4_muon",
        overrides={
            "--muon-start-epoch": start_epoch,
            "--stage-transition-reset-moments": True,
            "--softmax-temp-start": 0.05,  # freeze tau at the l7-end value
            "--softmax-temp-end": 0.05,
        },
        epochs_delta=window,
        notes="Muon finisher; is it the d_seg finisher (conditioning) or no?",
    )


def DirectionalBasis(weight: float = 0.5, start_epoch: int = 300,  # noqa: N802
                     window: int = 100) -> Lever:
    """Turn the lane-edge directional term ON (the completed run had weight 0).
    The all-class directional/tangent basis measured -48% d_seg earlier.
    Carries a warm-start window (else dead-arm, review C1)."""
    return Lever("directional_basis",
                 overrides={"--lane-edge-weight": weight,
                            "--lane-edge-start-epoch": start_epoch},
                 epochs_delta=window,
                 notes="all-class directional tangent basis (was OFF: weight 0)")


def TauFrozen(value: float = 0.05, window: int = 100) -> Lever:  # noqa: N802 — A1b isolation
    """A1b: freeze tau (start==end) to isolate an l7 effect from the tau anneal.

    MUST carry an ``epochs_delta`` (the warm-start window) or the arm runs ZERO
    gradient steps when resumed from an end-of-run ckpt (DSL review C1, 2026-06-28:
    epochs==resume_epoch → empty range → scientifically-dead arm)."""
    return Lever("A1b_tau_frozen",
                 overrides={"--softmax-temp-start": value, "--softmax-temp-end": value},
                 epochs_delta=window,
                 notes="freeze tau to isolate l7-loss vs tau-anneal (diff refutation)")


def SoftBoundary(beta: float = 2.0, window: int = 100) -> Lever:  # noqa: N802
    """Anti-aliased SOFT boundary (lower HOSC beta) — tests Signal's hypothesis that
    a soft edge carries sub-pixel boundary position through R better than a hard
    step (β→∞). Replaces the confounded constant-β≈16 'beta_steplim' arm (review H2)."""
    return Lever("soft_boundary",
                 overrides={"--hosc-beta": beta},
                 epochs_delta=window,
                 notes="soft anti-aliased edge (low beta) for sub-pixel R-survival")


def FiLMFix(per_layer: bool = True, concat_code: bool = True,  # noqa: N802 — LEVER-A
            rank_floor_weight: float = 0.0, rank_floor_target: float = 4.0,
            window: int = 100) -> Lever:
    """LEVER-A (FiLM-rank-fix): attack the MEASURED per-pair FiLM modulation participation-ratio
    collapse (3.34@CE -> 1.27@tau -> 1.19@l7; 91.8% of per-pair variation in ONE axis) that caps
    d_seg AND held-out amortization. Composes three default-OFF trainer routes:

      * ``per_layer`` -> ``--film-per-layer``: SEPARATE per-layer RESIDUAL FiLM (identity at init) =
        more INDEPENDENT multiplicative modulation routes.
      * ``concat_code`` -> ``--film-concat-code``: an ADDITIVE per-pair code-injection (folded concat;
        identity at init) = a NON-collapsing per-pair TRANSLATION route (what a moving lane needs).
      * ``rank_floor_weight`` > 0 -> ``--film-rank-floor-weight``/``--film-rank-floor-target``: a soft
        participation-ratio FLOOR penalty so the curriculum cannot funnel the modulation to rank-1.

    Emits ONLY flags that are turned on (store_true flags are never emitted False, per DSL review C2).
    Carries a warm-start ``window`` (else dead-arm when resumed at end-of-run, review C1)."""
    ov: dict = {}
    if per_layer:
        ov["--film-per-layer"] = True
    if concat_code:
        ov["--film-concat-code"] = True
    if rank_floor_weight > 0.0:
        ov["--film-rank-floor-weight"] = rank_floor_weight
        ov["--film-rank-floor-target"] = rank_floor_target
    return Lever("A_film_rank_fix", overrides=ov, epochs_delta=window,
                 notes="FiLM rank-fix: per-layer + concat-code + rank-floor (attacks PR collapse)")


def LanePrior(weight: float = 1.0, start_epoch: int = 300,  # noqa: N802 — LEVER-B
              lane_class: int = 1, radius: int = 4, target: float = 0.5,
              window: int = 100) -> Lever:
    """LEVER-B (thin-lane dropped-dash prior): up-weight the realized through-R seg margin hinge on
    THIN GT-lane structures the unweighted mean loss drops (MEASURED: 57% Road<->Lane confusion, PC0
    = Lane->Road DROP, 52.7% of GT-lane components wholesale-missed, miss-fraction monotone in dash
    size). A precomputed thin-lane weight map (local lane density in a (2r+1)^2 window) concentrates
    pressure on the thin dashes. Carries a warm-start ``window`` (else dead-arm, review C1).

    NOTE: this is the ``--lane-thin-*`` realized-margin prior; it is DISTINCT from the
    ``--lane-prior-phi1`` structured-init lane-SDF flag (a different mechanism)."""
    return Lever("B_lane_thin_prior",
                 overrides={"--lane-thin-weight": weight,
                            "--lane-thin-start-epoch": start_epoch,
                            "--lane-thin-class": lane_class,
                            "--lane-thin-radius": radius,
                            "--lane-thin-target": target},
                 epochs_delta=window,
                 notes="thin-lane dropped-dash prior (realized margin hinge weighted by thinness)")


def AnalyticLaneRenderBand(  # noqa: N802 — FEED-dv render-band lever
    softness: float = 1.0, dash_forward_max_m: float = 55.0,
    uncertainty_source: str = "witness", tau: float = 0.85, eps: float = 0.35,
    weight: float = 1.0, start_epoch: int = 300, window: int = 200,
) -> Lever:
    """FEED-dv (#203/#213/#215) analytic-lane RENDER-BAND: composite the analytic
    openpilot lane band OVER the witness render BEFORE R (the ``compose_fn`` hook), so
    the frozen SegNet reads the composited frame and the d_seg loss backprops into the
    witness. NON-NAIVE form (the naive band HURT +0.00082, sizing c3): AA-SDF coverage x
    RANGE-DEPENDENT dash gate (#215) x WITNESS-UNCERTAINTY mask (rides #141 margin) so the
    band paints ONLY where the witness ERASES the lane, killing the dash-gap FALSE-POSITIVE.

    Impl: ``tac.boundary_math.analytic_lane_render_band.make_lane_band_compose_fn`` passed
    to the trainer's ``render_fn`` (compose hook). MEASURED post-hoc n600: the levers take
    naive +0.00082 -> ~break-even; the NET-NEGATIVE win is realized by TRAINING WITH the
    band active (the witness re-adapts its boundaries; sizing VERDICT).

    PAIRED WITH the trainer wire-in (docs/analytic_lane_render_band_wire_in_spec.md): the
    ``--lane-band-*`` flags LAND WITH the compose wire-in; until then this Lever will be
    REFUSED by the never-invent-flags validator (fail-closed = correct)."""
    return Lever("FEED_dv_analytic_lane_render_band",
                 overrides={"--lane-render-band": True,
                            "--lane-band-softness": softness,
                            "--lane-band-dash-forward-max-m": dash_forward_max_m,
                            "--lane-band-uncertainty-source": uncertainty_source,
                            "--lane-band-tau": tau,
                            "--lane-band-eps": eps,
                            "--lane-band-weight": weight,
                            "--lane-band-start-epoch": start_epoch},
                 epochs_delta=window,
                 notes="analytic-lane render-band compose (AA-SDF x range-dash-gate x "
                       "witness-uncertainty); FP-killed non-naive form; realized THROUGH R")


def StiefelW(window: int = 100) -> Lever:  # noqa: N802 — DM1a
    """DM1a (Stiefel-W): per-step project film.weight onto orthonormal columns (WᵀW=I) so W is an
    ISOMETRY => PR(M)=PR(cov(code)) to the projection's ~1e-2 residual (the byte-free root half-1 of the
    FiLM rank-collapse cure; design memo per_stage_fractal_optimizer §0/§4). store_true => emitted ONLY
    when on (never False, review C2). Carries a warm-start ``window`` (else dead-arm when resumed at
    end-of-run, C1)."""
    return Lever("DM1a_stiefel_w",
                 overrides={"--film-stiefel": True},
                 epochs_delta=window,
                 notes="Stiefel-orthonormal film.weight (WᵀW=I => PR(M)~PR(cov code) to ~1e-2 residual; "
                       "global-magnitude WD on W neutralized)")


def CodeSpectralEntropy(beta: float = 0.01, window: int = 100) -> Lever:  # noqa: N802 — DM1b
    """DM1b (code spectral-entropy): add the CAPACITY penalty -beta*log(PR(cov(code))) keeping all
    code directions live (the byte-free root half-2; design memo §0/§4). A value flag (not store_true);
    omitted when beta<=0 (off). Carries a warm-start ``window`` (else dead-arm when resumed, C1)."""
    ov: dict = {}
    if beta > 0.0:
        ov["--code-spectral-entropy-weight"] = beta
    return Lever("DM1b_code_spectral_entropy",
                 overrides=ov,
                 epochs_delta=window,
                 notes="spectral-entropy CAPACITY penalty on cov(code) (raises PR(cov code) => PR(M))")


def DM1Minimal(beta: float = 0.01, window: int = 100) -> tuple[Lever, Lever]:  # noqa: N802 — A3
    """DM1 minimal cure = Stiefel-W + code-spectral-entropy (design memo §4 the 80/20). Returns the
    two composable levers (use ``BASELINE.with_lever(*DM1Minimal())``); both halves target DIFFERENT
    params (W via projection, code via penalty) so they compose without double-counting (§3 routing).
    The per-stage moment-reset (the third minimal item) is the existing ``--stage-transition-reset-moments``
    (already wired); add ``Muon(...)`` or ``StiefelW(window=...)`` arms to engage it at a boundary.

    The warm-start ``window`` is carried ONCE (on the Stiefel lever); the entropy lever uses
    ``window=0`` so composing both extends epochs by ``window`` (not ``2*window``)."""
    return StiefelW(window=window), CodeSpectralEntropy(beta=beta, window=0)


def MarginSaliency(weight: float = 1.0, start_epoch: int = 900,  # noqa: N802 — LEVER (KKT waterfill)
                   tau: float = 0.5, target: float = 0.5, window: int = 100) -> Lever:
    """Margin-saliency hinge (KKT waterfill on margin-saliency, `boundary_routing.py`) engaged
    LATE (l7/Muon finetune; from-scratch margin starves the interior). Composes with ``UniWARD``
    (the texture mask) + ``DirectionalBasis`` (the curvelet basis) in the synergy map. Carries a
    warm-start ``window`` (else dead-arm when resumed at end-of-run, review C1)."""
    return Lever("margin_saliency",
                 overrides={"--margin-saliency-weight": weight,
                            "--margin-saliency-start-epoch": start_epoch,
                            "--margin-saliency-tau": tau,
                            "--margin-saliency-target": target},
                 epochs_delta=window,
                 notes="KKT-waterfill margin-saliency hinge (late finetune)")


def UniWARD(weight: float = 1.0, start_epoch: int = 900, beta: float = 4.0,  # noqa: N802 — LEVER-4
            tau: float = 0.5, target: float = 0.5, window: int = 100) -> Lever:
    """LEVER-4 (UniWARD inverse-steganalysis, Fridrich; BUILT + smoke-verified, `uniward_texture.py`
    `compute_texture_probability` + `uniward_delta.py`): margin-saliency with the UNIWARD texture
    DOWN-weight ``sal /= (1+beta*tex)`` (tex = stop-grad spatial-gradient energy of the realized
    frame) -> let error HIDE in textured (SegNet-undetectable) regions, CONCENTRATE correctness on
    the smooth boundary. On-theme: the contest IS inverse steganalysis. A LATE-STAGE (l7/Muon) A/B
    arm; composes with ``MarginSaliency`` + ``DirectionalBasis`` (the curvelet directional basis) in
    the synergy map. ``--margin-saliency-uniward`` is store_true -> emitted True ONLY (never False,
    review C2). Carries a warm-start ``window`` (else dead-arm, review C1)."""
    return Lever("LEVER4_uniward",
                 overrides={"--margin-saliency-weight": weight,
                            "--margin-saliency-start-epoch": start_epoch,
                            "--margin-saliency-uniward": True,
                            "--margin-saliency-uniward-beta": beta,
                            "--margin-saliency-tau": tau,
                            "--margin-saliency-target": target},
                 epochs_delta=window,
                 notes="UniWARD texture-masked margin-saliency (Fridrich; concentrate on smooth boundary)")


def WarpRealLumaFrame0(  # noqa: N802 — DSL constructor
    w_pose: float = 1.0, start_epoch: int = 0, window: int = 0,
) -> Lever:
    """POSE CARRIER B — warp-real-luma FRAME0 (``tac.boundary_math.warp_real_luma_frame0``).

    Engages the pose term (``--w-pose``, default 0.0 = pose-blind) so the SE(3)-twist
    residual trains to close d_pose from the deterministic warp floor (~2.6-10.5 at n600
    advisory) toward ~3.4e-5. The CARRIER ITSELF is a CODE wire-in (NOT a trainer flag):
    the parent routes the f0 render slot through
    ``WarpRealLumaFrame0Carrier.make_pair_render_dispatch(...)`` and passes it to
    ``make_loss_fn(render_fn=...)`` (even code_idx=f0 -> warp(gt_f0, xi) through R; odd=f1
    -> witness). frame0 is seg-free (upstream/modules.py:108) so this lever CANNOT disturb
    d_seg — it side-steps the W8 d_seg-vs-d_pose warp-scale crux. Byte cost = the per-pair
    6-DOF twist (~875 B/600 fp16, ~325 B low-rank r2; dual-use with the stored pose).

    Gauge: ``WarpGauge.SCREW_TWIST`` x ``PoseGauge.LOW_RANK`` (the twist IS the pose).
    Composes with every d_seg lever (orthogonal frame). MEASURED reference: FEED-lj/W7.
    ``w_pose`` is the pose-term weight; ``--pose-eps`` is left at the trainer default.
    """
    return Lever("warp_real_luma_frame0_pose_carrier",
                 overrides={"--w-pose": w_pose},
                 epochs_delta=window,
                 notes=("pose carrier B: warp-real-luma frame0 (SE(3)-twist ground-homography, seg-free); "
                        "render_fn code wire-in via make_pair_render_dispatch; --w-pose>0 trains the "
                        "rank-6 twist residual to d_pose~3.4e-5 (FEED-lj/W7; advisory pointer 0.19110)"))


# ---------------------------------------------------------------------------
# The FIXED, KNOWN OPENING of the from-scratch openpilot-seeded d_seg curriculum.
# (S0 seed -> S1 short-CE -> S2 tau_softplus). l7 + Muon are STACKED ADAPTIVELY by
# ``campaign.plan_adaptive_step`` off this opening's measured per-stage checkpoints.
# Deep-math anchors: FEED-bv (measured per-stage d_seg dirs), FEED-fs (separatrix
# seed), FEED-fz/-bu (reheat), anneal-memo (tau=0.3 == reachability floor), FEED-fi
# (Muon = spectral conditioner -> stacked, not fixed). DAG FEED-ln.
# ---------------------------------------------------------------------------
def openpilot_seeded_opening(  # noqa: N802 — DSL constructor
    out_dir: str,
    gt_cache: str,
    num_pairs: int = 200,
    *,
    ce_to: int = 300,
    tau_window: int = 300,
    tau: float = 0.3,
    w_pose: float = 0.0,
    rewarmup_epochs: int = 8,
    rewarmup_floor: float = 0.1,
    seed: int = 0,
    mlx_device: str = "gpu",
) -> WitnessProgram:
    """The FIXED, KNOWN OPENING (S0 seed -> S1 short-CE -> S2 tau_softplus) as ONE program.

    The curriculum is NOT fully fixed up front (operator riff 2026-06-29): we KNOW the
    opening; l7 + Muon are STACKED ADAPTIVELY from the MEASURED tau-stage d_seg trajectory
    off the per-stage checkpoints (see ``campaign.plan_adaptive_step`` / ``decide_next_stage``).

    S0 (pre-train seed, NOT an epoch stage): ``--structured-init`` + ``--lane-prior-phi1``
       inject the openpilot deg-3 centerline SIGNED-DISTANCE field into the phi1 (lane)
       channel of the structured-init pretrain target (FEED-fs separatrix residual 1.9e-5)
       -> the level-set homotopy STARTS in-basin AT the Road<->Lane separatrix. NTK view:
       the seed supplies the LOW-FREQUENCY lane structure free, so CE need not learn it.
    S1 CE [1, ce_to): SHORT confidence-calibration over all pixels. The seed gives the
       geometry (zero-level-set placement); CE only calibrates per-pixel argmax confidence
       -> SHORTENED (not eliminated). Measured CE descent 0.01045 -> 0.00643 (FEED-bv).
    S2 tau_softplus [ce_to, ce_to+tau_window): the PRIMARY measured d_seg drop
       (0.00643 -> 0.00396, FEED-bv). ``tau=0.3`` == the anneal-memo reachability floor
       Delta_min ~= 0.3 (the margin-RESONANCE T*=Delta for the *fixable* boundary flips;
       grad ∝ (1/T)e^{-Delta/T} peaks at T=Delta).

    l7 + Muon parked: ``--l7-start-epoch`` is set to ``epochs`` (no-op tail) so the opening
    is EXACTLY ce->tau and the trainer validator ``tau_start < l7_start <= epochs`` holds;
    the adaptive engine engages l7 (and then the Muon finisher) by warm-starting from the
    preserved tau checkpoint.

    REHEAT (FEED-fz BUILD 1 / FEED-bu, "different stages need different treatment") is ON at
    every transition: ``--stage-transition-rewarmup-epochs`` (LR floor->1x over the window,
    measured 0.1x/~8ep) + ``--stage-transition-reset-moments`` (zero stale AdamW 2nd-moments)
    -> the ce->tau boundary is stable BY CONSTRUCTION. smooth + lambda/sigma stages are
    SKIPPED (smooth measured to RAISE d_seg +6.8%; the trainer has no such curriculum stages
    so the skip is STRUCTURAL, not a flag).

    Pose rides the stored Quantizr-style sidecar -> ``w_pose=0`` (the witness's sole
    controllable job is d_seg). DETERMINISTIC-REPRODUCIBLE: single recorded ``--seed``;
    per-stage + periodic checkpoints ON (PRESERVE clause); EMA-shadow saved; ``--resume-from``
    compatible. FROM-SCRATCH: ``resume_from=None`` (the structured-init IS the seed, not a ckpt).
    """
    epochs = ce_to + tau_window
    base = dict(BASELINE.base)
    base.update({
        "--w-pose": w_pose,
        "--tau-softplus-tau": tau,
        "--structured-init": True,
        "--structured-init-include-lane": True,
        "--lane-prior-phi1": True,
        "--lane-prior-phi1-mode": "replace",
        "--lane-prior-phi1-dash-gate": True,
        "--stage-transition-rewarmup-epochs": rewarmup_epochs,
        "--stage-transition-rewarmup-floor": rewarmup_floor,
        "--stage-transition-rewarmup-shape": "linear",
        "--stage-transition-reset-moments": True,
        "--seed": seed,
    })
    return WitnessProgram(
        out_dir=out_dir,
        gt_cache=gt_cache,
        epochs=epochs,
        num_pairs=num_pairs,
        temp=Anneal(1.0, 0.05),  # RENDER-partition sharpness anneal (NOT the seg-surrogate tau);
        stages=(                  # frozen at 0.05 by the Muon finisher (FEED-fm FIX-2).
            Stage("CE", None, None),
            Stage("tau_softplus", "--tau-softplus-start-epoch", ce_to),
            # l7 PARKED at epochs (no-op tail); engaged adaptively via warm-start continuation.
            Stage("l7_softplus", "--l7-start-epoch", epochs),
        ),
        regularizers=(
            Regularizer("--eikonal-weight", 0.01),
            Regularizer("--length-weight", 0.001),
        ),
        preserve=Preserve(stage_boundaries=True, ckpt_every=25),
        contain=Contain(),
        authority=Authority(),
        base=base,
        resume_from=None,  # FROM SCRATCH (structured-init seed, not a checkpoint)
        mlx_device=mlx_device,
    )
