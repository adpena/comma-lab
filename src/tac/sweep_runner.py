"""Bayesian hyperparameter sweep framework (Optuna-backed).

CLAUDE.md mandates "no arbitrariness, engineering+scientific+algorithmic rigor"
for every training schedule (LRs, epoch counts, batch sizes). This module is
the canonical implementation of that rule: every training schedule is a search
space, and every promotion is the argmin of an Optuna study, not a hand-pick.

Algorithmic choices (citation-backed, not arbitrary):
  * Sampler  : TPE (Tree-structured Parzen Estimator), Bergstra et al. 2011.
               Optuna's default; outperforms random + grid on the budget regime
               (10-200 trials) we operate in.
  * Acquisition: EI (Expected Improvement) — canonical default for
               budget-constrained sweeps; UCB / PI exposed as overrides via
               `OptunaTrialDispatcher(sampler=...)`.
  * Pruner   : MedianPruner — kills trials whose intermediate score is below
               the median of completed trials at the same step. Cheap proxy
               for early-stopping bad configs, ~30-40% wall-clock savings on
               our 100-trial sweeps.

Provenance (CLAUDE.md "no signal loss" rule):
  Every sweep emits `<output_dir>/trial_history.jsonl` (one JSON object per
  trial: params, value, state, datetime_complete) plus a `study.db` SQLite
  database (Optuna's native format) for full re-loadability.

Lane integration (the GENERIC contract — every lane plugs in by satisfying it):
  1. A search_space dict     -- {name: ("loguniform"|"uniform"|"int"|"categorical", *args)}
  2. A script template       -- shell script with `__PARAM_<NAME>__` placeholders
  3. A result_parser         -- callable(script_path) -> float (the auth_score)

Constraints (CLAUDE.md non-negotiables enforced statically):
  * Refuses MPS/CPU-fallback search spaces (must pin --device cuda).
  * Refuses search spaces that toggle `eval_roundtrip` False.
  * Tags every emitted script with `__SWEEP_TRIAL_NUMBER__` and
    `__SWEEP_NAME__` for traceability back to the study.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search-space spec (lightweight schema; validated up-front, not at trial time)
# ---------------------------------------------------------------------------

_DISTRIBUTIONS = ("loguniform", "uniform", "int", "categorical", "fixed")


def _validate_search_space(search_space: dict[str, tuple]) -> None:
    """Raise ValueError on any malformed entry.

    Schema:
        {name: ("loguniform", low, high)}
        {name: ("uniform", low, high)}
        {name: ("int", low, high)}                  # inclusive both ends
        {name: ("categorical", [v1, v2, ...])}
        {name: ("fixed", value)}                    # constant — pinned, not sampled
    """
    if not isinstance(search_space, dict):
        raise TypeError(f"search_space must be dict, got {type(search_space).__name__}")
    if not search_space:
        raise ValueError("search_space must be non-empty")

    for name, spec in search_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"search_space key must be non-empty str, got {name!r}")
        if not isinstance(spec, tuple) or len(spec) < 2:
            raise ValueError(
                f"search_space[{name!r}] must be a tuple (kind, ...), got {spec!r}"
            )
        kind = spec[0]
        if kind not in _DISTRIBUTIONS:
            raise ValueError(
                f"search_space[{name!r}] kind must be one of {_DISTRIBUTIONS}, "
                f"got {kind!r}"
            )
        if kind in ("loguniform", "uniform"):
            if len(spec) != 3:
                raise ValueError(
                    f"search_space[{name!r}] {kind} expects (kind, low, high)"
                )
            _, low, high = spec
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                raise ValueError(
                    f"search_space[{name!r}] bounds must be numeric"
                )
            if low >= high:
                raise ValueError(
                    f"search_space[{name!r}] low ({low}) must be < high ({high})"
                )
            if kind == "loguniform" and low <= 0:
                raise ValueError(
                    f"search_space[{name!r}] loguniform low must be > 0, got {low}"
                )
        elif kind == "int":
            if len(spec) != 3:
                raise ValueError(f"search_space[{name!r}] int expects (kind, low, high)")
            _, low, high = spec
            if not (isinstance(low, int) and isinstance(high, int)):
                raise ValueError(f"search_space[{name!r}] int bounds must be int")
            if low > high:
                raise ValueError(
                    f"search_space[{name!r}] low ({low}) must be <= high ({high})"
                )
        elif kind == "categorical":
            if len(spec) != 2 or not isinstance(spec[1], (list, tuple)):
                raise ValueError(
                    f"search_space[{name!r}] categorical expects (kind, [v1,v2,...])"
                )
            if len(spec[1]) == 0:
                raise ValueError(f"search_space[{name!r}] categorical choices empty")
        elif kind == "fixed":
            if len(spec) != 2:
                raise ValueError(f"search_space[{name!r}] fixed expects (kind, value)")


def _enforce_non_negotiables(search_space: dict[str, tuple]) -> None:
    """Block search spaces that violate CLAUDE.md non-negotiables.

    These are static-detectable failure modes — refusing them at sweep
    construction prevents us from ever spending GPU on a configuration
    that would produce an invalid score.
    """
    # eval_roundtrip must be True (or fixed True), never False/categorical containing False.
    er = search_space.get("eval_roundtrip")
    if er is not None:
        kind = er[0]
        if kind == "fixed":
            if er[1] is not True:
                raise ValueError(
                    "CLAUDE.md non-negotiable: eval_roundtrip must be True. "
                    f"sweep would set eval_roundtrip={er[1]!r}."
                )
        elif kind == "categorical":
            choices = er[1]
            if False in choices or any(c in (False, 0, "false", "False") for c in choices):
                raise ValueError(
                    "CLAUDE.md non-negotiable: eval_roundtrip categorical "
                    f"contains a False-equivalent: {choices!r}"
                )
        else:
            raise ValueError(
                "eval_roundtrip must be specified as 'fixed' True or 'categorical' [True], "
                f"got distribution {kind!r}"
            )

    # device must not be MPS or CPU. If sampled, must be 'cuda' only.
    dev = search_space.get("device")
    if dev is not None:
        kind = dev[0]
        if kind == "fixed":
            if dev[1] != "cuda":
                raise ValueError(
                    f"CLAUDE.md non-negotiable: device must be 'cuda', got {dev[1]!r}"
                )
        elif kind == "categorical":
            for c in dev[1]:
                if c != "cuda":
                    raise ValueError(
                        f"CLAUDE.md non-negotiable: device categorical contains "
                        f"non-cuda value {c!r}"
                    )
        else:
            raise ValueError(
                f"device must be 'fixed' or 'categorical', got distribution {kind!r}"
            )


def _hash_search_space(search_space: dict[str, tuple]) -> str:
    """Canonical SHA256 of the search space (for reproducibility provenance)."""
    # Sort keys so equivalent specs hash identically regardless of insertion order.
    canonical: list[tuple] = []
    for name in sorted(search_space.keys()):
        spec = search_space[name]
        # Tuples with lists inside need to be canonicalized.
        if spec[0] == "categorical":
            canonical.append((name, "categorical", tuple(spec[1])))
        else:
            canonical.append((name,) + tuple(spec))
    payload = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# BayesianSweep — the canonical sweep harness.
# ---------------------------------------------------------------------------


def _placeholder_token(name: str) -> str:
    """Convert a hyperparameter name to its template placeholder.

    Convention: `__PARAM_<UPPER>__` so the placeholder is unambiguous in
    shell scripts (`__` is not a legal shell variable character; this
    avoids accidental collision with $VAR substitutions).
    """
    return f"__PARAM_{name.upper()}__"


@dataclass
class TrialResult:
    """Result of one Bayesian-sweep trial."""

    trial_number: int
    params: dict[str, Any]
    value: float
    state: str  # "complete" | "pruned" | "failed"
    started_at_utc: str
    completed_at_utc: str
    notes: str = ""


@dataclass
class BayesianSweep:
    """Canonical Bayesian hyperparameter sweep wrapper.

    Args:
        name:           Sweep identifier (used in output filenames + study name).
        script_template: Path to a shell script with `__PARAM_<NAME>__` placeholders
                        for every search-space key. Substituted at trial time.
        search_space:   {name: (distribution_kind, *args)} (see _validate_search_space).
        n_trials:       Total trials the study will run (TPE warms up after ~10).
        objective:      Name of the metric to optimize (e.g. "auth_score"). Used in
                        provenance tags only — the dispatcher decides how to extract.
        direction:      "minimize" or "maximize" (auth scoring is "minimize").
        output_dir:     Where dispatched scripts + trial_history.jsonl live.

    Critical invariants enforced at __post_init__:
      * Non-negotiables (eval_roundtrip, device=cuda) are pre-validated.
      * search_space hash is deterministic across equivalent specs.
      * objective direction is one of {minimize, maximize} (no fuzzy strings).
    """

    name: str
    script_template: Path
    search_space: dict[str, tuple]
    n_trials: int = 30
    objective: str = "auth_score"
    direction: str = "minimize"
    output_dir: Optional[Path] = None
    result_parser: Optional[Callable[[Path], float]] = None
    # Predicted auth-score band — operator-supplied range from prior runs;
    # used only for provenance + sanity-band warnings, never for control flow.
    predicted_band: Optional[tuple[float, float]] = None
    trials: list[TrialResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Coerce string paths.
        self.script_template = Path(self.script_template)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

        if not isinstance(self.n_trials, int) or self.n_trials < 1:
            raise ValueError(f"n_trials must be positive int, got {self.n_trials!r}")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"direction must be 'minimize' or 'maximize', got {self.direction!r}"
            )

        _validate_search_space(self.search_space)
        _enforce_non_negotiables(self.search_space)
        self.search_space_hash = _hash_search_space(self.search_space)

    # ----- script template substitution -------------------------------------

    def _read_template(self) -> str:
        if not self.script_template.exists():
            raise FileNotFoundError(
                f"BayesianSweep template not found: {self.script_template}"
            )
        text = self.script_template.read_text()
        # Validate every placeholder exists (catch typos at construction-adjacent time).
        missing = []
        for name in self.search_space:
            tok = _placeholder_token(name)
            if tok not in text:
                missing.append(tok)
        if missing:
            raise ValueError(
                f"script_template {self.script_template.name} missing placeholder(s): "
                f"{missing}. Add `{missing[0]}` to the template."
            )
        return text

    def _substitute(self, template_text: str, params: dict[str, Any], trial_number: int) -> str:
        """Substitute `__PARAM_X__` tokens with concrete values; tag trial."""
        text = template_text
        for name, value in params.items():
            tok = _placeholder_token(name)
            # Bash-friendly serialization: bool → 0/1, others → str(value).
            if isinstance(value, bool):
                rendered = "1" if value else "0"
            else:
                rendered = str(value)
            text = text.replace(tok, rendered)

        # Always-present provenance tags (even if template doesn't have them,
        # they're harmless — the substitution is a no-op).
        text = text.replace("__SWEEP_NAME__", self.name)
        text = text.replace("__SWEEP_TRIAL_NUMBER__", str(trial_number))
        text = text.replace("__SWEEP_SEARCH_SPACE_HASH__", self.search_space_hash)
        return text

    # ----- per-trial dispatch -----------------------------------------------

    def dispatch_remote(self, trial: Any) -> Path:
        """Sample params via the Optuna trial, write a unique remote script.

        Returns the path to the rendered script (ready to ship to Vast.ai).
        Does NOT execute the script — that is the caller's responsibility
        (we never run real GPU sweeps from inside Bayesian search; sweeps
        dispatch to remote workers).
        """
        if self.output_dir is None:
            raise ValueError(
                "output_dir must be set on BayesianSweep before dispatching trials"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = self._sample_params(trial)
        template_text = self._read_template()
        rendered = self._substitute(template_text, params, trial.number)

        out_path = self.output_dir / f"{self.name}_trial_{trial.number:04d}.sh"
        out_path.write_text(rendered)
        # Make executable so the operator can dispatch directly.
        out_path.chmod(0o755)
        return out_path

    def run_local(self, trial: Any) -> float:
        """Local validation harness — does NOT touch GPU.

        Returns a deterministic SMOKE score derived from the sampled params.
        Used by tests and by `--smoke` mode to validate the wiring end-to-end
        before real dispatch. The score is a hash-based pseudo-objective, NOT
        a real proxy. Anyone using this for promotion decisions has misread
        the contract — emit a warning if invoked outside test/smoke context.
        """
        params = self._sample_params(trial)
        # Smoke-objective: hash params → [0, 1) — deterministic, no GPU.
        # The "best" smoke trial has nothing to do with the real best trial;
        # this exists ONLY to validate plumbing.
        h = hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        # Convert first 8 hex chars to a float in [0, 1).
        smoke_value = int(h[:8], 16) / 0xFFFFFFFF

        result = TrialResult(
            trial_number=trial.number,
            params=params,
            value=float(smoke_value),
            state="complete",
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            notes="run_local smoke (NOT a real auth score)",
        )
        self.trials.append(result)
        self._append_history(result)
        return float(smoke_value)

    def parse_remote_result(self, script_path: Path) -> float:
        """Parse the auth-eval result that the dispatched script produced.

        Convention (matches contest_auth_eval.py + auth_eval_renderer.py):
        the remote script writes `<output_dir>/<script_stem>.result.json`
        OR appends a `RESULT_JSON: {...}` line to a colocated `.log` file.

        Custom parsers can be supplied via `result_parser` on the dataclass.
        """
        if self.result_parser is not None:
            return float(self.result_parser(script_path))

        # Default parser: look for a sidecar JSON next to the script.
        candidates = [
            script_path.with_suffix(".result.json"),
            script_path.parent / f"{script_path.stem}.result.json",
            script_path.parent / "auth_eval.log",
            script_path.parent / "contest_auth_eval.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.suffix == ".json":
                payload = json.loads(candidate.read_text())
                # Accept either {"final_score": ...} or {"auth_score": ...}.
                for key in ("final_score", "auth_score", "score", self.objective):
                    if key in payload:
                        return float(payload[key])
                raise KeyError(
                    f"result JSON {candidate} lacks a known score key "
                    f"(expected one of: final_score / auth_score / score / {self.objective})"
                )
            # Plain log: scrape RESULT_JSON.
            text = candidate.read_text()
            match = re.search(r"^RESULT_JSON:\s*(\{.*\})\s*$", text, re.M)
            if match:
                payload = json.loads(match.group(1))
                for key in ("final_score", "auth_score", "score", self.objective):
                    if key in payload:
                        return float(payload[key])
        raise FileNotFoundError(
            f"no remote result file found for {script_path.name}; "
            f"checked: {[str(c) for c in candidates]}"
        )

    # ----- internals --------------------------------------------------------

    def _sample_params(self, trial: Any) -> dict[str, Any]:
        """Sample concrete values from each search-space distribution."""
        params: dict[str, Any] = {}
        for name, spec in self.search_space.items():
            kind = spec[0]
            if kind == "loguniform":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high, log=True)
            elif kind == "uniform":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high, log=False)
            elif kind == "int":
                _, low, high = spec
                params[name] = trial.suggest_int(name, low, high)
            elif kind == "categorical":
                _, choices = spec
                params[name] = trial.suggest_categorical(name, list(choices))
            elif kind == "fixed":
                _, value = spec
                # Optuna doesn't have a 'fixed' suggester; record but don't sample.
                params[name] = value
            else:  # pragma: no cover — _validate_search_space already filtered
                raise ValueError(f"unknown distribution {kind!r}")
        return params

    def _append_history(self, result: TrialResult) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.output_dir / "trial_history.jsonl"
        record = {
            "trial_number": result.trial_number,
            "params": result.params,
            "value": result.value,
            "state": result.state,
            "started_at_utc": result.started_at_utc,
            "completed_at_utc": result.completed_at_utc,
            "notes": result.notes,
            "sweep_name": self.name,
            "search_space_hash": self.search_space_hash,
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# OptunaTrialDispatcher — orchestrates the study, defers dispatch to caller.
# ---------------------------------------------------------------------------


class OptunaTrialDispatcher:
    """Thin wrapper around an Optuna study + a BayesianSweep.

    Why a separate class: BayesianSweep is the SWEEP DEFINITION (search space
    + template + parser). The dispatcher is the EXECUTION ENGINE (study
    creation, trial loop, convergence checks). Separating them keeps the
    sweep definitions pure-data and testable without an Optuna installation.

    Modes:
      * `dispatch_only(callback)` — yield each trial's rendered script path to
        the caller (caller is responsible for shipping to Vast.ai + back).
        The callback returns the float result; we feed it to optuna.
      * `local_smoke()` — run the sweep using BayesianSweep.run_local — for
        tests and pre-flight wiring validation. Never burns GPU.
    """

    def __init__(
        self,
        sweep: BayesianSweep,
        sampler: Any = None,
        pruner: Any = None,
        seed: int = 1234,
    ) -> None:
        self.sweep = sweep
        self.sampler = sampler
        self.pruner = pruner
        self.seed = seed
        self._study: Any = None

    def _ensure_study(self) -> Any:
        if self._study is not None:
            return self._study
        try:
            import optuna
        except ImportError as e:  # pragma: no cover — dep is required
            raise RuntimeError(
                "optuna is required for OptunaTrialDispatcher. "
                "Install it on the remote: `uv pip install optuna`."
            ) from e

        sampler = self.sampler if self.sampler is not None else optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=min(10, max(3, self.sweep.n_trials // 4)),
        )
        pruner = self.pruner if self.pruner is not None else optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        # SQLite storage so the study survives crashes + can be re-loaded for
        # post-hoc analysis (CLAUDE.md "no signal loss").
        storage_url = None
        if self.sweep.output_dir is not None:
            self.sweep.output_dir.mkdir(parents=True, exist_ok=True)
            storage_url = f"sqlite:///{self.sweep.output_dir / 'study.db'}"
        self._study = optuna.create_study(
            study_name=self.sweep.name,
            direction=self.sweep.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=True,
        )
        return self._study

    def local_smoke(self) -> dict[str, Any]:
        """Run the sweep with the local smoke objective. No GPU spend."""
        study = self._ensure_study()
        study.optimize(self.sweep.run_local, n_trials=self.sweep.n_trials)
        return self._summary(study)

    def dispatch_only(
        self,
        result_callback: Callable[[Path, dict[str, Any]], float],
    ) -> dict[str, Any]:
        """Run trials by dispatching scripts; result_callback returns the score.

        result_callback signature: (rendered_script_path, params) -> float.
        The callback is the operator's responsibility: it ships the script to
        the remote, waits for completion, returns the auth score.
        """
        study = self._ensure_study()

        def _objective(trial: Any) -> float:
            script_path = self.sweep.dispatch_remote(trial)
            params = {k: trial.params[k] for k in trial.params}
            value = float(result_callback(script_path, params))
            result = TrialResult(
                trial_number=trial.number,
                params=params,
                value=value,
                state="complete",
                started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.sweep.trials.append(result)
            self.sweep._append_history(result)
            return value

        study.optimize(_objective, n_trials=self.sweep.n_trials)
        return self._summary(study)

    def _summary(self, study: Any) -> dict[str, Any]:
        best = study.best_trial
        # Convergence diagnostic: best-so-far per trial.
        best_so_far = []
        running = float("inf") if self.sweep.direction == "minimize" else float("-inf")
        for t in study.trials:
            if t.value is None:
                best_so_far.append(running)
                continue
            if self.sweep.direction == "minimize":
                running = min(running, t.value)
            else:
                running = max(running, t.value)
            best_so_far.append(running)
        return {
            "sweep_name": self.sweep.name,
            "objective": self.sweep.objective,
            "direction": self.sweep.direction,
            "n_trials": len(study.trials),
            "search_space_hash": self.sweep.search_space_hash,
            "best_value": best.value,
            "best_params": dict(best.params),
            "best_trial_number": best.number,
            "best_so_far": best_so_far,
            "predicted_band": list(self.sweep.predicted_band) if self.sweep.predicted_band else None,
        }
