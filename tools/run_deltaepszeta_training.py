"""δεζ-aware training driver — Phase 3 keystone.

Per Grand Council bilevel-optimization deliberation
(``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``),
**Phase 3** of the bilevel trajectory targets score 0.165 by training a
substrate whose native description length is lower under the chosen codec.
This driver wires:

- :func:`tac.shannon_h2_loss.shannon_h0_loss` — differentiable rate proxy.
- ``tools.build_deltaepszeta_training_targets`` JSON — per-tensor weights.
- :class:`tac.codec_pipeline_deltaepszeta_callback.CodecPipelineAwareTrainingCallback`
  — end-of-epoch ground-truth archive-bytes signal from the canonical
  :class:`tac.codec_pipeline.CodecPipeline`.
- :func:`tools.run_bilevel_optimization.detect_substrates` — substrate
  auto-detection (called via the shared helper module).

The training loop:

1. Loads a state_dict from disk and constructs an ``nn.ParameterDict``
   over its tensors (or the operator-supplied subset).
2. At each step, computes:
   - distortion proxy: caller-supplied callable (default = MSE vs the
     reference state_dict, suitable for a sanity-loop).
   - rate proxy: weighted ``shannon_h0_loss`` over per-tensor weights.
   - combined Lagrangian:  ``loss = distortion + λ * rate``.
3. Per-epoch:
   - dual-ascent on λ when rate proxy exceeds ``rate_budget``.
   - invokes the CodecPipelineAwareTrainingCallback for the empirical
     archive-bytes signal.
4. Writes a JSONL log + final state_dict checkpoint.

CLAUDE.md compliance:
    - Strict-scorer-rule: NO scorer load. Distortion is operator-supplied
      (default = MSE-vs-reference). For Phase 3 sanity-loop a simple MSE is
      fine; real δεζ runs supply a perceptual surrogate.
    - No /tmp paths: log_dir under ``experiments/results/<lane_id>/...``.
    - No score claims: this driver reports loss / rate / bytes only.
    - Pure CPU (PyTorch CPU tensors) by default; torch will auto-device only
      if the operator threads `device=` explicitly. A future GPU dispatch path
      must be added separately and gated by the canonical mandatory-CUDA
      wrapper in :mod:`tac.device_helpers` (NOT in this file).

Cross-references:
    - Council memo: ``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``
    - Differentiable rate term: :mod:`tac.shannon_h2_loss`
    - Pipeline-aware callback: :mod:`tac.codec_pipeline_deltaepszeta_callback`
    - Per-tensor weight builder: :mod:`tools.build_deltaepszeta_training_targets`
    - Lane registry id: ``lane_run_deltaepszeta_training`` (phase 4
      operator infrastructure)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from datetime import UTC, datetime
from typing import Callable, Iterable

import torch

# Make `tac` importable from tools/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tac.codec_pipeline import CodecPipeline  # noqa: E402
from tac.codec_pipeline_deltaepszeta_callback import (  # noqa: E402
    CodecPipelineAwareTrainingCallback,
)
from tac.shannon_h2_loss import shannon_h0_loss  # noqa: E402


# ---------------------------------------------------------------------------
# Driver dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DeltaEpsZetaTrainingConfig:
    """Operator-supplied configuration.

    Attributes:
        n_epochs: total epochs to train (each epoch = ``steps_per_epoch`` steps).
        steps_per_epoch: optimizer steps per epoch.
        learning_rate: SGD step size on the substrate parameters.
        lambda_init: initial Lagrange multiplier on the rate term.
        lambda_step: dual-ascent step size when the rate proxy exceeds
            ``rate_budget_bits``.
        rate_budget_bits: per-symbol H0 ceiling. The dual-ascent loop ramps λ
            up while the measured H0 exceeds this; ramps down when below.
            (Lower-bound is 0.0; multiplier never goes negative.)
        per_tensor_weights: optional dict ``{tensor_name -> weight}`` that
            scales the rate term per tensor. Sum should normally be 1.0.
        rate_n_bits: alphabet size for the soft-histogram rate proxy.
        rate_temperature: softmax temperature for the rate proxy.
        log_dir: directory for the JSONL log and final checkpoint. Must NOT
            be under /tmp.
        run_label: human-readable label for the JSONL log filename.
        seed: torch / numpy seed for reproducibility.
    """
    n_epochs: int = 1
    steps_per_epoch: int = 4
    learning_rate: float = 1e-3
    lambda_init: float = 1e-3
    lambda_step: float = 1e-3
    rate_budget_bits: float = 7.0
    per_tensor_weights: dict[str, float] | None = None
    rate_n_bits: int = 8
    rate_temperature: float = 1.0
    log_dir: pathlib.Path = dataclasses.field(
        default_factory=lambda: pathlib.Path("experiments/results/lane_run_deltaepszeta_training/run")
    )
    run_label: str = "deltaepszeta"
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class StepReport:
    """One JSONL row written per training step."""
    epoch: int
    step: int
    distortion: float
    rate_bits: float
    lambda_value: float
    loss: float
    timestamp_utc: str

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Reference distortion: MSE-vs-reference (sanity-loop default)
# ---------------------------------------------------------------------------

def make_mse_distortion_fn(
    reference_state_dict: dict[str, torch.Tensor],
    *,
    parameter_keys: Iterable[str] | None = None,
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """Build a distortion callable that returns weighted MSE against a frozen
    reference state_dict.

    The closure captures the reference tensors detached. This is the default
    distortion for the Phase 3 sanity-loop — a real δεζ run supplies a
    perceptual surrogate (e.g., logits MSE on a held-out feature map) but the
    pattern is identical: take a state_dict, return a scalar tensor.
    """
    keys = list(parameter_keys) if parameter_keys is not None else list(
        reference_state_dict.keys()
    )
    ref = {k: reference_state_dict[k].detach().clone() for k in keys}

    def distortion_fn(state: dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0)
        for k in keys:
            if k not in state:
                continue
            d = state[k] - ref[k]
            total = total + (d * d).mean()
        return total

    return distortion_fn


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class DeltaEpsZetaTrainingDriver:
    """δεζ-aware training driver.

    Composition:
        - state_dict (operator-supplied, on disk) -> ``nn.ParameterDict``.
        - distortion_fn(state_dict) -> scalar loss term.
        - rate_term = sum_i w_i * shannon_h0_loss(state[i]).
        - loss = distortion + lambda * rate_term.
        - λ ramps via dual-ascent against ``rate_budget_bits``.
        - end-of-epoch: pipeline-aware callback fires for empirical archive
          bytes; result appended to JSONL log.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        config: DeltaEpsZetaTrainingConfig,
        *,
        distortion_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
        callback: CodecPipelineAwareTrainingCallback | None = None,
    ) -> None:
        # CLAUDE.md /tmp guard.
        log_dir_str = str(config.log_dir)
        if log_dir_str.startswith("/tmp/") or log_dir_str == "/tmp":
            raise ValueError(
                f"log_dir must not be under /tmp (transient-evidence trap); "
                f"got {log_dir_str!r}. Use experiments/results/<lane_id>/... "
                "or .omx/state/ instead."
            )
        if config.n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0, got {config.n_epochs}")
        if config.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0, got {config.steps_per_epoch}"
            )
        if config.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {config.learning_rate}"
            )
        if config.lambda_init < 0:
            raise ValueError(
                f"lambda_init must be >= 0, got {config.lambda_init}"
            )

        torch.manual_seed(config.seed)

        self._config = config
        self._reference = {k: v.detach().clone() for k, v in state_dict.items()}
        # Build trainable parameters as float32 leaf tensors.
        self._params: dict[str, torch.nn.Parameter] = {
            k: torch.nn.Parameter(v.detach().clone().to(torch.float32))
            for k, v in state_dict.items()
        }

        # Default distortion = MSE-vs-reference (sanity-loop).
        if distortion_fn is None:
            distortion_fn = make_mse_distortion_fn(self._reference)
        self._distortion_fn = distortion_fn

        self._lambda = float(config.lambda_init)
        self._optimizer = torch.optim.SGD(
            list(self._params.values()), lr=config.learning_rate
        )

        # Log dir.
        self._log_dir = pathlib.Path(config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{config.run_label}_step_log.jsonl"

        # Pipeline-aware callback (optional; tests construct without one).
        self._callback = callback

    # ---------------------------------------------------------------
    # API
    # ---------------------------------------------------------------

    @property
    def lambda_value(self) -> float:
        return self._lambda

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: p.detach().clone() for k, p in self._params.items()}

    @property
    def log_path(self) -> pathlib.Path:
        return self._log_path

    def _rate_term(self) -> torch.Tensor:
        """Per-tensor weighted H0 rate proxy."""
        weights = self._config.per_tensor_weights or {}
        if not weights:
            # Uniform: 1/N over all params.
            n = max(1, len(self._params))
            weights = {k: 1.0 / n for k in self._params}
        total = torch.tensor(0.0)
        for name, p in self._params.items():
            w = float(weights.get(name, 0.0))
            if w == 0.0:
                continue
            h0 = shannon_h0_loss(
                p,
                n_bits=self._config.rate_n_bits,
                temperature=self._config.rate_temperature,
            )
            total = total + w * h0
        return total

    def step_once(self) -> StepReport:
        """Run one optimizer step; return the StepReport for that step."""
        self._optimizer.zero_grad(set_to_none=True)
        state = {k: p for k, p in self._params.items()}
        distortion = self._distortion_fn(state)
        rate = self._rate_term()
        loss = distortion + self._lambda * rate
        loss.backward()
        self._optimizer.step()
        report = StepReport(
            epoch=-1,
            step=-1,
            distortion=float(distortion.detach().item()),
            rate_bits=float(rate.detach().item()),
            lambda_value=float(self._lambda),
            loss=float(loss.detach().item()),
            timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return report

    def _ramp_lambda(self, observed_rate_bits: float) -> None:
        """Dual-ascent: ramp λ when rate exceeds budget, ramp down otherwise.
        λ is clamped at 0 (no negative multipliers).
        """
        budget = self._config.rate_budget_bits
        step = self._config.lambda_step
        if observed_rate_bits > budget:
            self._lambda = max(0.0, self._lambda + step)
        else:
            # Slowly relax to avoid oscillation.
            self._lambda = max(0.0, self._lambda - 0.5 * step)

    def _append_log(self, report: StepReport) -> None:
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            )

    def train(self) -> list[StepReport]:
        """Full training loop. Returns the list of StepReport rows written."""
        all_rows: list[StepReport] = []
        for epoch in range(self._config.n_epochs):
            last_rate = 0.0
            for step in range(self._config.steps_per_epoch):
                base = self.step_once()
                stamped = dataclasses.replace(base, epoch=epoch, step=step)
                all_rows.append(stamped)
                self._append_log(stamped)
                last_rate = stamped.rate_bits
            # End-of-epoch: dual-ascent on λ + callback fires.
            self._ramp_lambda(last_rate)
            if self._callback is not None:
                self._callback.report(
                    self.state_dict,
                    epoch=epoch,
                    notes=f"deltaepszeta epoch={epoch}",
                )
        return all_rows

    def save_checkpoint(self, path: pathlib.Path | str) -> pathlib.Path:
        """Bit-faithfully save the current state_dict to disk."""
        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, out)
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="δεζ-aware training driver (Phase 3 keystone)",
    )
    p.add_argument(
        "--state-dict",
        required=True,
        help="path to state_dict .pt file",
    )
    p.add_argument(
        "--targets-json",
        default=None,
        help="optional path to build_deltaepszeta_training_targets JSON; "
        "per-tensor weights are read from per_tensor[*]['name'] + "
        "per_tensor[*]['loss_weight_normalized']",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=1,
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=4,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--lambda-init",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--lambda-step",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--rate-budget-bits",
        type=float,
        default=7.0,
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="default: experiments/results/lane_run_deltaepszeta_training_<UTC>/run",
    )
    p.add_argument(
        "--run-label",
        default="deltaepszeta",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = p.parse_args(argv)

    state_dict_path = pathlib.Path(args.state_dict)
    if not state_dict_path.exists():
        print(f"FATAL: state_dict not found: {state_dict_path}", file=sys.stderr)
        return 2
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    weights: dict[str, float] | None = None
    if args.targets_json:
        targets_path = pathlib.Path(args.targets_json)
        if not targets_path.exists():
            print(f"FATAL: targets JSON not found: {targets_path}", file=sys.stderr)
            return 2
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        weights = {
            row["name"]: float(row.get("loss_weight_normalized", 0.0))
            for row in targets.get("per_tensor", [])
        }

    log_dir = pathlib.Path(args.log_dir) if args.log_dir else pathlib.Path(
        f"experiments/results/lane_run_deltaepszeta_training_{_utc_timestamp()}/run"
    )
    cfg = DeltaEpsZetaTrainingConfig(
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        lambda_init=args.lambda_init,
        lambda_step=args.lambda_step,
        rate_budget_bits=args.rate_budget_bits,
        per_tensor_weights=weights,
        log_dir=log_dir,
        run_label=args.run_label,
        seed=args.seed,
    )
    driver = DeltaEpsZetaTrainingDriver(state_dict=state_dict, config=cfg)
    rows = driver.train()
    ckpt_path = driver.save_checkpoint(log_dir / "final_state_dict.pt")
    print(f"trained {len(rows)} steps")
    print(f"final λ:        {driver.lambda_value:.6g}")
    print(f"log:            {driver.log_path}")
    print(f"checkpoint:     {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
