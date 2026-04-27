"""Centralised eval_roundtrip enforcement gate.

CLAUDE.md non-negotiable: `eval_roundtrip` MUST default True everywhere.
The only escape hatch is the env var `TAC_ALLOW_NO_ROUNDTRIP=1`. Without
the round-trip, proxy-auth gap can grow 2-11x on PoseNet — every TTO /
training run without it is a wasted run.

History (the bug class this replaces):
  - Each script (~16 of them) re-implemented `_enforce_eval_roundtrip(args)`.
    The duplicated copies were sticky — they only printed the banner when
    `args.eval_roundtrip` was already False. If an operator left
    `TAC_ALLOW_NO_ROUNDTRIP=1` exported in their shell or tmux session,
    later programmatic / config-driven `eval_roundtrip=False` runs picked
    up the relaxed gate WITHOUT any per-run acknowledgement banner.
  - 2026-04-27 codex R5-4 #4 fix: this module centralises the helper and
    fixes the sticky-env-var bug.

Behaviour matrix:

  args.eval_roundtrip | TAC_ALLOW_NO_ROUNDTRIP | result
  --------------------+------------------------+-------------------------
  True                | unset                  | clean — no warning
  True                | set ('1')              | WARN: env var unused
  False               | unset                  | SystemExit (FATAL)
  False               | set ('1')              | WARN: roundtrip DISABLED
                                                  (proceeds with banner)

For provenance: callers can pass an `output_dir=Path(...)` and we write
the resolved roundtrip state + env-var presence into a sidecar JSON so
the run record carries the operator's decision.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_VAR = "TAC_ALLOW_NO_ROUNDTRIP"
SENTINEL_VALUE = "1"


@dataclass(frozen=True)
class RoundtripGateResult:
    """Resolved state of the eval_roundtrip gate after enforcement."""

    eval_roundtrip: bool          # final value (post-enforcement)
    env_var_present: bool         # was TAC_ALLOW_NO_ROUNDTRIP set?
    env_var_value: str | None     # raw value of the env var, if any
    proceeded_via_escape_hatch: bool  # True iff False+env-set combo

    def to_provenance_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict suitable for run_record / provenance."""
        return {
            "eval_roundtrip": self.eval_roundtrip,
            "env_var_name": ENV_VAR,
            "env_var_present": self.env_var_present,
            "env_var_value": self.env_var_value,
            "proceeded_via_escape_hatch": self.proceeded_via_escape_hatch,
        }


def _print_banner(msg: str) -> None:
    sep = "!" * 78
    print("\n" + sep + "\n" + msg + "\n" + sep + "\n",
          file=sys.stderr, flush=True)


def enforce_eval_roundtrip(
    args: Any | None = None,
    *,
    eval_roundtrip: bool | None = None,
    output_dir: str | Path | None = None,
    write_provenance: bool = True,
) -> RoundtripGateResult:
    """Enforce the eval_roundtrip non-negotiable.

    Resolves the desired `eval_roundtrip` value from either:
      - keyword `eval_roundtrip=...` (programmatic callers), OR
      - `args.eval_roundtrip` if `args` is provided (argparse callers).

    Always inspects the env var presence (codex R5-4 #4 fix: the previous
    helper only checked the env var when args.eval_roundtrip was already
    False, which made the env var sticky across runs).

    Behaviour:
      • eval_roundtrip=True, env unset → clean, return.
      • eval_roundtrip=True, env set    → WARN that env var is present
        but unused; remind the operator to unset it.
      • eval_roundtrip=False, env unset → SystemExit (FATAL).
      • eval_roundtrip=False, env set   → WARN with DANGER banner;
        proceed with the relaxed gate.

    Side effects:
      • Prints WARN banners to stderr.
      • If `output_dir` is given AND `write_provenance` is True, writes
        `<output_dir>/eval_roundtrip_gate.json` with the resolved state.
    """
    if eval_roundtrip is None:
        if args is None:
            raise ValueError(
                "enforce_eval_roundtrip requires either `eval_roundtrip=...` "
                "or `args` (with .eval_roundtrip attribute)."
            )
        eval_roundtrip = bool(getattr(args, "eval_roundtrip", True))

    env_value = os.environ.get(ENV_VAR)
    env_present = env_value is not None
    proceeded_via_hatch = False

    if not eval_roundtrip:
        if env_value != SENTINEL_VALUE:
            raise SystemExit(
                f"FATAL: eval_roundtrip is False but {ENV_VAR}={SENTINEL_VALUE} "
                f"is not set. Set the env var explicitly for diagnostic ablation."
            )
        proceeded_via_hatch = True
        _print_banner(
            f"DANGER: eval_roundtrip is DISABLED via {ENV_VAR}={SENTINEL_VALUE}.\n"
            f"  Proxy-auth gap will be 2-11x. Tag results "
            f"[no-roundtrip-ablation]."
        )
    else:
        if env_present:
            _print_banner(
                f"WARNING: {ENV_VAR}={env_value!r} is set in the environment but "
                f"eval_roundtrip is True for this run.\n"
                f"  The env var has NO effect here. If you intended to disable "
                f"eval_roundtrip, set --no-eval-roundtrip on the CLI (where "
                f"supported); otherwise unset {ENV_VAR} so future runs in this\n"
                f"  shell aren't accidentally relaxed."
            )

    result = RoundtripGateResult(
        eval_roundtrip=eval_roundtrip,
        env_var_present=env_present,
        env_var_value=env_value,
        proceeded_via_escape_hatch=proceeded_via_hatch,
    )

    if output_dir is not None and write_provenance:
        try:
            d = Path(output_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "eval_roundtrip_gate.json").write_text(
                json.dumps(result.to_provenance_dict(), indent=2),
            )
        except OSError as e:
            print(
                f"  warn: could not write eval_roundtrip_gate.json under "
                f"{output_dir}: {e}",
                file=sys.stderr,
            )

    return result
