# SPDX-License-Identifier: MIT
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
  - 2026-04-27 codex R5-r6 #3 fix: the helper used to drop the sidecar
    JSON when `output_dir` was None at call time. Live scripts called
    `enforce_eval_roundtrip(args, output_dir=getattr(args, 'output_dir',
    None))` BEFORE resolving the timestamped default output dir → no
    sidecar was written for the common default-output-dir invocation.
    Fix: `output_dir_callback` lets callers pass a thunk that materialises
    the path AFTER defaults are resolved; `write_sidecar_now()` triggers
    the deferred write.

Behaviour matrix:

  args.eval_roundtrip | TAC_ALLOW_NO_ROUNDTRIP | result
  --------------------+------------------------+-------------------------
  True                | unset                  | clean — no warning
  True                | set ('1')              | WARN: env var unused
  False               | unset                  | SystemExit (FATAL)
  False               | set ('1')              | WARN: roundtrip DISABLED
                                                  (proceeds with banner)

For provenance: callers can pass an `output_dir=Path(...)` AND/OR an
`output_dir_callback=lambda: ...`. If callback is given, the sidecar
write is deferred — call `write_sidecar_now()` on the result after the
script has resolved its default output dir. This lets the gate enforce
SystemExit at the TOP of main() (so a bad config dies before preflight
burns startup time) while still landing the sidecar at the resolved path.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ENV_VAR = "TAC_ALLOW_NO_ROUNDTRIP"
SENTINEL_VALUE = "1"


@dataclass
class RoundtripGateResult:
    """Resolved state of the eval_roundtrip gate after enforcement.

    NOT frozen: codex R5-r6 #3 added `write_sidecar_now()` which mutates
    `_sidecar_written` so callers can audit whether the deferred sidecar
    was ever materialised.
    """

    eval_roundtrip: bool          # final value (post-enforcement)
    env_var_present: bool         # was TAC_ALLOW_NO_ROUNDTRIP set?
    env_var_value: str | None     # raw value of the env var, if any
    proceeded_via_escape_hatch: bool  # True iff False+env-set combo
    _output_dir_callback: Callable[[], Path | str] | None = field(
        default=None, repr=False,
    )
    _sidecar_written: bool = field(default=False, repr=False)

    def to_provenance_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict suitable for run_record / provenance."""
        return {
            "eval_roundtrip": self.eval_roundtrip,
            "env_var_name": ENV_VAR,
            "env_var_present": self.env_var_present,
            "env_var_value": self.env_var_value,
            "proceeded_via_escape_hatch": self.proceeded_via_escape_hatch,
        }

    def write_sidecar_now(
        self, output_dir: str | Path | None = None,
    ) -> Path | None:
        """Materialise the deferred sidecar write.

        Used in conjunction with `enforce_eval_roundtrip(...,
        output_dir_callback=lambda: ...)` so the script can resolve its
        timestamped default output dir AFTER the gate has run.

        If `output_dir` is given, it overrides the callback (operator
        already knows the final path). Otherwise the callback is invoked
        to obtain the path. Returns the sidecar Path (None if no callback
        / no override AND none materialised).
        """
        path: Path | None = None
        if output_dir is not None:
            path = Path(output_dir)
        elif self._output_dir_callback is not None:
            try:
                path = Path(self._output_dir_callback())
            except Exception as e:
                print(
                    f"  warn: eval_roundtrip_gate sidecar deferred-callback raised "
                    f"{type(e).__name__}: {e}; sidecar not written.",
                    file=sys.stderr,
                )
                return None
        if path is None:
            return None
        try:
            path.mkdir(parents=True, exist_ok=True)
            sidecar = path / "eval_roundtrip_gate.json"
            sidecar.write_text(
                json.dumps(self.to_provenance_dict(), indent=2),
            )
            self._sidecar_written = True
            return sidecar
        except OSError as e:
            print(
                f"  warn: could not write eval_roundtrip_gate.json under "
                f"{path}: {e}",
                file=sys.stderr,
            )
            return None


def _print_banner(msg: str) -> None:
    sep = "!" * 78
    print("\n" + sep + "\n" + msg + "\n" + sep + "\n",
          file=sys.stderr, flush=True)


def enforce_eval_roundtrip(
    args: Any | None = None,
    *,
    eval_roundtrip: bool | None = None,
    output_dir: str | Path | None = None,
    output_dir_callback: Callable[[], Path | str] | None = None,
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
        `<output_dir>/eval_roundtrip_gate.json` immediately.
      • If `output_dir_callback` is given (and `output_dir` is not), the
        sidecar write is DEFERRED — the caller must invoke
        `result.write_sidecar_now()` once the script has resolved its
        default output dir. This is the codex R5-r6 #3 fix for the case
        where `args.output_dir is None` at gate time but later resolves
        to a timestamped path. (Both args silently no-op if
        `write_provenance=False`.)
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
        _output_dir_callback=output_dir_callback if write_provenance else None,
    )

    if write_provenance and output_dir is not None:
        try:
            d = Path(output_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "eval_roundtrip_gate.json").write_text(
                json.dumps(result.to_provenance_dict(), indent=2),
            )
            result._sidecar_written = True
        except OSError as e:
            print(
                f"  warn: could not write eval_roundtrip_gate.json under "
                f"{output_dir}: {e}",
                file=sys.stderr,
            )

    return result
