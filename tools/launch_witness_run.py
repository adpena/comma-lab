#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical ONE-COMMAND launcher for a level-set witness run — the automated
value-generator for launch + observability (operator 2026-06-30: "That needs to
happen automatically in the future" + the guiding principle "build the automated
value-generator, not ad-hoc").

It folds the whole error-prone hand-assembly into ONE reusable, flag-validated,
no-silent-failure path:

  (a) DERIVE the config from the clip's GT cache (tac.witness_autoconfig) and
      FLAG-VALIDATE every emitted flag against the trainer's REAL argparse
      (never-invent-a-flag — NO-FAKE);
  (b) WRITE the command into ``<out_dir>/launch.sh`` (a SCRIPT — so the daemon
      cmd is a clean ``["bash", launch.sh]`` argv with NO word-split fragility,
      the exact class of bug that collapsed a whole command into argv[0]);
  (c) LAUNCH durably via tools/spawn_durable_daemon.py, which now AUTO-VERIFIES
      the child survived exec (dead launch -> nonzero + detailed debug);
  (d) VERIFY the perf-env line ``custom_grouped_backward active=true`` is in the
      log (the ~17x fast path; an unset env is a silent slow-run footgun);
  (e) CONFIRM the dashboard is up — once up it AUTO-TRACKS this (and every future)
      run every refresh tick, so no manual repoint/reload is ever needed for a
      new run (only NEW dashboard CODE needs a zero-downtime reload).

Determinism/resumability/observability are preserved: the emitted command carries
``--seed``, ``--ckpt-every``, ``--stage-checkpoints`` (resumable per-stage), and
the run is rendered live by the dashboard.

means != ends: this LAUNCHES an advisory [macOS-MLX] run. Only a byte-closed exact
n600 row < 0.19110 moves the pointer. Use ``--dry-run`` to emit + validate + write
launch.sh WITHOUT spawning (CPU-only, GPU-free, safe).

Usage:
    .venv/bin/python tools/launch_witness_run.py \\
        --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \\
        --num-pairs 600 --epochs 1000              # real launch
    .venv/bin/python tools/launch_witness_run.py ... --dry-run   # emit+validate only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

from tac import witness_autoconfig as wac  # noqa: E402
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    TRAINER_PATH,
)
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    real_boolean_flags as _cdsl_real_boolean_flags,
)
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    real_store_true_flags as _cdsl_real_store_true_flags,
)
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    real_trainer_flags as _cdsl_real_trainer_flags,
)
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    schedule_epoch_budget_violations as _schedule_epoch_budget_violations,
)

_TRAINER = TRAINER_PATH  # canonical single-source: curriculum_dsl.TRAINER_PATH


def _composable_lever_names() -> tuple[str, ...]:
    """The --dsl-lever help-text enumeration, derived from the DSL's own composability
    predicate (tac.witness_dsl.lever_registry.name_composable_levers) — never a hand-typed
    list (the hand-typed list is how Muon/DM1Minimal ended up advertised-but-crashing).
    A registry failure degrades the HELP TEXT only (loud placeholder, not a launcher crash);
    actual --dsl-lever resolution still refuses via the typed error in main()."""
    try:
        from tac.witness_dsl.lever_registry import name_composable_levers
        return name_composable_levers()
    except Exception as exc:  # degraded help, loud — composition itself still fail-closes
        return (f"<composable-lever list unavailable: {type(exc).__name__}: {exc}>",)


def _admission_override_ok(text: str | None) -> bool:
    """Reject empty / placeholder admission-override rationales (per Catalog #287 discipline)."""
    if not text or not text.strip():
        return False
    low = text.strip().lower()
    return low not in {"<rationale>", "<reason>", "placeholder", "tbd", "todo", "n/a"} and len(low) >= 8


# ───────────── safe-frac policy (operator memory policy 2026-07-04; review XC-ii/L5) ────────────
SAFE_FRAC_SINGLE_WORKLOAD = 0.85   # sole-workload: no artificial ceiling (>=10 GiB fail-safe floor
#                                    + ~10 GiB margin => 0.85 is the physics-derived safe fraction)
SAFE_FRAC_CONCURRENT = 0.70        # only under admitted concurrency (coexistence headroom), and the
#                                    CONSERVATIVE fallback when the governor state is unreadable


HEAVY_MIN_PROJECTED_GIB = 4.0      # a governed row is a HEAVY workload iff its recorded projection
#                                    is at least this (telemetry daemons record no/near-zero peaks)


def _governed_active_jobs() -> list[dict]:
    """READ-ONLY view of the governor's admitted/running HEAVY jobs: the durable-daemon registry
    rows with ``status == "running"`` AND a live pid (stale rows for dead pids are dropped) AND a
    heavy-workload signature — a recorded ``projected_peak_gib >= HEAVY_MIN_PROJECTED_GIB`` (the
    governed admission path), or, when no projection was recorded, a cmd matching the governor's
    own OUR_JOBS_PATTERN heavy vocabulary. Control-plane/telemetry daemons (memory_blackbox,
    dashboards) record neither and must NOT pin the box at 0.70 after the real run stops.
    Consumes ONLY the governor's read helpers — never its action surface, never mutates state."""
    import system_memory_governor as _gov  # tools/ is on sys.path (same dir as this launcher)

    jobs: list[dict] = []
    for r in _gov._running_registry_jobs(_gov._load_registry_rows()):
        try:
            pid = int(r.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)              # liveness probe only — signal 0 sends nothing
        except ProcessLookupError:
            continue                      # dead pid => stale registry row, not a live workload
        except PermissionError:
            pass                          # alive but not ours => still a live workload
        proj = r.get("projected_peak_gib")
        if proj is not None:
            try:
                heavy = float(proj) >= HEAVY_MIN_PROJECTED_GIB
            except (TypeError, ValueError):
                heavy = True              # malformed projection => count it (conservative)
        else:
            cmd = r.get("cmd")
            cmd_str = " ".join(str(t) for t in cmd) if isinstance(cmd, list) else str(cmd or "")
            heavy = _gov.matches_our_jobs(cmd_str)
        if not heavy:
            continue
        jobs.append({"label": str(r.get("label", "")) or f"pid{pid}", "pid": pid})
    return jobs


def derive_safe_frac(explicit: float | None) -> tuple[float, str, str]:
    """Policy-aware ``--mem-preflight-safe-frac`` (operator memory policy 2026-07-04, memory
    ``operator_memory_policy_sole_workload_no_artificial_ceiling_20260704``): 0.85 when NO other
    governed heavy job is admitted/running (sole-workload — no artificial ceiling), 0.70 only
    under admitted concurrency. An EXPLICIT CLI value always wins; unreadable governor state
    falls back CONSERVATIVE to 0.70. Returns ``(safe_frac, branch, why)`` for observability."""
    if explicit is not None:
        return (float(explicit), "explicit",
                "CLI --mem-preflight-safe-frac overrides the policy derivation")
    try:
        jobs = _governed_active_jobs()
    except Exception as exc:  # read-only consumption must never crash the launcher
        return (SAFE_FRAC_CONCURRENT, "fallback_conservative",
                f"governor state unreadable ({type(exc).__name__}: {exc}) -> conservative "
                f"{SAFE_FRAC_CONCURRENT:.2f}")
    if jobs:
        names = ", ".join(sorted(j["label"] for j in jobs)[:4])
        return (SAFE_FRAC_CONCURRENT, "concurrent",
                f"{len(jobs)} governed heavy job(s) admitted/running ({names}) -> "
                f"{SAFE_FRAC_CONCURRENT:.2f} (coexistence headroom)")
    return (SAFE_FRAC_SINGLE_WORKLOAD, "single_workload",
            f"no other governed heavy job admitted/running -> {SAFE_FRAC_SINGLE_WORKLOAD:.2f} "
            f"(sole-workload policy 2026-07-04: >=10 GiB fail-safe floor + ~10 GiB margin)")


# ───────────────────────── never-invent-a-flag guard ─────────────────────────
def real_trainer_flags() -> frozenset[str]:
    """The SET of real ``--flag`` names parsed from the trainer's argparse, INCLUDING the
    ``--no-<flag>`` negation forms argparse auto-generates for ``BooleanOptionalAction`` flags.

    (CLASS-fix 2026-07-07, islands-arm launch): the DSL merge deliberately renders a ``False``
    override as ``--no-<flag>`` (``witness_autoconfig._merge_dsl_levers`` mirroring
    ``curriculum_dsl.WitnessProgram.compile_trainer_argv``), so a lever that turns a
    BooleanOptionalAction base flag OFF (e.g. ``Mod32SegOnlyControlBase`` negating
    ``--lane-prior-phi1``) emits a REAL argparse token this validator previously mis-refused as
    invented. Only BooleanOptionalAction flags gain the negation; ``store_true`` flags do NOT
    (their ``--no-`` form would be a genuine invention — the DSL's C2 guard refuses those
    upstream, and this validator still refuses them here)."""
    # Regex-scanning is single-sourced in curriculum_dsl (never-invent-flags canonical).
    # BooleanOptionalAction-only = boolean flags MINUS store_true flags (the negation-eligible
    # set; store_true flags get NO ``--no-`` form per the C2 guard).
    flags = set(_cdsl_real_trainer_flags(TRAINER_PATH))
    bool_opt = _cdsl_real_boolean_flags(TRAINER_PATH) - _cdsl_real_store_true_flags(TRAINER_PATH)
    flags.update(f.replace("--", "--no-", 1) for f in bool_opt)
    return frozenset(flags)


def validate_emitted_flags(cfg, out_dir: str) -> tuple[bool, list[tuple[str, bool]]]:
    """Validate every emitted flag against the real argparse. Returns
    ``(all_pass, [(flag, ok), ...])``."""
    real = real_trainer_flags()
    results = [(flag, flag in real) for flag, _ in cfg.to_trainer_flags(out_dir)]
    return all(ok for _, ok in results), results


# ───────────────────────── extra-trainer-flags passthrough (C5, SEAL review 2026-07-04) ─────────
def parse_extra_trainer_flags(text: str | None) -> tuple[list[str], list[str]]:
    """Shell-split an ``--extra-trainer-flags`` string and validate every ``--flag`` token against
    the trainer's REAL argparse (:func:`real_trainer_flags` — the same never-invent-a-flag guard the
    derived config goes through). Returns ``(tokens, invented_flags)``; empty/None -> ``([], [])``.
    """
    import shlex

    if not text or not text.strip():
        return [], []
    toks = shlex.split(text)
    real = real_trainer_flags()
    invented = [t for t in toks if t.startswith("--") and t.split("=", 1)[0] not in real]
    return toks, invented


# ───────────────────────── emit-side confound fixes (confound_hunt_synthesis_20260705.md) ───────
# The launcher composes the derived-config argv (cfg.to_trainer_flags) with the passthrough
# --extra-trainer-flags into ONE launch.sh command. These pure helpers make that composed argv
# CLEAN for the next launch: no duplicate long-flags (C13), palliative resume flags coupled to a
# weights-only warm-start (C8), --seed-anneal-epochs relative to the resume epoch (C16), and the
# per-group grad-clip opted in (C4). Trainer BEHAVIOR is unchanged — only what the launcher EMITS.
_PALLIATIVE_RESUME_FLAGS = ("--resume-clear-spike-guard", "--resume-allow-lever-drift")
_SEED_ANNEAL_WINDOW_DEFAULT = 200
_EPOCH_TOKEN_RE = re.compile(r"ep(?:och)?[_-]?(\d+)")


def _extra_flag_names(extra_flags: list[str]) -> list[str]:
    """The long-flag NAMES (``--foo`` from a bare ``--foo`` or ``--foo=bar`` token) in a token list."""
    return [t.split("=", 1)[0] for t in extra_flags if t.startswith("--")]


def duplicate_long_flags(flag_names: list[str]) -> list[str]:
    """The distinct long-flag names that appear MORE THAN ONCE (order-preserving), else ``[]``."""
    seen: set[str] = set()
    dups: list[str] = []
    for f in flag_names:
        if f in seen and f not in dups:
            dups.append(f)
        seen.add(f)
    return dups


def _flag_value(tokens: list[str], flag: str) -> str | None:
    """The value of ``--flag value`` or ``--flag=value`` in a token list (first hit), else ``None``."""
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def _replace_flag_value(tokens: list[str], flag: str, new_value: str) -> list[str]:
    """Return a copy of ``tokens`` with every ``--flag <v>`` / ``--flag=<v>`` value set to
    ``new_value`` (space or ``=`` form preserved)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == flag and i + 1 < len(tokens):
            out += [t, str(new_value)]
            i += 2
            continue
        if t.startswith(flag + "="):
            out.append(f"{flag}={new_value}")
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def inject_per_group_grad_clip(extra_flags: list[str], config_flag_names: list[str],
                               *, enable: bool = True) -> tuple[list[str], str | None]:
    """Fix 4 (confound C4): opt IN ``--per-group-grad-clip`` for witness launches. The trainer
    default is OFF (for byte-identity), so the volatile eikonal gradient can hijack the SHARED
    grad-clip budget and starve the seg step (measured seg-step throttle ~300-900x). The launcher
    opts in unless the user already set either polarity (``--per-group-grad-clip`` /
    ``--no-per-group-grad-clip``) or ``enable=False``. Returns ``(extra_flags, note|None)``."""
    if not enable:
        return extra_flags, None
    names = set(_extra_flag_names(extra_flags)) | set(config_flag_names)
    if "--per-group-grad-clip" in names or "--no-per-group-grad-clip" in names:
        return extra_flags, None
    return [*extra_flags, "--per-group-grad-clip"], (
        "Fix 4 (C4): injected --per-group-grad-clip (trainer default OFF for byte-identity) so the "
        "eikonal + seg gradients are clipped per-group — the volatile eikonal term can no longer "
        "hijack the shared clip budget and starve the seg step. Pass --no-per-group-grad-clip to opt out.")


def couple_palliative_warm_start(extra_flags: list[str],
                                 config_flag_names: list[str]) -> tuple[list[str], str | None]:
    """Fix 2 (confound C8): a resume that carries a PALLIATIVE flag
    (``--resume-clear-spike-guard`` / ``--resume-allow-lever-drift``) but restores optimizer state
    keeps the poison (stale ep-N moments restored into a drifted loss geometry) while using only
    warm-start's cosmetic side-effects. Make the safe coupling structural: if a palliative flag is
    emitted without ``--warm-start-weights-only``, inject it (the trainer then auto-resolves
    ``--resume-model-from`` to ``ema``). Returns ``(extra_flags, note|None)``."""
    names = set(_extra_flag_names(extra_flags)) | set(config_flag_names)
    if not any(f in names for f in _PALLIATIVE_RESUME_FLAGS):
        return extra_flags, None
    if "--warm-start-weights-only" in names:
        return extra_flags, None
    return [*extra_flags, "--warm-start-weights-only"], (
        "Fix 2 (C8): a palliative resume flag (--resume-clear-spike-guard / "
        "--resume-allow-lever-drift) is emitted without --warm-start-weights-only; injected it so "
        "the trainer auto-resolves --resume-model-from ema and does NOT restore stale ep-N optimizer "
        "moments into the drifted loss geometry.")


def _resume_start_epoch(extra_flags: list[str]) -> int | None:
    """Best-effort parse of the resume epoch E from ``--resume-from <ckpt>`` — the epoch the run
    RESUMES at, which is ``ckpt_epoch + 1`` (a ckpt saved after completing epoch N continues at
    N+1; the confound anchor: an ``ep100`` ckpt resumes at start_epoch 101). Extracts the LAST
    ``ep<NNN>`` / ``epoch_<NNN>`` integer from the ckpt basename (then the whole path) and adds 1.
    Returns ``None`` if absent/unparseable. Conservative: if the true off-by-one differs, firing one
    epoch early only extends the seed window (a safe direction)."""
    val = _flag_value(extra_flags, "--resume-from")
    if not val:
        return None
    matches = _EPOCH_TOKEN_RE.findall(Path(val).name) or _EPOCH_TOKEN_RE.findall(val)
    return int(matches[-1]) + 1 if matches else None


def seed_anneal_relative_to_resume(extra_flags: list[str], config_flag_names: list[str],
                                   *, anneal_window: int = _SEED_ANNEAL_WINDOW_DEFAULT
                                   ) -> tuple[list[str], str | None]:
    """Fix 3 (confound C16): when the config seeds islands (``--seed-islands``) AND resumes at
    epoch E (``--resume-from``) with a seed-anneal window ``--seed-anneal-epochs N``, N must be
    RELATIVE to E. An absolute ``N <= E`` withdraws the seed crutch BEFORE the resumed run even
    begins (the confound: seed-anneal-epochs 101 with resume start_epoch 101 -> off 899/900 epochs).
    Auto-correct ``N`` to ``E + anneal_window`` and log. Returns ``(extra_flags, note|None)``."""
    names = set(_extra_flag_names(extra_flags)) | set(config_flag_names)
    if "--seed-islands" not in names:
        return extra_flags, None
    n_str = _flag_value(extra_flags, "--seed-anneal-epochs")
    if n_str is None:
        return extra_flags, None  # no anneal window emitted -> nothing to make relative
    try:
        n = int(n_str)
    except ValueError:
        return extra_flags, None
    start = _resume_start_epoch(extra_flags)
    if start is None or n > start:
        return extra_flags, None  # fresh run, or the window already extends past the resume epoch
    corrected = start + int(anneal_window)
    return _replace_flag_value(extra_flags, "--seed-anneal-epochs", str(corrected)), (
        f"Fix 3 (C16): --seed-anneal-epochs {n} <= resume start_epoch {start} would withdraw the "
        f"island seed before the resumed run begins; corrected to {corrected} (E {start} + "
        f"{anneal_window} anneal window) so the seed crutch persists past resume start.")


def apply_emit_side_confound_fixes(extra_flags: list[str], config_flag_names: list[str],
                                   *, per_group_grad_clip: bool = True
                                   ) -> tuple[list[str], list[str], list[str]]:
    """Apply the four emit-side confound fixes to the passthrough ``extra_flags`` given the derived
    config's flag names, then C13-CHECK the FINAL combined argv for duplicate long-flags. Returns
    ``(extra_flags, notes, duplicate_long_flags)``; a non-empty duplicate list means the caller
    MUST refuse the launch (argparse last-wins silently shifts schedules)."""
    notes: list[str] = []
    for fn in (
        lambda ef: inject_per_group_grad_clip(ef, config_flag_names, enable=per_group_grad_clip),
        lambda ef: couple_palliative_warm_start(ef, config_flag_names),
        lambda ef: seed_anneal_relative_to_resume(ef, config_flag_names),
    ):
        extra_flags, note = fn(extra_flags)
        if note:
            notes.append(note)
    dups = duplicate_long_flags(config_flag_names + _extra_flag_names(extra_flags))
    return extra_flags, notes, dups


# ───────────────────────── launch.sh (no word-split fragility) ─────────────────────────
def config_family(cfg) -> str:
    """The canonical named-config FAMILY this cfg renders, derived from the cfg's own
    selector fields (factual — never a guess). Stamped into the launch.sh RUN-IDENTITY
    header so run-identity consumers (dashboard) can cite it as evidence."""
    # crucible_v7 is a DSL TypedWitnessConfig (name field), not a witness_autoconfig dataclass —
    # detect it by its declared name so the run-identity header does not MISLABEL it proven_base.
    if getattr(cfg, "name", "") in ("crucible_v7", "crucible_v752", "crucible_v753",
                                    "v9_cgauge", "v9_cgauge_432",
                                    "v9_cgauge_truly_optimal_core",
                                    "v9_cgauge_ideal_mod19", "v9_cgauge_ideal_mod19_sR",
                                    "v9_cgauge_ideal_mod32",
                                    "v9_cgauge_432_taper_off",
                                    "v9_cgauge_432_horizon_iso",
                                    "v9_cgauge_432_step_iso",
                                    "next_launch_all_levers_20260713",
                                    "next_launch_all_levers_trimmed_20260713",
                                    "throughput_component_timer_async_20260713",
                                    "throughput_component_timer_solo_20260713"):
        return cfg.name
    if getattr(cfg, "crucible_v6", False):
        return "crucible_v6"
    if getattr(cfg, "fresh_seeded", False):
        return "fresh_seeded"
    if getattr(cfg, "sealed_205", False):
        return ("store_nothing_205"
                if getattr(cfg, "pose_carrier_source", "real_keyframe") == "generated"
                else "sealed_205")
    if getattr(cfg, "all_levers", False):
        return "all_levers"
    return "proven_base"


def _identity_header(cfg, *, dsl_compile_hash: str | None = None) -> str:
    """Machine-readable RUN-IDENTITY header lines for launch.sh (the run dir's config
    record; operator 2026-07-07 run-identity row). ``# tac-config-family:`` is always
    stamped (factual, derived from the cfg itself); ``# tac-run-purpose:`` only when a
    purpose was DECLARED (--purpose -> WitnessConfig.purpose) — dashboards render that
    verbatim with provenance "declared". Comment lines only: every launch.sh consumer
    (trainer argv extraction, flag parsers, memory preflight, DSL schedule read-back,
    the sealed_205 argv byte-identity gate) skips ``#`` lines, so the header is
    provenance-neutral to training + argv byte-identity."""
    lines = [f"# tac-config-family: {config_family(cfg)}\n"]
    if dsl_compile_hash:
        lines.append(f"# dsl_compile_hash: {dsl_compile_hash}\n")
    purpose = getattr(cfg, "purpose", None)
    if purpose:
        lines.append(f"# tac-run-purpose: {' '.join(str(purpose).split())}\n")
    return "".join(lines)


def _readiness_deferral_header(cfg) -> str:
    """Render typed config-owned readiness deferrals as score-neutral comments.

    A held named config must survive every deterministic ``launch.sh`` regeneration;
    requiring an operator to repeat already-authored causal-isolation reasons on the
    CLI makes config freshness depend on shell history.  The values come only from
    the typed launch manifest and never become trainer arguments.
    """

    manifest = dict(getattr(cfg, "dsl_program_manifest", None) or {})
    deferrals = dict(manifest.get("readiness_deferrals", {}) or {})
    lines: list[str] = []
    for rung, reason in sorted(deferrals.items()):
        rung_text = str(rung).strip()
        reason_text = " ".join(str(reason).split())
        if not rung_text or len(reason_text) < 8 or "\n" in rung_text:
            raise ValueError(
                f"invalid typed readiness deferral {rung!r}={reason!r}; "
                "need a rung and substantive single-line reason"
            )
        lines.append(f"# LAUNCH_READINESS_DEFER:{rung_text}={reason_text}\n")
    return "".join(lines)


def build_launch_sh(cfg, out_dir: str, repo_root: Path | None = None,
                    extra_flags: list[str] | None = None,
                    dsl_compile_hash: str | None = None) -> str:
    """Render the launch.sh body. The trainer command goes into a SCRIPT so the
    daemon cmd is ``bash launch.sh`` (2 clean tokens) — never a space-bearing
    single argv[0]. Includes the perf-env prefix (TAC_MLX_CUSTOM_GROUPED_BACKWARD=1)
    via cfg.to_command(perf_env=True). ``extra_flags`` (already flag-validated
    tokens from --extra-trainer-flags) are appended as a trailing continuation
    line, so the memory preflight parses them off the SAME launch.sh it gates."""
    import shlex

    repo = str(repo_root or _REPO)
    cmd = cfg.to_command(out_dir, perf_env=True)
    if extra_flags:
        cmd += " \\\n  " + " ".join(shlex.quote(t) for t in extra_flags)
    dsl_admission_exports = ""
    if dsl_compile_hash:
        dsl_admission_exports = (
            f"export TAC_DSL_COMPILE_HASH={dsl_compile_hash}\n"
            f"export TAC_DSL_PROVENANCE_PATH="
            f"{shlex.quote(str(Path(out_dir) / 'dsl_provenance.json'))}\n"
            f"export TAC_DSL_LAUNCH_SH_PATH="
            f"{shlex.quote(str(Path(out_dir) / 'launch.sh'))}\n"
        )
    return (
        # cross-platform-by-default (operator 2026-07-07): env-resolved bash finds
        # a modern bash on PATH; macOS pins /bin/bash to 3.2. The durable-daemon
        # invokes this as `bash launch.sh` (shebang unused there), but a bare
        # `./launch.sh` on any fleet node must resolve portably.
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        # RUN-IDENTITY header (family always; purpose only when declared) — comments only.
        f"{_identity_header(cfg, dsl_compile_hash=dsl_compile_hash)}"
        # Config-freshness readiness deferrals are typed manifest metadata, not trainer flags.
        # Render them on every regeneration so held treatment-isolation decisions cannot vanish.
        f"{_readiness_deferral_header(cfg)}"
        f"cd {repo}\n"
        # Trainer memory telemetry (mem_probe rows). The #205 run was SILENT because this env-gated
        # default-off flag was never set in launch.sh (memory mine 2026-07-04 §1/§6) — the launcher
        # now sets it so every future run feeds the projection ledger's reconcile path. Telemetry
        # only: the gate emits mem_probe log rows; it never touches training numerics.
        "export TAC_MEM_PROBE=1\n"
        # (#254) launch.sh is a GOVERNED artifact (only the launcher emits it), so stamp the
        # admission marker so a manual `bash launch.sh` resume passes the trainer's admission
        # guard. Raw `python train_...py ...` that never went through a governed path lacks this
        # and is refused when enforce is armed (tac.admission_guard.GOVERNED_MARKER_ENV).
        "export TAC_GOVERNED_ADMISSION=1\n"
        # The trainer-side admission guard independently re-opens and recomputes
        # this exact binding.  A marker without these three values has no authority.
        f"{dsl_admission_exports}"
        f"{cmd}\n"
    )


def write_launch_sh(cfg, out_dir: Path, repo_root: Path | None = None,
                    extra_flags: list[str] | None = None,
                    dsl_compile_hash: str | None = None) -> Path:
    """Write launch.sh ATOMICALLY (tmp + os.replace, same dir → new inode).

    NEVER ``write_text`` in place: bash reads scripts incrementally from an open
    fd, so truncating the SAME inode under a live run shifts the byte offset and
    bash executes an orphaned continuation line as a command when the long
    trainer command returns. Empirical anchor: mod32cap 20260706T115554Z —
    launch.sh was regenerated ~5.5h into the live run; at trainer exit bash
    resumed mid-file and died on ``line 60: --ckpt-every: command not found``.
    ``os.replace`` gives every rewrite a fresh inode; a running bash keeps its
    fd on the old bytes and finishes cleanly (hardening sweep 2026-07-08)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    launch = out_dir / "launch.sh"
    body = build_launch_sh(
        cfg,
        str(out_dir),
        repo_root,
        extra_flags=extra_flags,
        dsl_compile_hash=dsl_compile_hash,
    )
    tmp = out_dir / f".launch.sh.tmp.{os.getpid()}"
    tmp.write_text(body)
    tmp.chmod(0o755)
    os.replace(tmp, launch)
    return launch


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write a launch-admission JSON artifact atomically in its run directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_dsl_compile_artifacts(
    cfg,
    launch_sh: Path,
    provenance_document: dict,
) -> tuple[Path, Path]:
    """Write and recompute the #405 DSL admission unit beside ``launch.sh``.

    The compile token itself excludes run identity, while this run manifest binds
    the exact artifact bytes the launcher/governor will consume.  Neither file is
    trusted on its own: :func:`verify_dsl_provenance_artifacts` re-hashes both and
    round-trips the shell argv back to the DSL compile.
    """

    from tac.v9_provenance_gates import (
        DSL_LAUNCH_MANIFEST_SCHEMA,
        extract_trainer_argv_from_launch_sh,
        verify_dsl_provenance_artifacts,
    )
    from tac.witness_run_artifacts import DSL_PROVENANCE_JSON, LAUNCH_MANIFEST_JSON

    launch_sh = Path(launch_sh)
    provenance_path = launch_sh.with_name(DSL_PROVENANCE_JSON)
    manifest_path = launch_sh.with_name(LAUNCH_MANIFEST_JSON)
    _write_json_atomic(provenance_path, provenance_document)
    launch_bytes = launch_sh.read_bytes()
    provenance_bytes = provenance_path.read_bytes()
    exact_argv = extract_trainer_argv_from_launch_sh(launch_bytes.decode("utf-8"))
    manifest = {
        "schema": DSL_LAUNCH_MANIFEST_SCHEMA,
        "config_family": config_family(cfg),
        "spec_id": provenance_document["spec_id"],
        "dsl_compile_hash": provenance_document["dsl_compile_hash"],
        "launch_sh": launch_sh.name,
        "launch_sh_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "dsl_provenance": provenance_path.name,
        "dsl_provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "resolved_launch_argv": list(exact_argv),
        "non_authoritative_context": {"written_at_utc": _utc()},
    }
    _write_json_atomic(manifest_path, manifest)
    ok, detail = verify_dsl_provenance_artifacts(
        launch_sh,
        provenance_path=provenance_path,
        launch_manifest_path=manifest_path,
        expected_hash=str(provenance_document["dsl_compile_hash"]),
    )
    if not ok:
        raise RuntimeError(detail)
    return provenance_path, manifest_path


def compile_dsl_document_for_config(
    cfg,
    out_dir: Path | str,
    *,
    program_name: str | None = None,
) -> tuple[dict, dict, str]:
    """Compile and validate the one DSL authority document for a launcher config."""

    from tac.v9_provenance_gates import build_dsl_compile_provenance_document
    from tac.witness_dsl.typed_config import verify_launch_manifest as _verify_dsl_manifest

    typed = getattr(cfg, "typed", None)
    if typed is None and hasattr(cfg, "to_program") and hasattr(cfg, "model_dump"):
        typed = cfg
    if typed is None:
        raise RuntimeError("launch config has no TypedWitnessConfig/WitnessProgram compile custody")
    dsl_manifest = dict(getattr(cfg, "dsl_program_manifest", {}) or {})
    dsl_ok, dsl_detail = _verify_dsl_manifest(
        dsl_manifest, list(_emitted_flag_names(cfg, str(out_dir)))
    )
    if not dsl_ok:
        raise RuntimeError(dsl_detail)
    if dsl_manifest.get("typed_config_hash") != typed.typed_config_hash():
        raise RuntimeError("typed config hash differs from its carried DSL program manifest")
    document = build_dsl_compile_provenance_document(
        program_name=str(program_name or getattr(typed, "name", config_family(cfg))),
        typed_config=typed,
        compiler_manifest=dict(getattr(cfg, "constants_manifest", {}) or {}),
        repo_root=_REPO,
    )
    return document, dsl_manifest, dsl_detail


def write_dsl_bound_launch(
    cfg,
    out_dir: Path,
    *,
    program_name: str | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Compile, emit, persist, reopen, and recompute an internal launch unit."""

    document, _, _ = compile_dsl_document_for_config(
        cfg, out_dir, program_name=program_name
    )
    launch_sh = write_launch_sh(
        cfg,
        out_dir,
        dsl_compile_hash=str(document["dsl_compile_hash"]),
    )
    provenance_path, manifest_path = write_dsl_compile_artifacts(
        cfg, launch_sh, document
    )
    return launch_sh, provenance_path, manifest_path, document


def with_internal_dsl_lever(cfg, *, name: str, overrides: dict[str, object]):
    """Compose a launcher-owned bounded-smoke delta through a typed DSL Lever."""

    from tac.witness_dsl.typed_config import TypedLever

    typed = getattr(cfg, "typed", None)
    rebind = getattr(cfg, "_rebind_typed", None)
    if typed is None or not callable(rebind):
        raise RuntimeError(
            "internal witness launch delta requires a typed launch adapter with _rebind_typed"
        )
    lever = TypedLever(
        name=name,
        overrides=dict(overrides),
        notes="launcher-owned bounded smoke/resume custody; Catalog #406 DSL-authored",
    )
    rebound = typed.model_copy(update={"levers": (*typed.levers, lever)})
    return rebind(rebound)


def write_constants_manifest(cfg, out_dir: Path) -> Path | None:
    """(#351 LawRef migration) Write ``constants_manifest.json`` beside launch.sh when the config
    carries LawRef-compiled constants (``cfg.constants_manifest`` — currently the crucible_v6
    CONSUMED trio + LR-hold). Provenance-only: every value is BIT-IDENTICAL to the sealed literal it
    replaces (value-identity is the law), so the manifest documents WHERE each launch constant came
    from (equation_id + typed inputs + artifact shas + ladder class) WITHOUT changing the emitted
    launch.sh. Returns the path written, or ``None`` when the config has no compiled constants (every
    non-crucible path — no file is created, so those runs are byte-and-file-identical to before).
    Written atomically (tmp + os.replace)."""
    manifest = dict(getattr(cfg, "constants_manifest", {}) or {})
    if not manifest:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "constants_manifest.v1",
        "config_family": config_family(cfg),
        "generated_at": _utc(),
        "note": ("LawRef-compiled launch constants (#351). Each value is BIT-IDENTICAL to the sealed "
                 "literal it replaces (VALUE-IDENTITY IS THE LAW); this manifest carries the "
                 "equation_id + typed inputs + artifact sha256 + value-provenance ladder class per "
                 "constant. Provenance-only — never a trainer flag; the emitted launch.sh is unchanged."),
        "constants": manifest,
    }
    path = out_dir / "constants_manifest.json"
    tmp = out_dir / f".constants_manifest.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)
    return path


# ───────────────────────── perf-env verification ─────────────────────────
def verify_perf_env(run_log: Path, timeout_s: float = 30.0, poll_s: float = 1.0) -> tuple[str, str | None]:
    """Wait (bounded) for the trainer's ``{"stage": "custom_grouped_backward", ...}``
    line and report whether the ~17x fast path is active. Returns
    ``("active"|"inactive"|"not_seen", raw_line|None)``. An unset env -> "inactive"
    (a silent slow-run footgun the launch gate must catch)."""
    deadline = time.time() + timeout_s
    run_log = Path(run_log)
    while time.time() < deadline:
        try:
            text = run_log.read_text(errors="replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if '"stage": "custom_grouped_backward"' in line or '"stage":"custom_grouped_backward"' in line:
                try:
                    active = bool(json.loads(line.strip()).get("active"))
                except Exception:
                    active = '"active": true' in line or '"active":true' in line
                return ("active" if active else "inactive", line.strip())
        time.sleep(poll_s)
    return ("not_seen", None)


# ───────────────────────── dashboard ensure ─────────────────────────
def _healthz(port: int, timeout: float = 3.0) -> int | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as r:
            return r.getcode()
    except Exception:
        return None


def ensure_dashboard(port: int) -> bool:
    """Confirm the dashboard is serving. Once up it AUTO-TRACKS this run (the
    resolver re-resolves the newest run every refresh tick) — no manual repoint.
    Returns True iff healthz==200; otherwise prints an actionable command (does NOT
    auto-start, to stay contained)."""
    code = _healthz(port)
    if code == 200:
        print(f"[launch-witness] dashboard :{port} is UP (healthz 200) — it AUTO-TRACKS "
              f"this run every refresh tick (no manual repoint/reload needed).")
        return True
    # NOTE: no --tau/--l7 in the hint — the dashboard DERIVES stage boundaries from
    # the run's own config via the DSL schedule read-back; hardcoded hint constants
    # (the old "--tau 300 --l7 600") were the exact mislabel class that fix extincted.
    print(f"[launch-witness] WARNING: dashboard :{port} NOT serving (healthz={code}). "
          f"This run is launched + durable regardless; to observe it, start/reload the "
          f"dashboard:\n    .venv/bin/python tools/dashboard_reload.py --port {port}",
          file=sys.stderr)
    return False


# ───────────────────────── throughput gate (compute pass) ─────────────────────────
def _emitted_flag_names(cfg, out_dir: str) -> set[str]:
    return {flag for flag, _ in cfg.to_trainer_flags(out_dir)}


def _config_wall_clock_budget_days(cfg) -> float | None:
    """Read a config's DECLARED wall-clock budget (days) as a float, or None if undeclared.
    Handles a Provenanced wrapper (TypedWitnessConfig.wall_clock_budget_days) via ``.value`` and a
    bare numeric (a future WitnessConfig field). Pure."""
    raw = getattr(cfg, "wall_clock_budget_days", None)
    if raw is None:
        return None
    val = getattr(raw, "value", raw)  # Provenanced -> .value; numeric -> itself
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def resolve_wall_clock_budget(accept_days: float | None, declared_days: float | None,
                              epochs: int) -> tuple[float | None, str, bool]:
    """Resolve the effective wall-clock budget for the launch (DEFAULT-ON, pure, unit-testable).

    Priority: (1) operator ``--accept-wall-clock`` override (stamps the run dir) > (2) the config's
    DECLARED budget > (3) a launcher-DERIVED fallback from the throughput anchor x epochs x slack (so
    a legacy config that never declared a budget STILL gets a default-on refuse — the gate never
    silently disappears). Returns ``(budget_days | None, source_label, is_operator_override)``;
    None only when epochs<=0 and nothing was supplied/declared (the gate then stays silent)."""
    if accept_days is not None:
        return float(accept_days), "operator --accept-wall-clock (override, stamped)", True
    if declared_days is not None:
        return float(declared_days), "config-declared (typed DERIVED budget)", False
    if epochs and epochs > 0:
        from tac.local_acceleration.scorer_throughput_gate import derive_wall_clock_budget_days
        return (derive_wall_clock_budget_days(int(epochs)),
                "launcher-derived fallback (anchor min/ep x epochs x slack; config declared none)",
                False)
    return None, "unavailable (no budget; epochs unknown)", False


def dsl_config_gate_action(
    *, ok: bool, detail: str, manifest_absent: bool, config: str,
    dry_run: bool, skip: bool, enforce: bool, allow_rationale: str | None,
) -> tuple[str, str]:
    """Pure decision for the DSL-authored-config gate (unit-testable, no I/O).

    Catalog #406 retires every migration/override branch.  The legacy arguments
    remain in the signature so older callers receive a deterministic refusal
    instead of an argparse/import crash, but none can authorize a hand-ruled run.
    """
    if ok:
        return "ok", detail
    requested_bypass = bool(skip or allow_rationale)
    suffix = " Retired bypass option supplied; it has no authority." if requested_bypass else ""
    return "refuse", (
        f"Catalog #406 DSL compile binding REFUSED for config {config!r}: {detail}. "
        "A typed WitnessProgram compile, recomputable dsl_compile_hash, and exact argv "
        f"round-trip are mandatory in dry-run and real-launch modes.{suffix}"
    )


def _run_throughput_gate(cfg, out_dir, *, threshold_ms: float | None,
                         accept_wall_clock_days: float | None = None) -> int:
    """Pre-spawn SegNet fwd+bwd throughput assertion (the ~17x fast-path gate) + the
    conditional --compile-step bit-identical requirement + the L45 DEFAULT-ON WALL-CLOCK GATE.
    Returns 0 (proceed) / nonzero (REFUSE). NEVER blocks on unavailability (measured-slow only)."""
    try:
        from tac.local_acceleration.scorer_throughput_gate import (
            ABS_THRESHOLD_MS,
            evaluate_throughput,
            project_launch_wall_clock,
        )
    except Exception as exc:  # helper import failure must not block the launch
        print(f"[launch-witness] WARNING: throughput gate unavailable (import: {exc}); "
              f"proceeding (perf-env log check still runs post-spawn).", file=sys.stderr)
        return 0
    thr = float(threshold_ms) if threshold_ms is not None else ABS_THRESHOLD_MS
    print(f"# throughput gate: measuring SegNet fwd+bwd (B=8, custom fast path) — REFUSE if median "
          f"> {thr:.0f}ms (measured ON~396 / OFF~6713)")
    verdict = evaluate_throughput(abs_threshold_ms=thr)
    if verdict.status == "fast":
        print(f"[launch-witness] throughput OK: SegNet fwd+bwd {verdict.segnet_fwd_bwd_ms:.1f}ms "
              f"<= {thr:.0f}ms (custom-grouped-backward fast path ACTIVE).")
    elif verdict.status == "unavailable":
        print(f"[launch-witness] WARNING: throughput gate could not measure "
              f"({verdict.reason}); proceeding (perf-env log check still runs post-spawn).",
              file=sys.stderr)
    else:  # slow
        print(f"[launch-witness] ERROR: REFUSING to launch — {verdict.reason}", file=sys.stderr)
        return 3
    # sub-part 3: --compile-step (or any --compile* flag) requires bit-identical compiled step.
    compile_flags = {f for f in _emitted_flag_names(cfg, str(out_dir)) if f.startswith("--compile")}
    if compile_flags:
        print(f"[launch-witness] NOTE: {sorted(compile_flags)} emitted — the trainer MUST assert "
              f"assert_compile_bit_identical at construction (compiled step == uncompiled + "
              f"deterministic). See tac.local_acceleration.scorer_throughput_gate."
              f"assert_compile_step_bit_identical.")
    # sub-part 4 (L45): DEFAULT-ON WALL-CLOCK GATE. PROJECT total wall-clock from the measured SegNet
    # ms + the run-1 anchor and REFUSE against the DECLARED budget (config typed field) — no opt-in
    # flag needed (operator 2026-07-08 "default on always"). Budget resolution: --accept-wall-clock
    # override (stamps) > config-declared > launcher-derived anchor fallback (so a legacy non-declaring
    # config STILL gets a refuse). This also couples throughput to budget (fix #3): a bench that passes
    # the 700ms absolute gate but is slower than the budget-implied ceiling STILL REFUSES here (catches
    # a non-env perf regression — kernel not loading / wrong device / thermal — even with the env set).
    # Projection is off a same-class (B=1-accum) MEASURED anchor, NOT a fresh per-ep measurement.
    epochs = int(getattr(cfg, "epochs", 0) or 0)
    budget, budget_src, is_override = resolve_wall_clock_budget(
        accept_wall_clock_days, _config_wall_clock_budget_days(cfg), epochs)
    if is_override and budget is not None:
        try:
            stamp = Path(out_dir) / "wall_clock_accept.txt"
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(
                "OPERATOR WALL-CLOCK ACCEPT (--accept-wall-clock)\n"
                f"accepted_budget_days: {budget}\n"
                "reason: operator knowingly accepted a wall-clock budget overriding the config-derived "
                "ceiling (never silent; L45 default-on gate).\n")
            print(f"[launch-witness] WALL-CLOCK ACCEPT (operator): budget {budget:.2f} days — stamped {stamp}.",
                  file=sys.stderr)
        except Exception:  # stamp is best-effort provenance; never block on it
            print(f"[launch-witness] WALL-CLOCK ACCEPT (operator): budget {budget:.2f} days.", file=sys.stderr)
    proj = project_launch_wall_clock(verdict.segnet_fwd_bwd_ms, epochs, budget_days=budget)
    if proj is not None and epochs > 0:
        print(f"[launch-witness] wall-clock gate (L45, default-on): {proj.detail} [budget: {budget_src}]")
        if proj.over_budget:
            print(f"[launch-witness] ERROR: REFUSING to launch — {proj.detail} [budget: {budget_src}]. "
                  f"This machine's measured SegNet bench projects a run OVER the declared/derived "
                  f"budget (a slower-than-anchor machine or a non-env perf regression). Land a compute "
                  f"lever / reduce --epochs / free the machine, or pass --accept-wall-clock <days> to "
                  f"knowingly accept a longer run.", file=sys.stderr)
            return 8
    return 0


# ───────────────────────── named-config derivation (shared by launch + calibration) ─────────────
def _derive_named_config_unchecked(config: str, gt_cache: str, *, num_pairs: int,
                                   epochs: int | None, overfit: bool):
    """Resolve a canonical named config to a derived trainer config at the given scale. The
    RSS-calibration smoke reuses this with a SMALL num_pairs/epochs but the SAME config name, so
    the calibration exercises the REAL flag set (not a toy variant).

    ``epochs=None`` => the config family's OWN sealed default applies (the kwarg is simply not
    passed, so each ``derive_*``/``compile_*`` signature default — the single source of truth —
    wins). An explicit int OVERRIDES the sealed default (NEW-1 fix: the launcher's old hardcoded
    default=1000 silently trampled crucible_v6/v7's sealed 3000)."""
    _ek: dict = {} if epochs is None else {"epochs": int(epochs)}
    if config == "sealed_205":
        # The #205 P3 SEALED capstone config fixes its own knobs (mod-dim 32 etc.); overfit N/A.
        return wac.derive_sealed_205_config(gt_cache, num_pairs=num_pairs, **_ek)
    if config == "store_nothing_205":
        # The sealed capstone + STORE-NOTHING pose-carrier source (Track B) — the A/B pose arm.
        return wac.derive_store_nothing_205_config(gt_cache, num_pairs=num_pairs, **_ek)
    if config == "fresh_seeded":
        # The 2026-07-04 SEAL-review REVISED run-1 argv (sealed_205 + seed/control deltas; C5).
        return wac.derive_fresh_seeded_config(gt_cache, num_pairs=num_pairs, **_ek)
    if config == "crucible_v6":
        # T5 CRUCIBLE v6.2 launch candidate (seal-round-2 BLOCKER-1 fix): store_nothing_205 +
        # ABSOLUTE schedule pins (tau@300 / anneal-den 3000 x hold 0.2 = descent 600 / Muon 726)
        # + tau_end 0.31 + fused-R + the v6 §1.1 DSL levers; pose block inherited (MAJOR-A2/#314).
        return wac.derive_crucible_v6_config(gt_cache, num_pairs=num_pairs, **_ek)
    if config == "crucible_v7":
        # T5 CRUCIBLE v7 restart — the FIRST requirement-V-native config, authored AS a
        # TypedWitnessConfig (DSL-emitted argv). compile_* produces the typed cfg + BOTH provenance
        # manifests (constants + DSL-program) + the governance DICT; .to_launch_config() wraps them
        # into the ONE object that satisfies the WHOLE launcher cfg protocol — the emit adapters AND
        # the gate-chain manifests (seal v7 r1 BLOCKER #1 + MAJOR #2). num_pairs/epochs flow through;
        # overfit is N/A (v7 fixes its own knobs, inherited from the sealed v6 substrate).
        return wac.compile_crucible_v7_config(
            gt_cache, num_pairs=num_pairs, **_ek).to_launch_config()
    if config == "crucible_v752":
        # T5 CRUCIBLE-2 v7.5.2 launch-1 SELF-ORIENT-OFF (owed-16 P9 RESOLVED-REFUTING; operator GO
        # 2026-07-10 #385 ADDENDUM v2). The deliberately-deferred P8-wall launcher wire-in, NOW
        # authorized. self_orient=False is the GO'd amendment (the realized −48% transfer measured ≈0,
        # 47 GiB RAM tax removed). compile_crucible_v752_launch_config returns the SAME launcher-facing
        # cfg protocol as crucible_v7 (.to_command / .to_trainer_flags / .name / dsl_program_manifest /
        # constants_manifest / schedule_governance), with the v7-identical constants+governance reused.
        # amber=True (OI-5 realization; operator elevation 2026-07-10 "amber is important to pursue" +
        # coordinator crash-recovery directive "FOLD AMBER INTO THE SAME RELAUNCH"): the SPEC §1.1
        # 4-value stability set composed as EXPLICIT flags (grad-clip 0.5 / pose-grad-coeff-max 25 /
        # grad-normalize per-param / per-group-grad-clip); the expected_stability manifest + the (d2)
        # startup-telemetry assertion verify the RESOLVED values actually reached the trainer.
        return wac.compile_crucible_v752_launch_config(
            gt_cache, num_pairs=num_pairs, self_orient=False, amber=True, **_ek)
    if config == "crucible_v753":
        # T5 CRUCIBLE-2 v7.5.3 (fractal-synthesis typed-delta over v7.5.2,
        # fullstack_fractal_optimal_synthesis_20260710.md §3). DEFAULT branch = trunk_basis='off' (the
        # §2 pre-registered owed16v2 OFF-arm) with every A/B arm OFF ⇒ argv byte-identical to the GO'd
        # v7.5.2 self-orient-OFF launch; the Δ2/Δ3 arms + Δ1 'on' branch + Δ5 MC-finisher are composed
        # by derive_crucible_v753_config kwargs (default-OFF, duty-to-measure). Same launcher-facing cfg
        # protocol as crucible_v752 (v7-identical constants+governance reused). means != ends: a MEANS.
        return wac.compile_crucible_v753_launch_config(
            gt_cache, num_pairs=num_pairs, **_ek)
    if config == "v9_cgauge_432":
        # Task #432 — the V9·CGauge COHERENT STATE-GATED-SCHEDULE ARM (the #430 bundle's
        # witness-DSL compile on the V9·CGauge base; vehicle_v9_cgauge_naming_20260711 the design
        # spec). = crucible_v752(self_orient=False, amber-equivalent explicit stability) + the V9
        # T1 phase-advection LEVER (0.4 @ ep726 static approx; label_floor event N7 BUILD-OWED) +
        # mod-dim 19 (cgauge_whitney_moddim_v1; the arm doubles as #299 Arm-A on the SPEC_v9 base).
        # Cascade gates ride the wired trainer sensors (lane_nucleus / annulus_plateau /
        # powerlaw_meat / sigma_min_plateau / tau-event / birth-completion); per-class-λ budget
        # shifts remain ORGAN-ADVISORY. FRESH start (mod-19 cannot warm-start mod-32 checkpoints).
        # Same launcher-facing cfg protocol as crucible_v752 (v7 constants+governance reused).
        # CONTROL = the #205 banked mod-32 baseline. means != ends: a MEANS.
        from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config
        return compile_v9_cgauge_432_launch_config(
            gt_cache, num_pairs=num_pairs, **_ek)
    if config in ("v9_cgauge_truly_optimal_core", "v9_cgauge_ideal_mod19",
                  "v9_cgauge_ideal_mod19_sR", "v9_cgauge_ideal_mod32"):
        # 2026-07-13 held event-native V9. The core is the mod19 main bet; the two
        # ideal_mod* names are the decisive matched FAMILY A/B. Both scientific arms
        # compile the same actuated flow; only --mod-dim differs (plus custody out-dir).
        # CONTAINMENT: derive only. launch remains operator-GO and is held until the
        # 95%-kill P0 completes to avoid GPU-timing contention.
        from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_ideal_launch_config

        mod_dim = 32 if config == "v9_cgauge_ideal_mod32" else 19
        return compile_v9_cgauge_ideal_launch_config(
            gt_cache, num_pairs=num_pairs, mod_dim=mod_dim,
            program_name=config,
            with_reachability=(config == "v9_cgauge_ideal_mod19_sR"), **_ek)
    if config in ("v9_cgauge_432_taper_off", "v9_cgauge_432_horizon_iso",
                  "v9_cgauge_432_step_iso"):
        # Task #432 top-3 duty-to-measure typed one-delta arms.  Each compiler starts from the
        # identical ideal-mod19 scientific control and fail-closes on the exact argv delta plus
        # LawRef/parser/trainer-consumer custody before returning a launchable config.
        from tac.witness_dsl.spec_v9_cgauge import (
            compile_v9_cgauge_432_horizon_iso_launch_config,
            compile_v9_cgauge_432_step_iso_launch_config,
            compile_v9_cgauge_432_taper_off_launch_config,
        )

        _iso_compilers = {
            "v9_cgauge_432_taper_off": compile_v9_cgauge_432_taper_off_launch_config,
            "v9_cgauge_432_horizon_iso": compile_v9_cgauge_432_horizon_iso_launch_config,
            "v9_cgauge_432_step_iso": compile_v9_cgauge_432_step_iso_launch_config,
        }
        return _iso_compilers[config](
            gt_cache_path=gt_cache, num_pairs=num_pairs, **_ek)
    if config in ("next_launch_all_levers_20260713", "next_launch_all_levers_trimmed_20260713"):
        # 2026-07-13 operator-GO-only ticket.  The compiler starts from the
        # ideal mod19 lineage, composes every compatible speed/init/observer
        # lever through the typed DSL, and carries fail-closed launch blockers
        # for the exact D-A/D-B/causal dependency slots.  Pure derive only.
        from tac.witness_dsl.spec_next_launch_all_levers_20260713 import (
            compile_next_launch_all_levers_ticket,
        )

        variant = ("trimmed_compliant"
                   if config == "next_launch_all_levers_trimmed_20260713" else "full")
        return compile_next_launch_all_levers_ticket(
            gt_cache, num_pairs=num_pairs, variant=variant, **_ek)
    if config in ("throughput_component_timer_async_20260713",
                  "throughput_component_timer_solo_20260713"):
        # Bounded n24, four-epoch, tau=1 (CE-exact) D-A timer and its
        # matched no-async control. Pure compile here; real actuation still
        # traverses every governed launcher gate and remains operator-GO-only.
        from tac.witness_dsl.spec_throughput_component_timer_20260713 import (
            compile_throughput_component_timer_ticket,
        )

        variant = ("solo_control"
                   if config == "throughput_component_timer_solo_20260713"
                   else "async_current")
        return compile_throughput_component_timer_ticket(
            gt_cache, num_pairs=num_pairs, epochs=int(_ek.get("epochs", 4)), variant=variant)
    # fail-LOUD default (seal v7 r1 BLOCKER #1): ONLY proven_base / all_levers ride the derive_config
    # fall-through (all_levers => --all-levers). ANY OTHER name is an unknown config and MUST RAISE —
    # never silently fall through to a proven_base WitnessConfig. That silent fall-through is its own
    # bug class: it is exactly how `--config crucible_v7` LOOKED launchable while actually running
    # proven_base (config_family would even MISLABEL it). A new named config MUST add an explicit
    # branch above; an unmapped name is a hard error, not a quiet substitution.
    if config in ("proven_base", "all_levers"):
        return wac.derive_config(gt_cache, num_pairs=num_pairs, overfit=overfit, **_ek,
                                 all_levers=(config == "all_levers"))
    raise ValueError(
        f"derive_named_config: unknown config name {config!r} — no derive branch resolves it. Known "
        f"configs: proven_base, all_levers, sealed_205, store_nothing_205, fresh_seeded, crucible_v6, "
        f"crucible_v7, crucible_v752, crucible_v753, v9_cgauge_432, "
        f"v9_cgauge_truly_optimal_core, v9_cgauge_ideal_mod19, "
        f"v9_cgauge_ideal_mod19_sR, v9_cgauge_ideal_mod32, "
        f"v9_cgauge_432_taper_off, v9_cgauge_432_horizon_iso, "
        f"v9_cgauge_432_step_iso, "
        f"next_launch_all_levers_20260713, next_launch_all_levers_trimmed_20260713, "
        f"throughput_component_timer_async_20260713, "
        f"throughput_component_timer_solo_20260713. "
        f"Add an explicit branch (NEVER "
        f"silently fall through to proven_base).")


def derive_named_config(config: str, gt_cache: str, *, num_pairs: int, epochs: int | None,
                        overfit: bool):
    """Derive a named config and reject an impossible curriculum before writes.

    The wrapper covers typed and legacy config families alike and therefore runs
    for dry-run, calibration, dry-start, and real launch callers.  Epoch-budget
    feasibility is a compiler/config-construction invariant, not an advisory
    launcher check.
    """

    cfg = _derive_named_config_unchecked(
        config, gt_cache, num_pairs=num_pairs, epochs=epochs, overfit=overfit
    )
    emitted = cfg.to_trainer_flags("SCHEDULE_FEASIBILITY_AUDIT")
    violations = _schedule_epoch_budget_violations(emitted, TRAINER_PATH)
    if violations:
        raise ValueError(
            f"named config {config!r} failed the DSL schedule epoch-budget gate: "
            + " | ".join(violations)
        )
    return cfg


# ───────────────────────── RSS calibration smoke (BUILD #294 piece B; optional, default OFF) ────
_SAFE_RUN_PEAK_MIB_RE = re.compile(r"peak_rss=(\d+(?:\.\d+)?)MiB")
_SAFE_RUN_JSON_PEAK_RE = re.compile(r'"peak_rss_mib"\s*:\s*(\d+(?:\.\d+)?)')


def parse_safe_run_peak_mib(text: str) -> float | None:
    """Parse safe_run's exit peak-RSS telemetry (detail line or --json row). PURE."""
    peaks = [float(m) for m in _SAFE_RUN_PEAK_MIB_RE.findall(text)]
    peaks += [float(m) for m in _SAFE_RUN_JSON_PEAK_RE.findall(text)]
    return max(peaks) if peaks else None


def calibration_verdict(projected_gib: float, actual_gib: float,
                        overrun_pct: float) -> tuple[bool, str]:
    """PURE overrun check: REFUSE when the MEASURED calibration peak already exceeds the projection
    by more than overrun_pct (the projection under-modeled the config => the full-scale projection
    cannot be trusted)."""
    limit = float(projected_gib) * (1.0 + float(overrun_pct) / 100.0)
    ok = float(actual_gib) <= limit
    detail = (f"calibration actual {actual_gib:.2f} GiB vs projected {projected_gib:.2f} GiB "
              f"(limit +{overrun_pct:.0f}% = {limit:.2f} GiB)")
    return (True, f"OK: {detail}") if ok else (
        False, f"OVERRUN: {detail} — the projection under-models this flag set; REFUSING the "
               f"full-scale launch (recalibrate the preflight constants before launching)")


def _run_rss_calibration(args, config: str, overfit: bool, out_dir: Path, label: str,
                         extra_flags: list[str] | None, wmp) -> int:
    """Run the emitted config at SMALL scale (REAL flag set, governed safe_run path, FOREGROUND,
    minutes) capturing actual peak RSS; write calibration_rss.json next to launch.sh; REFUSE
    (rc=5) on projection overrun > --calibrate-overrun-pct. Also appends projection+reconcile rows
    to the margin ledger — every calibration feeds calibrated_margin()."""
    import subprocess

    calib_dir = out_dir / "calibrate_rss"
    cfg_c = derive_named_config(config, args.gt_cache, num_pairs=args.calibrate_pairs,
                                epochs=args.calibrate_epochs, overfit=overfit)
    if extra_flags:
        print(
            "[launch-witness] ERROR: calibration received post-DSL trainer flags; "
            "Catalog #406 requires a typed Lever (rc=8).",
            file=sys.stderr,
        )
        return 8
    try:
        launch_c, _, _, _ = write_dsl_bound_launch(cfg_c, calib_dir)
    except Exception as exc:
        print(
            "[launch-witness] ERROR: calibration DSL compile binding REFUSED rc=8: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 8
    proj_c = wmp.project_from_launch_sh(launch_c, safe_frac=args.mem_preflight_safe_frac)
    print(f"# calibrate-rss: n={args.calibrate_pairs} epochs={args.calibrate_epochs} "
          f"projected peak {proj_c.projected_peak_gib} GiB — running FOREGROUND via safe_run "
          f"(timeout {args.calibrate_timeout_s:.0f}s)")
    try:
        wmp.record_projection(calib_dir, launch_c, proj_c, note="calibrate_rss")
    except Exception as exc:
        print(f"[launch-witness] WARNING: calibration ledger append failed ({exc})", file=sys.stderr)

    cmd = [sys.executable, str(_REPO / "tools" / "safe_run.py"),
           "--rss-mb", str(int(args.rss_cap_mb)), "--timeout", str(float(args.calibrate_timeout_s)),
           "--json", "--label", f"calib_{label}", "--", "bash", str(launch_c)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=float(args.calibrate_timeout_s) + 120.0)
    except subprocess.TimeoutExpired:
        print("[launch-witness] ERROR: calibration smoke exceeded its outer timeout — REFUSING "
              "the full-scale launch (no measured peak).", file=sys.stderr)
        return 5
    peak_mib = parse_safe_run_peak_mib((res.stderr or "") + (res.stdout or ""))
    if peak_mib is None:
        print(f"[launch-witness] ERROR: calibration smoke produced NO safe_run peak telemetry "
              f"(rc={res.returncode}) — REFUSING the full-scale launch (never launch on an "
              f"unmeasured calibration). stderr tail: {(res.stderr or '')[-500:]}", file=sys.stderr)
        return 5
    actual_gib = peak_mib / 1024.0
    # P11′ (T5 recess R3; P5 second-red-team verdict §3 item 1, F4): the small calibration smoke
    # (calibrate_epochs < the 25-ep verdict cadence) never fires the chunked n600 verdict, so its
    # measured peak silently MISSES the +~6 GiB chunked-verdict transient every real run WILL pay —
    # the "5-ep smoke never fires a verdict" false-green half of the F4 gap. Add the MEASURED
    # chunked-verdict floor to the smoke actual before the overrun check (conservative: if the
    # smoke DID fire a verdict this double-counts toward REFUSE, never toward a false SAFE).
    verdict_delta_gib = float(wmp.VERDICT_FLOOR_GIB)
    actual_gated_gib = actual_gib + verdict_delta_gib
    ok, reason = calibration_verdict(proj_c.projected_peak_gib, actual_gated_gib,
                                     args.calibrate_overrun_pct)
    report = {
        "config": config, "calibrate_pairs": args.calibrate_pairs,
        "calibrate_epochs": args.calibrate_epochs,
        "projected_peak_gib": proj_c.projected_peak_gib,
        "actual_peak_gib": round(actual_gib, 3),
        "verdict_delta_added_gib": verdict_delta_gib,
        "actual_gated_gib": round(actual_gated_gib, 3),
        "overrun_pct_limit": args.calibrate_overrun_pct,
        "smoke_rc": res.returncode, "ok": ok, "reason": reason,
        "note": ("small-n calibration validates the model's fixed-overhead term; the smoke never "
                 "fires the epoch-cadence chunked verdict, so the measured chunked-verdict floor "
                 "is ADDED to the smoke actual before the overrun check (P11' F4 amendment); the "
                 "n600 cf-cache scaling term is validated by the ledger's full-run reconciles"),
        "ts": _utc(),
    }
    (out_dir / "calibration_rss.json").write_text(json.dumps(report, indent=2))
    try:
        # LG-F1 (throughput review 2026-07-04): calib_dir has no run.log and safe_run never
        # registers it with the governor/blackbox, so the source-sniffing reconcile leg was dead
        # code. The measured peak is ALREADY in hand (safe_run telemetry parsed above) — pass it
        # as the explicit actual_override so every calibration truly feeds calibrated_margin().
        wmp.reconcile_run_dir(calib_dir, actual_override=(
            actual_gib, "safe_run peak_rss telemetry (calibrate_rss smoke, parsed by launcher)"))
    except Exception as exc:
        print(f"[launch-witness] WARNING: calibration reconcile append failed ({exc})",
              file=sys.stderr)
    if not ok:
        print(f"[launch-witness] ERROR: REFUSING to launch — {reason}", file=sys.stderr)
        return 5
    print(f"[launch-witness] calibrate-rss {reason}")
    if res.returncode != 0:
        print(f"[launch-witness] WARNING: calibration smoke exited rc={res.returncode} (peak was "
              f"still measured; inspect {calib_dir} before trusting the run).", file=sys.stderr)
    return 0


# ───────────────────── FULL-CONFIG DRY-START gate (owed-2 / SYNTHESIS §C item 2) ─────────────
def parse_dry_start_run_metrics(run_log: Path) -> dict:
    """Parse a bounded dry-start run.log (JSONL, the trainer's own telemetry): the MAX epoch stepped
    (``ep``/``epoch``), the gt-load overhead (``{"stage":"gt","secs":...}``), whether a resume ckpt was
    written (a ``checkpoint`` row naming a ``resume_latest``), and — on the resume pass — the resume
    evidence (a ``resume_model_source`` row + a ``resume_start_epoch`` field the trainer emits when it
    restores from disk). PURE: same file -> same dict. Missing file -> all-absent verdict (never raises).
    """
    epochs_completed = -1
    gt_secs: float | None = None
    ckpt_written = False
    resume_source = False
    resume_start_epoch: int | None = None
    try:
        text = Path(run_log).read_text()
    except OSError:
        text = ""
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        ep = d.get("ep", d.get("epoch"))
        if isinstance(ep, (int, float)) and not isinstance(ep, bool):
            epochs_completed = max(epochs_completed, int(ep))
        stg = d.get("stage")
        if stg == "gt" and isinstance(d.get("secs"), (int, float)):
            gt_secs = float(d["secs"])
        if stg == "checkpoint" and d.get("resume_latest"):
            ckpt_written = True
        if stg == "resume_model_source":
            resume_source = True
        rse = d.get("resume_start_epoch")
        if isinstance(rse, (int, float)) and not isinstance(rse, bool):
            resume_start_epoch = int(rse)
    return {
        "epochs_completed": epochs_completed,
        "gt_secs": gt_secs,
        "checkpoint_written": ckpt_written,
        "resume_model_source": resume_source,
        "resume_start_epoch": resume_start_epoch,
    }


def dry_start_sec_per_ep(wall_s: float, gt_secs: float | None,
                         epochs_completed: int) -> tuple[float | None, float | None]:
    """(gross, marginal) sec/ep from a bounded pass. gross = wall/epochs (amortizes the one-time boot
    over few epochs — an UPPER bound); marginal = (wall - gt_load)/epochs (subtracts the measured
    gt-load overhead, closer to the steady-state per-epoch cost the wall-clock budget wants). PURE."""
    e = int(epochs_completed)
    if e <= 0 or not (wall_s > 0.0):
        return None, None
    gross = wall_s / e
    marginal = (wall_s - float(gt_secs or 0.0)) / e
    return round(gross, 2), round(max(marginal, 0.0), 2)


def dry_start_boot_ok(p: dict) -> bool:
    """PASS-1 boots+steps+ckpts (regardless of the terminal rc — the pass is INTENTIONALLY wall-clock
    bounded, so safe_run's timeout SIGTERM is the EXPECTED terminus, not a failure): >=1 epoch stepped
    AND a resume ckpt written AND a peak measured. PURE."""
    return bool(p.get("epochs_completed", -1) >= 1 and p.get("checkpoint_written")
                and p.get("peak_rss_gib") is not None)


def dry_start_resume_ok(p2: dict) -> bool:
    """PASS-2 resumed from disk (again regardless of the terminal timeout rc): the trainer logged a
    resume_model_source row AND a positive resume_start_epoch (it actually restored the epoch position),
    AND stepped at least one more epoch past it (proving the resumed state trains). PURE."""
    rse = p2.get("resume_start_epoch") or 0
    return bool(p2.get("resume_model_source") and rse >= 1
                and p2.get("epochs_completed", -1) >= rse)


def _run_dry_start(args, config: str, overfit: bool, out_dir: Path, label: str,
                   extra_flags: list[str] | None, wmp,
                   projected_peak_gib: float | None = None) -> int:
    """FULL-CONFIG DRY-START (owed-2 / SYNTHESIS §C item 2). Reached AFTER the whole gate chain has run
    on the REAL n600 config (flag-validate, launch.sh, perf-env, constants, schedule-provenance,
    DSL-config, memory-preflight, safe-compile, system-admission, throughput) — so start-ability of the
    real config is already PROVEN by the gates. This step then proves the trainer BOOTS + builds the
    model + STEPS + writes a crash-resumable checkpoint, and that the checkpoint RELOADS (resume
    round-trip), WITHOUT a real multi-hour run — running the EXACT REAL launch.sh (unmodified real
    schedule / caps / levers; crucible_v7 pins an ABSOLUTE 3000-epoch schedule whose interlocking
    stage-stagger validators a shrunk-epochs config cannot satisfy) but WALL-CLOCK BOUNDED:

      PASS 1  fresh boot, REAL config + --ckpt-every 1, safe_run --timeout sized to ~N (<=3) epochs
              (governed, FOREGROUND). safe_run SIGTERMs it at the timeout — the intended bound, which
              DOUBLES as a crash simulation. Capture peak RSS + wall + epochs stepped + a written ckpt.
      PASS 2  RESUME round-trip: relaunch the REAL config --resume-from the PASS-1 dir, same timeout;
              assert the trainer logs a resume_model_source row + a positive resume_start_epoch (it
              restored from disk) AND steps past it (the resumed state trains), not a silent fresh start.

    Records peak RSS + sec/ep MEASURED to dry_start_report.json. EXITS cleanly (rc 0 on both passes
    green; rc 6 otherwise) — NEVER proceeds to the real spawn. Route: the SAME governed launcher
    machinery (safe_run RSS cap + timeout + admission); no raw-python bypass.
    """
    import subprocess
    import time as _time

    n = int(args.dry_start)
    if not (1 <= n <= 3):
        print(f"[launch-witness] ERROR: --dry-start must be in 1..3 (bounded scope; a longer run is a "
              f"real launch, not a dry-start); got {n}.", file=sys.stderr)
        return 2
    # Size each pass's wall-clock timeout to ~n epochs: a boot budget (gt-load + model build + first-step
    # MLX compile) + n * a conservative per-epoch upper bound. safe_run SIGTERMs at this budget → the
    # bound is wall-clock, not epochs (the trainer has no schedule-independent epoch cap), so the actual
    # epochs stepped are READ BACK from run.log and reported. Both knobs are CLI-tunable.
    pass_timeout = float(args.dry_start_boot_budget_s) + n * float(args.dry_start_per_ep_budget_s)

    def _pass(sub_name: str, resume_from: Path | None) -> dict:
        sub = out_dir / sub_name
        cfg_b = derive_named_config(config, args.gt_cache, num_pairs=args.num_pairs,
                                    epochs=None, overfit=overfit)  # REAL sealed epochs → all validators pass
        if extra_flags:
            raise RuntimeError(
                "Catalog #406 dry-start received post-DSL trainer flags; compose a typed Lever"
            )
        overrides: dict[str, object] = {"--ckpt-every": 1}
        if resume_from is not None:
            overrides["--resume-from"] = str(resume_from)
        cfg_b = with_internal_dsl_lever(
            cfg_b,
            name=f"catalog406_dry_start_{sub_name}",
            overrides=overrides,
        )
        launch_b, _, _, _ = write_dsl_bound_launch(cfg_b, sub)
        cmd = [sys.executable, str(_REPO / "tools" / "safe_run.py"),
               "--rss-mb", str(int(args.rss_cap_mb)),
               "--timeout", str(pass_timeout),
               "--projected-gib", (str(round(float(projected_peak_gib), 3))
                                    if projected_peak_gib else "0"),
               "--json", "--label", f"drystart_{label}"]
        # Thread the launcher's admission-override rationale into the inner safe_run so its OWN
        # system-admission gate honors the SAME operator decision the launcher already made (else a
        # tight-but-overridden ceiling refuses the bounded smoke with rc=5, never booting the trainer).
        if _admission_override_ok(args.admission_override_rationale):
            cmd += ["--admission-override-rationale", args.admission_override_rationale]
        cmd += ["--", "bash", str(launch_b)]
        t0 = _time.perf_counter()
        outer_timeout = False
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=pass_timeout + 300.0)
            rc, blob = res.returncode, (res.stderr or "") + (res.stdout or "")
        except subprocess.TimeoutExpired:
            outer_timeout, rc, blob = True, None, ""
        wall = _time.perf_counter() - t0
        peak_mib = parse_safe_run_peak_mib(blob)
        # (dry-start FIX 2026-07-10; attempt-1/-3 epochs=-1 root-cause) PERSIST the captured child
        # output to sub/run.log BEFORE parsing. Under this path the trainer's stdout is INHERITED by
        # safe_run (its Popen does not redirect) and captured by THIS subprocess.run — nothing writes
        # the run.log that real (durable-daemon) launches get from the daemon redirect, so
        # parse_dry_start_run_metrics read a MISSING file and reported epochs=-1/ckpt=False even when
        # the trainer had stepped + checkpointed (attempt-3 wrote levelset_resume_state.npz and would
        # STILL have failed the gate). The persisted file is also the durable inspection artifact the
        # FAILED-path message points at ("inspect ... run.log"). Append-mode: never clobber.
        try:
            with (sub / "run.log").open("a") as fh:
                fh.write(blob)
        except OSError as exc:
            print(f"# dry-start WARN: could not persist captured output to {sub / 'run.log'}: {exc!r}")
        m = parse_dry_start_run_metrics(sub / "run.log")
        gross, marginal = dry_start_sec_per_ep(wall, m["gt_secs"], m["epochs_completed"])
        return {"dir": str(sub), "rc": rc, "outer_timeout": outer_timeout,
                "wall_s": round(wall, 1), "pass_timeout_s": round(pass_timeout, 1),
                "peak_rss_gib": (round(peak_mib / 1024.0, 3) if peak_mib is not None else None),
                "sec_per_ep_gross": gross, "sec_per_ep_marginal": marginal, **m}

    print(f"# dry-start PASS 1: fresh boot at REAL n={args.num_pairs} config={config!r}, wall-clock bounded "
          f"~{n} epoch(s) (safe_run --timeout {pass_timeout:.0f}s; SIGTERM = intended bound + crash sim)")
    try:
        p1 = _pass("dry_start", None)
    except Exception as exc:
        print(
            "[launch-witness] ERROR: dry-start DSL compile binding REFUSED rc=8: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 8
    boot_ok = dry_start_boot_ok(p1)
    print(f"#   PASS 1: rc={p1['rc']} epochs={p1['epochs_completed']} peak={p1['peak_rss_gib']} GiB "
          f"ckpt={p1['checkpoint_written']} sec/ep(gross~{p1['sec_per_ep_gross']}) -> boot_ok={boot_ok}")

    p2: dict | None = None
    resume_ok = False
    if boot_ok:
        print(f"# dry-start PASS 2: RESUME round-trip (--resume-from {p1['dir']}, wall-clock bounded)")
        try:
            p2 = _pass("dry_start_resume", Path(p1["dir"]))
        except Exception as exc:
            print(
                "[launch-witness] ERROR: dry-start resume DSL compile binding REFUSED rc=8: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 8
        resume_ok = dry_start_resume_ok(p2)
        print(f"#   PASS 2: rc={p2['rc']} resume_source={p2['resume_model_source']} "
              f"resume_start_epoch={p2['resume_start_epoch']} epochs={p2['epochs_completed']} "
              f"-> resume_ok={resume_ok}")
    else:
        print("# dry-start PASS 2 SKIPPED (PASS 1 did not boot+step+ckpt).", file=sys.stderr)

    report = {
        "gate": "full_config_dry_start",
        "owed": "owed-2 / SYNTHESIS §C item 2",
        "config": config, "num_pairs": args.num_pairs, "dry_start_target_epochs": n,
        "pass_timeout_s": round(pass_timeout, 1),
        "boot_ok": boot_ok, "resume_round_trip_ok": resume_ok,
        "green": bool(boot_ok and resume_ok),
        "peak_rss_gib": p1["peak_rss_gib"],
        "sec_per_ep_gross": p1["sec_per_ep_gross"],
        "sec_per_ep_marginal": p1["sec_per_ep_marginal"],
        "pass1": p1, "pass2": p2,
        "note": ("runs the EXACT REAL launch.sh (unmodified real schedule/caps/levers), wall-clock bounded "
                 "to ~N epochs by safe_run --timeout (crucible_v7 pins an atomic 3000-epoch schedule whose "
                 "interlocking stage-stagger validators a shrunk-epochs smoke cannot satisfy — so the bound "
                 "is wall-clock, and safe_run's SIGTERM at the budget doubles as a crash sim for the resume "
                 "round-trip). The full gate chain already ran on the REAL n600 config above (memory "
                 "preflight + admission + throughput). peak_rss_gib + sec_per_ep are MEASURED; "
                 "sec_per_ep_gross=wall/epochs is an UPPER bound (includes the one-time boot); "
                 "sec_per_ep_marginal=(wall-gt_load)/epochs is closer to steady-state. Feeds §B.wall_clock / "
                 "the #385 dual-chain brief. NEVER proceeds to the real launch."),
        "ts": _utc(),
    }
    (out_dir / "dry_start_report.json").write_text(json.dumps(report, indent=2))
    print(f"[launch-witness] wrote {out_dir / 'dry_start_report.json'}")
    if report["green"]:
        print(f"[launch-witness] DRY-START GREEN: boot+step+ckpt+resume all pass at REAL n={args.num_pairs}. "
              f"peak={p1['peak_rss_gib']} GiB, sec/ep(gross~{p1['sec_per_ep_gross']}). "
              f"NOT launching (dry-start exits cleanly).")
        return 0
    print(f"[launch-witness] DRY-START FAILED: boot_ok={boot_ok} resume_ok={resume_ok} — inspect "
          f"{out_dir}/dry_start* run.log before a real launch.", file=sys.stderr)
    return 6


# ───────────────────── shadow observer auto-start (#247 agent-native) ─────────────────────
def _observer_label(out_dir: Path) -> str:
    return f"costate_obs_{out_dir.name}"


def _observer_already_running(label: str) -> bool:
    """Idempotency: True iff the durable-daemon registry has a RUNNING row with this
    label AND its pid is alive (stale rows for dead pids do NOT count)."""
    try:
        reg = json.loads((_REPO / ".omx" / "state" / "durable_daemons.json").read_text())
        rows = reg if isinstance(reg, list) else list(reg.values()) if isinstance(reg, dict) else []
        for r in rows:
            if not (isinstance(r, dict) and r.get("label") == label and r.get("status") == "running"):
                continue
            try:
                os.kill(int(r.get("pid", 0)), 0)
                return True
            except (ProcessLookupError, ValueError, TypeError):
                continue
            except PermissionError:
                return True
    except Exception:
        pass
    return False


def ensure_shadow_observer(out_dir: Path) -> None:
    """AUTO-START the score-neutral #247 shadow observer for this run (CLAUDE.md
    "'Off' is a tracked queue": read-only observability DEFAULTS ON — it must never
    depend on a human remembering to start it). The observer is SENSE-only: each tick
    is a short-lived ``costate_shadow_report --write`` subprocess (reads run.log +
    launch.sh; writes ONLY the advisory ``costate_shadow.jsonl`` sidecar the trainer
    never reads -> training byte-identity preserved by construction; the package has
    no actuation capability, source-scan-tested). Self-terminates when the trainer
    exits. Idempotent: skipped when a live observer with this label already exists."""
    label = _observer_label(out_dir)
    if _observer_already_running(label):
        print(f"[launch-witness] shadow observer already running (label={label}); not double-starting.")
        return
    import spawn_durable_daemon as sdd
    rc = sdd.main([
        "--log", str(out_dir / "observer.log"), "--label", label,
        "--projected-gb", "0.2", "--projected-peak-gib", "0.1",
        "--min-free-gb", "2", "--rss-cap-mb", "2048",
        "--", sys.executable, str(_REPO / "tools" / "costate_observer_loop.py"),
        "--run-dir", str(out_dir),
    ])
    if rc == 0:
        print(f"[launch-witness] shadow observer STARTED (label={label}; SENSE-only, "
              f"score-neutral; log={out_dir / 'observer.log'}).")
    else:
        print(f"[launch-witness] WARNING: shadow observer failed to start (rc={rc}); the run "
              f"is unaffected — observe manually via tools/costate_shadow_report.py.",
              file=sys.stderr)


# ───────────────────────── main ─────────────────────────
def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", required=True, help="path to the clip's GT cache (e.g. .../gt_n600.npz)")
    ap.add_argument("--num-pairs", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs. OMIT to use the named config's SEALED default (each "
                         "derive function declares its own; crucible_v6/v7 seal 3000, older "
                         "families 1000). An explicit value OVERRIDES the sealed default and is "
                         "stamped as a loud provenance note (NEW-1: the launcher's old hardcoded "
                         "default=1000 silently trampled sealed values and compiled a wrong "
                         "wall-clock budget that passed every gate).")
    ap.add_argument("--aggressive", action="store_true",
                    help="overfit=False: aggressive Whitney-floor mod-dim (rate-saving)")
    ap.add_argument("--config", default=None,
                    choices=["proven_base", "all_levers", "sealed_205", "store_nothing_205",
                             "fresh_seeded", "crucible_v6", "crucible_v7", "crucible_v752",
                             "crucible_v753", "v9_cgauge_432",
                             "v9_cgauge_truly_optimal_core",
                             "v9_cgauge_ideal_mod19", "v9_cgauge_ideal_mod19_sR",
                             "v9_cgauge_ideal_mod32",
                             "v9_cgauge_432_taper_off",
                             "v9_cgauge_432_horizon_iso",
                             "v9_cgauge_432_step_iso",
                             "next_launch_all_levers_20260713",
                             "next_launch_all_levers_trimmed_20260713",
                             "throughput_component_timer_async_20260713",
                             "throughput_component_timer_solo_20260713"],
                    help="canonical named config resolved from the triality (tac.witness_autoconfig): "
                    "proven_base (attribution-clean baseline; the default when neither --config nor "
                    "--all-levers is given), all_levers (== --all-levers), sealed_205 (the #205 "
                    "Phase-3 SEALED capstone argv — the deep-math-OPTIMAL all-levers base + the 4 "
                    "SEALED deltas mod-dim 32 / adam-beta2 0.999 / w-pose 1.0 + pose-carrier table; "
                    "reproduces the sealed §7 launch.sh BYTE-IDENTICALLY modulo --out-dir), "
                    "store_nothing_205 (the sealed capstone + --pose-carrier-source generated = Track B "
                    "STORE-NOTHING-but-xi: frame0 = warp(witness's OWN render, xi), stores ONLY xi/H; "
                    "the A/B pose arm vs sealed_205), or fresh_seeded (the REVISED run-1 argv from the "
                    "2026-07-04 pre-launch SEAL review: sealed_205 + the lane-nucleation seed fix "
                    "[paint + --seed-islands] + eikonal 0.05->0.10 + constant tau=1.0 + mod-dim 19 + "
                    "film-stiefel + muon warm-start/final-frac + band 350 + rewarmup 20-cosine + "
                    "closed-loop control; event-triggered curriculum + bank-6 deliberately EXCLUDED "
                    "per C1/C2/C3), or crucible_v6 (the T5 CRUCIBLE v6.2 launch candidate = "
                    "store_nothing_205 + ABSOLUTE schedule pins [tau@300, anneal-den 3000 x "
                    "hold-frac 0.2 = tau descent completes ep600 and HOLDS 0.31, Muon cap 726 — "
                    "NEVER family-scaled] + --softmax-temp-end 0.31 + --fused-r-kernel + the v6 "
                    "S1.1 DSL levers + ChromaBoundarySharpen + V=5 co-predicate; pose block "
                    "inherited from store_nothing_205 per seal-round-2 MAJOR-A2/#314), or crucible_v7 "
                    "(the T5 CRUCIBLE v7 restart = the FIRST requirement-V-native config, authored AS "
                    "a tac.witness_dsl.typed_config.TypedWitnessConfig with a DSL-emitted argv: v6 "
                    "substrate + witness-native continuous L_tau [--seg-form-unify-tau, removes the "
                    "last PR95 curriculum bone] + geometric tau anneal [floor tau*=0.31] + TAIL_k "
                    "warm-restart + LADDER island-birth homotopy; the three transitions FIRE on wired "
                    "sensors [powerlaw_meat / lane_nucleus / annulus_plateau] with tagged fail-safe "
                    "backstop caps [0 naked epochs]; ships a DSL-provenance manifest [b0.6 VERIFIES] + "
                    "the v6-inherited LawRef constants manifest; pose block verbatim from v6), or "
                    "crucible_v752 (the T5 CRUCIBLE-2 v7.5.2 launch-1 SELF-ORIENT-OFF config = the "
                    "sealed crucible_v7 trunk + #121 d_seg-aware taper MINUS σ_cc′ MINUS the self-orient "
                    "directional basis [owed-16 P9 RESOLVED-REFUTING: realized −48%% transfer measured "
                    "≈0, 47 GiB RAM tax removed; operator GO 2026-07-10]; reuses v7's constants + "
                    "schedule-governance manifests with its own DSL-provenance fingerprint), or "
                    "crucible_v753 (the fractal-synthesis typed-delta over v7.5.2; DEFAULT branch = "
                    "argv byte-identical to the GO'd v7.5.2 launch, A/B arms composed via kwargs), or "
                    "v9_cgauge_432 (task #432: the V9·CGauge COHERENT STATE-GATED-SCHEDULE ARM = "
                    "crucible_v752 self-orient-OFF + explicit amber stability + the T1 phase-advection "
                    "LEVER 0.4@726 + mod-dim 19 [cgauge_whitney_moddim_v1; doubles as #299 Arm-A on "
                    "the SPEC_v9 base]; the #430 cascade rides the wired trainer sensors; FRESH start "
                    "— mod-19 cannot warm-start mod-32 checkpoints; CONTROL = the #205 banked "
                    "mod-32 baseline).")
    ap.add_argument("--extra-trainer-flags", default=None,
                    help="(C5 passthrough) EXTRA trainer flags appended verbatim to the emitted "
                    "launch.sh command (shell-split; e.g. \"--eikonal-weight 0.07 --seed-islands\"). "
                    "Every --flag token is validated against the trainer's REAL argparse "
                    "(never-invent-a-flag) and the memory preflight re-parses the final launch.sh, so "
                    "memory-relevant extras (e.g. --bank-n-scales) are gated too. This is the governed "
                    "escape hatch — raw heavy python launches remain FORBIDDEN.")
    ap.add_argument("--per-group-grad-clip", action=argparse.BooleanOptionalAction, default=True,
                    help="(Fix 4 / confound C4) EMIT --per-group-grad-clip to the trainer (default "
                    "ON): clips the eikonal + seg gradients per-group so the volatile eikonal term "
                    "cannot hijack the SHARED grad-clip budget and starve the seg step. The trainer's "
                    "own default is OFF (for byte-identity); this launcher opts in. --no-per-group-"
                    "grad-clip restores the trainer default.")
    ap.add_argument("--all-levers", action="store_true",
                    help="emit the deep-math-OPTIMAL all-levers from-scratch config (#205 artifact); "
                    "equivalent to --config all_levers. "
                    "--render-aa none + analytic coverage-integrated lane-render-band (Wave D AA "
                    "correction; supersample DISQUALIFIED: hurts -49%% + decode over budget) + "
                    "persistence/topology loss + "
                    "island-birth amplification + annealed hosc 1->4 + l7 DEMOTED + verdict-pairs 0 + "
                    "mod-dim 19 (Whitney floor) + adam-beta2 0.9999999. Default OFF => attribution-clean "
                    "proven_base baseline.")
    ap.add_argument("--dsl-lever", action="append", default=[], metavar="NAME",
                    help="(#332) compose a named DSL Lever factory over the base config "
                    "(repeatable), e.g. --dsl-lever SeedIslandEased --dsl-lever EventTriggeredCurriculum. "
                    "Resolved from tac.witness_dsl.curriculum_dsl (the config-generating SoT); each "
                    "lever's overrides are merged over the base, and every emitted flag is argparse-"
                    "validated. Composable (zero-arg, single-Lever — derived from the DSL predicate, "
                    "never hand-typed): " + " ".join(_composable_lever_names()) + ". "
                    "Factories needing explicit args (Muon) or returning composites (DM1Minimal) are "
                    "refused with a clear error before any gate/spawn work.")
    ap.add_argument("--out-dir", default=None,
                    help="run out-dir (default: experiments/results/levelset_n<N>_witness_<utc>)")
    ap.add_argument("--label", default=None, help="daemon label (default: derived from out-dir)")
    ap.add_argument("--readiness-defer", action="append", default=None, metavar="RUNG=REASON",
                    help="(repeatable) record a launch-readiness DEFER for a fire-now duty rung "
                         "the config deliberately excludes: writes a canonical "
                         "'# LAUNCH_READINESS_DEFER:<rung>=<reason>' line into the emitted "
                         "launch.sh so the daemon's readiness gate honors it. Reason must be "
                         "substantive (the gate rejects placeholders). Added 2026-07-11 after the "
                         "#432 fire required hand-editing launch.sh + a direct sdd spawn because "
                         "the launcher regenerates launch.sh and had no defer passthrough.")
    ap.add_argument("--purpose", default=None,
                    help="DECLARED one-line intent of THIS run (e.g. 'clean baseline / control', "
                         "'A/B arm: eikonal 0.07 vs mod32cap parent', 'frontier candidate'). "
                         "Stamped into launch.sh as '# tac-run-purpose:' (the run dir's config "
                         "record) so dashboards render it with provenance 'declared'; unset -> "
                         "consumers show a LABELLED derived heuristic. Metadata only — never a "
                         "trainer flag, zero effect on training or argv byte-identity.")
    ap.add_argument("--rss-cap-mb", type=int, default=90000,
                    help="per-arm RSS cap (MiB) for safe_run layer-3 (default 90000)")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="OOM launch-preflight free-memory floor (default 10; operator-relaxed)")
    ap.add_argument("--mem-preflight-safe-frac", type=float, default=None,
                    help="(#205 OOM self-protection) REFUSE launch if projected peak RSS exceeds this "
                    "fraction of total RAM. DEFAULT = POLICY-DERIVED at runtime (operator memory "
                    "policy 2026-07-04): 0.85 when NO other governed heavy job is admitted/running "
                    "(sole-workload — no artificial ceiling), 0.70 under admitted concurrency; "
                    "governor state unreadable -> conservative 0.70. The read is READ-ONLY (registry "
                    "rows + pid liveness). An explicit value here always wins over the policy.")
    ap.add_argument("--skip-mem-preflight", action="store_true",
                    help="bypass the projected-peak-RSS memory preflight (WARN instead of REFUSE)")
    ap.add_argument("--calibrate-rss", action="store_true",
                    help="(BUILD #294 optional pre-launch hardening) run the emitted config for a few "
                    "epochs at small n (REAL flag set, governed safe_run path, FOREGROUND, minutes) "
                    "capturing actual peak RSS next to the projection; REFUSE the full launch if the "
                    "measured peak already exceeds the projection by > --calibrate-overrun-pct. "
                    "Default OFF — the default launch path is unchanged.")
    ap.add_argument("--calibrate-pairs", type=int, default=24,
                    help="n-pairs for the RSS calibration smoke (default 24)")
    ap.add_argument("--calibrate-epochs", type=int, default=3,
                    help="epochs for the RSS calibration smoke (default 3)")
    ap.add_argument("--calibrate-overrun-pct", type=float, default=15.0,
                    help="REFUSE the launch when calibration actual peak exceeds its projection by "
                    "more than this percent (default 15)")
    ap.add_argument("--calibrate-timeout-s", type=float, default=1800.0,
                    help="safe_run timeout for the calibration smoke (default 1800s)")
    ap.add_argument("--admission-override-rationale", default=None,
                    help="operator-quoted rationale to OVERRIDE a SYSTEM admission REFUSAL (the "
                         "SUM-over-RAM crash gate); the ONLY non-skip bypass (placeholder/empty rejected)")
    ap.add_argument("--verify-s", type=float, default=4.0,
                    help="seconds spawn_durable_daemon verifies the child survived exec")
    ap.add_argument("--perf-env-timeout-s", type=float, default=45.0,
                    help="seconds to wait for the custom_grouped_backward perf line")
    ap.add_argument("--skip-throughput-gate", action="store_true",
                    help="skip the pre-spawn SegNet fwd+bwd throughput micro-bench (the ~17x fast-path "
                    "assertion). Default runs it (a measured gate, not a flag-grep).")
    ap.add_argument("--accept-wall-clock", type=float, default=None, metavar="DAYS",
                    help="(L45 operator escape hatch) SUPPLY/OVERRIDE the wall-clock budget in DAYS and "
                    "LOUDLY stamp wall_clock_accept.txt in the run dir. Wall-clock gating is DEFAULT-ON "
                    "(the config declares its DERIVED budget; a non-declaring legacy config uses the "
                    "launcher-derived anchor fallback), so the projection REFUSES an over-budget launch "
                    "WITHOUT this flag; pass it only to knowingly accept a longer run (never silent).")
    ap.add_argument("--skip-schedule-provenance-gate", action="store_true",
                    help="(operator 2026-07-09 'move from hardcoded epochs to event based') downgrade "
                    "the schedule-provenance gate from REFUSE to WARN. The gate refuses a REAL launch "
                    "whose emitted --*-start-epoch schedule TRIGGERS are naked hardcoded epochs — not "
                    "event-governed (a co-emitted --curriculum-event-triggered/--curriculum-nucleus-"
                    "guard/--plateau-trigger/--closed-loop-control declared in schedule_governance), "
                    "not LawRef-DERIVED (in constants_manifest.json), and not a TAGGED fail-safe cap. "
                    "Default ENFORCES (--dry-run is always advisory: it prints the table, never refuses).")
    ap.add_argument("--skip-dsl-config-gate", action="store_true",
                    help="RETIRED compatibility option. Catalog #406 never downgrades the DSL compile "
                    "hash gate; an invalid/missing binding is REFUSED with rc=8 in dry-run and real "
                    "launch modes.")
    ap.add_argument("--enforce-dsl-config-gate", action="store_true",
                    help="RETIRED compatibility option. DSL compile-hash enforcement is now always on "
                    "and fail-closed (rc=8).")
    ap.add_argument("--allow-non-dsl-config", default=None, metavar="RATIONALE",
                    help="RETIRED compatibility option. A rationale cannot authorize a hand-ruled "
                    "config; missing/mismatched DSL compile custody is always REFUSED with rc=8.")
    ap.add_argument("--throughput-threshold-ms", type=float, default=None,
                    help="override the SegNet fwd+bwd median ms gate (default 700; measured ON~396 / "
                    "OFF~6713). >threshold => REFUSE (custom-grouped-backward fast path not active).")
    ap.add_argument("--dashboard-port", type=int, default=8790)
    ap.add_argument("--no-dashboard", action="store_true", help="skip the dashboard up-check")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit + flag-validate + write launch.sh, but DO NOT spawn (CPU-only, safe)")
    ap.add_argument("--dry-start", type=int, default=0, metavar="EPOCHS",
                    help="(owed-2 / SYNTHESIS §C item 2) FULL-CONFIG DRY-START GATE. Run the ENTIRE gate "
                    "chain on the REAL n600 config (flag-validate, launch.sh, perf-env, constants, "
                    "schedule-provenance, DSL-config, memory-preflight, safe-compile, system-admission, "
                    "throughput), then — INSTEAD of the unbounded durable spawn — execute a BOUNDED "
                    "EPOCHS (<=3) trainer run at the REAL n/levers FOREGROUND via the governed safe_run "
                    "path (proves the trainer BOOTS + builds the model + steps + writes a resume ckpt), "
                    "then a RESUME ROUND-TRIP (relaunch --resume-from the written ckpt for +1 epoch), then "
                    "EXIT cleanly (NEVER the real launch). Records peak RSS + sec/ep MEASURED to "
                    "dry_start_report.json. Bounded to <=3 (a longer run is a real launch); same governed "
                    "machinery — no raw-python bypass. 0 (default) = OFF.")
    ap.add_argument("--dry-start-boot-budget-s", type=float, default=300.0,
                    help="(dry-start) estimated one-time boot overhead (gt-load + model build + first-step "
                    "MLX compile) in seconds; the per-pass safe_run timeout = boot-budget + N*per-ep-budget "
                    "(default 300).")
    ap.add_argument("--dry-start-per-ep-budget-s", type=float, default=90.0,
                    help="(dry-start) conservative per-epoch upper bound (seconds) used to size the "
                    "wall-clock timeout to ~N epochs (default 90; the real n600 anchor is ~42 s/ep).")
    args = ap.parse_args(argv)

    # (L5/XC-ii) POLICY-AWARE safe-frac: derive the memory-preflight fraction from the operator
    # memory policy (2026-07-04) unless the CLI pinned it. Printed loud so the fired branch (and
    # why) is always in the launch record; every downstream consumer (b1 preflight + calibrate-rss)
    # reads the resolved value from args.
    _sf, _sf_branch, _sf_why = derive_safe_frac(args.mem_preflight_safe_frac)
    args.mem_preflight_safe_frac = _sf
    print(f"# mem-preflight safe-frac {_sf:.2f} [{_sf_branch}] — {_sf_why}")

    overfit = not args.aggressive

    # Resolve the canonical named config (triality selector). Backward-compat: --all-levers is an
    # alias for --config all_levers; the historical default (neither given) is proven_base.
    config = args.config
    if config is None:
        config = "all_levers" if args.all_levers else "proven_base"
    elif args.all_levers and config != "all_levers":
        print(f"[launch-witness] ERROR: --config {config} conflicts with --all-levers "
              f"(--all-levers == --config all_levers); pass exactly one.", file=sys.stderr)
        return 2

    cfg = derive_named_config(config, args.gt_cache, num_pairs=args.num_pairs,
                              epochs=args.epochs, overfit=overfit)
    # (fresh-eyes advisory P0-1 required gate, 2026-07-10) EXPECTED-ACTIVE-LEVER manifest re-check at
    # the LAUNCHER (compile already enforced it fail-closed; this catches a cfg object mutated/patched
    # between compile and launch, and runs for dry-run, dry-start, AND real spawns). Applies to any
    # config whose dsl_program_manifest carries "expected_active_levers"; others are unaffected.
    _exp_levers = (getattr(cfg, "dsl_program_manifest", None) or {}).get("expected_active_levers")
    if _exp_levers is not None:
        _got_levers = list(getattr(cfg, "dsl_levers", ()) or ())
        if sorted(_got_levers) != sorted(_exp_levers):
            print(f"[launch-witness] ERROR: expected-active-lever manifest MISMATCH for {config!r}: "
                  f"composed={sorted(_got_levers)} != expected={sorted(_exp_levers)} — a lever was "
                  f"dropped/added after typed authoring (built-but-not-composed class, advisory P0-1). "
                  f"REFUSING (rc=10).", file=sys.stderr)
            return 10
        print(f"# expected-active-lever manifest: OK ({len(_exp_levers)} levers match the pinned "
              f"expectation for {config!r})")
    # NEW-1 (seal v7 round-2 docket): epochs provenance. When --epochs is OMITTED the config
    # family's SEALED default just applied (read it back from cfg for the header/telemetry);
    # when EXPLICIT, stamp a LOUD note — the wall-clock gate DERIVES its budget from whatever
    # epochs it is handed, so it cannot catch a wrong hand on its own.
    if args.epochs is None:
        args.epochs = int(getattr(cfg, "epochs", 0) or 0)
        print(f"# epochs: {args.epochs} (config-sealed default for {config!r})")
    else:
        _cfg_ep = int(getattr(cfg, "epochs", 0) or 0)
        _note = "matches derived config" if _cfg_ep == int(args.epochs) else "EXPLICIT OVERRIDE"
        print(f"# epochs: {args.epochs} ({_note}; wall-clock budget scales with THIS value — "
              f"verify intent)")
    # (#332) compose selected DSL levers over the named base; the config DELEGATES to the DSL SoT.
    # Each name is EAGERLY resolved through the DSL composability predicate HERE — BEFORE any
    # gate/spawn work — so a non-composable name (Muon needs explicit args; DM1Minimal returns a
    # composite) is a clean one-line refusal, never a raw traceback mid-launch (CLASS-fix,
    # review 2026-07-06).
    if args.dsl_lever:
        import dataclasses as _dc

        from tac.witness_dsl.lever_registry import (
            LeverCompositionError,
            resolve_composable_lever,
        )
        try:
            for _lv in args.dsl_lever:
                resolve_composable_lever(_lv)
        except LeverCompositionError as exc:
            print(f"[launch-witness] ERROR: {exc}", file=sys.stderr)
            return 2
        # APPEND to (never clobber) any levers the named config pre-composes. Requirement-V
        # configs expose a typed adapter rather than a dataclass ``dsl_levers`` field; compose
        # into that typed DSL and regenerate its manifest instead of hand-mutating argv.
        if hasattr(cfg, "with_dsl_lever_factories"):
            cfg = cfg.with_dsl_lever_factories(*args.dsl_lever)
        else:
            cfg = _dc.replace(cfg, dsl_levers=(*cfg.dsl_levers, *args.dsl_lever))
        print(f"[launch-witness] DSL levers composed: {', '.join(cfg.dsl_levers)}")

    # RUN-IDENTITY: thread the DECLARED purpose into the config (metadata only; the
    # identity header in launch.sh renders it; the argv is untouched by construction).
    if args.purpose:
        import dataclasses as _dcp

        _purpose = " ".join(str(args.purpose).split())
        cfg = (
            cfg.with_purpose(_purpose)
            if hasattr(cfg, "with_purpose")
            else _dcp.replace(cfg, purpose=_purpose)
        )
        print(f"# declared run purpose: {cfg.purpose}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO / "experiments" / "results" / f"levelset_n{args.num_pairs}_witness_{_utc()}")
    label = args.label or f"levelset_witness_{out_dir.name}"

    # (pointer-only discipline: the exact pointer lives in
    # .omx/state/canonical_frontier_pointer.json — never a hardcoded literal here)
    print(f"# launch_witness_run {wac.ADVISORY_TAG}  pointer UNMOVED (SoT: canonical_frontier_pointer.json)")
    print(f"# clip={args.gt_cache} num_pairs={args.num_pairs} epochs={args.epochs} "
          f"overfit={overfit} config={config}")
    print(f"# out_dir={out_dir}")
    if not (_REPO / args.gt_cache).exists() and not Path(args.gt_cache).exists():
        print(f"# NOTE: gt-cache {args.gt_cache} not found on disk -> gt regen required at launch",
              file=sys.stderr)

    # (a) FLAG-VALIDATE (never-invent-a-flag, NO-FAKE) — refuse before writing anything.
    # Any residual LeverCompositionError from the DSL merge (e.g. a lever override setting
    # False on a plain store_true flag) is the same clean one-line refusal, never a traceback.
    from tac.witness_dsl.lever_registry import LeverCompositionError as _LCE
    try:
        all_pass, results = validate_emitted_flags(cfg, str(out_dir))
    except _LCE as exc:
        print(f"[launch-witness] ERROR: {exc}", file=sys.stderr)
        return 2
    n = len(results)
    n_ok = sum(1 for _, ok in results if ok)
    print(f"# flag validation: {n_ok}/{n} flags exist in the trainer argparse")
    if not all_pass:
        bad = [f for f, ok in results if not ok]
        print(f"[launch-witness] ERROR: REFUSING to launch — emitted invented flag(s) "
              f"not in the trainer argparse: {bad}", file=sys.stderr)
        return 2

    # (a2) EXTRA-TRAINER-FLAGS passthrough (C5): same never-invent-a-flag bar as the derived config.
    extra_flags, invented = parse_extra_trainer_flags(args.extra_trainer_flags)
    if invented:
        print(f"[launch-witness] ERROR: REFUSING to launch — --extra-trainer-flags contains "
              f"invented flag(s) not in the trainer argparse: {invented}", file=sys.stderr)
        return 2
    if extra_flags:
        print(f"# extra trainer flags (requested, validated): {' '.join(extra_flags)}")

    # (a3) EMIT-SIDE CONFOUND FIXES (confound_hunt_synthesis_20260705.md). Compose the derived-config
    # flag names with the passthrough extras, then: Fix 4 opt-in --per-group-grad-clip (C4), Fix 2
    # palliative-implies-warm-start (C8), Fix 3 seed-anneal relative to resume epoch (C16), and the
    # highest-value Fix 1 C13 GUARD — REFUSE any duplicate long-flag across the FINAL emitted argv
    # (argparse last-wins silently shifts schedules; the v6 launch had 5 dup flags that flattened
    # eikonal-weight-end and shifted tau/lane-band/persistence by 100 epochs).
    config_flag_names = [flag for flag, _ in cfg.to_trainer_flags(str(out_dir))]
    extra_flags, _fix_notes, _dups = apply_emit_side_confound_fixes(
        extra_flags, config_flag_names, per_group_grad_clip=args.per_group_grad_clip)
    for _note in _fix_notes:
        print(f"# {_note}")
    if _dups:
        print(f"[launch-witness] ERROR: REFUSING to launch — the emitted argv would contain "
              f"DUPLICATE long-flag(s) {_dups} (argparse last-wins silently shifts schedules; "
              f"confound C13). Remove the duplicate from --extra-trainer-flags (the derived config "
              f"already emits it), or reconcile the config.", file=sys.stderr)
        return 2
    if extra_flags:
        print(f"# extra trainer flags (final, post emit-side fixes): {' '.join(extra_flags)}")

    # (a4 / Catalog #406) COMPILE THE CRYPTOGRAPHIC DSL BINDING.  Any trailing
    # semantic token is, by construction, outside the WitnessProgram and has no
    # #332 Lever owner.  It cannot acquire a valid hash by being argparse-valid.
    if extra_flags:
        print(
            "[launch-witness] ERROR: REFUSING to launch — rc=8 Catalog #406: "
            f"post-DSL semantic argv tokens have no #332 Lever owner: {extra_flags}. "
            "Move the change into a typed DSL Lever and recompile; "
            "--extra-trainer-flags cannot authorize a hand-ruled argv.",
            file=sys.stderr,
        )
        return 8
    try:
        _dsl_document, _dsl_manifest, _dsl_detail = compile_dsl_document_for_config(
            cfg,
            out_dir,
            program_name=config,
        )
        _dsl_compile_hash = str(_dsl_document["dsl_compile_hash"])
    except Exception as exc:
        print(
            "[launch-witness] ERROR: REFUSING to launch — rc=8 Catalog #406 DSL "
            f"compile binding unavailable: {type(exc).__name__}: {exc}. "
            "Rule chain: TypedWitnessConfig -> WitnessProgram.compile_trainer_argv() "
            "-> #332 bijection/LawRef manifest -> dsl_compile_hash.",
            file=sys.stderr,
        )
        return 8

    # (b) WRITE launch.sh (script-based command — no word-split fragility) with
    # the compile hash in its header and trainer-side admission environment.
    launch_sh = write_launch_sh(
        cfg,
        out_dir,
        dsl_compile_hash=_dsl_compile_hash,
    )
    if getattr(args, "readiness_defer", None):
        # Persist operator-recorded readiness DEFERs into the emitted launch.sh (the
        # surface the readiness gate reads). Atomic rewrite (tmp+os.replace) per the
        # same live-bash-fd rule write_launch_sh honors.
        _body = launch_sh.read_text()
        _defer_lines = []
        for _d in args.readiness_defer:
            _rung, _sep, _reason = str(_d).partition("=")
            if not _sep or len(_reason.strip()) < 8:
                print(f"[launch-witness] ERROR: --readiness-defer needs RUNG=<substantive reason>: {_d!r}",
                      file=sys.stderr)
                return 2
            _defer_lines.append(f"# LAUNCH_READINESS_DEFER:{_rung.strip()}={_reason.strip()}\n")
        _lines = _body.splitlines(keepends=True)
        _idx = next((i for i, _ln in enumerate(_lines) if not _ln.startswith("#") and _ln.strip()), 0)
        _tmp = out_dir / f".launch.sh.tmp.defer.{os.getpid()}"
        _tmp.write_text("".join(_lines[:_idx]) + "".join(_defer_lines) + "".join(_lines[_idx:]))
        _tmp.chmod(0o755)
        os.replace(_tmp, launch_sh)
        print(f"[launch-witness] readiness-defer recorded in launch.sh: "
              f"{', '.join(d.split('=')[0] for d in args.readiness_defer)}")
    try:
        _dsl_provenance_path, _launch_manifest_path = write_dsl_compile_artifacts(
            cfg, launch_sh, _dsl_document
        )
    except Exception as exc:
        print(
            "[launch-witness] ERROR: REFUSING to launch — rc=8 Catalog #406 exact "
            f"artifact recomputation failed: {type(exc).__name__}: {exc}. "
            "No launch/governor path may proceed without an exact DSL argv round-trip.",
            file=sys.stderr,
        )
        return 8
    run_log = out_dir / "run.log"
    print(f"# wrote {launch_sh}")
    print(f"# wrote {_dsl_provenance_path}")
    print(f"# wrote {_launch_manifest_path}")
    print(f"# dsl_compile_hash={_dsl_compile_hash}")

    # (b-perf) PERF-ENV CLASS GUARD (operator 2026-07-08 "shouldn't have had to be caught manually").
    # Assert the EMITTED launch.sh env block carries every REQUIRED_PERF_ENV var (the ~17x custom-
    # grouped-backward fast path). A config path that forgets the perf-env emission (the exact v7
    # orphan the compute audit fixed by hand) is now a STRUCTURAL REFUSE (rc=9) naming the missing
    # var — not an audit finding. SoT = the PERF_ENV_PREFIX constant (parsed, never a duplicate list).
    try:
        from tac.witness_dsl.typed_config import REQUIRED_PERF_ENV, missing_perf_env_vars
        _missing = missing_perf_env_vars(Path(launch_sh).read_text())
        if _missing:
            print(f"[launch-witness] ERROR: REFUSING to launch — the emitted launch.sh is MISSING "
                  f"required perf-env var(s) {_missing} (of {sorted(REQUIRED_PERF_ENV)}). The config's "
                  f"to_command dropped the ~17x custom-grouped-backward prefix; the run would be ~17x "
                  f"SLOW. Fix the config's to_command emission (see "
                  f"tac.witness_dsl.typed_config.PERF_ENV_PREFIX).", file=sys.stderr)
            return 9
        print(f"# perf-env guard: launch.sh carries {sorted(REQUIRED_PERF_ENV)} (~17x fast path emitted).")
    except Exception as exc:  # the guard's helper import must never wedge the launch (loud fail-open)
        print(f"[launch-witness] WARNING: perf-env class guard unavailable ({type(exc).__name__}: {exc}); "
              f"the post-spawn perf-env LOG check (step d) remains the backstop.", file=sys.stderr)

    # (b0, #351) WRITE constants_manifest.json beside launch.sh when the config carries LawRef-
    # compiled constants (crucible_v6). Provenance-only (value-identity: every value bit-matches the
    # sealed literal), so launch.sh is unchanged; the manifest records each constant's law + inputs +
    # artifact shas + ladder class. No file for non-crucible configs (byte-and-file-identical to before).
    manifest_path = write_constants_manifest(cfg, out_dir)
    if manifest_path is not None:
        print(f"# wrote {manifest_path} ({len(cfg.constants_manifest)} LawRef-compiled constants)")

    # (b0.5) SCHEDULE-PROVENANCE GATE (operator 2026-07-09, verbatim fury: "Fuck pr95... Never do it
    # again. Add a gate and hook... move from hardcoded epochs to event based and deep math governed
    # and costate controller"). Every emitted --*-start-epoch schedule TRIGGER carrying a POSITIVE
    # hardcoded epoch must be EVENT-TRIGGERED (a named co-emitted sensor declared in schedule_
    # governance), DERIVED (a LawRef value in constants_manifest.json), or a TAGGED fail-safe CAP —
    # else it is the PR95-skeleton regression the operator prohibits and the launch REFUSES (rc=6).
    # ADVISORY on --dry-run + under --skip-schedule-provenance-gate (prints the table, proceeds);
    # ENFORCING on a real launch. Fail-OPEN on infra error (a gate crash must never wedge the ONE
    # launch path); the deterministic classification of valid inputs does not raise.
    try:
        import schedule_provenance_gate as spg  # tools/ on sys.path (same dir as this launcher)

        sched_pairs = list(cfg.to_trainer_flags(str(out_dir)))
        sched_pairs += spg.extra_flag_pairs(extra_flags or [])
        _trainer_text = _TRAINER.read_text()
        sched_verdicts = spg.classify_launch(
            sched_pairs,
            registry=spg.schedule_when_flags(_trainer_text),
            manifest_keys=set(getattr(cfg, "constants_manifest", {}) or {}),
            governance=getattr(cfg, "schedule_governance", {}) or {},
            # (operator override 2026-07-08) surface the co-emitted --*-start-event WIRINGS as
            # EVENT_TRIGGERED transitions alongside their FAIL_SAFE_CAP backstops.
            event_registry=spg.event_start_flags(_trainer_text))
        sched_ok, sched_viol, sched_table = spg.gate_report(sched_verdicts)
        print(sched_table)
        if not sched_ok:
            if args.dry_run or args.skip_schedule_provenance_gate:
                why = "DRY-RUN advisory" if args.dry_run else "--skip-schedule-provenance-gate set"
                print(f"[launch-witness] WARNING ({why}): schedule-provenance gate has "
                      f"{len(sched_viol)} NAKED primary-epoch schedule trigger(s); proceeding.\n"
                      + spg.LEGAL_PATHS_MSG, file=sys.stderr)
            else:
                print(f"[launch-witness] ERROR: REFUSING to launch — {len(sched_viol)} NAKED "
                      f"primary-epoch schedule trigger(s) (operator 2026-07-09 no-hardcoded-epochs):",
                      file=sys.stderr)
                for v in sched_viol:
                    print(f"    {v.flag} {v.value}  — {v.detail}", file=sys.stderr)
                print(spg.LEGAL_PATHS_MSG, file=sys.stderr)
                return 6
    except Exception as exc:  # infra/import failure must never wedge the launcher (fail-open, loud)
        print(f"[launch-witness] WARNING: schedule-provenance gate unavailable "
              f"({type(exc).__name__}: {exc}); proceeding (no naked-epoch protection this launch).",
              file=sys.stderr)

    # (b0.6) LEGACY MANIFEST CHECK, now a fail-closed rc=8 subset of the stronger
    # exact artifact recomputation above.  The old WARN/override migration paths
    # are deliberately dead; keeping this call protects legacy API consumers.
    try:
        from tac.witness_dsl.typed_config import verify_launch_manifest as _verify_dsl_manifest

        _emitted_dsl_names = list(_emitted_flag_names(cfg, str(out_dir)))
        _dsl_ok, _dsl_detail = _verify_dsl_manifest(_dsl_manifest, _emitted_dsl_names)
        _action, _msg = dsl_config_gate_action(
            ok=_dsl_ok, detail=_dsl_detail, manifest_absent=not _dsl_manifest, config=config,
            dry_run=args.dry_run, skip=args.skip_dsl_config_gate,
            enforce=args.enforce_dsl_config_gate, allow_rationale=args.allow_non_dsl_config)
        if _action == "ok":
            print(f"# dsl-config gate: OK — {_dsl_detail}")
        else:  # "refuse"
            print(f"[launch-witness] ERROR: REFUSING to launch — {_msg}", file=sys.stderr)
            return 8
    except Exception as exc:
        print(f"[launch-witness] ERROR: REFUSING to launch — rc=8 legacy DSL manifest "
              f"recheck failed ({type(exc).__name__}: {exc}).", file=sys.stderr)
        return 8

    # (b1) MEMORY PREFLIGHT (#205 OOM self-protection). Project peak RSS from the EMITTED launch.sh
    # using MEASURED constants (2026-07-02 ledger) and REFUSE a config whose projected peak busts a
    # control-plane-safe fraction of RAM. The throughput gate measures COMPUTE at B=8; it NEVER
    # projected memory at the real n600 config, so the #205 OOM config passed it. This closes that
    # gap: it refuses e.g. --verdict-batch 0 at n600 (the ~66 GiB verdict spike) or n so large the
    # resident cf_mx_cache alone busts RAM. safe_run's --rss-cap-mb remains the runtime backstop.
    projected_peak_gib: float | None = None
    wmp = None
    try:
        import witness_memory_preflight as wmp  # tools/ is on sys.path (same dir as this launcher)

        proj = wmp.project_from_launch_sh(launch_sh, safe_frac=args.mem_preflight_safe_frac)
        projected_peak_gib = proj.projected_peak_gib
        if not args.dry_run:
            # BUILD #294 piece D: every gated launch appends its projection to the margin ledger
            # ({run_dir, projected_peak, config_hash, ts}); --reconcile closes the loop post-run.
            try:
                wmp.record_projection(out_dir, launch_sh, proj, note=f"launcher_b1:{config}")
            except Exception as exc:  # ledger telemetry must never block a launch
                print(f"[launch-witness] WARNING: projection ledger append failed ({exc})",
                      file=sys.stderr)
        print(f"# mem-preflight: projected peak {proj.projected_peak_gib} GiB "
              f"(fixed {proj.fixed_overhead_gib} + cf_mx_cache {proj.cf_cache_gib} + gt {proj.gt_gib} "
              f"+ verdict {proj.verdict_transient_gib}); safe ceiling {proj.safe_ceiling_gib} GiB "
              f"({proj.safe_frac:.0%} of {proj.total_ram_gib} GiB)")
        rss_cap_gib = args.rss_cap_mb / 1024.0
        if proj.projected_peak_gib > rss_cap_gib:
            print(f"# mem-preflight: NOTE projected peak {proj.projected_peak_gib} GiB > "
                  f"--rss-cap-mb {args.rss_cap_mb} MiB ({rss_cap_gib:.1f} GiB) — safe_run would kill it.")
        if not proj.safe:
            if args.skip_mem_preflight:
                print(f"[launch-witness] WARNING: mem-preflight would REFUSE ({proj.reason}) but "
                      f"--skip-mem-preflight set; proceeding.", file=sys.stderr)
            else:
                print(f"[launch-witness] ERROR: REFUSING to launch — {proj.reason} "
                      f"(pass --skip-mem-preflight to override, or reduce --num-pairs / raise "
                      f"--verdict-batch / free RAM).", file=sys.stderr)
                return 4
    except Exception as exc:  # projection must never crash a launch; WARN and continue.
        print(f"[launch-witness] WARNING: mem-preflight unavailable ({type(exc).__name__}: {exc}); "
              f"safe_run --rss-cap-mb {args.rss_cap_mb} remains the runtime backstop.", file=sys.stderr)

    # (b2) SAFE-COMPILE MANIFEST FRESHNESS (#252 v2 per-chip trust). When the launch.sh arms
    # --safe-compile-regions != none, the CERTIFICATE it will activate must have been measured on THIS
    # host (fp-contraction is per-chip). REFUSE (rc=4) a stale/absent manifest so a certificate carried
    # from another chip/macOS/MLX can never silently activate a compiled region. Byte-identical configs
    # (default 'none') skip this entirely. Advisory in --dry-run.
    try:
        _sc_text = Path(launch_sh).read_text()
        _sc_spec = _flag_value(_sc_text.split(), "--safe-compile-regions")
        if _sc_spec and _sc_spec.strip().lower() not in ("none", "off", ""):
            from tac.mlx_safe_compile import CertificationManifest, manifest_fingerprint_ok
            _sc_mpath = _flag_value(_sc_text.split(), "--safe-compile-manifest") or str(
                _REPO / ".omx" / "state" / "mlx_safe_compile_manifest.json")
            _sc_man = CertificationManifest.load(_sc_mpath) if os.path.exists(_sc_mpath) else None
            _fp_ok, _fp_reason = manifest_fingerprint_ok(_sc_man)
            if _sc_man is None:
                _fp_ok, _fp_reason = False, f"safe-compile manifest absent at {_sc_mpath} (recertify on this host)"
            print(f"# safe-compile: spec={_sc_spec!r} manifest={_sc_mpath} fingerprint_ok={_fp_ok} — {_fp_reason}")
            if not _fp_ok and not args.dry_run:
                print(f"[launch-witness] ERROR: REFUSING to launch — safe-compile {_fp_reason}. "
                      f"Recertify: python -m tac.mlx_safe_compile --certify --out {_sc_mpath}",
                      file=sys.stderr)
                return 4
    except Exception as exc:  # freshness check must never wedge a launch; WARN and continue.
        print(f"[launch-witness] WARNING: safe-compile freshness check unavailable "
              f"({type(exc).__name__}: {exc}); trainer resolve_enabled_regions remains the runtime "
              f"backstop (also fail-closed on stale fingerprint).", file=sys.stderr)

    # (b1-sys) SYSTEM ADMISSION HARD GATE — the SUM-over-RAM crash guard (the P0 fix). The per-run
    # projection above is blind to what ELSE is running; this composes THIS run's projected peak with
    # the live system-wide used RAM + all active jobs' remaining growth vs the adaptive ceiling. REFUSE
    # (rc=4) when the SUM would bust the machine — the exact multi-job overflow that crashed us. NOT a
    # dry-run advisory: it BLOCKS the launch. Bypass only via an operator-quoted override rationale.
    if projected_peak_gib is not None and wmp is not None:
        try:
            ctx = wmp.system_aware_admission(projected_peak_gib)
            d = ctx.decision
            print(f"# system-admission: {'ADMIT' if d.admit else 'REFUSE'} — {d.reason}")
            print(f"#   total={ctx.snapshot.total_gib:.0f} used={ctx.snapshot.used_gib:.1f} "
                  f"available={ctx.snapshot.available_gib:.1f} ceiling={ctx.ceiling.adaptive_ceiling_gib:.1f} "
                  f"budget={ctx.ceiling.training_budget_gib:.1f} active_jobs={len(ctx.active_jobs)} "
                  f"fail_safe={ctx.snapshot.fail_safe} GiB")
            if not d.admit and not args.dry_run:
                import system_memory_governor as _gov  # tools/ on sys.path
                if _admission_override_ok(args.admission_override_rationale):
                    print(f"[launch-witness] ADMISSION OVERRIDE (operator rationale): "
                          f"{args.admission_override_rationale!r} — proceeding despite: {d.reason}",
                          file=sys.stderr)
                elif _gov.admission_enforcing():
                    print(f"[launch-witness] ERROR: REFUSING to launch — SYSTEM admission [ENFORCE]: "
                          f"{d.reason}\n  Free RAM / wait for an active job to finish / reduce this run's "
                          f"peak, or pass --admission-override-rationale \"<operator verbatim>\".",
                          file=sys.stderr)
                    return 4
                else:
                    print(f"[launch-witness] WOULD-REFUSE (ADVISORY) — SYSTEM admission: {d.reason}\n"
                          f"  Gate ships ADVISORY pending independent adversarial review; PROCEEDING. "
                          f"Flip to enforce with {_gov.ADMISSION_ENFORCE_ENV}=1 after review.",
                          file=sys.stderr)
        except Exception as exc:  # governor unavailable => fail-open here; spawn's gate is the backstop.
            print(f"[launch-witness] WARNING: system admission unavailable ({type(exc).__name__}: {exc}); "
                  f"spawn_durable_daemon's admission gate remains the backstop.", file=sys.stderr)

    # (b1-ticket) CONFIG-DECLARED DEPENDENCY BLOCKERS.  A held research ticket may need the whole
    # zero-dollar chain (compile/flag/schedule/DSL/memory/fingerprint/governor) to materialize its
    # launch.sh while still being structurally forbidden from spawning.  The typed DSL manifest is
    # the single source: a non-empty ``launch_blockers`` list REFUSES every real launch and returns a
    # distinct rc=11 on --dry-run after all preceding $0 gates have run.  There is deliberately no
    # override flag; the config compiler must re-derive an empty list from landed dependencies.
    _ticket_blockers = list(
        (getattr(cfg, "dsl_program_manifest", None) or {}).get("launch_blockers") or []
    )
    if _ticket_blockers:
        print(f"[launch-witness] TICKET BLOCKED: {len(_ticket_blockers)} declared dependency blocker(s):",
              file=sys.stderr)
        for _blk in _ticket_blockers:
            if isinstance(_blk, dict):
                print(f"    {_blk.get('id', 'UNNAMED')}: {_blk.get('detail', '')}", file=sys.stderr)
            else:
                print(f"    {_blk}", file=sys.stderr)
        if args.dry_run:
            print("# DRY-RUN: launch.sh written and every preceding $0 gate completed; NOT spawning. "
                  "The ticket remains fail-closed until its typed manifest re-compiles with zero "
                  "launch_blockers.")
        else:
            print("[launch-witness] ERROR: REFUSING to launch — clear the typed ticket dependency "
                  "slots and recompile; no runtime override exists.", file=sys.stderr)
        return 11

    if args.dry_run:
        print("# DRY-RUN: launch.sh written + flags validated; NOT spawning. "
              "Re-run without --dry-run to launch.")
        return 0

    # (b1.5) OPTIONAL RSS CALIBRATION SMOKE (BUILD #294 piece B; default OFF — the default launch
    # path is unchanged). Runs the REAL flag set at small n FOREGROUND via the governed safe_run
    # path, measures actual peak RSS, and REFUSES (rc=5) if the measurement already busts the
    # projection by > --calibrate-overrun-pct.
    if args.calibrate_rss:
        if wmp is None:
            print("[launch-witness] ERROR: --calibrate-rss requires the memory preflight module "
                  "(unavailable above) — refusing the calibration-gated launch.", file=sys.stderr)
            return 5
        calib_rc = _run_rss_calibration(args, config, overfit, out_dir, label,
                                        extra_flags or None, wmp)
        if calib_rc != 0:
            return calib_rc

    # (b2) THROUGHPUT GATE (compute pass): a MEASURED SegNet fwd+bwd micro-bench (B=8), NOT a flag-grep.
    # REFUSE if the custom-grouped-backward ~17x fast path is not actually active on this machine
    # (median > threshold => the ~6713ms reference accumulator, not the ~396ms fast path). Unavailable
    # (no MLX/scorer/GPU) => WARN, never block. Sub-part 3: if a --compile* flag is emitted, the compiled
    # step must be bit-identical (asserted at trainer construction; the gate flags the requirement here).
    if not args.skip_throughput_gate:
        gate_rc = _run_throughput_gate(cfg, out_dir, threshold_ms=args.throughput_threshold_ms,
                                       accept_wall_clock_days=args.accept_wall_clock)
        if gate_rc != 0:
            return gate_rc

    # (b3) FULL-CONFIG DRY-START (owed-2 / SYNTHESIS §C item 2). When --dry-start N is set, the whole
    # gate chain above has already validated + memory-projected + admission-checked + throughput-benched
    # the REAL n600 config; this proves BOOT + STEP + CKPT + RESUME with a bounded <=3-epoch governed
    # run, then EXITS (never the unbounded spawn). Placed AFTER the throughput gate (so start-ability is
    # fully gated) and BEFORE the durable spawn (so it replaces, never precedes, a real launch).
    if args.dry_start:
        return _run_dry_start(args, config, overfit, out_dir, label, extra_flags or None, wmp,
                              projected_peak_gib=projected_peak_gib)

    # (c) LAUNCH durably (spawn_durable_daemon auto-verifies the child survived exec + auto-starts the
    # black box + re-checks the SYSTEM admission gate as a defense-in-depth backstop). We pass the
    # projected peak so the registry records it (the NEXT launch's admission gate sums it) + the
    # operator override rationale (so the daemon's gate honors the same override the launcher did).
    import spawn_durable_daemon as sdd  # late import: heavy-ish + only needed for real launch
    spawn_argv = [
        "--log", str(run_log), "--label", label,
        "--rss-cap-mb", str(int(args.rss_cap_mb)),
        "--min-free-gb", str(float(args.min_free_gb)),
        "--verify-s", str(float(args.verify_s)),
    ]
    if projected_peak_gib is not None:
        spawn_argv += ["--projected-peak-gib", str(round(float(projected_peak_gib), 3))]
    if _admission_override_ok(args.admission_override_rationale):
        spawn_argv += ["--admission-override-rationale", args.admission_override_rationale]
    spawn_argv += ["--", "bash", str(launch_sh)]
    rc = sdd.main(spawn_argv)
    if rc != 0:
        print(f"[launch-witness] ERROR: durable launch FAILED (spawn rc={rc}); see the "
              f"detailed debug above and the log: {run_log}", file=sys.stderr)
        return rc

    # (c.1) ACTIVATION LEDGER — record a "fired" event for every DSL lever this real launch used, so
    # "off" is a TRACKED queue the #247 costate SENSE drains (CLAUDE.md "'Off' is a tracked queue"; the
    # ledger is the anti-orphan surface). Only reached on a SUCCESSFUL spawn (not --dry-run, not a
    # refused/failed launch). NON-FATAL: a ledger write must never break a launch that already fired.
    if cfg.dsl_levers:
        # cfg.dsl_levers = the config's pre-composed levers (crucible_v6) + any --dsl-lever appends:
        # every lever this launch actually fires gets its ledger row (never only the CLI subset).
        try:
            from tac.witness_dsl.activation_ledger import EVENT_FIRED, record_activation
            for _lv in cfg.dsl_levers:
                # str(out_dir): record_activation json-serializes the row; a PosixPath raised
                # TypeError and silently dropped ALL fire records (caught live, v752 pilot launch
                # 2026-07-10 — backfilled by tools/… backfill call in the same landing).
                record_activation(_lv, EVENT_FIRED, run_ref=str(out_dir),
                                  reason=f"launched via tools/launch_witness_run.py (config={config})",
                                  agent="launch_witness_run")
            print(f"[launch-witness] activation-ledger: recorded 'fired' for {', '.join(cfg.dsl_levers)}")
        except Exception as exc:  # telemetry must never break a fired launch
            print(f"[launch-witness] WARNING: activation-ledger record failed "
                  f"({type(exc).__name__}: {exc}); launch already fired, continuing.", file=sys.stderr)

    # (c.2) SHADOW OBSERVER AUTO-START (#247 agent-native core sense organ; operator
    # NON-NEGOTIABLE 2026-07-07 "does not require human or manual activation"). Score-neutral
    # SENSE-only telemetry — read-only observability DEFAULTS ON per the "'Off' is a tracked
    # queue" rule. NON-FATAL: an observer failure must never break a launch that already fired.
    try:
        ensure_shadow_observer(out_dir)
    except Exception as exc:  # telemetry must never break a fired launch
        print(f"[launch-witness] WARNING: shadow-observer auto-start failed "
              f"({type(exc).__name__}: {exc}); launch already fired, continuing.", file=sys.stderr)

    # (d) VERIFY the perf-env fast path (loud warning on the silent-slow footgun).
    status, line = verify_perf_env(run_log, timeout_s=args.perf_env_timeout_s)
    if status == "active":
        print("[launch-witness] perf-env OK: custom_grouped_backward ACTIVE (~17x fast path).")
    elif status == "inactive":
        print(f"[launch-witness] WARNING: custom_grouped_backward is INACTIVE — the run will be "
              f"~17x SLOW (TAC_MLX_CUSTOM_GROUPED_BACKWARD unset/disabled). line: {line}",
              file=sys.stderr)
    else:
        print(f"[launch-witness] WARNING: could not confirm the custom_grouped_backward perf line "
              f"within {args.perf_env_timeout_s:.0f}s (run is alive; check {run_log}).",
              file=sys.stderr)

    # (d2, OI-5 amber realization 2026-07-10) STARTUP-TELEMETRY assertion for a stability-composed
    # config: the trainer's runtime-resolved stability row must hold the manifest's expected values —
    # a preset/flag set that compiles but is runtime-defeated is the P0-1 built-not-composed class
    # (the resolver's grad_clip explicit-wins docstring is NOT implemented; explicit composition is
    # what this verifies actually reached the trainer). Poll run.log like verify_perf_env. NON-FATAL
    # WARN (the launch already fired) but LOUD — the operator sees a defeated composition immediately.
    _exp_stab = (getattr(cfg, "dsl_program_manifest", None) or {}).get("expected_stability")
    if _exp_stab:
        _stab_row = None
        _deadline = time.monotonic() + float(args.perf_env_timeout_s)
        while time.monotonic() < _deadline:
            try:
                for _ln in Path(run_log).read_text(errors="replace").splitlines():
                    if '"witness_stability_resolved"' in _ln:
                        _stab_row = json.loads(_ln.strip())
                        break
            except OSError:
                pass
            if _stab_row is not None:
                break
            time.sleep(2.0)
        if _stab_row is None:
            print(f"[launch-witness] WARNING: expected_stability declared but NO "
                  f"witness_stability_resolved row within {args.perf_env_timeout_s:.0f}s — verify "
                  f"the amber values reached the trainer (grep {run_log}).", file=sys.stderr)
        else:
            _want = float(_exp_stab.get("grad_clip", -1.0))
            _got = float(_stab_row.get("grad_clip", _stab_row.get("out_grad_clip", -2.0)))
            if abs(_got - _want) < 1e-9:
                print(f"[launch-witness] stability OK: runtime-resolved grad_clip={_got} matches "
                      f"the composed amber expectation.")
            else:
                print(f"[launch-witness] WARNING: STABILITY COMPOSITION DEFEATED AT RUNTIME — "
                      f"resolved grad_clip={_got} != expected {_want} (row: {_stab_row}). The amber "
                      f"values did NOT reach the trainer (P0-1 built-not-composed class).",
                      file=sys.stderr)

    # (e) CONFIRM dashboard observability (auto-tracks this run once up).
    if not args.no_dashboard:
        ensure_dashboard(args.dashboard_port)

    print(f"[launch-witness] LAUNCHED + VERIFIED: label={label} log={run_log}")
    print(f"  stop:   .venv/bin/python tools/spawn_durable_daemon.py --stop {label}")
    print("  status: .venv/bin/python tools/spawn_durable_daemon.py --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
