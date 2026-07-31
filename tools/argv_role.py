"""Observer-vs-launcher argv discrimination for process-classification guards (CLASS 2 fix, 2026-07-17).

WHY (sisters ``tunnel_always_up_supervisor_canonical_20260717`` #406 + the p0_512 same-outdir guard):
fail-closed guards that classify a live process by TOKEN PRESENCE in its joined cmdline misfire on
OBSERVERS/monitors that carry the trainer NAME as a flag VALUE (e.g. the dashboard supervisor's
``--training-sig train_levelset_witness``). The #406 admission gate was the first casualty (rc=8
blocked the always-on tunnel). The cure is STRUCTURAL, not token-deletion: strip the known observer
flag/value pairs BEFORE joining, so a monitor that merely NAMES a trainer is not classified as
launching one. Raw trainer argv (the thing we actually want to refuse) is untouched — verified both
directions. Deterministic; no psutil / FM dependency (FM role-classification, if used, is an advisory
tiebreaker layered on TOP of this, never a replacement — per the triality-detector pattern)."""

from __future__ import annotations

# Flags whose VALUE legitimately carries a trainer name because the process only WATCHES a trainer.
# ``--training-sig <name>`` is the canonical one (dashboard supervisor / chain watchdog / checkin).
OBSERVER_VALUE_FLAGS: tuple[str, ...] = ("--training-sig",)


def strip_observer_flag_values(
    cmdline: list[str] | tuple[str, ...] | None,
    observer_flags: tuple[str, ...] = OBSERVER_VALUE_FLAGS,
) -> list[str]:
    """Return ``cmdline`` with every ``<flag> <value>`` and ``<flag>=<value>`` observer pair removed.

    The result is what a trainer-token classifier should scan: a process that carries
    ``--training-sig train_levelset_witness`` (an OBSERVER) yields tokens with no trainer name, so it
    is not misclassified as a launch. A raw trainer argv keeps its trainer tokens and is still caught.
    """
    if not cmdline:
        return []
    out: list[str] = []
    skip_next = False
    for token in cmdline:
        text = str(token)
        if skip_next:
            skip_next = False
            continue
        if text in observer_flags:
            skip_next = True
            continue
        if any(text.startswith(f"{flag}=") for flag in observer_flags):
            continue
        out.append(text)
    return out


def is_observer_stripped_launch(
    cmdline: list[str] | tuple[str, ...] | None,
    trainer_tokens: tuple[str, ...] = ("train_levelset_witness", "train_witness"),
) -> bool:
    """True iff, after stripping observer flag values, the cmdline still names a trainer entrypoint
    (i.e. it is a genuine launch, not a monitor watching a launch).

    NOTE (ddm_gh1 #829): this is the LEGACY joined-substring form. It kills the observer-flag
    false positive only. For a guard that REFUSES A LAUNCH, prefer
    :func:`cmdline_names_entrypoint`, which additionally kills the reader/viewer false-positive
    class (a background ``grep``/``rg``/editor whose argv merely CONTAINS the token) and the
    unrelated-path-argument class (a ``--out .../train_levelset_witness_run/x.jsonl``).
    """
    joined = " ".join(strip_observer_flag_values(cmdline))
    return any(tok in joined for tok in trainer_tokens)


# ---------------------------------------------------------------------------
# ddm_gh1 #829 — precise entrypoint matching for slot-holder / refuse guards.
# ---------------------------------------------------------------------------
# Programs that READ, SEARCH, or DISPLAY source rather than run it. A process whose PROGRAM is one
# of these can never be holding a scorer slot, no matter what its arguments say. This is the leg
# that kills the measured false-refusal: an unrelated background `grep -rn train_levelset_witness`
# (or an agent's `rg`, an editor, a pager) made `tok in line` fire and the tool refused to launch.
READER_PROGRAMS: tuple[str, ...] = (
    "ack", "ag", "awk", "bat", "cat", "code", "cut", "diff", "du", "egrep", "emacs", "fgrep",
    "find", "fzf", "grep", "head", "less", "ls", "mdfind", "more", "nano", "nvim", "open",
    "pgrep", "ps", "rg", "ripgrep", "sed", "sort", "strings", "tail", "tee", "tr", "uniq",
    "vi", "view", "vim", "wc", "xargs",
)

# Characters that fence a path inside a quoted/embedded code string (`python -c "run('a/b.py')"`).
_PATH_FENCE = "\"'()[]{},;:`\\"


def _program_basename(pieces: list[str]) -> str:
    """Basename of the first piece that is not a leading ``VAR=value`` environment assignment."""
    for piece in pieces:
        if "=" in piece and not piece.startswith("-") and "/" not in piece.split("=")[0]:
            continue  # leading env assignment (e.g. `OMP_NUM_THREADS=1 python ...`)
        return piece.rsplit("/", 1)[-1]
    return ""


def _path_shaped_basenames(pieces: list[str]) -> list[str]:
    """Basenames of every piece that looks like a filesystem path (so a bare search PATTERN or a
    numeric flag value can never match a token)."""
    out: list[str] = []
    for piece in pieces:
        text = piece.strip(_PATH_FENCE)
        if not text or text.startswith("-"):
            continue
        if "/" not in text and not text.endswith(".py"):
            continue
        out.append(text.rsplit("/", 1)[-1].strip(_PATH_FENCE))
    return out


def cmdline_names_entrypoint(
    cmdline: str | list[str] | tuple[str, ...] | None,
    tokens: tuple[str, ...] | list[str],
    *,
    observer_flags: tuple[str, ...] = OBSERVER_VALUE_FLAGS,
) -> bool:
    """True iff this process actually RUNS an entrypoint named by one of ``tokens``.

    Precise where ``token in joined_cmdline`` is not. A token matches only when it appears inside
    the BASENAME of a PATH-SHAPED argument of a process whose program is not a reader/viewer:

    * ``python tools/sb1_seg_batch.py --pairs 4``            -> True  (basename ``sb1_seg_batch.py``)
    * ``grep -rn sb1_seg_batch tools/``                      -> False (program is a reader)
    * ``python x.py --out /d/sb1_seg_batch_run/t.jsonl``     -> False (basename is ``t.jsonl``)
    * ``python watch.py --training-sig train_levelset_witness`` -> False (observer flag value)

    Accepts a raw ``ps -axo command`` line (str) or an argv sequence. Whitespace inside an
    embedded code string (``python -c "...tools/pb1_qdbs.py..."``) is split too, so a genuinely
    live job wrapped in ``bash -c`` is still caught.
    """
    if not cmdline:
        return False
    argv = cmdline.split() if isinstance(cmdline, str) else [str(part) for part in cmdline]
    pieces: list[str] = []
    for part in strip_observer_flag_values(argv, observer_flags):
        pieces.extend(part.split())
    if not pieces:
        return False
    if _program_basename(pieces) in READER_PROGRAMS:
        return False
    basenames = _path_shaped_basenames(pieces)
    return any(token in name for token in tokens for name in basenames)


def process_table_entrypoint_holders(
    ps_output: str,
    tokens: tuple[str, ...] | list[str],
    *,
    self_tokens: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Lines of ``ps -axo command`` output that actually RUN one of ``tokens``.

    ``self_tokens`` are matched by the SAME precise rule, so a tool never sees itself — and, just
    as importantly, never excludes an unrelated process that merely mentions its own name.
    Returns the offending command lines so a refusal can name what it is waiting on instead of
    reporting a bare boolean.
    """
    holders: list[str] = []
    for line in ps_output.splitlines():
        if not line.strip():
            continue
        if self_tokens and cmdline_names_entrypoint(line, tuple(self_tokens)):
            continue
        if cmdline_names_entrypoint(line, tuple(tokens)):
            holders.append(line.strip())
    return holders
