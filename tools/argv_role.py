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
    (i.e. it is a genuine launch, not a monitor watching a launch)."""
    joined = " ".join(strip_observer_flag_values(cmdline))
    return any(tok in joined for tok in trainer_tokens)
