"""Provider error classifier + pre-dispatch lexical-compatibility lint — A4 of the
harness-engineering crosswalk (``.omx/research/harness_engineering_crosswalk_20260719_codex.md``).

CLOSES ledger class ``provider_content_filter_false_positive_kills_arm`` (row 63, prevention
owed: "arm contracts should avoid security-scanner phrasing in generated spec prose … consider
a phrasing lint in the delegate contract"). TWO surfaces:

1. ``classify_provider_error(text)`` — turn a raw provider error STRING into a typed cause so a
   provider-side refusal is NOT misattributed to the arm's own work. The crosswalk's two live
   examples MUST classify as provider faults, not arm failures:
     * "The 'sol' model is not supported"                → ``provider_model``
     * "You've hit your usage limit … try again at Jul 24th" → ``provider_quota``
   plus the incident's own cause, a cyber/security classifier false-positive → ``provider_content_filter``.

2. ``lint_provider_trigger_phrasing(text)`` — a NARROW pre-dispatch lexical fixture over
   generated spec/prompt prose. It detects ONLY known provider-trigger phrasings (the
   security-scanner vocabulary that tripped the codex cyber classifier: "destructive-operation
   scan", "strict-prefix … refuses", etc.), points at neutral equivalent wording, and PRESERVES
   the substantive constraint. It is advisory — an explicit reviewed exception is allowed via a
   caller-side allowlist; it never rewrites the spec itself.

Both are pure functions (no I/O), so they compose into the delegate contract without state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "PROVIDER_ERROR_CLASSES",
    "PhrasingFinding",
    "ProviderErrorVerdict",
    "classify_provider_error",
    "is_provider_fault",
    "lint_provider_trigger_phrasing",
]

#: Typed causes. The three ``provider_*`` classes are provider-side faults (NOT the arm's
#: fault — do not mark the arm failed / do not reset its review counters for these).
PROVIDER_ERROR_CLASSES = (
    "provider_quota",           # usage/rate/quota limit hit
    "provider_model",           # requested model unsupported/unknown/decommissioned
    "provider_content_filter",  # a safety/cyber classifier refused the request
    "arm_failure",              # a genuine failure inside the arm's own work
    "unknown",                  # no confident classification
)

_PROVIDER_FAULTS = frozenset({"provider_quota", "provider_model", "provider_content_filter"})


@dataclass(frozen=True)
class ProviderErrorVerdict:
    error_class: str
    matched: str          # the substring that drove the classification (evidence)
    is_provider_fault: bool

    def to_dict(self) -> dict:
        return {"error_class": self.error_class, "matched": self.matched,
                "is_provider_fault": self.is_provider_fault}


# Ordered rules: (error_class, compiled pattern). First match wins. Quota/model/content-filter
# are checked BEFORE any arm-failure signal so a provider refusal is never misattributed.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("provider_quota", re.compile(
        r"usage limit|quota|rate limit|rate[- ]?limited|too many requests|"
        r"try again (?:at|in|after)|resets? at|429\b|insufficient_quota|"
        r"exceeded your current", re.I)),
    ("provider_model", re.compile(
        r"model is not supported|not a (?:valid|supported) model|unknown model|"
        r"model .*(?:unsupported|decommissioned|deprecated|retired)|"
        r"no such model|model_not_found|does not support the model", re.I)),
    ("provider_content_filter", re.compile(
        r"content (?:policy|filter)|safety (?:filter|classifier|system)|"
        r"cyber(?:security)? classifier|flagged (?:by|as)|"
        r"refus(?:ed|al) .*(?:policy|safety|classifier)|"
        r"violat(?:es|ion) .*(?:policy|guidelines)|blocked by .*(?:filter|classifier)|"
        r"cannot assist with", re.I)),
    ("arm_failure", re.compile(
        r"traceback|assertion(?:error)?|test(?:s)? failed|exit code [1-9]|"
        r"rc=[1-9]|non-?zero exit|compile error|syntax ?error", re.I)),
)


def classify_provider_error(text: str) -> ProviderErrorVerdict:
    """Classify a raw provider error string into a typed cause (see PROVIDER_ERROR_CLASSES).

    Provider-side rules are evaluated FIRST so a quota/model/content-filter refusal is never
    misattributed to the arm. Empty/None-ish input → ``unknown``."""
    s = (text or "").strip()
    if not s:
        return ProviderErrorVerdict("unknown", "", False)
    for cls, pat in _RULES:
        m = pat.search(s)
        if m:
            return ProviderErrorVerdict(cls, m.group(0), cls in _PROVIDER_FAULTS)
    return ProviderErrorVerdict("unknown", "", False)


def is_provider_fault(text: str) -> bool:
    """True when the error is a provider-side fault (quota / model / content-filter)."""
    return classify_provider_error(text).is_provider_fault


@dataclass(frozen=True)
class PhrasingFinding:
    trigger: str          # the risky phrase matched
    matched: str          # the exact substring in the input
    suggestion: str       # neutral equivalent wording that preserves the constraint

    def to_dict(self) -> dict:
        return {"trigger": self.trigger, "matched": self.matched, "suggestion": self.suggestion}


# Narrow, KNOWN provider-trigger phrasings → neutral equivalents that preserve the substantive
# intent. Derived from the 2026-07-18 incident (codex cyber classifier tripped on
# "destructive-operation scan" / "strict-prefix … refuses" security-scan spec prose) and its
# sister reasoning-echo refusal-storm class. Each entry is (label, pattern, neutral suggestion).
_TRIGGER_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("destructive-operation scan",
     re.compile(r"destructive[- ]operation scan", re.I),
     "rephrase as 'audit for irreversible file operations' — describe the property, "
     "not a security scan"),
    ("security scanner / scan",
     re.compile(r"\b(?:security|vulnerability|malware|exploit|cyber)[- ]?(?:scan|scanner|scanning)\b", re.I),
     "rephrase as 'static check for <specific property>' — name the concrete check, "
     "not a scanner"),
    ("strict-prefix … refuses",
     re.compile(r"strict[- ]prefix[^.\n]{0,40}refus", re.I),
     "rephrase as 'the guard returns a non-zero code on a disallowed prefix' — describe "
     "the return behavior, not a refusal"),
    ("refuses / refusal (of a guard)",
     re.compile(r"\b(?:gate|guard|check|preflight)[^.\n]{0,30}\brefus(?:e|es|al)\b", re.I),
     "rephrase 'refuses' as 'returns a non-zero exit / raises' — behavioral, not adversarial"),
    ("payload / attack surface",
     re.compile(r"\b(?:attack surface|malicious payload|weaponiz)", re.I),
     "rephrase as 'input surface' / 'invalid input' — describe the data, not an attack"),
)


def lint_provider_trigger_phrasing(
    text: str, *, allow: tuple[str, ...] = (),
) -> list[PhrasingFinding]:
    """Scan generated spec/prompt prose for KNOWN provider-trigger phrasings.

    Returns a list of ``PhrasingFinding`` (empty = clean). Each finding names the risky phrase,
    the exact match, and a neutral suggestion that PRESERVES the substantive constraint. Pass
    ``allow`` (a tuple of trigger labels) to record an explicit reviewed exception — allowlisted
    triggers are not reported. Advisory only: this never edits the spec."""
    s = text or ""
    findings: list[PhrasingFinding] = []
    for label, pat, suggestion in _TRIGGER_RULES:
        if label in allow:
            continue
        m = pat.search(s)
        if m:
            findings.append(PhrasingFinding(trigger=label, matched=m.group(0),
                                            suggestion=suggestion))
    return findings
