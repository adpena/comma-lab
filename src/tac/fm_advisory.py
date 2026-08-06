# SPDX-License-Identifier: MIT
"""fm_advisory — the on-device FM (fmtools) ADVISORY semantic-classification SENSE layer
for the #247 costate organ (task #522).

Operator 2026-07-17 (verbatim): "We can also bake fmtools into the costate organ or costate
controller as well" — as an ADVISORY sense layer, never the actuating/blocking decision.

This is the ONE shared adapter that wraps the established detached-subprocess fmtools
pattern (tools/triality_drift_detector.fm_scope_advisories · tools/dashboard_fm_events ·
tools/magnitude_dismissal_detector) into a generic ``classify`` API the costate organ
consumes at four insertion points:

  (a) REGIME supplement   — classify recent run-log/telemetry text into the λ-organ regime
                            labels; emit ``fm_regime`` + agreement-with-numeric flag.
  (b) EVENT-intelligence  — classify verdict-reason / lever-engage / confound-alarm rows
                            into event classes (annotation feeding the #344 context).
  (c) DUTY-ranking hint   — an FM relevance judgment of top-K never-fired duty rows vs the
                            current regime (a SECONDARY sort hint; P8 floor-aware order stays base).
  (d) CONFOUND-alarm class— classify new harness-failure-ledger rows into known classes.

═══ THE ADVISORY BOUNDARY (binding; CONTAINMENT unchanged) ══════════════════════════════
FM outputs are ADVISORY ONLY. They land as an explicit ``fm_advisory`` field / digest
section and NEVER feed actuation, a block, a verdict, a promotion, or a numeric score.
The deterministic/numeric layers remain the decision; the FM is a semantic second opinion.
Score-neutral + read-only by construction (it only READS text + emits labels).

HOME (CONTAINMENT): this module lives at the ``tac`` top level, NOT inside
``tac.witness_control`` — that package carries the structural "no actuation" invariant
(source-token scan forbids any ``subprocess``/spawn token, so the controller cannot launch/
stop/mutate a run). This adapter MUST spawn a classification subprocess, so it sits beside
the other FM consumers' layer (tools/) rather than in the controller package. Spawning an
on-device text CLASSIFIER is NOT run-actuation: it reads text and returns labels — never
touching the trainer, config, or score.

ISOLATION (the established pattern): FM inference NEVER runs in the pact venv. We spawn a
SUBPROCESS under the fmtools-venv interpreter (``fm_python()``: env override → DASH_FM_PYTHON
→ ~/Projects/fmtools/.venv/bin/python). The pact venv gains ZERO deps. fmtools absent /
model unavailable / timeout ⇒ every entry point GRACEFULLY DEGRADES to ``None`` (the layer
is simply ABSENT — never a value-shaped stub). A tiny in-process cache keyed by content
hash avoids re-classifying identical text within a session (no disk write ⇒ callers that
must-not-write, e.g. the digest, stay write-free).

HONESTY: labels ride a closed ``anyOf`` set; the rationale restates only words present in
the text; the FM never invents numbers. Every emitted row is provenance-labeled
``apple-fm-on-device · advisory · NON-PROMOTABLE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

# ─────────────────────────── closed label sets ───────────────────────────
#: (a) the λ-organ regime labels (operator-named, task #522). Closed ``anyOf`` set.
LAMBDA_ORGAN_REGIME_LABELS: tuple[str, ...] = (
    "lane-erosion",
    "mixed-Lane-Road",
    "movable-island-unborn",
    "OTHER/novel",
)

#: (b) event-intelligence classes for verdict-reason / lever-engage / alarm text.
EVENT_CLASS_LABELS: tuple[str, ...] = (
    "stage-transition",
    "lever-engage",
    "guard-alarm",
    "convergence",
    "regression",
    "info",
)

#: (c) duty-relevance labels (secondary sort hint only).
DUTY_RELEVANCE_LABELS: tuple[str, ...] = ("high", "medium", "low")

#: Charter-level work-shape labels for the codex arm queue lint. Advisory only.
CHARTER_CLASS_LABELS: tuple[str, ...] = (
    "build_race_train_measure",
    "audit_analysis",
    "convocation",
    "mixed",
)

#: Mechanism-reduction smells. Each is classified independently as present/absent.
MECHANISM_REDUCTION_LANGUAGE_LABELS: tuple[str, ...] = (
    "quick-train",
    "undersized",
    "toy-scale",
    "convenience-basis",
)

AUTHORITY = "on-device FM · advisory · read-only · NON-PROMOTABLE"

# in-process cache: content-hash → {"label", "rationale", "classifier"}. Bounded; NO disk
# write (keeps write-free callers write-free). Cleared per-process; ample for one session.
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 512

# The generic single-label classifier, run under the fmtools venv. Reads ONE job from
# stdin {labels, instructions, items:[{id,text}]} → prints {ok, results:[{id,label,rationale}]}.
# fmtools absent ⇒ {"ok": False}. Mirrors the triality/dashboard scripts (closed anyOf schema,
# per-item fault tolerance, model never invents a label outside the set).
_FM_CLASSIFY_SCRIPT = r'''
import asyncio, json, sys
try:
    import apple_fm_sdk as fm
    from fmtools import local_extract
except Exception:
    print(json.dumps({"ok": False, "reason": "fmtools-absent"})); raise SystemExit(0)

try:
    job = json.load(sys.stdin)
except Exception:
    print(json.dumps({"ok": False, "reason": "bad-input"})); raise SystemExit(0)
if not isinstance(job, dict):
    print(json.dumps({"ok": False, "reason": "bad-input"})); raise SystemExit(0)

labels = [str(x) for x in (job.get("labels") or [])]
instructions = str(job.get("instructions") or "")
items = job.get("items") or []
if not labels or not items:
    print(json.dumps({"ok": True, "results": []})); raise SystemExit(0)

@fm.generable()
class Choice:
    label: str = fm.guide(
        anyOf=labels,
        description="The single best-matching label from the allowed set.")
    rationale: str = fm.guide(
        description="One short clause using ONLY words/facts present in the text; "
                    "never invent numbers or outcomes.")

@local_extract(Choice, retries=1, instructions=instructions)
async def _classify(text: str) -> Choice:
    """(instructions provided above)"""

async def _main():
    out = []
    for it in items:
        _id = it.get("id") if isinstance(it, dict) else None
        txt = str((it.get("text") if isinstance(it, dict) else it) or "")[:1800]
        try:
            r = await _classify(txt)
            lab = str(getattr(r, "label", "") or "")
            if lab not in labels:
                lab = None
            out.append({"id": _id, "label": lab,
                        "rationale": str(getattr(r, "rationale", ""))[:200],
                        "classifier": "apple-fm-on-device"})
        except Exception as exc:
            out.append({"id": _id, "label": None,
                        "rationale": "unclassified (%s)" % type(exc).__name__,
                        "classifier": "apple-fm-skip"})
    print(json.dumps({"ok": True, "results": out}))

asyncio.run(_main())
'''


# ─────────────────────────── venv resolution + availability ───────────────────────────
def fm_python() -> str | None:
    """The fmtools-venv interpreter (env override → DASH_FM_PYTHON → known default). None
    when absent — every consumer then degrades to None (the layer is ABSENT, never a stub)."""
    for cand in (
        os.environ.get("FM_ADVISORY_PYTHON"),
        os.environ.get("DASH_FM_PYTHON"),
        os.path.expanduser("~/Projects/fmtools/.venv/bin/python"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def available() -> bool:
    """True iff the fmtools venv interpreter exists. Cheap (a path stat); the definitive
    gate every insertion point checks so the digest section renders ONLY when present."""
    return fm_python() is not None


# ─────────────────────────── prose framing ───────────────────────────
def prosify(obj: object) -> str:
    """Frame a telemetry object as a plain English sentence. Mostly-numeric raw JSON trips
    the FM language guardrail ('Unsupported language or locale'); a k-is-v sentence does not
    (the measured dashboard_fm_events lesson). Dicts → 'k is v; …'; else str()."""
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
            except Exception:
                return s[:400]
        else:
            return s[:400]
    if isinstance(obj, dict):
        body = "; ".join(f"{k} is {v}" for k, v in list(obj.items())[:12])
        return f"A training run logged: {body}"
    return str(obj)[:400]


# ─────────────────────────── cache ───────────────────────────
def _cache_key(labels: tuple[str, ...] | list[str], instructions: str, text: str) -> str:
    h = hashlib.sha256()
    h.update("\x1f".join(labels).encode("utf-8", "replace"))
    h.update(b"\x1e")
    h.update(instructions.encode("utf-8", "replace"))
    h.update(b"\x1e")
    h.update(text.encode("utf-8", "replace"))
    return h.hexdigest()


def _cache_put(key: str, row: dict) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # cheap bounded eviction: drop an arbitrary ~10% (dict insertion order = FIFO-ish).
        for k in list(_CACHE)[: _CACHE_MAX // 10 or 1]:
            _CACHE.pop(k, None)
    _CACHE[key] = {k: row.get(k) for k in ("label", "rationale", "classifier")}


# ─────────────────────────── subprocess runner (mockable) ───────────────────────────
def _run_job(fm_py: str, job: dict, timeout: int) -> dict | None:
    """Run ONE classify job under the fmtools venv. Returns the parsed payload dict, or
    None on any failure/timeout (fail-silent by construction). Tests monkeypatch this."""
    try:
        proc = subprocess.run(
            [fm_py, "-c", _FM_CLASSIFY_SCRIPT],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads((proc.stdout or "").strip() or "{}")
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


# ─────────────────────────── core classify API ───────────────────────────
def classify(
    items: list,
    labels: tuple[str, ...] | list[str],
    instructions: str,
    *,
    timeout: int = 25,
    cache: bool = True,
) -> list[dict] | None:
    """Classify each item's text into exactly one of ``labels`` (semantic, on-device FM).

    ``items``  — list of strings, or list of ``{"id": ..., "text": ...}`` dicts.
    Returns    — a list aligned to ``items``, each ``{"id","label","rationale","classifier"}``
                 (``label`` is None when the FM abstained / that item failed), OR **None** when
                 the fmtools venv is absent OR the subprocess failed entirely with no cached
                 rows (the caller then treats the whole layer as ABSENT). Never raises.
    """
    fm_py = fm_python()
    if not fm_py or not labels or not items:
        return None
    labels = tuple(str(x) for x in labels)

    norm: list[dict] = []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            norm.append({"id": it.get("id", i), "text": str(it.get("text", ""))})
        else:
            norm.append({"id": i, "text": str(it)})

    by_id: dict = {}
    to_run: list[tuple[dict, str]] = []
    for it in norm:
        key = _cache_key(labels, instructions, it["text"]) if cache else ""
        cached = _CACHE.get(key) if cache else None
        if cached is not None:
            by_id[it["id"]] = {"id": it["id"], **cached, "cached": True}
        else:
            to_run.append((it, key))

    if to_run:
        job = {
            "labels": list(labels),
            "instructions": instructions,
            "items": [{"id": it["id"], "text": it["text"]} for it, _ in to_run],
        }
        payload = _run_job(fm_py, job, timeout)
        if payload is None or not payload.get("ok"):
            # subprocess failed / fmtools absent at runtime: honor cached-only, else ABSENT.
            if not by_id:
                return None
        else:
            fresh = {r.get("id"): r for r in (payload.get("results") or []) if isinstance(r, dict)}
            for it, key in to_run:
                r = fresh.get(it["id"])
                if r is None:
                    by_id[it["id"]] = {
                        "id": it["id"], "label": None,
                        "rationale": "no-result", "classifier": "none",
                    }
                    continue
                if cache and r.get("classifier") == "apple-fm-on-device":
                    _cache_put(key, r)
                by_id[it["id"]] = {**r, "id": it["id"]}

    return [
        by_id.get(
            it["id"],
            {"id": it["id"], "label": None, "rationale": "no-result", "classifier": "none"},
        )
        for it in norm
    ]


def classify_one(
    text: str,
    labels: tuple[str, ...] | list[str],
    instructions: str,
    *,
    timeout: int = 25,
    cache: bool = True,
) -> dict | None:
    """Single-text convenience over ``classify``. None when unavailable / failed."""
    out = classify([text], labels, instructions, timeout=timeout, cache=cache)
    return out[0] if out else None


# ─────────────────────────── (a) REGIME supplement ───────────────────────────
_REGIME_INSTRUCTIONS = (
    "You label the CURRENT dynamical regime of a witness segmentation-training run from its "
    "recent telemetry text, on a closed set. 'lane-erosion' = the Lane class (class 1) "
    "boundary is degrading / flickering / losing flips. 'mixed-Lane-Road' = both the Lane and "
    "Road (class 0) boundaries are moving together. 'movable-island-unborn' = the movable "
    "(class 3) islands never nucleated / stay unborn. 'OTHER/novel' = none of these clearly "
    "fits. Use ONLY facts in the text; pick the closest single label."
)


def numeric_regime_hint(annulus_data: dict | None, classification: dict | None = None) -> str | None:
    """Derive a NUMERIC regime hint from the measured annulus per-class flip shares (Road=0,
    Lane=1, Movable=3), for the agreement-with-numeric flag. Conservative: returns None when
    the per-class flip shares are unavailable (agreement is then unknown, never guessed).

    Heuristic (MEASURED shares, advisory): Lane-dominant → 'lane-erosion'; Lane and Road both
    material → 'mixed-Lane-Road'; Movable share ~0 while others move → 'movable-island-unborn';
    else 'OTHER/novel'. Pure + unit-testable."""
    if not isinstance(annulus_data, dict):
        return None
    ann = annulus_data.get("annulus") if "annulus" in annulus_data else annulus_data
    if not isinstance(ann, dict):
        return None
    shares = ann.get("per_class_annulus_flip_frac")
    if not isinstance(shares, dict) or not shares:
        return None

    def _f(k: str) -> float:
        try:
            v = shares.get(k)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    road, lane, movable = _f("0"), _f("1"), _f("3")
    total = road + lane + movable
    if total <= 0.0:
        return None
    lane_r, road_r, mov_r = lane / total, road / total, movable / total
    if lane_r >= 0.4 and road_r >= 0.3:
        return "mixed-Lane-Road"
    if lane_r >= 0.45:
        return "lane-erosion"
    if mov_r <= 0.02 and (lane_r + road_r) >= 0.6:
        return "movable-island-unborn"
    return "OTHER/novel"


def regime_supplement(
    telemetry_texts: list,
    *,
    numeric_hint: str | None = None,
    timeout: int = 25,
) -> dict | None:
    """(a) Classify recent telemetry text into the λ-organ regime labels and compare against
    the numeric hint. Returns ``{"fm_regime","rationale","numeric_hint","agrees_with_numeric"}``
    (agreement is None when there is no numeric hint), or None when the FM is unavailable."""
    if not telemetry_texts:
        return None
    items = [{"id": i, "text": prosify(t)} for i, t in enumerate(telemetry_texts)]
    out = classify(items, LAMBDA_ORGAN_REGIME_LABELS, _REGIME_INSTRUCTIONS, timeout=timeout)
    if not out:
        return None
    # majority label over the window (most recent last); ignore abstentions.
    labels = [r.get("label") for r in out if r.get("label")]
    if not labels:
        return None
    fm_regime = max(set(labels), key=labels.count)
    rationale = next((r.get("rationale") for r in reversed(out) if r.get("label") == fm_regime), "")
    agrees = (fm_regime == numeric_hint) if numeric_hint else None
    return {
        "fm_regime": fm_regime,
        "rationale": rationale,
        "numeric_hint": numeric_hint,
        "agrees_with_numeric": agrees,
        "n_classified": len(labels),
        "authority": AUTHORITY,
    }


# ─────────────────────────── (b) EVENT-intelligence ───────────────────────────
_EVENT_INSTRUCTIONS = (
    "You label ONE event from a machine-learning training run into a closed set. "
    "'stage-transition' = a curriculum/optimizer stage boundary (CE→tau, l7, Muon switch, "
    "structured_init). 'lever-engage' = a lever/loss term turning on or its weight changing. "
    "'guard-alarm' = a guard/spike/stale/clip/OOM/refusal alarm. 'convergence' = a "
    "descent/plateau/best-so-far note. 'regression' = a metric rising / erosion / decoupling. "
    "'info' otherwise. Use ONLY facts in the text; never invent an outcome."
)


def classify_events(event_texts: list, *, timeout: int = 25) -> list[dict] | None:
    """(b) Classify verdict-reason / lever-engage / alarm event text into event classes.
    Returns a list of ``{"text","event_class","rationale"}`` (annotation feeding the #344
    detector context), or None when the FM is unavailable."""
    if not event_texts:
        return None
    items = [{"id": i, "text": prosify(t)} for i, t in enumerate(event_texts)]
    out = classify(items, EVENT_CLASS_LABELS, _EVENT_INSTRUCTIONS, timeout=timeout)
    if not out:
        return None
    rows: list[dict] = []
    for src, r in zip(event_texts, out, strict=False):
        rows.append({
            "text": (str(src)[:160]),
            "event_class": r.get("label"),
            "rationale": r.get("rationale"),
            "classifier": r.get("classifier"),
        })
    return rows


# ─────────────────────────── (c) DUTY-relevance hint ───────────────────────────
def duty_relevance(
    duty_rows: list[dict],
    regime_text: str,
    *,
    top_k: int = 6,
    timeout: int = 25,
) -> list[dict] | None:
    """(c) FM relevance judgment of the top-K never-fired duty levers vs the current regime —
    a SECONDARY sort HINT only (P8 floor-aware ranking stays the base order). Returns a list of
    ``{"lever","relevance","rationale"}`` or None when unavailable. Advisory; never reorders on
    its own."""
    if not duty_rows or not regime_text:
        return None
    top = duty_rows[: max(1, top_k)]
    instr = (
        "You judge how RELEVANT a training lever is to the run's CURRENT regime described as: "
        f"'{regime_text}'. 'high' = the lever directly targets this regime's failure; 'medium' "
        "= plausibly related; 'low' = unrelated. Use only the lever's name/description; a "
        "relevance hint is advisory and never a decision."
    )
    items = []
    for i, r in enumerate(top):
        name = str(r.get("lever") or r.get("candidate") or "?")
        desc = str(r.get("why") or r.get("description") or r.get("form_class") or "")
        items.append({"id": i, "text": f"lever '{name}': {desc}"[:400]})
    out = classify(items, DUTY_RELEVANCE_LABELS, instr, timeout=timeout)
    if not out:
        return None
    rows: list[dict] = []
    for r_in, r_out in zip(top, out, strict=False):
        rows.append({
            "lever": str(r_in.get("lever") or r_in.get("candidate") or "?"),
            "relevance": r_out.get("label"),
            "rationale": r_out.get("rationale"),
        })
    return rows


# ─────────────────────────── charter lint helpers ───────────────────────────
_CHARTER_CLASS_INSTRUCTIONS = (
    "You classify ONE codex-arm charter into a closed set. "
    "'build_race_train_measure' = the charter asks the arm to implement, build, train, race, "
    "compose, solve, measure, or otherwise produce a mechanism or measured artifact. "
    "'audit_analysis' = the charter asks only to inspect, review, classify, audit, summarize, "
    "or analyze existing artifacts without building a new mechanism. 'convocation' = the "
    "charter is a discussion/symposium/planning council with no direct implementation step. "
    "'mixed' = both implementation/measurement and audit/convocation are material. Use only "
    "facts in the text; pick the closest single label."
)


def charter_class(text: str, *, timeout: int = 25) -> dict | None:
    """Classify a charter's semantic work-shape for queue lint enrichment.

    Returns ``{"charter_class","rationale","classifier","authority"}``, or None when
    fmtools is unavailable / failed. Advisory only; deterministic lint gates stay primary.
    """
    if not str(text).strip():
        return None
    row = classify_one(
        str(text)[:1800],
        CHARTER_CLASS_LABELS,
        _CHARTER_CLASS_INSTRUCTIONS,
        timeout=timeout,
    )
    if not row or not row.get("label"):
        return None
    return {
        "charter_class": row.get("label"),
        "rationale": row.get("rationale"),
        "classifier": row.get("classifier"),
        "authority": AUTHORITY,
    }


_MECHANISM_REDUCTION_DESCRIPTIONS: dict[str, str] = {
    "quick-train": (
        "language that proposes a quick training shortcut, short-run proxy, smoke-only "
        "descent, or hurry-up training as if it can stand for the real mechanism"
    ),
    "undersized": (
        "language that knowingly shrinks capacity, basis size, epoch count, sample count, "
        "or representation size in a way that may change the mechanism rather than only scope"
    ),
    "toy-scale": (
        "language that treats a toy, tiny, n=small, smoke, or miniature mechanism verdict as "
        "if it can transfer to n600/production without preserving mechanism fidelity"
    ),
    "convenience-basis": (
        "language that chooses a basis, metric, coordinate system, or substrate because it is "
        "convenient/easy/default rather than because it is the optimal form for the mechanism"
    ),
}

_MECHANISM_REDUCTION_INSTRUCTIONS = (
    "For each item, decide whether the PASSAGE contains the TARGET mechanism-reduction smell. "
    "Return 'present' only when the passage itself uses or endorses the smell, not when it "
    "warns against it, requires optimal form, or discusses the smell as a forbidden class. "
    "Use 'absent' otherwise. Use only facts in the passage."
)


def mechanism_reduction_language(text: str, *, timeout: int = 25) -> dict | None:
    """Flag quick-train / undersized / toy-scale / convenience-basis language.

    The four smells are classified independently through the same closed-label fmtools
    subprocess surface. Returns a dict with ``flags`` (possibly empty) or None when fmtools
    is unavailable / failed. Advisory only; no gate consumes this as authority.
    """
    passage = str(text).strip()
    if not passage:
        return None
    passage = passage[:1600]
    items = [
        {
            "id": label,
            "text": (
                f"TARGET {label}: {description}. PASSAGE: {passage}"
            ),
        }
        for label, description in _MECHANISM_REDUCTION_DESCRIPTIONS.items()
    ]
    out = classify(
        items,
        ("present", "absent"),
        _MECHANISM_REDUCTION_INSTRUCTIONS,
        timeout=timeout,
    )
    if out is None:
        return None
    rows: list[dict] = []
    flags: list[str] = []
    classified = 0
    for r in out:
        label = str(r.get("id") or "")
        if r.get("label") in {"present", "absent"}:
            classified += 1
        present = r.get("label") == "present"
        if present and label in MECHANISM_REDUCTION_LANGUAGE_LABELS:
            flags.append(label)
        rows.append({
            "label": label,
            "present": present,
            "rationale": r.get("rationale"),
            "classifier": r.get("classifier"),
        })
    if classified == 0:
        return None
    return {
        "flags": flags,
        "rows": rows,
        "authority": AUTHORITY,
    }


# ─────────────────────────── (d) CONFOUND-alarm classing ───────────────────────────
def classify_confounds(
    failure_texts: list,
    known_classes: list[str],
    *,
    timeout: int = 25,
) -> list[dict] | None:
    """(d) Classify new harness-failure-ledger rows into known failure classes (composes with
    the bug-sweep Class-5 machinery). Returns ``{"text","matched_class","rationale"}`` rows, or
    None when unavailable. 'none' when no known class matches (the FM may not invent an id)."""
    if not failure_texts or not known_classes:
        return None
    labels = (*(str(c) for c in known_classes), "none")
    instr = (
        "You match ONE harness-failure description to exactly one KNOWN failure-class id from "
        "the allowed set, or 'none' if it matches no known class. Use only the text; never "
        "invent a class id."
    )
    items = [{"id": i, "text": prosify(t)} for i, t in enumerate(failure_texts)]
    out = classify(items, labels, instr, timeout=timeout)
    if not out:
        return None
    rows: list[dict] = []
    for src, r in zip(failure_texts, out, strict=False):
        cls = r.get("label")
        if cls == "none":
            cls = None
        rows.append({
            "text": str(src)[:160],
            "matched_class": cls,
            "rationale": r.get("rationale"),
        })
    return rows


# ─────────────────────────── shadow-row bundle (insertion (a)+(b)) ───────────────────────────
def shadow_advisory(
    *,
    telemetry_texts: list,
    event_texts: list,
    annulus_data: dict | None = None,
    classification: dict | None = None,
    timeout: int = 25,
) -> dict | None:
    """The bundle that populates the costate shadow-observer row's additive ``fm_advisory``
    field: (a) regime supplement + (b) event intelligence. Returns None when the FM is
    unavailable (⇒ the row omits the field ⇒ byte-identical schema when absent). Advisory only."""
    if not available():
        return None
    hint = numeric_regime_hint(annulus_data, classification)
    regime = regime_supplement(telemetry_texts, numeric_hint=hint, timeout=timeout)
    events = classify_events(event_texts, timeout=timeout)
    if regime is None and not events:
        return None
    return {
        "regime": regime,
        "events": events or [],
        "authority": AUTHORITY,
    }
