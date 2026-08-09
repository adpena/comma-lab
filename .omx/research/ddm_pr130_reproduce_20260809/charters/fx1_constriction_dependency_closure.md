# ddm_fx1 — CURE 1: the decode path is not dependency-closed (`constriction`)

**Operator 2026-08-09: "Continue with all."** This is the review's headline finding and the only one
that blocks *evaluating anything derived from the PR130 base*. Fix it for real.

## THE FINDING (rr3, `09f667e2fa`, do not re-derive)

`RR3_HPAC_SHIPPING_AUDIT.md` MEASURED: the shipped decode entrypoint imports `constriction`
unconditionally at `code/inflate.py:13`, uses `RangeDecoder` at `:563-568`, and a clean bare venv
(`pip==26.1.2`, `ENABLE_USER_SITE=false`, `importlib.util.find_spec("constriction") is None`) running
the shipped `inflate.sh` **fails rc=1 in 0.03 s**. Denominator 1/1. Verdict: *not dependency-closed in
the current contest runtime tree* — a forward shipping blocker, NOT a retroactive rejection of PR130's
completed contest-CUDA row (their runtime had it).

## YOUR SCOPE

1. **Establish WHERE the cure lives.** The intake is READ-ONLY — you may not fix `inflate.py` in place.
   Determine what WE actually ship for a PR130-derived candidate: a derived runtime tree under our
   custody, or nothing yet. If nothing yet, that absence IS the first finding — say so, then build the
   minimal derived runtime tree that a candidate could ship, and cure it there.
2. **Apply the #666 e4 brotli precedent.** Read how `brotli` was declared as runtime-tree dep #2 with
   a fail-closed fallback on `ImportError`. Mirror the STRUCTURE, not the code: declare `constriction`,
   and make the failure path *loud and closed*, never a silent numeric fallback. An arithmetic decoder
   has no safe silent substitute — a wrong probability desynchronizes the whole stream.
3. **PROVE the bootstrap — the r5 lesson.** Never assume the host. Reproduce rr3's bare-venv harness,
   then show the CURED path passing it from a venv where `find_spec("constriction") is None` at start.
   Report install wall-clock, wheel availability for the contest runtime's platform/python, and whether
   a source build could be triggered (that is the 30-min-budget risk, not the import).
4. **Budget honesty.** `timeout-minutes: 30` is the WHOLE CI job (#835), and `evaluate.py:64` sums
   `rglob('*')` over `videos/` for the rate denominator (Catalog #812) — NOT the constant 37,545,489.
   Time the bootstrap inside that whole-job frame, not in isolation.
5. **Vendoring is a live alternative, price it.** CLAUDE.md L4 as amended states the PREFERENCE ORDER:
   vendored OSS inside inflate.py (zero install risk, rule-118-free) > fewer deps > more deps. Price
   both arms honestly: declared-dep+self-install vs vendoring the decoder path. Recommend one with the
   arithmetic shown. Do not vendor anything video-derived — rule-118 binds.

## OPTIMAL FORM

- **Reference form:** a cured derived runtime tree whose decode entrypoint passes a bare-venv
  dependency-closure smoke from a venv that provably lacks `constriction` at start, with measured
  install time inside the whole-job budget, plus the priced vendoring alternative.
- **SCOPE reductions (legal):** decode-entrypoint smoke rather than a full n600 decode (you have no
  Metal/CUDA); single platform measured with the contest platform named as the gap.
- **MECHANISM reductions (declare TOY-BRACKET):** answering by reading a requirements file instead of
  running the clean-env import; a fallback that silently degrades numerics; asserting wheel
  availability from PyPI metadata without attempting the install.
- **Provenance pins:** intake `e34f31bc4969042c0051ac81aa3c56884419a231`; rr3 audit + receipt at
  `.omx/research/ddm_pr130_reproduce_20260809/RR3_HPAC_SHIPPING_{AUDIT.md,RECEIPT.json}` (`09f667e2fa`).

## NON-NEGOTIABLES

- Intake READ-ONLY; never edit in place, never `git add` inside it. Upstream snapshot IMMUTABLE.
- No number without a locatable receipt. ABSENT is honest; restating is not.
- **Never consume a background job's output without asserting terminal status** — a partial read is
  INCOMPLETE, never a negative result. (MAIN published a false negative this way today.)
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py` (use `tools/review_tracker.py mark-file` ×2); fine for
  `.md`/`.json`.

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/FX1_DEPENDENCY_CLOSURE.md` — **§1 = the cured bare-venv
receipt** (or an explicit blocker naming what stopped you), then the declared-dep-vs-vendor pricing,
the whole-job budget arithmetic, ranked residual risks with falsifiers, and "could not check / why."
