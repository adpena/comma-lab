# ddm_fx5 — close FX1's named gap: the Linux x86_64 leg of the decode dependency

**Operator 2026-08-09: "Continue with all."** FX1 cured the constriction blocker and then labelled
its own platform statement a **TOY-BRACKET**. This arm converts that bracket into a receipt — or
reports honestly that it could not.

## THE STATE (FX1, `3e616f568a`, do not re-derive)

Derived runtime tree at `src/tac/pr130_runtime/fx1_runtime_tree` — five receiver modules **byte-for-byte**
from intake `e34f31bc4969042c0051ac81aa3c56884419a231` (0/5 changed), `constriction==0.5.0` declared and
pinned, wheel-only (`--only-binary`) fail-closed self-install into an isolated target with version+API
verification before atomic publish, nonzero exit on every closure failure, **no numeric fallback**.

MEASURED on macOS arm64 / CPython 3.11.15: from a venv with `ENABLE_USER_SITE=False` and
`find_spec("constriction") is None`, the cured `inflate.sh` installed, imported the real `inflate.py`,
printed `PR130_DEPENDENCY_READY constriction=0.5.0 receiver=inflate.py`, rc=0. Cold 17.50 s
(uv 111 ms resolve / 223 ms install); warm repeat 0.51 s; host venv still `find_spec is None` afterward.
Invalid-target positive control rc=65, 1/1. Target manifest sha
`bfe25761e26f32b1dca1f7114a45648fa9b25dc8f98f0fa2e98b199992fd4a4b`, 10 files / 1,931,498 B.

**The gap, in FX1's own words:** *"Linux wheel execution is not measured here. This platform statement
is a TOY-BRACKET, not a contest-host closure receipt."* The install used a locally retained wheel cache
because the sandbox cannot resolve PyPI DNS; the live-network trial terminated rc=2 in 6.61 s **without
publishing a partial target** (correct fail-closed behavior, but not a success receipt).

## YOUR SCOPE

1. **Get the Linux x86_64 receipt.** PyPI 0.5.0 lists a CPython 3.11 `manylinux_2_17_x86_64` wheel
   (displayed 410.4 kB, sha256 `eb7909d0ad4940d3b74696d98f0dc16dec7294e57f9e0797bc06d5ce7b3b1507`) and
   **no sdist**. Run the cured entrypoint on a real Linux x86_64 CPython 3.11 host from a venv that
   provably lacks `constriction` at start, with **network fetch** — the untested half. Measure: resolve
   time, download time, install time, import, and the receiver's `PR130_DEPENDENCY_READY` line.
2. **This is the compute-split exception, and it binds you.** Local Metal cannot execute a Linux
   x86_64 wheel — that is *physically impossible locally*, which is exactly and only when Modal is
   authorized, and it must be **SHORT**. CPU container, seconds-to-minutes, no GPU. Honor Modal
   single-flight + the dual-ledger + the ≤$20 cumulative envelope (#381); record the dispatch in both
   ledgers; if the envelope or single-flight refuses, that refusal is information — report it, do not
   route around it.
3. **Measure the whole-job frame, not the step.** `timeout-minutes: 30` is the entire CI job (#835).
   FX1's 159.598 s cross-host projection is explicitly labelled *budget context, not a measured total*
   — it double-counts receiver startup and mixes macOS with an RTX 5070. Produce a single-host Linux
   number, or state plainly that you produced only the dependency-bootstrap segment.
4. **Re-check the rate denominator while you are on a real host.** `evaluate.py:64` sums `rglob('*')`
   over `videos/` (Catalog #812) — NOT the constant 37,545,489. FX1 correctly refused to reuse the
   constant. Confirm what the denominator actually is on the contest-shaped tree and record it.
5. **If and only if the Linux leg passes, re-open the vendor comparison with real bytes.** FX1 ranked
   vendoring *structurally preferable but unexecuted* and refused to call it better on that basis. With
   the wheel under custody the arm becomes executable: price extraction/import glue and decode time
   against the declared-dep path. Do not adopt on preference — race it.

## OPTIMAL FORM

- **Reference form:** the cured entrypoint executing on Linux x86_64 CPython 3.11 from a
  provably-clean venv **over the network**, with the full timing breakdown and the receiver import
  proven, plus the measured rate denominator from that host.
- **SCOPE reductions (legal):** dependency-bootstrap + receiver-import only, without a full n600
  decode — state the boundary. A CPU container instead of the exact contest runner image, with the
  image delta named.
- **MECHANISM reductions (declare TOY-BRACKET):** a cached/pre-seeded wheel on the Linux host (that
  re-creates FX1's exact gap); asserting wheel compatibility from PyPI metadata alone; reporting a
  step time as if it were the whole-job total; allowing a source build (`--only-binary` must hold).
- **Provenance pins:** FX1 `3e616f568a`; target manifest sha `bfe25761e26f32b1dca1f7114a45648fa9b25dc8f98f0fa2e98b199992fd4a4b`;
  wheel sha256 `eb7909d0ad4940d3b74696d98f0dc16dec7294e57f9e0797bc06d5ce7b3b1507`; intake `e34f31bc…`.

## NON-NEGOTIABLES

- Intake READ-ONLY; upstream snapshot IMMUTABLE. Receiver modules stay byte-identical to intake.
- **Modal ONLY because a Linux wheel physically cannot execute locally, and SHORT.** Governed path,
  single-flight, dual-ledger, ≤$20 envelope. A governor REFUSE is information, not an obstacle.
- No numeric fallback on the decode path — a differing probability desynchronizes the whole stream.
- **Never consume a background job's output without asserting terminal status** (MAIN published a
  false negative this way today; FX4 caught an already-complete Modal job by checking).
- No number without a locatable receipt. ABSENT is honest; restating is not.
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py` (use `tools/review_tracker.py mark-file` ×2).

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/FX5_LINUX_CLOSURE.md` — **§1 = the Linux x86_64 network
bootstrap receipt** (or the exact blocker that stopped it, with the governed-path refusal quoted),
then the whole-job budget number with its scope stated, the measured rate denominator, the
declared-vs-vendored race if it became executable, ranked residuals with falsifiers, and
"could not check / why."
