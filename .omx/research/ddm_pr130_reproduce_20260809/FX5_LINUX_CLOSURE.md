# FX5 — the Linux x86_64 leg of the PR130 decode dependency closure

**Charter:** `charters/fx5_linux_wheel_closure.md`. **Executed by MAIN** (the arm's sandbox could not
reach Modal — see §6). Two Modal CPU dispatches, both short, both CPU-only, no GPU.
Compute-split compliance: a Linux x86_64 wheel **physically cannot execute on local Metal**, which is
the one condition that authorizes Modal.

Base unchanged: **PR130 CPR1 S = 0.172141297491896447** `[contest-CUDA, DALI GT, n600]`, archive
191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
Nothing in this arm is a score claim. `score_claim=false` on every row.

---

## §1 — THE LINUX RECEIPT (the TOY-BRACKET FX1 flagged is now closed)

Two runs, same mounted tree, same entrypoint, different image shape. Receipts:
`FX5_LINUX_RECEIPT.json` (bare) · `FX5_LINUX_RECEIPT_CONTEST_SHAPED.json` (contest-shaped).

**Host, both runs (MEASURED):** Linux · x86_64 · CPython 3.11.12 · glibc 2.36.
**Precondition, both runs:** `importlib.util.find_spec("constriction") is None` → **True**. The premise
held; this was a genuinely clean venv for the dependency under test.
**Custody, both runs:** all 5 receiver modules hashed at the remote and compared to FX1's manifest —
`receiver_sha256_matches_manifest: true`, 5/5 byte-identical to intake `e34f31bc…`. We measured OUR
cured tree, not a lookalike.

### Run B — contest-shaped image (numpy + torch 2.10.0+cpu present, constriction absent)

```
COLD  rc=0  5.302 s   PR130_DEPENDENCY_READY constriction=0.5.0 receiver=inflate.py
      uv:  Resolved 1 package in 49 ms | Prepared in 57 ms | Installed in 18 ms | + constriction==0.5.0
WARM  rc=0  3.010 s   PR130_DEPENDENCY_READY constriction=0.5.0 receiver=inflate.py
TARGET      10 files / 1,102,335 B  (constriction/constriction.cpython-311-x86_64-linux-gnu.so + dist-info)
CONTROL     pre-existing invalid target → rc=65, refused, did not overwrite  (1/1, PASS)
```

**This is the receipt FX1 could not produce.** Its install used a locally retained wheel cache because
that sandbox cannot resolve PyPI DNS; the live-network trial terminated rc=2 in 6.61 s. Here the wheel
was **fetched over the network** on a real Linux x86_64 CPython 3.11 host, the manylinux `.so` landed,
the version+API verification passed, and the real `inflate.py` imported and bound the verified module.
Cold-vs-warm delta ≈ 2.3 s ⇒ the fetch+install itself is ~2 s of the 5.3 s; the rest is interpreter
startup and the verification subprocesses.

`--only-binary constriction` held throughout — no source build was triggered, which was the actual
30-min-budget risk (not the import).

---

## §2 — WHAT THE BARE RUN FOUND: FX1's closure statement is INCOMPLETE

Run A used a bare `debian_slim` image — uv only, **no numpy, no torch**. It failed:

```
COLD  rc=1  4.779 s
      uv:  Resolved 1 package in 199 ms | Prepared in 25 ms | Installed in 10 ms | + constriction==0.5.0
      Traceback … File "/runtime/fx1_runtime_tree/inflate.py", line 14, in <module>
          import numpy as np
      ModuleNotFoundError: No module named 'numpy'
WARM  rc=1  0.290 s   (constriction now present → install skipped → same import failure)
```

**The constriction bootstrap SUCCEEDED even here** (wheel resolved, downloaded, installed, verified).
The failure is one step later, at receiver import, on a *different* dependency.

The receiver's true third-party closure, enumerated from the source (denominator: all 5 modules):

| package | imported by | in FX1's `runtime-dependencies.json`? |
|---|---|---|
| `constriction` | `inflate.py:13` | **yes** |
| `numpy` | `inflate.py:14`, `carrier_codec.py:10`, `integer_model_io.py:5` | **no** |
| `torch` | `inflate.py:15-17`, `hpac_integer.py:5-7`, `hpac_integer_sparse.py:7-8`, `integer_model_io.py:6` | **no** |

**3 third-party packages; 1 declared.**

**Severity, stated honestly.** This is **not** a shipping blocker. `upstream/pyproject.toml` lists
`numpy` as a core dependency and `upstream/uv.lock` pins `torch 2.10.0+cpu` (Linux) — the contest eval
host necessarily has both, because `evaluate.py` runs torch scorers. Run B is the faithful shape and it
passes. What is defective is the **statement of closure**: a document that names one dependency while
the code imports three cannot be used to reason about portability.

**And the instrument was blind to it.** FX1's macOS bare-venv smoke passed because that venv had numpy
and torch installed — "clean" meant *clean of constriction*, not clean. A closure smoke that only
removes the dependency you already suspect can only ever confirm what you suspected. This is the same
genus as today's other failures: an instrument that cannot see the thing it is supposed to certify.
Run A, which I built to be *wrong* about the contest shape, is the only reason we know.

**Cure (owed, small):** `runtime-dependencies.json` should enumerate all three with their provenance
class — `constriction` = **self-installed by our entrypoint**; `numpy`/`torch` = **contest-runtime
provided, pinned by `upstream/uv.lock`, asserted-not-installed**. The distinction matters: silently
self-installing torch would be a 2 GB download inside a 30-minute job.

---

## §3 — THE WHOLE-JOB BUDGET, with its scope stated

`timeout-minutes: 30` is the **entire CI job** (Catalog #835), not the decode step. What §1 measures is
**only the dependency-bootstrap + receiver-import segment**: **5.302 s cold**, **3.010 s warm**, on a
Modal CPU container. That is the honest boundary — this arm ran no n600 decode.

I am **not** reproducing FX1's 159.598 s cross-host projection. FX1 labelled it "budget context, not a
measured total" because it double-counts receiver startup and mixes macOS timings with an RTX 5070. A
number assembled from two hosts is not a single-host total, and restating it would launder it.

What is now MEASURED: the bootstrap costs **~5.3 s of the 1800 s job budget (0.29%)** on Linux x86_64,
network fetch included, wheel-only, fail-closed. The remaining terms (inflate decode over n600, then
`evaluate.py`) are unmeasured here and are the real budget question.

---

## §4 — THE RATE DENOMINATOR, MEASURED

`evaluate.py:64`:

```python
uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
```

Measured on the current tree: `upstream/videos/` contains **1 file** (`0.mkv`, 37,545,489 B), so the sum
is **37,545,489** — **delta 0** from the constant.

Two corrections to how this gets discussed:

1. It is `args.uncompressed_dir`, a **CLI-supplied path**, not a hardcoded `videos/`. The denominator is
   whatever directory the harness is pointed at.
2. The constant is correct **today by coincidence of tree contents**, not by definition. One `.DS_Store`,
   one extra video, one stray artifact under that directory and every score computed from the constant
   is silently wrong. Catalog #812's hazard is real; it is currently **non-binding**, which is exactly
   the state in which a guard is worth keeping and a hardcoded constant is not.

Measured locally, no dispatch: this is directory arithmetic and is platform-independent. Spending a
Modal mount of the video tree to re-learn it would have been waste.

---

## §5 — THE DECLARED-DEP vs VENDORED RACE: still not executable, and why

The charter authorized re-opening this "if and only if the Linux leg passes." It passed — but the race
still cannot be run honestly, for a reason the Linux run itself supplies.

`constriction` is a **compiled Rust extension**: the target contains
`constriction.cpython-311-x86_64-linux-gnu.so` (the bulk of the 1,102,335 B). "Vendoring inside
inflate.py" per CLAUDE.md L4 means vendoring **source you can read and ship as code** — you cannot
vendor a platform-specific binary `.so` as inflate.py code without either (a) shipping the binary in
`archive.zip`, which makes it **counted** and is ~1 MB against a ~131 KB corridor, or (b) reimplementing
the range decoder in pure Python, which is a real project and a decode-time risk, not a vendoring step.

So the honest table is:

| arm | install risk | counted bytes | status |
|---|---|---|---|
| declared dep + self-install | MEASURED: 5.302 s cold, wheel-only, fail-closed, rc=0 | 0 | **executable today** |
| vendor the `.so` | none | **~1.1 MB counted** — 8× the whole corridor | **structurally dominated** |
| reimplement decoder in Python | none | ~0 | **unpriced**; a project, not a swap |

FX1 ranked vendoring "structurally preferable but unexecuted." With the wheel now under custody I can
sharpen that: **vendoring the artifact is dominated on bytes**, and the only surviving vendor-shaped arm
is a pure-Python reimplementation, which nobody has priced. Recorded as unpriced, not as preferred.

No adoption on preference. The declared-dep path is the one with a receipt.

---

## §6 — WHY MAIN RAN THIS INSTEAD OF THE ARM

The fx5 arm's first attempt was refused by Modal single-flight (#513), naming three "active" claims. FX4
had already proven the #906 job **complete**; the claims were stale rows from superseded runs
(r2 / r3 / pending_spawn) that were never given terminal outcomes. I wrote three terminal rows
(`stale_superseded_by_r5`) via `tools/claim_lane_dispatch.py`; the reconciler then returned `rc=0`,
`live_modal_call_ids: []`, `active_modal_claims: []`, `consistent: true`, and fx5b independently
confirmed `SINGLE_FLIGHT_CLEAR`.

The arm's second blocker was different: `modal app list --json` **rc=1 after 56.57 s** in its sandbox.
From MAIN the same call returns rc=0 and PyPI answers in 112 ms. Modal is up, credentials work, network
is fine — the arm's sandbox cannot reach the provider. The Linux leg was never blocked; it was
**mis-assigned**. Dispatches belong to MAIN.

---

## §7 — BUDGET

Envelope check before firing (#381). The **local ledger is blind**: `modal_call_id_ledger.jsonl` has
**0 rows carrying cost**, so an envelope check reading it returns "remaining $20.00" regardless of truth.
That is the vacuity genus again — an empty field reads as clear.

Live billing from MAIN (`modal billing report --start 2026-07-09 --json`, rc=0) is the only authority:
**222 rows, cumulative $10.267810, envelope $20, remaining $9.7322 (51.34% used).** Top spenders:
comma-train-lane $5.6201 · clickpolish-pr110-cpu $3.6706 · comma-auth-eval-cpu $0.4983 ·
comma-auth-eval $0.3518. Both fx5 dispatches were seconds of CPU container time against that.

**Owed (two-landing):** #381 has been enforceable only via the billing API; the local ledger's cost
field is unpopulated. Either populate it at dispatch-completion or make the envelope check read billing
and fail closed when it cannot.

---

## §8 — RANKED RESIDUALS, each with its falsifier

1. **Declare all three deps with provenance class** (§2). *Falsifier:* a fourth third-party import
   exists that this enumeration missed — re-derive from AST over all 5 modules, not grep.
2. **Whole-job budget is still unmeasured** (§3). Only the 5.3 s bootstrap segment is measured.
   *Falsifier:* an n600 inflate + evaluate on one Linux host exceeding 1800 s total.
3. **Pure-Python range decoder is unpriced** (§5). *Falsifier:* a working implementation whose n600
   decode fits the budget — that flips the recommendation to zero-install-risk.
4. **#381 envelope reads an empty field** (§7). *Falsifier:* a ledger row carrying real `cost_usd`
   written by the normal dispatch path.
5. **numpy/torch presence is asserted from `upstream/uv.lock`, not observed on the contest host.**
   *Falsifier:* a contest-runtime manifest showing either absent.

## §9 — COULD NOT CHECK / WHY

- **n600 decode on Linux** — not attempted; needs the archive + video tree mounted and real minutes.
  Out of the SCOPE the charter permits, and stated rather than implied.
- **The exact contest runner image** — `debian_slim` + `torch 2.10.0+cpu` mirrors the pinned versions,
  not the image. Delta named: base image, preinstalled system libs, CPU model.
- **CUDA path** — this arm is CPU-only by the compute-split law. No CUDA statement is made.
- **Whether the contest host preinstalls constriction** — unknowable from here; the cure makes it
  irrelevant, since the entrypoint installs it fail-closed when absent and verifies when present.
