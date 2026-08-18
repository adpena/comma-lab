# ddm_seal1 — THE CANDIDATE-SEAL CONTRACT (task #1115)

**Date:** 2026-08-18 · **Axis:** byte identity only. Nothing here computes, claims, or implies
a score. · **Status:** LANDED, 36 new controls + 49 sister controls green.

> Operator 2026-08-18, verbatim: **"Things can be better frozen and constrained through
> engineering."**

## The finding first

"Sealed" was a word, not a checkable object. Five incidents in one day were all the same
shape — a load-bearing pin left as prose and patrolled by attention. This lands the freeze:
one typed document that pins everything a fire needs, one validator that re-derives every pin
from disk at consumption time, and one consumer — `tools/fire_modal_auth_eval.py --seal` —
that refuses to spend a paid call when any pin has moved.

The constraint is what makes it work. Every value the seal owns is **removed from the fire
command line**: pass `--seal` and the runtime dir, archive path, archive-sha pin, and contest
axis are *derived*. Supplying one of them by hand alongside a seal is itself a refusal. There
is no state in which the fired bytes and the sealed bytes can disagree and the fire still
takes.

## What a seal pins, and the measured failure each field extincts

| Field | Extincts |
|---|---|
| `archive` (path, sha256, bytes) | **rr2** — a hand-assembled tree was fired whose bytes were never the proved bytes (S 27.83 vs projected 0.1585). |
| `archive_member` (name, sha256, bytes) | The finest statement available: our archives are 1–2 members and `p`/`0.bin` carries ~99% of the payload. |
| `runtime` (content-only FILES digest, count, total bytes) | **ps1u r1** — a receiver sha pin drifted between seal and fire. Content-only because **r9m** deadlocked on two validators disagreeing over an environment-coupled tree hash while the bytes were identical. |
| `receiver_pins[]` (per-file relpath + sha + bytes) | A tree-level digest mismatch names nothing. Per-file pins let the refusal say *which* file drifted. |
| `admit_bar` **with `derivation`** (pointer axis, score at seal, pointer archive sha at seal, tolerance) | **qs4** — qs2's Schur compensation constant carried onto a different object, +2.4e-4 S. A bar without its baseline is unanchored, and baselines move. |
| `axis` ∈ contest_cuda / contest_cpu / advisory | **t1h r1/r2** — a single-axis waiver hand-supplied at fire time. CPU and CUDA are separate evidence spaces; the seal decides, not the shell history. |
| `retained_payload_paths[]` | ALWAYS KEEP THE PAYLOAD, *checked* rather than asserted. |
| `falsifiers[]` | Pre-registered, so a harvest cannot invent its own bar afterwards. |
| `seal_sha256` over the canonical serialization | An edited seal is a `SEAL_TAMPERED` refusal, not a quietly different fire. |

## Typed verdicts — no bare booleans

`SEAL_VALID` · `SEAL_SCHEMA_VIOLATION` · `SEAL_PLACEHOLDER_PIN` · `SEAL_FILE_MISSING` ·
`SEAL_SHA_DRIFT` · `SEAL_BYTE_DRIFT` · `SEAL_RUNTIME_DRIFT` · `SEAL_BAR_DRIFT` ·
`SEAL_TAMPERED`.

Ordering is deliberate and each stage **returns** rather than accumulating: placeholders are
reported before the signature check (a placeholder seal is honestly signed garbage, and
calling it tampering sends the reader hunting an attacker), and the signature is checked
before any drift (drift measured against a value a tamperer chose is not a measurement).

## Two constraints that make the document honest

**1. No placeholder passes.** Catalog #287's placeholder rejection, lifted from waiver
rationales to structured data pins. `""`, `"TBD"`, `"pending_ratification"`, `<anything>`, a
non-hex sha, the all-zero digest, a non-positive byte count, an empty `receiver_pins` — each
refuses as `SEAL_PLACEHOLDER_PIN`. A seal carrying a stand-in where a sha belongs is not a
weaker seal; it is not a seal.

**2. The digest is invariant under its own consumer's first stage.** `measure_runtime_digest`
covers exactly the files that *can reach the evaluator*: it skips what the transport zip skips
(host metadata, bytecode caches) **and** any hidden path part, because
`validate_runtime_upload_file` refuses dot-prefixed files outright. macOS re-creates
AppleDouble `._` litter on ExFAT the instant anything writes to a custody volume — had that
litter entered the digest, every seal written on the SSD tier would refuse itself minutes
later for a reason having nothing to do with the candidate. A seal its own consumer's stage 1
could invalidate would be a seal in name only.

**That second rule was found by the control, not by design.** The first cut skipped only the
transport-zip set; `test_the_runtime_digest_is_invariant_under_the_fire_paths_sanitize_stage`
went red on `._inflate.py` and the definition was corrected. The red direction earned its keep
before this shipped.

## Two defects the review passes caught (both fail-open, neither found by a failing run)

**A — a malformed seal CRASHED instead of refusing.** Stage 1 originally checked only that
required keys existed, not their types. A seal whose `archive` was a bare sha string sailed
past the placeholder scan (which skips non-objects) into the drift comparison and raised
`AttributeError: 'str' object has no attribute 'get'` **into the fire path** — a traceback a
reader can mistake for a tooling bug rather than a bad seal. Verified empirically by reverting
the fix: pre-fix `AttributeError` at `candidate_seal.py:1009`, post-fix 7 controls green.
Stage 1 now type-checks `archive` / `runtime` / `admit_bar` (objects), `receiver_pins` /
`retained_payload_paths` / `falsifiers` (lists).

**B — `"schema": null` was accepted.** The version check read
`document.get("schema") not in (SEAL_SCHEMA, None)`, so an unversioned seal passed as
understood. Removed the `None` escape hatch: claiming to understand an unversioned document is
how a future v2 seal gets validated by v1 rules.

Both were found by re-reading the validator, not by re-running it. Neither had a failing test
until one was written for it.

## Consumer wiring

`tools/fire_modal_auth_eval.py` gains **STAGE 0 SEAL**, ahead of sanitize/validate/pin/claims
and ahead of any subprocess:

1. Refuse `--seal` alongside `--runtime-dir` / `--archive` / `--require-archive-sha` /
   `--axis` — two sources for one truth is the hand-assembly hazard.
2. Refuse `--seal --repin-receiver` — repin mutates the receiver, invalidating the very seal
   being consumed. Re-pin at compose time, *then* seal.
3. `validate_seal` against disk; on any non-`SEAL_VALID` verdict, **rc=7**.
4. Refuse an `advisory` seal — a paid Modal row is contest evidence and would be mislabelled.
5. Otherwise derive runtime dir, archive, `--expected-archive-sha256`, and the worker
   entrypoint from the seal.

**The refusal is unmaskable.** t1h r1 returned rc=5 into a `| tail` and read as success. A
seal refusal writes the human line to **stderr**, a receipt to **`<seal>.REFUSED.json`** (where
the next reader of that candidate looks), and `FIRE_REFUSED.json` in the output dir. rc=7 is
distinct from every existing code (2 missing, 3 validators, 4 archive-sha pin, 5 dispatch,
6 receiver-pin mismatch).

Backward compatible: `--axis` now defaults to `None` as a sentinel and resolves to `cuda`, so
every existing no-seal invocation is unchanged. Proved by
`test_the_no_seal_path_is_unchanged`.

## Producer

`tools/make_candidate_seal.py` has **no flag for a hand-typed sha** — a producer that accepted
one would reproduce the failure the seal exists to stop. `--verify-archive-sha` is *checked*
against the measured bytes and refuses on mismatch; it never becomes the stored value. After
writing, the producer runs the **consumer's** validator on its own output and deletes the seal
if it does not pass: a seal that cannot be consumed is worse than none.

One usability defect was fixed at the same site: argparse's negative-number matcher does not
understand `-3.5e-6`, and every admission bar in this project is exactly that shape. The
matcher is widened (safe here — no option string begins with a digit) so the operator never
meets `expected one argument` while sealing by hand.

## Worked example — the sa1 FIRE_ORDERS base block

`FIRE_ORDERS.json`'s `base` names archive sha `35ac2b9b…`, 181,161 B, S 0.15853325034789678
`[contest-CUDA T4 n600]`, and its `runtime_note` says the shipping rr4 `candidate_runtime`
decodes it. Sealing that exact object (custody read-only throughout):

```bash
.venv/bin/python tools/make_candidate_seal.py \
    --candidate-id sa1_base_rr4_cuda_prob_reencode \
    --runtime-dir /Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime \
    --axis contest_cuda --admit-bar-net-ds -3.5e-6 --archive-member p \
    --retained-path /Volumes/APDataStore/pact/ddm_sa1/retained \
    --falsifier "net dS >= -3.5e-6 at n600 contest-CUDA refutes the rate credit" \
    --out <durable>/SEAL_sa1_base.json
```

Measured output — every number computed from disk, none typed:

```
SEALED
  candidate   sa1_base_rr4_cuda_prob_reencode [contest_cuda]
  archive     181,161 B sha 35ac2b9beb7e6fa8…      <- matches FIRE_ORDERS base exactly
  runtime     33 files, 517,725 B digest 1dfbec32a623513d…
  receivers   inflate.py, inflate.sh
  admit bar   net dS < -3.5e-06 vs contest_cuda 0.15771358 (tolerance 0)
  seal sha    516d30c64e340f3c7ecf717571b393e7e96184454199746a60af424cc4116203
  VALIDATED   SEAL_VALID
```

**Green:** `fire_modal_auth_eval.py --seal … --dry-run` emits a dispatch argv whose
`--archive`, `--submission-dir`, and `--expected-archive-sha256 35ac2b9b…` are all derived
from the seal, with the CUDA worker entrypoint selected by the seal's axis. Zero hand-typed
values.

**Red (same tree, copied so custody is never mutated):** appending an 18-byte comment to
`inflate.py` produces

```
SEAL REFUSED (rc=7): seal invalid: SEAL_RUNTIME_DRIFT
  receiver 'inflate.py' drifted: sealed 3ba93237bb811fc9…/2282 B, on disk 9a36d3058940508d…/2302 B
SEAL REFUSAL RECEIPT: …/SEAL_copy.json.REFUSED.json
```

No subprocess is reached. The drift is *named*, not merely detected.

Note what the bar block records: the live `contest_cuda` pointer is **0.15771358** on candidate
`debb025f…`, while sa1's base is **0.15853325** on `35ac2b9b…`. The frontier has already moved
past this base. The seal freezes today's baseline and, with `tolerance 0`, will refuse the
moment it moves again — which is the qs4 cure doing its job rather than an inconvenience.

## Landed surfaces

- `src/tac/candidate_seal.py` — brick 2 appended beside brick 1 (`SealValidation`, `AdmitBar`,
  `RuntimeDigest`, `build_seal`, `validate_seal`, `write_seal`, `measure_runtime_digest`,
  `runtime_digest_skip_reason`, `read_pointer_state`, `read_archive_member_identity`).
- `tools/fire_modal_auth_eval.py` — `--seal`, `refuse_seal`, `SEAL_AXIS_TO_FIRE_AXIS`,
  `SEAL_OWNED_FLAGS`, stage 0, `main(argv)`.
- `tools/make_candidate_seal.py` — the producer.
- `src/tac/tests/test_candidate_seal.py` — 29 controls (2 positive, 20 executed negatives,
  7 consumer-integration).
- `src/tac/tests/test_candidate_seal_pin_consistency.py` — repaired a **pre-existing** broken
  gate: `assert "return 6" in source` matched one formatting of the refusal and had already
  gone red on a line wrap. Now regex-based, and extended to assert the brick-2 rc=7 path.

## Honest limits

- The seal freezes **byte identity and the bar's baseline**. It does not verify that the
  runtime *decodes* — that remains the entry-point smoke receipt, a separate pin the schema
  can carry as a future field.
- `runtime.path` and `archive.path` are absolute. A relocated custody tree reports
  `SEAL_FILE_MISSING` and needs a re-seal; that is deliberate (a relocated tree is a different
  object until re-measured), but a `--relocate-root` would be a reasonable later affordance.
- `validate_seal(check_pointer=False)` exists for offline replay. It is a *narrower* check and
  callers must not read its pass as a bar re-derivation.
- Nothing here is a score. A valid seal means the object is what it was sealed as — it says
  nothing about whether the candidate admits.
