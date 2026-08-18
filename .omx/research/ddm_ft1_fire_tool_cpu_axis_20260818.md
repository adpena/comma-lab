# ddm_ft1 — the canonical fire tool gets a CPU axis, and receipts stop being discarded

**Task #1105 · two landings on `tools/fire_modal_auth_eval.py` (+ `tools/modal_endpoint_close.py`).**
**No Modal fire from this arm.** Build + dry-run only; MAIN fires the real row at packet freeze.
**Pointer UNMOVED.** Own-vehicle frontier stays fx1 **S 0.15816036933414834 @ 180,601 B
[contest-CUDA T4 n600]**. This arm shipped apparatus, not a score — that is the honest state.

---

## STORES CONSULTED

| Store | What it gave |
|---|---|
| `.omx/research/ddm_rv2_frontier_adversarial_review_r1_20260817.md` §FO-1/FO-2 (L55-91, L541-545) | The L2 defect, named at `tools/modal_endpoint_close.py:385-388`, with the regression census showing 08-14 CPU rows kept file-records and the 08-17 rr4 row did not |
| `.omx/research/ddm_pq1_submission_packet_prep_20260815/CPU_AXIS_SEALED_FIRE_ORDER.json` | The sealed CPU fire-order: archive `35ac2b9b…` @ 181,161 B, runtime dir, lane/job ids, waiver text, poller deadline 9000 s, `payload_retention.required: true` |
| `.omx/research/ddm_pq2_packet_polish_20260817.md:159-163` | The blocker in the operator's words: the fire tool "is **CUDA-only** … so a CPU row must currently be hand-assembled — precisely the hazard the hand-assembled-dispatch law names as an error factory. **Cure the tool before firing the row.**" |
| `.omx/research/ddm_hm1_model_byte_derivative_20260816.md:199-206` + `experiments/results/ddm_f26r_mc36_contest_cpu_20260814/` | #1054's receipts: the ONE prior `[contest-CPU]` row, MC36, `inflate_elapsed_seconds 831.5345…`, fired through `experiments/modal_auth_eval_cpu.py::main` via the *paired* dispatcher, closed by **recovery** not by a poller |
| `.omx/state/main_hot_state.md:28-31` | The t1h r1→r4 ladder: "attempts 1/2 refused by the paired-axis gate; attempt 1's rc=5 was MASKED by my tail pipe — **never pipe a fire command**" |
| `experiments/results/ddm_t1h_pass2_…_r3/` vs `…_r4/` + `src/tac/candidate_seal.py` | The r3→r4 cure: r3 fired the rr4 tree with the t1h archive and died remotely in 11.2 s on `extracted payload does not match archive.zip`; `--repin-receiver` + stage 3b SEAL is the landed cure |
| `src/tac/deploy/modal/auth_eval.py:111-146` | `validate_modal_auth_eval_pairing` — the paired-axis gate, already accepting `contest_cpu`, already called by both wrappers |
| `src/tac/deploy/modal/paired_dispatch.py` | The pre-existing axis command builder — reused for its **constants**, deliberately not for its command shape (see §L1) |
| memory `hand_assembled_dispatch_is_the_error_factory_20260817` · `.omx/research/ddm_er1_error_class_ledger_and_determinization_20260817.md` | Failures F1-F5 / E1-E11 the 7 stages already cure, and the three still-open debts owned by MAIN |

---

## PRIOR-LAW PREDICTION — verdict: **CONFIRMED**

> Charter: *"The #1054 CPU row proves the worker side exists; the debt is tool-surface plumbing.
> Predict: L1+L2 land in one arm-session with zero worker changes; falsifier: if the CPU route
> requires worker-image changes, STOP and report the delta."*

**The falsifier did not fire. Zero worker changes, zero image changes.** Verified two ways
(mine + an independent recall sweep): `experiments/modal_auth_eval_cpu.py` already carries
`@app.function(image=eval_image, cpu=8.0, memory=16*1024, timeout=9000)` with **no `gpu=`**, a
CPU-wheel torch image, its own `comma-auth-eval-cpu` app, a `prove_env` locked-env prover, and a
`main()` local entrypoint that hardcodes `axis="contest_cpu"` into the pairing gate. And
`tools/modal_endpoint_close.py:495-506` already allowlists that wrapper for closure. The entire
debt was one hardcoded string in the fire tool's dispatch template.

**The prediction was slightly *too pessimistic* in one direction and too optimistic in another:**

* Too pessimistic: I expected plumbing for `--inflate-timeout 1800`, `--evaluate-timeout 5400`
  and `--scorer-input-cache-tensor-volume-run-id`, all of which the sealed order passes
  explicitly. None was needed — the CPU entrypoint's own defaults are **exactly** 1800 / 5400,
  and the volume run-id defaults to `out_dir.name`, which for the sealed output dir *is*
  `ddm_pq2_rr4_exact_contest_cpu_20260817`. Only `--claim-policy` genuinely had no passthrough.
* Too optimistic: the sealed fire-order **as written would have failed**. See §Finding below.

---

## Finding: pq1's sealed `command_argv` was already dead, and the tool's F3 cure is why

The sealed order passes `--expected-runtime-tree-sha256 7acedb07e670e76c…` (the
`runtime_tree_sha256`). Both wrappers accept **only** `""`, `"auto"`, or the runtime **FILES**
digest — here `ba713a2578418497…` — and raise `SystemExit` on anything else, because the projected
and remote tree hashes are environment-coupled and structurally disagree (the 2026-08-04 r9m
deadlock). `7acedb07… ≠ ba713a25…`, so the hand-written command would have been refused.

It would have been refused *locally*, at $0, so this is a dead fire-order rather than a burned
call — but it is a clean demonstration of the charter's thesis: the by-hand path had a defect that
survived pq1's own by-hand argparse verification. The canonical tool pins `auto` unconditionally
(failure F3) on **both** axes, so this class cannot recur; `test_runtime_tree_sha_is_pinned_auto`
holds it there.

Second live finding, surfaced by the dry run itself: reconcile currently reports the
**provable-phantom** condition — `0 live ledger call_id(s), 1 active Modal claim(s)`
(`lane=ddm_fx1_logistic_mixer_t4_n600 job=fx1_mixer_r1`). Stage 4 recorded it without closing it,
correctly, because `--dry-run`. MAIN's real fire will auto-close it.

---

## L1 — CPU axis on the canonical tool

`--axis {cuda,cpu}` (default `cuda`) selects, from **one table**, three things that must never
drift apart: the worker entrypoint, the evidence tag, and the watch deadline.

| | `--axis cuda` | `--axis cpu` |
|---|---|---|
| entrypoint | `experiments/modal_auth_eval.py::main` | `experiments/modal_auth_eval_cpu.py::main` |
| evidence tag | `[contest-CUDA]` | `[contest-CPU]` |
| poller deadline | 2400 s | **9600 s** |

**Extended, never forked.** All seven stages — SANITIZE, VALIDATE, PIN, SEAL(+`--repin-receiver`),
CLAIMS, DISPATCH, ARM-POLLER, MANIFEST — run identically on both axes. A second CPU dispatcher
would have been the error-factory anti-pattern the charter forbids.

**Why the existing `paired_auth_eval_axis_command()` was reused for constants but not for shape.**
It is tempting (both axes, explicit entrypoint, already written) and it is wrong here: it takes
`pair_group_id` as required and always emits `--pair-group-id`, with **no**
`single_axis_waiver_reason` parameter at all — so it cannot express the waived single-axis fire
that cured t1h r1/r2, which is exactly what MAIN's CPU row is. It also emits the runtime-tree pin
only `if expected_runtime_tree_sha256` (weakening F3's always-`auto`) and appends `--gpu`, moving
the proven rr4/fx1 argv. Adopting its shape would have been a **mechanism reduction**. Its wrapper
*paths* are now imported, so "which file is the CPU worker" has one definition.

**The 9600 s deadline is not a round number.** The CPU worker is `timeout=9000`; the tool's old
fixed 2400 s would have abandoned every CPU row roughly a quarter of the way in, recreating
failure F5 (an unwatched paid call) on the very first CPU fire.
`test_cpu_poller_deadline_outlives_the_cpu_worker_timeout` asserts both `9600 > 9000` and that the
literal `timeout=9000` is still in the worker source, so the constant cannot silently decouple.

**Paired-axis gate: semantics unchanged.** The tool still just forwards `--pair-group-id` /
`--single-axis-waiver-reason`; the refusal lives in `validate_modal_auth_eval_pairing`, which
already handles `contest_cpu`. Neither flag set ⇒ the worker refuses. That is the gate, and CPU
inherits it verbatim rather than getting a second, weaker copy.

### Guards (both directions executed)

`tools/tests/test_fire_modal_auth_eval_axis.py`, 29 tests:

* **never-invent-flags, executed.** `test_every_emitted_flag_is_declared_by_its_entrypoint` parses
  the *real* `main()` signature out of each wrapper by AST and asserts every emitted flag maps to a
  declared parameter. pq1 did this by hand ("read the local_entrypoint signature … and confirmed");
  hand-verification is the step this determinizes. Premise guarded separately by
  `test_cuda_only_params_are_absent_from_the_cpu_entrypoint` (`gpu`/`scorer_device`/
  `inflate_device`/`inflate_env` in CUDA, none in CPU).
* **custody parity.** `test_cpu_and_cuda_carry_identical_custody_fields` — same flag set, identical
  values, the entrypoint token the *only* difference. The CPU route cannot be a weaker path.
* **CUDA regression.** `test_cuda_template_is_unchanged_regression` pins the proven rr4/fx1 argv.
* **axis-untagged rows REFUSED.** `write_fire_manifest` raises on a missing/blank/unknown axis tag,
  and leaves no artifact behind. Direction A: a tagged row writes. Direction B: each of the three
  required fields blanked ⇒ `ValueError` and no `FIRE_MANIFEST.json` on disk. CPU and CUDA are
  separate evidence spaces; an axis-less row is not weak evidence, it is none.
* **dry runs no longer mutate.** These fires target SEALED trees whose hashes another agent
  recorded; deleting litter changes the runtime FILES digest the seal names. `sanitize_litter(...,
  apply=False)` reports instead, and `validate_tree(..., skip=...)` skips the same set so both
  modes see one tree. Both directions asserted, including the premise that the validator *does*
  refuse litter without the skip set.

### Third, smaller cure (proactive-hardening law, 08-14): refusals land on disk

t1h r1 returned `rc=5` and a `| tail` swallowed it, so a refused fire read as a successful one and
cost two more attempts. A tool cannot stop a caller from piping, but it can refuse to be silent:
every refusal path (rc 2/3/4/5/6) now writes `FIRE_REFUSED.json` next to where the manifest would
have gone. **Presence of `FIRE_REFUSED.json` means no call exists.** Guarded by
`test_refuse_writes_a_disk_receipt_a_pipe_cannot_swallow`. The "never pipe a fire command" rule
remains convention; this makes the convention's failure survivable rather than silent.

---

## L2 — `modal_endpoint_close` stops discarding str artifacts

`persist_remote_result` wrote only `bytes` and recorded everything else as
`{"embedded_value_type": "str"}` — a measured **type** with the content thrown away, the canonical
measure-and-discard signature, applied to receipts. The rr4 CUDA row returned
`contest_auth_eval.json` and `report.txt` as `str`; both were dropped and had to be recovered from
the Modal result cache. `scripts/pre_submission_compliance_check.py --contest-final` requires both.

Now every artifact is persisted byte-for-byte through the same `atomic_bytes` + `file_record` path:
`bytes`/`bytearray`/`memoryview` raw, `str` as UTF-8, anything else as canonical JSON, and an
unserializable value as its `repr` marked `lossy_repr: true`. Raising instead would discard a whole
paid harvest over one bad entry; keeping it and saying so is the ALWAYS-KEEP-THE-PAYLOAD reading.

### Both-direction control — EXECUTED, same three-artifact fixture

| | `contest_auth_eval.json` (str) | `report.txt` (str) | `provenance.json` (bytes) |
|---|---|---|---|
| **before** | `{"embedded_value_type": "str"}`, **not on disk** | same, **not on disk** | file-record, sha `68c759aa…` |
| **after** | file-record, sha `851b3822…` | file-record, sha `45a7e14e…` | file-record, sha `68c759aa…` **unchanged** |

`returned_artifacts/` held **1 of 3** before and **3 of 3** after; the bytes path did not move.
Additionally, HEAD's pre-fix `persist_remote_result` was extracted by AST and executed against the
new fixtures to confirm the new tests genuinely **fail** on the old code
(`l2_head_prefix_control.py`). Both control scripts are retained beside the receipts.

Guards: `tools/tests/test_modal_endpoint_close_artifact_retention.py`, 16 tests — the positive
invariant is that **every** record names a real file with a size and a digest (not merely that the
old key name is gone), plus round-trips for 9 payload types, NaN→repr, and the preserved refusals
for unsafe names and non-mapping artifacts.

---

## Verification

* `tools/tests/test_fire_modal_auth_eval_axis.py` — 29 passed
* `tools/tests/test_modal_endpoint_close_artifact_retention.py` — 16 passed
* `tools/tests/test_modal_endpoint_close.py` — **41 passed** (pre-existing suite, no regression)
* `ruff check` clean on all four changed files
* CPU dry-run on MAIN's sealed tree: archive `35ac2b9b…` @ 181,161 B matched `--require-archive-sha`;
  `SEAL PIN CONSISTENT`; `AXIS: cpu [contest-CPU]`; rc=0
* CUDA control dry-run on the same tree: `AXIS: cuda [contest-CUDA]`, argv unchanged

Receipts live at `experiments/results/ddm_ft1_cpu_axis_dryrun_20260818/` — durable local path, but
`experiments/results/` is gitignored, so the serializer refused them (rc=13) and they are
**local-only evidence**. Shas, so the bytes are verifiable:

```
9f9640469d95cc3b04c274012bbbb1145679523fa1ab780169f4dd2a5a1508e7  CPU_DRYRUN_FIRE_MANIFEST.json
78a9a6a51865cba0a27f7e9d92f036a1ac2143bc2ddde3da98a7c60cf98de312  CUDA_CONTROL_DRYRUN_FIRE_MANIFEST.json
d60a3f1e39a594c617449df64964309b2be1138eff616859e5c17a553439d1f1  l2_artifact_retention_control.py
66a9de2c6b2539c2d85b054fc7d2e15563f47076fbc665f730c4997828d584d0  l2_head_prefix_control.py
```

The dry-run manifest was moved out of `experiments/results/ddm_pq2_rr4_exact_contest_cpu_20260817/`
so **MAIN's fire dir is left pristine** — a rehearsal must not plant artifacts where a real row goes.

### Answering MAIN's relay (received mid-task)

MAIN's recall relay matched this arm's own sweep on every point but one, and recommended
*"REUSE `paired_auth_eval_axis_command` … Do not fork a template."* **No template was forked**:
`build_dispatch_argv` is the fire tool's own pre-existing inline `cmd = [...]` list, extracted
into a function and parameterized on the entrypoint. There is still exactly ONE executing
dispatcher, and `paired_dispatch`'s wrapper-path constants **are** now imported.

Its command *shape* is not reused, for three measured reasons (see §L1): it cannot express a
waived single-axis fire — which is precisely what MAIN's sealed CPU row is — because it has no
`single_axis_waiver_reason` parameter and always emits `--pair-group-id`; it makes the F3
runtime-tree pin conditional; and it appends `--gpu`, moving the proven rr4/fx1 argv. Adopting it
would be the mechanism reduction the charter's OPTIMAL FORM clause forbids.

Two smaller corrections to the relay:

* Relay item 3 lists `--inflate-timeout 1800`, `--evaluate-timeout 5400` and
  `--scorer-input-cache-tensor-volume-run-id` as needing plumbing. Measured: they do not. The CPU
  entrypoint's own defaults are **exactly** 1800 and 5400, and the volume run-id defaults to
  `out_dir.name`, which for the sealed output dir *is* `ddm_pq2_rr4_exact_contest_cpu_20260817`.
  Only `--claim-policy` genuinely lacked a passthrough, and it now has one.
* Relay item 3 says to "grep the CPU wrapper's `add_argument` list". There is none — these are
  `@app.local_entrypoint()` functions and Modal derives the CLI flags from the **function
  signature**. The guard therefore parses that signature by AST, which is the actual truth source.

Relay item 5 (er1's E1/E5/E8): **not absorbed.** L2 touches only `persist_remote_result`; claim
closure at harvest (E5) lives in the poller/closer control flow and is untouched.

---

## What MAIN fires at packet freeze

Economics are unchanged and remain MAIN's call: pq1 measured ~$0.40 against ~$1.38 remaining
headroom (~29%), and its own adjudication note says *"PREPARED, NOT RECOMMENDED EITHER WAY."*
This arm cured the safety blocker; it does not argue the row is worth buying.

```bash
.venv/bin/python tools/fire_modal_auth_eval.py \
  --axis cpu \
  --runtime-dir /Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime \
  --output-dir experiments/results/ddm_pq2_rr4_exact_contest_cpu_20260817 \
  --lane-id lane_ddm_pq2_rr4_exact_contest_cpu_20260817 \
  --instance-job-id ddm_pq2_rr4_exact_contest_cpu_20260817 \
  --claim-agent MAIN \
  --claim-policy require_active \
  --require-archive-sha 35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956 \
  --single-axis-waiver-reason "CUDA axis already measured on these exact bytes; this row completes the pair"
```

Drop `--dry-run` is the only change from the rehearsed command. **Do not pipe it.** The tool arms
the detached poller itself at 9600 s; `--claim-policy require_active` means the sealed
`claim_argv` must run first.

## Open debt this arm did NOT close

1. **`--claim-policy require_active` needs a live claim** — the sealed `claim_argv` is still a
   separate by-hand step ahead of the fire. Determinizing that (fire tool opens its own claim) is
   the obvious next cure and is **not** in this landing.
2. **The "never pipe a fire command" rule is still convention.** `FIRE_REFUSED.json` makes its
   failure detectable, not impossible.
3. **er1's three open debts remain MAIN's**: proved-tree↔fired-tree receipt binding (E1), claim
   closure at harvest (E5), DESYNC per-frame telemetry (E8).
4. **rv2 FO-2(b) — the closer-side gate** that refuses a closure whose `materialized_artifacts`
   carries a content-free record — is not landed. After L2 the producer cannot emit one, so the
   gate would read clean today; it is worth landing only as defence against a *future* producer.
   Flagged rather than built, since a gate that can only pass is not yet evidence.
