# ddm_pl1 — the A2 payload loss: what I verified, the gate that was still owed, and the payload I got back

**Date:** 2026-08-16 · **Arm:** ddm_pl1 (Opus) · **Axis:** apparatus. No scorer ran, no launch fired,
no Modal spend. Own-vehicle frontier UNMOVED: **hv1 ep0634 S 0.15959729295498598 @ 182,759 B
`[contest-CUDA T4 n600]`**.

---

## ANSWER FIRST

1. **av3's F2 mechanism is CORRECT in every clause I could check.** The crash, the ordering, the
   `status=ok exit=1` receipt — all confirmed at source and in the retained receipts.
2. **The FIX was already landed — by a sister arm, four hours before I started.** `ddm_rg1b`
   (`60aefac081`, 2026-08-16T04:25:29-05:00) shipped all four instrument fixes av3 deferred. I verified
   each at source and ran its 26 tests green. **I did not re-land it.**
3. **The GATE was still owed, and that is my landing.** rg1b shipped a per-module test that
   string-matches two literal call spellings inside one function. It cannot see the other 11,015
   modules, and a rename disarms it silently. `check_no_bulk_write_strands_the_ready_record` is the
   repo-wide, mechanism-derived sister: **live count 10 over a MEASURED denominator of 11,016 modules**,
   landed **WARN-ONLY** because 10 is not 0.
4. **A2's payload is NOT lost. I recovered it.** The atomic tmp file was written successfully and only
   the rename failed, so the complete `result` — including the full history — survived inside
   `.checkpoints.32551.tmp`. This **refutes** av3's "unrecoverable without re-firing". 379 s of Metal
   recovered without re-running anything.
5. **F3 is confirmed and BROADER than av3 wrote.** Not just C0: **all five** lr1 arms report
   `verdict: PASS` at the identical step-0 value. None improved.
6. **The A1/A3 fire-block can be LIFTED.** Three independent cures are in place. Conditions for MAIN
   are in §7.

---

## 1. F2 verified at source, clause by clause

av3 is a fresh-eyes review, not gospel, so I checked each clause independently rather than restating it.

| av3 clause | verdict | evidence |
|---|---|---|
| `_atomic_write_json(result, args.out)` is ordered AFTER `_atomic_torch_save` | **CONFIRMED (at the time of the run)** | A2 `run.log` traceback: crash at `train_semantic_quantized_resumable.py:1295` in `_atomic_torch_save`, and no `result.json` exists. In the CURRENT tree the order is already inverted (JSON at :1515, save at :1542) — rg1b's fix, see §2 |
| the checkpoints-dir collision raised, because of an existing empty directory | **CONFIRMED** | `IsADirectoryError: [Errno 21] Is a directory: '.../A2/.checkpoints.32551.tmp' -> '.../A2/checkpoints'`; `A2/checkpoints/` exists, is a directory, is EMPTY, and is timestamped 03:19 — launch time, not finalize time |
| `safe_run` reports `status=ok` on a nonzero exit | **CONFIRMED** | `A2/safe_run_status.json` literally reads `"status": "ok"`, `"exit": 1`, `"elapsed_s": 379.307`. `tools/safe_run.py:510` sets `status = "ok"` and overrides only for timeout / oom / killed / interrupted |
| the sealed A1/A3 tickets REPRODUCE the crash | **CONFIRMED as written, ALREADY CORRECTED** | av3 itself fixed the rc2 §1 ticket `--save … /checkpoints` → `/ckpt` at line 114. A1/A3 then ran with `/ckpt` and both have a `result.json` |
| A2's `final_seg` is "unrecoverable without re-firing" | **REFUTED — see §4** | the orphan tmp holds the complete result |

**One clause I could not verify:** av3 states it did not sweep for `safe_run` consumers that key on
`status` alone. I did not sweep either. That gap is still open and is not mine to close silently.

### The mechanism nobody named: where the directory came from

`--save` is used as a **filename PREFIX** for periodic checkpoints (`checkpoints.periodic.step000100…`)
and as a **file path** for the final save. A2's launch created `A2/checkpoints/` as a *directory* at
03:19. Every periodic checkpoint therefore wrote happily as a sibling for 379 s, and the collision only
surfaced at the very last write. That is why the failure was maximally expensive: **the run's cheapest
artifact was gated behind its last and most fragile one.**

---

## 2. Landing 1 — the fix. Already done by ddm_rg1b; verified, not duplicated

`60aefac081` shipped four fixes. I verified each rather than trusting the commit message.

| fix | location | verified |
|---|---|---|
| `--save`/`--out` directory guard, at **parse time** | `train_semantic_quantized_resumable.py:968-979`, inside `parse_args` | ✓ refuses in milliseconds, before any compute |
| result JSON written **before** the checkpoint | `:1515` (JSON) vs `:1542` (save) | ✓ order inverted; the parity-refusal path at `:1474` also writes before raising |
| `best_step` / `improved_over_init` / `init_quantized_exact_seg` in the result | `:1491-1493` | ✓ and threaded through the checkpoint payload + a legacy-tolerant resume (`_restored_init_seg`, `:611-620`) |
| `safe_run` non-zero-exit signal | `tools/safe_run.py:446-449` | ✓ **additive**: `child_exit_nonzero` + `receipt_status_disagrees_with_exit`. `status` itself is deliberately unchanged for consumer compatibility |

`src/tac/pr130_lift/tests/test_av3_instrument_fixes.py` — **26 passed**.

**One honest caveat on the safe_run fix.** `status` still reads `"ok"` on a crashed child. rg1b chose
additive fields over changing the enum, and documented why (existing consumers key on `status`).
That is defensible, but it means **a consumer that reads `status` alone still reads a crash as a
success**. The new fields only help a consumer that knows to look. The unswept-consumer gap from §1
therefore still matters.

---

## 3. Landing 2 — the gate. This is my work

### Why rg1b's test was not the second landing

```python
source = inspect.getsource(trainer.run)
write_result = source.index("_atomic_write_json(result, args.out)")
save_final = source.index("_atomic_torch_save(final_payload, args.save)")
assert write_result < save_final
```

It guards **one function in one module**, by **exact literal call spelling**. Rename either helper and
it stops testing anything, silently. Per the two-landing rule the class-level refusal was still owed.

### The predicate, and why it is a genuine sister rather than a duplicate

`check_no_measure_and_discard_payload` refuses a run whose only persisted artifact is **scalars** while
bytes sat in memory — a defect of **DESIGN**. A2 persisted **both**, correctly, by design. That gate is
structurally silent here. The class is different:

> **the record was READY, and a fragile bulk write was scheduled ahead of it.**

Violation iff, as **sibling statements in one block**: a dict literal of ≥3 keys is built at `k`; an
**unguarded** bulk write runs at `i > k`; that dict is **serialised** at `j > i`.

Roles are derived from **the primitives a helper actually calls** — `_atomic_torch_save` is bulk because
its body calls `torch.save`, and stays bulk under any name. Two cures pass: persist the record first
(what rg1b did), or wrap the save in `try`.

### Measured, with the denominator stated

| | |
|---|---|
| **modules examined** | **11,016** (`src`, `tools`, `scripts`, `experiments`) |
| **excluded, and why** | `experiments/results/**` — 49,136 harvested run-output copies, not maintained source |
| **live count** | **10** |
| **unparseable** | reported by name in `payload_write_order_population()["unparsed"]`, never folded into the cleared count |
| **runtime** | 16.1 s (from 40.1 s, via a prefilter that skips the AST parse but **not** the population) |
| **strictness** | **WARN-ONLY.** 10 ≠ 0 |

**I could have reached zero by narrowing the scan, and refused.** Dropping `experiments/` and test files
would have bought a green strict gate over a population chosen to produce it — the vacuity==pass class
wearing a clean label.

### The ten live sites

**Runner context (8) — genuine:**

| site | record |
|---|---|
| `tools/train_ddm_cl1_hpac_capacity.py:1361` | `result` — **the exact A2 shape**, closest sibling of the incident |
| `src/tac/substrates/hi_nerv/archive_candidate.py:2742` | `bitstream_report` |
| `experiments/ddm_rt1_seg_roundtrip_decomposition.py:327` | `receipt` |
| `experiments/build_pr85_lossless_pure_rate_candidates.py:892` | `summary` |
| `experiments/probe_frozen_instance_horizon_crossframe.py:151` | `rep` |
| `experiments/probe_horizon_band_dseg_lever.py:224` | `rep` |
| `experiments/profile_fp4_layer_sensitivity.py:430` | `metadata` |
| `experiments/profile_hessian_per_weight.py:542` | `metadata` |

**Test-fixture context (2)** — the shape is present, no run product is at stake:
`src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py:263`,
`src/tac/optimization/tests/test_ddm_dm4_j5_counted_application.py:131`.
I kept them in scope and labelled them rather than excluding tests to flatter the count.

### Two false positives I found by READING the sites, not by trusting the count

1. `experiments/tests/test_ddm_cx2_trace_evaluate.py` — `evaluator_dependencies` is a dict of **fixture
   paths**, used as a subscript **receiver**, never serialised. Cure: the record must be an **argument to
   the serialiser**. Pinned by `test_a_dict_of_paths_used_as_a_subscript_receiver_is_not_a_record`.
2. `experiments/train_levelset_witness_realized_through_R_mlx.py` ×3 — `_tail_cycle_start_epoch =
   {"v": None}` is a **closure box**, and a `with` block 3,000 lines away was being read as a sibling.
   Cures: the ≥3-key floor, and recursing into blocks instead of flattening them. Pinned by
   `test_a_small_dict_is_a_closure_box_not_a_record`.

A third near-miss: the **cured** trainer ends with `print(json.dumps(result))`. My first predicate
flagged it — the gate would have refused its own fix. Requiring a real file argument fixed it, pinned by
`test_printing_the_result_is_not_persisting_it`.

### The positive control caught a hole in my own gate

The registered control plants the A2 shape with `with open(args.out, "w"): json.dump(result, fh)` — the
ordinary way to write JSON in Python. My first draft excluded **every** compound statement and **could
not see its own planted violation**. `with` does not branch, so it is now transparent; `if`/`for`/`while`
still are not. Without the control this gate would have shipped near-blind on real code.

### Registration, proven

- appended to `CONFOUND_GATES`; `tac.preflight` iterates that tuple, so membership *is* the wire-in
- **not** in `_CONFOUND_STRICT` — pinned by
  `test_the_gate_is_warn_only_in_preflight_while_its_live_count_is_nonzero`
- registered in `POSITIVE_CONTROLS`; `MIN_POSITIVE_CONTROL_COVERAGE` ratcheted **12 → 13** per the
  module's own instruction. Uncovered stays **18** (my gate ships covered)
- `src/tac/tests/test_payload_write_order_gate.py` — **25 tests, all passing**

---

## 4. A2's payload is recovered — av3's "unrecoverable" is REFUTED

`_atomic_torch_save` writes the tmp, fsyncs it, and *then* calls `os.replace`. Only the **rename**
failed. Nothing cleaned the tmp up. And the trainer sets `final_payload["result"] = result`.

So the complete result was sitting on disk the whole time:

```
/Volumes/APDataStore/pact/ddm_lr1/A2/.checkpoints.32551.tmp
  1,700,991 B · sha256 9f6ee1da777c64cb5034070989e617f44616af9b2ec1d67337295fa31a328f0c
```

Recovered by `torch.load`, **not** re-run:

| field | value |
|---|---|
| `verdict` | `PASS` |
| `quantized_exact_seg` | 0.00028616163465711804 |
| `packed_parameter_bytes` | 40,252 |
| `ema_deployed_argmax_parity.passed` | `True` |
| `history` | 0 → 2.8616e-4 · 100 → 5.1648e-4 · 200 → 4.0685e-4 · 300 → 3.8796e-4 · 400 → 3.6760e-4 · 500 → 3.5749e-4 · 600 → 3.5439e-4 |

Persisted, with provenance, to `/Volumes/APDataStore/pact/ddm_lr1/A2/result.recovered.json`
(schema `ddm_pl1_recovered_result.v1`; named `.recovered.` so it can never be mistaken for something the
run wrote). **A2's ladder row no longer has to be reconstructed from `run.log`.**

There is a lesson in the shape of the accident: the tmp survived **only because the failure path did not
clean up after itself**. A `finally: temporary.unlink()` would have destroyed the very bytes that saved
this. I am not proposing one.

---

## 5. F3 — confirmed at source, and broader than av3 wrote

`best_seg`/`best_state` are seeded from the step-0 EMA shadow before the loop, and `best_key` is compared
against every eval — so a run that only degrades never displaces step 0, and the top-level headline
reports its own input.

av3 named C0. **It is all five arms**, each reporting the identical step-0 number:

| arm | headline `quantized_exact_seg` | verdict | step-600 | improved? |
|---|---|---|---|---|
| C0 | 0.00028616163465711804 | PASS | 3.0451e-4 | no |
| A1 | 0.00028616163465711804 | PASS | 3.1861e-4 | no |
| A2 (recovered) | 0.00028616163465711804 | PASS | 3.5439e-4 | no |
| A3 | 0.00028616163465711804 | PASS | 7.8934e-4 | no |
| W1 | 0.00028616163465711804 | PASS | 3.9146e-4 | no |

Five runs, five `PASS`, zero improvements. The `PASS` is real arithmetic (2.86e-4 < 4e-4) applied to the
**initial condition**. rg1b's `best_step` / `improved_over_init` / `init_quantized_exact_seg` fix the
reporting going forward; these five files predate it (03:19–03:59 vs the 04:25 fix) and **do not carry
the fields**. So MAIN's INSTRUMENT WARNING stands for these five artifacts specifically: read `history`.

---

## 6. Pre-existing failures I did NOT fix, and did not launder

`src/tac/tests/test_confound_gates.py` had **9 failures before I touched anything**; my landing added a
10th (my gate had no bound entry). After the fix-forward: **7 remain, all pre-existing.** I am naming
them rather than re-baselining someone else's debt:

| still failing | measured vs bound | mine? |
|---|---|---|
| `check_levelset_hosc_requires_beta_end` | 10 vs 9 | no |
| `check_no_raw_virtual_memory_safety_basis` | 2 vs 0 | no |
| `check_process_guard_excludes_observer_flag_values` | 1 vs 0 | no |
| `check_no_stub_lever_factories` | 11 vs 10 | no |
| `check_checkpoint_saves_do_not_silently_drop_optimizer_state` | 1 vs 0 | no |
| `check_refusal_gates_have_live_positive_control` | uncovered 18 vs ceiling 17 | **no** — measured at 18 *before* my change (total 30, covered 12); my gate ships covered, so it stays 18 |
| `TestPositiveControlNegativeTwins831::test_control_registry_and_ratchets_are_consistent` | same uncovered 18 vs 17 | no |

The uncovered-ceiling breach means **some earlier gate landed without a positive control**. That is the
denominator-side ratchet doing its job, and it is a real open item for whoever owns those gates — the
ceiling must be lowered by covering gates, never raised to admit a bare one.

**Fixed forward (2):** `test_all_gates_registered` was asserting 29 against a live 30 —
`check_no_row_contract_error_quarantines_the_ledger` had been appended without updating it. Worse, its
absence from the `bounds` map made its parametrised case raise **KeyError**, which reports as a test
error and **masks** the live-count regression the bound exists to catch. Both it and my gate now have a
name entry and a bound (0 and 10). This follows the convention the file documents for itself.

---

## 7. A1/A3 fire-block — RECOMMEND LIFT

MAIN's `main_hot_state.md` NEXT_BOUNDARIES holds a FIRE-BLOCK. Its two owed conditions are now met, and
one of its premises is already stale.

**Stale premise:** A1 and A3 **have already run** (03:34–03:47), fired with `--save …/<arm>/ckpt`, and
both have a `result.json`. The block reads as if they are pending.

**Cures now in place — three, independent:**

1. **The ticket** — av3 corrected rc2 §1 to `--save /Volumes/APDataStore/pact/ddm_lr1/<arm>/ckpt`
   (verified at line 114).
2. **The code, fail-fast** — `parse_args` refuses a directory at `--save`/`--out` in milliseconds. Even
   the old ticket text can no longer burn 379 s.
3. **The code, fail-safe** — the result JSON is written before the checkpoint, so a save that raises can
   no longer strand the record.

**What MAIN should verify before lifting** (all cheap, all local):

```bash
.venv/bin/python -m pytest src/tac/pr130_lift/tests/test_av3_instrument_fixes.py -q   # expect 26 passed
.venv/bin/python -m pytest src/tac/tests/test_payload_write_order_gate.py -q          # expect 25 passed
grep -n -- "--save" .omx/research/ddm_rc2_regime_charter_and_lr_probe_20260816.md     # expect .../ckpt
```

**What does NOT clear, and should stay recorded:**
- the **INSTRUMENT WARNING (F3) stands** for the five existing `result.json` files — they predate the
  fix and lack `best_step`/`improved_over_init`. Read `history`.
- the **`safe_run` status-only consumer sweep** is still unswept (§1). `status=ok` on a crashed child is
  still readable as success by any consumer that does not know about the new fields.
- **A2 needs no re-fire.** Its result is recovered (§4). Re-running it would spend ~380 s of Metal to
  reproduce a number now sitting in `result.recovered.json`.

---

## 8. What this unit did NOT establish

- **No sweep of `status`-only `safe_run` consumers.** av3 checked one and cleared it; I checked none.
  Open.
- **No proof the ten live sites are all harmful.** I read all ten and judged 8 runner-context / 2
  test-fixture. I did not cure any of them — curing another arm's runner mid-flight is not my call.
- **No strict flip.** The gate is warn-only by measurement, not by preference.
- **No claim about `experiments/results/**`.** 49,136 modules are outside my denominator by an explicit,
  named choice.
- **No score, no launch, no spend.** Pointer UNMOVED.
- **The gate is one predicate, not the class boundary.** It catches the ready-record-made-to-wait shape.
  A run can still lose its product in ways this does not see — an unguarded write between two record
  writes, or a crash before the record is built at all.

---

## Cross-references

- `.omx/research/ddm_av3_fresh_eyes_review_20260816.md` — F2/F3, the review this answers
- `60aefac081` (ddm_rg1b) — the fix landing I verified rather than duplicated
- `src/tac/confound_gates.py::check_no_bulk_write_strands_the_ready_record` + `payload_write_order_population`
- `src/tac/tests/test_payload_write_order_gate.py` — 25 controls
- `src/tac/payload_retention_gate.py::check_no_measure_and_discard_payload` — the sister class
- `/Volumes/APDataStore/pact/ddm_lr1/A2/result.recovered.json` — the recovered payload
