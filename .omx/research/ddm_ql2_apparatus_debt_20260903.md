# ddm_ql2 — three owed apparatus rows closed: a strict gate driven 11 → 1, a broken tool given its real flag, a sealed receipt superseded append-only

**Arm:** `ddm_ql2_apparatus_debt`
**Charter:** `.omx/research/charters/ddm_ql2_apparatus_debt_20260903.md`
**Predecessor:** `.omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md` (f1f0abd27) — the four owed rows
**Craft:** `docs/operating_manual_craft_handoff.md` — §4 (re-derive, never recognize) decided every disposition
below; it is why ql1's three-field delta and three-source drift were reproduced from different code paths
before either was acted on, and why ql1's "scorer-class" cost estimate did not survive contact with the source.
**Commits:** `2cb39a8c3` (item 1), `05307d426` (items 2 + 3)

## HEADLINE

All three non-scorer rows closed. The gate went from **11 live violations to 1**, and the one that remains is
a live arm's file this arm was not permitted to edit. The bounded-target-G receipt is superseded **append-only**
— the sealed July bytes are retained AND now pinned by the test, so "we did not rewrite it" is checkable rather
than merely asserted. The stack-receipt tool has a real `--strict-source-reopen` flag with wiring tests.

Two things I did not expect. **The charter's prior-law prediction is FALSIFIED** — plainly, per its own
falsifier. And curing the gate surfaced a **real latent crash** that had been sitting in the tree: a probe that
writes an npz into a directory it has not created yet, whose record write is the line that would have been lost.

## PER-ITEM TABLE

| # | Item | BEFORE | AFTER | Evidence |
|---|---|---|---|---|
| 1 | `check_no_bulk_write_strands_the_ready_record` live count | **11** over 11,533 modules (0 unparseable) | **1** over 11,535 | 8 reorders + 2 waivers; bound lowered 10 → 1, never raised |
| 1b | latent cold-run crash in `probe_horizon_band_dseg_lever.py` | npz written before `OUT_DIR.mkdir` | mkdir → record → npz | found by curing, not by looking |
| 2 | `tools/build_taskspace_inverse_stack_receipt.py` `strict_source_reopen` | hardcoded `True`, tool unusable under the drift | real `--strict-source-reopen` / `--no-strict-source-reopen` + `--dry-run` | 6 new tests; degraded mode exercised live |
| 2b | V9/PBR2 teacher-census rematerialization | claimed scorer-class (ql1) | **NOT scorer-class; blocked on a packet re-seal** | measured; test stays quarantined, reason updated |
| 2c | V9 renderer source drift | 3 of 13 sources moved (ql1) | reproduced, **and output measured BIT-IDENTICAL** | `v9_renderer_source_drift_equivalence.json` |
| 3 | bounded-target-G receipt | 1 strict-xfail, receipt had no producer | receipt superseded, **test un-quarantined**, producer tool landed | July bytes retained at `854ffaf3…`, pinned |

## SUITE COUNTS

| suite | failed | errors | passed | xfailed | wall |
|---|---|---|---|---|---|
| `src/tac/witness_dsl` (ql1 AFTER, baseline) | 0 | 0 | 1354 | 21 | 1294.38 s |
| `src/tac/witness_dsl` (this arm, MEASURED) | **0** | **0** | **1361** | **20** | 972.06 s |

Run retained (committed, not gitignored) at
`.omx/research/ddm_ql2_apparatus_debt_20260903/runs/witness_dsl_full_suite_after_20260903.pytest.txt`,
with a machine-readable `*.counts.json` and the launch manifest beside it. Launched detached via
`tools/launch_detached_process.py` (`--nice 10 --nice-best-effort`), pid 16324.

The arithmetic reconciles: **1354 + 21 = 1375 → 1361 + 20 = 1381**, and the +6 total is exactly the six tests
in the new `test_build_taskspace_inverse_stack_receipt_cli.py`. The seven-test rise in `passed` is those six
plus the un-quarantined bounded-target-G test, and `xfailed` falls by exactly one — that same quarantine,
deleted because its lineage is repaired. I wrote that reconciliation down as a PREDICTION before the run
finished and it held exactly; saying so is worth more than presenting it as though it were discovered
afterwards. No test was deleted, skipped, or weakened; `xfail(strict=True)` means every remaining quarantine
still FAILS the moment its own lineage is fixed. (Wall clock is not comparable to ql1's 1294 s — different
machine load, and I make no claim from it.)

Supporting runs: `test_payload_write_order_gate.py` + `test_ddm_dm4_j5_counted_application.py` **38 passed**;
`test_confound_gates.py` **173 passed** (this is the module holding the live-count bound);
`test_taskspace_inverse_stack_receipt.py` **26 passed / 1 xfailed**;
`test_bounded_target_g_encoder.py` **7 passed** (was 6 passed / 1 xfail);
`test_build_taskspace_inverse_stack_receipt_cli.py` **6 passed**.

## ITEM 1 — the gate, and the prediction it falsified

The gate is **not** in `tac.preflight`'s namespace, as the charter warned. It lives at
`src/tac/confound_gates.py:5377`, with its declared denominator at `payload_write_order_population` (:5366)
and its scanner at `_pl1_scan` (:5313). I read all 11 violations rather than counting them.

**PRIOR-LAW PREDICTION: FALSIFIED.** The charter predicted "the 11 violations are ≤ 3 distinct write sites
(bulk writers repeat)", with the falsifier set at ≥ 6 distinct sites. **Measured: 11 violations in 11 distinct
files, 11 distinct sites, zero repeats.** I count it plainly. The premise behind the prediction — that a bulk
writer helper is shared and so repeats — is wrong for this gate by construction: `_pl1_helper_roles` classifies
*module-local* helpers by the primitives their bodies call, so a shared cross-module writer would be invisible
to it. The gate can only ever see per-module sites. A repeat would have required the same file to contain two
stranded records.

| # | site | record | disposition |
|---|---|---|---|
| 1 | `tools/train_ddm_cl1_hpac_capacity.py:1447` | `result` | REORDER — the exact A2 shape, closest sibling of the incident |
| 2 | `src/tac/substrates/hi_nerv/archive_candidate.py:2742` | `bitstream_report` | REORDER |
| 3 | `experiments/ddm_rt1_seg_roundtrip_decomposition.py:327` | `receipt` | REORDER |
| 4 | `experiments/probe_frozen_instance_horizon_crossframe.py:151` | `rep` | REORDER |
| 5 | `experiments/probe_horizon_band_dseg_lever.py:224` | `rep` | REORDER (+ real bug, below) |
| 6 | `experiments/profile_fp4_layer_sensitivity.py:430` | `metadata` | REORDER |
| 7 | `experiments/profile_hessian_per_weight.py:542` | `metadata` | REORDER |
| 8 | `src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py:263` | `trainer_result` | REORDER — the fixture mirrors the trainer's layout, so it now mirrors its order |
| 9 | `experiments/build_pr85_lossless_pure_rate_candidates.py:892` | `summary` | **WAIVE** |
| 10 | `src/tac/optimization/tests/test_ddm_dm4_j5_counted_application.py:131` | `row` | **WAIVE** |
| 11 | `experiments/ddm_ft1_verdict_bhw_pose.py:558` | `receipt` | **LEFT** — ft1's file, off-limits to this arm |

**Why the two waivers are honest and not gate-gaming.** Both are role misreads, and I say which mechanism
misreads them. `_pl1_primitive_role` classifies `write_bytes` as bulk and `write_text` as record *by spelling*.
Site 9's record is `(out_dir / "candidate_summary.json").write_bytes(_json_bytes(summary))` — the record write,
spelled `write_bytes` only because its helper returns encoded JSON — and the line the gate reads as the record
is `_write_markdown`, a human rendering of that same `summary`. The record already goes first; reordering would
put the rendering ahead of its own source. Site 10 writes the SAME `tmp_path` file twice on purpose: once with a
unique bucket, then again with a duplicated row to prove the loader refuses it. Swapping them destroys the test.
I could have made site 9 green by respelling the write as `write_text` for byte-identical output; I did not,
because a waiver that states the misread is worth more to the next reader than a spelling that hides it.

**Why the 11th is left.** `experiments/ddm_ft1_verdict_bhw_pose.py` belongs to the live arm ft1, which the
charter placed off-limits — and that includes adding a waiver comment to it. It is a genuine violation of the
same shape as the eight I cured. It is owed, with a trigger, below.

**The bound, lowered not raised.** `src/tac/tests/test_confound_gates.py` bounded this gate at 10 with the
instruction "Lower it as the ten sites are cured; never raise it." It is now **1**, the measured value, with a
comment naming the one remaining site and the condition under which the gate can flip STRICT (count 0). I did
not flip STRICT: the count is not zero, and a strict flip at 1 would be a lie.

**The latent crash.** At site 5 the record went LAST, behind an `np.savez_compressed(OUT_DIR / ...)` that ran
**before `OUT_DIR.mkdir(parents=True, exist_ok=True)`**. On a cold run the npz raises and `margin.json` never
lands — precisely the A2 failure the gate was built to predict, sitting in the tree unexecuted. Both bugs die in
one reorder: mkdir, then the cheap irreplaceable record, then the rebuildable row curves. This is the gate
earning its keep on a site nobody had run cold.

**Zero new lint.** Ruff with the repo config over all 11 files produces a per-file error set **identical**
before and after (`diff` of the two per-file counts is empty); the 5 remaining `--select F` findings all
pre-exist at HEAD in `probe_frozen_instance_horizon_crossframe.py` and are none of my doing.

## ITEM 2 — the tool's flag is real now; the rematerialization is not one step, and it is not scorer-class

**The flag.** I grepped the argparse before writing anything, per never-invent-flags. `parse_args` had exactly
one argument (`--output`) and `main` called `build_stack_receipt(..., strict_source_reopen=True)` as a literal.
It is now `argparse.BooleanOptionalAction`, default `True`, plus `--dry-run`. Six tests
(`test_build_taskspace_inverse_stack_receipt_cli.py`) pin the two things that make a flag real rather than
decorative: it EXISTS with the safe default, and its value REACHES the consumer instead of being shadowed.

The degraded mode **cannot publish**, and the tool now says so up front instead of failing inside the
publisher: `write_once_receipt` re-validates by rebuilding with `strict_source_reopen=True`
(`taskspace_inverse_stack_receipt.py:1056`), so a `NOT_RUN` receipt is unpublishable by construction.
Exercised live — `--no-strict-source-reopen` returns verdict
`REOPENED_EVIDENCE_STACK_BLOCKED_BEFORE_N600_CANDIDATE_AUTHORITY`, `source_reopen.teacher_census = NOT_RUN`,
and 16 exact blockers including `strict_teacher_and_harvest_source_reopen_not_run`. That restores exactly what
ql1 said was lost: the stack can be INSPECTED while an upstream producer pin is drifted.

**The rematerialization: two corrections to ql1, both measured.**

*Correction 1 — it is not scorer-class.* ql1 wrote "Rematerialization is scorer-class, so again out of reach
here." The census tool's own docstring says "No RGB/scorer transition is run", the sealed header carries
`decode_scorer_dependency: false`, and grepping the module finds no SegNet/PoseNet load. The full decode I ran
took **6.64 s on CPU**. Scorer authority is not what this needs.

*Correction 2 — what it actually needs is a packet re-seal.* `build_real_receipt` refuses at
`tools/measure_g1_teacher_atom_census.py:1248` because `receiver.source_manifest_sha256` (live
`715e3e57…`) ≠ `header["predictor_renderer_sha256"]` (sealed `92ab2350…`) — and that `header` is decoded FROM
the sealed PBR2 packet BYTES (`decode_progressive_geometry_residual(pbr2_payload).header`). The renderer
identity is baked into the packet. So "rematerialize the census" is not one step: it re-seals the PACKET, which
drags `EXPECTED_PBR2_SHA256`, the PBR2 materialization receipt and the n600 grammar cross-close with it. I
report that rather than manufacture a receipt.

**What I did measure, which nobody had.** ql1 named three drifted sources; I reproduced that count against the
seal commit `e153f2031` (3 of 13: `direct_description_carrier_compose.py` `8b097db2`→`6fef110d`,
`predictor_upgrade_xi_chart.py` `60f19569`→`46901b60`, `direct_description_minimizer.py`
`4029644b`→`980fdcf7`), and then measured the **output**. `renderer_source_sha256` is a hash of SOURCE TEXT — a
claim about the code, not about what it produces. Decoding the sealed V9 program with **today's** sources
yields `predictor_semantic_sha256 = 735c01a4…` over the full 64×384×512 label field: **bit-identical** to the
digest the sealed header asserts. Source-text drift, identical semantics. Receipt:
`.omx/research/ddm_ql2_apparatus_debt_20260903/v9_renderer_source_drift_equivalence.json`.

The receipt states what that does NOT license, because it is the tempting move: it does not license narrowing
`RENDERER_SOURCE_PATHS` until the hash matches, and it does not license rewriting the digest. A manifest shrunk
until it agrees is the vacuity==pass class wearing a green label. The equivalence is the evidence that a real
re-seal would be output-preserving — not a substitute for one. The test stays quarantined; its reason now
carries the measurement and the corrected fire trigger.

## ITEM 3 — the receipt superseded, append-only, and the missing producer built

The July receipt had **no producer**: the only thing that could rebuild it was the test that byte-compares
against it. A canonical artifact whose sole regeneration path is its own assertion cannot be refreshed when a
legitimate upstream pin moves — which is exactly the corner ql1 landed in. So the first deliverable here is
`tools/build_ep725_n2_bounded_target_g_receipt.py`, the missing producer: it recompiles the control,
round-trips its own bytes through the canonical parser before anyone can publish them, diffs field-by-field
against the retained receipt, and writes crash-atomically while **refusing to overwrite any existing file whose
bytes differ**.

I ran it `--dry-run` first. It reproduced ql1's three-field delta **exactly**, from a different code path:

| field | retained | current |
|---|---|---|
| `predictor_binding_sha256` | `a25ee501…` | `d5efebd7…` |
| `packet_sha256` | `9139f2a7…` | `d2a70126…` |
| `compile_receipt_binding_sha256` | `23d9775d…` | `32011a33…` |

`changed_field_count = 3`. Every substantive field holds: debt 60,217 → 0, events 21,323 (12,259 births /
9,064 deaths), `packet_bytes` **341,316**, decoded labels == target `6a9ee68a…`, `scorer_invoked` false. The
byte count standing still while the digest moves is the signature of a fixed-length substitution — one 32-byte
digest inside the packet — which is what a pin refresh produces and a regression does not.

**Append-only, and checkable.** The sealed July receipt is retained at file sha `854ffaf3…` — the value its own
July SPEC records — verified unchanged after publication. The new receipt is
`ep725_n2_bounded_target_g_v2_receipt_20260903.json` (file `2a958741…`, receipt `e5c9ac76…`), with a provenance
row beside it naming the EP725 pin-refresh cause, the two refresh commits, the three-field delta and the
unchanged substantive fields. The un-quarantined test now asserts **both** receipts: the current one
byte-exactly, the July one byte-exactly, and that the set of moved fields is exactly those three, cross-checked
against the provenance row's own `changed_field_count`. If a later arm rewrites the July bytes in place, that
test fails. "We superseded it rather than overwrote it" is now a property, not a promise.

## RECALL EVIDENCE

I did not work from the charter alone. Consulted and re-derived rather than adopted: ql1's memo and its four
owed rows (three of whose claims I reproduced independently and two of whose cost estimates I corrected);
`ddm_pl1_payload_loss_two_landing_20260816.md` for the gate's original ten sites and its two documented false
positives — which is how I recognised sites 9 and 10 as the same class and knew a waiver was the intended
disposition rather than a dodge; the gate's own source (`_pl1_primitive_role`, `_pl1_helper_roles`,
`_pl1_scan`) rather than its docstring, which is what showed the prediction's premise to be structurally
impossible for this gate; the July `SPEC_g3_label_local_transport_amendment_20260726.md` for the retained
receipt's sealed shas; `git show e153f2031:<path>` per-source blob hashing for the manifest drift;
`tools/measure_g1_teacher_atom_census.py` line by line for the scorer question, because ql1's "scorer-class"
label was the kind of claim that gets inherited forever if nobody opens the file; and memory
`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`, which is the genus of the whole item — a
pin recorded as law that nobody re-derives when its premise moves.

## DEAD-ENDS

- **`probe_census_delta.py`** — my first V9 probe tried to rebuild the census body and diff it. It refused
  before reaching the diff, at the pair-window/renderer check. Useful only in that the refusal *location*
  (before the decode, in the builder) is what proved the header comes from the sealed packet rather than being
  recomputed.
- **`probe_v9_equivalence.py`** — my second tried to run the full stage/accounting chain with live sources.
  `apply_progressive_geometry_residual` refuses on the same identity at
  `progressive_geometry_residual.py:1127`. The lesson is that the renderer identity is checked at *every* seam,
  not once — so no probe that goes through the residual path can measure equivalence. The third probe
  (`probe_v9_predictor_digest.py`) stops at the receiver's own decode, which is the only seam that does not
  re-check, and that is where the measurement lives.
- **Making site 9 green by respelling `write_bytes` → `write_text`.** Byte-identical output, would have removed
  the violation, and I rejected it: it makes the gate's misread invisible instead of documented.

## OWED — typed fire orders

1. **The 11th payload-write-order site.** `experiments/ddm_ft1_verdict_bhw_pose.py:558` writes a bulk payload
   ahead of the already-built `receipt` (built :478, persisted :561). Genuine, same shape as the eight cured.
   **Fires when ft1's arm lands and releases the file.** Cure: move the receipt write ahead of the save, then
   lower the bound in `test_confound_gates.py` from 1 to 0 and flip the gate STRICT in `tac.preflight`.
2. **V9/PBR2 packet re-seal** (1 test + the census tool's strict mode). Needs **no scorer**. Re-seal the PBR2
   packet with today's renderer identity and carry `EXPECTED_PBR2_SHA256`, the PBR2 materialization receipt and
   the n600 grammar cross-close with it; the equivalence receipt above is the evidence the re-seal is
   output-preserving. Then delete `_QUARANTINE_V9_RENDERER_MANIFEST_PIN`. **Never** narrow
   `RENDERER_SOURCE_PATHS`.
3. **V15 compile receipt** (19 tests). Unchanged from ql1: needs scorer authority. Out of scope here, quarantine
   left intact.
4. **`src/tac/tests/test_fit_ddm_cl1_hpac_capacity.py` is 15 failed / 2 passed — PRE-EXISTING, not mine.**
   I verified by stashing my two edits to that lineage and re-running at HEAD: same 15 failures, same error
   (`CL1FitError: row 0 training receipt names the wrong trainer`, `tools/fit_ddm_cl1_hpac_capacity.py:887`).
   It is outside this charter and outside `src/tac/witness_dsl`, so I did not chase it, but it is red on main
   and somebody owns it. **Fires now**, for whoever owns the CL1 fit lineage.

## NEXT_IF_RESUMED

Fire order 2 — it is CPU-only, its equivalence evidence is already in hand, and it unblocks both the census
tool's strict mode and the last quarantined test in that lineage. Then order 1 the moment ft1 releases its file,
because that single site is the only thing between this gate and a STRICT flip. Separately: ql1's untaken sweep
still stands — the repo-wide `timeout = 60` in `pyproject.toml:286` is likely too small for any other test that
drives a real subprocess decode, and neither arm has looked for siblings rather than following failures.

---

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** (afr1) — UNMOVED by this
arm, as expected: this is apparatus work. Its value is that a gate the suite could not enforce at 11 can be
enforced at 1, a tool that could not run can run, and a receipt that could only be asserted can now be rebuilt.

## ADDENDUM (MAIN, 2026-09-03) — the owed items closed same day

- **11th site cured + STRICT flip.** `experiments/ddm_ft1_verdict_bhw_pose.py` (released when ft1 finished): the
  bulk section write is now try-guarded so the built receipt is persisted even if the save raises
  (`retained_path=None` + `retained_error`), then re-raised. Both laws hold (payload-before-receipt on success;
  record survives a failed save). Fresh strict re-measurement: **11,535 modules, 0 violations** →
  `check_no_bulk_write_strands_the_ready_record` added to `_CONFOUND_STRICT` in `tac.preflight`; test bound 1 → 0.
- **Pre-existing red cured** (`test_fit_ddm_cl1_hpac_capacity.py` 15F at HEAD, "row 0 training receipt names the
  wrong trainer"): root cause MEASURED — `_normalize_argv` called `.resolve()` on argv[0], following the venv
  symlink `.venv/bin/python → python3.13` (rebuilt 2026-08-31), while `_expected_training_argv` deliberately
  leaves `PYTHON_PATH` unresolved (site-packages). Cure: `os.path.abspath` (normalize, never follow) — 17/17 pass.
  Genus: [[m35]] venv-symlink identity; sister of m123 (env-coupled digest).
