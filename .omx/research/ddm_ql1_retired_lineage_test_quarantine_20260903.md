# ddm_ql1 — the witness_dsl suite is honest again: 51 red tests adjudicated to 4 root causes, 1 pin refreshed with a receipt, 21 quarantined with provenance, 0 regressions

**Arm:** `ddm_ql1_retired_lineage_test_quarantine` (gen 2; gen 1 was a codex arm that died to a usage
limit with its work HELD, not landed).
**Charter:** `.omx/research/charters/ddm_ql1_retired_lineage_test_quarantine_20260903.md`
**Directive:** `.omx/research/ddm_cd1_directive_test_drift_ledger_20260903.md`
**Craft:** `docs/operating_manual_craft_handoff.md` — §4 (re-derive, never recognize) and §3 (put depth
where a silent error does the most damage) decided every call below; §6 is why the ep725 refresh was
re-derived from first principles instead of re-applying the held patch.

## HEADLINE

The 51 red tests were **4 root causes, not 8 modules**, and **none was a regression**. One source pin
was stale and is now refreshed with a bit-exact decode receipt; two custody pins belong to a retired
lineage whose cure needs the scorer and are quarantined with `xfail(strict=True)`; one fixture had
already been repaired by cd1. Peeling the pin then exposed a **fifth thing the red baseline was
hiding: a pytest-timeout budget too small for the tests' own declared cost**, and behind one of those
timeouts sat a real assertion failure that no run had ever reached.

**Suite: 35 failed / 16 errors → 0 failed / 0 errors.** Every remaining non-pass is a documented
strict-xfail that will FAIL the moment its lineage is repaired.

## SUITE COUNTS

| | failed | errors | passed | xfailed | total | wall |
|---|---|---|---|---|---|---|
| BEFORE (MAIN, clean, pre-cd1-repair tree) | 35 | 16 | 1324 | — | 1375 | 1749.98 s |
| AFTER (this arm, HEAD + these edits) | **0** | **0** | 1354 | 21 | 1375 | 1294.38 s |

The totals reconcile — 1324+35+16 = 1354+21 = 1375 — so no test was lost, skipped or deleted; the 51
red outcomes became 30 passes and 21 documented strict-xfails. The suite also got **455 s faster** on a
*more* loaded machine, which is the timeout fix showing up as arithmetic: four tests were each burning
a full 60 s stall before dying. AFTER log: `runs/after_20260903/run.log`.

BEFORE is MAIN's retained log
`/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/witness_dsl_full_suite_20260903.log`.
I also launched my own HEAD baseline and **discarded it**: it aborted with a pytest INTERNALERROR
(958 passed, not 1324) after contention-induced timeouts corrupted a longrepr. I say so rather than
quote it, because a number from a crashed run is not a number. Retained anyway at
`runs/before2_20260903/run.log`. MAIN's log has **zero** `Timeout (>60` hits, which is what let me
prove the timeouts I saw were mine, not the suite's.

## PER-MODULE DISPOSITION

| Module | Red before | Root cause | Disposition | After |
|---|---|---|---|---|
| `test_ep725_levelset_predictor_adapter` | 2 F | EP725 renderer source pin, then a 60 s budget | **PIN-REFRESH** + timeout budget | pass |
| `test_taskspace_g17_actuator_ir_v1` | 8 E | EP725 pin (fixture `real_pair_bundle`) | cured by the refresh | 8 pass |
| `test_taskspace_monolithic_pga_receiver` | 20 F | EP725 pin (~13), then pytest-timeout (7) | refresh + **TIMEOUT BUDGET** | 35 pass |
| `test_bounded_target_g_encoder` | 1 F | EP725 pin → downstream digest chain | **QUARANTINE** (refresh owed, receipted) | 1 xfail |
| `test_taskspace_selected_preimage_program_v1` | 1 F | V15 producer custody pin | **QUARANTINE** | 1 xfail |
| `test_taskspace_selected_preimage_program_v2` | 7 F | V15 producer custody pin | **QUARANTINE** | 7 xfail |
| `test_taskspace_g17_g49_..._bridge_v1` | 3 E | V15 producer custody pin | **QUARANTINE** | 3 xfail |
| `test_taskspace_g72_..._factor_compiler_v1` | 1 F | V15 producer custody pin | **QUARANTINE** | 1 xfail |
| `test_taskspace_g82_tsppv2_pvsa1_lowering_v1` | 2 F + 5 E | V15 producer custody pin | **QUARANTINE** | 7 xfail |
| `test_taskspace_inverse_stack_receipt` | 2 F | V9/PBR2 manifest pin (1); stale target fixture (1) | **QUARANTINE** (1); cd1 already fixed (1) | 1 xfail |

**REGRESSIONS: 0.** The charter's falsifier was "any module whose current output differs from its
receipt AND has a live consumer". No module in scope has one: every caller of every red module sits
inside the same July taskspace/ep725 lineage. `submissions/semantic_joint_ctxmix/`,
`src/tac/semantic_pipeline/`, `src/tac/submission_chain.py` and `src/tac/contest_score*` import none
of it.

Two near-misses, both chased down rather than waved off:

- `src/tac/canonical_frontier_pointer.py` names `taskspace_inverse_stack_receipt` at line 339 **inside
  a docstring**. There is no import; the frontier pointer does not run that path.
- A Track-B staged packet *does* reach a quarantined module transitively:
  `submissions/robust_current/taskspace_pvsa_staging/inflate.py` → `taskspace_g85_pvsa_public_receiver_v1`
  → `taskspace_g74_v15_roleaware_overlay_decoder_v1` → `taskspace_selected_preimage_program_v1`. I
  state it rather than round it down to "no consumer". It still does not change the disposition, for
  two measured reasons: g74 imports exactly one name from that module — the enum
  `SelectedPreimageFrameSelectorV1` — and **none** of g74, g85 or the compact container calls
  `verify_v15_semantic_compile_lineage`, which is the function that raises. The staged packet's runtime
  path never touches the drifted check. `robust_current` is also the Track-B research track, not the
  shipped vehicle.

## ROOT CAUSE 1 — EP725 renderer source pin → PIN-REFRESH, with the receipt cd1 said was owed

`EP725_CANONICAL_RENDERER_SHA256` hashes the **source text** of
`tools/levelset_byte_close_and_eval.py`. It is a source pin, not an output hash. Two commits moved it,
not one — cd1's ledger named only the second:

| commit | date | producer sha | what it touched |
|---|---|---|---|
| (pinned) | 2026-07-21 | `1cecaa3e…` | — |
| `f73bfb4e8` | 2026-08-03 | `00106018…` | `_parse_evaluate_report` + dropped `import re` |
| `90d537745` | 2026-08-26 | `35d6ce6b…` | `run_inflate`: `extractall` → `safe_extract_zip` |

Neither touches the three callables the adapter requires. That is not an argument, it is measured
twice: I ran the decode, and a second agent hashed the function bodies at `f73bfb4e8~1` and HEAD —
`_read_blob_bytes_full` `5dab864b913d78ee`, `_dequant_blob` `c46719fe7fcc56e0`,
`numpy_oracle_reference_frames` `728a5c1ec5cb29f7`, byte-identical on both sides.

**The receipt** (`receipts/ep725_decode_receipt_20260903.json`, real custody, n1 and n2):

- both decoders deterministic; shipped runtime raw == canonical NumPy raw;
- `chronological_camera_frame_sha256`, `chronological_raw_prefix_sha256`, `labels_sha256` —
  **bit-exact to the frozen values**;
- only three digests moved, and they moved because `predictor_renderer_sha256` is an *input* to
  `_expected_semantic_binding`.

I re-derived all six refreshed digests from first principles rather than reading them back from a
test (`receipts/ep725_receipt_sha_derivation_20260903.json`), and they reproduce the held codex
patch's values exactly. That independent agreement is why the patch was safe to land — not the fact
that it made tests pass. cd1's condition ("PIN-REFRESH-WITH-RECEIPT or quarantine; a hash-only edit
is forbidden") is met.

## ROOT CAUSE 2 — V15 producer custody pin → QUARANTINE (19 tests)

The sealed 2026-07-26 compile receipt pins two producer sources. One is clean; one drifted:

| producer | pinned | HEAD |
|---|---|---|
| `tools/measure_ddm_v15_scorer_solved_templates.py` | 33,966 B `a39dad6f…` | identical |
| `src/tac/optimization/direct_description_carrier_compose.py` | 156,551 B `3e1f69bb…` | 160,470 B `6fef110d…` |

Chain: `b8f24ae63` (the pinned blob) → `9934d488b` → `36f4b2947`, all 2026-08-20. The delta is exactly
3 changed defs, 2 added, 0 removed — and all three **are** on the path the pinned producer calls, so
this is not dismissible as unrelated hardening. It is nonetheless measured output-equivalent: the
drifted receiver still decodes the sealed archive and refuses all 5 mutation samples with identical
coverage (133,941 B); the only receipt delta is an **added** key, `non_empty_member_payload_count`,
that no check reads.

**Why quarantine and not refresh, given that evidence.** The pin lives in a SEALED custody receipt.
Rewriting its `producer_custody` to match today's sources would make that receipt assert it was
compiled by sources that did not exist at compile time — a false statement in a canonical data field.
The honest cure is a NEW receipt from a fresh compile, and
`tools/measure_ddm_v15_scorer_solved_templates.py` solves *"through exact R + SegNet"*: it runs the
frozen scorer, which this charter forbade. So the receipt is structurally out of this arm's reach, and
I say that rather than manufacture one.

Also recorded for whoever fires it: the same file text is re-checked at
`taskspace_selected_preimage_program_v1.py:1923` and `..._v2.py:723`. Only a receipt refresh cures all
three sites; waiving the `:572` check alone would not.

## ROOT CAUSE 3 — V9/PBR2 renderer manifest pin → QUARANTINE (1 test)

`CensusError: exact V9/PBR2 pair window or renderer identity differs` ORs five conditions. Isolated by
probe: `pair_start` 448, `window` 64, and both receiver fields all still match. **Only the renderer
identity differs**: sealed `92ab2350…` vs live `715e3e57…`. cd1 recorded one drifted source; there are
**three** of the manifest's 13: `predictor_upgrade_xi_chart.py` (merge `8181d8763`, 2026-07-27),
`direct_description_minimizer.py` and `direct_description_carrier_compose.py` (both `b8f24ae63`
onward). The seal itself is faithful — all 13 sealed digests reproduce exactly at `e153f2031`, main's
tip at seal time. Rematerialization is scorer-class, so again out of reach here.

**Owed beyond the test:** `tools/build_taskspace_inverse_stack_receipt.py:34` calls
`build_stack_receipt(strict_source_reopen=True)` with no way to disable it, so this drift breaks that
**tool**, not only its test. Retired lineage, no submission-chain consumer — but it is a broken tool,
and that belongs in the ledger.

## ROOT CAUSE 4 — stale competitive-target fixture → ALREADY FIXED by cd1

`test_changed_pointer_recomputes_target_and_conditional_ceiling` forced the upstream row to `0.16` and
expected the effective frontier to follow. Once afr1 crossed below `0.162`, our own CUDA row became the
minimum and the premise expired. cd1 repaired it in `563b093e3` (target `0.1`). I verified the live
pointer independently: local CUDA `0.14797617125559104`, local CPU `0.1880443979880752`, upstream best
`0.162`, effective = the CUDA row. The production code implements CLAUDE.md's `min(...)` rule
correctly; only the fixture had gone stale. Textbook
`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`.

**The three canonical-equation "surfaces once" tests: VERIFIED PASS** (3 passed in 0.59 s). cd1's
registry repair held.

## WHAT THE RED BASELINE WAS HIDING — the finding I did not expect

`pyproject.toml:286` sets a repo-wide `timeout = 60`. The ep725 cold decode runs **four** full decodes
(canonical NumPy ×2, shipped-runtime subprocess ×2). Measured cost, with the timeout raised:

```
73.01 s  test_real_explicit_member_runtime_decode_is_source_independent_and_causally_bound
63.86 s  test_real_ep725_n2_v2_control_exactly_reconstructs_frozen_target_without_transport
63.70 s  test_real_ep725_n2_double_decode_is_hash_bound_bit_exact_and_label_owned
```

All three exceed the ceiling; one shipped n2 decode alone costs 15.6–16.7 s. The fixtures ask the
adapter for `timeout_seconds=120.0`/`180.0` — the harness was contradicting the tests' own declared
intent. The PGA file already carried a single `@pytest.mark.timeout(180)` and that one test passed,
which is the whole explanation of its `……FFFFFFF……` pattern: the cold surface is a module-scoped
cache, and only the test that happened to fill it had the budget.

So the budget belongs at module scope, not on one test — any test can be the one that pays. 180 s is
not a number I invented: it is what the module's own decode call asks for and what the file's
pre-existing mark already used, and it is 2.5× the worst measured cost. This raises a watchdog only;
no assertion changed. **PGA went 7 failed / 588 s → 35 passed / 176 s.**

**And behind one of those timeouts sat a real assertion no run had ever reached.** With the budget
raised, `test_bounded_target_g_encoder.py:179` failed on `packet_sha256` — masked first by the pin,
then by the clock. Measured (`receipts/bounded_target_g_delta_20260903.json`), the fresh receipt
differs from the retained one in **exactly three fields**, all provenance digests from one cause:

| field | retained | fresh |
|---|---|---|
| `predictor_binding_sha256` | `a25ee501…` | `d5efebd7…` |
| `packet_sha256` | `9139f2a7…` | `d2a70126…` |
| `compile_receipt_binding_sha256` | `23d9775d…` | `32011a33…` |

`d5efebd7…` is precisely the n2 state binding I re-derived in the EP725 receipt: the packet folds
`predictor_state.binding_sha256` into `pbr1_sha256`. Every substantive field is unchanged — debt
60,217, events 21,323, births 12,259, deaths 9,064, packet_bytes 341,316, decoded labels == target
`6a9ee68a…`, `scorer_invoked` False. The byte count holding still while the digest moves is the
signature of a fixed-length substitution. **A pin-refresh consequence, not a regression.** I
quarantined rather than refreshed it because the cure needs the sealed July receipt
`ep725_n2_bounded_target_g_v2_receipt.json` regenerated, and I had just refused that same class of
move for V15. The evidence is sufficient; the authority is not mine. One step for MAIN.

## QUARANTINE MECHANISM

`pytest.mark.xfail(strict=True, reason=…)`, never a deletion, never a weakened assertion. Verified by
probe before use (`probe/test_xfail_setup_probe.py`): it **covers a fixture SETUP error**, so the 16
ERRORs are quarantinable; and it reports **XPASS as FAILED**, so every quarantine self-reports the
moment its lineage is repaired. Each reason carries the pin, the drift commits, the measured evidence,
the owning memos, and a typed FIRE TRIGGER.

## SEMANTIC-VEHICLE TESTS — one pre-existing red, NOT mine

The charter required the live vehicle's tests to stay green and separate. `test_semantic_pipeline.py`
+ `test_confound_gates.py`: **1 failed, 184 passed**. The failure is
`test_real_repo_live_count_bounded[check_no_bulk_write_strands_the_ready_record]` — `assert 11 <= 10`,
a gate whose repo-wide live count has grown past its cap.

It is not mine, and I checked rather than asserted that: I ran the gate directly and read all 11
violations. They sit in `src/tac/optimization/tests/`, `src/tac/substrates/hi_nerv/`, `src/tac/tests/`,
`tools/` and `experiments/`. **Not one is under `src/tac/witness_dsl/`**, and my diff adds no write or
save call anywhere — only comments, marks and sha literals. The count grew through another arm's
landing. I left it alone: the three newest offenders belong to other lineages, and editing them
mid-flight is not this arm's business. Flagged for MAIN as an unrelated live-count debt.

## OWED — typed fire orders for MAIN

1. **V15 compile receipt** (19 tests). A scorer-authorized arm re-runs
   `tools/measure_ddm_v15_scorer_solved_templates.py`, emits a fresh receipt, and either confirms
   byte-identity — then deletes `_QUARANTINE_V15_PRODUCER_PIN` — or records real output drift. Fires
   when scorer authority exists. Note the two extra hash sites above.
2. **V9/PBR2 rematerialization** (1 test + 1 broken tool). Reseal the renderer manifest; three of 13
   sources moved. No blind rewrite. Also unblocks `tools/build_taskspace_inverse_stack_receipt.py`.
3. **bounded-target-G receipt** (1 test). Regenerate
   `.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_bounded_target_g_v2_receipt.json`
   and refresh three literals against it. **No scorer needed** — pure CPU, and the delta is already
   measured. Cheapest of the three; fires on an authority decision, not on new evidence.

## RECALL EVIDENCE

I did not work from the charter alone. Consulted: cd1's landing memo and its four-root ledger (which
my independent derivation reproduced, and refined in three places — a second EP725 drift commit, three
V9 manifest rows instead of one, and the PGA cluster being timeouts rather than pins); cd1's directive
and its verbatim failure list; MAIN's retained full-suite log, which I mined for root-cause grouping
and for the zero-timeout fact that exposed my own contention confound; the live
`canonical_frontier_pointer.json` rather than the log's stale reading of it — which is how I learned
cd1 had already fixed the fixture the log showed failing; `git log -S` and per-commit blob hashing for
every pin; the AST import graph over `src/tac/witness_dsl` plus repo-wide greps for the live-consumer
question; and memory `[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`, which named
root cause 4 before I measured it. Two opus agents ran the per-module investigation on disjoint file
groups; I re-ran their decisive claims myself rather than adopting them.

## LIVE-HYPOTHESES

- The V15 drift is output-equivalent for the sealed archive, so a fresh compile will confirm
  byte-identity and fire order 1 will end in a deletion, not a drift record. (Measured on the
  fail-closed proof and the decode; unmeasured on the compile itself.)
- The 60 s ceiling is likely too small elsewhere in the repo for any test that drives a real
  subprocess decode. This arm fixed the four files it measured; it did not sweep for siblings.

## DEAD-ENDS

- My own HEAD baseline suite. Launched concurrently with two investigating agents; the contention
  produced timeouts that MAIN's quieter run did not have, and pytest then crashed formatting one of
  their longreprs. Recorded and discarded. The lesson is the ordinary one: do not measure a suite while
  two agents are running it.
- Re-applying the held codex patch directly. It was correct, but landing it on its own evidence would
  have been agreeing with the test. Deriving the same six digits independently cost one probe and made
  the refresh defensible.

## NEXT_IF_RESUMED

Fire order 3 first — it is CPU-only, one step, and its delta is already measured. Then sweep the repo
for other tests whose declared `timeout_seconds` exceeds the 60 s ceiling; this arm found four such
files by following failures, not by looking.

---

Own-vehicle frontier: **S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** (afr1) — UNMOVED by
this arm, as expected: this is apparatus work. Its value is that a suite with 51 known-red tests could
not have caught a 52nd, and now it can.
