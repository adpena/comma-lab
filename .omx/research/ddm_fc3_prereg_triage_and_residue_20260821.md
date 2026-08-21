# ddm_fc3 — prereg triage + the fc2 residue, drained (task #1179)

**All four owed items are closed. The three suspect preregs adjudicate as MUTATED-MATERIAL (the
already-known W3-F14, confirmed not reopened), FALSE-POSITIVE (not a prereg at all), and
MUTATED-BENIGN (its edit window closes before any measurement of its arm existed) — so no
additional F14-class mutation was found. Two corrections to the fc2 memo, both found by measuring
the object: the extrapolation-label class is 8 sites, not 6, and the extrapolation is not merely
mislabelled — it is numerically WRONG, overstating by 1.034×–1.056×, worth +0.0066 S on the largest
emitter.** That last number turns fc2's labelling finding into a numbers finding, which fc2
explicitly scoped out.

| item | verdict / fix | sha | controls both directions |
|---|---|---|---|
| 1 | 3 preregs adjudicated; 2 of 21 registered | `f99d1d4ea2`, `22faf021df`, `0a1c891199` | disposition write/refuse; register/verify INTACT |
| 2 | 8 keys relabelled `DERIVED_extrapolated` + error MEASURED | `78dfa54ae0` | 6 controls incl. PRE-fix vs POST-fix value identity |
| 3 | caller-layer `assert_treatment_was_applied()` | `7c5a6aec7c` | 7 controls + 3 CLI-level exit codes |
| 4 | selftest fixture disclosed | (this memo, §4) | n/a |

---

## ITEM 1 — the three suspect preregs

I re-derived every timestamp from the epoch myself. **All times below are explicit UTC**; macOS
`stat -t` prints LOCAL and this host is UTC−5, which is the trap the round-4 reviewer fell into and
fc2 corrected. Filesystems matter here too: `/Volumes/APDataStore` is **exfat**,
`/Volumes/VertigoDataTier` and the repo are **apfs**.

### S1 — `ddm_fs3/FS3_DROP_FALSIFIER.json` → **MUTATED-MATERIAL** (confirmed, not reopened)

Born `2026-08-21T03:42:20Z`, rewritten `04:07:33Z` (+1,513 s). Already adjudicated by fc2 as W3-F14.
I confirmed independently rather than transcribing: original sha `bd0b293c7441c074…` and disposition
sha `828124d9a453ac42…` both match the fc2 memo exactly, and the disposition receipt is on disk.

**Not registered, deliberately.** Registering an already-mutated file would enshrine its
**post**-mutation bytes in git *as if they were birth bytes* — worse than not registering it. The
disposition receipt is the correct instrument for this file.

### S2 — `.omx/research/ddm_hg1_truncation_and_prediction.json` → **FALSE-POSITIVE**

**It is not a pre-registration.** The census token matched the word `PREDICTION` inside the *probe
name* `hg1_probe3_truncation_and_prediction`. Its keys are a measured truncation ledger plus p1
result rows — a results artifact. It carries no `registered_before_*` / `recorded_before_*` claim.

The filesystem flag is real but benign: the 694 s window (`12:52:58Z → 13:04:32Z`) closes **2m42s
before** its single commit `77cb4a9223` at `13:07:14Z`, and the live file is byte-identical to the
HEAD blob with a clean worktree. It is authoring, then commit, then immutability.

*Residual caveat, stated not hidden:* it does carry `p1_verdict.registered_refutation_band =
"|rs| < 0.5 for ALL four => P1 REFUTED"` co-located with its own outcome, so I cannot prove the band
predates the computation. What bounds the risk is **margin, not provenance**: measured
`best_abs_spearman = 0.8571…` against a band edge of `0.5` is 1.71× clear, so no small threshold
move flips it. Labelled DERIVED-bound, not MEASURED-provenance.

### S3 — `ddm_cx1_20260803/cx1_PREDICTION_prereg.json` → **MUTATED-BENIGN**

This one *is* a real prereg (`recorded_before_gate: true`, a `falsifier`, `predicted_*` fields), so
it got four instruments, not one.

1. **Filesystem** — born `05:34:05Z`, last write `05:40:54Z` (+409 s).
2. **Ordering of the arm's own artifacts** — the window closes **1m53s before** the first measured
   artifact (frame-parity receipt `05:41:26Z`), before the gate log even opens (`05:42:47Z`), and
   **19m02s before** the gate receipt (`05:59:56Z`). This is the exact inverse of the S1 signature,
   where the rewrite landed 26 s *after* the receipt it cites.
3. **Re-derivation** — every material value is recoverable from artifacts that predate the file's
   own birth. `predicted_rate = 0.2355862`; I recomputed `25 × 353808 / 37545489 = 0.2355862244…`
   from an archive fixed at `05:33:42Z`, 23 s *before* the prereg was born. There was nothing to fit.
4. **Committed corroboration** — `ddm_cx1_pj2_container_compose_20260802.md` (`497189b36c`, committed
   `06:00:57Z`) quotes the same predicted value the live file still carries. Bound stated honestly:
   this pins content from `06:00:57Z` forward; instruments 2 and 3 are what cover the actual window.

*Incidental check, cleared:* the cited `archive.zip` carries mtime `06:00:16Z`, **after** its gate
receipt — but its content is unchanged, sha `1d3ab694c337…` byte-identical to the pin in **both**
`cx1_gate_receipt.json` and `cx1_build_receipt.json`, size 353,808 as recorded. Metadata touch only.

Receipts: `.omx/research/ddm_fc3_prereg_mtime_adjudication_20260821.json` (committed, all four
instruments) and `cx1_PREDICTION_prereg.DISPOSITION.json` beside the subject. **No subject file was
edited**; the cx1 sha is `0e4d0a2b4000…` before and after, and the disposition writer REFUSES on a
second run (rc=1).

### The registration split — 2 registered of 21, and why not 21

Two instruments, reconciled rather than trusted:

| instrument | count |
|---|---|
| the tool's census, `--max-depth 4` (fc2's setting) | **24** rows |
| my own `find`, case-insensitive, full depth | **29** files |
| difference | 5 files, all at depth ≥ 5, **all clone duplicates** of 2 already-counted repo files inside retained SSD repo copies |

So the depth bound hides no distinct prereg. Of the 24 rows, 2 are the birth-copy/provenance pair
the tool wrote for itself and 1 is fc2's synthetic fixture → **21 real prereg files**.

**Measured: 0 of 21 are awaiting their measurement.** Every one has a verdict artifact, gate receipt
or `result.json` timestamped after it — rt1 `R4_LADDER_VERDICT.json`, sf1 `SF1_VERDICT.json`, a1s a
committed verdict memo quoting the prereg's own band, up3 `T4_ROW_FALSIFIER_CLOSEOUT.json`, iv1
"§8 — T4 VERDICT: REFUSED", cx1 `prediction_miss 1.25e-08`, fl1 its own scoped verdict, fs3
terminally refused on the measured pose leg, cw1/cd1 their own results. Registering 21 post-verdict
birth copies is exactly the spam the instruction warns against.

I registered the **2** whose authorized work is still ahead, which is a different and narrower claim
than "live":

- **`CD1_PREREGISTERED_DECISION_TABLE`** (`22faf021df`) — its verdict was *"BUILD the corrector
  port"* and the port is not built; its thresholds (529.332 s / 169.332 s) will be re-read when it
  lands. Live lineage: cd1's archive `f3bce5d2…` at 180,625 B is the direct ancestor of the frontier.
- **`CW1_LR1E5D_PREREGISTRATION`** (`0a1c891199`) — its own `source` field says the un-swept
  direction is DOWN, so its seed-ensemble band is the reference for any further down-rung.

Both `verify` **INTACT**, rc=0. Everything else: **DEAD/historical, one line above, no birth copy.**

Not registered for a *structural* reason rather than deadness: the 2 repo-resident preregs
(`ddm_ws1_j5_slope_falsifier_20260724.json`, `ddm_hg1_truncation_and_prediction.json`) are already
immutable in git — I verified live == HEAD blob for both — so a birth copy would duplicate git with
no added guarantee.

**Class-population line.** 21 real preregs · 2 registered · 2 already-immutable-in-git · 2 mutated
and therefore ineligible · 15 post-verdict historical. The `mtime > birthtime` flag remains **one
instrument**: 4 files carry it (the 3 above plus fc2's fixture) and only a committed birth copy
yields a real INTACT/MUTATED verdict.

> **A live false-positive of that instrument, for the record.**
> `ddm_hm1_failed_source_repo_20260810/.omx/research/ddm_hg1_truncation_and_prediction.json` shows
> birth `2026-08-10T12:29Z` and mtime `12:23Z` — mtime *before* birth, the `cp -p` signature. The
> instrument behaves exactly as the caveat describes, in both directions.

**verdict_scope: INSTANCE** per adjudication. The census is a census.

---

## ITEM 2 — the extrapolation-labelled MEASURED keys (`78dfa54ae0`)

**Correction to fc2: the class is 8 sites, not 6.** `lateral_extent_poly_byte_cost` has a degenerate
`n_fit == 0` early-return branch that also emitted both keys; the PRE-fix control confirms it did.
All 8 are fixed across `bulk_boundary_byte_cost`, `horizon_poly_xi_byte_cost` and
`lateral_extent_poly_byte_cost`.

The defect shape is fs3's carrier exactly: `per_frame = best_measured / n` then
`full = round(per_frame * n_frames)`, emitted as `full_bytes_at_n_frames_MEASURED`. Reading the code
found a **second, subtler** extrapolation the memo did not name: the two poly emitters divide by
`n_frames_FITTED`, so an unfittable frame is silently assumed to cost the same as a fitted one. That
is why the emitted provenance names its `extrapolation_basis` rather than assuming one denominator.

**Divergence from the fc2 pattern, deliberately.** A pure rename would relabel the honest case too:
when the coded count equals `n_frames` the value genuinely *is* measured. So the key NAME is stable
in both cases — consumers should not branch on a key name — and the per-call truth lives in
`is_extrapolated`, `extrapolation_factor`, `extrapolation_basis` and a `label_superseded` note that
says which case the call actually is. My first pass got this wrong (it emitted "DERIVED by
extrapolation, NOT measured" even at factor 1.0); **review pass 2 caught it and control 6 covers it.**

**The extrapolation is not merely mislabelled — it is wrong, and I measured how wrong.** fc2
scoped this out (*"I did not measure whether the extrapolation is numerically wrong there"*).
Extrapolating an n=150 prefix of the real `gt_n600` cache to 600 and comparing against coding all
600 for real:

| emitter | extrapolated | measured n600 | error | ratio | ΔS error |
|---|---:|---:|---:|---:|---:|
| `horizon_poly_xi_byte_cost` | 4,360 B | 4,167 B | +193 B | 1.046× | +0.000129 |
| `lateral_extent_poly_byte_cost` | 6,788 B | 6,426 B | +362 B | 1.056× | +0.000241 |
| `bulk_boundary_byte_cost` | 302,984 B | 293,050 B | **+9,934 B** | 1.034× | **+0.006615** |

All three **OVERSTATE**, across two different coders (zlib and brotli) and both bases — the
direction the temporal-context argument predicts. The bulk error alone is +0.0066 S, not negligible
against a 0.148 frontier. The measured magnitudes now travel inside the emitted note itself.

**Committed-memo check, cleared — do not misread the relabel.**
`t5_crucible3/P4_recess_20260709.md` quotes `full_bytes_at_n_frames_MEASURED = 4167`. Re-derived:
4,167 is exactly the n=600 value, where the factor is 1.0. That memo's **number was genuinely
measured** and needs no supersession; only the code's key name was overclaiming.

**Controls executed, both directions.** Every one ran against the PRE-fix HEAD module too, so none
is vacuous.

| control | result |
|---|---|
| all 3 emitters: old keys absent POST-fix, new keys present | PASS |
| all 3 emitters: **values byte-identical** PRE vs POST (5925→5925, 2100→2100, 8925→8925) | PASS |
| **negative:** `n_frames == coded` → `is_extrapolated False`, factor 1.0, `full == best_measured_bytes` | PASS |
| **non-vacuity:** degenerate `n_fit == 0` branch really runs, and PRE-fix it emitted the old keys | PASS |
| review-pass-2 fix: `label_superseded` matches `is_extrapolated`, the two notes differ | PASS |
| consumers migrated: 3 test assertions; **0 other code consumers** (the bare `score_rate_contribution` hits elsewhere are an unrelated key computed from real archive bytes) | PASS |
| the 3 migrated assertions **actually executed** — 15 passed, **0 skipped**, cache present | PASS |
| ruff `--select F` clean; full ruff **8 findings PRE and 8 POST, identical rule codes** — 0 new | PASS |

**Class-population line.** 8 emitted keys in 1 module, all fixed. 3 test assertions migrated. 0 other
code consumers. 2 committed memos cite the old key names — both APPEND-ONLY historical provenance,
**not mutated**, and the one quotable figure in them re-derives as genuinely measured.

**verdict_scope: INSTANCE-SET** for the label fix (the 8 sites). The extrapolation-error measurement
is **INSTANCE-SET and PREFIX-BASED**: measured for these 3 emitters, on this cache, with an n=150
prefix at factor 4. Per the prefix-bias law the first frames are a scene block, not a random sample,
so the 1.034×–1.056× *magnitudes* are specific to this prefix; the *sign* is what the mechanism
predicts and what all three show. A seeded random-sample re-test would tighten the magnitude — not
run here.

---

## ITEM 3 — the archive.zip-must-differ assertion, at the caller (`7c5a6aec7c`)

fc2 deliberately left this open, and correctly: the comparator answers *"do these trees differ ONLY
by payload?"*, and a tree compared with itself is an honest `PAYLOAD_ONLY` for **that** question —
which is why a pre-registered control requires it to keep PASSING there. *"Was a treatment applied at
all?"* is a different question and it now lives at the layer that consumes the verdict to compute a
candidate-vs-base seg/pose/rate delta: `run()` in the same module, the only entry point (no external
caller constructs the args).

`assert_treatment_was_applied()` refuses in two fail-closed ways: **no `archive.zip` present in both
trees** (the `VACUITY==PASS` shape — the comparator would report its strongest verdict having
compared no payload), and **byte-identical `archive.zip`** (the two receipts describe the same
shipped bytes, so any nonzero leg is an instrument artifact). `--allow-identical-archive` exists for
a deliberate null control and is **off by default**, so the null case must be declared, never
stumbled into. Its result is written into the emitted receipt, so the decision is observable.

The presence/difference facts come from the **same hash pass** the comparator already ran, surfaced
as observability fields (`archive_zip_relpaths_in_both`, `archive_zip_differs`) rather than re-hashed
at the caller — deliberately, to avoid two instruments that can disagree.

**Controls executed, both directions.**

| control | comparator | caller |
|---|---|---|
| the **real** fs3 pair | `PAYLOAD_ONLY`, 61 files | **PASSES** |
| retained-receipt regression: 6 shipped verdict fields | **6/6 compared, 0 mismatched** | — |
| **same dir twice** (pre-registered control) | **still `PAYLOAD_ONLY`** | **REFUSES** |
| **distinct dirs, byte-identical archive.zip** | `PAYLOAD_ONLY`, `same_directory=False` | **REFUSES** |
| **negative:** same input + `--allow-identical-archive` | — | **PASSES** |
| **vacuity:** no `archive.zip` in either tree | — | **REFUSES** |
| **negative:** archives genuinely differ | — | **PASSES** |
| non-vacuity: `assert_treatment_was_applied` absent from HEAD | — | confirmed |

CLI exit codes, captured correctly (my first readout took `$?` after a pipe and reported `tail`'s
status — corrected): real pair `rc=1` at the **pre-existing, unrelated** `pact_commit` gate having
passed my check; same-tree-twice `rc=1` with the **new** refusal; same-tree-twice
`--allow-identical-archive` `rc=1` back at the pre-existing gate — proving the flag bypasses my gate
and nothing else. Retained receipt sha `826924405915…` unchanged throughout.

**Class-population line.** 1 caller of `runtime_trees_differ_only_by_payload`, and it is now guarded.
ruff clean PRE and POST; 6 related tests pass.

**verdict_scope: FORMULATION** for "the caller is the right layer" — that follows from the
comparator's pre-registered same-dir control, which I did not re-open.

---

## ITEM 4 — the selftest fixture, disclosed

`FC2_SELFTEST_FALSIFIER` (both `/Volumes/APDataStore/pact/ddm_fc2/selftest/…` and the birth copy at
`.omx/research/preregs/…`) is a **synthetic fixture that ddm_fc2 registered and then deliberately
mutated in place as the MUTATED control** for `register_prereg.py verify`. It is the 4th file
carrying `mtime > birthtime` and it reports `MUTATED` in the census **by design**. It is a control
fixture, **not evidence about any real row**, and must not be filed as an F14 instance.

---

## Still owed

1. **The extrapolation-error magnitude is prefix-based.** A seeded **random-sample** n≥120 re-test
   would replace the 1.034×–1.056× figures with population estimates. The sign is safe; the
   magnitudes are not. Nothing consumes these numbers today, so this is a sharpening, not a blocker.
2. **`bulk_boundary_byte_cost`'s +0.0066 S overstatement** should be re-checked by whoever next
   prices the Road↔Undrivable carrier — the honest n600 figure is **293,050 B / 0.195130**, not the
   prefix-extrapolated 302,984 B / 0.201745.
3. **19 post-verdict preregs remain unregistered by choice.** If any dead arm is ever reopened, its
   prereg should be registered *at that moment* — registering it now would only prove what it says
   today, which is not what a reopened arm needs.
4. **The census's `--max-depth 4` default** silently excludes depth-≥5 files. Harmless today (the 5
   excluded are clone duplicates), but it is a bound, not a proof, and a future arm nesting a prereg
   deeper would be invisible to the routine census.

## Bearing on the pointer

**None.** Every item here is apparatus. No archive, verdict, or score claim moved, and the fs3 row is
still dead on the measured pose leg.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — unmoved by this
unit, as expected.
