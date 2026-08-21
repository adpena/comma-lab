# ddm_fc2 — rv17 wave-3 round-4 findings, cured (task #1178)

**All five round-4 findings are cured, each with controls executed in BOTH directions. Three
comparator fail-opens are closed and the real fs3 pair still reproduces its retained receipt
byte-for-byte on every verdict field. The terminal verdict is untouched: the fs3 row was REFUSED on
the MEASURED pose leg and nothing here changes that — these are instrument cures, not verdict
cures.** Two corrections to the round-4 memo and one divergence from my own instructions are
recorded below, both found by measuring the object instead of transcribing it.

| item | finding | fix sha | status |
|---|---|---|---|
| 1 | W3-F14 prereg rewritten in place | `2619bfdb5d` (+ `4ec02640b2`) | cured, class-guard landed |
| 2 | W3-F15 withdrawn label never reached the code | `6c31ccd21c` | cured, 3 surfaces |
| 3 | W3-F16/F17/F18 comparator fail-opens | `e8388b837c` | cured, A/B controlled |

---

## ITEM 1 — W3-F14: a pre-registered falsifier was rewritten in place

**Defect.** `/Volumes/APDataStore/pact/ddm_fs3/FS3_DROP_FALSIFIER.json` was rewritten **1,513 s
(25m13s)** after birth. It cites `FS3_RATE_BASELINE_RECEIPT.json`, which was born **1,487 s (24m47s)
AFTER the falsifier's own birth** — and the falsifier was rewritten **26 s after that receipt
appeared**, which is the signature of a post-hoc annotation pass. Its first field still reads
`"registered_before_any_build": true`. Consequence: the threshold
`row_dies_if_token_stream_shrinks_by_less_than_bytes = 651.5704466342045` is **unauditable** — I can
prove neither that it moved nor that it did not, which is exactly the question append-only exists to
answer.

I re-derived all three timestamps from the filesystem myself (epochs `1787283740` / `1787285253` /
`1787285227`). **Correction to the round-4 memo:** it records these with a `Z` suffix, but macOS
`stat -t` prints **local** time and this machine is UTC−5. True UTC is `2026-08-21T03:42:20Z` (birth),
`04:07:33Z` (mtime), `04:07:07Z` (cited receipt birth). The **deltas the finding rests on are
unaffected**, so the finding stands unchanged — but the absolute times in the memo are local, not UTC.

**Cure (a) — append-only disposition.** `FS3_DROP_FALSIFIER.DISPOSITION.json` written beside the
original (sha `828124d9a453ac42…`), recording both instruments, both current-state sha256s, the
mutation window, the memo correction, and the plain statement that immutability was violated by a
**good-faith post-hoc annotation through the wrong mechanism** — a mechanism defect, not an integrity
one. **The original was not touched**: its sha is `bd0b293c7441c074…` before and after.

**Cure (b) — the structural class guard.** `tools/register_prereg.py`. `register` copies a prereg to
`.omx/research/preregs/<name>.json` and commits it through the canonical serializer **in the same
call**, so birth bytes enter git history at registration and any later in-place rewrite is diffable
forever. Write-once with no `--force`, refusing on both the working tree **and** git history (a birth
copy committed then deleted from disk is still registered). `verify` compares the live file against
the **committed** blob and reports INTACT/MUTATED; `census` reports coverage.

The guarantee is deliberately narrow and the tool says so in its own docstring: it proves what the
file said **at registration time**, never that registration preceded the build. A prereg registered
late is still a late prereg — this makes that visible instead of arguable.

**Controls executed, both directions.**

| control | result |
|---|---|
| disposition write, file absent | writes (4,645 B) |
| disposition write, file present | REFUSES, rc=1 |
| original falsifier after both runs | sha unchanged `bd0b293c…` |
| register → verify unmutated | **INTACT**, rc=0 |
| mutate live in place → verify | **MUTATED**, rc=1, `json_key_diff` = key added + `FALSIFIER` changed |
| re-register same name (worktree) | REFUSES, rc=2 |
| re-register, birth copy deleted from worktree but in git | REFUSES, rc=2 |
| verify a never-registered name | REFUSES, rc=2 |
| `--max-depth 4` vs a file at depth 5 / depth 5 | excluded / included |
| relative `--source` from another cwd | resolves against caller's cwd |

The MUTATED control reproduces the **exact W3-F14 signature** — a key added and the `FALSIFIER`
threshold changed — which is the point: this is the instrument that would have caught it.

Two defects in my own code were found by the review passes, not by the tests: `census` walked the
entire SSD via `rglob` before applying `--max-depth` (now a pruning `os.walk`, 11.7 s), and a
`os.chdir(REPO_ROOT)` would have silently re-rooted a relative `--source` (removed).

**Class-population line.** **25** prereg-like files (`FALSIFIER`/`PREREG`/`PREREGISTERED`/`PREDICTION`)
across `/Volumes/APDataStore/pact`, `/Volumes/VertigoDataTier/pact` and `.omx/research`. **25 of 25
have no committed birth copy.** 3 show `mtime > birthtime`, including `FS3_DROP_FALSIFIER.json`
itself. That count is a **COUNT, not a verdict**: `mtime > birthtime` is a single instrument (APFS)
and only a committed birth copy yields a real INTACT/MUTATED verdict.

**verdict_scope: INSTANCE** for the F14 mutation. The census is a census.

---

## ITEM 2 — W3-F15: the withdrawal reached the memo and the receipt, never the code

**Defect.** `carrier_MEASURED_leg2` was withdrawn as overclaimed (W3-F7, `fs3:769`) and **both
emitters still published it**, so any future run would re-mint the withdrawn label into a fresh
receipt. Reading the code found a **third** surface the memo did not list: the `compose_legs`
docstring also asserted `carrier -- MEASURED by leg 2`.

The value is a multiplication — `pairs * CARRIER_COMPENSATION_BYTES_PER_PAIR * S_PER_ARCHIVE_BYTE`,
i.e. 45 B measured over 454 pairs in ONE build, re-multiplied to a different pair count.

**Fix** (`6c31ccd21c`), all three surfaces:
- `ddm_fs3_jg5_real_price_reopen.py:583` → `leg_carrier_DERIVED_extrapolated_leg2`
- `ddm_fs3_compose_reopen_candidate.py:377` → `carrier_DERIVED_extrapolated_leg2`
- the `compose_legs` docstring, plus the `CARRIER_COMPENSATION_BYTES_PER_PAIR` comment
- both emitters now also write `carrier_label_superseded`, citing the terminal measurement

I verified the terminal measurement at source before citing it (`fs3:877`): the real build
**180,625 → 179,961 = −664 B** puts the +45 B splice on **both** sides, so leaving the carrier
unchanged makes the carrier **BYTE** leg **EXACTLY ZERO**. The annotation carries the memo's own
caveat rather than stopping at the flattering half: *zero bytes buys a STALE carrier on the changed
pairs, so the COST does not vanish even though the BYTES do.*

The annotation is deliberately placed **outside** the `legs` dict — every value in `legs` is printed
with `:+.6e`, so a prose string there would crash the emitter.

**Controls executed, both directions.**

| control | result |
|---|---|
| compose emitter, **real** CLI path on real retained receipts | new key lands, old key absent, annotation present |
| jg5 `compose_legs` executed on synthetic sweeps | new key lands, old absent, annotation present |
| jg5 control non-vacuous? | `pairs` reopened = 1, carrier `+6.599924e-08` — the carrier branch really ran |
| old key still emitted anywhere in `.py`? | **none** — only inside the withdrawal text that names it |
| ruff / ty | clean; **2 ty errors are pre-existing at HEAD**, 0 new |

**Class-population line.** 294 `MEASURED`-ish occurrences across `experiments/ src/ tools/`, but the
overwhelming majority are authority **values** (`"MEASURED"` as a tag), not keys. Emitted **keys**
that assert MEASURED while their value is computed by multiplication/extrapolation:

- **fs3's 2 — FIXED here.**
- **`src/tac/boundary_math/road_undriv_bulk_field.py` — 6 (3 functions × 2 keys), QUEUED.**
  `per_frame = best_measured / n` then `full = round(per_frame * n_frames)`, emitted as
  `full_bytes_at_n_frames_MEASURED`, with `score_rate_contribution_MEASURED` derived from it. Same
  defect shape as fs3's carrier: a per-unit price measured on a subset, re-multiplied to a different
  count, labelled MEASURED. The code even carries `n_frames_measured` beside it, so it already knows
  the two counts differ — the label is the overclaim.
- Checked and **cleared** as genuine direct measurements: `intrinsic_annulus_rate_MEASURED` (mean of
  a measured array), `primary_byte_delta_MEASURED` (a direct byte difference), `seg_MEASURED`,
  `rate_MEASURED_real_reencode`.

**verdict_scope: INSTANCE-SET.** I read the computation at the 6 queued sites and classified the
*label*; I did **not** measure whether the extrapolation is numerically wrong there. That is a
labelling finding, not a numbers finding.

---

## ITEM 3 — W3-F16/F17/F18: three comparator fail-opens

All three were latent this run and all three returned the module's **strongest** verdict,
`PAYLOAD_ONLY`, on unscrutinised input. Fixed in `e8388b837c`.

**(a) F16 — the pin normaliser erased whole expressions.** It replaced `node.value` with no check on
the right-hand side, so an RHS with executed side effects normalised away. Now the RHS must be a
single literal, and anything else raises `PoseLegError` naming the file, the pin and the node type.

> **Divergence from my instructions, and why.** The instruction said "a single `ast.Constant`
> **string** literal". I measured the real pins first: `ARCHIVE_SHA256` is a **str**, `ARCHIVE_BYTES`
> is an **int** (`180625` / `179961`). Requiring a *string* would refuse the real pair and break the
> instruction's own control ("the real candidate/base pair still returns `ast_identical True`"). I
> implemented **single `ast.Constant` of type str or int** — `bool` excluded via `type(...) in`
> rather than `isinstance`, since `bool` subclasses `int`. This closes the defect exactly: the hole
> was "an arbitrary *expression* is erased", not "a non-string is erased".

**(b) F17 — a nested `inflate.py` bypassed every check.** `unexpected` filtered on **basename** while
the AST check keyed on the exact string `"inflate.py"`, so a differing `lib/inflate.py` satisfied the
filter *and* missed the AST check. Both now key on the **exact relative path**, and **every**
differing file whose basename is `inflate.py` is AST-checked.

**(c) F18 — the gate passed vacuously.** Empty and nonexistent trees now raise `PoseLegError`.
`is not False` became a **conditional** `is True`.

> **Two notes where I did not follow the memo's cure, deliberately.**
> 1. The memo proposed refusing when `base.resolve() == cand.resolve()` and requiring `archive.zip`
>    in `differing`. My instructions state the opposite control — *same dir twice must still PASS
>    (legitimately identical)* — and I followed the instruction. This comparator answers "do these
>    trees differ only by payload?"; "was the treatment applied at all?" is the **caller's** question.
>    To keep the vacuity visible rather than hidden I added an observability field,
>    `base_and_candidate_are_the_same_directory`. **The archive.zip-differs assertion is QUEUED for
>    the caller layer, not adopted here.**
> 2. An **unconditional** `is True` contradicts that same-dir control (`differing=[]` → `None` →
>    would refuse). So the invariant is: *if any differing path is named `inflate.py`, the check must
>    have run and returned True.* The `None` fail-open is closed **structurally** by the (b) fix —
>    "the check never ran" can no longer coexist with "an `inflate.py` differs".

**Correction to the round-4 memo (F18).** The memo says "Nonexistent dirs, two empty dirs, and the
same dir twice **all** return `PAYLOAD_ONLY`". Measured pre-fix: **both** dirs nonexistent →
`PAYLOAD_ONLY` (a genuine fail-open), but **one** dir nonexistent → `RECEIVER_CHANGED` (already
refused, incidentally, via the set difference rather than by any precondition check). The claim is
right for the symmetric cases and overstated for the asymmetric one.

**Controls executed, both directions.** Every control was run against the **PRE-fix (HEAD)** module
as well, so none of them is vacuous:

| control | PRE-fix | POST-fix |
|---|---|---|
| (a) pin RHS = call with side effect | `PAYLOAD_ONLY` | REFUSED |
| (a) pin RHS = f-string / tuple / walrus | `PAYLOAD_ONLY` | REFUSED |
| (a) pin RHS = bare name | `RECEIVER_CHANGED` | REFUSED |
| (a) **negative:** plain literal pins differ | `PAYLOAD_ONLY` | `PAYLOAD_ONLY` |
| (b) differing `lib/inflate.py` | `PAYLOAD_ONLY` | `RECEIVER_CHANGED` |
| (b) **negative:** differing `lib/other.py` | `RECEIVER_CHANGED` | `RECEIVER_CHANGED` |
| (c) both dirs nonexistent | `PAYLOAD_ONLY` | REFUSED |
| (c) two empty dirs | `PAYLOAD_ONLY` | REFUSED |
| (c) **same dir twice (must PASS)** | `PAYLOAD_ONLY` | `PAYLOAD_ONLY` |
| (c) no `inflate.py` differs → `None` legitimate | `PAYLOAD_ONLY` | `PAYLOAD_ONLY` |
| CLI-level typo'd `--base-runtime` | — | refuses, **rc=1** |

**Regression — the shipped comparison is unchanged.** Re-running the real pair
(`ddm_jg5/candidate_runtime_jg5` vs `ddm_fs3/candidate_runtime_drop137`) against the retained
`FS3_SAME_INSTRUMENT_LEGS.json`: `files_only_in_base`, `files_only_in_candidate`, `differing_files`,
`unexpected_differing_files`, `inflate_py_AST_identical_once_pin_normalised`, `verdict` — **all six
identical, zero mismatches.** New observability: `inflate_py_paths_ast_checked=['inflate.py']`,
`files_compared=61`, `base_and_candidate_are_the_same_directory=False`. Note the candidate tree does
contain a nested `cpr1/inflate.py`; it is byte-identical between the trees, which is why F17 was
latent rather than live.

**Class-population line.**
- **basename-vs-exact-path:** 1 other site in our own code, `ddm_ck2_build_receiver_overlay.py:240`.
  It iterates a flat `dest.iterdir()`, where `p.name` **is** the relative identity, and it is
  fail-closed. **Not a defect; 0 queued.** All other hits were vendored `site-packages`.
- **`is not False`:** ~12 sites in our code. Every one sampled sits in a **REFUSE** conjunction
  (`if X is not False: raise`), where `None` → refuse — the *correct* polarity. The comparator was
  the single instance where it sat inside an **acceptance** conjunction (`ok = ... and (...)`), where
  `None` → PASS. **0 queued.**
- **Empty/nonexistent-input vacuity** (the `VACUITY==PASS` genus) was **not** swept exhaustively. I
  state that as scope rather than claiming a negative-existence result.

**verdict_scope: FORMULATION** for "these three are latent, not live, in this run" — that is a
property of *this* pair of trees (plain-literal pins, no differing nested `inflate.py`), not of the
comparator's inputs in general.

---

## Owed / queued

1. **6 extrapolation-labelled MEASURED keys** in `src/tac/boundary_math/road_undriv_bulk_field.py`
   (3 functions × 2 keys) — same defect class as fs3's carrier. Not fixed here per instruction.
2. **The `archive.zip`-must-differ assertion** — belongs to the caller of
   `runtime_trees_differ_only_by_payload`, deliberately not added to the comparator.
3. **25/25 prereg-like files have no committed birth copy.** `tools/register_prereg.py register` now
   exists; the live preregs still need registering.
4. **Self-test artifacts, disclosed:** `FC2_SELFTEST_FALSIFIER` is a synthetic prereg registered at
   `4ec02640b2` and then **deliberately mutated in place** as the MUTATED control. It is a control
   fixture, not evidence about any real row.

## Bearing on the terminal verdict

**None.** The fs3 row is dead on the measured pose leg and every finding above is an instrument
finding. What changes is that the 1.9 % margin's second caveat is now recorded in an append-only
receipt beside the file it concerns, the withdrawn carrier label can no longer be re-minted by a
future run, and three fail-opens that would have returned `PAYLOAD_ONLY` on unexamined input now
refuse.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]** — unmoved by this
unit, as expected: this is apparatus, not a score mover.
