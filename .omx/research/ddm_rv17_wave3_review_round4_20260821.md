# ddm_rv17 — wave-3 round 4: 5 findings adopted from arm `ad3653c5680247bda`, ALL re-verified at source

**Counter: 0/3** (already reset in round 3; these findings hold it there).

**Provenance.** Arm `ad3653c5680247bda` (items 2 and 3 of the scoring round) returned a report.
Per my standing rule I adopted nothing on its word. I re-verified every load-bearing claim below
at source with my own instruments. **All five survived.** This is the first arm output this wave
that survived verification intact on every checked claim, and it did so because the arm practiced
the three instrument laws I bound it to — it disclosed its own instrument sharing unprompted and
disproved eight of its own suspicions before filing.

---

## W3-F14 (HIGH) — a pre-registered falsifier was rewritten in place

| field | value |
|---|---|
| **file** | `/Volumes/APDataStore/pact/ddm_fs3/FS3_DROP_FALSIFIER.json` |
| **instrument 1 (filesystem)** | `birth=2026-08-20T22:42:20Z`, `mtime=2026-08-20T23:07:33Z` — a **25-minute gap**. |
| **instrument 2 (content)** | The file cites `FS3_RATE_BASELINE_RECEIPT.json`, whose `birth=2026-08-20T23:07:07Z` — **25 minutes after the falsifier was born**. A file cannot at birth cite an artifact created later. It also contains `W3_F6_baseline_receipting`, `W3_F3_F8_control_triad_honesty`, `W3_F5_control_a_vacuity` — wave-3 material that did not exist at 22:42. |
| **independence** | These two cannot fail together. One is APFS metadata; one is file content. If birthtime were unreliable, the citation still stands. My own instrument-diversity law is satisfied. |
| **the claim it contradicts** | The file's own first field is `"registered_before_any_build": true`. |
| **what this is, and is not** | The material added is **the disposition of my own wave-3 findings**. This is a good-faith act of recording, performed through the wrong mechanism — a successor was owed, an in-place edit was taken. It is a MECHANISM defect, not an integrity one, and I state that plainly. |
| **consequence** | A prereg's entire value is immutability. `row_dies_if_token_stream_shrinks_by_less_than_bytes = 651.5704466342045` is now **unauditable**: I can prove neither that the threshold moved nor that it did not. That is precisely the question append-only exists to answer, and it is the threshold the mirror's 664 B survival (12.4 B, 1.9%) is measured against. |
| **bearing on the terminal verdict** | **None.** The row was REFUSED on the measured pose leg regardless. But the memo instructs that the 1.9% margin be quoted with its caveat wherever it travels — and there is now a **second** caveat: the line it clears sits in a file whose immutability cannot be shown. |
| **verdict_scope** | INSTANCE |
| **cure** | Re-register as `FS3_DROP_FALSIFIER.v2.json` with `supersedes` + `edit_disclosed: true`; state in the memo that v1's threshold immutability is unprovable. Never touch v1. |

## W3-F15 (MED-HIGH) — the withdrawal reached the memo and the receipt, not the code

`experiments/ddm_fs3_jg5_real_price_reopen.py:561` → `"leg_carrier_MEASURED_leg2": carrier,`
`experiments/ddm_fs3_compose_reopen_candidate.py:359` → `"carrier_MEASURED_leg2": carrier,`
`grep -c "DERIVED_extrapolated\|withdrawn"` over both files returns **0 and 0**.

The label was withdrawn as overclaimed (`fs3:769`), and both emitters still publish it. **Any future
run of either script re-publishes the withdrawn MEASURED label into a fresh receipt.** A withdrawal
that lives only in prose is not durable. This is my W3-F1 genus reaching its furthest surface yet:
the caveat failed to travel not into a headline or a registry, but into the code that mints new claims.
*Cure:* rename both keys to `carrier_DERIVED_extrapolated`, emit `carrier_label_superseded` alongside.

## W3-F16 (HIGH, latent) — a receiver-behaviour change can hide inside pin normalization

`experiments/ddm_fs3_same_instrument_pose_leg.py:147-150`
```
if isinstance(node, _ast.Assign):
    if isinstance(target, _ast.Name) and target.id in PIN_CONSTANTS:
        node.value = _ast.Constant(value="<PIN>")
```
The normalizer replaces `node.value` — **the entire right-hand expression** — with no check that the
RHS is itself an `ast.Constant`. Any expression assigned to a pin name is erased, including one with
executed side effects. The arm demonstrated it: an RHS binding a module global
(`globals().__setitem__("DECODE_MODE","ALT_RECEIVER")`) normalizes away and the comparator returns
`PAYLOAD_ONLY | ast_identical: True` for two files with **different executed semantics**. A walrus does
the same. This is the crux question answered: **yes, a receiver-behaviour change can hide inside the
normalization step.** *Cure:* refuse unless the pin RHS is an `ast.Constant`.

## W3-F17 (HIGH, latent) — a nested `inflate.py` bypasses the AST check entirely

`:160` filters on **basename** — `[k for k in differing if Path(k).name not in PAYLOAD_FILES]` —
while `:163` keys the AST check on the **exact relative-path string** `if "inflate.py" in differing`.
A differing `lib/inflate.py` therefore satisfies the basename filter (so it is never "unexpected")
**and** fails the exact-string test (so it is never AST-checked). Zero scrutiny, and it needs no
adversarial intent: a vendored or backup copy triggers it. *Cure:* compare on full relative path;
AST-check **every** differing path whose basename is `inflate.py`.

## W3-F18 (HIGH) — the fail-closed gate passes vacuously on an empty file set

`:172` reads `and (inflate_body_identical is not False)` — so `None`, the never-ran case, passes.
`grep` finds no empty-set guard and no same-directory guard. Nonexistent dirs, two empty dirs, and
the same dir twice all return `PAYLOAD_ONLY` — the strongest-sounding verdict in the module — from a
typo'd `--base-runtime`. There is also no assertion that `archive.zip` actually differs, i.e. that
the treatment was applied at all. *Cure:* refuse on empty hashes; refuse when
`base.resolve() == cand.resolve()`; require `"archive.zip" in differing`.
**The arm disclosed that its three probes here share one instrument** (`_file_hashes` returning an
empty-or-equal dict) and are one check with three inputs. I adopt that framing; the defect is
confirmed by reading the code, which is a second and independent instrument.

---

## What I credit the arm for, and what I still hold

It **disproved eight of its own suspicions**, including two I would have been tempted by: that the
F2 mislabel flips `clears_sub_015` (it does not at 4.1379 — 0.14851 < 0.15; only the *aggregation
level* flips it), and that the transfer caveat lives only in the docstring (it is a key in the
written receipt). It also confirmed the shipped comparison reproduces exactly, so **F16 and F17 are
latent in this run, not live** — the real pin lines are plain literals and neither tree holds a
nested `inflate.py`. The shipped verdict stands on this run's evidence.

Two of its rows I record but do not adopt as findings, because they rest on a single instrument the
arm itself flagged (`stat` birth-vs-mtime, uncorroborated): the census and retention-manifest edits.
Rated LOW/MED by the arm on exactly that ground, which is the correct call.

## Honest state

- **Counter 0/3.** Seven open findings this wave (W3-F12, F13, F14–F18).
- **The substance still has not moved.** No round in 20 + 11 + 4 has found a wrong score, a wrong
  pin, a wrong digest, a mis-scoped receipt, or an unverifiable archive claim. The terminal verdict
  (REFUSED on the measured pose leg) is untouched by every finding above.
- **Wave 3 is producing exactly one genus, and it has now been traced to its end.** A correct number
  whose obligation stopped one surface short: the warrant (F12), the registry (F13), the successor
  receipt (F14), **the code that mints the next claim (F15)**. F15 is the deepest instance —
  everything upstream of it is a document, and a document does not mint new receipts. That is where
  a withdrawal has to land to be durable.
- **The comparator findings are a different genus** and worth naming separately: a normalizer that
  erases more than it was built to erase, a filter and a check that disagree about what a path is,
  and a gate that reports its strongest verdict on an empty set. All three are **fail-open in the
  reassuring direction**, and all three are latent rather than live in this run.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
