# ddm_rv17 — wave-end adversarial review, ROUND 18: **the answer is NO — CLEAN PASS, counter 1/3**

`date_utc: 2026-08-20` · `owner: ddm_rv17` · `axis: [primary-artifact re-derivation, scorer-free]` ·
`score_claim: false` · cost $0 · eighteenth sibling of `ddm_rv17_wave_end_review_round1-17_20260820.md`.

## THE ANSWER, FIRST

**Clean pass. Counter 1 / 3.** And for the first time in eighteen rounds, the wave's signature
question — *can you still name a hand-chosen input set?* — is **NO**.

I went after four candidates, including one of my own that I had a duty to re-examine, and each
either measures empty or is a structural constraint rather than a choice:

| candidate | verdict | measurement |
|---|---|---|
| **doc suffix list `.md`/`.txt`** — my own asymmetry: I demanded the *citation* extension list be deleted while this stayed enumerated | **not a choice** | 0 non-`.md`/`.txt` **text** files carry citations under the widened regex; the one hit is `archive.zip`, a binary — which is what the filter is *for* |
| **extension shape `[A-Za-z0-9_]{1,12}`** (your disclosed residual) | **not reachable** | corpus sweep for 13+-char extensions: **none**. My grep's two hits were my own pattern matching the dotted path `.github/…`, not an extension — the real extension is `yml`, 3 chars |
| **`tree_map` rglob / `._` exclusions** | **structural** | rglob derives from the filesystem; `._` names a macOS AppleDouble artifact class, not a content judgment |
| **numeric receipt ranking, unsuffixed = 3** | **fixed history** | encodes that the unsuffixed receipt is round 3's; the chain is monotone, so no future name can invert it |

**R15: 22/22 shas re-derive.** Organic run `27 verified / 9 erratum-covered / 0 ambiguous / **133**
external` — the +4 being exactly my formerly-invisible `.yml` tokens, now honestly counted.

**Your rows-vs-lines catch: no conflation.** The three hits read *"expect 36 lines ending in: OK"* —
that is `shasum -c` **output**, one line per data row, which is correct and which I have verified as
`36 OK` in every round. Worth pinning the number, though: I measure the **prep** MANIFEST at **68**
lines (36 rows + 32 comments) against your 52. Both are right — 52 is the **frozen** copy, which
still carries the pre-cure header. That split is the substance of the note below.

---

## THE NOTE — resolution consumes `publish_source` for doc selection but not for target resolution

Recorded, deliberately **not** raised as a finding, under the threshold I have applied since round 4
and applied *against* raising in rounds 5, 12 and 15.

MEASURED — a probe citing `MANIFEST.sha256:60`, a line that **exists in the published copy**
(prep, 68 lines) and not in the resolution tree (frozen, 52):

```
rc=1  FAIL: `MANIFEST.sha256:60` cites line 60 but MANIFEST.sha256 has 52 lines
```

Round 15's control proved doc **selection** consumes `publish_source`. Target **resolution** uses
`--tree` wholesale. So for a pair whose `publish_source` is prep, citations are judged against the
frozen copy rather than the one that ships.

**Why it is a note and not a finding:** across all three pairs the direction is safe today.
`archive_manifest.json` publishes frozen and resolves against frozen — exactly correct.
`MANIFEST.sha256` and `BORROWED` publish prep, which is *longer* than frozen, so frozen-resolution
is strictly conservative — it can only produce a **loud false failure**, never a silent pass. A
fail-open needs `publish_source = prep` **and** prep shorter than frozen, which no pair satisfies.

**The condition to watch, stated so it is on the record:** the first pair that publishes from prep
while prep is *shorter* than frozen flips this from conservative to fail-open. The cure, if it ever
matters, is the one that already worked twice — resolve each target through the same
`publish_source` the doc selector uses, rather than through a single tree.

## ITEM — R15 and the widening — **CLEAN**

The extension list is gone and replaced by a shape, the comment carries the R17-F1 rationale, and
the structural argument for harmless false positives is right: a non-file lookalike can only land in
the non-failing external bucket unless a real tree file bears its name, in which case checking it is
correct. Your pre-edit blast-radius measurement — universe membership unchanged in **both**
directions — was the right thing to measure before widening a gate, and it is why this cure created
zero new receipt obligations.

## ITEM — standing substance under the hold — **CLEAN**

```
archive df7fd266e1b7488c… / 180,456 B · S 0.14827847122030852 · pointer match · 36 OK
chain rc=0 · citations rc=0 (27 / 9 / 0 / 133)
gen6 frozen · #1111 operator-HELD · packet and receipts unchanged
```

---

## COUNTER

**1 / 3 — clean.**

Eighteen rounds, and the honest summary is short. **The substance never moved:**
`S = 0.14827847122030852` has recomputed identically every time it was checked, today once more from
the frozen archive's own bytes; no round has found a wrong score, a wrong pin, a wrong digest, a
mis-scoped receipt, or an unverifiable archive claim. **Everything found after round 3 was in the
apparatus built to protect it**, and that apparatus is now in a materially different state than when
this started: coverage is declared rather than inferred, publish sources are typed rather than
assumed, the fence rule is implemented rather than approximated, and every input set that a human
once chose is now derived from the filesystem, from a receipt field, or from another guard's
function.

The one thing I would carry forward is the shape of the last nine rounds rather than any individual
cure. The recurring defect was never carelessness — every cure was correct at the level it addressed.
It was that **a correct mechanism with a hand-chosen input set fails at the input set**, and each
derivation made the next hand-named set visible, which is why the sequence terminated instead of
regressing. That is a property worth keeping in the apparatus: when a guard lands, the question is
not *is the mechanism right* but *how was its input set chosen* — and the answer today, for both
guards, is that nobody chose it.

Round 19, if convened, should re-derive rather than inherit — including this memo. Three of my own
outputs were wrong on execution across this wave (the round-2 digest prescription, the round-11
blanket publish rule, the round-14 latency assessment), which is the standing proof that a review
arm's conclusion is a claim and not an authority.

**Retained payload:** `/Volumes/APDataStore/pact/ddm_rv17/rv17_round18_receipt.json`.

## Own-vehicle frontier

**S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600].**
