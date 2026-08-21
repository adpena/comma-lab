# ddm_rv17 — wave-3 seal sweep: NOT CLEAN. One finding survives. Counter 2/3 → 0/3.

**The wave does not seal.** The full-surface sweep found one real, measured, unrouted defect —
**and it is in a retained receipt, surfaced by my own adoption error.** I am recording it rather
than sealing, which is the system working.

---

## What the sweep verified CLEAN (with instruments different from prior rounds)

**Lens (a) — cross-surface number join** (a join, not a re-grep; four surfaces at once). The mirror
price tells one story: `5.3280` appears memo=9 / registry=3 / DAG=5 with no conflicting value; the
falsifier `5.2282` sits in memo and DAG and correctly **not** in the registry (the registry holds the
law, not the threshold); `5.9467` sits in memo and registry. The task ledger carries no prices at all,
which is correct — a task ledger holds states, not measurements.
*Rounding note, examined and dismissed:* both `5.3279` and `5.3280` appear. `664·8/997 =
5.327983951…`, so these are a truncate and a round of one value, not two numbers. Not a defect; I
record the check so a later reader does not re-open it.

**Lens (c) — residue routing.** `#1179` is present in the canonical task ledger. The three
`mtime > birthtime` preregs and the six queued labels are routed, not lost.

**The score, by a different denominator than I have used before.** The live rc2 frontier reads
`180,456 B / sha df7fd266` straight from the pointer file. Unchanged, and untouched by the entire
fs3 arc — as it must be, since the row was refused.

---

## W3-F19 (MED) — a retained baseline receipt names an object it does not measure

| field | value |
|---|---|
| **file** | `/Volumes/APDataStore/pact/ddm_fs3/FS3_RATE_BASELINE_RECEIPT.json` |
| **what it says** | *"SHIPPED 455-edit token stream of **the live rc2 body** — the object the drop removes tokens FROM"* |
| **what it measures** (my instruments, read off disk, independent of any receipt field) | jg5's archive: **180,625 B / sha `f3bce5d2`** |
| **what "the live rc2 body" is** (from `canonical_frontier_pointer.json`) | **180,456 B / sha `df7fd266`** |
| **delta** | **169 bytes and an entirely different sha.** These are two different objects. |
| **routing** | `grep` over the canonical task ledger for the phrase or either byte count returns **0**. It lives in three of my own review memos and nowhere else. **A review memo is not a work queue.** |
| **severity / bearing** | MED. Not decision-bearing today — the row was REFUSED on the measured pose leg, and S is untouched. It becomes live **the moment pose is fixed**, because the whole fs3 rate arc is measured against this baseline and a reader will take the receipt at its word. |
| **genus** | `measured object vs named object` — already a named genus in our memory, recurring in a **retained receipt**, which is the surface with the longest half-life. |
| **verdict_scope** | INSTANCE |
| **cure** | One clause: name the object as jg5's `180,625 B / f3bce5d2…` shipped stream, and state that it is **not** the live rc2 body. Then route it. |

### Why this survived four rounds: my declination named the wrong rows

The arm reported this in round 4. I adopted five of its rows (F14–F18) and wrote that I was
declining two others *"because they rest on a single instrument the arm itself flagged."* But the
rows I named there were the **census and retention-manifest edits**, from a different table. This
row and the `clears_sub_015` band received **neither adoption nor a stated reason**. My declination
paragraph made the disposition *look* complete while leaving two rows in silence.

That is my own wave-2 law recurring against me: **a ledger closes only on the rows it contains.** I
had applied it to other people's ledgers and then wrote a declination that closed on rows I had not
actually enumerated. The new form worth naming: **a declination that names the wrong rows reads as
complete disposition** — it is more dangerous than an empty one, because it signals that the sweep
happened.

---

## Honest state

- **Counter 2/3 → 0/3.** Eight findings this wave; seven cured and verified, one new (W3-F19).
- **The substance has still never moved** in 38 rounds. No wrong score, pin, digest, mis-scoped
  receipt, or unverifiable archive claim. The terminal verdict — task #1176 REFUSED on the measured
  pose leg — is untouched by every finding, every cure, and this one.
- **I did not manufacture this.** I looked for it because the seal instruction asked what had never
  been examined from the right angle, and the honest answer was: the rows I had passed over without
  saying why. It measured out real on two independent reads.
- **The wave's genus held to the end.** Every wave-3 finding has been one shape — a correct number
  whose obligation stopped one surface short. It travelled from a headline, to a registry, to a
  successor receipt, to the code that mints claims, and finally to a **retained receipt** and to my
  own declination paragraph. The last two are the surfaces that outlive the work, which is why the
  sweep was worth running instead of sealing.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4 n600]**
