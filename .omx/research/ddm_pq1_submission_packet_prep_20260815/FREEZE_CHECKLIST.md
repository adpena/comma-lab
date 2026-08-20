# FREEZE CHECKLIST — packet generation 6 (the composed rider × native port)

**Author:** ddm_pq11 · **Date:** 2026-08-20 · **State:** PREPARED, FROZEN, NOT SUBMITTED

This file lists **exactly what remains** between the staged packet and publication.
Nothing on this list has been done. No push, hosting, or PR action of any kind has
been taken.

**Staged packet:**
`/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed/`
· **Receipts:** `…/generations/gen6_receipts/`

| Property | Value |
|---|---|
| Archive SHA-256 | `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` |
| Archive bytes | 180,456 |
| Runtime tree SHA-256 | `fdd5774921319a317a385a9594489aa97e45cebc0f6f20cdc50fe8aaeb08a7f2` (36 rows) |
| Score `[contest-CUDA T4, n600]` | **0.14827847122030852** ± 3.633e-06 |
| Decode budget `[contest-CUDA T4]` | **PASS** — 498.476 s charged vs an 822 s cold-cache ceiling |
| CPU axis | **MEASURED WALL-INFEASIBLE** — no score exists, none claimed |
| Compliance | **OWED** — not re-bought for these bytes |
| Census | **OWED** — not re-run for these trees |
| Review counter | **0 of 5** — round 14 is the first review of these bytes |

---

## (a) Operator one-line confirm — REQUIRED, BLOCKING

Per `SWAP_PROCEDURE.md`, an actual push, hosting action, or PR opening without
explicit operator authorization is a **refusal condition**. Two distinct
authorizations are gated behind the confirm:

1. **Hosting authorization** — the archive has never been published. Until it is,
   the PR body's download field is blank by construction and the compliance red
   `hosted_archive_manifest_supplied` cannot go green. This is the operator's, not
   an arm's.
2. **Opening the pull request** — reserved to the operator in the same procedure.

## (b) GPU-routing variant decision — REQUIRED, BLOCKING

See `GPU_ROUTING_VARIANTS.md`. **The two variants are not symmetric and the choice
is not free.**

- **Variant (b) CURRENT / AUTO** — what is staged. `inflate.sh` unmodified, tree
  `fdd57749…`, **authority row valid**, cost **zero**. Routing depends on the
  maintainer selecting the `linux-nvidia-t4` runner; the PR body states the
  requirement explicitly.
- **Variant (a) GPU-REQUIRED-EXPLICIT** — prepared, not applied. The one-line CUDA
  fail-fast edits a shipped runtime file, so it moves the tree hash and **the
  0.14827847122030852 row does not apply to it**. Cost: **one new T4 exact-eval row
  plus a full re-stage and re-review.**
  *Do not carry the old variant-(a) tree hash forward.* `75a1aeef…` was measured on
  generation 5's 33-row tree. This is a different 36-row tree; the flipped hash for
  it has not been measured, and quoting the old one would be exactly the
  cross-regime constant transfer this packet has already paid for twice.

Both remain open. The decision is the operator's.

## (c) The wc2 corrector port — FOLDED. It ships.

Formerly listed here as optional. It landed, and it is the reason the decode budget
went from over-ceiling to PASS.

`runtime/f26_corrector_native.c` and `runtime/native_free_corrector.py` are rows 20
and 31 of the shipped manifest, and the authority receipt records
`free_corrector: NativeFreeCorrector` on **both** axes — the Python fallback did not
engage. Measured effect on the shipping axis: inflation fell from
**1419.904212624 s** (generation 5) to **458.752594349 s**, a **3.10×** drop, at
**zero** change to any decoded value.

The cost structure named in the old entry was paid honestly rather than avoided:
folding it moved the tree hash, so the packet bought a **new T4 row** rather than
carrying generation 5's. That row is this generation's authority.

## (c2) The rr5 lossless rider — FOLDED. It ships.

Formerly assessed here as DO NOT FOLD BY DEFAULT, for three reasons. Two were about
sequencing and one was about arithmetic; all three were answered by doing the work in
the order the entry demanded, not by waiving it.

1. **"Its measured value is on a DIFFERENT body."** Correct, and the warning was
   quantitatively right. Re-measured on the final body the rider is worth **169 B**,
   not the 183 B measured on the older body.
2. **"The −1.85e-4 budget figure is not measured."** Also correct, and it stays
   withdrawn. The realized number is now the measured **−1.1253016e-04**, taken from
   two exact rows rather than from any budget.
3. **"Lossless does not mean free — the resulting score would be DERIVED."** This
   was the decisive objection and it was answered the only way it can be: the
   composed archive was **evaluated in its own right** on contest-CUDA T4. The score
   is a measured row, not a derived one.

What ships: reserved header flag `0x08` engages `restore_carrier_body` on the
receiver, restoring a 22,316 B carrier blob. The decoded state does not move — the
n600 inflated output is byte-identical to generation 5 on the shipping axis, both
hashing to `6bf8acf8…` at 3,662,409,600 B.

## (d) Hosted archive URL — REQUIRED for the PR body, BLOCKING

`PR_BODY_DRAFT.md` §"upload zipped archive.zip" reads **"Download: PENDING
PUBLICATION — the URL is deliberately blank."** That sentence is deliberate and must
be replaced with a real URL pinned to the commit that carries **these** bytes —
never a placeholder, which the checker refuses
(`hosted_archive_public_text_has_no_placeholder`).

Generation 5's URL is **not** reusable: it serves a different archive under a
different runtime tree. `SWAP_PROCEDURE.md` step 4A is the instrument — push the
exact bytes, derive a raw URL pinned to that 40-character commit, download it fresh,
and require HTTP 200 plus SHA-256 and byte-count equality before publication.

## (e) Review counter — 0 of 5, NOT a blocker the arm can clear

`SWAP_PROCEDURE.md` step 7 requires **five consecutive clean passes**. The counter is
`0/5`; rounds 1–13 reviewed superseded bytes and do not carry. Round 14 is the first
review of these bytes, and its examination list is written in
`ADVERSARIAL_REVIEW_SCAFFOLD.md`. **This arm cannot run it** — it staged the packet.

## (f) Compliance and census — OWED for these bytes, BLOCKING

Neither was re-bought at this swap, deliberately. Under the receipt-freshness law a
receipt is a joint measurement of BYTES × INSTRUMENT × WORLD, and at this swap **all
three moved**: the archive and runtime changed, every checker-scanned surface was
rewritten, and the frontier pointer advanced to this candidate. Generation 5's
`83 GREEN / 4 RED of 87` is therefore stale on every axis and is **not** carried
forward or cited as this generation's state.

Owed, in order: `tools/packet_census_guard.py` over the staged generation **and** the
prep tree (rc must be 0), then the strict chain with
`--expected-archive-sha256 df7fd266…` and `--expected-archive-size-bytes 180456`,
preserving every red with a typed disposition.

## (g) Owed items, named rather than closed

1. **`tools/stage_contest_submission_packet.py` still has no tests.** It is
   load-bearing for the identity proof. Its mechanism was independently re-derived
   by an outside reviewer, but tests remain owed.
2. **Reproduction NOT re-verified for these bytes.** `ddm_pq2_compress_e2e.py` has
   not been re-run for `df7fd266…`, and no prior VERIFIED label transfers.
3. **3 of 36 runtime rows are not yet in version control** — `inflate.py`,
   `inflate.sh` and `runtime/residual_archive.py`, the two entry points and one
   receiver module the composed object rewired. The other 33 are already present
   byte-identically. This closes with the same operator push that pins the URL.
4. **The CI residual window is a PROJECTION.** Our 498.476 s is measured; the
   822–1302 s it has to fit inside is not, and no per-step CI second has ever been
   timed by us on a real runner. This is now the largest open risk on the runtime
   axis — smaller than generation 5's, which exceeded both ends of that window, but
   still the one thing about the budget we cannot show you.

---

## What is DONE and needs no further action

- Packet staged and **proved** byte-identical to the evaluated tree: 36/36 manifest
  rows re-hashed after copy, and `runtime_tree_sha256` **re-derived from the freshly
  measured staged bytes** equals the receipt's `fdd57749…`.
- Archive verified `df7fd266…` at 180,456 B; single stored member `p`, 180,356 B,
  name-safety checked.
- Both axes measured on these exact bytes: a contest-CUDA T4 n600 score row, and a
  contest-CPU run that establishes wall-infeasibility rather than leaving the axis
  open.
- Decode identity against generation 5 proved on the shipping axis by identical raw
  output, which is what makes the inter-generation score delta exact.
- Score recomputed from components; the evaluator's rounded `0.15` display is
  disclosed everywhere the claim appears.
- Public docs refreshed against this object: `report.txt`, `README.md`,
  `archive_manifest.json`, `MANIFEST.sha256` (36 rows), `PR_BODY_DRAFT.md`,
  `ARCHIVE_MANIFEST.json`, `PACKET_TARGET.json`, and the accounting.
- Every `GATED-ON-RC2` marker resolved from the measured receipts, except the CI
  setup wall, which the receipts genuinely do not measure and which stays labelled
  a projection.
