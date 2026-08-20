# FREEZE CHECKLIST — packet generation 5 (jg5, the first sub-0.15 row)

**Author:** ddm_pq3 · **Date:** 2026-08-20 · **State:** PREPARED, FROZEN-READY, NOT SUBMITTED

This file lists **exactly what remains** between the staged packet and publication.
Nothing on this list has been done. No push, hosting, or PR action of any kind has
been taken.

**Staged packet:**
`/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen5_jg5_waterfill/`
(38 files) · **Receipts:** `…/generations/gen5_receipts/` (10 files)

| Property | Value |
|---|---|
| Archive SHA-256 | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` |
| Archive bytes | 180,625 |
| Runtime tree SHA-256 | `2103073d739fc3f27d329ea0785ea3010307360c2380af0476e16d0f5b57cb9b` |
| Score `[contest-CUDA T4, n600]` | **0.14839100138338618** ± 3.633e-06 |
| Compliance | 83 GREEN / 4 RED of 87, strict `--contest-final` |
| Census | `CENSUS_CLEAN` / `PREP_CLEAN` / `RECEIPTS_CLEAN`, rc=0 |
| Review counter | **0 of 5** — round 13 is the first review of these bytes |

---

## (a) Operator one-line confirm — REQUIRED, BLOCKING

Per `SWAP_PROCEDURE.md`, an actual push, hosting action, or PR opening without
explicit operator authorization is a **refusal condition**. Two distinct
authorizations are gated behind the confirm:

1. **Hosting authorization** — compliance red #4
   (`hosted_archive_manifest_supplied`) cannot go green until the archive is hosted
   and a manifest supplied. This is the operator's, not an arm's.
2. **Opening the pull request** — reserved to the operator in the same procedure.

## (b) GPU-routing variant decision — REQUIRED, BLOCKING

See `GPU_ROUTING_VARIANTS.md`. **The two variants are not symmetric and the choice
is not free.**

- **Variant (b) CURRENT / AUTO** — what is staged. `inflate.sh` unmodified, tree
  `2103073d…`, **authority row valid**, cost **zero**. Routing depends on the
  maintainer selecting the `linux-nvidia-t4` runner; the PR body states the
  requirement explicitly.
- **Variant (a) GPU-REQUIRED-EXPLICIT** — prepared, not applied. MEASURED: the
  one-line CUDA fail-fast moves the tree hash to `75a1aeef…`, so **the
  0.14839100138338618 row does not apply to it**. Cost: **one new T4 exact-eval
  row plus a full re-stage and re-review.**

Both remain open. The decision is the operator's.

## (c) Optional — fold the wc2 corrector port, then RE-PIN and RE-PROVE identity

The `ddm_wc2` arm is porting the rr2 corrector to close the wall-clock gap. Its
latest word is a **self-correction that softened its own verdict**: the jg5 body is
**WARN, not REFUSE**, fitting the optimistic end of the CUDA residual window by
about 10.7 s, with the port still on the critical path because 10.7 s of
warm-cache-only margin is noise.

If the port lands and is folded in, it touches shipped runtime files, so:
the tree hash moves, the seal must be re-pinned, and the packet needs a **new T4
row** before it can carry any score claim. Same cost structure as variant (a).

**Coordination note:** this arm did not touch wc2's files. Read its state from
`.omx/research/ddm_wc2_wall_clock_pass_20260820.md` and commits `cba7268e2a`,
`d2e1d067ef`, `0c8aab6f5e` before folding anything.

## (c2) Optional — the rr5 lossless rider — EVALUATED: DO NOT FOLD BY DEFAULT

The charter asked whether the rr5 rider composes onto jg5. **Assessment: it does
not compose without re-derivation, and folding it would cost the measured row.**

1. **Its measured value is on a DIFFERENT body.** `ddm_rr5`'s receipt measures
   **183 B / ΔS −1.2185e-4 on the then-pointer body**, not on jg5's. The pointer has
   since moved to jg5. The number does not transfer; it must be re-derived.
2. **The −1.85e-4 the charter cites is not a measured figure.** rr5's own memo
   corrects it: ra2 measured `+263 B raw / ~230 B realised` and labelled the
   realised half **PROVISIONAL**; a later memo added an unreceipted 48 B leg to
   reach ~278 B ⇒ −1.85e-4, and downstream memos inherited that sum as "MEASURED".
   The **lossless claim is sound** (round-trip exact, 27,648/27,648 symbols); only
   the size was overstated. This is the cross-regime constant-transfer genus.
3. **Decisive: lossless does not mean free.** The rider changes archive BYTES.
   Even with decode-identity proven — so `d_seg` and `d_pose` are unchanged by
   construction and the rate leg is exactly computable — the resulting score would
   be a **DERIVED** number, not a measured row. Shipping a derived score as the
   claim is the NO-FAKE #8 surrogate trap.

**Therefore the default stands: ship the MEASURED `f3bce5d2` bytes.** If the
operator wants the rider, it is staged as variant (c2) and needs its own T4 row.

## (d) Hosted archive URL — REQUIRED for the PR body, BLOCKING

`PR_BODY_DRAFT.md` §"upload zipped archive.zip" currently reads *"pending
operator-authorized hosting. No URL is claimed here, because no URL exists yet."*
That sentence is deliberate and must be replaced with a real URL — never a
placeholder, which the checker refuses
(`hosted_archive_public_text_has_no_placeholder`).

## (e) Review counter — 0 of 5, NOT a blocker the arm can clear

`SWAP_PROCEDURE.md` step 7 requires **five consecutive clean passes**. The counter
is `0/5`; rounds 1–12 reviewed superseded bytes and do not carry. Round 13 is the
first review of these bytes, and its examination list is written in
`ADVERSARIAL_REVIEW_SCAFFOLD.md`. **This arm cannot run it** — it staged the packet.

## (f) Owed items, named rather than closed

1. **`tools/stage_contest_submission_packet.py` has no tests.** It is new, it is
   load-bearing for the identity proof, and only its author has read it.
2. **Reproduction NOT re-verified for these bytes.** `ddm_pq2_compress_e2e.py` has
   not been re-run for `f3bce5d2…`.
3. **No contest-CPU row on these bytes**, and the axis is expected to be infeasible.
4. **The wall-clock WARN is DERIVED, not measured in CI.** 10.7 s of margin against
   a derived residual window is the largest open risk on this submission.
5. **The terminal dispatch claim row was appended by this arm**, turning three
   compliance reds green. Flagged for independent verification in round 13.

---

## What is DONE and needs no further action

- Packet staged and **proved** byte-identical to the evaluated tree: 33/33 manifest
  rows re-hashed after copy, and `runtime_tree_sha256` **re-derived from the staged
  rows** equals the receipt's `2103073d…`.
- Archive verified `f3bce5d2…` at 180,625 B; single stored member `p`.
- Score recomputed from components; the evaluator's rounded `0.15` display is
  disclosed everywhere the claim appears.
- Public docs written: `report.txt`, `README.md`, `archive_manifest.json`,
  `BORROWED_SUBSTRATE_ACCOUNTING.md` (§9 appended, §§1–8 preserved verbatim).
- PR body rewritten for this candidate, with `report.txt` embedded byte-verbatim.
- Compliance 83/87 with the same red SET as generation 4; four r1 reds cured at
  source rather than adjudicated away.
- Census clean on all three trees; 51 AppleDouble sidecars caught and purged.
- Harvested receipts decoded from Python bytes-reprs with round-trip proof.
