# ddm_tmx1 — the tail coder's 40 counted mixer weights: what they are, what they were fit to, and why the refit is not yet priced

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
axis `[macOS-CPU advisory; exact bytes, scorer-free]` ·
`# FORMALIZATION_PENDING: this unit registers no new law. The law it would have tested — hpr1's "a counted section fit to a superseded state of the object is owed a refit", still n=1 — is CITED, never re-registered, because this arm produced no exact byte price to price it with.`

**STATE — INCOMPLETE, and the headline is the gap.** The rail and the fit are built, committed
(`71e3d8597`), lint-clean and reviewed twice; the falsifiers were pre-registered before any number
existed (`d5e58db44`); deliverable 1 is fully MEASURED below. **The live-loop control never ran to
completion, so no refit price exists and none is claimed.** The blocker is the storage tier, not the
mixer: Vertigo stands at **35.32 GiB free against the 40 GiB fail-closed reserve**, and at free below
the reserve every write refuses regardless of its size. I did not lower the reserve.

Frontier line: `composition S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600] (move 47)` —
UNMOVED by this arm.

---

## 1. What the 40 weights ARE — MEASURED, read off move 47's shipped archive

Producer: the rider parsed out of `d1fab05d69f31c90…` with the shipping
`residual_archive.read_residual_archive`. Rider sha `76f10171e42d27e6a1966a5a4a0a530c18541415b2a60a5fe9f8c2bb8af3d91a`,
exactly the value hpr1's staleness audit names.

The 60 B counted rider is `variant(1) + 40 int8 weights + 19 B geometry` — `FORMAT` in
`runtime/rlc1_geometry.py` is `<BHH8B6B`, 19 B, and `LaneMixer.__init__` refuses anything but
`41 + FORMAT.size`. So **the 60 B state's SHAPE is fixed by the shipped schema**: a refit may change
the 40 VALUES and nothing else. It cannot grow.

Weights (int8, receiver scale 1/32), tc1's 35 as `K=5` winning classes × `F=7` features, in the order
`SharedMixer.features` walks `LEVELS`:

| winner class | spatial2 | spatial3 | previous | run | rowband | temperature | hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 Road | −2 | 26 | −20 | 15 | 17 | −1 | 3 |
| 1 Lane | −7 | 25 | −1 | 11 | 17 | −1 | 2 |
| 2 Undrivable | 5 | 19 | −18 | 7 | 14 | −2 | 5 |
| 3 Movable | 7 | 20 | −12 | 2 | 10 | −2 | 5 |
| 4 MyCar | 11 | 13 | −14 | 9 | 10 | −2 | 8 |

tc3's 5 lane coefficients, one per winner class of **tc1's OUTPUT**: `[29, 23, 6, 10, 5]` — the exact
vector tc3's memo records as "these round to int8 [29, 23, 6, 10, 5]".

Geometry (unchanged by any refit): `FORMAT.unpack` = `(1, 128, 320, 16, 4, 4, 36, 2, 2, 3, 24, 0, 1,
2, 4, 8, 16)`.

### How they reach the coded row — the cascade, read from the receiver

`runtime/residual_archive.py` builds `LaneMixer(parts.tc1_weights)` when the rider is 60 B. Per
coding group:

1. `SharedMixer.features` takes the HPAC-prior-plus-corrector float32 row, builds RC64's integer
   `freq` (sum 2³¹, winner balanced), takes `arg = freq.argmax`, and fills a `(n, K, 7)` feature
   block: five Q10 fixed-point `log2` lookups of frame-frozen KT count/expected ratios indexed by
   `arg * levels + context`, one `log2(freq/2³¹)` temperature channel, and a `Q = 1024` indicator at
   the winner. `LEVELS` = spatial2 36, spatial3 216, previous 6, run 64, rowband 12.
2. `mix_probabilities` forms `exponent_k = Σ_j phi[k,j] · w[arg, j]`, then
   `p ∝ freq/2³¹ · 2^(exponent / (Q·SCALE))`, `Q·SCALE = 32768`. **So the 7 weights of the winner's
   bank, and only those, tilt that row.**
3. `LaneMixer` repeats the same arithmetic once more on tc1's OUTPUT with a single extra feature —
   the 9-bin lane-geometry table — and one weight per class; rows whose lane bin is the null bin 8,
   or whose class weight is zero, fall back to tc1's row unchanged.
4. RC64 codes the final row.

**The consequence that shapes any refit:** the online KT tables tc1 reads are accumulated from
`self.base` (the freq of the row that ENTERS the mixer) and the decoded truth, so **tc1's features
are weight-independent**. The lane table is accumulated from the freq of tc1's OUTPUT, so it is
weight-dependent. That asymmetry is why the refit tool treats the lane table as a first-order carry
and says so, and why the real encode — which recomputes everything — is the authority.

## 2. What they were FIT to — with receipts

| object | fitted on | field then | citation |
|---|---|---|---|
| tc1's 35 | the move-33/34 body, composed into the archive at move 36 | `a73289e0a30dd765…` — tc1's own memo names it as "the decoded label field", 117,964,800 symbols | `ddm_tc1_token_tail_bound_and_shared_mixer_pricing_20260909.md`; weights sha `35d56667911d1b59…` |
| cmp1's composition | **did not refit them** | `361cc6c9749fdec1…` (move 37's admitted field) | `ddm_cmp1_compose_rc3_tc1_20260909.md`, verbatim: *"The unchanged 35 counted int8 weights have SHA-256 `35d56667911d1b59…`"* |
| tc3's 5 | move 40, with the 35 **frozen** | move-40 field | `ddm_tc3_lane_predictor_receiver_seal_20260910.md`, verbatim: *"It freezes the original 35 mixer weights and fits five new shared coefficients on move40; these round to int8 [29, 23, 6, 10, 5]."* |
| rlc5's 19 B geometry | move-40 geometry, re-encoded on move 43's stream | — | `99625f32f` |

The field last moved at **move 43** (sj1 pass 6, `48109233e`) and is `a92e7d902a449896…` today.
**tc1's 35 have coded four later field changes without a refit; tc3's 5 have coded three.**

## 3. The prior negatives, accounted before the fit — and one that sharpens the expectation

The charter names mxo3, ls1/ls2, cl2/cl3 and the container lottery. The search of this arm's own
target family turned up a closer one the charter did not name, and it is the most informative:

**tc3's variant B jointly refit all 35 + 5 on the move-40 field and returned 37 actual archive
bytes**, against variant A's 79 B with the 35 FROZEN. tc3 records the confound in its own words —
*"B changes both the map and refitting, so this A/B cannot isolate their contributions"* — so it does
not close a pure refit. But it is the closest prior measurement on this exact object and it points at
**tens of bytes, not hundreds**.

The magnitude expectation was therefore pre-registered (`ddm_tmx1_falsifiers_preregistered_20260911.md`
§F7, committed before the fit tool ran): the mixer's entire measured value when FRESH was **549 B**
(tc1) + **79 B** (tc3) = **628 B**, so a staleness refit can recover only the DRIFT — a fraction of
628 B, never a multiple — and a bigger result would itself be suspect.

## 4. A measured correction to the fire bar's noise floor

The standing law prices a candidate against a **container-break lottery of sd 34.8 B**, which sits
ABOVE the charter's own fire bar (ΔS < −2e-5 ⇔ **ΔB < −30.04 B** at 6.658589531221714e-7 S/B). On
this axis that comparison does not apply, and the reason is MEASURED:

> move 47's `archive.zip` holds **one** member, `p`, `compress_type 0` (STORE), 179,259 B, with
> exactly **100 B** of ZIP overhead. The tail is the RAW RC64 stream, and the 60 B rider's length is
> fixed. A change in the token stream's length therefore reaches archive bytes **one-for-one**, with
> no re-segmentation and no recompression.

So a tail-length delta on this object is deterministic, not a one-sample draw, and the twin encodes
prove it reproducible. **A −30 B win here would be a real −30 B.** (The lottery still governs edits
that move a brotli'd member — the `hpac` section, for instance — which is precisely what hpr1's
refit did and this one does not.)

## 5. What was BUILT, and what it is one command from doing

- `experiments/ddm_tmx1_mixer_price.py` — the exact-byte rail. A fork of hpr1's move-47
  `ddm_hpr1_shape_price.py` with **one** structural change: the treatment replaces
  `parts.tc1_weights`, so the `hpac` member is carried through untouched and no RC3 re-encode runs.
  hpr1's geometry build is IMPORTED, not re-derived. Treatments `control47 / refit47 / control48 /
  refit48`; the move-48 base is hpr1's in-flight composition archive `d830edd37164…` @ 179,111 B,
  copied read-only, so MAIN can rebase without a re-run. The control's falsifier is that its archive
  must equal the base sha; the output-lossless proof is the in-loop frame-by-frame assertion.
- `experiments/ddm_tmx1_refit.py` — the offline refit. It replays the **shipped** cascade, imported
  from the priced runtime copy, over a retained 1-in-32 systematic sample (hashed on
  `(frame, position)` so the sample does not follow the coding GROUP lattice), scores it with the
  coder's own ideal length, and fits the 40 int8 values by deterministic coordinate descent on the
  grid. Held-out is by **frame parity, in both directions** — the same fold definition hpr1 used
  with mxo3's `cross_bits`; `cross_bits` itself does not transfer, because it scores a per-cell
  calibration table and the object here is a 40-parameter global vector with no cells, so applying it
  would be a category error rather than a shared instrument.
- **The fit's own control:** the stride-scaled surrogate must reproduce the loop's EXACT in-loop
  ideal bits (accumulated from the same hook the arithmetic encoder is fed from) to within 1 %, or
  the tool refuses and nothing downstream is admissible.

Resume is one command:
`.venv/bin/python experiments/ddm_tmx1_mixer_price.py --treatment control47 --collect`, then
`experiments/ddm_tmx1_refit.py`, then `--treatment refit47 --weights <fit>/weights_i8.bin`.

## 6. Why it did not run — the two failures, exactly, and a boundary correction

**Failure 1 (Vertigo, 452 s, frame ~175):**
`PriceError: STORAGE_BLOCK at the durable prior checkpoint` — the 40 GiB reserve fired inside the
encoder-state save. Not lowered. Run set aside at
`/Volumes/VertigoDataTier/pact/ddm_tmx1/aborted/control47_run1_storage_block` (moved, never deleted).

**Failure 2 (5 s):** `Jg2Error: immutable ddm_tmx1 retained payload changed on retry: INPUTS.json`.
**My bug, and worth keeping.** I had bound two volatile observations — free bytes at bind — INSIDE
`INPUTS.json`. Free space changes between runs, so the binding was not reproducible: the immutability
guard refused the second bind, and a resume would have refused too. Cured by moving the observation
into its own uniquely named `STORAGE_ROUTING_<utc>.json` receipt, outside the binding. This is hpr1's
own rule at a new surface — *volatile observation times are preserved as non-authoritative context
outside the hash* — and it is a general trap: **anything a binding hash covers must be a property of
the inputs, never of the moment.**

**Boundary correction (MAIN, 2026-09-11).** Between those failures I routed this arm's payloads to a
local-disk store to keep the Vertigo reserve intact. MAIN ruled that local disk is not a tier and the
charter names Vertigo. All 58 files written there (1.1 MB — the aborted prepare only; no fit trace,
no encode ever reached the local store) were copied back to
`/Volumes/VertigoDataTier/pact/ddm_tmx1/boundary_correction_20260911/`, **hash-verified file by file**
with a `MOVED_BACK.sha256` manifest of all 58, and the local copy was then removed. The second store
root is deleted from the producer; there is one root and one reserve.

**The footprint was cut as MAIN directed**, and the cut is right regardless of the tier: the decoded
field is no longer re-retained (112.5 MB per run of a raster that must EQUAL the bound input field by
the in-loop assertion — the sha is the receipt), and the fit trace samples 1 in 32 instead of 1 in 8.
Worst case for all four treatments is now about **200 MB**.

## 7. The blocker is the TIER, not the footprint — P0 for MAIN

MAIN's instruction assumed the refusal was a footprint problem and that Vertigo held 43 GiB. It held
43 GiB at 21:39 UTC. **MEASURED at 22:05 UTC, two independent ways, no local snapshots:**

| instrument | reading |
|---|---|
| `df -k /Volumes/VertigoDataTier` | **35.32 GiB free** |
| `diskutil apfs list disk5` → Capacity Not Allocated | 37,921,574,912 B = **35.32 GiB** |
| `tmutil listlocalsnapshots` | none — nothing is hidden behind a snapshot |

The guard's test is `free < RESERVE + len(payload)`. At 35.32 GiB against a 40 GiB reserve, **a 3 KB
source-file copy refuses** — I verified exactly that. Shrinking a footprint cannot clear a refusal
whose cause is the free-space term. This is a tier-level condition, and it binds **every** producer
carrying the 40 GiB reserve (`ddm_hpr1_shape_price`, `ddm_hpr1_shape_inputs`, `ddm_hpr1_public`,
this rail), not only this arm.

**What consumed it, MEASURED:** `/Volumes/VertigoDataTier/pact/ddm_hpr1/diagnostics/` holds
**7.1 GiB**, written 22:00–22:01 UTC by a sister arm's two live full-video `inflate.sh` pose
diagnostics — `pair_base/output/0.raw` and `pair_candidate/output/0.raw`, **3,662,409,600 B each**,
plus two 112.5 MB token checkpoints. The tier fell 43 → 35.3 GiB inside that window.

**A reclaim candidate, certified as far as this arm may go.** `pair_base` decodes move 47's promoted
tree, which is the live pointer, so its `0.raw` should be byte-identical to the pointer's retained raw
`2b762eba4a20a315…` that hpr1's own `RESULT.json` and the committed move-47 seal already hold.
Verified by streaming it: sha `2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc` — **identical**. **That is 3.41 GiB of duplicate bytes** whose claim is
preserved by three surviving receipts. It is hpr1's directory and I did not touch it —
**MAIN owns that decision.** The candidate raw is NOT a reclaim candidate: hpr1 states in its own memo
that the composition row is unsealed and its bytes are live evidence.

**And the arithmetic says a reclaim alone is not enough:** 35.32 + 3.41 = 38.73 GiB, still under 40.
So MAIN has two levers and this arm can pull neither: authorise a larger cross-arm reclaim, or rule
on the reserve itself.

**On the reserve, a derivation offered rather than an action taken.** The 40 GiB is not a derived
number anywhere in this repo — sister producers carry 1, 4, 8, 16, 20, 32 and 40 GiB, and hpr1's own
comment says only that it is "matched to the sister HPAC producers". This arm's worst-case footprint
is ~200 MB. A reserve 200× the job it guards is a cargo-culted constant, and the modal sister value
(8 GiB, in `db1`, `dc1s`, `df1`, `bl1`, `bhw2`, `ld1`) would admit this arm today with 35 GiB of
headroom left. **I did not change it.** Changing a reserve in the minute after it refuses is the
forbidden shape whatever the arithmetic says; the derivation is handed to MAIN, not applied.

## 8. What is NOT claimed

No refit price. No held-out number. No candidate. No pointer move. No Modal dispatch, no scorer run,
no seal, $0 spent. The sister receipt that the harness CAN reproduce move 47 — hpr1's own
`price/control47/PRICE.json`, 118,511 B stream, archive `d1fab05d69f31c90…` @ 179,359 B, Δ 0, twins
agreeing, output-lossless — is cited as evidence about the harness, **not** as this arm's control.
This arm's control is owed.

## 9. Boundaries honoured

`upstream/`, the PR tree, the sealed move-47 promoted tree and every sister arm's directory
(`ddm_hpr1`, `ddm_pr19`, `ddm_cons2`) were read and copied, never written. No receiver file was
patched — the rail asserts its runtime copy equals the base tree byte for byte, every file, which is
what makes a weights-only row a normal-seal shape. The 40 GiB reserve was never lowered; the local
store was removed and its bytes returned hash-verified; APDataStore was read-only for the field and
never written.
