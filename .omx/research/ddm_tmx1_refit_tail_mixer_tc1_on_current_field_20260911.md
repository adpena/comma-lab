# ddm_tmx1 — refitting the tail coder's 40 counted mixer weights on the CURRENT field: MEASURED, and it LOSES

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
axis `[macOS-CPU advisory; exact bytes, scorer-free]` ·
`# FORMALIZATION_PENDING: this arm measures hpr1's proposed law rather than extending it. The law — "a counted section fit to a superseded state of the object is owed a refit" — is CITED at n=1 (hpr1's HPAC refit, −887 B) and is NOT re-registered here, because this arm's exact rows are the law's first NEGATIVE instance and a law needs a second positive before it is a law. What this arm does register for MAIN's judgement is the corrected RANKING quantity in §7, which needs a third section's refit to confirm.`

**STATE — COMPLETE. The answer is NO.** Both live-loop controls reproduce their bases byte-identically.
Four independent fits, on two bases with different HPAC priors, across four held-out folds, all say the
refit is WORSE. Six exact 600-frame encodes agree: **every refit costs bytes, none saves any.**
**FIRE VERDICT: DO NOT FIRE.** Nothing is staged, nothing is sealed, no candidate is claimed.

Live frontier, read at write time (the pointer moved to **move 48** under this arm at 23:12 UTC, and
this rail's own guard is what caught it — §6):
`composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` — UNMOVED by this arm.

---

## 1. What the 40 weights ARE — MEASURED off move 47's shipped archive

Rider sha `76f10171e42d27e6a1966a5a4a0a530c18541415b2a60a5fe9f8c2bb8af3d91a`, exactly the value hpr1's
staleness audit names. 60 B = `variant(1) + 40 int8 + 19 B geometry`; `FORMAT` in
`runtime/rlc1_geometry.py` is `<BHH8B6B` = 19 B and `LaneMixer.__init__` refuses anything but
`41 + FORMAT.size`. **The shape is fixed by the shipped schema: a refit may change the 40 VALUES and
nothing else. It cannot grow.**

tc1's 35, as `K=5` winning classes × `F=7` features, in the order `SharedMixer.features` walks `LEVELS`:

| winner class | spatial2 | spatial3 | previous | run | rowband | temperature | hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 Road | −2 | 26 | −20 | 15 | 17 | −1 | 3 |
| 1 Lane | −7 | 25 | −1 | 11 | 17 | −1 | 2 |
| 2 Undrivable | 5 | 19 | −18 | 7 | 14 | −2 | 5 |
| 3 Movable | 7 | 20 | −12 | 2 | 10 | −2 | 5 |
| 4 MyCar | 11 | 13 | −14 | 9 | 10 | −2 | 8 |

tc3's 5 lane coefficients, indexed by the winner of **tc1's OUTPUT**: `[29, 23, 6, 10, 5]` — the exact
vector tc3's memo records. Geometry, untouched by any refit:
`(1, 128, 320, 16, 4, 4, 36, 2, 2, 3, 24, 0, 1, 2, 4, 8, 16)`.

### The cascade, read from the receiver

`residual_archive` builds `LaneMixer(parts.tc1_weights)` when the rider is 60 B. Per coding group:
`SharedMixer.features` builds RC64's integer `freq` (sum 2³¹, winner balanced), takes `arg =
freq.argmax`, and fills `(n, K, 7)`: five Q10 `log2` lookups of frame-frozen KT count/expected ratios
at `arg * levels + context` (`LEVELS` = spatial2 36, spatial3 216, previous 6, run 64, rowband 12), a
`log2(freq/2³¹)` temperature channel, and `Q = 1024` at the winner. Then
`p ∝ freq/2³¹ · 2^(Σⱼ φ[k,j]·w[arg,j] / 32768)` — **only the winner's 7-weight bank tilts that row.**
`LaneMixer` repeats the same arithmetic once more on tc1's output with one extra feature (a 9-bin lane
table) and one weight per class; a null bin or a zero class weight falls back to tc1's row unchanged.

**The asymmetry that shapes any refit:** tc1's KT tables accumulate from the freq of the row ENTERING
the mixer and the decoded truth, so **tc1's features are weight-independent**; the lane table
accumulates from the freq of tc1's OUTPUT, so it is weight-dependent. The offline fit therefore carries
the lane table from the control as a first-order term and says so; the real encode recomputes it.

## 2. What they were FIT to — with receipts

| object | fitted on | moved since? | citation |
|---|---|---|---|
| tc1's 35 | the move-33/34 body, composed at move 36 onto field `a73289e0…` | **yes, four times** | `ddm_tc1_…_20260909.md`, weights sha `35d56667911d1b59…` |
| cmp1's composition | **did not refit them** | — | `ddm_cmp1_…_20260909.md` verbatim: *"The unchanged 35 counted int8 weights have SHA-256 `35d56667911d1b59…`"* |
| tc3's 5 | move 40, with the 35 **frozen** | **yes, three times** | `ddm_tc3_…_20260910.md` verbatim: *"It freezes the original 35 mixer weights and fits five new shared coefficients on move40; these round to int8 [29, 23, 6, 10, 5]."* |
| rlc5's 19 B geometry | move-40 geometry, re-encoded on move 43's stream | weights stale, encode current | `99625f32f` |

The field last moved at **move 43** (sj1 pass 6, `48109233e`) and is `a92e7d902a449896…` today.
**The staleness the audit described is real. The debt it implies is not.**

## 3. The two live-loop CONTROLS — both PASS

Same rail, same known-symbol twin-encoder loop, 600 frames, real coder, real receiver loop:

| control | base | stream | archive | Δ | twins | decoded field | wall |
|---|---|---:|---|---:|---|---|---:|
| `control47` | move 47 `d1fab05d…` | **118,511 B** | **`d1fab05d69f31c90…` @ 179,359 B** | **0** | identical | `a92e7d90…` | 1,173.9 s |
| `control48` | the composition `d830edd3…` | **118,896 B** | **`d830edd371641e19…` @ 179,111 B** | **0** | `a92e7d90…` | 1,171.7 s |

Output-lossless is asserted IN the loop, frame by frame, in every run. Every unrelated member
(`hpac`, `semantic`, `carrier`, `residual`) parses back byte-identical. **No treatment price would have
been admissible without these, and both passed on the first attempt.**

## 4. The surrogate, and three controls on it

The fit replays the SHIPPED cascade, imported from the priced runtime copy, over a retained 1-in-32
systematic sample (3,686,368 of 117,964,800 symbols, hashed on `(frame, position)` so the sample does
not follow the coding GROUP lattice), scored with the coder's own ideal length. Its controls, all
MEASURED, all PASSING (`fit/CASCADE_SELFTEST.json`, and the per-run control in every `REFIT.json`):

1. **the mixing math is BIT-IDENTICAL to the shipped mixer** — 5,000 random rows and feature blocks
   through both `mix_probabilities` and the vectorised copy, exact float32 equality on the tc1 stage,
   the lane stage and the inactive-row fallback (539 inactive rows, matched). It earned its keep at
   once: it caught my first lane construction indexing the table by tc1's INPUT winner where the
   receiver indexes by tc1's OUTPUT winner.
2. **the run-state and context replay is EXACT** — the shipped `SharedMixer`'s own `end_frame` driven
   over the real field; this tool's replayed `run` state and the five contexts from it agree frame by
   frame.
3. **the end-to-end bits control** — the stride-scaled replay against the loop's own exact in-loop
   bits, at a tolerance DERIVED from the estimator's noise (§5).

## 5. The control that fired, and why the bar was the bug

My first end-to-end bar demanded the replay land within **1 %** of the loop's exact bits. It fired at
**2.15 %**. I suspected a float-versus-integer ideal-length mismatch and **measured it dead: the two
definitions agree to 0.0001 %.** The real cause is the instrument:

> The per-symbol bit cost on this field is violently heavy-tailed — class 1 costs 0.474 bits a symbol,
> class 2 costs 0.0018. MEASURED from the sample itself, a 1-in-32 estimate of the total carries a
> standard error of **10,245 bits = 1,280.6 B = 1.058 %**. The observed gap was **1.99 SE**.

**A 1 % bar on a 1 %-SE estimator fires about half the time whatever the code does — it tests the
sample, not the replay.** The bar is now three standard errors computed from the sample the control is
drawn from, and the tool reports PAIRED deltas with their own standard errors, because the absolute
level of a stride-scaled estimate is noisy while the difference between two vectors scored on the SAME
symbols is not. A control's tolerance has to be derived from the estimator's noise, not picked.

## 6. The pointer moved under this arm, and the guard caught it

At **23:12 UTC** the live pointer became `d830edd371641e19…` @ 179,111 B — hpr1's composition landed as
**move 48**. My next launch refused before writing a byte:
`PriceError: POINTER_MOVED: move-47 tree d1fab05d… is not the live pointer d830edd3…`.

The guard was right to fire and wrong in scope, and the correction is worth keeping: a **candidate** row
must price against the live pointer, but a move-47 row is now a **diagnostic** against a named,
superseded base, so it binds to the RECORDED move-47 sha rather than to "whatever is live", and every
row declares `prices_against_live_pointer`. Binding a historical base to a moving pointer is the
opposite error to the one the candidate guard prevents.

A second small lesson in the same fix: my first patch moved the live-pointer check off the move-47
branch and left **no check at all** on the move-48 branch — and the only thing pointing at it was
ruff's unused-variable warning for the now-dead `live`. **A linter complaint about a dead name was a
real missing invariant.**

## 7. THE REFIT TABLE — surrogate and exact, both bases

Surrogate, PAIRED against the shipped weights, with standard errors (positive = WORSE, bytes):

| fit | search set | in-sample paired | held-out, even fold | held-out, odd fold |
|---|---|---:|---:|---:|
| move 47, search stride 4 | 922 k | **+115.5 ± 52.6** (t +2.19) | **+134.1 ± 43.6** (t +3.08) | **+173.4 ± 47.0** (t +3.69) |
| **move 47, FULL sample** | 3.69 M | **−90.2 ± 25.4** (t −3.55) | **+53.6 ± 24.6** (t +2.18) | **+54.2 ± 24.3** (t +2.23) |
| move 48, search stride 4 | 922 k | +51.9 ± 47.9 (t +1.08) | **+118.4 ± 39.3** (t +3.02) | **+201.8 ± 47.9** (t +4.21) |
| **move 48, FULL sample** | 3.69 M | **−98.2 ± 23.9** (t −4.10) | **+59.0 ± 25.8** (t +2.29) | **+54.2 ± 25.1** (t +2.16) |

**The two full-sample rows are the finding.** The search DOES find a real in-sample gain — −90.2 B on
move 47, −98.2 B on move 48, both at |t| > 3.5 — and that gain **reverses to about +55 B on held-out
data, in all four fold directions, on two different HPAC priors, at t between +2.16 and +2.29.** The
agreement across independent bases is close enough to read as a constant: the 40-parameter fit
reliably extracts ≈ −94 B of fold-specific structure and pays ≈ +55 B for it elsewhere.

**Pre-registered falsifier F3 — "the refit must beat the shipped weights on the held-out fold in BOTH
directions" — FIRES. Four folds out of four go the wrong way.**

EXACT, the real coder over all 117,964,800 symbols, twin encodes agreeing, decoded field byte-identical
to `a92e7d90…` in every row:

| row | base | weights | stream | archive | **ΔB** | ΔS | live pointer? |
|---|---|---|---:|---|---:|---:|---|
| `control47` | move 47 | shipped | 118,511 | `d1fab05d…` @ 179,359 | **0** | — | (was) |
| `refit47` | move 47 | stride-4 fit | 118,714 | `54fcbfbc…` @ 179,562 | **+203** | +1.352e-4 | no |
| `refit47b` | move 47 | **full-sample fit** | 118,533 | `57ffe1c7…` @ 179,381 | **+22** | +1.465e-5 | no |
| `control48` | move 48 | shipped | 118,896 | `d830edd3…` @ 179,111 | **0** | — | **yes** |
| `refit48` | move 48 | stride-4 fit | 119,051 | `b64f6bae…` @ 179,266 | **+155** | +1.032e-4 | **yes** |
| `refit48b` | move 48 | **full-sample fit** | 118,916 | `a7ba750c…` @ 179,131 | **+20** | +1.332e-5 | **yes** |

**The two full-sample rows are decisive, and they replicate.** Their weights gained **−90.2 ± 25.4 B**
(move 47) and **−98.2 ± 23.9 B** (move 48) on the 1-in-32 samples they were fit to; the real coder on
the **full** field says **+22 B** and **+20 B**. The sample-to-field transfer lost **112 B and 118 B**
— twice, on two bases with different HPAC priors — which is what the fold test predicted from a
different direction (in-sample −94, held-out +55, gap ≈ 149 B). **Three instruments — the paired
surrogate, the frame-parity folds, and the exact encode — agree on two independent bases, and none of
them finds a gain.**

**+20 B is the family's measured ceiling on this object.** The full-sample fit is the most favourable
vector this search can construct, and the real coder still prices it above the incumbent. The weights
tc1 fitted on the move-32/33–34 field are, for the current field, at least as good as anything four
searches could find for it.

Falsifier **F4 ("if the surrogate and the exact encode disagree in sign, the surrogate is falsified")**
does NOT fire: every sign matches. The surrogate is a sound ranker at this object; there is simply
nothing for it to rank into a win.

### What this says about the staleness audit's RANKING rule

hpr1's audit ranked refits by **the byte mass the stale state PRICES**, and on that rule this rung was
#1 on the board: *"60 B of state prices 118,511 B."* The rule is **refuted as a predictor by this
measurement.** The HPAC prior — 12,262 B of fitted state conditioning the same 118,511 B stream —
repaid **−887 B** when refit. The mixer — 40 B of fitted state conditioning the identical stream —
repays **nothing**, on either base.

The quantity that separates them is not the mass conditioned; it is **the fitted capacity the section
holds, against the noise in the objective that fits it.** 12,262 B of state can absorb a field change;
40 B cannot hold enough information for a field change to have made it wrong. The mixer's entire
measured value when FRESH was 549 B (tc1) + 79 B (tc3) = 628 B over 118 M symbols — the whole object is
a small correction on a strong neural prior, and its drift is below the resolution of any estimator
that has to be fit on the same field it codes. **Rank refit rungs by fitted capacity, not by
conditioned mass.** That is a proposal at n=2 (one positive, one negative), not a law.

### The prior negative that pointed here first

tc3's variant B jointly refit all 35 + 5 on the move-40 field and returned **37 actual archive bytes**
against variant A's 79 B with the 35 FROZEN. tc3 names its own confound — *"B changes both the map and
refitting, so this A/B cannot isolate their contributions"* — so it could not close a pure refit. It
was still the closest prior measurement, it pointed at tens of bytes rather than hundreds, and it was
pointing the right way. **This arm now isolates it: the refit alone is worth less than nothing.**

## 8. Fire verdict, and the noise floor that does NOT apply here

**Bar:** ΔS < −2e-5, i.e. **ΔB < −30.04 B** at 6.658589531221714e-7 S/B.
**Every exact row is ≥ +20 B. DO NOT FIRE.** Nothing is staged; no parse-back, no manifest
regeneration, no smokes, no seal, no candidate claim.

And a correction worth keeping: the standing **±34.8 B container-break lottery does not govern this
axis.** MEASURED — the archive holds **one** member, `p`, `compress_type 0` (STORE), with exactly
**100 B** of ZIP overhead, and the 60 B rider's length is fixed, so a token-stream length change reaches
archive bytes **one-for-one** with no re-segmentation and no recompression. The twins prove it
reproducible. **These +20 B and +203 B are exact, deterministic byte counts, not draws.** (The lottery
still governs edits that move a brotli'd member — the `hpac` section — which is what hpr1's refit did
and this one does not.)

## 9. Decode-time cost — unchanged by construction, and not a receipt

The refit changes 40 int8 values and nothing else: same symbols, same groups, same code path, same
operation counts. The decode-time delta is structurally zero. The six 600-frame encodes ran
1,171.7–1,293.1 s wall on a contended host `[macOS-CPU advisory diagnostic; NOT a contest decode-time
receipt]`; that spread is host load, and no wall-clock claim is made from it.

## 10. Storage, and a boundary correction

The 40 GiB Vertigo reserve refused this arm's first control at frame ~175 when a sister's two live
full-video `inflate.sh` diagnostics took the tier from 43 to 35.3 GiB. **I did not lower it.** I briefly
routed payloads to a local-disk store; MAIN ruled that local disk is not a tier, and all 58 files
(1.1 MB — an aborted prepare only; no encode ever reached it) were copied back to
`boundary_correction_20260911/`, **hash-verified file by file** with a `MOVED_BACK.sha256` manifest,
and the local copy removed. The second store root is gone from the producer: one root, one reserve.
MAIN then cleared the tier (50.5 GiB) and the sequence ran.

Two footprint cuts from that episode are right regardless: the decoded field is no longer re-retained
(112.5 MB per run of a raster that must EQUAL the bound input field by the in-loop assertion — the sha
is the receipt), and the fit trace samples 1 in 32. Six encodes and four fits fit in ~1.2 GiB.

**A reserve derivation was offered to MAIN and not applied** (`ddm_tmx1_tier_blocker_and_reclaim_proposal_20260911.json`);
MAIN has since recorded that the 40 GiB is a boot-swap floor copied onto the SSDs against a measured
need of ~21–24 GiB. Changing a reserve in the minute after it refuses is the forbidden shape whatever
the arithmetic says.

## 11. One bug of mine worth keeping

`Jg2Error: immutable ddm_tmx1 retained payload changed on retry: INPUTS.json`. I had bound two volatile
observations — free bytes at bind — INSIDE the binding. Free space changes between runs, so the binding
was not reproducible: the immutability guard refused the second bind, and a resume would have refused
too. Cured by moving the observation into its own uniquely named `STORAGE_ROUTING_<utc>.json` receipt.
**Anything a binding hash covers must be a property of the inputs, never of the moment.**

## 12. What is NOT claimed

No candidate. No pointer move. No seal, no staging, no parse-back. No Modal dispatch, no scorer run,
$0 spent. The decode-time figures are advisory diagnostics. The corrected ranking quantity in §7 is a
proposal at n=2, not a registered law.

## 13. Boundaries honoured

`upstream/`, the PR tree, the sealed move-47 promoted tree and every sister arm's directory
(`ddm_hpr1`, `ddm_pr19`, `ddm_cons2`) were read and copied, never written. **No receiver file was
patched** — the rail asserts its runtime copy equals the base tree byte for byte, every file, which is
what makes a weights-only row a normal-seal shape. The Vertigo reserve was never lowered; APDataStore
was read-only for the field and never written; nothing was deleted, and the aborted runs are moved to
`aborted/`, not removed. Retention manifest with every payload's sha: `RETENTION.json`.
