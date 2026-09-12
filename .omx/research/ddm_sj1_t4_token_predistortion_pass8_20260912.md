# ddm_sj1 PASS 8 on the move-49 field — MEASURED, and the family CLOSES on this field

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_sj1_t4_token_predistortion_pass8_first_measurement_20260912`. Axis
`[macOS-CPU advisory, jg1 instrument, DALI GT lineage]` for the seg and pose legs; bytes
EXACT. **No score is claimed here.** Only `upstream/evaluate.py` on the shipped bytes is a
score, and MAIN fires.

---

## 1. The headline this pass did NOT expect to write

The charter routed pass 8 through the FIRST-MEASUREMENT seal chain, "exactly as ntb2 (move
46) and hpr1 (move 48) did". That route is **CLOSED to this candidate**, and it is closed
for a reason that has nothing to do with pass 8: **the `candidate_prefire_intent.v1`
contract, as frozen at ddm_ffi1, admits only RATE-ONLY, distortion-unchanged candidates.**
ntb2 and hpr1 were exactly that — a frame-even rounding change and a prior retrain, both
byte-identical in their decoded output. Pass 8 moves distortion by construction.

Three independent gates, all in `src/tac/candidate_seal.py`, each MEASURED by a real call:

1. `:2167-2169` — `raw_identity_n600.candidate_raw.sha256 == pointer_raw.sha256`,
   unconditional. A token-pre-distortion candidate changes the decoded field; its raw cannot
   equal the pointer's.
2. `:2077-2080` with `:2658-2659` — `rate_only_precheck` must carry
   `raw_identity_required: True` and a `derived_net_dS` computed **purely from the archive
   byte delta**, no d_seg and no d_pose term, and that number must clear the bar. At the
   −2e-05 bar the archive would have to SHRINK by more than 30.04 B. Pass 7 GREW the archive
   by 42 B and still cleared by 3.25×, because the seg credit paid for the bytes. This
   arithmetic cannot price a distortion candidate at all.
3. `:2371-2372` — pr19's identity-class risk mode, the mode the charter names, requires every
   declared `t4_direct` leg to carry the **candidate's own** `decoded_token_sha256`. MEASURED:
   moves 44, 46 and 48 all carry `a92e7d90…`; move 49 carries `fdf2255f…`; pass 8 will carry
   a third. The class is (this receiver, this decoded token plane), so a field change leaves
   it by construction and the first candidate carrying a new plane can never find a leg
   carrying it.

**No intent can therefore be EMITTED, let alone committed.** `tools/make_candidate_seal.py`
re-validates its own output and `unlink()`s it on failure, so there is no artifact to hand
over and no sha to quote. Saying otherwise would be a fabricated deliverable.

## 2. The route that IS open, and it costs nothing

The charter's premise about the timing leg is correct and its conclusion is one step short.
MEASURED, both refusals, by real calls:

| route | verdict |
|---|---|
| inherit move 48's `t4_direct` leg under the move-49 pointer | REFUSED — *"source measurement is not the pointer archive"* (`decode_wall_clock.py:589`) |
| inherit move 49's own leg | REFUSED — *"inheritance must point directly to a measured or t4_direct leg (never to an inherited one)"* (`:585-586`) |
| inherit a `t4_direct` leg MINTED FOR MOVE 49 | **OPEN** |

Move 49's fire on Tesla T4 **already paid** for a cold n600 public-entrypoint decode of the
shipped archive, and its harvest carries the receiver's own report line. Only the LEG was
never minted, because a row that inherits is never asked for one.
`tac.decode_wall_clock.build_t4_direct_leg` mints it from that receipt; MEASURED:
`validate_decode_wall_clock` returns **zero problems**, `measured_t4_decode_seconds =
1106.219164097` against the 1,260 s limit, and `inherit_decode_wall_clock` on it then
SUCCEEDS with `behavior_digests_equal True`.

**The general shape, worth a guard:** a row that seals by INHERITING becomes the pointer;
its own leg is then mode `inherited`; and its successor can neither inherit it nor reach back
to the grandparent's direct leg. Every fired pointer has the measurement already; minting the
`t4_direct` leg **at harvest time**, in the packet apparatus, keeps the chain open and needs
no fire, no Modal call and no authorization. Memory:
`an_inherited_decode_leg_blocks_its_own_successor_mint_t4_direct_at_harvest_20260912`.

## 3. What the 12,127-cell residual is actually made of

MEASURED on the two shipped bodies, and it reframes the family:

| body | flips | stored token WRONG | stored token ALREADY RIGHT |
|---|---:|---:|---:|
| move 48's field (pass 6 shipped) | 12,196 | 1,371 | 10,825 |
| move 49's field (pass 7 shipped) | 12,127 | 1,368 | 10,759 |

Of the **69 cells pass 7 repaired, 66 sat where the stored token was ALREADY CORRECT** and
only 3 where it was wrong. The wrong-token pool moved 1,371 → 1,368 across an entire pass and
holds at 11.3 % of the residual. **The family is repairing RENDERER BOUNDARY JITTER by
perturbing a NEIGHBOURING token, not correcting encoder error** — the same object
`[[residual_seg_debt_is_renderer_boundary_jitter_at_correct_tokens_20260908]]` measured at
86 % by a different route, now measured a second way on this body at 88.7 %. Every repair
therefore costs a token move the rate leg must pay for.

## 4. The gates that were measured before the search's output could be believed

| gate | verdict |
|---|---|
| base binding | move 49's SHIPPED decode and this arm's own admitted field agree on **600 of 600 planes, 0 differing**; the npz written is route (a), the shipped decode |
| **F6** control encode | the RLC1 pricer re-encodes move 49's own field to its archive **byte-identically** — 179,153 B, sha `73e41a66…`, twins agreeing, two independent processes |
| **F7** pose base | 4.543568679770593e-06 on move 49's own configuration, gate ratio **0.998587**, `instrument_gate_pass`; it reproduces pass 7's own resolved value exactly |
| **F12** rider | 64 B `RLC1` rider sha `0ebda100…`, stream 118,938 B sha `e2afd028…`, taken from the receiver's own reader |
| **F13** close identity | splicing move 49's own carrier codes back into its body reproduces `73e41a66…` at 179,153 B **exactly** |

## 5. Two instrument defects this pass caught, both by controls rather than by inspection

**(a) The successor reach instrument.** A search row's `accepted` entries are DICTS, so
`tuple(move)` yields the KEY tuple — identical for every move. That collapses 218 distinct
positions into 130, reports a wrong carryover column, and still passes F14. What caught it
was running the new instrument on pass 6 → pass 7 and demanding it reproduce every published
figure; it did not. The canonical key is `(token_y, token_x, new_class)`, MEASURED against
pass 7's own stored expectation at **154/154**, where `(token_x, token_y, new_class)`,
`(site_y, site_x, new_class)` and `(token_y, token_x, old_class)` find 0, 3 and 0. After the
fix the control reproduces pass 7 exactly: 221 cells / 218 tokens / 130 pairs, fresh 66 cells
/ 64 tokens over 161 moved pairs at 0.40993788819875776 per pair, carryover 155 cells / 154
tokens on 90 pairs, stale-new 0, F11 154/154 with zero pairs disagreeing.

**(b) The native-library class defect.** Pass 7 lost a parse-back AND a public smoke to
`KeyError: 'RLC1_GEOMETRY_LIBRARY'`, and cured it by exporting the library externally before
each launch — which cures the run and leaves the class alive. `build_receiver_libraries` now
reads the list off `inflate.sh`'s own build block: three libraries under the default
`F26_TOKEN_DECODER=python` (rc64 backend, the float64 free corrector with
`-ffp-contract=off -fno-fast-math`, and the RLC1 lane geometry), and `f26_hpac_native` only
under `native-hpac`, where the harness REFUSES rather than guess at the libomp recipe.
MEASURED after the cure, with no external export: the public path probe on move 49's shipped
tree returns `REACHED_TOKEN_DECODE` — the exact leg that died.

## 6. Pre-registration, written before the search was launched

Residual 12,127 cells. Charter band 0.5–1.2 % = **60.6–145.5 cells**. The arm's own point
prior is **168 cells** (17.2 fresh + 151 carryover), which DISAGREES with the band by 1.154×,
entirely through the carryover column, and that disagreement was written down before the
measurement. Stop rules: **A** below 52.1 cells (cannot clear even at the best banked
economics), **B** 52.1–175.5 (marginal), **C** above 175.5 (clears at even the worst). The
expected branch, said in advance, is **B** — so a B outcome cannot be narrated as a win.
Fifteen falsifiers pre-registered, including F14 (the arm's own reach law put at risk) and
F15 (the seal door proven by a real attempt, whatever the verdict).

---



## 6b. A point prediction, written before the merge

The charter's band and the arm's prior are both about REACH. The question that decides the
pass is the ADMISSION, so this pass pre-registered a point prediction of that instead.

Re-running the arm's own Lagrange sweep over ONLY the carryover set — the 88 pairs pass 7
edited and its sweep dropped, whose renders are unchanged at move 49 — with their own
measured seg, bits and resolved pose, against MOVE 49's base:

> **the optimal subset is the EMPTY one, at every λ.** The carryover half pays ZERO.

That is not a surprise once stated: pass 7 dropped those pairs *because* they did not pay,
and nothing about move 49 makes them pay more — their base pose is byte-identical (MEASURED,
558 of 558 unchanged pairs, max abs difference 0.0), their seg credit is the same and their
rate cost is the same. So the entire pass-8 yield must come from the FRESH column, the cells
on the 42 pairs pass 7's own repairs re-rendered, which the reach law puts near 42 × 0.410 =
17.2 cells — worth about −5.0e-06 at pass 7's own economics, **a quarter of the −2e-05 bar.**

The null control is the part worth keeping: dropping everything scores 0.13631822891280948
against the pointer's own 0.13632299781031237, a −4.769e-06 gap that is exactly the
local-versus-T4 pose-print class (local base 4.5435687e-06 against the T4 print 4.55e-06).
The first draft of this prediction carried MOVE 48's pose base for the non-kept pairs and
showed a +2.72e-05 phantom instead; the control is what caught it.

Three things would falsify the prediction: the carryover pairs admitting anything once pass
8's OWN re-solve replaces pass 7's resolved pose as the proxy; the fresh column exceeding
~34 cells, twice the reach law's rate; or a fresh cell landing on a pair whose pose damage is
NEGATIVE, which would make it a pose credit and change the economics the way pass 7's subset
did.

## 6c. A read-only tree that did not come back untouched

This arm imported from move 49's own shipped tree — the parse-back path and a carrier-splice
control — and Python dropped 29 `.pyc` files into 3 new `__pycache__` directories inside the
live pointer's runtime. **No gate would ever have fired:** the T4 runtime manifest
(`8f3c5026…`, identical to the fire's), both receiver digests and the archive sha all skip
bytecode caches. That is exactly why it was worth curing — the failure mode is not a refusal,
it is a true-sounding sentence in the next arm's memo. The files were removed, every digest
re-measured identical, and `sys.dont_write_bytecode = True` now covers the in-process imports
the probe's own `PYTHONDONTWRITEBYTECODE` never reached.
## 7. What pass 8 measured

**The reach was fine. The admission is what failed.**

| | measured |
|---|---:|
| residual entering | 12,127 cells |
| reach | **165 cells / 163 tokens / 96 pairs** = 1.3606 % |
| — carryover (pairs whose render did not move) | 152 cells / 151 tokens on 88 pairs |
| — fresh (pairs pass 7's own repairs re-rendered) | 13 cells / 12 tokens on 42 pairs |
| — new on a stale pair | **0** |
| admitted subset | **11 pairs / 19 cells / 19 tokens** |
| admission fraction | **0.115** (pass 7: 0.312, pass 6: 0.689) |
| net ΔS vs the pointer | **−1.424218456666515e-05** = **0.712 × the −2e-05 bar** |
| net like-for-like vs this instrument's null control | −9.473287063771485e-06 |

Three-leg decomposition of the admitted subset: **seg −1.6117e-05 · pose −4.864e-06 · rate
+6.739e-06**, summing to −1.4242e-05 (recomputed from components to 1e-15, never from a
rounded field). The pose leg carries the local-versus-T4 print class: this instrument's base
is 4.5435687e-06 against the T4's 4.55e-06, which alone accounts for −4.769e-06 of the total.

**Why it does not clear, measured rather than guessed:** 165 reach cells bought only 19
admitted. The rate leg is the reason. Pass 8's full field prices at **8.93 bits per changed
token against a 10.31 break-even** — a margin of 1.154× where pass 7's subset had 2.09× — so
most pairs' seg credit is eaten by their own bytes before the pose multiplier is applied at
all. The full 96-pair field scores 0.13643067, **+1.08e-04 WORSE than the pointer**; only the
subset helps, and not enough.

### Against the priors

| prior | cells | measured / prior |
|---|---:|---:|
| charter — 0.5–1.2 % of the residual | 60.6–145.5 | **1.134× the band's top** |
| this arm — 17.2 fresh + 151 carryover | 168 | **0.982× — essentially exact** |

The arm's prior won because it counted the CARRYOVER explicitly: 151 of the 168 it predicted
were positions pass 7 found and its sweep dropped, and 151 of the 165 measured are exactly
those positions. Stop-rule branch **B (MARGINAL)** fired, which is the branch the
pre-registration said to expect.

### The reach law, a third time

`fresh cells per re-rendered pair`: **1.156** (pass 6, pairs moved by sister arms'
argmax-NEUTRAL edits) → **0.410** (pass 7, moved by this arm's own repairs) → **0.310**
(pass 8, moved again by its own repairs). n = 3, and the sign has never reversed. F14 holds:
**not one new repair on a pair whose render did not move**, three passes running. F11 holds:
**151 of 151** prior-dropped positions re-found, zero pairs disagreeing.

### The prediction, graded

Falsifier #1 **FIRED**. The prediction said the carryover half would admit ZERO; **5 of the
11 admitted pairs are carryover.** The named cause was the right one — pass 8's OWN carrier
re-solve, not pass 7's resolved pose, decides those pairs, and the prediction used the latter
as its proxy. The prediction was directionally right and quantitatively pessimistic by 2.8×
on the net, and it still called the outcome.

### Frame 0, declined a third time

Swept on all 43 pose-bound dropped pairs. The adopt stage reports 12 pairs, +11 B selector,
standalone **−3.980e-05**. Inside the admission it **LOSES by +9.384e-07**, and its fixed
point BREAKS on 2 of the 12 adopted pairs (376, 501). Pass 6 declined it on a build blocker;
pass 7 measured +1.09e-06; pass 8 measures +9.38e-07. The standalone gain never transfers,
for the same reason each time: it is measured on pairs the Lagrange sweep DROPS, whose
shipped plane is the live row's, so its pose credit belongs to the base configuration and not
to this candidate.

## 8. The verdict

**The token pre-distortion family is CLOSED on the move-49 field, at FORMULATION scope.**

No candidate was built, no archive staged, nothing sealed. The charter's fire bar is
`net ΔS < −2e-05 on the RESOLVED pose`; the best subset reaches 0.712 of it. Building past
that would have been spending on a row that cannot be admitted.

The scope is FORMULATION and it is not a paradigm kill. This family re-opened at pass 6 and
again at pass 7 because the OBJECT changed underneath, and the reach law now says exactly how
much a re-render is worth: **about 0.31 cells per re-rendered pair, and falling.** A sister
arm that re-renders pairs re-opens it; nothing else does.

### The curve, so the closure is read off a trajectory and not one pass

| pass | residual | reach | admitted | admission fraction | net ΔS | per admitted cell |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 12,614 | 237 | 77.85 | 0.328 | −2.701e-05 | −3.469e-07 |
| 6 | 12,540 | 499 | 344 | 0.689 | −1.916e-04 | −5.571e-07 |
| 7 | 12,196 | 221 | 69 | 0.312 | −6.492e-05 | −9.408e-07 |
| **8** | **12,127** | **165** | **19** | **0.115** | **−1.424e-05** | **−7.496e-07** |

The per-cell economics did not collapse; the **admission fraction** did, from 0.689 → 0.312 →
0.115. That is the rate leg tightening as the cheap tokens are spent, and it is the thing to
watch on any successor field.

## 9. What this pass does NOT claim

- **No score of any kind.** Every S here is a PROJECTION on measured legs; only
  `upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.
- **No candidate, no archive, no seal.** Nothing was built past the admission.
- **No paradigm closure.** The scope is FORMULATION, on THIS field.
- **No end-to-end first-measurement attempt on a pass-8 candidate**, because no pass-8
  candidate exists. F15 was pre-registered as "prove the door by a real attempt"; what was
  actually run is every component of that door against real objects — including the LIVE
  POINTER itself, which is the same candidate class — and both inheritance routes through
  the real validator. That is a stronger measurement than a single refusal on one candidate,
  and it is still not the end-to-end attempt the falsifier named. Said plainly rather than
  rounded up.

## 10. Custody

- Store `/Volumes/VertigoDataTier/pact/ddm_sj1_pass8/` — 3.5 GB on disk; **45 custody
  payloads, 3,570,018 B, every one with bytes and sha256** in `RETENTION_MANIFEST.json`,
  under the 8 GiB cap. The 1.75 GB overlay and the encode checkpoints are rebuildable from
  the retained field npzs by the commands in `PIPELINE_PLAN.md` and are deliberately not
  custody payloads.
- Second copy
  `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass8_20260912/custody_pass8/` —
  all 45 files copied and **sha-verified on the destination**, not on the label.

<!-- # FORMALIZATION_PENDING: a closure verdict and an apparatus finding; no pointer row is produced, so there is no equations leg for tools/pointer_move_packet.py to write. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm):
**S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]** (move 49).
