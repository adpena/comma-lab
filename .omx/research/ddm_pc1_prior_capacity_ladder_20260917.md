# ddm_pc1 — the PRIOR's capacity/rate trade: the ladder cannot be built on this receiver, and the coordinate that can be is already closed

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
`# FORMALIZATION_PENDING: the staleness slope is n=1 and ranks a refit rung; it is registered as a canonical equation only when a second byte-closed drift row exists, per the standing "first-order token price is a ranking not a charge" law.`

**Arm:** ddm_pc1 (Opus, 2026-09-17), charter `.omx/research/ddm_pc1_prior_capacity_ladder_against_the_shipped_field_charter_20260917.md` (commit 201610366).
**Pointer at spawn and at close:** move 55 — S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600], archive sha
`ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a`. **UNMOVED by this arm.**

Axis of every byte and site count below: `[macOS-CPU advisory; exact counts over retained payloads]`.
No scorer ran. No Modal. No training. Labels: MEASURED / DERIVED / INFERRED / ASSUMED as marked.

---

## Result first

**The chartered ladder cannot be built on the shipped receiver, and I established that before spending a GPU-hour.**
Three things, in the order they decide the question:

1. **STOP, charter step 4 (MEASURED).** Every width/depth rung is a RECEIVER CODE CHANGE. The archive carries prior
   WEIGHTS and nothing else; every shape the loader needs it takes from the model the *receiver* constructs from its own
   hardcoded constants. A wider prior therefore cannot be byte-closed on move 55's tree — it is another unit.
2. **The one capacity coordinate the receiver DOES parse is already closed in both directions (RECALL, MEASURED by three
   prior arms).** That coordinate is the per-output-channel bit depth, which the trainer moves through `--rate-lambda`.
   cl2 measured the more-model direction at **+506 B**; cl3 measured the less-model direction at **+224 B** and fired its
   pre-registered whole-axis falsifier; dpi1 landed on the knee at **+37 B** with a secant of −1.0925 against the −1
   break-even. The charter's falsifier — "no rung nets ≤ −25 B after real coding" — is therefore **FIRED**, by
   measurement that already existed.
3. **The one receiver-safe rung left, the 1.0× refit of charter step 2, is refused on a measured drift of 179 sites.**
   The shipped prior's training field and move 55's field differ at **179 token sites of 117,964,800** (1.52e-6). The
   drift that bought hpr1 its −887 B row was **11,128 sites — 62× larger**. At the measured conversion the rung is worth
   **−14.3 B** at its optimistic best, inside the 34.8 B container lottery and short of the −25 B bar.

**verdict_scope: formulation** — the shipped architecture's capacity knob, on this receiver, at this λ, on move 55's
field. The width/depth FAMILY is not closed; it is *unreachable from here* and belongs to a first-measurement unit.

I nearly got (3) wrong, and the way I nearly got it wrong is the durable part of this arm. §5.

---

## 1. The recipe (charter step 1) — MEASURED, recovered by proactive recall

The shipped prior is produced by a trainer the lab owns. Nothing had to be invented.

| element | what it is |
|---|---|
| producer | `tools/train_ddm_cl1_hpac_capacity.py` (in-tree, 65.6 KB) |
| profile | `cl2_shipped_ladder` (identical config to `jf1_joint_refit`, `device: mps`); `hpr1_shape_rungs` inherits it |
| law | 60-epoch cosine, warm start from the shipped prior's own weights, batch 8, lr 3e-3, QAT fraction 0.5, seed 20260716, λ = 1.0 |
| training data | the decoded n600 token field as `{"seg": uint8 (600,384,512), "spatial_token_sha256": ...}`; the cache contract is checked by `_verify_jf1_cache_payload` and the caller must pin the field sha |
| loss | the field's surprisal under the prior plus a λ-weighted rate penalty on the per-channel bit depths (`lr_bits`, `bit_eps`, `init_bits`) |
| weight coding | IHS1: one bit depth per output channel, nibble-packed, then that many two's-complement bits per weight; packed by `ddm_rx2_mc36_identity_race._pack_terminal_ihs1`, recoded by `rc1_adaptive_model_sections.restore_hpac`, then a Brotli q0–q11 race |
| pricing rail | `experiments/ddm_hpr1_shape_price.py` (model change on a fixed field) and `experiments/ddm_sj1_rlc1_price.py` (field change on a fixed model); both drive the SHIPPED decode loop with the symbols injected |
| ancestry | shipped weights are our own rx2 lineage (a ~634-epoch Metal burn with post-hoc epoch selection), not a repack of a third party's |

**The recipe is re-runnable.** Its two inputs are retained: hpr1's cache and init at
`/Volumes/VertigoDataTier/pact/ddm_hpr1/train_inputs/`, cl2's at
`/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/inputs/`. So charter step 1 does not STOP the arm.
Step 4 does.

**Which prior is actually shipped at move 55 — MEASURED by byte identity, not by a memo.** I split the RX1M member of
three archives and hashed the prior section:

| archive | prior section | sha256 (first 16) | identical to move 55 |
|---|---:|---|---|
| move 47 (`hpr1` retrain control), 179,359 B | 12,262 B | `fe913038ecc98105` | no |
| move 48 (`hpr1` retrain_frame_even), 179,111 B | **11,629 B** | **`925adb48dca8a77e`** | **yes** |
| move 55 (LIVE pointer), 179,255 B | **11,629 B** | **`925adb48dca8a77e`** | — |

So the shipped prior IS move 48's, unchanged for seven pointer moves, and the field it was fit on is hpr1's
`a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8`. That is an identity, not an inference, and it is what
makes §4's drift count the right one.

## 2. The STOP: every width/depth rung is a receiver change (charter step 4) — MEASURED, three independent receipts

**(a) The receiver hardcodes the topology.** `cpr1/inflate.py` in move 55's own tree carries `HPAC_PATCH = 64`,
`HPAC_CHANNELS = 64`, `HPAC_DELTA`, `HPAC_FILM_DIM` as module constants and passes them to the model constructor
(lines 33–35, 257–260). The reviewable receiver agrees: `submissions/mrs7/inflate.py` line 2100 constructs
`IntegerPrior(..., channels=PRIOR_CHANNELS, ...)` from `PRIOR_CHANNELS = 64`.

**(b) The stream carries no shape.** `cpr1/integer_model_io.py` enumerates the coded modules from `model.modules()` and
slices the depth vector by `module.weight.shape[0]`. Every row width comes from the model the RECEIVER built. The prior
stream is a bare depth vector plus packed weights whose parse is entirely determined by receiver-side constants — change
the width and the existing loader does not parse the new size, it mis-parses it.

**(c) The constants are inside the receiver's behaviour digest.** `runtime/f26_inflate.py` folds `HPAC_PATCH`,
`HPAC_DELTA`, `HPAC_CHANNELS` and `HPAC_FILM_DIM` into the `renderer_contract` that is hashed into the token-decoder
fingerprint — the `9f6e7168…` digest the addenda track as "receiver behaviour unchanged".

This reproduces, from primary sources, what hpr1 wrote on 2026-09-11 for the *shape* axis and it extends to *width*:
*"Not one shape bit ships in `archive.zip` … Every shape rung is therefore a RECEIVER CHANGE and routes to the
first-measurement chain — none can take the normal seal path."* Width is strictly worse than shape: a shape rung moves
taps and holds the stored value count, while a width rung changes the value count too.

Per charter step 4 I stopped rather than train a ladder that could not be byte-closed.

## 3. The coordinate the receiver DOES parse is closed in both directions — RECALL, all MEASURED

The only capacity-like quantity the archive carries is the **per-output-channel bit depth**. It is a precision knob, not
a width knob, and the trainer moves it with `--rate-lambda`. Three arms have already priced it through the real coder,
with the field held bit-identical so only model bytes and tail bytes move:

| arm | rung | Δ model B | Δ tail B | Δ joint B | verdict |
|---|---|---:|---:|---:|---|
| cl2 (2026-09-05) | λ 1.0 → 0.5 (more model) | +350 | +156 | **+506** | prior law FALSIFIED; secant +0.446 vs the −1 break-even |
| cl3 (2026-09-05) | λ 1.0 → 2.0 (less model) | −457 | +681 | **+224** | whole-axis falsifier FIRED; λ=4.0 not fired |
| dpi1 (2026-09-12) | restored bit-depth state | −400 | +437 | **+37** | on the knee; secant −1.0925; flat across a 1.84 bit/row swing |

cl3's own words: the capacity axis is **closed in BOTH directions on the shipped object**. dpi1's mechanism sentence is
the one that matters for this charter: *"the prior already sits on its joint model-vs-tail knee; restoring a dropped
state moves the solution ALONG the curve, not off it."*

The charter's falsifier ("no rung nets ≤ −25 B after real coding") is therefore already fired on the coordinate that can
be byte-closed. Re-firing it with a fourth losing rung would have bought nothing.

## 4. The last receiver-safe rung: the 1.0× refit — MEASURED drift, DERIVED refusal

Charter step 2 asks for a retrain on move 55's field. That rung is receiver-safe (identical topology, identical op
counts, only weight values move) and it has paid before, so it deserved a real answer rather than an assumption.

**MEASURED, token field against token field:**

| pair | drifted sites | of 117,964,800 |
|---|---:|---:|
| cl2's fit field `cc10a7b0…` → hpr1's field `a92e7d90…` (the drift hpr1's refit consumed) | **11,128** | 9.43e-5 |
| shipped prior's fit field `a92e7d90…` → move 55's field `ce13cdf3…` (the drift on offer now) | **179** | 1.52e-6 |
| move 54 → move 55 (pd8's pass 3) | 29 | 2.46e-7 |

**The conversion, from the only two byte-closed instances the lab has:**

| instance | drift | archive Δ | legs |
|---|---:|---:|---|
| hpr1 → move 47 | 11,128 sites | **−887 B** | model +351, tail −1,238 |
| dpi1 on move 48 | 0 sites | **+37 B** | model −400, tail +437 |

Slope −0.0797 B per drifted site (n=1). At 179 sites the rung is worth **−14.3 B** if the whole staleness credit is
spent and the model leg is charged nothing; add the one measured zero-drift row as an intercept and it is **+22.7 B**.
The optimistic number misses the −25 B bar and sits at 0.41 σ of the 34.8 B container-break lottery. **REFIT_NOT_OWED.**

An independent bound agrees and does not lean on n=1. The tail is 119,162 B over 117,964,800 symbols — **0.00808
bits/symbol**. A changed token perturbs only the predictions whose receptive field contains it: conv_a reads 23 taps,
conv_b1 14, conv_b2 5, conv_past 9 (hpr1's reconstruction). Generously ≤ 200 downstream predictions per changed token,
so ≤ 35,800 perturbed predictions. Even at 20× the mean rate the whole re-pricable mass is **< 725 B**, of which a refit
recovers a fraction — against a model leg that moved ±351–400 B in both measured instances.

Label: the 179 and 11,128 counts are **MEASURED**; the −14.3 B is **DERIVED** from an n=1 slope and ranks the rung, it
does not price it. I did not train, and I say so plainly: on this evidence a 60-epoch burn plus a full price would have
cost about five hours to land inside the noise, and the nearest measurement in the family (dpi1) landed on the losing
side of the bar.

## 5. What this arm nearly got wrong, and the guard that now exists

I first measured the drift as **27,189 sites** and read it as 2.44× hpr1's — a strong signal to launch the refit. It was
wrong by 150×. I had picked up `/Volumes/APDataStore/pact/ddm_pd8/base/argmax_move54.npy`: a file with the right name,
the right dtype and the right `(600, 384, 512)` shape, which is the **SegNet argmax**, not the coded token field. The
two planes differ at 27,303 sites *by construction* — Addendum 50 measured that the token field is an optimized
pre-image of renderer∘scorer and renders 2.825× better than the true partition, so they are SUPPOSED to disagree. A
scorer output substituted for a coder input and read as staleness.

The catch was a cheap cross-check: pd8's own pricing control `rlc1_bulk/fields/control.u8` disagreed with the file I had
taken for the same field. Two files claiming one identity is the signal.

**The guard is landed, not remembered.** `experiments/ddm_pc1_prior_refit_headroom.py` (commit `82b83a649`, 14 tests)
answers "is a refit owed at the live pointer?" in seconds and refuses to answer on an unverified plane:

- **Guard 1 — name the prior by byte identity.** It splits the prior section out of the live archive and each candidate
  ancestor and reports which ancestor is byte-identical. That ancestor's training cache is the fit field. §1's table is
  its output: `produced_by: ["move48"]`.
- **Guard 2 — every field arrives with a declared sha256 and is refused if its bytes disagree.** Run against the argmax
  plane under the token field's sha it refuses with the reason in the message. A negative control, run and recorded.

Receipt: `/Volumes/APDataStore/pact/ddm_pc1/probe/HEADROOM.json`, sha
`2766306da95243b88e69455d655cce90042c1262b73d95820ef43118ad314f3b`.

The honest limit of guard 2: it catches a plane whose bytes are not the ones its provenance names, which is the failure
that happened. It cannot catch a caller who hashes the wrong file and declares that hash. Only a decode of the live
archive settles that, and it costs a full pass.

## 6. Decode time and prior size (charter question)

- **A refit rung changes decode time by nothing measurable, by construction.** Same shapes, same op counts, same group
  order; only weight values differ. cl2 measured decode wall-clock across its rungs and found no rung slower than the
  shipped decode, with scheduling noise dominating the deltas.
- **A width rung WOULD change decode time.** The prior's forward pass over 190 groups × 600 frames is the dominant
  per-symbol cost, and channel count scales it roughly linearly. That is a second, independent reason a width rung
  routes to the first-measurement chain: it needs its own measured `t4_direct` leg, not an inherited one. Move 55's
  inherited leg (`SEAL_ddm_pd8_price_first_pass3_contest_cuda.json.decode_wall_clock.json`, 1,072.3 s) would not carry
  it. Note the live headroom: move 55's first T4 run inflated in 1,602.7 s against a 1,260 s ceiling on a
  byte-identical receiver (Addendum 71), so the decode budget is not a place with room to spend.

## 7. Sub-0.12 arithmetic, RE-DERIVED at move 55 (binding numbers expire at every pointer move)

S 0.13603403441098336 = rate 0.11935854664191482 + distortion 0.016675487769068534. Gap to 0.12: **0.016034034**.
Exchange 25/37,545,489 = 6.658589531221714e-07 S/B; one bar (25 B) = 1.665e-5 S.

- **RATE corner** at held distortion: archive ≤ **155,174.8 B** → demand **−24,080.2 B**.
- **DISTORTION corner** at held bytes: distortion ≤ 0.00064145 → **26.0× reduction**.
- Zero-distortion B_max 180,218.3 B: the archive is 963.3 B under the threshold at zero distortion.

**What this arm says about that demand.** The prior section is 11,629 B — **6.49 %** of the archive. Even deleting it
entirely and coding the tail for free would not reach −24,080 B. The token section it conditions is 119,162 B (66.48 %),
and ls1/ls2 already measured the receiver rung of that axis 8,365 B short. So the capacity/rate trade on the shipped
prior was never a corner-sized lever; this arm's contribution is to say so with the receiver gate and the drift
measurement rather than with a projection.

## 8. What I did NOT do, plainly

- **No training.** No rung was trained, so the charter's ladder table has no rows of mine. §2 says why for width/depth
  (unbuildable here) and §3–§4 say why for the two knobs that are buildable (already measured; refused on drift).
- **No 1.0× instrument check.** It is charter step 2, and I skipped it knowingly: its purpose is to validate rungs, and
  after the step-4 STOP there are no rungs to validate. Running it would have been apparatus for its own sake.
- **No receiver edit, no seal, no candidate, no Modal, no scorer, no dispatch.** No sister arm's rail was edited —
  sj1's docstring records why (a rail binds its own sha into every receipt, so an edit refuses in-flight resumes), and
  hpr1's and dpi1's stores were read only.
- **Nothing written to Vertigo.** cl2's, hpr1's and dpi1's payloads there were read.
- **No canonical equation registered.** The staleness slope is n=1. It goes in this memo labelled DERIVED, exactly as
  tmx1 labelled its own n=2 refit law "a proposal, not a law".

## 9. What outlives the row

**A prior refit is owed in proportion to the field's drift, and drift is a ten-second numpy diff.** The refit law now
stands at n=4 with one positive (hpr1 −887 B at 11,128 sites) and three negatives (tmx1 +20 B on the mixer, dpi1 +37 B
at 0 sites, and this arm's derived refusal at 179 sites). Before any future arm charters a prior refit, run the probe:
it names the shipped prior by byte identity, counts the drift, and returns OWED / NOT_OWED against the fire bar. That
converts a five-hour question into a seconds-long one and it refuses to answer on a plane it cannot identify.

**The next real question about the prior is a receiver-change unit, and it should be chartered as one or not at all.**
Its cost is now known up front: three receiver constants in `cpr1/inflate.py`, a matching change in
`runtime/f26_hpac_native.c` where `conv_past` is re-implemented with its stencil baked in (hpr1's silent-divergence
hazard — patch one and not the other and the archive decodes one field in the default path and another in native mode),
a changed behaviour digest, and a freshly measured decode leg against a ceiling the pointer is already brushing.

## Custody

Store `/Volumes/APDataStore/pact/ddm_pc1/` — `probe/HEADROOM.json`
(sha `2766306da95243b88e69455d655cce90042c1262b73d95820ef43118ad314f3b`), 640 KiB total, well inside the 2 GiB budget.
APDataStore free space 22 GiB at open and at close; nothing was written to Vertigo; no bulk payload was created, so no
reclaim was needed. Read-only inputs cited by path and sha in §1 and §4. Lane
`ddm_pc1_prior_capacity_ladder_20260917`. Commits: `82b83a649` (probe + 14 tests), this memo.

## Frontier line

**move 55 — S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600]** — UNMOVED by this arm. No candidate, no seal.
