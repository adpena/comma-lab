# DDM HV1 — harvest composition from the selected e960 checkpoint (2026-08-15)

**Charter:** `.omx/research/charters/ddm_hv1_harvest_compose_ep508_20260815.md`.
**Axis:** `[macOS-CPU advisory, scorer-free lossless composition]`. `score_claim: false`.
No scorer ran here, no Modal dispatch was performed, and the live run's store was
never written.

**RETARGET (MAIN adjudication, mid-arm).** The charter names ep508. The e960 burn was
governed-early-stopped and the landed selector then ran over ALL 81 retained periodic
checkpoints; the true argmin is **ep634** (estimated joint 130,393 B, top1 0.0018945397,
482 B better than ep508 at equal-or-better top1). Receipt:
`/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/gpu_race/full_e480b_e960/endpoint_closure/checkpoint_selection.json`.
The candidate carried into byte-close, the advisory comparison, and the sealed T4 fire
order is therefore the **ep634** build. The ep508 receipts are retained as the A/B point
under `ep0508/`. Both checkpoint SHAs were re-verified locally before use:
ep508 `68da5ee0135613ec9aebb1c323d26b475d1292e2006bfaad33bf1bd87659fa7a`,
ep634 `5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec`.

---

## VERDICT

**A byte-closed candidate 743 B smaller than the incumbent, at provably identical
distortion.** `s1p25_c1p0` @ brotli_q10 — **182,759 B**, SHA-256
`80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.

| | e480b v2 (incumbent) | hv1 ep634 | Δ |
|---|---:|---:|---:|
| archive bytes | 183,502 | **182,759** | **−743** |
| d_seg contribution | 0.029611 | 0.029611 (inherited) | 0 |
| d_pose contribution | 0.008294576541331089 | 0.008294576541331089 (inherited) | 0 |
| rate contribution | 0.12218644961582469 | 0.12169171641365491 | −0.0004947332021697733 |
| **S** | **0.1600920261571558** | **0.15959729295498598** | **−4.947332021697733e-4** |

Checked two ways, which agree to the last place: incumbent S plus the rate delta gives
0.159597292954986, and recomputing from components
(`0.029611 + 0.008294576541331089 + 25·182759/37545489`) gives 0.15959729295498598.

The incumbent row is `[contest-CUDA] T4 n600`. The hv1 row is a **projection**, axis
`[macOS-CPU advisory]`, `score_claim: false` — it is the incumbent's measured CUDA
components plus an exactly-computed rate delta, valid because the decoded frames are
byte-identical. It is **not a score** and does not move any pointer. The exact T4 row is
sealed and owed; MAIN fires it.

For scale: |ΔS| = 4.947e-4 is **49× the 1e-5 canonical naming bar** and **141× the
±3.5e-6 8dp report band**, so unlike the qs/re micro-edit family this candidate is not
sub-band — it would be a nameable move if the T4 row confirms.

The cost was ~53 minutes of local CPU on already-purchased training compute, and no
dispatch. Nothing was retrained; this is harvest.

---

## What this candidate is, and what it is not

The composition is a **pure rate object on a frozen distortion**. The RX2/MC36 chain
re-codes the *fixed* MC36 spatial-token stream with a better-trained HPAC probability
model and admits the result only under exact decoded-token identity plus full raw-output
byte identity against the MC36 CPU decode. Consequently:

- `d_seg` and `d_pose` are **EXACTLY** the incumbent's measured values by construction —
  identical decoded frames feed identical scorers. They are inherited, not re-measured.
- Only the **rate** term moves. `ΔS = 25 · Δbytes / 37,545,489`.

This is why the advisory stage of this arm is a byte count plus two identity proofs
rather than a local CPU-torch scorer run: raw-output byte identity is *strictly stronger*
evidence than re-running a scorer on a non-authoritative axis, because any deterministic
function of identical bytes is identical.

**The one residual assumption, stated so it is not smuggled.** The identity I MEASURED is
on the CPU decode axis: this candidate's full raw output is byte-identical to the MC36 CPU
decode, which is the same raw output the incumbent produces
(`…/ddm_rx2_current_mc36_label_hpac/retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json`,
raw `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`, 3,662,409,600 B).
The incumbent's `d_seg`/`d_pose` were measured on the **CUDA** axis. Carrying them across
therefore assumes the decoder is device-deterministic — a well-founded assumption for an
integer receiver, and one the chain's own design rests on, but it is an ASSUMPTION on this
arm's evidence, not a measurement. The T4 row in the sealed fire order is what converts it
to a measurement; until it fires, the projected score below is `[macOS-CPU advisory]` and
carries no score claim.

### Base lineage — explicitly, for the record

`experiments/ddm_rx1_rate_representation_attack.py:40` pins the conventions base to the
MC35/MC36 micro35 candidate `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip`
(186,269 B, sha `f0ba4bb4…`). That is the **same base the incumbent e480b v2 was built
from** — `BUILD_RESULT_s1p25_c1p0.json` records exactly that `base` and e480b v2 as its
`winner`. The comparison is therefore apples-to-apples by construction. Measured proof
that only the HPAC changed:

| frozen section | hv1 sha256 (mine) | identical to e480b v2 |
|---|---|---|
| semantic stream (34,763 B shipped) | `4099eab6fc18af5b…` | **yes** |
| carrier stream (22,161 B shipped) | `fd14aabcb9daa5f1…` | **yes** |
| base residual (100 B) | `bd27a2ddb1706799…` | **yes** |
| base tokens (115,238 B) | `b44367b18f5f6258…` | **yes** |
| HPAC IHS1 | ep634 `e8c0cfd73d32…` | **no — this is the only change** |

All admission arithmetic in this arm is against **e480b v2** (183,502 B; seg 0.029611 /
pose 0.008294576541331089 / rate 0.12218644961582469 / S 0.1600920261571558
`[contest-CUDA] T4 n600`). The delegated RX2 receipts also carry a field
`projected_score_if_mc36_distortion_held`, which is the DONOR's arithmetic against the
186,269 B mc35 base — that field is **not** this arm's admission number and must not be
read as one.

---

## Which weights the export consumed (reviewer H7)

**The EMA shadow, not the live weights — measured, not inferred.** The packer loads
`checkpoint["state_dict"]` (`experiments/ddm_rx2_mc36_identity_race.py:268`). On the ep634
checkpoint that tensor set is **byte-identical to `checkpoint["ema"]["shadow"]` on all 37
tensors and differs from `checkpoint["live_state_dict"]` on all 37**:

```
deployment_weights           = "ema_shadow"
history[-1].evaluated_weights = "ema_shadow"
state_dict vs ema.shadow     : 37 identical / 0 differ
state_dict vs live_state_dict:  0 identical / 37 differ
ema decay = 0.9998720867875375, num_updates = 47,550
```

This matters twice over. It satisfies the EMA non-negotiable (inference and archive bytes
come from the shadow, never the live final-step weights). And it makes the selector's
arithmetic honest: the `estimated_joint_bytes = 130,393` that chose ep634 was itself
computed with `evaluated_weights: "ema_shadow"`, so the selector priced **the same tensors
the export serialized**. A live-weight export would have been a different object from the
one that was ranked, and the ranking would not transfer.

## Charter erratum — stage (3) as written was wrong

Recorded per MAIN's adjudication (the fold is ADMITTED; MAIN re-verified both receipts at
source). The charter's stage (3) directed re-applying qs2 and re1 as banked, un-spent,
additive edits. Against the bytes they are neither banked nor un-spent: they were consumed
into MC36 and ship inside the frozen carrier of every archive in this lineage. The
instruction, executed literally, would have produced a **cross-regime double-count** — the
same class as transplanting a constant across lattices — and would have broken the
decoded-token and raw-identity gates on the way. The corrective is structural and general:
*before re-applying a banked edit, read the target object's bytes for the edit's own magic
rather than trusting the ledger's "banked" label.* A ledger records what was measured; it
does not track what a later merge absorbed.

## Stage (3) — the micro-edit recompile: FOLDED, with evidence

The charter directs re-applying the banked qs2 (−4.374914e-6) and re1 Round-1
(−1.2068738e-6) micro-edits, recompiled against this archive's coder. **That work is
already done and shipped: both edits are ALREADY INSIDE the frozen carrier this
candidate composes on.** Re-applying them would double-count ancestry and would break the
two identity gates that define the RX2 lineage.

Measured, by me, from this arm's own frozen carrier payload
(`ep0634/retained/models/mc36_frozen/carrier.br`, 22,161 B, sha `fd14aabcb9daa5f1…`):

```
brotli -d  →  22,219 B, sha 065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f
             = 22,183 B packed CAP1 body  +  36 B tail
tail[0:4]  =  b'Q2C1'                      (experiments/ddm_qs2_compensation_overlay_runtime.py MAGIC)
decode     →  pairs (7, 96, 105, 176, 178, 517, 523), 12-dim int12 delta codes per pair
```

Pairs 7 and 96 are re1's; 105/176/178/517/523 are qs2's. The union build that consumed
them is explicit in code: `experiments/ddm_mc35_micro35_union_build.py:90-101`
(`QS2_PAIRS = (105,176,178,517,523,532)`, `RE1_PAIRS = (96, 7)`), and
`.omx/research/ddm_na7_negative_signal_audit_20260814.md:59,73,121,224` records both as
**CONSUMED / FOLDED by MC36 — "not additive residue", "adding it again would
double-count ancestry"**.

Two consequences, stated plainly:

1. There is no admissible qs2/re1 recompile on this lineage. The charter's stage (3)
   premise — that these are banked, un-spent, additive edits — does not hold against the
   bytes.
2. Their ΔS is **already inside** the incumbent's measured 0.029611 seg / 0.0082946 pose
   components. Adding their headline deltas to this candidate's projection would be
   double-counting; the projection below does not do that.

A genuinely new micro-edit family would be a *distortion* change on a new support
(qs5's `solve_exact_object` protocol re-solved in-compile — it does re-target, e.g.
`ddm_mc35_micro35_union_build.py:1114,1150`), and would require its own dual-axis T4 row.
That is a separate arm, not this one.

---

## The chain that ran

Delegation, not reinvention: `experiments/ddm_hv1_harvest_compose.py` imports
`experiments/ddm_rx2_mc36_identity_race.py` (sha
`27d22573963374d73380465095393dcd7953e9c69c7facf32993b8e5262bc6ce` at use) and drives its
stages. Two documented deltas, both forced by the situation and neither reducing a custody
invariant:

1. **Epoch-scoped store redirection.** RX2's roots are module globals. Run unmodified it
   would have overwritten the custodied e480b v2 artifacts *and* silently resumed its 600
   completed base-probability frames — pairing one epoch's weights with another epoch's
   logits, a silent wrong number. The runner rebinds both roots to
   `…/ddm_hv1_harvest_compose/ep<NNNN>/` and asserts the rebinding took effect. Bulk lives
   on APDataStore because the Vertigo tier is at capacity (975 MB free at arm start —
   flagged for MAIN); the small payload set and every receipt are mirrored to the
   charter-mandated Vertigo retention root.
2. **Preflight adaptation.** RX2's preflight demands a terminal trainer report with
   `history[-1].epoch == packed epoch` plus the artifact manifest; a mid-run periodic
   checkpoint has neither. `hv1_preflight` keeps every check on objects that exist (base
   archive, expected spatial, source event-order digest `f4149ab6…`, checkpoint
   schema/phase/profile/epoch, causal-state hash **re-verification**) and replaces the
   report check with the pinned selection sha, the checkpoint-embedded telemetry row, and
   the resume-lineage pin to the e480b parent (`cd89907b…`, epoch 480). Scope substitution
   on the report surface only.

---

## Measured: the HPAC pack A/B

| checkpoint | IHS1 raw B | shipped (brotli q10) B | Δ shipped vs e480b v2 |
|---|---:|---:|---:|
| ep480 — e480b v2 incumbent | 17,996 | 13,619 | — |
| ep508 — charter's original pick | 18,023 | 13,600 | −19 |
| **ep634 — selected** | **17,952** | **13,515** | **−104** |

Trainer telemetry for the same three, re-derived from the primary logs rather than from
prose (advisory estimates, `ADVISORY_ESTIMATE_NOT_SERIALIZED`):

| checkpoint | est. token B | est. model B | est. joint B | top1 error |
|---|---:|---:|---:|---:|
| ep480 | 113,229 | 17,991 | 131,220 | 0.0019019402398003473 |
| ep508 | 112,857 | 18,018 | 130,875 | 0.0018965996636284722 |
| **ep634** | **112,446** | **17,947** | **130,393** | **0.0018945397271050348** |

ep634 improves *both* terms against ep480 (−783 B estimated token, −44 B estimated model)
at a lower top-1 error, which is why it is the argmin. Note the estimate↔realization
identity measured at ep480: estimated token 113,229 equalled the **realized neutral RC64
stream exactly** (`retained/coders/neutral/RC64_RESULT.json`).

## Measured: the n600 RC64 race

Every row is a complete n600 native-RC64 stream on the real symbols, closed through the
shipped receiver with decoded-token identity — not an entropy estimate and not a subset.
The build stage races all 13 lossless model representations per variant and takes the
minimum; `brotli_q10` won every time, as it did at ep480.

| variant | model repr | model B | token B | archive B | Δ vs e480b v2 | token identity / repeat |
|---|---|---:|---:|---:|---:|---|
| `neutral` (control) | brotli_q10 | 70,453 | 112,446 | 183,095 | −407 | true / byte-identical |
| **`s1p25_c1p0`** | **brotli_q10** | **70,453** | **112,110** | **182,759** | **−743** | **true / byte-identical** |
| `s1p0_c1p25` | brotli_q10 | 70,453 | 112,114 | 182,763 | −739 | true / byte-identical |
| `s1p0_c1p0` | brotli_q10 | 70,453 | 112,121 | 182,770 | −732 | true / byte-identical |
| `s0p75_c0p75` | brotli_q10 | 70,453 | 112,121 | 182,770 | −732 | true / byte-identical |

Candidate denominator: 5 variants × 13 lossless model representations = **65 complete
archives**, every one receiver-closed. The winner is `s1p25_c1p0` @ brotli_q10:

**archive 182,759 B, SHA-256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.**

**Determinism, at two strengths.** In-process, every candidate's `archive.repeat.zip` is
byte-identical to its `archive.zip` (the column above). Beyond that, the chain was run a
second time in a **separate process from a separate governed launch**, which rebuilt the
winner from the checkpoint through the pack, the table fit, the coder, and the packer —
and reproduced the same SHA-256 exactly. The table fit also re-selected the identical
variant set. Cross-process reproduction from the checkpoint is a stronger receipt than an
in-process repeat, because it re-executes every stage rather than re-serializing one
in-memory object.

**Where the −743 B comes from, decomposed:**

| section | e480b v2 | hv1 ep634 | Δ |
|---|---:|---:|---:|
| model (HPAC + frozen semantic + frozen carrier + 14 B wrapper) | 70,557 | 70,453 | **−104** |
| token (n600 native RC64) | 112,749 | 112,110 | **−639** |
| residual (compact RCF1) | 96 | 96 | 0 |
| member | 183,402 | 182,659 | −743 |
| **archive** | **183,502** | **182,759** | **−743** |

The model saving is entirely the HPAC (13,619 → 13,515); the frozen semantic and carrier
sections are byte-identical, as the table above proves.

**The token saving decomposes, and the decomposition is a finding.** Both checkpoints were
raced with a neutral control, so the model's contribution and the table's separate:

| | ep480 (incumbent) | ep634 (this arm) | Δ |
|---|---:|---:|---:|
| neutral token (model alone) | 113,229 | 112,446 | **−783** |
| fitted-table gain over neutral | −480 | −336 | **+144** |
| realized winner token | 112,749 | 112,110 | −639 |

The better model buys 783 B, and then **gives 144 B of it back** because the re-fit table
has less left to correct. That is the expected shape — the table is a residual corrector on
the model's boundary-bucket miscalibration, so a better-calibrated model leaves it less
work — but it means the two levers are **substitutes, not addends**. Any future projection
that adds a model improvement to a fixed table gain will overestimate.

---

## DEAD-ENDS

1. **qs2/re1 micro-edit recompile — structurally excluded (see stage (3) above).** Not a
   failed attempt: the edits are already consumed into the base. Any future arm proposing
   to "re-apply the bank" onto an RX2-lineage object should read the carrier tail first.
   The general lesson is the charter erratum above: a "banked" ledger label survives the
   merge that spent it, so the bytes — not the ledger — decide whether an edit is still
   additive.
2. **Running RX2 unmodified against a new checkpoint — silent-wrong-number hazard.** The
   export stage resumes from existing per-frame receipts, so a shared store would have
   paired ep634 weights with ep480 logits and produced a plausible, wrong archive. The
   cure is structural (epoch-scoped roots + fail-closed rebinding assertions), not
   procedural.
3. **A conservative memory projection is not free.** The first detached launch was
   REFUSED by the system admission gate (rc=5, receipt fired at 15:43:51Z) because I
   priced it at a guessed 24 GiB against a 116 GiB adaptive ceiling with 70.1 GiB already
   used. The refusal was correct and is the apparatus working. Cure: measure the real
   peak on a 6-frame slice (**1,625 MiB**), then price the launch at 4 GiB — admitted
   immediately. Guessing high is not the safe direction; it is a different failure.
4. **One projected peak for a chain of unlike stages.** The coding stages sit at ~1.6 GiB
   while the full-raw decoder is larger (measured 1.9 GiB peak). Pricing them together
   forces the whole chain to the decoder's number. Cure: `--stop-after`, so each phase is
   admitted at its own measured footprint.
5. **A hardcoded label survived a retarget.** The first sealed fire order carried
   `candidate_id: hv1_ep508_…`, plus an `ep508` lane id, instance-job id, and results
   directory — all string literals frozen before the ep508→ep634 retarget, all pointing at
   an epoch that produced none of the sealed bytes. The archive hashes were right; the
   NAMES were wrong, which is the phantom-directory bug class (a downstream reader trusts
   the label, not the hash). Caught by reading the emitted JSON back instead of trusting
   that the emitter was correct. Cure: every label is now derived from `--epoch`, and the
   order was re-emitted and re-verified to contain no `ep508` anywhere. The general form:
   **a retarget must sweep string literals, not just the data they describe.**

## LIVE-HYPOTHESES

1. **The trainer's `estimated_token_bytes` is an EXACT predictor of the realized neutral
   RC64 stream — now measured twice, promoted from hypothesis toward law.**

   | checkpoint | estimated token B | realized neutral RC64 B | error |
   |---|---:|---:|---:|
   | ep480 | 113,229 | 113,229 | **0** |
   | ep634 | 112,446 | 112,446 | **0** |

   Two independent checkpoints, exact to the byte. The estimate is evidently the same
   arithmetic the coder realizes, not a correlate. Consequence: the training telemetry is
   a **free, zero-compute ranker** for any future checkpoint harvest — the 18-minute
   base-probability export is needed only to BUILD a candidate, never to RANK one. The
   selector already exploits this; this arm is the second confirmation that it is safe to.
   Falsifier for a third arm: a checkpoint whose realized neutral stream differs from its
   estimate by even one byte.
2. **The fitted table is genuinely checkpoint-specific — measured, not assumed.** ep480's
   development screen selected `{s1p25_c1p0, s1p25_c1p25, s1p25_c0p75, s1p5_c1p25}`;
   ep634's selected `{s1p25_c1p0, s1p0_c1p25, s1p0_c1p0, s0p75_c0p75}` — a different
   family, with the shrink optimum moving down. Transplanting ep480's table onto ep634's
   probabilities would have been exactly the cross-regime constant transfer the charter
   forbids, and the fit disagreement is the direct evidence that the prohibition has
   teeth here rather than being ceremonial. **The gain magnitude does NOT transfer either**
   — measured −480 B at ep480 against −336 B at ep634. Model quality and table gain are
   SUBSTITUTES: a better-calibrated model leaves the residual corrector less to correct.
   Consequence for ranking: `estimated_joint_bytes` alone slightly *overstates* the
   realized advantage of a better checkpoint, because it does not know the table will give
   some back. The ordering is preserved (both terms move the same way); the magnitude is
   not. Falsifier: a checkpoint pair where the better model also earns a larger table gain.
3. **The model section still has room the burn is not buying.** ep634's IHS1 raw is 44 B
   *smaller* than ep480's while its shipped brotli is 104 B smaller — the QAT tail is
   improving compressibility faster than it is shrinking the raw object. Whether that
   continues past ep634 is the open question the early stop closed off; the mz1 result
   (8/8 lossless model-section races lost) says the *coder* has no slack left, so any
   further model-section byte must come from the training tail or from a different
   representation.
4. **A checkpoint-harvest ladder may be cheaper per byte than any coder work.** This arm
   spends ~35 minutes of local CPU to convert already-purchased training compute into
   archive bytes. If the projection below holds, that is a better bytes-per-hour rate than
   the mz1 model-section race (0 B for a full lossless race) achieved.

---

## Retention (ALWAYS KEEP THE PAYLOAD)

Per-epoch stores, both tiers, with SHA manifests:

- Vertigo (charter-mandated retention root):
  `/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/ep0634/` — winning archive,
  determinism-repeat archive, member, model section, token stream, residual, `SHA256SUMS`,
  and every receipt. `ep0508/` holds the A/B point's receipts.
- APDataStore (bulk):
  `/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/` — all 600 base-probability
  frames, every candidate variant's quantized probabilities and RC64 stream, every
  representation's archive, the full raw decode, and the retention inventory.

Checkpoint payloads (`retained/epoch_0508.pt`, `retained/epoch_0634.pt`) are copies; the
live run's files were never opened for write.

---

## Provenance pins

| object | pin |
|---|---|
| selector | `tools/select_hpac_checkpoint.py` (landed `5624ef8bdc`) |
| selection receipt | `…/full_e480b_e960/endpoint_closure/checkpoint_selection.json` |
| early-stop receipt | `…/full_e480b_e960/endpoint_closure/governed_early_stop_receipt.json` |
| conventions donor | `experiments/ddm_rx2_mc36_identity_race.py` sha `27d22573…` |
| base archive | mc35 micro35 candidate 186,269 B sha `f0ba4bb4…` |
| incumbent | e480b v2 183,502 B sha `e3e6f440b45bbb92…` |
| ep634 checkpoint | 1,103,503 B sha `5007beae7af77897…` |
| ep508 checkpoint | sha `68da5ee0135613ec…` |
| swap procedure | `.omx/research/ddm_pq1_submission_packet_prep_20260815/SWAP_PROCEDURE.md` |
| runner | `experiments/ddm_hv1_harvest_compose.py` |
| sealed T4 order | `.omx/research/ddm_hv1_t4_sealed_fire_order_ep0634_20260815.json` |

---

## The identity gates — MEASURED, both green

The whole projection rests on these two, so they are the load-bearing rows of this memo:

| gate | measured | meaning |
|---|---|---|
| full raw-output decode | 3,662,409,600 B, SHA-256 `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` | **byte-identical to the MC36/e480b v2 CPU decode** |
| decoded spatial tokens | `raw_identity_vs_mc36_cpu: true`, `decoded_token_identity: true` | the receiver reproduces the exact token field |

Decode wall 1,369.7 s through the custodied F26P lifted CPU runtime, four threads, axis
`[macOS-CPU advisory, four-thread lifted F26]`. Receipt:
`ep0634/retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json`.

Because the decoded bytes are identical, every deterministic function of them is identical
— including SegNet's argmax and PoseNet's output. That is what licenses carrying the
incumbent's d_seg and d_pose across unchanged, and it is why no local scorer was run.

## The sealed T4 fire order

`.omx/research/ddm_hv1_t4_sealed_fire_order_ep0634_20260815.json`, schema
`ddm_hv1_t4_sealed_fire_order.v1`, disposition **QUEUED_WITH_A_FIRE_ORDER**, owner **MAIN**,
`score_claim: false`. **Nothing was dispatched by this arm.** The order pins every hash off
disk at seal time (re-verifying custody rather than copying numbers from prose), refuses if
the in-process repeat is not byte-identical, and refuses if either identity gate is false.

Its four preconditions keep MAIN's ownership intact: the CUDA runtime tree must be staged
and verified through the landed `SWAP_PROCEDURE.md` steps VERIFY_SOURCE and
STAGE_NEW_GENERATION (runtime custody is MAIN-owned, so the command template leaves those
two fields as explicit placeholders rather than guessing); reconciliation must report zero
live remote calls and zero active claims; the unique lane claim must succeed; and every
sealed hash must still match on disk. The CPU axis is noted separately — pq1's sealed CPU
order targets e480b v2 at swap generation 0, and no e480b receipt transfers to changed
bytes.

## Own-vehicle frontier

**Unmoved: e480b v2 S 0.1600920261571558 @ 183,502 B `[contest-CUDA T4 n600]`.**

This arm did not move it and cannot: an advisory projection is not a score. What it
produced is a byte-closed candidate that would move it to **0.15959729295498598 @ 182,759 B**
if the sealed T4 row confirms — a −4.947e-4 step, closing 4.9% of the 0.0100920 gap to
sub-0.15. The honest state is: the candidate exists, is retained, is receiver-closed, is
determinism-checked across processes, and is waiting on one MAIN-owned dispatch.

