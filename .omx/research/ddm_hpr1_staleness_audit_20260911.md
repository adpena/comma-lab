# ddm_hpr1 — the STALENESS AUDIT: every counted section, what it was fit to, and whether that moved

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false ·
`# FORMALIZATION_PENDING: an audit of provenance and a ranked refit queue; the one law it proposes (a section fit to a superseded state of the object is owed a refit before any structural rung on it is priced) is registered as a canonical equation only when a second section's refit prices it, the hpac refit being n=1.`

Base: move 47, S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600], archive
`d1fab05d69f31c90…`. MAIN commissioned this after the day measured the same thing three times.

---

## 0. Why this audit exists

Move 47 was won by retraining the HPAC prior on the token field it actually codes. It bought **−887 B
for +351 B of model** — and it was found by accident, as the CONTROL of a shape experiment that
failed. The generalisation is the question this audit answers: **which other counted sections are fit
to a state of the object that has since moved, and what is the cheapest refit that keeps the receiver
unchanged?**

## 1. The census — MEASURED

Read with the shipping member splitter off move 47's promoted archive:

| member | bytes | what it is |
|---|---:|---|
| `header` | 14 | RX1M section lengths |
| `hpac` | 12,262 | the arithmetic coder's PRIOR (IntegerHPAC weights) |
| `semantic` | 29,862 | the renderer / SM3R body + its shared-mixer coding |
| `carrier` | 18,450 | the pose carrier: int12 coefficient codes + the CPR1 AR(1)+bias predictor |
| `tail` | 118,671 | 96 B residual prefix + **64 B counted RLC1 rider** + **118,511 B arithmetic stream** |
| member total | 179,259 | |
| ZIP overhead | 100 | |
| **archive** | **179,359** | |

## 2. The identity history — MEASURED, by re-reading 52 archives on both tiers

I hashed the `semantic` and `carrier` members of every `candidate_runtime/archive.zip` on both SSD
tiers. The distribution is the finding:

- **6 distinct `semantic` members have ever existed; 15 distinct `carrier` members have.** The carrier
  has been worked; the renderer has not.
- **`semantic` = `786950a5…` @ 29,862 B is byte-identical from the `ddm_tc3` move-39 tree through
  moves 45, 46 and 47** — unchanged across every archive in that span.
- **`carrier` = `7e222f00…` @ 18,450 B is byte-identical across moves 45, 46, 47** (pc3's refit).
- **The tail's counted STATE is byte-identical across moves 45, 46, 47**: `tc1_weights` 60 B sha
  `76f10171e42d…` and the 96-byte prefix `8ab2fe748ab7…`. Only the coded stream moves
  (119,749 → 120,107 → 118,511).

**So of the four counted sections, exactly one was refit in the last three moves, and it is the one
that produced the last two pointer moves.**

## 3. Provenance — from the store (arm-searched; every row cites its memo or commit)

### 3a. A correction to my own framing, first

I wrote in `ddm_hpr1_hpac_receptive_field_shape_rung_20260911.md` that "the field moved at move 32".
That is a FIRST-change-after-cl2 statement, not a last-change statement, and as a staleness figure it
is wrong. **MEASURED from `.omx/state/pointer_move_events.jsonl`: the token field moved EIGHT times —
moves 31, 32, 35, 38, 39, 40, 42 and 43 — and last moved at move 43 (sj1 pass 6, commit `48109233e`),
producing the current `subset6.u8` / `a92e7d90…`.** Moves 44–47 are field-invariant by receipt. The
correction makes the HPAC prior's staleness *worse*, not better: cl2 fit it at move 26 and it then
coded eight different fields. The −887 B was the accumulated bill.

### 3b. The table

| section | last CONTENT fit | fit to | moved since? | citation |
|---|---|---|---|---|
| **`hpac` prior** 12,262 B | **move 47** (this arm) | the **current** field `a92e7d90…` | **no — CURRENT** | commit `40518b844` |
| **`semantic`** 29,862 B | **never in the move 24→47 window.** Body `17e0fd0b…` (SM3R v1 mode 6, from fx5 / stage-08 `3948ccfc…`, August lineage). Only its CODING moved: rc1 at move 27, sm1/cmp2 at move 37 | **unrecorded** — the store names the checkpoint, not its training data | **STALE; and the staleness cannot even be dated** | `ddm_sm1_…_20260909.md`, `ddm_cmp2_…_20260909.md` (`310729f13`), `ddm_wd4_warm_lineage_width_20260821.md` |
| **carrier codes** (in 18,450) | **move 43** (sj1 pass 6 re-solve, +8 B) | pose, on the candidate's own renders of the current field | **no — CURRENT** | `48109233e` |
| **carrier CPR1 predictor** (in 18,450) | **move 45** (`ddm_pc3`, −160 B) | the shipped codes, bit-identical | **no — CURRENT** | `01f2b66ad` |
| **tail: tc1 35-weight mixer** (in the 64 B rider) | fit on the move-33/34 body, composed at **move 36** | the **move-32** field `a73289e0…` | **STALE by 11 moves; it has coded four later field changes (35, 38–40, 42, 43) without a refit** | `ddm_tc1_…_20260909.md`; `ddm_cmp1_…_20260909.md` (`127a9b6c1`) states verbatim that the 35 weights are **unchanged** through the composition |
| **tail: tc3 +5 weights, 6th context map** | **move 41** | the move-40 field | **STALE by 3 moves** | `359146e2f` |
| **tail: 60 B counted rider geometry** | **move 44** (`ddm_rlc5`; rider built on move 40) | move-40 geometry, re-encoded on move 43's stream | **weights stale, encode current** | `99625f32f` |

### 3c. Successor checks — what is genuinely closed, and at what scope

- **ntb1, lossless non-tail levers:** *"Charter status: PARTIAL … No lossless winner exists among 13
  tested formats (26 real encodes) … HPAC pruning with a model-dependent tail re-encode was NOT
  implemented or measured. Renderer precision cuts with seg re-solve … were NOT run … Those are live
  obligations, not negative verdicts."* Scope **INSTANCE** (13 named formats). Does not close a refit.
- **ntb2, renderer precision:** *"The renderer's 29,862 B are not purchasable"* — damage-per-byte 102×
  to 1,864× the score's exchange rate — **but** *"The renderer closure is scoped to 3-bit at the
  shipped per-axis max-absolute quantizer, fixed tokens, no weight refit. **A quantization-aware refit
  is NOT closed.**"* Scope **FAMILY-at-one-formulation**. **The refit door is explicitly open and ntb2
  says so in its own words.**
- **pc3, carrier lattice:** *"CLOSED at formulation scope: on move 45's carrier, no coefficient-lattice
  refinement pays … the n600 ceiling independently closes the global rungs by cost alone (914 B against
  415.5 B payable)."* Scope **FORMULATION**; payable bytes for the whole family **349.9 B, CI [284.7,
  415.5]**. (The phrase "pc3 closed the carrier lattice at n600" is a paraphrase; it appears nowhere
  verbatim. The scope above is the record.)
- **Is there any record of the renderer being retrained on the current field? NO — a clean negative**,
  searched across `.omx/research`, the pointer-move ledger, and `git log -S`. The only fine-tune ever
  run (`ddm_ft1`, 2026-09-03, *before* the pre-distortion line began) **fired its own falsifier at the
  first evaluated epoch: d_seg +31.23 %.** That is a real warning attached to this door, not a closure.

## 4. The ranked refit queue

Ranked by **the byte mass the stale state PRICES**, not by the stale object's own size — that is the
lesson of move 47, where 351 B of model bought 1,238 B of stream.

| # | rung | stale state | bytes it prices | receiver? | cost | why this rank |
|---|---|---|---:|---|---|---|
| **1** | **Refit the tc1 tail mixer's 35 int8 weights on the current field** | fit to the **move-32** field, 11 moves stale | **118,511 B** (the stream) | **UNCHANGED** — the weights are counted data under the shipped mixer | one fit + one 600-frame encode on the existing rail; no training run | Best lever arm on the board: **60 B of state prices 118,511 B.** Exactly the genus that just paid −887 B, on the largest counted object, and the store has never priced it. `ddm_cmp1` states in its own words that it composed onto a changed field **without** refitting these weights. |
| **2** | **Refit tc3's +5 weights / 6th context map** with rung 1 | move-40 field, 3 moves stale | same stream | UNCHANGED | folds into rung 1's fit | Same instrument, same encode; separating them would waste an encode. Do 1 and 2 as one rung with both ON/OFF arms. |
| **3** | **Quantization-aware refit of the SM3R renderer body** | never refit in this window; training data unrecorded | **29,862 B** of its own + whatever d_seg slack it returns | UNCHANGED if only codes/scales move | a real training run + a seg re-solve; the expensive one | Biggest single prize and **explicitly left open by ntb2**. But carries ft1's measured warning (+31.23 % d_seg at the first epoch) and an unrecorded training provenance that must be reconstructed first. |
| — | hpac prior | — | — | — | — | **CURRENT at move 47. No action.** |
| — | carrier codes + CPR1 predictor | — | — | — | — | **CURRENT at moves 43/45.** pc3 closes refinement at formulation scope, 349.9 B payable. No action. |

**Before rung 3, one cheap owed step:** reconstruct and record what the SM3R body was trained against.
The store names the checkpoint (`3948ccfc…`) and not its data. A refit cannot be scoped, and its result
cannot be attributed, until that is written down.

## 5. The law this audit proposes

**A counted section fit to a superseded state of the object is owed a refit before any structural rung
on it is priced.** gs3 Addendum 44 already states it; move 47 is its first measured instance (−887 B);
rung 1 above is the test that would make it a law rather than an anecdote. Until a second section's
refit prices it, **n = 1** — say so.

## 6. Boundaries

$0. No Modal, no scorer, no launches beyond unit (1)'s composition rows. Read-only on every sister
arm's tree and on move 46's promoted tree; all writes under `/Volumes/VertigoDataTier/pact/ddm_hpr1/`.
