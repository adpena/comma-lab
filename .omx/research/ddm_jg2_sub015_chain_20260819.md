# ddm_jg2 — the sub-0.15 chain: replace the modelled rate leg with a measurement

- **arm** `ddm_jg2` (task #1139 — the successor to `ddm_jg1`'s joint solve)
- **date** 2026-08-19
- **axis** every number is `[macOS-CPU advisory]` unless it carries an explicit DALI-lineage
  tag. `score_claim=false` · `promotable=false`. This arm fires **no Modal job**; MAIN owns
  the T4 slot.
- **cost** $0.
- **store** `/Volumes/APDataStore/pact/ddm_jg2/`
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.
  **Pointer UNMOVED** at contest-CUDA `0.15652626435208142` until a T4 row says otherwise.

## THE BASE (re-read from `.omx/state/canonical_frontier_pointer.json` at arm start)

| term | value | S contribution |
|---|---:|---:|
| `d_seg` | 0.00030309 | 0.030309 |
| `d_pose` | 7.649246787e-06 | 0.008746 |
| archive | 176,420 B | 0.117471 |
| **S** | | **0.15652626435208142** |

`archive.zip` sha `7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f`.
**Gap to sub-0.15 = 0.006526.**

## THE INHERITED PROJECTION, AND THE ONE LEG THAT IS NOT MEASURED

`ddm_jg1` (memo `.omx/research/ddm_jg1_joint_solve_20260819.md`) established, at $0:

1. a **validated** local contest-axis seg instrument (`0.99995x` of the T4 seg leg);
2. the **move-class law** — single-cell token coordinate moves repair ~1.5 argmax cells per
   changed token and compose within a sparse pass; block/dilation moves realize worse;
3. the **hard negative and its reversal** — token seg edits destroy pose (`x387`), but
   re-running the carrier's own coordinate descent against the edited frame recovers
   `d_pose` to `1.073x` of original at ~0 bytes. **The actuators compose.**

Its rate leg is **modelled, not measured**: `+4.718 bits` per changed token, computed from
the **hm1/182,759 B body's** probability model, then transferred to the to1/up3 body we
actually ship. jg1 names three reasons that constant is suspect, and **all three point the
same way — the real price is likely HIGHER**:

| # | risk (jg1 S1d caveats 3-5) | direction |
|---|---|---|
| 3 | cross-body transfer: to1's model is **sharper** (0.007446 vs 0.007603 bits/token) | costs MORE |
| 4 | context coupling: the HPAC model decodes in 190 groups, feeding decoded tokens forward | costs MORE |
| 5 | the table correction is omitted from the marginal number | unknown sign |

Two extrapolations exist and they disagree, which is itself information:

| source | repaired cells | changed tokens | net S |
|---|---:|---:|---:|
| jg1 §S1e "honest extrapolation" | ~11,400 | ~7,800 | **-0.0066** |
| jg1 §S2 first-pass scale-up (the charter's headline) | ~18,000 | ~11,600 | **-0.0104** |

The gap is 0.006526. **The honest one barely clears it; the headline clears it with room.**
Both rest on the same modelled constant. That is why S1 runs before anything else.

## STAGE LEDGER

| stage | what it settles | status |
|---|---|---|
| S1 | REAL `ΔB` for jg1's retained 3-pair edit set, through a real encoder on the to1 body | IN PROGRESS |
| S2 | n600 joint solve, seeded-random pair order, rate-aware acceptance | GATED on S1 |
| S3 | byte-close + identity control + determinism + seal | GATED on S2 |

**HONESTY RAIL (charter, binding).** `-0.0104 S` is a 3-pair extrapolation. Realized-vs-
projected is printed at every scale rung. A smaller honest win still seals and fires; an
honest refusal with the measured curve is a first-class landing.

STORES CONSULTED: `.omx/state/canonical_frontier_pointer.json` (re-read at arm start) ·
`.omx/research/ddm_jg1_joint_solve_20260819.md` (full) ·
`/Volumes/APDataStore/pact/ddm_jg1/JG1_RETENTION_MANIFEST.json` + all 12 retained files ·
memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` ·
memory `the_denominator_and_the_falsifier_can_both_be_vacuous_20260816` ·
memory `concavity_helps_when_you_pay_the_axis_upward_20260818`.

---

## S1 — THE REAL RATE

### S1a — the coordinate correction, measured before anything was encoded

Two things jg1 recorded needed fixing, and both are structural rather than cosmetic:

1. **The tail is not all coder output.** `read_residual_archive` (`runtime/residual_archive.py:478-494`)
   splits it: **109,792 B tail = 96 B compact fixed residual table + 109,696 B RC64
   stream.** A re-encoder that compares its output against the whole tail is 96 B off by
   construction. This module carries the 96 B prefix through untouched and byte-checks
   only the stream.
2. **The pointer body is `ddm_up3/candidate_runtime/`, not the to1 tree.** The to1 tree's
   `inflate.py:19` pins `ARCHIVE_SHA256 = "50e56145..."` and would refuse the pointer
   archive. MEASURED here: the two archives' `runtime/` and `cpr1/` trees are identical,
   and section-wise `hpac`, `semantic` and `tail` are **byte-identical** — **only the
   carrier differs** (`f59210d7...` vs `4ef50093...`). So jg1's tail work transfers, but
   the splice target is the pointer.

### S1b — THE FINDING THAT DECIDES THE METHOD: there is no per-token price

jg1's `+4.718 bits/token` is a **per-symbol marginal**: the cost of flipping one token
holding every other probability fixed. Reading the shipped decoder
(`decode_production_tokens`, `:600-649`) shows that quantity does not exist for this
coder. **Four independent feedback paths make a token's price depend on other tokens'
VALUES:**

| # | path | file:line | reach |
|---|---|---|---|
| 1 | `sparse.selected_logits(current, context, group)` — `current` is the partially-decoded frame, one-hot'd into the conv | `residual_archive.py:617`, `cpr1/hpac_integer_sparse.py:161-170` | the 189 later groups of the SAME frame |
| 2 | `context = model.prepare_frame_context(index, previous)` — `conv_past` + SPM on the previous decoded frame | `:603`, `cpr1/hpac_integer.py:330-362` | all of frame n+1 |
| 3 | `boundary = _boundary_buckets(previous_cpu)` — the fixed-table row index is a distance-to-class-edge map of the previous frame | `:605-608`, `:515-534`, `:621` | all of frame n+1 |
| 4 | `FreeCorrector` Krichevsky-Trofimov counters, updated only from decoded symbols and **never reset per frame** | `free_corrector.py:290-308`, `173-175` | **the entire remaining stream** |

Path 4 alone makes the blast radius of one changed token **global and unbounded** — one
edit in pair 283 perturbs the probability of every symbol through pair 599. There is no
local window to price, and `ddm_hm1`'s retained `base_logits_int16_n600.i16` — which
`ddm_rr2`'s encoder memmaps — becomes **wrong** the moment a token changes. `ddm_rr2`
states its own precondition at `:40-43`: its logits are valid *"because the decoded field
is unchanged by construction."* A jg2 token edit is exactly what breaks that.

**So the modelled 4.718 was never going to be checkable by a better model. It is
replaceable only by encoding the whole 600-frame stream along the new trajectory and
stat'ing `archive.zip`.** That is what `experiments/ddm_jg2_tail_reencode.py` does: it is
`decode_production_tokens` line for line with the decode call replaced by an encode of the
known symbol, importing the model, group plan, boundary map, fixed table, corrector and
probability quantization from the shipped runtime rather than reimplementing any of them.

### S1c — the encoder is the exact inverse (smoke, n=2 frames)

The RC64 encoder half is not in the shipped tree (`runtime/entropy/rc64_backend.c`, sha
`05839d14...`, is **decoder-only**). It comes from the `ddm_rr2` lineage source pinned at
sha `5c75e2c7...` plus `route_b_rc64.RC64_CHECKPOINT_EXTENSION`. That pairing is not
assumed to invert this decoder — it is **tested**:

| quantity | value |
|---|---:|
| frames encoded | 2 |
| emitted bytes | 555 |
| **prefix bytes agreeing with the shipped stream** | **554** |
| ideal code length from the probability rows | 554.78 B |
| wall clock | 2.88 s (1.44 s/frame -> ~14.4 min for n600) |

The single trailing byte is the coder's end-of-stream flush at frame 2, which the shipped
stream does not have there. **554 of 555 bytes agree**, and the emitted length sits 0.22 B
above the ideal code length — the coder tax, measured rather than assumed. The full
600-frame control (which must be byte-identical, not prefix-identical) is the binding
proof and is running.
