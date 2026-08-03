---
schema: ddm_cp1_composition_matrix.v1
date_utc: 2026-08-03
arm: ddm_cp1 (do the measured wins COMPOSE?)
lane_id: "lane_ddm_cp1_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: INSTANCE
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. Section-disjointness measured by
  SHA-256 over the actual ZIP members; byte deltas byte-exact through the REAL builder;
  d_pose realized n600 through the REAL receiver + frozen PoseNet with a canary.
  NO training, NO paid dispatch, NO exact gate fired, NO pointer mutation."
consumes:
  - .omx/research/ddm_pj2_pose_scale_degeneracy_20260802.md
  - .omx/research/ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802.md
  - .omx/research/ddm_ms8_menu_selector_solver_st_codebook_20260802.md
  - .omx/research/ddm_ix1_index_compaction_ladder_20260802.md
  - .omx/research/ddm_cr2_composition_row_ep854_base_20260801.md
  - .omx/research/ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md
  - .omx/research/ddm_mt1_menu_triage_79_20260802.md
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_{pw1,ms8,dc1_fold,pj2,cr2_ep854,gd3_coarse12x16_pooled,qa66_celldrop50}_archive.zip
produces:
  - tools/cp1_compose_pose_stack.py
  - src/tac/tests/test_ddm_cp1_compose_pose_stack.py (17 tests)
  - /Volumes/VertigoDataTier/pact/ddm_cp1_20260802/{cp1_PREDICTION_prereg.json,
    cp1_fold_receipt.json,final_cp1_pj2_fold.jsonl,cp1_score_shard*.jsonl,cp1_report.json}
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cp1_pj2_fold_archive.zip
    (360,339 B, built + decode-verified, MEASURED NEGATIVE — do NOT gate it)
consumers: [MAIN, ddm_pj2, ddm_dc1, ddm_ms8, ddm_ix1, ddm_gd3, "#850", "#873", "#882"]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_cp1 — four of the five "wins" are ONE archive member, and the fifth is the only one that composes

## §0 POINTER HONESTY, and the answer first

**The exact pointer is UNMOVED and I fired no gate.** Everything below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`.

The charter asked whether four (later five) measured wins add. **They do not add, and the reason is
structural rather than numeric.** One SHA-256 pass over the actual ZIP members settles most of it:

```
member                     pw1        ms8        dc1_fold   pj2        cp1_pj2_fold
state/tokens.dr7t          305a2be9   305a2be9   305a2be9   305a2be9   305a2be9    346,478 B
state/renderer.sec         71e776c3   71e776c3   71e776c3   71e776c3   71e776c3      3,341 B
state/selector.sec         8fc5c3fc   8fc5c3fc   8fc5c3fc   8fc5c3fc   8fc5c3fc        535 B
state/pose_stub.sec        01cfd661   01cfd661   01cfd661   01cfd661   01cfd661         83 B
state/pose_warp.stp        5fbd3d4c   9bdb88e8   b72a2909   b64d9cf1   18a9c95e   <-- ONLY THIS
manifest.json              (differs)  (differs)  (differs)  (differs)  (differs)   <-- and this
```

**`pw1`, `ms8`, `dc1_fold` and `pj2` are not four stack layers. They are four successive solutions of
ONE member** (`state/pose_warp.stp`, ~8.7 KB = 2.4% of the archive) plus its manifest. Their byte
deltas reconcile to the byte: ms8 `+53` on pose_warp `−2` on manifest `= +51`; dc1_fold `−66 / +1
= −65`; pj2 `+32 / 0 = +32`. **Any sum of their ΔS against a common base double-counts.**

**And the one composition the charter thought was safest is a MEASURED NET LOSS.** I built
`pj2 × dc1_fold` end to end — pj2's solved pose VALUES re-expressed at dc1_fold's incumbent `s_t`
REPRESENTATION — byte-closed at **360,339 B**, decode-verified, n600-scored:

| | archive B | seg | pose | rate | composed S |
|---|---:|---:|---:|---:|---:|
| `dc1_fold` — live own-vehicle base | 360,309 | 0.431179 | 0.2272826 | 0.2399150 | 0.8983766 |
| **`pj2` — best measured** | 360,406 | 0.431179 | **0.1597263** | 0.2399796 | **0.8308849** |
| `cp1_pj2_fold` — this composition | **360,339** | 0.431179 | 0.1598056 | 0.2399349 | 0.8309196 |

**The fold refunds `−4.461e-05` S of rate and spends `+7.931e-05` S of pose. Net `+3.469e-05` —
worse than `pj2` alone.** `pj2` stays the candidate; **do not gate `cp1_pj2_fold`.**

**The composable win is `ix1`,** and I verified its transfer on my own bytes rather than inferring it.

---

## §1 THE COMPOSITION MATRIX

Section-disjointness is MEASURED by SHA-256 over the actual ZIP members (§0), never inferred from a
memo. `manifest.tokens_sha256` is **not** used as a custody check anywhere here: `ddm_gd3` measured
it matches neither payload and is never read (counted-but-inert, #417) — I hash the members.

| pair | verdict | mechanism, MEASURED |
|---|---|---|
| **pj2 × ms8** | **SAME WIN — pj2 CONTAINS ms8** | pj2 started from ms8's archive and 31 of its 600 rows are literally `source: ms8_st_refit`. Counting both is a double-count of `−0.0491770`. |
| **pj2 × dc1_fold** | **COMPOSES-WITH-INTERACTION → NET LOSS (§2)** | same member, different coordinates (values vs `s_t` representation). Bytes compose to within **2 B** of additive; distortion composes exactly by algebra but **not** through the f16 storage lattice. |
| **dc1_fold × ms8** | **SAME WIN, opposite sign on rate** | dc1_fold's `−65 B` is the REFUND of ms8's `+51 B` (plus `−13` on `tp_member`, `−1` net manifest). Its ΔS-vs-`pw1` **IS** ms8's win re-realized, not a second one. |
| **pj2 × ix1** | **ORTHOGONAL** (one precondition, §3) | disjoint members; ix1 is lossless so the decoded lattice — hence every rendered frame — is bit-identical, so pj2's per-pair solve is not stale. Precondition: ix1's manifest rung may not migrate a **fitted** `st_grid`. |
| **dc1_fold × ix1** | **ORTHOGONAL, MEASURED on my bytes** | ix1's baseline IS dc1_fold. I decoded `tokens.dr7t` from both `dc1_fold` and `cp1_pj2_fold` → lattice `(600,24,32,4)` **bit-identical, sha `f6055f31`** — and re-raced ix1's ladder: `soa_axis0_minor / nibble_lane` = **339,970 B**, reproducing ix1's number exactly. |
| **ix1 × gd3** | **STRONGLY SUB-ADDITIVE (unmeasurable today)** | ix1's `−5,184 B` is 1.5% of a 346,478 B stream; ds=32's stream is 101,636 B. Naive scaling puts the transfer near `−1.5 KB`, not `−5.2 KB`. Moot in practice — see next row. |
| **anything × gd3** | **NOT A MEMBER TODAY** | `ddm_tr1_runtime._conv_shapes` sets `n_upsample = log2(ds)`, so ds=32 is a **7-conv decoder against the shipped 6**; `up4` has no trained weights in custody. No archive exists to compose. Its rate row (`−0.1722397`, 23.72% of gap) is real and **costs a training run**. |
| **anything × cr2** | **BLOCKED AT THE BASE (refuted)** | `ddm_cr2r`'s matched control: same solver, 74 matched pairs, celldrop50 `0.0778` vs ep854 `11.5904`, ep854 better on 1/74. **46× over break-even at the floor.** The defect is the base, not the solver. |
| **pj2 × cr2** | **CONFLICT** | cr2's `pose_warp` is byte-identical to `pw1`'s (`b64d9cf1`) — it is *behind* pj2 on the same member — and its tokens/renderer differ, so pj2's solve would be stale anyway. |
| **dc1_fold × cr2** | **CONFLICT, same reason** | cr2 carries `pw1`'s pose member and a different token stream. |

### §1.1 The staleness test, applied — and it already has a MEASURED price

MAIN's correction ("any token change makes the pose carrier stale, because `f0 := a·warp(f1) + b` was
solved against the *original decoded* frame_1") is right, and **its cost is not hypothetical on this
vehicle**. Applying the test to every row:

* **`dc1_fold` does NOT touch the tokens.** `state/tokens.dr7t` is byte-identical across the whole
  pose line (§0). *(This also corrects the charter's framing of `dc1_fold` as a "dead-codeword refit
  of the token codebook" — the token member is untouched; the fold is a pose-side move. `dc1`'s own
  §1 says so; the `−65 B` lands entirely on `pose_warp` + manifest.)* **No staleness.**
* **`ix1` re-encodes the token stream losslessly** and proves bit-identity of the decoded lattice from
  the frame alone. Same lattice + same `renderer.sec` bytes ⇒ same frames ⇒ **no staleness.**
  *(Scope: I did not re-verify ix1's frame decoder; I verified that the lattice ix1 operates on is
  byte-identical to the one my archive ships, which is what makes its number transfer.)*
* **`cr2` and `gd3` both move the tokens ⇒ staleness ⇒ MEASURED at 37.877** for the transplant, and
  **11.59 after a fresh full re-solve** (`cr2r`), against a break-even of `0.0132`.
  **The corollary matters more than the law:** cr2r's matched control proves the residual is a
  property of the BASE, not the solver — so a token-touching change owes a **training run with pose
  in the loop**, and a post-hoc re-solve does not buy it back.

---

## §2 THE BUILD — prediction stated first, then measured

**PREDICTION, recorded to `cp1_PREDICTION_prereg.json` before any scorer ran:**

| quantity | predicted | measured | miss |
|---|---:|---:|---:|
| archive bytes | 360,340 ± 5 | **360,339** | **1 B** |
| `d_pose` mean | 0.0025517105 | **0.0025537846** | +2.07e-06 |
| composed S | 0.8308554 | **0.8309196** | **+6.42e-05** |

The byte half landed to 1 B. **The distortion half missed by 26× the byte-close fidelity anchor
(1.8e-6–2.5e-6), and that miss is the finding**, so it is reported as one rather than smoothed over.

**Legs, in order:**

1. **ALGEBRAIC (free).** Folding `p'[0:3] = p[0:3]·(s_fitted/s_incumbent)` and decoding at the
   incumbent `s_t` leaves the homography invariant: **max relative difference 3.945e-16** over 600
   pairs, through the same `pose_to_homography` object the scorer path uses. (Independent agreement
   with `mq1` 5.98e-16, `dc1` 4.539e-16, `pj2` 7.38e-16.)
2. **BYTE (free, real encoder).** `tp_member` 6,401 → **6,388 B (−13)**; `st_coded` back to the
   incumbent **189 B**; `pose_warp` 8,752 → **8,686 B (−66)**. Built through
   `ddm_v4d_build_composed_archive.py --dim0-offset auto`, **no `--st-override`**, so the builder's
   `assert_inherited_st_grid_is_vendored` guard had to pass.
3. **MUTATION CONTROL.** The shipped `st_grid` is the **vendored ladder**, character for character,
   identical to `pw1`/`dc1_fold` and unlike `ms8`/`pj2`. The folded archive ships **no menu change at
   all** — it would decode on the pre-ms8 receiver.
4. **DECODE VERIFIED.** `ddm_v4d_verify_decode.py`: `A_ok` (parse-back consumption bijection #417),
   `B_pose_reconstruct_exact`, `B_ab_bit_exact`, `B_selector_exact`, `B_beta_exact`,
   `C_recompute_byte_exact` on 7 sampled pairs incl. a two-plane and a `beta≠0` pair. Archive sha
   `81ee7cd0fcc6e36289b244bd6f1f8b1ad9f11e9d9769c3e6357fcac3db507611`, **360,339 B**.
5. **CANARY.** Re-scoring the UNFOLDED solution on my harness reproduces pj2's reported per-pair
   values to **max abs err 3.313e-07**, against the measured instrument floor 1.2e-05. The report
   *aborts* above that floor — an instrument that cannot reproduce a known row may not publish a
   verdict from a new one.
6. **REALIZED n600.** All 600 pairs, quantized exactly as the builder stores them, through the
   receiver's own `f0` branch structure and the frozen PoseNet.

### §2.1 Why it loses — and the law that generalizes

```
d_pose mean  0.0025512505 -> 0.0025537846   (+2.534e-06)
345 / 600 pairs IDENTICAL · 188 WORSE · 67 BETTER · max |delta| 6.137e-04
```

The homography is invariant by algebra; **the f16 STORAGE lattice is not.** Rescaling `p[0:3]` by
`k` moves the stored values onto different float16 cells, and `p[0]` ships as a residual off a
re-derived manifest offset (32.1875 → 32.75), whose absolute quantum depends on residual magnitude.

**The asymmetry (188 worse vs 67 better) is the mechanism, not noise.** `pj2`'s solver uses
**realized-acceptance at the shipped quantization**: it proved local optimality *on the lattice it
was storing to*. Re-expressing the answer on a different lattice moves it off an optimum the solver
had certified. That is pj2's own §3 convergence proof read in reverse.

**The transferable law, MEASURED twice with opposite signs:**

| folded solution | its `d_pose` mean | fold's Δ`d_pose` | relative |
|---|---:|---:|---:|
| ms8's (`dc1`'s measurement) | 0.00516574 | **−4.6e-07** (a gain) | −0.009% |
| **pj2's (this unit)** | 0.00255125 | **+2.53e-06** (a cost) | **+0.099%** |

**A representation change costs a TIGHTER solve relatively more — here 11× more — and the sign is not
predictable.** It is f16 lattice noise whose *variance*, not mean, is what matters; `dc1` drew
favourably and I drew unfavourably. **Corollary for every future arm: a "free" re-representation of a
solved field is only free while the solution is loose. Price it against the solution you actually
intend to ship, never against the one it was measured on.** This is the one thing in this memo I
expect to be reused.

### §2.2 The cure, priced honestly — and why I did not spend it

Re-run pj2's solve with the **incumbent grid pinned**, so realized-acceptance is evaluated on the
lattice that actually ships in the folded configuration. Then the `−67 B` is free by the solver's own
monotone guard. **Ceiling: `−4.461e-05` S = 0.0061% of the gap** — the rate refund and nothing more.

I did not run it, and I recommend nobody does: **the entire `pj2 × dc1_fold` composition is worth at
most 0.006% of the gap even when executed perfectly, and as built it is negative.** The pose member
is 2.4% of the archive; composition inside it is CLOSED at the 1e-4 scale. That verdict is worth more
than the 4e-05 would have been.

---

## §3 THE BEST COMPOSABLE STACK

`ix1` is the only row that composes freely, because it is lossless and lands on a member that is
byte-identical across the whole pose line. Both variants, recomputed from components:

| stack | archive B | seg | pose | rate | S | ΔS vs base | % of gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dc1_fold` (live base) | 360,309 | 0.431179 | 0.2272826 | 0.2399150 | 0.8983766 | — | — |
| `pj2` | 360,406 | 0.431179 | 0.1597263 | 0.2399796 | **0.8308849** | −0.0674917 | 9.294% |
| **`pj2 × ix1`** (st_grid NOT migrated) | 354,331 | 0.431179 | 0.1597263 | 0.2359363 | **0.8268398** | **−0.0715368** | **9.850%** |
| `cp1_pj2_fold × ix1` (full migration) | 354,195 | 0.431179 | 0.1598056 | 0.2358457 | 0.8268286 | −0.0715481 | 9.852% |

The two are within **1.1e-05** of each other — the fold's byte advantage and its pose cost cancel
again. **`pj2 × ix1` is the recommendation**, because it needs no re-solve and carries no rule-118
question.

*(The `pj2 × ix1` row charges back **69 B** for retaining a fitted `st_grid` in the manifest — that is
the raw-JSON length of the field, DERIVED, not a re-measurement of ix1's stack. It moves the row by
4.6e-06 S and changes no verdict.)*

**`ix1` is MEASURED but NOT BUILT as an archive.** Its `IX1SOA02` frame needs receiver support that
`inflate_runner_v4d.py` does not have. The `−6,144 B` is a byte-exact ladder with a proved bit-identity
roundtrip, not a byte-closed archive; **the composed rows above are therefore DERIVED, and only the
`pj2` and `cp1_pj2_fold` rows are byte-closed.** Building the receiver is the highest-value next unit
on this axis by an order of magnitude (0.563% of gap vs 0.006%).

### §3.1 A rule-118 flag on ix1's manifest rung — the fold is what makes it legal

`ix1` migrates `manifest.st_grid` into `inflate.py` as "GENERIC — quantization GEOMETRY, not
assignment." **That is true only when the grid is the vendored constant.** MEASURED:

```
pw1  / dc1_fold / cp1_pj2_fold : [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
                                  == ddm_pfs1_ep_warp_pose_solve.ST_GRID  (vendored)  => GENERIC
ms8  / pj2                     : [0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16]
                                  fitted by ms8's solver ON THIS CLIP              => VIDEO-DERIVED
```

Migrating a **fitted** grid into `inflate.py` is the hide-data-in-code fake (rule 118, NO-FAKE #6/#7).
It is **69 B** — trivial as rate, not trivial as compliance. So on `pj2` the manifest rung must retain
`st_grid` (the row above does), and **`dc1`'s fold is exactly what converts that field back to
generic.** That is a real, non-numeric way in which the two compose.

**A second flag, OWED not claimed:** `manifest.rs_beta_mags` (**75 B**) is likewise migrated by ix1's
rung, but `ddm_v4d_build_composed_archive.derive_beta_table` builds it *from the magnitudes the solve
actually chose on this clip* — video-derived by construction, on **every** v4d-line archive including
`dc1_fold`. I did not re-measure ix1's stack; I read its table and the builder's derivation. Routed to
`ddm_ix1`/`ddm_ix2`: the manifest rung's `−840 B` may owe back ~75 B.

---

## §4 WHAT I DID NOT DO / OWED

* **No exact gate fired.** MAIN owns the slot. **My recommendation is that it goes to `pj2`
  (already staged, `bash experiments/stage_v4d_realized_gate.sh cpu pj2`) and NOT to my archive** —
  `cp1_pj2_fold` is a measured negative and would waste the slot. The two differ only in the pose
  member and the byte count, both measured here on the same instrument with a 3.3e-07 canary, so the
  ranking does not need a gate to be trusted.
* **d_seg was not re-measured, and does not need to be.** `state/tokens.dr7t` and `state/renderer.sec`
  are byte-identical across every archive in §0, so `d_seg` is inherited by construction, not by
  assumption — verified by SHA-256, not by a manifest field.
* **I did not re-verify ix1's `IX1SOA02` frame decoder.** I verified that the lattice it operates on
  is byte-identical to mine and that its layout race reproduces on my bytes (339,970 B).
* **The `rs_beta_mags` rule-118 question is FLAGGED, not measured.**
* **Single instrument, no restart census.** The per-pair objective is deterministic (canary exact);
  the *fold* has no search, so there is no start-bias here — but the §2.2 cure would inherit pj2's
  un-censused restart question (`sv1` §2b, `uv1` §4).
* No training, no paid dispatch, no `upstream/` edit, pointer untouched.

## §5 FALSIFIERS

1. A gate on `pj2` returning an S outside `0.8308849 ± 1e-4` ⇒ the fidelity anchor is broken and
   every advisory row on this line reopens, mine included.
2. A re-solve with the incumbent grid pinned (§2.2) that buys **more** than `−4.461e-05` ⇒ the fold's
   cost was not pure lattice re-rounding and §2.1's law is wrong.
3. `ix1`'s `−6,144 B` failing to reproduce on a `pj2`-based archive ⇒ the token member's byte identity
   is not sufficient for the transfer and §1's orthogonality verdict is wrong.
4. A third fold measurement landing at |Δ`d_pose`| ≈ 4.6e-07 on a solution tighter than ms8's ⇒ §2.1's
   "tighter is more fragile" law is a two-point coincidence, not a law.

## §6 NEXT-IF-RESUMED

1. **Build the `IX1SOA02` receiver path in `inflate_runner_v4d.py`** and byte-close `pj2 × ix1`.
   **−0.0715 S / 9.85% of the gap, scorer-free, no re-solve.** By far the largest composable item.
2. **Settle the `rs_beta_mags` rule-118 question** (§3.1) before ix1's manifest rung is claimed.
3. **Do NOT spend on `pj2 × dc1_fold`** (§2.2, ceiling 0.006% of gap).
4. **`gd3` and `cr2` both need a training run with pose in the loop** (§1.1), not a solver.
