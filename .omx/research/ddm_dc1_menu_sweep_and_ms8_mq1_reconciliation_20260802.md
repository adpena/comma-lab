---
schema: ddm_dc1_menu_sweep.v1
date_utc: 2026-08-02
arm: ddm_dc1 (sweep every discrete menu for the ms8 defect x audit ddm_mq1 as a negative)
lane_id: "lane_ddm_dc1_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: FORMULATION
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. n600 sweep + byte-close through the
  REAL builder and the REAL receiver. NO training, NO paid dispatch, NO exact gate fired,
  NO pointer mutation."
consumes:
  - .omx/research/ddm_ms8_menu_selector_solver_st_codebook_20260802.md   (the finding generalized)
  - .omx/research/ddm_mq1_pose_menu_rd_audit_20260801.md                 (the negative audited)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md               (the clipping discriminator)
  - .omx/research/ddm_lg2_binary_inventory_20260802.md
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/final_pw1.jsonl
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip
  - /Volumes/VertigoDataTier/pact/ddm_ms8_20260802/ms8_st_override.json
  - /Volumes/VertigoDataTier/pact/ddm_v4c_20260730/solve_celldrop50.partial.jsonl
produces:
  - tools/dc1_menu_sweep.py
  - src/tac/tests/test_ddm_dc1_menu_sweep.py (29 tests)
  - /Volumes/VertigoDataTier/pact/ddm_dc1_20260802/{dc1_inventory_receipt.json,
    dc1_selcurves_shard*.jsonl,dc1_seldesign_receipt.json,dc1_degen_shard*.jsonl,
    dc1_degen_meta.json,final_dc1_fold.jsonl}
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_dc1_fold_archive.zip
    (360,309 B, staged, NOT gated)
consumers: [MAIN, ddm_ms8, ddm_mq1, ddm_pj2, ddm_bs3, "#873", "#882"]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_dc1 — the ms8 menu was never a quantizer, and its whole win is reachable at the INCUMBENT menu for −14 bytes

## §0 POINTER HONESTY, and the headline

**The exact contest pointer is UNMOVED and no gate was fired.** Everything below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`.

My charter was: generalize ms8's dead-codeword discriminator across the shipped stack. **The
generalization does not follow, and the measurement that shows why is a strictly better version of
ms8's own result.**

| own-vehicle candidate | archive B | seg | pose | rate | composed S |
|---|---:|---:|---:|---:|---:|
| pw1 — live own-vehicle frontier | 360,323 | 0.431179 | 0.276504 | 0.239924 | **0.9476070** |
| ms8 — fitted `s_t` codebook (+51 B) | 360,374 | 0.431179 | 0.227293 | 0.239958 | 0.8984300 |
| **dc1 — same geometry, INCUMBENT menu, folded into the pose** | **360,309** | 0.431179 | 0.227283 | 0.239915 | **0.8983766** |

**ΔS vs pw1 = −0.0492304 for −14 BYTES** (ms8 paid +51 B for −0.0491770). Against the gap to the bar
(PR130 0.172141, gap 0.7754660) that is **6.35% of the gap, bought at negative rate cost.**

The composed S is a **PREDICTION**; its fidelity anchor on this vehicle is the QA78 v4d gate residual
(1.8e-6) and the pw1 gate residual (2.5e-6). I did not self-fire the gate. The archive is staged:
`bash experiments/stage_v4d_realized_gate.sh cpu dc1_fold`.

**`state/tokens.dr7t` is byte-identical to pw1's**, so d_seg is inherited unchanged and this composes
with the seg line rather than competing with it.

---

## §1 THE RECONCILIATION — mq1 and ms8 are BOTH right; ms8 mislabelled its own mechanism

MAIN chartered me to audit mq1 as a negative under three failure modes. **Only one of the three
applies, and it is not the one MAIN hypothesised.** Verdict on each, in MAIN's order:

| MAIN's hypothesis | verdict |
|---|---|
| (a) NAIVE/TOY SCOPE — mq1's n=48 vs n600 | **NOT the explanation.** mq1's own §0 table is explicitly scoped and its canary is exactly 0.0. I did not need to re-run ms8 on mq1's 48 because (c) resolves it structurally. |
| (b) BINARY INTERPRETATION — "degenerate"/"never binding" | **PARTIALLY.** mq1's headline "format loses to search by 33×" is a ratio, not a binary, and it is *correct*. What is binary is the implicit map from "menu" to "format", which is what mis-files ms8. |
| (c) WRONG PROJECTION — mq1 priced RATE, ms8's axis was DISTORTION | **MAIN'S HYPOTHESIS IS WRONG.** mq1 did not confuse the axes; §4 explicitly decomposes distortion into `gap_lattice` (what a finer LATTICE buys) and `gap_search` (what a better SEARCH buys). mq1 measured distortion on both. |

**The actual resolution, MEASURED.** `pfs1_warp_receiver.py:45-46` is

```
t = s_t * [p2, p1, p0]        R = expmap(s_r * [p3, p4, p5])
```

so **`s_t` and the stored translation triple are exactly multiplicatively degenerate.** Scaling
`(p0,p1,p2)` by `k` and `s_t` by `1/k` leaves the homography invariant. I re-derived this on ms8's
OWN per-pair scale factors through the same receiver object the scorer path uses:
**max relative homography difference 4.539e-16 over all 600 pairs** (mq1 §2 measured 5.98e-16 on
arbitrary factors; independent agreement).

Therefore **`gap_lattice` for `s_t` is IDENTICALLY ZERO**: the `s_t` menu imposes no reachability
limit at all, because a continuous coordinate already ships in the same direction. Every point the
menu can express is reachable without it. **ms8's entire −0.049177 is `gap_search` in mq1's own
decomposition** — a better search over the effective translation, not a better code for it.

So mq1's ceiling ("every STORAGE-FORMAT lever ≤ 0.056% of the gap") and ms8's −0.049177 are **not
in the same bucket and never contradicted each other.** ms8's "PLACEMENT vs SELECTION" split is
real as arithmetic — the two arms differ by 5.7× — but both halves are search: SELECTION is a
better index at today's reachable set, PLACEMENT is a *larger reachable set*. Neither is a
quantizer property. **The contradiction was a labelling error in ms8, not a measurement error in
either arm.** mq1's `verdict_scope: FORMULATION` on the codebook framing stands, and it should have
been enough to stop the generalization a day before I was chartered.

### §1.1 The decisive consequence, byte-closed

If the win is search reach and `s_t` is exactly degenerate, then ms8's gain must be reachable at the
**incumbent** menu by folding its chosen scale into the pose that already ships. Tested:

```
FOLD:  p'[0:3] = p[0:3] * (s_ms8 / s_shipped)     s_t index UNCHANGED (incumbent, 189 B, verbatim)
```

| arm | mean d_pose | ΔS_pose | archive B |
|---|---:|---:|---:|
| CTRL shipped (pw1) | 0.00764543 | 0.000000 | 360,323 |
| MS8 fitted codebook | 0.00516620 | −0.049211 | 360,374 (**+51**) |
| **FOLD at incumbent menu** | **0.00516574** | **−0.049221** | **360,309 (−14)** |

**Recovery: 100.02% of ms8's ΔS_pose.**

**HONEST QUALIFIER (round-2 self-review).** The equality is a POPULATION-MEAN equality, not a
per-pair identity: 127 pairs are worse than ms8 and 93 better, max |Δd_pose| 1.87e-03 — real f16
storage-rounding differences, three orders above the 1.085e-05 instrument floor, not noise. The mean
difference is 4.6e-07 (0.009%). **Do not read the fold as "better distortion than ms8"; read it as
"the same geometry, for 65 fewer bytes."** The byte win is the finding; the distortion is a tie.

The fold also *needs no receiver change*. ms8 had to make the receiver read `manifest['st_grid']`;
the folded archive's receiver reads the **identical incumbent table** (MEASURED below), so this
candidate would decode on the pre-ms8 receiver too.

---

## §2 BYTE-CLOSE + MUTATION CONTROL

Built through the REAL builder
(`ddm_v4d_build_composed_archive.py --final-jsonl <folded> --dim0-offset auto`, **no `--st-override`**):

```
archive 360,309 B   sha256 9fb9f4e9e460f91c1ea87bd7cbafd8da48cf6a8d5938e9859029a78b6290d3cb
tp_member  6,378 ->  6,365 B (-13)   st_coded 189 B (UNCHANGED, copied verbatim from the base)
manifest carries the VENDORED ladder;  st_grid override: null;  sel/beta/tokens/renderer untouched
```

`ddm_v4d_verify_decode.py`: **all checks ok** — `A_ok`, `B_pose_reconstruct_exact`, `B_ab_bit_exact`,
`B_selector_exact`, `B_beta_exact`, `C_recompute_byte_exact` on 7 sampled pairs including a two-plane
and a beta≠0 pair.

**MUTATION CONTROL** — instantiating the real `Decoder` on both archives:

```
[pw1     ] receiver st_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
[dc1_fold] receiver st_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
```

**IDENTICAL — the folded archive ships no menu change at all**, and re-scoring end to end through the
receiver's own `f0` + the frozen PoseNet reproduces the predicted per-pair d_pose exactly:

| pair | k | s_t pw1 | s_t fold | d_pose pw1 | d_pose fold | predicted |
|---:|---:|---:|---:|---:|---:|---:|
| 547 | 2.000 | 0.08 | 0.08 | 0.002185 | 0.000859 | 0.000859 |
| 416 | 2.000 | 0.06 | 0.06 | 0.017415 | 0.005192 | 0.005192 |
| 460 | 2.000 | 0.08 | 0.08 | 0.001257 | 0.000266 | 0.000266 |
| 366 | 1.750 | 0.08 | 0.08 | 0.000682 | 0.000327 | 0.000327 |
| 38 | 1.750 | 0.08 | 0.08 | 0.149965 | 0.003587 | 0.003587 |
| 0 / 2 / 599 (k=1) | 1.000 | 0.08 | 0.08 | 0.000788 / 0.000190 / 0.000259 | **identical** | — |

85/600 pairs carry k≠1 (range 0.375–2.000). Pairs ms8 did not move fold to a bit-identical pose.

---

## §3 THE INVENTORY, WITH ITS DENOMINATOR — and the denominator is where I was weakest

`tools/dc1_menu_sweep.py --mode inventory`, occupancy on the REAL n600 solution:

| menu | K | occupancy | **dead** | dead % | mode % | verdict |
|---|---:|---|---:|---:|---:|---|
| `st_grid` (`s_t` scale) | 11 | `[0,0,0,0,0,0,22,364,156,58,0]` | **7** | **63.6%** | 60.7% | the ms8 row — but see §1: NOT a quantizer |
| `selector` (1-plane vs 2-plane) | 2 | `[376, 224]` | 0 | 0.0% | 62.7% | no placement DOF; SELECTION measured in §4 |
| `rs_beta_mags` | 13 | `[5,5,1,10,15,420,66,52,13,1,7,1,4]` | 0 | 0.0% | 70.0% | **0 dead BY CONSTRUCTION** (`derive_beta_table` = `sorted(set(chosen))`) |
| `token_quant_levels` (96.2% of the archive) | 16 | `[556003,51054,…,10222,57708]` | 0 | 0.0% | 30.2% | 0 dead; **but see §5** |
| r7 SMEVR temporal mode-base | 16 | `[…]` over 3072 cells | 0 | 0.0% | 33.8% | adaptive/derived; no defect |
| `AUTO_CODECS` · `DEFLATE_MEMBERS` · 9 pinned `selector.sec` fields | 2·2·9 | — | — | — | — | UNREACHED, each with a stated reason |

**`st_grid` is the ONLY menu over the 10% threshold, and ms8 already refit it.** My occupancy
reproduces ms8's exactly (`[0,0,0,0,0,0,22,364,156,58,0]`, 7 dead) from a *different source* — I
decoded the shipped index stream out of the archive; ms8 read the d1 solve JSONL. The shipped bytes
and the solver's record are the same story.

**MY DENOMINATOR WAS TOO SMALL, and I am reporting that as a defect of this arm.** An independent
repo-wide sweep of the receiver + builder import closure found **36 candidate menus, 19 of them
reaching the archive or receiver**; my tool enumerates **8 (5 occupancy-measured, 3 unreached)**.
The gap is 19 vs 8 on the archive-reaching set. The 11 I did not enumerate are architectural,
single-value, seed-regenerated, or adaptive-context menus (`SECTION_CONTRACT`, `MEMBER_ORDER`, the
lotto bank alphabet, `code_width`, `grid_downsample`, `token_ste`, `token_temporal_mode`,
`token_codec`, `pose_stub`, and the SMEVR adaptive context tables) — for each of those a *per-pair
occupancy* is undefined, so no dead-codeword verdict is possible; but **"undefined for my
discriminator" is not "swept".** The receipt now records both numbers so the smaller one is never
mistaken for the population. Three findings from that sweep worth carrying (not mine, recorded so
they are not re-derived):

1. `manifest["beta_idx_counts"]` (13 ints) and `manifest["selector_num_two"]` are written by the
   builder and **read by no receiver** — the #417 counted-but-inert class, same as the `st_grid`
   field ms8 fixed. The builder's own comment says `st_idx_counts` was deliberately kept out for
   exactly this reason; `beta_idx_counts` was left in.
2. `ST_GRID` is hand-duplicated at **9 repo sites** plus 4 volume copies of `pfs1_warp_receiver.py`,
   and **two tool copies are 10-entry** (they drop the leading `0.0`). `#907` owns this; the fold
   landed here reduces the shipped path's exposure to none of the copies (it changes no table).
3. `pfs1_warp_receiver.py` **is not in the git repo** — it exists only on the external volume.

---

## §4 THE SELECTOR — the follow-on ms8 named as owed, MEASURED

ms8 left `selector ∈ {0,1}` unmeasured ("~1 line of the same harness"). Measured here at n600,
`--mode selcurves` + `--mode seldesign`, one f1 render per pair reused across 8 configurations
(2 `s_t` variants × 2 selectors × 2 pose sources).

**POSITIVE CONTROL (mandatory, ABORTS on failure): canary max |d_ctrl − d_shipped| = 1.085e-05,
578/600 EXACT** — reproducing ms8's canary to the digit from an independently written harness. The
`MS8_only` arm reproduces ms8's headline `d_pose 0.00516620` and `ΔS_total −0.049177` exactly, which
is an independent replication of ms8 by a second arm.

| arm | d_pose | ΔS_pose | pairs moved | ΔB | ΔS_total |
|---|---:|---:|---:|---:|---:|
| RESEL_sel_on_ms8_st | 0.00511590 | −0.050320 | 5 | +51 | **−0.050286** |
| BRANCHPOSE_flip_st_ms8 | 0.00516394 | −0.049261 | 3 | +45 | −0.049231 |
| MS8_only (replication) | 0.00516620 | −0.049211 | 0 | +51 | −0.049177 |
| **RESEL_sel_at_shipped_st** | 0.00760412 | −0.000748 | **6** | **0** | **−0.000748** |
| BRANCHPOSE_flip_st_ship | 0.00764317 | −0.000041 | 3 | −6 | −0.000045 |
| CTRL_shipped | 0.00764543 | 0.000000 | 0 | 0 | 0.000000 |

**The selector re-selection is REAL, FREE, and SMALL: ΔS −0.000748 at ZERO bytes, 6/600 pairs**
(the brotli-packbits stream happens to cost the same). On the ms8/fold geometry it grows to
−0.001109 over `MS8_only`, so it **composes** rather than being absorbed. Against the gap that is
**0.096%** — 300× the fidelity anchor, and tidy-up scale, not a lever.

**BIAS DISCLOSED:** the pose was GN-solved for the shipped branch, so a flip is evaluated with a
mismatched pose — an adverse bias, making −0.000748 a floor. The `BRANCHPOSE` arms address it by
adopting each branch's own v4c GN pose (available for 250/600 pairs) and **LOSE** to the plain flip
(3 pairs adopt): the v4c branch poses are un-refined by pw1, so the counterfactual is worse than the
mismatch it fixes. `verdict_scope: FORMULATION` on that arm — a better counterfactual would re-run
pw1's refinement on the flipped branch, which is search and belongs to ddm_pj2.

---

## §5 THE 96.2%-OF-BYTES MENU — 0 dead codewords, but a MEASURED boundary discontinuity

`token_quant_levels` is a **hardcoded generic uniform lattice**, `v = 2·code/15 − 1`
(`ddm_tr1_runtime.py:1224-1227`). No table ships, so it has never been fitted. Measured occupancy
over the 1,843,200 shipped codes:

```
code    0      1      2      3      4      5      6      7      8      9     10     11    12    13    14     15
 %   30.17   2.77   4.84   7.21   7.69   8.01   7.98   7.18   5.58   4.81   3.77   2.94  2.02  1.35  0.55   3.13
```

**Dead-codeword fraction 0/16 — this menu PASSES my discriminator.** It fails the *other* one:
`code 0` is **10.9×** `code 1` and `code 15` is **5.65×** `code 14`, while the interior is smooth.
That is the clipping-fold signature of `np.clip(combined, -1, 1)` in the encoder, and it confirms
bs2's 5.65× at the top end and adds a much larger one at the bottom. It is exactly pw1's
discriminator firing where mine does not — the two are orthogonal, as ms8 §10 said.

**I am NOT claiming this is a defect.** The lattice was trained in the loop through an STE, so mass
at the bound is partly deliberate; and the one prior instance where a "bound pile" was read as a
defect (`s_t`) turned out to be a search-reach artifact. Naming the rung with its blocker instead:
a Lloyd–Max-style non-uniform dequant table (16 floats, ~100 counted B, one receiver line) is
**derivable only from the PRE-quantization activations**, which need the trained checkpoint, and its
objective is d_seg, which needs a scorer job. Out of this arm's authority (no heavy launch).

**Also measured, $0, and new:** **1,544 of the 3,072 token grid cells (50.3%) are CONSTANT across
all 600 pairs**, 752 of them at `code 0`. Their per-pair residual is always zero, so the SMEVR mode
base carries them exactly — this costs almost nothing in the *current* coder and is reported as a
property of the trained token field, not as a lever.

---

## §6 FALSIFIER VERDICT (pre-registered before measuring)

> Family CLOSED if: every remaining menu has dead-codeword fraction < 10% AND no refit at matched
> bytes wins > 3× the vehicle's fidelity anchor (1.8e-6 / 2.5e-6) across ≥ 3 attempts.

* **Clause 1 — SATISFIED.** Every menu other than `st_grid` has dead fraction **exactly 0%**
  (selector 0/2, beta 0/13 by construction, tokens 0/16, SMEVR mode-base 0/16).
* **Clause 2 — FAILED.** The selector re-selection wins −0.000748 = **299×** the 2.5e-6 anchor at
  matched (zero) bytes, on the first attempt.

**Verdict: the family is NOT CLOSED, but it is NARROW** — the only remaining measured win is a
0.096%-of-gap zero-byte *selection* move, and no *placement* refit is available anywhere because no
other menu has a dead codeword.

**And I am failing my own falsifier's clause 2 as a pre-registration.** Setting the threshold at 3×
the instrument floor made it nearly unfalsifiable — almost any real effect clears 7.5e-6. The
denominator that matters is the gap (0.7754660), against which −0.000748 is tidy-up. I record this
as a defect in my own pre-registration, not a result.

---

## §7 THE DISCRIMINATOR, CORRECTED AGAIN — and my charter's premise NARROWED

ms8 corrected pw1 (mode share is not the discriminator; deadness is). **This unit corrects ms8: on
this vehicle, deadness was not the discriminator either — degeneracy was.**

> A menu whose value is **exactly degenerate** with a continuous coordinate that already ships is a
> **search-reach parameter, not a quantizer.** Its "dead codewords" are unreachable search points,
> not wasted cells; its `gap_lattice` is identically zero; and its win is always available at zero
> menu cost by folding the choice into the continuous coordinate. Test degeneracy BEFORE fitting a
> codebook — it is a 10-line algebraic check on the receiver, and it is free.

`s_t` is the **unique** exactly-degenerate menu on this vehicle, checked at source
(`pfs1_warp_receiver.py:45-46`):

* `s_t` multiplies the translation triple → **degenerate**, and it is the one menu that refit.
* `beta` multiplies the rotation triple, but the receiver applies **two** scales (`1 ± β/2`) blended
  by row, so no single rescale of `p[3:6]` reproduces it → **NOT degenerate**, genuine row-gradient DOF.
* `selector` is a structural branch; `token_quant_levels` codes the content itself → **NOT degenerate**.

That is why exactly one menu was refit-able, and why the sweep found nothing else. **My charter's
premise — "a proven method, applied once, with the rest of the stack unswept" — is narrowed: the
method was not a menu method. It was a degeneracy method with a sample size of one.**

This is plausibly the **third instance today of one genus** (dt1's loss scalar, ms8's mode-share,
this) — a real effect attributed to the wrong mechanism, where the wrong attribution then licenses a
generalization that does not hold. **ddm_bs3 owns that genus; this row is offered to it.** MAIN's
hypothesis (c) is a fourth candidate instance but is itself an example of the genus — the
misattribution was in ms8, not in mq1.

---

## §8 WHAT I DID NOT DO / OWED

* **No exact gate.** The composed S is a prediction. MAIN fires
  `bash experiments/stage_v4d_realized_gate.sh cpu dc1_fold`.
* **The FOLD + selector re-selection combination is NOT byte-closed.** Arithmetic predicts
  d_pose ≈ 0.005116 at ~360,309 B ⇒ S ≈ 0.898. **This is a prediction from two separately measured
  arms, not a measurement** — it needs one builder run and an n600 re-score. Named, not claimed.
* **My denominator is 8 against a repo-wide 36/19** (§3). The 11 archive-reaching menus I did not
  enumerate need a discriminator that is not per-pair occupancy.
* **mq1's n=48 vs ms8's n600 scope question is unresolved by direct test.** I resolved the
  contradiction structurally instead, which is stronger, but MAIN's (a) was never falsified — only
  made unnecessary. If §1's algebra were wrong, (a) reopens.
* **`token_quant_levels`** stays where bs2 and ms8 left it, with the blocker now named precisely:
  pre-quantization activations (checkpoint) + a d_seg scorer job.
* **The `beta_idx_counts` / `selector_num_two` inert manifest fields** (§3 finding 1) are unfixed.
* No training, no paid dispatch, no `upstream/` edit, pointer untouched.

## §9 FALSIFIERS FOR THIS UNIT

1. The exact gate on `v4d_composed_dc1_fold_archive.zip` returns a composed S outside
   `0.8983766 ± 1e-4` ⇒ the byte-close fidelity anchor on this vehicle is broken and every advisory
   pose row on this line reopens, mine and ms8's alike.
2. A menu is found on this vehicle with a nonzero dead-codeword fraction that is **not** exactly
   degenerate with a shipped continuous coordinate, and refitting it wins ⇒ §7's correction is too
   strong and deadness recovers independent content.
3. The fold's advantage is reversed by the gate (i.e. the realized d_pose of the folded archive is
   materially worse than ms8's) ⇒ f16 storage rounding on the rescaled columns matters more than the
   4.6e-07 mean difference suggests, and mq1 §2's "deleting `s_t` loses" extends further toward
   ms8's scale factors than measured here.
