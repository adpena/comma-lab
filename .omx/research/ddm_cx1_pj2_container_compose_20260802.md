---
schema: ddm_cx1_container_composition.v1
date_utc: 2026-08-03
arm: ddm_cx1 (compose pj2 x the ix2 single-member container, byte-close, gate once)
lane_id: "lane_ddm_cx1_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false  # exact contest pointer 0.19108 UNMOVED; this is the own-vehicle line
verdict_scope: INSTANCE
axis: "[macOS-CPU advisory - real evaluator, real bytes] NON-PROMOTABLE. Container is
  EXACTLY lossless; frame parity measured n600 through the REAL receiver; S recomputed
  from components, never from the rounded report field. NO training, NO paid dispatch,
  NO pointer mutation."
consumes:
  - .omx/research/ddm_ix2_renderer_split_and_decoder_20260802.md
  - .omx/research/ddm_cp1_composition_matrix_20260802.md
  - .omx/research/ddm_pj2_pose_scale_degeneracy_20260802.md
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pj2_archive.zip
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_pj2/report.txt
produces:
  - tools/cx1_build_ix2_container_archive.py
  - tools/cx1_verify_frame_parity.py
  - src/tac/tests/test_ddm_cx1_container_compose.py
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cx1_pj2ix2_archive.zip
    (353,808 B, sha 1d3ab694c337f3f7374fa42034664b0494d0dfda1be479b1d367e964da78701f, GATED)
  - /Volumes/VertigoDataTier/pact/ddm_cx1_20260803/{cx1_PREDICTION_prereg.json,
    cx1_build_receipt.json,cx1_frame_parity_n600.json,cx1_gate_receipt.json}
  - eval_root/submissions/v4d_cx1_pj2ix2/report.txt (the n600 gate)
consumers: [MAIN, ddm_ix2, ddm_cp1, ddm_pj2]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_cx1 — the container composes onto `pj2` exactly, and the trap was one level sharper than "count it"

## §0 THE ANSWER FIRST — MEASURED, n600, one gate

**`pj2 × ix2-container` is BUILT, BYTE-CLOSED, n600 FRAME-PARITY VERIFIED, and GATED.**

| | archive B | seg | pose | rate | S |
|---|---:|---:|---:|---:|---:|
| `pj2` — the previous best, MEASURED | 360,406 | 0.4311790 | 0.1597320 | 0.2399796 | **0.8308905** |
| **`cx1` = `pj2 × ix2`, MEASURED** | **353,808** | 0.4311790 | 0.1597320 | 0.2355862 | **0.8264972** |

**ΔS −0.0043933 for −6,598 B — 0.667 % of the gap from `pj2` to the PR130 bar
(0.605 % of the original 0.7262358 gap). New gap to bar: 0.6543562.**

**The gate returned `d_seg = 0.00431179` and `d_pose = 0.00255143` — identical to `pj2`'s
to every printed digit**, exactly as the losslessness argument requires. Recomputed from
components: **S = 0.8264972 against a pre-registered prediction of 0.8264972, miss
1.25e-08.**

**Both `report.txt` files print `Final score: … = 0.83`.** The rounded field cannot tell the
two archives apart at all; it rounds away this arm's entire result. Every number above is
recomputed from `Average SegNet Distortion` / `Average PoseNet Distortion` /
`Submission file size`, and the `pj2` row is re-derived from
`eval_root/submissions/v4d_pj2/report.txt` rather than copied from a memo.

**This was a CONFIRMATION, not a search** — the container is exactly lossless, so `d_seg`
and `d_pose` are invariant by construction and only the rate term could move. The prediction
was written to `cx1_PREDICTION_prereg.json` before any scorer ran.

**And the strongest single check is the gates' own artifacts:** the two runs' inflated
outputs — the exact pixels the frozen scorer consumed — are **byte-identical**,
`sha256 988785e7cadfd6137d918b53d020fe5f34e5735765433a4873241cb15f7e200e` on 3,492.7 MB
each. Not a re-implementation of the decode: the contest inflate path's own output.

**This also beats `ddm_cp1`'s DERIVED `pj2 × ix1` row (354,331 B / S 0.8268398) by 523 B and
−0.0003426 S**, and it is byte-closed and gated where that row was derived.

**Two things I want on the record above the byte count:**

1. **The rule-118 trap on this base is one level sharper than `ddm_ix2` named it.** It is
   not merely "`st_grid` is FITTED on `pj2` so COUNT it instead of migrating it." When I
   counted it, `pack_config_section` **refused outright**: `pj2`'s fitted ladder
   (`0.06, 0.065, 0.07, …`) is **not representable in float16 at all**. Counting a fitted
   table is only honest if it is counted **EXACTLY** — a silently rounded `s_t` moves the
   homography, moves the rendered frames, and forfeits the very invariance exemption this
   whole line rests on. §2.
2. **The staging script would have run the WRONG RECEIVER.** `stage_v4d_realized_gate.sh`
   copied the receiver from the SSD, and that copy was byte-identical to the **pre-ix2**
   repo file — no container path at all. A container archive staged through it would have
   taken the legacy 6-member branch and died looking for a `manifest.json` that no longer
   exists. §5.

---

## §1 THE BUILD

`ddm_ix2` landed the container primitives and the receiver but no encoder that carries a
live 6-member archive into container form. That encoder is
`tools/cx1_build_ix2_container_archive.py`.

| | legacy `pj2` | container `cx1` |
|---|---:|---:|
| `manifest.json` | 1,451 (752 deflated) | **deleted** — replaced by a 36 B counted config section |
| `state/tokens.dr7t` | 346,478 | 341,295 (`IX2TOK01`, cell-major nibble) |
| `state/renderer.sec` | 3,341 | 3,266 (`IX2REN01`, field-split) |
| `state/selector.sec` | 535 | 535 (into the shared small-group coder) |
| `state/pose_stub.sec` | 83 | **deleted** — a constant string, migrated |
| `state/pose_warp.stp` | 8,752 | 8,752 **byte for byte** |
| ZIP framing | 686 (6 members, filenames twice) | **108** (one member, `0.bin`) |
| **total** | **360,406** | **353,808** |

`tokens` is kept OUT of the shared coder stream: at 96 % of the archive and already at its
coder's entropy it would only dilute the model. That is `ddm_ix2`'s two-tier split, and it
is why the module has a `stored` rung at all.

**The pose member is copied byte for byte.** That is deliberate and it is the entire
safety argument: `ddm_cp1` measured that a re-representation costs a *tighter* solve
relatively more (`ms8` −0.009 % = a gain, `pj2` +0.099 % = a cost, **11× worse**), because
the f16 storage lattice is not invariant even where the algebra is. `pj2` is the tightest
solve we have. This composition is exempt **only** because nothing in it re-quantizes; the
moment a rung re-quantizes rather than re-frames, cp1's law lands hardest on exactly this
base. Stated in the tool's module docstring so the next caller cannot miss it.

---

## §2 THE TRAP — `st_grid` on `pj2` is FITTED **and** not f16-representable

`ddm_ix2` migrated `st_grid` into `inflate.py` on `dc1_fold` because it is byte-equal to the
vendored `ST_GRID` there, and warned that the verdict is a `(field, ARCHIVE)` pair property.
It is, and `classify_against_vendored` returns `VIDEO_DERIVED` on `pj2` exactly as predicted.
The encoder counts it rather than migrating it, and does not reuse `ix2`'s `dc1_fold`
migration number.

**What the inherited plan could not do is COUNT it.** `pack_config_section` raised:

```
st_grid is not exactly representable in f16 — widen the section rather than
silently rounding a fitted codebook
```

That guard was right and it fired on the first real case. `ms8`'s solver authored the ladder
as decimals (`0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16`); none of
them is a dyadic rational. Rounding to f16 would move `s_t`, hence
`pose_to_homography`, hence every warped `frame_0`. The container would then be LOSSY, the
"`d_seg`/`d_pose` invariant by construction" exemption would be void, and this arm would be
publishing a rate saving that had quietly spent distortion it never measured.

**The cure: a ladder of EXACT counted-table encodings**, admitted only when the decoded
float64 equals the input float64 element for element — `TABLE_F16` / `TABLE_F32` /
`TABLE_F64` / `TABLE_SCALED_INT` (`code / 10**k`), cheapest first, with `f64` as the lane
that always qualifies so the ladder never fails closed on a finite table.

A fitted ladder is authored as decimal literals, so the integer-over-a-power-of-ten lane is
both exact and the narrowest:

| table | f16 | f32 | f64 | scaled-int | shipped |
|---|---:|---:|---:|---:|---|
| `rs_beta_mags` (13 halves) | 26 | 52 | 104 | **15** (`k=1`, i1) | scaled-int |
| `st_grid` (11 fitted decimals) | **impossible** | 44 | 88 | **13** (`k=3`, u1) | scaled-int |

**Counted config section on `pj2`: 36 B** = 8 B header + 15 B β codebook + 13 B fitted ladder.
The same section on `dc1_fold` is 23 B (grid proved generic, omitted) against `ix2`'s 32 B —
so the exactness ladder **pays for itself by 9 B even on the base that did not need it**.

**The exactness test is the non-obvious half.** It is `code / 10**k == v`, never "`v * 10**k`
is an integer": MEASURED, `1.001 * 1000` is `1000.9999999999999` in float64 while
`round(…)/1000` reproduces `1.001` exactly, so an integrality test would REJECT a lane that
is in fact lossless. (My first version of that test asserted `0.07 * 1000 != 70.0`; it is
exactly `70.0`, and the test caught my own wrong premise before it reached the memo.)

### §2b I PRICED THE ROUNDING I REFUSED — and it is enormous

Refusing to round is easy to write and easy to over-claim, so I built the rounded container
and measured it against the exact one through the real receiver:

```
st_grid exact : 0.06     0.065     0.07     0.075    …
st_grid f16   : 0.0599975…  0.0650024…  0.0700073…  0.0750122…  …
max abs error 3.42e-05      max REL error 2.98e-04
```

**MEASURED over the first 60 pairs: 60 of 60 frames CHANGED, max |frame_0 delta| = 205 grey
levels, 2,437,111 differing subpixels.**

A **3e-4 relative** perturbation of one counted table moves pixels by up to **80 % of full
range**. `s_t` scales the translation inside `K(R − t nᵀ/h)K⁻¹`, so it moves the sampling
grid of the whole warp; near occlusion boundaries a sub-pixel shift is a large photometric
step. The fitted ladder is a **high-gain** input, not a cosmetic constant.

So this was never pedantry. Had the section silently rounded, `d_seg` and `d_pose` would
both have moved, the "invariant by construction" exemption would have been void, and this
arm's rate-only ΔS would have been a **fake score claim** — a rate saving quietly paid for
with unmeasured distortion. **The guard `ddm_ix2` wrote is what caught it, on the first
archive that exercised it.**

---

## §3 THE SIX MANIFEST HASHES — deleted, by construction

`ddm_ix2` audited all six and recommended deletion: **0 read+correct, 4 unread+correct,
2 unread+WRONG** (`tokens_sha256` from `ddm_gd3`, `tr1_packet_sha256` new in ix2, with
`build_packet` verified deterministic so the field is false rather than the rebuild being
nondeterministic).

In container form there **is no manifest**, so all six are gone by construction rather than
by a policy decision — and so are `beta_idx_counts`, `selector_num_two`, `schema`, `base`
and `pose_carrier`, none of which any decode step reads. Build-time provenance lives in
`cx1_build_receipt.json` at zero archive cost.

I did **not** make one hash real and read. That would be a new counted field plus a new
fail-closed obligation, on a line whose parse already closes exactly
(`offset == len(payload)` at every level, the #417 counted-vs-inert bijection), which is a
strictly stronger check than a digest the receiver could compute for itself.

---

## §4 LOSSLESSNESS — proved from the container bytes alone, then again through the receiver

**Leg 1, encoder-side (`verify_container_bytes`), from `payload` and the receiver's generic
constants only — never from the archive it replaced:**

```
tokens_bit_identical             renderer_mask_bit_identical
renderer_floats_bit_identical    selector_bit_identical
pose_warp_bit_identical          dim0_offset_exact
beta_mags_exact                  st_grid_exact
```

8/8. The encoder also **refuses to run** unless every value the receiver carries as a
migrated constant (`frame0_policy`, `tr1_metadata`, `pose_stub`, the section order) is
proved byte-equal to this archive's own copy — a migrated constant that silently differs is
a decode defect wearing a rate saving.

### §4b FULL n600 FRAME PARITY — 0 mismatched pairs of 600

`tools/cx1_verify_frame_parity.py` decodes **both** archives through the real
`inflate_runner_v4d.Decoder` and compares the whole clip. MEASURED:

```
pairs_compared            600
all_frames_bit_identical  true
mismatched_pairs          []
state_equal               n_pairs, dim0_offset, beta_mags, st_vals,
                          p_best, st_idx, sel, ab, beta_idx   -> all true
parsed_packet_equal       metadata, selector, section_payloads, token_codes,
                          pose_stub_consumed, masks, gains, biases -> all true
```

Every `frame_0` and every `frame_1` rendered from the container is **bit-identical** to the
one rendered from the legacy `pj2` archive — the whole clip, not a spot check. So `d_seg`
and `d_pose` cannot move, and the only term left is `25 · bytes / 37,545,489`.

**Leg 3, the gate's own artifacts.** Both gate runs retained their inflated output. The two
`inflated/0.raw` files — 3,492.7 MB each, the exact pixel stream the frozen SegNet and
PoseNet consumed — hash to the **same** `sha256 988785e7cadfd6137d918b53d020fe5f34e5735765433a4873241cb15f7e200e`.
This is the check that needs no argument at all: it compares the contest inflate path's own
output, not my reconstruction of it, and it is why the measured `d_seg` and `d_pose` are
identical rather than merely close.

Two design notes on the verifier, because a verification harness that cannot return the
negative proves nothing:

* It **refuses to run** if the container archive has no `0.bin` — otherwise the receiver
  would take the legacy branch and the whole comparison would be vacuously true.
* `_packet_equal` names every field the renderer consumes and lets a missing one raise
  `AttributeError`. My first version used `hasattr` and reported `null` for the packet
  check — an instrument that can only answer "yes" or "unknown" is the vacuity trap, and it
  is exactly the class this repo has been bitten by three times today.

---

## §5 THE STALE RECEIVER — the defect that would have made the gate meaningless

`stage_v4d_realized_gate.sh` took its receiver from
`${V4D_DIR}/inflate_runner_v4d.py`. MEASURED: that file was **byte-identical to
`git show bed39893b4^:experiments/inflate_runner_v4d.py`** — i.e. the repo receiver from
*before* `ddm_ix2` landed the container path. It has no `0.bin` branch at all.

A container archive staged through it would have selected the legacy 6-member path and died
on the missing `manifest.json`. Not a wrong score — a crash — but the class is the one that
does produce wrong scores: **the deployed copy silently trailing the repo.**

Three changes, all additive:

1. `stage_v4d_realized_gate.sh` now prefers the **repo** receiver, falls back to the SSD
   snapshot, and **prints which one it used**.
2. It copies `src/tac/optimization/ddm_ix2_archive_container.py` into the run submission dir
   as vendored decode code (free generic code, rule-118; copied from the repo so the encoder
   and the decoder cannot drift). This closes `ix2`'s owed vendoring item.
3. `inflate_runner_v4d.py` imports the vendored module first and falls back to the `tac`
   package, so the receiver no longer needs `tac` installed in the contest runtime.
   The SSD snapshot was re-synced from the repo.

**A related measurement worth keeping:** the vendored `ddm_tr1_runtime.py` /
`ddm_r7_token_coder.py` / `pfs1_warp_receiver.py` in the submission template are **not
textually identical** to the repo sources (303 and 549 diff lines). I did not assume that
was harmless — I measured it on every path this build touches: token decode, `ST_GRID`, the
renderer raw frame and `_encode_tokens` all agree **bit for bit**. The encoder resolves the
vendored copies anyway (`--runtime-tree`), so it and the decoder read the same modules.

---

## §6 ROUND-1 ADVERSARIAL REVIEW OF MY OWN RESULT

Both prior arms said this was the most valuable thing they did. Four attacks, all run.

1. **"Is the −6,598 a framing artifact — was the `pj2` BASELINE badly built?"** This is the
   attack that would have inflated the headline, so it is the first one I ran. Re-zipping the
   six untouched members with minimal fixed-timestamp framing reproduces **360,406 B exactly**
   — and so does a rebuild that gives the baseline every advantage by picking the smaller of
   stored/deflate-9 per member. The 6-member baseline is already minimal. **The −6,598 is a
   true delta, not a comparison against a bloated control.**

2. **"Is the counted config section actually CONSUMED, or is the parity pass vacuous?"**
   Mutation control: perturb one fitted `st_grid` codeword by 0.001 and rebuild. The parity
   harness **detects it** (`st_vals: False`), and at the pixel level `frame_0` moves by up to
   **247 grey levels on exactly the 5 sampled pairs whose `st_idx == 3`** and by **0** on
   control pairs using other codewords. The section reaches the pixels; the check can return
   the negative. (§2b then re-ran the same instrument on the f16 rounding itself.)

3. **"Would the GATE actually run the code I verified?"** The staged submission dir's
   `inflate_runner.py` and `ddm_ix2_archive_container.py` are byte-identical to the repo
   files, and the gate log prints `receiver=… [src=repo]` and `container form: 0.bin`. Before
   the §5 fix this attack would have FAILED — which is how the stale receiver was found.

4. **"Is the encoder's own byte accounting honest?"** Round-2 review of my own tool found two
   reporting defects and both are fixed: it reported `manifest.json` as
   `len(json.dumps(manifest))` (**1,545 B**) rather than the member's real size (**1,451 B**)
   — an accounting that measures the reporter, not the archive — and it listed deleted
   manifest keys unconditionally, crediting itself with removing fields some archives never
   carried. Separately I verified rather than assumed that `set -e` does not exit on the
   failing left operand of `[ -f … ] && cp …`, because guessing wrong there would break every
   future gate on a host without the repo.

**Where I was wrong, before review:** my first parity harness reported the packet check as
`null` (a `hasattr` probe), and my first version of the scaled-int test asserted
`0.07 * 1000 != 70.0`, which is false. The test caught my own wrong premise; the `null`
was the vacuity trap in miniature. Both are fixed and both are now regression-tested.

**Rebuild determinism:** rebuilding the container from the same legacy archive reproduces
sha `1d3ab694c337f3f7374fa42034664b0494d0dfda1be479b1d367e964da78701f` / 353,808 B across
**three** independent runs, byte-identical to the archive the gate scored.

---

## §7 WHAT I DID NOT DO / OWED

* **No paid dispatch, no training, no `upstream/` edit, pointer UNMOVED.**
* **I did not touch the pose member.** cp1's ceiling for `pj2 × dc1_fold` is 0.006 % of gap
  and it is negative as built; that verdict is consumed, not re-derived.
* **I did not re-race any member.** `ix2` closed the token member across 6 layouts × 4
  coders × 3 transforms × 5 context models, and measured `renderer.sec` and `pose_warp.stp`
  as honest zeros. Re-racing them would be rediscovery.
* **The remaining lossless headroom on this vehicle is ~50 B** (`ix2` §8: colex on
  `sel_coded`, order-0 arithmetic on `beta_coded`, `pose_warp` framing). The lossless axis is
  essentially exhausted. Everything past it is LOSSY and inherits cp1's tighter-solve law.
* **I did not re-solve pose on the container.** `pj2`'s GN census says the solve is still
  bound-limited (`trust_radius_cap` 1,462 of 1,504), so pose is not exhausted — but that is a
  solver unit, not a container unit, and it composes with this row rather than competing:
  the pose member is untouched here and `d_seg` is inherited bit-unchanged from the tokens.

## §8 NEXT-IF-RESUMED

1. **The new live best is `cx1` at S 0.8264972 / 353,808 B.** Rate is now **0.2355862 of a
   0.6543562 gap**; against the PR130 floor's rate term (0.127214) there is **0.1084 of rate
   gap left**, but only ~50 B of it is reachable losslessly on this vehicle. The next real
   rate move is a REPRESENTATION change (fewer tokens / a smaller lattice), which is LOSSY and
   inherits `ddm_cp1`'s tighter-solve law — and `pj2` is now the tightest solve on the line.
2. **Seg is still the majority of what is left**: 0.4311790 against the bar's 0.02966 = a
   0.4015 gap, **61 % of the remaining 0.6544**, and it has not moved by one ULP across
   `v4d → pw1 → ms8 → dc1_fold → pj2 → cx1`. Every win on this line has been pose or rate.
3. **Do NOT re-race any archive member** (`ix2` closed tokens across 6 layouts × 4 coders ×
   3 transforms × 5 context models; `renderer.sec` and `pose_warp.stp` are measured zeros),
   and **do not quote an oracle conditional entropy as realizable headroom** (`ix2` §4).
4. **Any future counted table must go through `encode_exact_table`.** §2b measured that a
   3e-4 relative rounding of one counted table moves frames by up to 205 grey levels; the
   ladder is the structural guard, and `pack_config_section` is the only sanctioned caller.
* **`ix2`'s own `dc1_fold` container was never gated**, and its config *section* shrinks from
  32 B to 23 B under this arm's exactness ladder (MEASURED on the section; the archive delta
  is not exactly −9 B because the section is inside the jointly-coded group, so I do not
  quote an archive number I did not build). Not re-measured here: `dc1_fold` is superseded by
  `pj2`, and spending a gate slot on it would be spending it on the second-best base.
