---
schema: ddm_mt1_menu_triage.v1
date_utc: 2026-08-02
arm: ddm_mt1 (task #871 continuation — triage the unmeasured discrete choice points)
lane_id: "lane_ddm_mt1_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: INSTANCE
axis: "[macOS-CPU $0 static scan + byte-exact coder measurement; d_seg NOT measured]
  NON-PROMOTABLE. NO training, NO scorer job, NO paid dispatch, NO gate fired, NO pointer mutation."
consumes:
  - .omx/research/ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801.md  (the 84 denominator)
  - .omx/research/ddm_lg2_binary_inventory_20260802.md                                (the 64 denominator)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md                            (AT_A_BOUND, the positive)
  - .omx/research/ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802.md           (DEGENERATE, the null)
  - .omx/research/ddm_cv1_seven_surface_convocation_20260802.md §4                    (the charter)
  - sub015_DAG…20260611.md FEED-pb2                                                   (DIRECTION, the negative)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip       (shipped tokens + selector)
  - /Volumes/VertigoDataTier/pact/ddm_dc1_20260802/dc1_inventory_receipt.json
produces:
  - tools/mt1_menu_triage.py                 (the AST inventory, with a full exclusion ledger)
  - tools/mt1_structural_rate_ladder.py      (the byte-exact structural rate measurement)
  - /Volumes/VertigoDataTier/pact/ddm_mt1_20260802/{mt1_inventory.json,mt1_rate_ladder.json}
consumers: [MAIN, ddm_bs2, ddm_lg2, ddm_dc1, ddm_tw1 "#869", "#871", "#882", "#907"]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_mt1 — the 79 unmeasured choice points, triaged; and the largest one is a structural rate bound nobody had priced

## §0 POINTER HONESTY, and the headline

**The exact contest pointer is UNMOVED** (`0.1910828242 [contest-CPU]` submittable) **and no gate was
fired.** Everything below is `[macOS-CPU advisory]`, `score_claim=false`. This arm ran zero scorer
jobs (MAIN holds the slot) and zero training.

Anchoring, per the "no unanchored ΔS" rule: **LIVE BEST = `dc1_fold` composed
S 0.8983766 at 360,309 B** (pw1's 0.9476070 is superseded). Bar = PR130 **0.172141**.
**Gap = 0.7262356.**

**The headline is a measurement, not a classification.** The charter said *"if a $0 row is both
AT_A_BOUND and cheap, MEASURE IT."* One was, so I did:

> **`grid_downsample` is shipped at 16 — the COARSE ENDPOINT of its own menu `choices=[8, 16]` —
> and the RECEIVER already admits 32.** Re-gridding the shipped token tensor to the receiver-legal
> 12×16 and re-encoding through the REAL SMEVR coder saves **259,413 B of tokens** on the
> adversarial control arm, against a **+778 B** renderer cost (below), for a net
> **258,635 B**: **ΔS_rate = −0.172214 = 23.71% of the entire remaining gap**, lossless
> round-trip verified. It pays iff d_seg rises by less than **1.722e-03** (39.9% of the live
> d_seg 0.00431179).

That is 3.7× the deepest rung of the `token_quant_levels` ladder that `ddm_bs2` priced, and it was
sitting in the un-triaged 79. Its mechanism is *exactly* pw1's: **the capability was built; only
the menu pinned it.**

**The +778 B is a correction I owe to my own round-1 review**, and it is a structural point, not an
arithmetic nudge. `grid_downsample` sets `n_upsample = log2(ds)` (`ddm_tr1_runtime:282`), so ds=32
adds a **fifth** `up4` conv of shape `(24,3,3,24)`: params 22,248 → 27,432. The renderer is a
seed-regenerated lotto bank whose only counted content is the mask, so the cost scales with param
count: 3,341 B → **~4,119 B** (DERIVED by linear scaling from the shipped section, not measured).
So **`ds` is not a pure rate lever** — it trades token bytes for decoder depth. Two consequences:
(a) the net figure above, and (b) **the extra depth is free generic decoder capacity**, which
partially offsets the resolution loss on the d_seg side and makes the break-even easier to clear
than a naive "coarser grid = worse seg" reading suggests. My first draft priced only the tokens
and would have overstated the row by 0.3% while missing the mechanism entirely.

## §1 DENOMINATOR HONESTY — the point of this arm

I did **not** reproduce bs2's 84 or lg2's 64, because **neither published its per-row table**, so
neither is re-checkable. I re-derived the inventory from source with a deterministic AST scan
(`tools/mt1_menu_triage.py`) over an explicitly named 10-file scope, and I report my own
denominator with every exclusion counted.

| | rows | note |
|---|---:|---|
| raw syntactic candidates | **241** | every menu / bool / mode / control-flow accept found by AST |
| — `integrity_guard` | 81 | fail-closed structural validation (`sha`, `magic`, `len`, `shape`, schema). **No numeric DOF**: it passes or the program dies. Not a choice point. |
| — `local_state_var` | 23 | `accepted = False` inside a loop: state, not an admissible set |
| — `dispatch_arm_of_counted_menu` | 16 | `codec == "smevr"` etc. — one arm of `CODEC_IDS`, already counted once |
| — `entrypoint` | 6 | `__name__ == "__main__"` |
| — `dunder_export_list` | 2 | `__all__` |
| **IN SCOPE** | **113** | 11 discrete menus · 36 mode strings · 10 boolean flags · 56 accept/reject rules |

**THE DENOMINATOR IS A MOVING TARGET, and that is a finding.** This scan returned **112** in-scope
rows early in the session and **113** at the end, because sister arms edited
`ddm_v4c_resolve.py` and `ddm_v4d_resolve.py` **while this arm was running** — every line number
downstream of their edits shifted, and my first class-assignment table (keyed on positional row
ids) silently stopped resolving. **A triage keyed to a position in a live tree mis-attributes
without saying so.** `tools/mt1_triage_assign.py` now keys on `(basename, symbol, occurrence)` and
**REFUSES rather than guesses** when a key stops resolving. Any successor quoting "84" or "64" or
"113" must also quote the commit it was measured at.

**The class histogram is GENERATED, not transcribed** (`tools/mt1_triage_assign.py`, artifact
`mt1_triage.json`). Hand arithmetic over overlapping sections is exactly where a denominator
silently goes wrong — my own first draft of §4 said "56 residue" and the generated count is **50**.

| class | rows | % of 113 |
|---|---:|---:|
| `NO_DOF` (residue, §4) | **50** | 44.2% |
| `NO_OCCUPANCY_DATA` (§3.4) | **36** | 31.9% |
| `DIRECTION` (§3.2) | **9** | 8.0% |
| `AT_A_BOUND` (§3.1) | **8** | 7.1% |
| `INTERIOR_CLOSED` (§3.0) | **5** | 4.4% |
| `DEGENERATE` (§3.3) | **5** | 4.4% |
| **explicitly assigned** | **63** | 55.8% |

**And per `ddm_na1`, a third column the charter asked for: how many are DEGENERATE.** Direct
answer: **5 of 113 rows are exactly degenerate** (2 `ST_GRID` copies + 3 `BETA_MAGS` copies), and a
further **5 rows act on a degenerate coordinate** (the dim0 bracket, class A; the beta bracket,
class B in part) — see the step-0 pre-screen in §2.2. **10 of 113 = 8.8% of the in-scope inventory
is touched by a degeneracy class**, and every one of them was previously classified by some arm as
if it imposed a limit.

**Reconciliation, stated as a difference not a correction.** bs2 scoped to the live chain and
counted mode-strings separately (84). lg2 added the TR1 trainer's argparse and dropped
accept/reject (64). Mine keeps all four kinds and adds the trainer *and* the decode-path adapter
`repair_entropy_coder_runtime_adapters.py` that lg2 explicitly named as its sharpest unread gap.
**All three denominators are defensible under their own scope; none is wrong.** What was missing in
all three was the per-row table, which is §3.

**Scope, named so a successor can re-aim.** The 10 files are the 5 the gate actually stages into
the eval submission (`stage_v4d_realized_gate.sh:41-44` = the true decode set) plus the 5
encode/solve producers. **No claim here is repo-wide.** NOT scanned: `terminal_pose_gn.py`
(lg2 verified not live), `ddm_su2_qa43_tail_solver.py`, `ddm_kl1_pose_field_receiver.py`,
`inflate_runner_v4c.py`, the `mq1_*` source.

**How many of the 84 did I classify?** All 112 of my own rows carry a class (§3 + §4). Mapping onto
bs2's numbering is **not possible** — bs2 published 8 named rows of 84. Of those 8, **8/8 appear in
my inventory** and my classes agree with bs2/lg2/dc1/pw1 on every one (§3.0). That 8/8 agreement on
the overlap is the only cross-arm check available, and I report it as such rather than claiming to
have "covered the 79."

## §2 THE RULE, AND THE FIVE COLUMNS THE OPERATOR ADDED

Three receipts fix the discriminator, one per sign:

| receipt | sign | law |
|---|---|---|
| `ddm_pw1` | **+** | occupancy piled AT A BOUND ⇒ freeing pays (0.9639878 → 0.9476091) |
| `FEED-pb2` | **−** | a DIRECTION resolved by a myopic probe got WORSE when freed; the achievable bound is the per-pair min of BOTH continuations |
| `ddm_dc1` | **0** | `s_t`'s 7/11 dead codewords bought nothing in FORMAT — it is exactly multiplicatively degenerate with the shipped translation triple (rel diff 4.539e-16, n600) |

⇒ classes **AT_A_BOUND · DIRECTION · DEGENERATE · NO_OCCUPANCY_DATA**.

### §2.1 THE DISCRIMINATOR'S OWN SCOPE — re-derived after `ddm_na1`, and it is WEAKER than the charter states

`ddm_na1` (commit `0d5717c2da`) found that pw1's **negative control was degenerate**: `s_t` is a
member of degeneracy class **A** (`s_t` ↔ `[p2,p1,p0]` ↔ `CAMERA_HEIGHT_M`), and a coordinate with
zero net DOF **cannot saturate**, so the control could not have failed. That is a control-validity
failure and I am not entitled to call the rule "validated." I re-derived the instrument's basis
from source rather than accept either framing:

| pw1 arm | coordinate | degeneracy | verdict on the evidence |
|---|---|---|---|
| **negative control** | `s_t` (`ST_GRID`) | **class A** | **INVALID.** na1 is right; a degenerate coordinate's occupancy is gauge-determined. |
| **positive #1** | `dim0` | **class A** — `pfs1_warp_receiver.py:45` reads `t = s_t·[pose6[2], pose6[1], pose6[0]]`, so **`dim0` = `pose6[0]` IS a member of the translation triple** | **RE-SCOPED.** pw1's *measured* win stands (byte-closed 0.9639878 → 0.9476091); its *attribution* ("97.89% outside the shipped search's reach") is scoped to the dim0-alone parameterisation — the same point is reachable by moving `s_t`. dc1's `gap_lattice`/`gap_search` split applies. |
| **positive #2** | `beta` | **partly class B.** `inflate_runner_v4d.py:177,180` pass `rot` into the **`s_r` slot** of `pose_to_homography`, and `s_r` ↔ `[p3,p4,p5]`. So β's *overall scale* is absorbable. | **SURVIVES, and it is the ONE clean positive.** DERIVED, not measured: for β≠0 the receiver blends **two** warps at `s_r = 1∓β/2` and `1±β/2` (`:196-201`), and a class-B rescale by `c` maps both to `c(1∓β/2)` — **the RATIO `(1+β/2)/(1−β/2)` is invariant**, so the spread β controls is *not* absorbable. The sign flip `β → −β` reorders the two warps against `alpha_row` and is manifestly not a rescale. pw1's own MEASURED decomposition puts **72.8% of the beta gain (29 pairs, 0.21963 of 0.30155) in the bucket that needs BOTH magnitude and sign** — i.e. in the non-degenerate part. |

**⇒ THE INSTRUMENT'S HONEST STATUS: one valid positive, NO valid negative control.**
It can **FLAG** a candidate; it **cannot CERTIFY a null.** That is precisely the
"probe that cannot return the negative" class (memory `m50`), now found in the campaign's own
triage rule. Consequences, applied throughout §3:

* **`AT_A_BOUND` rows are CANDIDATES**, not verdicts. The flag direction has one valid positive.
* **`INTERIOR_CLOSED` is the null direction and the discriminator cannot support it.** Of my 5
  such rows, **3 survive on evidence that is not the discriminator at all** — `selector` (direct
  376/224 census), `AUTO_CODECS`/`CODEC_IDS` (all 9 raced byte-exact), `RS_GLOBAL_G` (a downstream
  join showing 22/101 escaped). Those stand. The other 2 are re-scoped to INSTANCE.
* **`DEGENERATE` is now a STEP-0 PRE-SCREEN, not a peer class** (§2.2).

**What does NOT depend on the discriminator at all: the §0 headline.** `grid_downsample` was
*found* via the bound heuristic but is *established* by a direct byte-exact coder measurement.
And it is **non-degenerate by na1's own accepted argument** — the one it used to certify `ba29`
SOUND: *no shipped continuous coordinate can absorb a byte count.* A change in the **number of
token cells** cannot be absorbed by rescaling any pose coordinate. **The headline survives na1
intact.**

### §2.2 STEP-0 DEGENERACY PRE-SCREEN (new, per na1)

Before asking "is it at a bound?", ask "**does it impose a limit at all?**" Screened every in-scope
row against na1's three classes:

| class | members | in my 112-row scope |
|---|---|---|
| **A** | `s_t` ↔ `[p2,p1,p0]` ↔ `CAMERA_HEIGHT_M` | `A1`, `I1` (`ST_GRID` ×2) — already DEGENERATE; **plus `E3`/`E4`, the dim0 bracket, which acts on `pose6[0]` ∈ class A. NEWLY FLAGGED.** |
| **B** | `s_r` ↔ `[p3,p4,p5]` | `E7`/`E8`/`E9` (the beta bracket) — **partially**: overall scale absorbable, ratio + sign are not (§2.1) |
| **C** | `NATIVE_FX/FY = 910` ↔ `p0`, on the `R=I` ground path only | no menu row; scalar constants, outside my scan's kind set |

**Screened NON-degenerate, with the reason** (a screen that reports only its positives is the
defect it audits): `grid_downsample`, `code_width`, `token_quant_levels` — all three change the
**number or depth of token symbols**, and no continuous pose coordinate can absorb a symbol count
(na1's own ba29 argument). `selector` — builds its second homography with a **literal `0.0`** for
`s_t`, and a literal zero survives any fold (na1 confirms). `rs_beta_mags` — dc1's not-degenerate
ruling, which na1 re-derived at the decode path and calls **SOUND**.

Per the 2026-08-02 operator directives, every row also carries:

* **ΔS split.** `seg+rate` is the DECISION column; `pose` is reported separately and is
  **repairable by methods we already hold**. A row that helps seg or rate and costs pose is
  **FIREABLE**, not a wash.
* **LEVEL** — L1 program / L2 description / L3 receiver / L4 scorer-feature. A knob at the wrong
  level reads inert when it is actually mis-placed; that is a distinct failure from a knob at a bound.
* **UNIT** — per-cell / per-pair / per-pixel / global. The scorer's unit is the argmax **cell**;
  a per-pixel and a per-cell choice point are not comparable.
* **DOF** — the true degrees of freedom, which is often **below** the arity.
* **CONTIGUITY** — whether the occupancy receipt is population-scoped or prefix/subset-scoped.

## §3 THE TYPED TABLE

### §3.0 Already measured by a sister arm (8 rows) — carried, not re-run

| row | site | occupancy (MEASURED) | class | LEVEL/UNIT/DOF | agreement |
|---|---|---|---|---|---|
| `ST_GRID` ×2 copies | `pfs1_ep_warp_pose_solve:61`, `pfs1_warp_receiver:18` | `[0,0,0,0,0,0,22,364,156,58,0]` n600 | **DEGENERATE** | L2 / per-pair / **DOF 0** (dc1: exactly degenerate with the shipped translation triple) | pw1 = bs2 = lg2 = dc1 |
| `BETA_MAGS` seed | `ddm_v4d_resolve:66` (+2 copies) | 76 pairs at the top entry, 26.4% of mass | **AT_A_BOUND — CURED** | L2 / per-pair / DOF 1 | pw1 fixed; manifest now ships 13 entries |
| `dim0` bracket | `ddm_v4d_resolve:177-184` | 124/600 at the bound, 37.4% of mass, 2.3× interior | **AT_A_BOUND — CURED** | L2 / per-pair / DOF 1 | pw1 fixed |
| `selector` | `inflate_runner_v4d:179` | 376 / 224 | **INTERIOR** | L3 / per-pair / DOF 1 | dc1; no blended compose is representable |
| `AUTO_CODECS` / `CODEC_IDS` | `ddm_r7_token_coder:55,56` | all 9 raced byte-exact; smevr wins by 51,546 B | **CLOSED** | L3 / global / DOF 1 | bs2 |
| `RS_GLOBAL_G` | `ddm_v4c_resolve:93` | 415 / 84 / **101 at top**, 28.05% mass | **SATURATED SEED, not a binding bound** | L2 / per-pair / DOF 1 | lg2: 22/101 escaped downstream; pw1's bracket already cures it |
| `token_quant_levels` | `ddm_tr1_runtime:83,342,533` | shipped 16 **== the SMEVR ceiling** | **AT_A_BOUND** | L2 / per-cell / DOF 1 | bs2 "CLIPPING-SUSPECTED"; **re-priced in §3.1** |
| entry-probe direction ± | `ddm_v4d_resolve:215,313` | 8/109 changed outcome; 2 pairs carry 96% | **DIRECTION** | L1 / per-pair / DOF 1 | pb2 MEASURED −3.745e-05 = 0.005% of gap |

**Directive (1) re-check of the already-measured rows — one verdict FLIPS.** `token_quant_levels`
was judged by bs2 on a JOINT break-even against d_seg. Under the split rule its whole effect is on
**rate**, its cost is on **seg**, and **it has no pose term at all** — so the joint framing never
disadvantaged it and the verdict stands. The row that *does* move is
`grid_downsample` (§3.1), which no prior arm scored at all. **No already-measured verdict is
overturned by the split; I checked all 8 and report that as a negative.**

### §3.1 AT_A_BOUND — freeing may pay (8 rows). Ranked by measured |ΔS_seg+rate|

| # | row | site | occupancy / bound evidence | ΔS **seg+rate** | ΔS **pose** | LEVEL/UNIT/DOF | falsifier | cost |
|---|---|---|---|---|---|---|---|---|
| **1** | **`grid_downsample`** | `train_tr1…:1692` `choices=[8,16]`, shipped **16 = coarse endpoint** | shipped value IS the endpoint; **receiver admits 32** (`ddm_tr1_runtime:283` power-of-two, `:331` grid×ds==(384,512); 32→(12,16) integer; `grid_downsample` is a free key in `_SELECTOR_BASE_KEYS`, pinned to no value) | **rate −0.172214 MEASURED net** (23.71% of gap; tokens −259,413 B decimated control, renderer +778 B DERIVED; pooled token arm −273,161 B is the optimistic one). **seg: pays iff Δd_seg < 1.722e-03** | **0** — tokens carry no pose term | L2 / **per-cell** / DOF 1 | train at ds=32; if Δd_seg ≥ 1.722e-03 the coarse grid is under-resolved and the bound is correct | 1 training run |
| **2** | `code_width` | `train_tr1…:1693` `choices=[2,4,6]`, shipped 4 | **INTERIOR on the menu** — listed here because the menu is a 3-point sample of an integer continuum with no receiver limit (`ddm_tr1_runtime:276` `minimum=1`) | rate **−0.117283 MEASURED** at cw=2 (16.15% of gap); −0.060684 at cw=3 | 0 | L2 / per-cell / DOF 1 | as #1 | 1 training run |
| **3** | `token_quant_levels` | `ddm_tr1_runtime:83` `_R7_SMEVR_MAX_LEVELS=16`; shipped **16** | **the default IS the codec ceiling** (`:342`, `:533`). General guard allows ≤256 (`:340`) | rate −0.0158 … −0.0704 MEASURED (2.17%…9.69% of gap) downward; **upward is BLOCKED by the codec, not by the data** | 0 | L2 / per-cell / DOF 1 | widen SMEVR past 16 and re-price; if the pre-quantisation activation is genuinely saturated the ceiling is correct | $0 down; codec work up |
| 4 | lane-guard budget | `lane_guard.py` via `train_tr1…:1706` | λ=0 on **64/64** gates; budget one value all run | rate 0 · **seg: covers 0.050213 S-units of Lane erosion that the constant budget was blind to** | 0 | L1 / global / DOF 1 | bs2's ratchet; a guard that engages while GT Lane cost falls refutes the calibration | built, **default-off** |
| 5 | `verdict_chunk > 120` | `train_tr1…:1929` | a hard refuse; the n600 OOM cure capped it | 0 (advisory verdict only) | 0 | L1 / per-pair / DOF 1 | none needed — measurement-side, score-neutral | n/a |
| 6 | `max(lengths) > 15` | `ddm_r7_token_coder:1250` | Huffman code-length ceiling; `huffman_nibble` is +50% and never selected | rate 0 while huffman is unselected | 0 | L3 / global / DOF 1 | only binds if huffman ever wins the race — it does not | $0, closed |
| 7 | `len(table) > 256` | `ddm_v4d_build_composed_archive:135` | uint8 beta-table ceiling; live tables **13** (pw1) and **44** (mq1) | 0 — **not near the bound** | 0 | L2 / global / DOF 1 | a solve wanting >256 beta entries | $0, closed |

**Rows 1–3 are the same physical object seen on three axes.** `codes = 600 × (384/ds) × (512/ds) ×
cw`, coded at ~log2(levels) bits. **On the RATE axis alone these three are mutually degenerate up
to reparametrisation** — any target byte count is reachable by any of them — so pricing them
independently and taking the cheapest is the wrong move. The correct object is the **(ds, cw,
levels) frontier at fixed rate, selected by d_seg**, and their d_seg costs are *not*
interchangeable: `ds` trades **spatial resolution of the description**, `cw` trades **per-cell
descriptive richness**, `levels` trades **lattice depth**. That is a coordinate fact, and reading
any one of them alone as "the rate lever" is the error dc1 caught in a different dress.

### §3.2 DIRECTION — freeing may HURT (4 rows)

| row | site | why DIRECTION | verdict |
|---|---|---|---|
| dim0 entry probe | `ddm_v4d_resolve:215-219` | `for sign in (1.0,-1.0): … break` — `−1` is **never evaluated** when `+1` improves at all | pb2 MEASURED: neither entry rule dominates (`+` wins 2 of 8, `−` wins 6); achievable bound = per-pair min of both. **−3.745e-05 composed = 0.005% of gap.** Banked, not fired. |
| beta entry probe | `ddm_v4d_resolve:313-317` | identical structure | same |
| `d >= best_d` doubling stop | `ddm_v4d_resolve:326` | first-failure stop on a possibly non-unimodal line | pb2: the entry-probe LOSER can win the continuation (pair 326 ends 4.58e-05 worse) |
| `cval < cur` / `cv < curB` greedy accepts | `pfs1…:204`, `ddm_v4c_resolve:431,484,689` | strict-improvement accept ⇒ **monotone-safe by construction**; cannot report an unrealised win | **correct as written.** Freeing them (accepting equal or worse) would break the safety property pw1's canary depends on. **Do not free.** |

### §3.3 DEGENERATE — imposes no limit (3 rows, +1 NEW)

| row | degenerate with | evidence |
|---|---|---|
| `ST_GRID` | the shipped translation triple `[p2,p1,p0]` | dc1, n600, two independent derivations, rel homography diff **4.539e-16** |
| **`ST_GRID` ×2 source copies** (`pfs1_ep_warp_pose_solve:61` ⟂ `pfs1_warp_receiver:18`) | **each other** | **NEW.** Two literal copies of the same 11-tuple in two files. Not a second choice point — **DOF 1, not 2.** Sister of #907's `ST_GRID` ×5 hand-duplication with one DIVERGED copy. |
| **`BETA_MAGS` ×3 source copies** (`ddm_v4d_resolve:66`, `ddm_v4d_build_composed_archive:44`, `inflate_runner_v4d:70` as `DEFAULT_BETA_MAGS`) | **each other** | **NEW, and the sharpest of the three.** The live manifest ships a **13-entry** `rs_beta_mags`, but all three source copies still read `(0.0, 0.5, 1.0)`. The receiver reads the manifest (`inflate_runner_v4d:127`) so the shipped path is correct — **but any successor reading the constants sees pw1's superseded 3-entry seed.** That is exactly the stale-constant confound, live, in three places. |

**Actively hunted, and these are the degeneracies I found.** I also checked and **REJECTED** one
candidate degeneracy: `code_width` × `token_quant_levels` are **not** degenerate (`code_width` is
the token tensor's channel count, `levels` its per-channel lattice depth — orthogonal factors of
`levels^code_width` per cell). Reported because a hunt that reports only its positives commits the
selection defect it audits.

### §3.4 NO_OCCUPANCY_DATA — no receipt exists (the largest class)

**34 rows.** These are the never-fired / unswept levers. Grouped by whether a read is possible at all:

| group | rows | why no data | cheapest read |
|---|---:|---|---|
| **never-fired mode levers, default OFF** | 14 | `token_ste{round,dither}` · `token_temporal_mode{shared_base,independent}` · `token_init_mode{zero,solve_project}` · `adam_bias_correction{off,on}` · `margin_weighted_loss` · `token_quant_anneal{off,at_knee}` · `token_quant_margin_coupling` · `token_delta_group_sparsity` · `delta_sparsity_engage` · `delta_sparsity_weight_field` · `head_range_relax` · `basin_handoff` · `boundary_probe` · `rate_model{entropy,smevr_surrogate}` — **every one shipped at its control value**; none ever swept on this vehicle | 1 training run each; **`token_ste` is the cheapest** and is coupled to the §3.1 #3 bound |
| **head/architecture modes** | 3 | `renderer_head_mode{rgb,class_field,class_field_photo}` (shipped `rgb`; `class_field` collapses the head to **1** output channel — the topology-matched representation for an argmax partition) · `distill_form{kd_logits,margin_field,argmax_ce}` · `seg_form_start{ce,tau_softplus,unify_tau,margin_hinge}` | changes the TRAINED model ⇒ occupancy undefined at $0 | 1 training run |
| **welding bools** (two axes on one switch) | 4 | `AB_START_POLICIES{neutral,derived}` welds *how many* restarts to *which* restarts (lg2 B10) · `beta = beta_mag * sign(pose[5])` welds magnitude to a hard 0-threshold on yaw (lg2 A6/C2) · `two_all` · `pose_source{shiptable,resolve}` | the two settings are **not a 2-point sample of one continuum** — a bool here is a UI over two different objects | re-run `--mode photo` (see below) |
| **produced-and-discarded** | 2 | `ab_trace` bound at `ddm_v4d_resolve:372` and referenced nowhere (~5 LOC to log) · `ab_stop`/`ab_relins`/`ab_damp_used` written by `ddm_v4c_resolve:818-822` but **`{'ABSENT': 600}`** in the shipped artifact | the signal is generated and thrown away | **the cheapest un-taken read in the chain**: ~5 LOC, then a `--mode photo` re-run |
| **GN internals, no record at all** | 6 | `for scale in (1.0, 0.5)` at 3 sites with *different* damping constants · `RELINS=4` · `for _damp in range(4)` · FD steps | no code path records which value won | **unmeasurable without adding telemetry first** |
| **run-control / device** | 5 | `mlx_device{gpu,cpu}` · `deterministic_r` · `full_confirm` · `telemetry_v9_port` · `variant{plain,lotto}` | score-neutral or architecture-pinned | n/a |

**Directive (2) — every FORMULATION/INSTANCE-scoped negative owes a named follow-on, marked P0:**

| negative | scope | **named P0 follow-on** |
|---|---|---|
| `RS_GLOBAL_G` "saturated seed, not a binding bound" (lg2) | INSTANCE (this base, this solve) | **P0:** re-read occupancy on the *next* solve. 22/101 escaping is not a null — it is a 21.8% escape rate whose seed is still 2-point. Widen `RS_GLOBAL_G` at the SEED and re-measure the escape fraction. |
| `ST_GRID` interior / dead-codeword null (dc1) | FORMULATION (dead-codeword framing) | **P0:** the degeneracy means `s_t` cannot be exhausted by menu work — the follow-on is on the **SEARCH** axis (dc1's own bucket), not FORMAT. Re-solve `s_t` jointly with the translation triple instead of re-fitting the codebook. |
| pb2 entry-probe `−3.745e-05` "not worth a slot" | INSTANCE (this composition) | **P0:** archive `v4d_composed_pb2_bestof_archive.zip` sha `6e1b80e901` is BANKED — fold it into the next pose re-solve that fires for a larger reason, per pb2's own decision line. |
| bs2 `token_quant_levels` "CLIPPING-SUSPECTED, not resolved at $0" | FORMULATION (post-hoc re-quantisation) | **P0:** the discriminator is the **pre-quantisation activation distribution**, which needs the trained model. Fold it into the §5 #1 training run as a free side-read — that run loads the model anyway. |

## §4 THE REMAINING IN-SCOPE ROWS, ACCOUNTED

**50 rows** (GENERATED count, not hand arithmetic) are accept/reject comparisons that survived the
automated exclusion filter but are, on a hand read, **further integrity guards my
`_INTEGRITY_TOKENS` list did not catch** (e.g. `payload[:8] != MAGIC`, `consumed != bit_count`,
`symbol < 0`, `original_length != (count+1)//2`, `coder_family == 'range'` dispatch arms). Each is
fail-closed with **DOF 0**.

**I am reporting these as a residue, not as triaged rows.** They are counted and individually named
in `mt1_triage.json`, and I assert only that a hand read found no numeric degree of freedom in them
— not that a deeper read never will. **50/113 = 44.2% of my in-scope denominator is closed by a
weaker argument than the other 55.8%.** Naming that asymmetry is the honest report, and it is the
single largest soft spot in this arm.

## §5 THE TOP 3 FIREABLE

Ranked by **measured |ΔS_seg+rate|**, per directive (1) — pose effect is zero on all three:

1. **`grid_downsample` 16 → 32.** `ΔS_rate = −0.172732` **MEASURED byte-exact** (decimated control;
   the pooled arm gives −0.181887 and is the optimistic one — **quote the control**). 23.78% of the
   gap. Pays iff `Δd_seg < 1.727e-03`. **Receiver already legal; only `choices=[8,16]` blocks it.**
   Cost: one training run at `--grid-downsample 32` after widening the menu. Falsifier is
   pre-registered and symmetric.
2. **`code_width` 4 → 2.** `ΔS_rate = −0.117283` MEASURED, 16.15% of the gap. Pays iff
   `Δd_seg < 1.173e-03`. Cost: one training run. **Fire second, not in parallel** — see §3.1's
   degeneracy note: it shares the rate axis with #1, so a joint sweep confounds them.
3. **`token_ste` `round` → `dither`.** ΔS unknown, but it is the **cheapest probe of the §3.1 #3
   bound**: it changes exactly how near-edge values quantise, which is the mechanism
   `token_quant_levels = 16 = ceiling` is suspected of. Never swept anywhere on this vehicle
   (bs2 and lg2 concur). Cost: one training run, and it can ride along with #1.

**Fire #1 first.** It is the only row in the entire 112 whose seg+rate effect is both measured and
larger than a fifth of the remaining gap.

## §6 CONTIGUITY + WHAT I DID NOT DO

**CONTIGUITY (directive 3).** Every occupancy figure quoted here is **population-scoped n600**, not
a prefix: `ST_GRID` (600), `selector` (600), `rs_beta_mags` (600), `RS_GLOBAL_G` (600), lane_guard
(64/64 gates over ep644→945). **No subset-scoped receipt is used**, so the `m88` prefix caveat does
not bind any number in this memo. The one exception is bs2's lane-guard gate estimator, which is a
**fixed 36-of-600-pair** subset (`gate_ids_n == 36`) carrying gd1's MEASURED **+3.34% Lane design
error** — bs2's ratchet cancels that offset by construction, and I carry the caveat forward rather
than re-deriving it.

**Not done, named:** no scorer job (MAIN holds the slot), no training, no gate fired, no pointer
mutation, no `upstream/` edit. **d_seg is NOT measured on any §3.1 arm** — every seg column is a
pre-registered break-even. The §3.1 #1 rate figure is exact for a 12×16 token tensor through the
real coder, but its *content* is derived from a tensor trained at 24×32; a run trained natively at
12×16 will emit different content and therefore different bytes. The decimated control bounds that
risk (18.8% between the two reductions) but does not eliminate it.

## §7 FALSIFIERS

1. **#1:** train at `grid_downsample=32`; if `Δd_seg ≥ 1.727e-03` the coarse grid is genuinely
   under-resolved and the endpoint is a *correct* bound, not a pinned one.
2. **Degeneracy claim:** if a native ds=32 run emits token bytes ≥ 346,478 (i.e. the coarse grid
   does not actually reduce rate once trained), the "three axes, one rate object" framing in §3.1
   is refuted and each axis must be priced independently after all.
3. **The `BETA_MAGS` ×3 stale-copy finding:** if any live code path reads the source constant
   rather than the manifest, the shipped 13-entry table is not what runs, and pw1's measured win is
   scoped narrower than believed. (I verified the receiver reads the manifest at
   `inflate_runner_v4d:127`; the falsifier is a path I did not trace.)

## §7.1 WHAT SURVIVES IF THE DISCRIMINATOR IS ONLY INSTANCE-VALID ON `s_t` (na1's question, answered)

| my claim | depends on the discriminator? | survives? |
|---|---|---|
| **§0/§5 #1 `grid_downsample` −0.172214** | **NO** — a direct byte-exact coder measurement; the heuristic only pointed me at the file | **YES, fully.** Also non-degenerate by na1's own accepted ba29 argument. |
| §5 #2 `code_width` −0.117283 | NO — same instrument | **YES, fully** |
| §3.1 rows 3–8 flagged AT_A_BOUND | YES, for the *flag* | **As CANDIDATES.** One valid positive (beta's sign/ratio DOF) supports the flag direction. |
| §3.0 `selector` / `AUTO_CODECS` / `RS_GLOBAL_G` nulls | NO — direct census / exhaustive race / downstream join | **YES** |
| §3.0 remaining INTERIOR_CLOSED nulls | YES, for the *null* | **RE-SCOPED to INSTANCE.** The instrument cannot certify a null. |
| §3.2 DIRECTION rows | NO — pb2 measured them at n600 | **YES** |
| §3.3 DEGENERATE rows | NO — algebraic | **YES, and ENLARGED by na1** |

**Net: the two fireable rows and both measured numbers are discriminator-independent.** What na1
costs me is the confidence of my *nulls*, not my *candidates* — and I have re-scoped the nulls
rather than defended them.

## §8 NEXT-IF-RESUMED

* **Fire §5 #1** — widen `train_tr1…:1692` to `choices=[8,16,32]`, train at 32, byte-close, gate.
  Everything else in this memo is subordinate to that one row.
* **Take the two free reads in §3.4 "produced-and-discarded"** — ~5 LOC at
  `ddm_v4d_resolve:372` to log `ab_trace`, then a `--mode photo` re-run to populate the
  `{'ABSENT': 600}` `ab_stop` census. Cheapest un-taken occupancy read in the chain.
* **Fix the ×3 `BETA_MAGS` / ×2 `ST_GRID` source duplication** (#907's class) — one constant,
  one home, read from the manifest. This is a stale-constant confound sitting live in five places.
* **The 50-row residue in §4** deserves one adversarial pass by an arm that has not already
  concluded they are inert — I closed them with a weaker argument than the rest and said so.
* **na1's P0-5 is now PARTLY paid and partly still owed.** §2.1 re-derives the discriminator's
  basis from source and finds ONE valid non-degenerate positive (beta's ratio/sign DOF, DERIVED)
  and NO valid negative control. **What is still owed is a MEASURED non-degenerate negative
  control** — a coordinate with real DOF whose occupancy is interior and which, when freed,
  measurably does NOT pay. Until that exists the rule can flag but cannot clear, and every
  "closed"/"exhausted"/"interior" verdict in the campaign inherits that limit.
* **`pitch = 0.0`** (na1's surfaced target) is hardcoded at every call site and is *genuinely*
  non-degenerate — `n` is unit-norm by construction so pitch rotates the plane-normal direction and
  no scale absorbs it. It is **not in my 113** because it is a scalar literal, not a menu: my scan's
  kind set structurally cannot see it. **Naming my own instrument's blind spot:** hardcoded scalar
  constants are a whole choice-point family this arm did not enumerate.

Pointer `0.1910828242 [contest-CPU]` UNMOVED. No prior negative re-opened without evidence.
`[no-triality] [p0-ledger-ok]`
