# ddm_mf1_manufactured_seg_repair — 90.47% of dx2's seg error is MANUFACTURED and 78.71% of it is born at the native render; is the breaking addressable, and what does it buy in the distortion currency? (owning memo: ddm_mst1_manufactured_stage_split_20260822.md)

## MANDATE

Operator 20260823: *"work for as long as it takes autonomously with full authority... Believe in
yourself and feel free to be creative and weird and think divergently"*

For two days every arm attacked the ARCHIVE. Six arms measured the token stream; four measured the
model; all of them priced BYTES. The two-currency reading of the demand
(`ddm_tl1_teacher_ledger_20260822.md`) says that was one of two available routes, and the other one
is untouched: shed 42,382 B at fixed distortion **OR** shed 150 B at ZERO distortion. **Distortion is
worth 42,235 B of the gap. Seg alone is worth 30,248 B.**

And seg is not label noise. Two independent arms measured it:
- `ddm_ms9`: **90.47% of dx2's seg error is MANUFACTURED**, not label error — 27,363 B-equivalent,
  64.6% of the whole demand. The realization path **FIXES 6,918 pixels while BREAKING 21,493.**
- `ddm_mst1`: **78.71% of the manufactured error appears at the NATIVE RENDER**, and **R + uint8 are
  net REPAIRERS, not destroyers** — the downstream pipeline is cleaning up, not causing.

Manufactured error is a defect with an address. This arm asks the question nobody asked: **where in
the native render are the 21,493 broken pixels born, is that mechanism addressable, and at what byte
cost does fixing it convert into the 30,248 B seg currency?**

**Why this survives the sharp-optimum law (the thing that closed everything else).** Five concordant
arms measured the HPAC model and the field as jointly at a local optimum, SHARP in every direction —
so perturbing the current object loses. This arm does not perturb that object. It changes the RENDER,
which is the object every seg closure was priced against. `ddm_sy2`'s composition law is explicit:
a closed leg survives only when another leg first CHANGES THE OBJECT it was priced on. That is
precisely the claim under test, and it is falsifiable below.

## SCOPE

1. LOCALIZE the breaking. From ms9's and mst1's retained per-pixel fields, decompose the 21,493
   broken pixels by: class (Lane is 0.59% of area but the worst distortion class, IoU 0.263, ~19% of
   flips — `ddm_bl1_per_position_bit_allocation_20260822.md`) · spatial band · margin at break ·
   whether the break is a NEW error or an amplified existing one. If a field is absent from disk,
   say so and re-derive it from the pinned categorical field — never from a headline.
2. NAME the mechanism per cluster. Candidate mechanisms already in the corpus, each to be confirmed
   or refuted AT SOURCE, not assumed: paint ordering · AA coverage at class boundaries · uint8
   amplitude floors · prototype-color choice vs the frozen affine head · pre-R sub-pixel placement.
   Cite `ddm_v14_realization_fidelity` lineage where it already measured one of these.
3. PRICE each addressable cluster in BOTH currencies: the d_seg it would recover (MEASURED on the
   current object through the real R + uint8 + frozen SegNet, never interpolated — exponent 16.7)
   and the bytes the fix costs, if any. A zero-byte fix (a render-side ordering or placement change)
   is the highest-value shape and must be separated from any fix that adds a counted stream.
4. Confront `ddm_ld1_lane_lossy_drop_exchange_20260822.md` DIRECTLY: every lossy Lane rung makes the
   archive BIGGER, so Lane is defended on both sides. If a cluster's fix touches Lane, it must show
   why it is not an ld1 rung in a new costume, or exclude Lane by design and say so.
5. Exchange rate 6.658590e-07 S/B — CITE `ddm_tx1_toolbox_crosswalk_20260819.md` §0, do NOT re-derive.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- NO Metal slot: jf1 holds it (six k-point CPU fits at epoch 30/60). Derivation + CPU measurement only.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/`
  (Vertigo near-full). sha256 + bytes on every persisted payload, never scalars alone.
- Any recovered d_seg is MEASURED on the current object or it is reported as UNMEASURED with the
  blocker named. NEVER interpolate distortion between rungs (amplification exponent ~16.7).
- File-ownership: sibling arms ddm_na12/ddm_w96f/ddm_hr3 are LIVE — do not touch their surfaces.
  jf1's receipts under `.omx/tmp/arm_receipts_local/ddm_jf1_*` are SACRED.
- Every negative-existence claim states its SEARCH SCOPE or is not made (m53).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ld1_lane_lossy_drop_exchange_20260822.md` — every lossy Lane rung makes the archive BIGGER.
  Lane defended on BOTH sides. Confront or exclude; do not walk into it.
- `ddm_rj1` — renderer re-representation REFUSED 3.51×, and **d_pose is 97.7% of that refusal**: the
  first contest-CUDA proof that per-block FiLM carries pose. Any render-side change therefore has a
  POSE channel and must price it, not assume seg-only. `ddm_gv1`-adjacent: PoseNet scores the FRAMES.
- `ddm_nr1` (349×) · `ddm_ni1` (247.69×) · `ddm_rc1`/`ddm_ri1` — whole-body lossy CLOSED on two
  authority rows. This arm is NOT a lossy rung; if it becomes one, it inherits their graveyard.
- `ddm_et1` + `ddm_hg1` — the container route is closed on both horns (535,761 B / 460,408 B vs a
  137,986 B cap). Do not route the fix through a container.
- **THE SHARP-OPTIMUM LAW** (`ddm_oe1_*`, `ddm_ld1_*`, `ddm_ae1_*`, `ddm_ni1_*`, `ddm_wj1_*`) — five
  concordant arms. The escape clause this charter invokes is object-change, and it is under test.
- `ddm_jf1` (LIVE) — its mandatory positive control FAILED by 7,554 B. Any jf1 figure inherits that
  caveat explicitly.

## OPTIMAL FORM

- **PROVENANCE PINS (verify before reading; do not work from memory):**
  - the manufactured fraction: `ddm_ms9_dx2_seg_manufactured_fraction_20260822.md`
  - the stage split: `ddm_mst1_manufactured_stage_split_20260822.md`
  - the concentration: `ddm_bl1_per_position_bit_allocation_20260822.md`
  - the composition law: `ddm_sy2_composition_synergy_deep_pass_20260823.md`, commit `fe2ba12dc2`,
    memo sha prefix `32fc8fcc206bf76c`
  - current object: archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`
    (180,368 B); categorical field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
- Family exemplar: the **reference** form for per-stage realization diagnosis is the
  `ddm_v14_realization_fidelity` lineage (exact mask → painted RGB → R → uint8 → SegNet argmax, with
  the named leaks). This arm is that instrument aimed at the CURRENT object with the ms9/mst1 split
  already measured — a scope reduction (one object, measured stages) not a mechanism reduction.
- SCOPE reductions LEGAL, declared per row: a class subset, a pair subset with its selection stated
  (NOT a prefix — `ddm_na4`/m88: a prefix of a skewed population is a different population; seg
  prefix bias ≈0.96×, small but the selection must still be declared), bounding a fix rather than
  solving it. MECHANISM reductions FORBIDDEN: no entropy-estimate pricing, no interpolated distortion,
  no "seg-only" claim on a render change without its pose channel measured (rj1's 97.7%).
- **PRIOR-LAW PREDICTION (falsifiable, deliberately NOT optimistic).** The sharp-optimum law has held
  in every measured direction for two days, and rj1 refused renderer re-representation at 3.51× with
  pose carrying 97.7% of the refusal. So the honest prior is that the manufactured error is mostly
  the renderer's CAPACITY expressing itself, not a fixable defect — and that any render change large
  enough to recover meaningful seg pays a pose tax that swamps it. I predict **REFUTED: no cluster
  yields a MEASURED net-negative joint ΔS at zero or near-zero byte cost.** FALSIFIER: one addressable
  cluster with a measured d_seg recovery whose joint ΔS (seg + pose + rate, all three measured) is
  negative on the current object. If that lands, it is the first live distortion-currency route in
  the campaign and MAIN fires the scorer row. If it does not, the two-currency reading is closed on
  its seg half and the campaign fact is that BOTH currencies are shut from the current object —
  state that plainly, it is the more decision-relevant outcome.

## DELIVERABLE

`.omx/research/ddm_mf1_manufactured_seg_repair_20260823.md` — typed rows: per cluster {class, spatial
band, pixel count, mechanism CONFIRMED-AT-SOURCE | REFUTED | UNDETERMINED, measured d_seg recovery or
UNMEASURED-with-blocker, byte cost, measured pose channel, joint ΔS}; the ld1 confrontation as its own
section; totals vs 137,986 B in both currencies; the prior-law prediction adjudicated CONFIRMED or
REFUTED with its number; verdict_scope on the narrowest rung the evidence supports; a
`STORES CONSULTED:` line (the contract's literal key) naming what was loaded, honestly including
"none" where none. Commit via the serializer. End with the own-vehicle frontier line.
