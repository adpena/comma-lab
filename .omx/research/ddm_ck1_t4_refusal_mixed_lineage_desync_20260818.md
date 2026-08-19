# ck1 T4 refusal S=79.4 — mixed-lineage silent arithmetic desync (typed + cured)

`verdict_scope`: **INSTANCE** — archive `35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3`
@ 177,182 B staged from the ck1 GENERATION runtime, contest-CUDA T4 n600
(call `fc-01M0BGYVF1RBYZKJPMA9KN7K9B`, ~$0.16, wall 456 s). The CLASS finding
(mixed-lineage assembly desyncs silently) is DERIVED from the module diff and the
receipt pair; the cured recompose is byte-identical on the archive side and
re-measured through the class-cure advisory below.

## 1. The row

d_seg **0.41580084** · d_pose **142.15991211** · S 79.40216174747616 — both scorers
destroyed. rc=0: the decode RAN to completion and produced garbage frames.
Receipt: `/Volumes/APDataStore/pact/ddm_ck1/t4_row/MODAL_REMOTE_RESULT.json` (persisted, P0).

## 2. Mechanism (EXACT, verified at source)

The compiled candidate paired **sz1's token bytes** with **ck1's runtime**:

- sz1's `runtime/free_corrector.py` imports `fx1_logistic_mixer_corrector` (+ `rr4_free_corrector`)
  — the fx1/rr4 probability chain the sz1 hpac section was ENCODED for.
- ck1's generation runtime forked BEFORE fx1/rr4: both modules ABSENT, its
  `free_corrector.py` diverges by 980 diff-lines. Same entry name, different
  probability model.
- Arithmetic decoding under a mismatched probability model never errors — it
  emits wrong symbols from the first divergent bin onward. Every downstream frame
  is garbage; both scorers pay.

Why nothing upstream caught it:
- **Parse-back is structure-level** (sections + shas round-trip) — it cannot see a
  probability-model mismatch.
- **All the scoring receipts in the admit arithmetic were measured on DIFFERENT
  objects**: the ck1 generation archive through ck1's runtime (advisory, authority
  solve) and sz1's archive through sz1's runtime. The COMPOSED candidate — sz1
  sections × ck1 runtime — was never decoded-and-scored anywhere before the paid row.
- NOT the rr2 DEVICE class (#1096) despite the surface resemblance: this desync is
  device-independent and would reproduce on CPU. The typing discriminator is
  "would CPU reproduce it" — here yes.

Bonus divergence found on the same diff: the RECEIVERS forked BOTH WAYS
(ck1 added SM3R mode-6; sz1 added SD1M depth rows). Neither tree alone decodes the
composed candidate.

## 3. Cures (landed this session)

1. **Recompose via `runtime_overlays`** (`experiments/ddm_sa3_rebase_sz1.py`):
   per-row list of (tree, relpath) files copied over the sz1 base runtime.
   ck1 row = sz1 runtime (fx1/rr4 chain intact) + ck1's
   `cpr1/ddm_mp2_semantic_receiver.py` (the only file that knows mode 6).
   Overlay refuses to ADD files blind (target must exist in base).
   r4 compile: parse-back PASS both variants, archive byte-identical
   (`35c318d5…` @ 177,182 — container assembly is deterministic).
2. **Pattern-based inflate.py re-pin, fail-closed** (same session, sister fix):
   the literal-substitution re-pin silently missed non-sz1 pins; now re-pins by
   assignment regex and refuses if no pin moved.
3. **CLASS CURE — composed-archive CPU advisory gate**: any candidate whose
   sections and runtime come from different lineages MUST be decoded AND scored
   end-to-end on local CPU (full n600 advisory) before a paid axis row. The
   keep01 chain got away without it because all its ingredients were
   same-lineage; ck1 is the first mixed-lineage compile and the first casualty.
   Fired as `/Volumes/APDataStore/pact/ddm_ck1/advisory_rebased/attempt_0001`.
   Structural (tool-enforced) version: follow-on — teach `stage_generation` (or
   the seal producer) to demand an advisory receipt keyed to the staged archive
   sha when `runtime_overlays`/`source_runtime` deviates from the base lineage.

## 4. Genus links

Sister of [[modal_dispatch_five_failure_classes_permanent_fixes_20260804]] (the
2-validators-disagree → env-coupled-digest law: here the two "validators" were
parse-back and the T4 scorer, and the missing shared object was a scored decode)
and of #1096 (rr2 device desync — same silent-arithmetic genus, different
trigger). The measured-object-vs-named-object law
([[measured_object_vs_named_object_20260816]]) applies exactly: the admit
arithmetic quoted receipts for objects that were not the fired object.

## 5. Chain state at memo time

Advisory attempt_0001 died at t=10s, rc=2 — NOT a decode failure: the #929
bare-`python` class through the LOCAL harness. The sz1-lineage `inflate.sh`
probes `python` (absent on this macOS host) → falls into its Brotli bootstrap
branch → `uv pip install --python python` → uv's "No interpreter found for
executable name `python`" → exit 2 under `set -e`, 0.0 s elapsed. Modal's T4
container has `python` on PATH, which is why the same script never failed
remotely. The V-series/keep01 advisories survived via
`/Volumes/APDataStore/pact/ddm_sa1/pyshim/python` (exec-wrapper → repo venv,
per `python-shim-must-be-exec-wrapper-never-symlink`) prepended to PATH in the
launch env — my attempt_0001 launch omitted that prefix. Cure = harness-env
fix ONLY (pyshim PATH + PYTHONDONTWRITEBYTECODE=1); the shipped tree stays
byte-identical. With the shim, the venv's Brotli 1.2.0 satisfies the probe and
the uv branch is skipped entirely.

Attempt_0002 relaunched with the corrected env (the proven keep01 template).
If d_seg lands ≈0.00043 band and pose residual ≈1.48e-4 band (ck1's own
AUTHORITY.json) → re-seal (new runtime digest; r3 seal STALE post-r4 overlay
re-stage) → T4 refire (~$0.16). If the advisory refuses on SCORE → the
authority solve's decoded-state identity assumption is next in the fault tree
(ck1's L_ck1 vs sz1's L_sz1 lattice).

## 6. VERDICT — the TENTH pointer move (appended at harvest, 2026-08-19 ~00:00Z)

The chain completed end-to-end and the row was **ADMITTED**:

- **Advisory attempt_0002 CLEAN** (rc=0, inflate 941 s, evaluate 407 s):
  canonical_score 0.19982266166528362, d_seg 0.00043336 (in the AUTHORITY band),
  d_pose 0.00014829 (matches the authority solve to 8dp — the decoded-state
  identity assumption HELD, ending the fault tree). The composed-archive CPU
  advisory class gate (§3.3) PASSED on its first real use.
- **r4 seal** `a64b3483…` (runtime digest `5d7bd6f6…`, 33 files, seal OUTSIDE
  the runtime tree per the r3 lesson) → `fire_modal_auth_eval.py --seal` with
  the standing F26 single-axis waiver (#1049/#1054).
- **T4 row** (call `fc-01M0BKKHWT2S2ZTET8BKXNPEXW`, 1,302 s, ~$0.16):
  **S 0.15710198138050818 @ 177,182 B [contest-CUDA T4, n600]** — d_seg
  0.00030309 · d_pose 7.77e-6 · rate 0.11797822. Receipt persisted at
  `/Volumes/APDataStore/pact/ddm_ck1/t4_row_r4/MODAL_REMOTE_RESULT.json` (P0).
- **Net vs keep01** (0.1571619225142182 @ 177,576 B): **−5.994113e-05,
  ADMITTED** (17.1× the sealed −3.5e-6 bar). Leg split (sums exactly):
  rate −2.6235e-4 (−394 B) · seg +1.7400e-4 · pose +2.8407e-5.
- **Pointer moved with ZERO manual steps**: the firer's scanner-visible anchor
  mirror (`experiments/results/modal_auth_eval_mirror/…ck1_r4…json`, the rv8f
  cure) let `refresh_canonical_frontier.py` pick up the row directly —
  effective_frontier now 0.1571019814 [contest-CUDA]. Gap to 0.15: 0.00710.

**HONEST SHORTFALL (new empirical gap, do not re-forget):** realized −5.99e-5
is **35% of the −1.72e-4 projection**. The pose leg transferred fine, the
rate leg is exact by construction — the miss is entirely the SEG leg, which
landed at **+1.7400e-4 on T4** vs a much smaller CPU-modeled delta. We have a
measured CPU→CUDA transfer law for POSE (ABSOLUTE+13%) and **none for SEG**.
Until a seg sibling law is measured, composed-candidate projections that lean
on a CPU-advisory seg delta must be treated as UPPER BOUNDS on the realized
win, and the admit bar arithmetic should discount the seg leg accordingly.
Memory: `cpu_to_cuda_seg_transfer_has_no_law_20260819`.
